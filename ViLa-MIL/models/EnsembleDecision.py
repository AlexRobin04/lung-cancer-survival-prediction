"""
决策级（后期）集成：五路基线 MIL 各自前向得到分类 logits，再对 logits 或概率做融合。
不涉及特征对齐 / 门控 MLP / 跨支路 Transformer。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections

from models.ensemble_branch_utils import (
    ENSEMBLE_BRANCH_ORDER,
    _ALL_BRANCH_SET,
    _parse_ensemble_exclude,
    _branch_prior_probs_tensor,
    _decision_branch_weights_tensor,
)
from utils.core_utils import *  # nll_loss / nll_loss_soft


class EnsembleDecisionMIL(nn.Module):
    """
    决策级（后期）融合：五路基线各自完整前向得到 logits，按**简单概率均值**融合，**不训练融合头**。

    核心规则（唯一）：
      - avg_prob: 对各活跃支路 softmax(logits) 后做概率加权平均；
        当存在 branch_prior/decision_branch_weights 时使用其归一化权重（自动化按模型可靠度投票），
        否则退化为等权平均（例如两路某类概率 0.8 / 0.7 -> 0.75）。
    """

    @staticmethod
    def branch_expected_risk_max(parts: torch.Tensor, mask1: torch.Tensor) -> torch.Tensor:
        """每样本在活跃支路上 max_b Σ_k k·softmax(logits_b)_k，形状 (B,)。"""
        probs_b = F.softmax(parts, dim=-1)
        k = int(probs_b.size(-1))
        cls_idx = torch.arange(k, device=parts.device, dtype=probs_b.dtype).view(1, 1, -1)
        exp_per_br = (probs_b * cls_idx).sum(dim=-1)
        exp_per_br = exp_per_br.masked_fill(mask1 < 0.5, float("-inf"))
        mx = exp_per_br.max(dim=1).values
        return torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

    def __init__(
        self,
        config,
        n_classes: int,
        feat_dim: int = 512,
        freeze_base: bool = True,
        decision_fusion: str = "avg_prob",
        ensemble_exclude=None,
        branch_prior: str | None = None,
        branch_prior_scale: float = 1.25,
        branch_prior_temperature: float = 1.0,
        decision_branch_weights: str | dict | None = None,
    ):
        super().__init__()
        from models.RRT import RRTMIL
        from models.AMIL import AMIL
        from models.WiKG import WiKG
        from models.DSMIL import MILNet
        from models.S4MIL import S4Model

        self.rrtmil = RRTMIL(config=config, n_classes=n_classes, input_dim=feat_dim)
        self.amil = AMIL(config=config, n_classes=n_classes, input_dim=feat_dim)
        self.wikg = WiKG(config=config, n_classes=n_classes, dim_in=feat_dim, dim_hidden=512)
        self.dsmil = MILNet(config=config, in_size=feat_dim, num_class=n_classes, dropout=0.25)
        self.s4mil = S4Model(
            config=config,
            in_dim=feat_dim,
            n_classes=n_classes,
            dropout=0.1,
            act="relu",
            d_model=512,
            d_state=16,
        )

        # 兼容旧任务传入的 fusion 名称，统一收敛到 avg_prob。
        _ = str(decision_fusion or "avg_prob").strip().lower()
        self.decision_fusion = "avg_prob"
        self.register_buffer("_ed_decision_tag", torch.tensor(0, dtype=torch.int8))

        excl = _parse_ensemble_exclude(ensemble_exclude)
        excluded_set = frozenset(excl)
        if not excluded_set.issubset(_ALL_BRANCH_SET):
            raise ValueError(f"ensemble_exclude 含未知键: {sorted(excluded_set - _ALL_BRANCH_SET)}")
        if len(excluded_set) >= len(_ALL_BRANCH_SET):
            raise ValueError("ensemble_exclude 不能排除全部五路基线")
        mask_list = [0.0 if b in excluded_set else 1.0 for b in ENSEMBLE_BRANCH_ORDER]
        self.register_buffer("ensemble_branch_mask", torch.tensor(mask_list, dtype=torch.float32))

        # 兼容旧 checkpoint：保留 fusion_logits buffer（当前用于构造概率融合权重）。
        fusion_logits = torch.zeros(5)
        dw = _decision_branch_weights_tensor(decision_branch_weights, excluded_set)
        prior_probs = _branch_prior_probs_tensor(branch_prior, excluded_set)
        scale = float(branch_prior_scale)
        tp = float(max(branch_prior_temperature, 0.05))
        eps = 1e-7
        if dw is not None:
            fusion_logits = scale * torch.log(dw + eps) * self.ensemble_branch_mask
            if abs(tp - 1.0) > 1e-6:
                fusion_logits = fusion_logits / tp
        elif prior_probs is not None:
            fusion_logits = scale * torch.log(prior_probs + eps) * self.ensemble_branch_mask
            if abs(tp - 1.0) > 1e-6:
                fusion_logits = fusion_logits / tp
        self.register_buffer("fusion_logits", fusion_logits)

        # 概率融合权重：优先显式 decision_branch_weights，其次自动 branch_prior（C-index 先验），否则等权。
        if dw is not None:
            vote_w = dw
        elif prior_probs is not None:
            vote_w = prior_probs
        else:
            vote_w = self.ensemble_branch_mask.clone()
            s = float(vote_w.sum())
            vote_w = vote_w / max(s, 1e-8)
        # 简单保底：自动先验投票时仅保留 Top-2 强模型，减少弱模型拖累。
        # 若用户显式传了 decision_branch_weights，则尊重用户权重，不做截断。
        if dw is None:
            active_idx = torch.where(self.ensemble_branch_mask > 0.5)[0]
            if int(active_idx.numel()) > 2:
                vals = vote_w[active_idx]
                topk_local = torch.topk(vals, k=2, dim=0).indices
                keep = active_idx[topk_local]
                pruned = torch.zeros_like(vote_w)
                pruned[keep] = vote_w[keep]
                s2 = float(pruned.sum())
                if s2 > 0:
                    vote_w = pruned / s2
        self.register_buffer("branch_vote_weights", vote_w)

        if freeze_base:
            for m in (self.rrtmil, self.amil, self.wikg, self.dsmil, self.s4mil):
                for p in m.parameters():
                    p.requires_grad = False

        self.hard_or_soft = config.hard_or_soft
        self.loss_ce = nn.CrossEntropyLoss()

    def load_pretrained(self, rrt_ckpt, amil_ckpt, wikg_ckpt, dsmil_ckpt, s4mil_ckpt, device="cpu"):
        self.rrtmil.load_state_dict(torch.load(rrt_ckpt, map_location=device))
        self.amil.load_state_dict(torch.load(amil_ckpt, map_location=device))
        self.wikg.load_state_dict(torch.load(wikg_ckpt, map_location=device))
        self.dsmil.load_state_dict(torch.load(dsmil_ckpt, map_location=device))
        self.s4mil.load_state_dict(torch.load(s4mil_ckpt, map_location=device))

    def set_decision_weights(self, weights: torch.Tensor) -> None:
        """
        动态更新 weighted 融合分支权重（长度 5，按 ENSEMBLE_BRANCH_ORDER）。
        会自动与 ensemble_branch_mask 相乘并归一化，再写回 fusion_logits。
        """
        # 当前固定为 avg_prob；保留接口仅为兼容旧代码调用。
        return
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=self.fusion_logits.dtype, device=self.fusion_logits.device)
        w = weights.to(device=self.fusion_logits.device, dtype=self.fusion_logits.dtype).view(-1)
        if int(w.numel()) != 5:
            raise ValueError(f"decision weights length must be 5, got {int(w.numel())}")
        mask = self.ensemble_branch_mask.to(device=w.device, dtype=w.dtype).view(-1)
        w = torch.clamp(w, min=0.0) * mask
        s = float(w.sum())
        if s <= 0:
            # 回退到活跃分支等权，避免写入无效权重。
            active = mask > 0.5
            n_act = int(active.sum().item())
            if n_act <= 0:
                return
            w = torch.zeros_like(mask)
            w[active] = 1.0 / float(n_act)
        else:
            w = w / w.sum().clamp_min(1e-8)
        self.fusion_logits.copy_(torch.log(w.clamp_min(1e-8)) * mask)

    def _extract_rrtmil_feature(self, x_s):
        x = x_s.float().unsqueeze(0)
        x = self.rrtmil.patch_to_emb(x)
        x = self.rrtmil.dp(x)
        x = self.rrtmil.online_encoder(x)
        x, _ = self.rrtmil.pool_fn(x, return_attn=True)
        return x

    def _extract_amil_feature(self, x_s):
        h = x_s
        A, h = self.amil.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        h = torch.mm(A, h)
        return h

    def _extract_wikg_feature(self, x_s):
        x = x_s.unsqueeze(0)
        x = self.wikg._fc1(x)
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5
        e_h = self.wikg.W_head(x)
        e_t = self.wikg.W_tail(x)
        attn_logit = (e_h * self.wikg.scale) @ e_t.transpose(-2, -1)
        topk_weight, topk_index = torch.topk(attn_logit, k=self.wikg.topk, dim=-1)
        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)
        Nb_h = e_t[batch_indices, topk_index_expanded, :]
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.wikg.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum("ijkl,ijkm->ijk", Nb_h, gate)
        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)
        if self.wikg.agg_type == "gcn":
            embedding = e_h + e_Nh
            embedding = self.wikg.activation(self.wikg.linear(embedding))
        elif self.wikg.agg_type == "sage":
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.wikg.activation(self.wikg.linear(embedding))
        elif self.wikg.agg_type == "bi-interaction":
            sum_embedding = self.wikg.activation(self.wikg.linear1(e_h + e_Nh))
            bi_embedding = self.wikg.activation(self.wikg.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding
        h = self.wikg.message_dropout(embedding)
        h = self.wikg.readout(h.squeeze(0), batch=None)
        h = self.wikg.norm(h)
        return h

    def _extract_s4mil_feature(self, x_s):
        x = x_s.unsqueeze(0)
        x = self.s4mil._fc1(x)
        x = self.s4mil.s4_block(x)
        x = torch.max(x, axis=1).values
        return x

    def stack_branch_logits(self, x_s, x_l=None):
        """五路基线各自 logits，形状 (B,5,K) 与掩码 (B,5)；x_l 保留参数与 forward 一致（当前支路仅用 x_s）。"""
        # 使用各基线模型原生 forward 输出 logits，确保与单模型推理口径一致。
        def _branch_logits(m):
            try:
                lg, _yp, _ls = m(x_s, None, x_l, None, None)
            except TypeError:
                lg, _yp, _ls = m(x_s, None, x_l, None)
            return lg

        l_rrt = _branch_logits(self.rrtmil)
        l_amil = _branch_logits(self.amil)
        l_wikg = _branch_logits(self.wikg)
        l_dsmil = _branch_logits(self.dsmil)
        l_s4 = _branch_logits(self.s4mil)
        parts = torch.stack([l_rrt, l_amil, l_wikg, l_dsmil, l_s4], dim=1)
        m = self.ensemble_branch_mask.to(device=parts.device, dtype=parts.dtype).view(1, 5, 1)
        bsz = int(parts.size(0))
        mask1 = m.expand(bsz, -1, -1).squeeze(-1)
        return parts, mask1

    def fuse_logits_from_parts(self, parts: torch.Tensor, mask1: torch.Tensor) -> torch.Tensor:
        """由 (B,5,K) 支路 logits 与 (B,5) 掩码得到融合后 (B,K) logits（与 forward 内规则一致）。"""
        probs = F.softmax(parts, dim=-1)
        base_w = self.branch_vote_weights.to(device=parts.device, dtype=parts.dtype).view(1, -1)
        w_br = (base_w * mask1).clamp_min(0.0)
        w_sum = w_br.sum(dim=1, keepdim=True).clamp_min(1e-8)
        w_norm = (w_br / w_sum).unsqueeze(-1)
        prob_fused = (probs * w_norm).sum(dim=1)
        return torch.log(prob_fused.clamp_min(1e-8))

    def forward(
        self,
        x_s,
        coord_s=None,
        x_l=None,
        coords_l=None,
        label=None,
        staus=None,
        time=None,
        disc=None,
        soft_0=None,
        soft_1=None,
        soft_2=None,
        soft_3=None,
        return_attn=True,
        no_norm=False,
    ):
        parts, mask1 = self.stack_branch_logits(x_s, x_l)
        # 兼容旧 API 字段：使用融合后期望风险作为标量。
        self._last_predict_risk_max = None

        logits = self.fuse_logits_from_parts(parts, mask1)

        loss = None
        if (disc is not None) and (staus is not None):
            disc = disc.unsqueeze(1)
            staus = staus.unsqueeze(1)
            if self.hard_or_soft:
                loss = nll_loss_soft(
                    logits, disc, staus, soft_0, soft_1, soft_2, soft_3, alpha=0.4, eps=1e-7, reduction="mean"
                )
            else:
                loss = nll_loss(logits, disc, staus, alpha=0.4, eps=1e-7, reduction="mean")
        elif label is not None:
            loss = self.loss_ce(logits.view(1, -1), label.view(1))

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, Y_prob, loss
