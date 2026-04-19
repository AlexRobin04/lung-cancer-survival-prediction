"""
Feature-Level Ensemble MIL Model
=================================
Integrates 5 baseline models (RRTMIL, AMIL, WiKG, DSMIL, S4MIL) by extracting
their bag-level feature representations, concatenating them, and passing through
a learnable fusion head for final classification.

Architecture:
    x_s -> [RRTMIL, AMIL, WiKG, DSMIL, S4MIL] -> [f1..f5]
    -> 各分支 LayerNorm + Linear(->D) 对齐
    -> fusion_mode:
       - "gate": concat(z*) 过 gate_mlp -> softmax 得 5 维权重，对 z* 加权求和 -> MLP 分类
       - "concat": concat(z*) -> MLP 分类（基线）
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections

from utils.core_utils import *  # nll_loss / nll_loss_soft（与其它 MIL 模型一致）

# 分支顺序：与 ensemble_branch_mask、gate 堆叠顺序一致
ENSEMBLE_BRANCH_ORDER = ("RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL")
_ALL_BRANCH_SET = frozenset(ENSEMBLE_BRANCH_ORDER)


def _normalize_ensemble_branch(name: str) -> str | None:
    u = str(name).strip().upper().replace("-", "_")
    if u in ("S4", "S4MIL"):
        return "S4MIL"
    if u in _ALL_BRANCH_SET:
        return u
    return None


def _parse_ensemble_exclude(exclude) -> list[str]:
    if exclude is None:
        return []
    if isinstance(exclude, str):
        parts = [x.strip() for x in exclude.replace(";", ",").split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in exclude]
    out: list[str] = []
    for p in parts:
        n = _normalize_ensemble_branch(p)
        if n and n not in out:
            out.append(n)
    return out


class EnsembleFeatureMIL(nn.Module):
    def __init__(
        self,
        config,
        n_classes,
        feat_dim=512,
        freeze_base=True,
        feature_align_dim=512,
        fusion_mode="gate",
        ensemble_exclude=None,
    ):
        """
        Args:
            config: ml_collections.ConfigDict with at least `hard_or_soft`
            n_classes: number of output classes
            feat_dim: input feature dimension of patch features (default 512)
            freeze_base: if True, freeze all 5 baseline model parameters and only
                         train the fusion head; if False, fine-tune everything
            feature_align_dim: 拼接前将各基线 bag 特征投影到该维度（配合 LayerNorm 缓解尺度差异）
            fusion_mode: "gate" 为门控加权融合；"concat" 为对齐后直接拼接再 MLP（消融用）
            ensemble_exclude: 要掩码掉的基线名称列表（如 ["RRTMIL"]），对齐后该路特征置零，用于 6.1 留一等消融；不可全部排除
        """
        super(EnsembleFeatureMIL, self).__init__()

        # ---- instantiate 5 baselines ----
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
            config=config, in_dim=feat_dim, n_classes=n_classes,
            dropout=0.1, act="relu", d_model=512, d_state=16,
        )

        # ---- feature dimensions at the bag-level ----
        # RRTMIL: online_encoder.final_dim
        # AMIL:   256
        # WiKG:   dim_hidden = 512
        # DSMIL:  B from BClassifier = in_size = feat_dim
        # S4MIL:  d_model = 512
        rrt_dim = self.rrtmil.online_encoder.final_dim
        amil_dim = 256
        wikg_dim = 512
        dsmil_dim = feat_dim
        s4mil_dim = 512
        d_align = int(feature_align_dim)
        if d_align <= 0:
            raise ValueError("feature_align_dim must be positive")

        # ---- 分支对齐：LN + 投影到统一维度（再拼接），减轻尺度不一导致的梯度主导 ----
        self.feature_align_dim = d_align
        self.align_rrt = nn.Sequential(nn.LayerNorm(rrt_dim), nn.Linear(rrt_dim, d_align))
        self.align_amil = nn.Sequential(nn.LayerNorm(amil_dim), nn.Linear(amil_dim, d_align))
        self.align_wikg = nn.Sequential(nn.LayerNorm(wikg_dim), nn.Linear(wikg_dim, d_align))
        self.align_dsmil = nn.Sequential(nn.LayerNorm(dsmil_dim), nn.Linear(dsmil_dim, d_align))
        self.align_s4mil = nn.Sequential(nn.LayerNorm(s4mil_dim), nn.Linear(s4mil_dim, d_align))

        mode = str(fusion_mode or "gate").strip().lower()
        if mode not in ("gate", "concat"):
            raise ValueError('fusion_mode must be "gate" or "concat"')
        self.fusion_mode = mode

        # ---- 门控：由 5 路对齐特征联合决定各基模型权重（样本自适应）----
        self.gate_mlp = None
        if self.fusion_mode == "gate":
            self.gate_mlp = nn.Sequential(
                nn.Linear(5 * d_align, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 5),
            )
            fusion_in = d_align
        else:
            fusion_in = 5 * d_align

        # ---- fusion classification head ----
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

        self.hard_or_soft = config.hard_or_soft
        self.loss_ce = nn.CrossEntropyLoss()

        excl = _parse_ensemble_exclude(ensemble_exclude)
        excluded_set = frozenset(excl)
        if not excluded_set.issubset(_ALL_BRANCH_SET):
            raise ValueError(f"ensemble_exclude 含未知键: {sorted(excluded_set - _ALL_BRANCH_SET)}")
        if len(excluded_set) >= len(_ALL_BRANCH_SET):
            raise ValueError("ensemble_exclude 不能排除全部五路基线")
        mask_list = [0.0 if b in excluded_set else 1.0 for b in ENSEMBLE_BRANCH_ORDER]
        self.register_buffer("ensemble_branch_mask", torch.tensor(mask_list, dtype=torch.float32))

        # ---- freeze baseline parameters if requested ----
        if freeze_base:
            for param in self.rrtmil.parameters():
                param.requires_grad = False
            for param in self.amil.parameters():
                param.requires_grad = False
            for param in self.wikg.parameters():
                param.requires_grad = False
            for param in self.dsmil.parameters():
                param.requires_grad = False
            for param in self.s4mil.parameters():
                param.requires_grad = False
            # align_* 与 fusion_head 保持可训练

    def load_pretrained(self, rrt_ckpt, amil_ckpt, wikg_ckpt, dsmil_ckpt, s4mil_ckpt, device='cpu'):
        """Load pre-trained checkpoint for each baseline model.

        Args:
            rrt_ckpt, amil_ckpt, wikg_ckpt, dsmil_ckpt, s4mil_ckpt: paths to .pt files
            device: device to load weights onto
        """
        self.rrtmil.load_state_dict(torch.load(rrt_ckpt, map_location=device))
        self.amil.load_state_dict(torch.load(amil_ckpt, map_location=device))
        self.wikg.load_state_dict(torch.load(wikg_ckpt, map_location=device))
        self.dsmil.load_state_dict(torch.load(dsmil_ckpt, map_location=device))
        self.s4mil.load_state_dict(torch.load(s4mil_ckpt, map_location=device))

    def _extract_rrtmil_feature(self, x_s):
        x = x_s.float().unsqueeze(0)
        x = self.rrtmil.patch_to_emb(x)
        x = self.rrtmil.dp(x)
        x = self.rrtmil.online_encoder(x)
        # pool_fn returns (feature, attention) when return_attn=True
        x, _ = self.rrtmil.pool_fn(x, return_attn=True)
        return x  # [1, final_dim]

    def _extract_amil_feature(self, x_s):
        h = x_s
        A, h = self.amil.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        h = torch.mm(A, h)  # [1, 256]
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
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)
        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.wikg.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.wikg.activation(self.wikg.linear(embedding))
        elif self.wikg.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.wikg.activation(self.wikg.linear(embedding))
        elif self.wikg.agg_type == 'bi-interaction':
            sum_embedding = self.wikg.activation(self.wikg.linear1(e_h + e_Nh))
            bi_embedding = self.wikg.activation(self.wikg.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        h = self.wikg.message_dropout(embedding)
        h = self.wikg.readout(h.squeeze(0), batch=None)
        h = self.wikg.norm(h)  # [1, 512]
        return h

    def _extract_dsmil_feature(self, x_s):
        feats, classes = self.dsmil.i_classifier(x_s)
        # BClassifier returns (C, A, B) where B is [1, n_classes, feat_dim]
        C, A, B = self.dsmil.b_classifier(feats, classes)
        # B: [1, n_classes, feat_dim] -> average over class dim -> [1, feat_dim]
        return B.mean(dim=1)

    def _extract_s4mil_feature(self, x_s):
        x = x_s.unsqueeze(0)
        x = self.s4mil._fc1(x)
        x = self.s4mil.s4_block(x)
        x = torch.max(x, axis=1).values  # [1, d_model]
        return x

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
        # ---- extract bag-level features from each baseline ----
        f_rrt = self._extract_rrtmil_feature(x_s)
        f_amil = self._extract_amil_feature(x_s)
        f_wikg = self._extract_wikg_feature(x_s)
        f_dsmil = self._extract_dsmil_feature(x_s)
        f_s4mil = self._extract_s4mil_feature(x_s)

        # ---- 对齐后拼接 ----
        z_rrt = self.align_rrt(f_rrt)
        z_amil = self.align_amil(f_amil)
        z_wikg = self.align_wikg(f_wikg)
        z_dsmil = self.align_dsmil(f_dsmil)
        z_s4mil = self.align_s4mil(f_s4mil)
        zs = torch.stack([z_rrt, z_amil, z_wikg, z_dsmil, z_s4mil], dim=1)
        m = self.ensemble_branch_mask.to(device=zs.device, dtype=zs.dtype).view(1, 5, 1)
        zs = zs * m
        cat_z = zs.flatten(1)

        if self.fusion_mode == "gate":
            gate_logits = self.gate_mlp(cat_z)
            w = F.softmax(gate_logits, dim=-1)
            fused = (w.unsqueeze(-1) * zs).sum(dim=1)
        else:
            fused = cat_z

        # ---- fusion head -> logits ----
        logits = self.fusion_head(fused)  # [B, n_classes]

        # ---- loss ----
        loss = None
        if (disc is not None) and (staus is not None):
            disc = disc.unsqueeze(1)
            staus = staus.unsqueeze(1)
            if self.hard_or_soft:
                loss = nll_loss_soft(
                    logits, disc, staus, soft_0, soft_1, soft_2, soft_3,
                    alpha=0.4, eps=1e-7, reduction="mean"
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
