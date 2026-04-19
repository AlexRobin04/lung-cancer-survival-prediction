"""
Feature-Level Ensemble MIL Model
=================================
Integrates 5 baseline models (RRTMIL, AMIL, WiKG, DSMIL, S4MIL) by extracting
their bag-level feature representations, concatenating them, and passing through
a learnable fusion head for final classification.

Architecture:
    x_s -> [RRTMIL, AMIL, WiKG, DSMIL, S4MIL] -> [f1, f2, f3, f4, f5]
    -> concat -> Fusion Head -> logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import os

from utils.core_utils import *  # nll_loss / nll_loss_soft（与其它 MIL 模型一致）


class EnsembleFeatureMIL(nn.Module):
    def __init__(self, config, n_classes, feat_dim=512, freeze_base=True):
        """
        Args:
            config: ml_collections.ConfigDict with at least `hard_or_soft`
            n_classes: number of output classes
            feat_dim: input feature dimension of patch features (default 512)
            freeze_base: if True, freeze all 5 baseline model parameters and only
                         train the fusion head; if False, fine-tune everything
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
        total_feat_dim = rrt_dim + amil_dim + wikg_dim + dsmil_dim + s4mil_dim

        # ---- fusion classification head ----
        self.fusion_head = nn.Sequential(
            nn.Linear(total_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

        self.hard_or_soft = config.hard_or_soft
        self.loss_ce = nn.CrossEntropyLoss()

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

        # ---- concatenate ----
        fused = torch.cat([f_rrt, f_amil, f_wikg, f_dsmil, f_s4mil], dim=-1)

        # ---- fusion head -> logits ----
        logits = self.fusion_head(fused)  # [1, n_classes]

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
