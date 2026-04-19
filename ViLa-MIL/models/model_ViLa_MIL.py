# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import math
import os
import sys
import warnings
from .model_utils import *
from utils.core_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F


def _ensure_conch_on_path():
    """conch 来自 mahmoodlab/CONCH；未 pip install -e 时，若仓库在 ViLa-MIL/CONCH 也可直接 import。"""
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    for key in ("CONCH_REPO_PATH", "CONCH_SRC"):
        v = os.environ.get(key)
        if v:
            candidates.append(os.path.abspath(v.strip()))
    candidates.append(os.path.join(_root, "CONCH"))
    for base in candidates:
        if base and os.path.isdir(os.path.join(base, "conch")):
            if base not in sys.path:
                sys.path.insert(0, base)
            return


_ensure_conch_on_path()

# 文本分支使用 CONCH（conch）自带 tokenizer，不依赖 OpenAI CLIP 包
try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "未找到 Python 包 conch。请先克隆 https://github.com/mahmoodlab/CONCH 到 "
        "ViLa-MIL/CONCH（或设置环境变量 CONCH_REPO_PATH 指向该仓库根目录），"
        "再执行: conda activate ViLa-MIL && pip install -e \"$CONCH_REPO_PATH\" 。"
        "也可运行: bash scripts/install_conch.sh"
    ) from e

class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        # Get dtype from one of the model's parameters
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        # Rest of the code remains the same
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, conch_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        
        # Get the tokenizer
        self.tokenizer = get_tokenizer()

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            # Use the correct tokenize function with both arguments
            prompt = tokenize(self.tokenizer, [ctx_init])
            with torch.no_grad():
                embedding = conch_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if False:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        # Process class names
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [name for name in classnames]
        
        # Use the correct tokenize function with both arguments
        tokenized_prompts = tokenize(self.tokenizer, prompts)
        
        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        # Use the tokenizer's encode method for getting lengths
        self.name_lens = [len(self.tokenizer.encode(name, 
                                                   max_length=127,
                                                   truncation=True)) 
                         for name in classnames]
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ViLa_MIL_Model(nn.Module):
    def __init__(self, config, num_classes=3):
        super(ViLa_MIL_Model, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        conch_model_cfg = getattr(config, "conch_model_cfg", "conch_ViT-B-16")
        conch_checkpoint_path = (
            getattr(config, "conch_checkpoint_path", None)
            or os.environ.get("CONCH_CKPT_PATH")
            or os.path.join(os.path.dirname(os.path.dirname(__file__)), "ckpt", "conch.pth")
        )
        if not os.path.exists(conch_checkpoint_path):
            raise FileNotFoundError(
                "CONCH checkpoint not found: {}. "
                "Pass --conch_checkpoint_path /path/to/conch.pth or set CONCH_CKPT_PATH.".format(
                    conch_checkpoint_path
                )
            )
        conch_model, preprocess = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)





        self.prompt_learner = PromptLearner(config.text_prompt, conch_model.float())
        self.text_encoder = TextEncoder(conch_model.float())

        self.norm = nn.LayerNorm(config.input_size)
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)
        # 输出维度应与类别数一致；原实现固定为 4 会在二分类等任务下报 shape mismatch
        self.norm1 = nn.LayerNorm(self.num_classes)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.norm4 = nn.LayerNorm(512)
        self.norm5 = nn.LayerNorm(512)
        


        self.learnable_image_center = nn.Parameter(torch.Tensor(*[config.prototype_number, 1, config.input_size]))
        trunc_normal_(self.learnable_image_center, std=.02)
        self.hard_or_soft = config.hard_or_soft

    @staticmethod
    def _mha_kv_from_patches(x):
        """MultiheadAttention(batch_first=False) 需要 (L,N,E)；patch 特征常为 (N_patches,E)，补 batch 维得到 (N_patches,1,E)。"""
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x

    def forward(
        self,
        x_s,
        coord_s,
        x_l,
        coords_l,
        label=None,
        staus=None,
        time=None,
        disc=None,
        soft_0=None,
        soft_1=None,
        soft_2=None,
        soft_3=None,
    ):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # x_s / x_l: (N_patches, E)；learnable_image_center: (P,1,E) —— key/value 必须与 query 同为 3 维
        M = x_s.float()
        M_kv = self._mha_kv_from_patches(M)
        compents, _ = self.cross_attention_1(self.learnable_image_center, M_kv, M_kv)
        compents = self.norm(compents + self.learnable_image_center)

        M_high = x_l.float()
        M_high_kv = self._mha_kv_from_patches(M_high)
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high_kv, M_high_kv)
        compents_high = self.norm(compents_high + self.learnable_image_center)

        H = compents.squeeze().float()
        A_V = self.attention_V(H)  
        A_U = self.attention_U(H)  
        A = self.attention_weights(A_V * A_U) 
        A = torch.transpose(A, 1, 0)  
        A = F.softmax(A, dim=1)  
        image_features_low = torch.mm(A, H)  

        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)  
        A_U_high = self.attention_U(H_high)  
        A_high = self.attention_weights(A_V_high * A_U_high) 
        A_high = torch.transpose(A_high, 1, 0)  
        A_high = F.softmax(A_high, dim=1)  
        image_features_high = torch.mm(A_high, H_high)  

        text_features_low = text_features[:self.num_classes]
        image_context = torch.cat((compents.squeeze(), M), dim=0)
        image_context_kv = self._mha_kv_from_patches(image_context)


        # print(image_context.shape)

        # print(text_features_low.unsqueeze(1).shape)

        text_context_features, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1), image_context_kv, image_context_kv
        )
        text_features_low = text_context_features.squeeze() + text_features_low

        text_features_high = text_features[self.num_classes:]
        image_context_high = torch.cat((compents_high.squeeze(), M_high), dim=0)
        image_context_high_kv = self._mha_kv_from_patches(image_context_high)
        text_context_features_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1), image_context_high_kv, image_context_high_kv
        )
        text_features_high = text_context_features_high.squeeze() + text_features_high


        image_features_high = self.norm2(image_features_high)
        image_features_low = self.norm3(image_features_low)
        text_features_low = self.norm4(text_features_low)
        text_features_high = self.norm5(text_features_high)

        # 与输入/模型同设备（CPU 或 CUDA）；勿硬编码 .cuda()，否则 Mac 等无 CUDA 环境会报错
        _dev = image_features_low.device
        logits_low = image_features_low @ text_features_low.T.to(_dev)
        logits = logits_low

        # 某些配置下 text_features 可能只有 num_classes（没有 high 部分），此时跳过 high 分支
        if text_features_high is not None and getattr(text_features_high, "numel", lambda: 0)() > 0:
            logits_high = image_features_high @ text_features_high.T.to(_dev)
            if logits_high.shape == logits_low.shape:
                logits = logits_low + logits_high
        logits = self.norm1(logits)
        # print(logits)

        if disc is not None and staus is not None:
            disc = disc.unsqueeze(1)
            staus = staus.unsqueeze(1)

        # print(logits.shape)
        # print(disc.shape)
        # print(staus.shape)

        loss = None
        if disc is not None and staus is not None:
            if self.hard_or_soft:    #  True 使用软标签
                loss = nll_loss_soft(
                    logits,
                    disc,
                    staus,
                    soft_0,
                    soft_1,
                    soft_2,
                    soft_3,
                    alpha=0.4,
                    eps=1e-7,
                    reduction='mean',
                )
            else:
                loss = nll_loss(logits, disc, staus, alpha=0.4, eps=1e-7, reduction='mean')
        Y_prob = F.softmax(logits, dim = 1)
        # Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]
        

        return logits, Y_prob, loss

