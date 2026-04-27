"""
五路基线集成共用：分支顺序、ensemble_exclude 解析、分支先验张量。
供 EnsembleDecision 与 api_server / core_utils 使用。
"""

from __future__ import annotations

import json

import torch

# 与 ensemble_branch_mask、checkpoint 中分支顺序一致
ENSEMBLE_BRANCH_ORDER = ("RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL")
_ALL_BRANCH_SET = frozenset(ENSEMBLE_BRANCH_ORDER)


def _normalize_ensemble_branch(name: str) -> str | None:
    u = str(name).strip().upper().replace("-", "_")
    if u == "WIKG":
        return "WiKG"
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


def _branch_prior_dict_from_spec(spec: str | None) -> dict[str, float] | None:
    """解析 'RRTMIL:0.62,AMIL:0.44' 或 JSON 对象字符串为分支名 -> 分数（如单模队列 C-index）。"""
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    out: dict[str, float] = {}
    if s.startswith("{"):
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict):
            return None
        for k, v in raw.items():
            n = _normalize_ensemble_branch(str(k))
            if n:
                out[n] = float(v)
    else:
        for part in s.replace(";", ",").split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            a, b = part.split(":", 1)
            n = _normalize_ensemble_branch(a.strip())
            if n:
                try:
                    out[n] = float(b.strip())
                except ValueError:
                    pass
    return out if out else None


def _branch_prior_probs_tensor(
    spec: str | None,
    excluded: frozenset[str],
    default_score: float = 0.5,
) -> torch.Tensor | None:
    """
    在 ENSEMBLE_BRANCH_ORDER 上构造先验分布：未写明的分支用 default_score；
    ensemble_exclude 中的分支概率为 0，其余归一化后和为 1。
    """
    d = _branch_prior_dict_from_spec(spec)
    if d is None:
        return None
    mask = torch.tensor([0.0 if b in excluded else 1.0 for b in ENSEMBLE_BRANCH_ORDER], dtype=torch.float32)
    vals: list[float] = []
    for b in ENSEMBLE_BRANCH_ORDER:
        if b in excluded:
            vals.append(0.0)
        else:
            vals.append(float(d.get(b, default_score)))
    t = torch.tensor(vals, dtype=torch.float32) * mask
    s = float(t.sum())
    if s <= 0:
        return None
    return t / s


def _decision_branch_weights_tensor(
    spec: str | dict | None,
    excluded: frozenset[str],
    default_weight: float = 1.0,
) -> torch.Tensor | None:
    """
    显式支路相对权重（与 branch_prior 共用 `RRTMIL:2,AMIL:1` 或 JSON 字符串格式）。
    未列出的未排除分支默认 default_weight；<=0 的项视为不参与（该路权重为 0，其余再归一化）。
    仅在 decision_fusion=weighted 时使用；若解析失败或有效权重和为 0 则返回 None。
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        d: dict[str, float] = {}
        for k, v in spec.items():
            n = _normalize_ensemble_branch(str(k))
            if n:
                d[n] = float(v)
    else:
        s = str(spec).strip()
        if not s:
            return None
        parsed = _branch_prior_dict_from_spec(s)
        if not parsed:
            return None
        d = parsed
    w = torch.zeros(len(ENSEMBLE_BRANCH_ORDER), dtype=torch.float32)
    for i, b in enumerate(ENSEMBLE_BRANCH_ORDER):
        if b in excluded:
            continue
        if b in d:
            val = float(d[b])
            if val <= 0:
                continue
            w[i] = val
        else:
            w[i] = float(default_weight)
    ssum = float(w.sum())
    if ssum <= 0:
        return None
    return w / ssum
