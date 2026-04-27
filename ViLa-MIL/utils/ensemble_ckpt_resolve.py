"""
五路基线集成（如 EnsembleDecision）：自动解析五个基模型预训练权重路径。

默认从 ViLa-MIL 根目录下的
  uploaded_features/best_models.json
  uploaded_features/tasks.json
读取各基模型在当前癌种、mode 下的 bestTaskId，并在对应任务的 resultsDir 中查找
  s_{fold_idx}_checkpoint.pt
（与 main_LUSC 多折训练保存方式一致）。

若未指定 ensemble_ckpt_dir，core_utils 将调用本模块；若仍无法凑齐 5 个文件则打印告警并跳过加载。
"""

from __future__ import annotations

import json
import os
from typing import Any


def _read_json(path: str, default: Any) -> Any:
    if not path or not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _best_key(cancer: str, model_type: str, mode: str) -> str:
    return f"{str(cancer).strip()}:{str(model_type).strip()}:{str(mode).strip()}"


def _tasks_index(tasks_path: str) -> dict[str, dict[str, Any]]:
    data = _read_json(tasks_path, {"tasks": []})
    tasks = data.get("tasks") if isinstance(data, dict) else []
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(tasks, list):
        return out
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("taskId") or t.get("id") or "").strip()
        if tid:
            out[tid] = t
    return out


def resolve_fold_checkpoint(results_dir: str, fold_idx: int) -> str | None:
    """在单次训练任务的 resultsDir 下解析当前折的权重文件。"""
    if not results_dir or not os.path.isdir(results_dir):
        return None
    fold_idx = int(fold_idx)
    flat = os.path.join(results_dir, f"s_{fold_idx}_checkpoint.pt")
    if os.path.isfile(flat):
        return flat
    nested = os.path.join(results_dir, str(fold_idx), f"s_{fold_idx}_checkpoint.pt")
    if os.path.isfile(nested):
        return nested
    # 宽松：resultsDir 下任意子目录中的 s_{fold}_checkpoint.pt
    try:
        for root, _dirs, files in os.walk(results_dir):
            name = f"s_{fold_idx}_checkpoint.pt"
            if name in files:
                return os.path.join(root, name)
    except Exception:
        pass
    return None


def auto_resolve_ensemble_pretrained_paths(
    *,
    fold_idx: int,
    cancer: str,
    mode: str,
    vila_root: str,
    best_models_path: str | None = None,
    tasks_path: str | None = None,
) -> dict[str, str] | None:
    """
    返回 load_pretrained 所需的五个路径键：rrt_ckpt, amil_ckpt, wikg_ckpt, dsmil_ckpt, s4mil_ckpt。
    无法凑齐 5 个则返回 None。
    """
    bm_path = best_models_path or os.path.join(vila_root, "uploaded_features", "best_models.json")
    tk_path = tasks_path or os.path.join(vila_root, "uploaded_features", "tasks.json")

    bm = _read_json(bm_path, {})
    by_key = (bm.get("byKey") or {}) if isinstance(bm, dict) else {}
    tasks_by_id = _tasks_index(tk_path)

    order: list[tuple[str, str]] = [
        ("rrt_ckpt", "RRTMIL"),
        ("amil_ckpt", "AMIL"),
        ("wikg_ckpt", "WiKG"),
        ("dsmil_ckpt", "DSMIL"),
        ("s4mil_ckpt", "S4MIL"),
    ]
    out: dict[str, str] = {}
    for ck_key, mtype in order:
        key = _best_key(cancer, mtype, mode)
        rec = by_key.get(key)
        if not isinstance(rec, dict):
            return None
        tid = str(rec.get("bestTaskId") or "").strip()
        if not tid:
            return None
        task = tasks_by_id.get(tid)
        if not task:
            return None
        rd = str(task.get("resultsDir") or "").strip()
        ck = resolve_fold_checkpoint(rd, fold_idx)
        if not ck:
            return None
        out[ck_key] = ck
    return out
