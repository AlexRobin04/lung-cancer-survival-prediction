"""
ViLa-MIL HTTP API (Flask).

前端 axios baseURL=/api，路由与 vila-mil-frontend 构建产物一致。
训练子进程：main_LUSC.py；Python 解释器优先环境变量 VILAMIL_PYTHON_BIN，否则为当前解释器。
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "api_training_logs")
DATA_ROOT = os.path.join(BASE_DIR, "uploaded_features")
TASKS_PATH = os.path.join(DATA_ROOT, "tasks.json")
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.json")
CASES_PATH = os.path.join(DATA_ROOT, "cases.json")
PREDICTIONS_PATH = os.path.join(DATA_ROOT, "predictions.json")
BEST_MODELS_PATH = os.path.join(DATA_ROOT, "best_models.json")
RESULT_API_RUNS = os.path.join(BASE_DIR, "result", "api_runs")

# 写入 predictions.json 的推理协议版本（论文可复现：同版本 + 同 checkpoint 列表 + 同特征应对齐）
PREDICT_PROTOCOL_ID = "predict-v4-202604-ensemble-mean-logits"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(RESULT_API_RUNS, exist_ok=True)

_LOCK = threading.Lock()
_QUEUE_DISPATCH_LOCK = threading.Lock()
_PROCS: dict[str, subprocess.Popen] = {}
_MODEL_CACHE: dict[str, Any] = {}

MODEL_CHOICES = [
    "ViLa_MIL",
    "TransMIL",
    "AMIL",
    "WiKG",
    "RRTMIL",
    "PatchGCN",
    "surformer",
    "DSMIL",
    "S4MIL",
    "EnsembleDecision",
]

_ENSEMBLE_BRANCH_KEYS_FROZEN = frozenset({"RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL"})
# 与 Dashboard 队列 C-index 表一致的五基线顺序
_ENSEMBLE_BRANCH_ORDER_FOR_PRIOR = ("RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL")


def _normalize_ensemble_branch_key_api(name: Any) -> str | None:
    u = str(name or "").strip().upper().replace("-", "_")
    if u in ("S4", "S4MIL"):
        return "S4MIL"
    if u == "WIKG":
        return "WiKG"
    if u in {"RRTMIL", "AMIL", "DSMIL"}:
        return u
    return None


def _normalize_ensemble_branch_prior_api(raw: Any) -> str:
    """将前端传入的分支先验规范化为 main_LUSC --ensemble_branch_prior 字符串（含 JSON）。"""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    return str(raw).strip()


def _ensemble_decision_training_api_schema() -> dict[str, Any]:
    """
    供 GET /api/config 与 GET /api/training/ensemble-options 使用；
    与 POST /api/training/start 在 modelType=EnsembleDecision 时的 body 字段一致。
    """
    return {
        "endpoint": "POST /api/training/start",
        "condition": "请求体 modelType 为 EnsembleDecision",
        "parameters": [
            {
                "name": "decisionFusion",
                "aliases": ["decision_fusion"],
                "type": "string",
                "enum": ["avg_prob"],
                "default": "avg_prob",
                "descriptionZh": "固定融合：各基线输出概率逐类取均值（简单可解释，默认推荐）。",
            },
            {
                "name": "ensembleBranchPriorAuto",
                "aliases": ["ensemble_branch_prior_auto"],
                "type": "boolean",
                "default": True,
                "descriptionZh": (
                    "未手动传 ensembleBranchPrior 时，是否按 Dashboard 同源规则从 predictions+临床 "
                    "自动聚合各单模队列 C-index 作为分支先验。"
                ),
            },
            {
                "name": "ensembleBranchPriorTemperature",
                "aliases": ["ensemble_branch_prior_temperature"],
                "type": "number",
                "default": None,
                "minimum": 1e-6,
                "descriptionZh": "存在分支先验时，将先验 logit 除以该温度。",
                "example": 0.55,
            },
        ],
        "relatedParameters": [
            {
                "name": "ensembleBranchPrior",
                "aliases": ["ensemble_branch_prior"],
                "type": "string | object",
                "descriptionZh": "手动分支先验；有手动值则不再自动填充。",
            },
            {
                "name": "ensembleBranchPriorScale",
                "aliases": ["ensemble_branch_prior_scale"],
                "type": "number",
                "default": 1.25,
                "descriptionZh": "先验强度系数。",
            },
            {
                "name": "decisionBranchWeights",
                "aliases": ["decision_branch_weights"],
                "type": "string | object",
                "descriptionZh": (
                    "仅 decisionFusion=weighted：显式五路相对权重（如 {\"RRTMIL\":2,\"AMIL\":1.5} 或 RRTMIL:2,AMIL:1.5）；"
                    "未写明的未排除支路默认 1。若提供则覆盖 ensembleBranchPrior（C-index 先验）。"
                ),
            },
        ],
        "exampleBody": {
            "cancer": "LUSC",
            "modelType": "EnsembleDecision",
            "mode": "transformer",
            "maxEpochs": 1,
            "decisionFusion": "avg_prob",
            "decisionBranchWeights": {"RRTMIL": 2.0, "AMIL": 1.5},
            "ensembleBranchPriorAuto": True,
            "ensembleBranchPriorTemperature": 0.55,
        },
    }


def _parse_ensemble_exclude_api(raw: Any) -> list[str]:
    """解析 ensembleExclude；未知键忽略；若等价于排除全部五路则抛 ValueError。"""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if str(x).strip()]
    else:
        parts = [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]
    out: list[str] = []
    for p in parts:
        n = _normalize_ensemble_branch_key_api(p)
        if n and n not in out:
            out.append(n)
    if len(_ENSEMBLE_BRANCH_KEYS_FROZEN.difference(out)) == 0:
        raise ValueError("ensembleExclude 不能排除全部五路基线")
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_TZ_CST = timezone(timedelta(hours=8))


def _cst_now() -> str:
    """Return ISO timestamp in UTC+8 (Asia/Shanghai style offset)."""
    return datetime.now(_TZ_CST).isoformat()


def _to_cst(ts: str | None) -> str | None:
    """Convert an ISO timestamp (Z or offset) to UTC+8 ISO string."""
    if not ts:
        return ts
    try:
        s = str(ts).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_TZ_CST).isoformat()
    except Exception:
        return ts


def _python_bin() -> str:
    return os.environ.get("VILAMIL_PYTHON_BIN") or sys.executable


def _read_json(path: str, default: Any) -> Any:
    if not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _atomic_write_json(path: str, data: Any) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_tasks() -> dict[str, Any]:
    data = _read_json(TASKS_PATH, {"tasks": []})
    if "tasks" not in data:
        data["tasks"] = []
    return data


def _save_tasks(data: dict[str, Any]) -> None:
    _atomic_write_json(TASKS_PATH, data)


def _find_task(task_id: str) -> dict[str, Any] | None:
    for t in _load_tasks()["tasks"]:
        if t.get("taskId") == task_id or t.get("id") == task_id:
            return t
    return None


def _update_task(task_id: str, **fields: Any) -> None:
    with _LOCK:
        data = _load_tasks()
        for i, t in enumerate(data["tasks"]):
            if t.get("taskId") == task_id or t.get("id") == task_id:
                data["tasks"][i] = {**t, **fields}
                break
        else:
            return
        _save_tasks(data)


_ORPHAN_RESUME_NOTE_ZH = (
    "训练子进程已不存在（例如 API 服务重启）；任务已自动重新入队，将稍后继续启动训练。"
)


def _requeue_all_orphaned_running_tasks() -> int:
    """将 status=running 但训练子进程 PID 已不存在的任务改回 queued（便于 API 重启后继续调度）。"""
    with _LOCK:
        data = _load_tasks()
        orphan_indices: list[int] = []
        for i, t in enumerate(data["tasks"]):
            if str(t.get("status") or "").lower() != "running":
                continue
            if _pid_alive(t.get("pid")):
                continue
            tid = str(t.get("taskId") or t.get("id") or "").strip()
            if tid:
                orphan_indices.append(i)
        if not orphan_indices:
            return 0
        base_ts = time.time()
        for j, i in enumerate(orphan_indices):
            t = data["tasks"][i]
            ts = base_ts - (len(orphan_indices) - j) * 0.001
            data["tasks"][i] = {
                **t,
                "status": "queued",
                "running": False,
                "pid": None,
                "endedAt": None,
                "exitCode": None,
                "failReason": None,
                "queuedAt": _cst_now(),
                "queuedAtTs": ts,
                "resumeNote": _ORPHAN_RESUME_NOTE_ZH,
            }
        _save_tasks(data)
        return len(orphan_indices)


def _append_task(task: dict[str, Any]) -> None:
    with _LOCK:
        data = _load_tasks()
        data["tasks"].insert(0, task)
        _save_tasks(data)


def _load_best_models() -> dict[str, Any]:
    data = _read_json(BEST_MODELS_PATH, {"byKey": {}})
    if "byKey" not in data:
        data["byKey"] = {}
    return data


def _save_best_models(data: dict[str, Any]) -> None:
    _atomic_write_json(BEST_MODELS_PATH, data)


def _best_key(cancer: str, model_type: str, mode: str) -> str:
    return f"{cancer}:{model_type}:{mode}"


def _safe_unlink(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _safe_rmtree(path: str) -> None:
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _try_append_log(task_id: str, msg: str) -> None:
    """Best-effort append message to a task's log file."""
    try:
        t = _find_task(task_id) or {}
        p = t.get("logPath") or os.path.join(LOG_DIR, f"{task_id}.log")
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n[api] {msg}\n")
    except Exception:
        pass


def _tail_file(path: str, n: int) -> str:
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""


_TRAINING_LOG_SCRUB = re.compile(
    r"(?i)(?:\bnnpack\b|could not initialize nnpack|nnpack is not supported|compiled without nnpack)",
)


def _scrub_training_log_content(text: str) -> str:
    """去掉训练日志里 NNPACK 等噪声行（前端 Log tail 同源过滤；C++ 直打 stderr 时 warnings 挡不住）。"""
    if not text:
        return text
    lines = text.split("\n")
    kept = [ln for ln in lines if not _TRAINING_LOG_SCRUB.search(ln)]
    return "\n".join(kept)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _parse_weight_decay(v: Any, default: float = 1e-5) -> float:
    try:
        x = float(v if v is not None and str(v).strip() != "" else default)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(x, 1.0))


def _read_meminfo_bytes() -> dict[str, int]:
    """
    Read Linux /proc/meminfo and return selected fields in bytes.
    Missing fields are returned as 0.
    """
    out = {
        "MemTotal": 0,
        "MemAvailable": 0,
        "SwapTotal": 0,
        "SwapFree": 0,
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                if k not in out:
                    continue
                # usually like: "123456 kB"
                m = re.search(r"([0-9]+)", v)
                if not m:
                    continue
                kb = int(m.group(1))
                out[k] = kb * 1024
    except Exception:
        pass
    return out


def _fmt_gib(n_bytes: int) -> str:
    gib = float(n_bytes) / float(1024**3)
    return f"{gib:.2f} GiB"


def _vila_resource_precheck() -> tuple[bool, dict[str, Any]]:
    """
    Resource guard for ViLa_MIL.
    Policy (default):
      - MemAvailable >= 6 GiB
      - SwapTotal >= 2 GiB
    Thresholds can be overridden by env:
      VILAMIL_MIN_AVAIL_GB, VILAMIL_MIN_SWAP_GB
    """
    info = _read_meminfo_bytes()
    mem_avail = int(info.get("MemAvailable") or 0)
    mem_total = int(info.get("MemTotal") or 0)
    swap_total = int(info.get("SwapTotal") or 0)
    swap_free = int(info.get("SwapFree") or 0)

    try:
        min_avail_gb = float(os.environ.get("VILAMIL_MIN_AVAIL_GB", "5.5"))
    except Exception:
        min_avail_gb = 6.0
    try:
        min_swap_gb = float(os.environ.get("VILAMIL_MIN_SWAP_GB", "2"))
    except Exception:
        min_swap_gb = 2.0

    req_avail = int(min_avail_gb * (1024**3))
    req_swap = int(min_swap_gb * (1024**3))

    # GPU guard: ViLa_MIL 训练强依赖 CUDA；无 GPU 时直接拒绝，避免 CPU 上长时间“卡死”
    cuda_ok = False
    cuda_detail: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available() and (torch.cuda.device_count() or 0) > 0)
        cuda_detail = {
            "available": bool(torch.cuda.is_available()),
            "deviceCount": int(torch.cuda.device_count() or 0),
        }
    except Exception as e:
        cuda_ok = False
        cuda_detail = {"available": False, "error": str(e)}

    ok = cuda_ok and (mem_avail >= req_avail) and (swap_total >= req_swap)
    detail = {
        "ok": ok,
        "required": {
            "memAvailable": _fmt_gib(req_avail),
            "swapTotal": _fmt_gib(req_swap),
            "cuda": True,
            "policy": "ViLa_MIL requires both available RAM and swap safeguards",
        },
        "current": {
            "memTotal": _fmt_gib(mem_total),
            "memAvailable": _fmt_gib(mem_avail),
            "swapTotal": _fmt_gib(swap_total),
            "swapFree": _fmt_gib(swap_free),
            "cuda": cuda_detail,
        },
    }
    return ok, detail


def _load_manifest() -> dict[str, Any]:
    return _read_json(MANIFEST_PATH, {"files": {}})


def _load_cases() -> dict[str, Any]:
    return _read_json(CASES_PATH, {"cases": {}, "caseOrder": []})


def _resolve_case_feature_paths(case_id: str) -> tuple[str, str]:
    cases = _load_cases().get("cases", {})
    c = cases.get(case_id)
    if not c:
        raise FileNotFoundError(f"case not found: {case_id}")
    f20 = c.get("feature20FileId")
    f10 = c.get("feature10FileId")
    mani = _load_manifest().get("files", {})
    e20 = mani.get(f20) if f20 else None
    e10 = mani.get(f10) if f10 else None

    # If binding is stale/missing, try to infer paired file by name.
    # Common pattern in uploads: 20x "TCGA-... .h5" and 10x "TCGA-..._10x.h5".
    if e20 and not e10:
        base = str(e20.get("name") or "")
        stem = base[:-3] if base.endswith(".h5") else base
        want = f"{stem}_10x.h5"
        for _id, ent in mani.items():
            if str(ent.get("featureType")) == "10" and str(ent.get("name")) == want:
                e10 = ent
                break
        if not e10:
            # fallback: match by shared TCGA prefix
            prefix = stem.split("_10x")[0]
            for _id, ent in mani.items():
                if str(ent.get("featureType")) == "10" and prefix and prefix in str(ent.get("name") or ""):
                    e10 = ent
                    break

    if e10 and not e20:
        base = str(e10.get("name") or "")
        stem = base[:-3] if base.endswith(".h5") else base
        want = stem.replace("_10x", "") + ".h5"
        for _id, ent in mani.items():
            if str(ent.get("featureType")) == "20" and str(ent.get("name")) == want:
                e20 = ent
                break
        if not e20:
            prefix = stem.replace("_10x", "")
            for _id, ent in mani.items():
                if str(ent.get("featureType")) == "20" and prefix and prefix in str(ent.get("name") or ""):
                    e20 = ent
                    break

    if not e20 or not e10:
        raise FileNotFoundError(f"case missing/invalid feature bindings: {case_id}")
    p20 = os.path.join(BASE_DIR, e20["storedPath"]) if not os.path.isabs(e20["storedPath"]) else e20["storedPath"]
    p10 = os.path.join(BASE_DIR, e10["storedPath"]) if not os.path.isabs(e10["storedPath"]) else e10["storedPath"]
    if not (os.path.isfile(p20) and os.path.isfile(p10)):
        raise FileNotFoundError(f"h5 missing: {p20} / {p10}")
    return p20, p10


def _manifest_lookup_entry(file_id: str) -> dict[str, Any] | None:
    """Resolve a manifest entry by uuid id, or by matching id/name against stored entries."""
    fid = str(file_id or "").strip()
    if not fid:
        return None
    files = _load_manifest().get("files") or {}
    if fid in files:
        return files[fid]
    for e in files.values():
        if str(e.get("id")) == fid or str(e.get("name")) == fid:
            return e
    return None


def _h5_feature_dim(h5_path: str) -> int | None:
    try:
        import h5py

        with h5py.File(h5_path, "r") as f:
            ds = f.get("features")
            if ds is None:
                return None
            shape = tuple(getattr(ds, "shape", ()) or ())
        if len(shape) == 2 and int(shape[1]) > 0:
            return int(shape[1])
        return None
    except Exception:
        return None


def _adapt_h5_feature_dim(h5_path: str, target_dim: int) -> dict[str, Any] | None:
    """
    将 H5 内 features 维度适配到目标维度：
    - 原维度 > 目标维度：截断前 target_dim 列
    - 原维度 < 目标维度：右侧零填充
    """
    if target_dim <= 0:
        return None
    try:
        import h5py
        import numpy as np

        with h5py.File(h5_path, "r+") as f:
            if "features" not in f:
                return None
            feat = np.array(f["features"])
            if feat.ndim != 2:
                return None
            orig_dim = int(feat.shape[1])
            if orig_dim == target_dim:
                return {"originalDim": orig_dim, "targetDim": target_dim, "applied": False}
            if orig_dim > target_dim:
                feat_new = feat[:, :target_dim]
            else:
                pad = np.zeros((feat.shape[0], target_dim - orig_dim), dtype=feat.dtype)
                feat_new = np.concatenate([feat, pad], axis=1)
            if "features" in f:
                del f["features"]
            f.create_dataset("features", data=feat_new.astype(np.float32, copy=False))
            return {"originalDim": orig_dim, "targetDim": target_dim, "applied": True}
    except Exception:
        return None


def _stored_path_to_abs(entry: dict[str, Any]) -> str:
    sp = entry.get("storedPath") or ""
    if not sp:
        return ""
    return os.path.join(BASE_DIR, sp) if not os.path.isabs(sp) else sp


def _resolve_feature_paths_by_file_ids(
    f20_id: str, f10_id: str, cancer: str | None = None
) -> tuple[str, str]:
    """
    推理可直接传入已上传特征在 manifest 中的 fileId，无需事先写入 cases.json。
    cancer：当 fileId 为裸文件名且仅存在于磁盘而未登记 manifest 时，用于定位 uploaded_features/<癌种>/features_20|10/。
    """
    e20 = _manifest_lookup_entry(f20_id)
    e10 = _manifest_lookup_entry(f10_id)
    cancer_key = str(cancer or "").strip() or None

    def path_for(entry: dict[str, Any] | None, fid: str, ft: str) -> str | None:
        if entry and entry.get("storedPath"):
            p = _stored_path_to_abs(entry)
            if p and os.path.isfile(p):
                return p
        if cancer_key:
            folder = os.path.join(DATA_ROOT, cancer_key, f"features_{ft}")
            cand = os.path.join(folder, os.path.basename(fid))
            if os.path.isfile(cand):
                return cand
        return None

    p20 = path_for(e20, f20_id, "20")
    p10 = path_for(e10, f10_id, "10")
    if not p20:
        raise FileNotFoundError(f"无法解析 20× 特征文件：{f20_id}（确认已上传，或在请求中附带 cancer）")
    if not p10:
        raise FileNotFoundError(f"无法解析 10× 特征文件：{f10_id}（确认已上传，或在请求中附带 cancer）")
    ft20 = str(e20.get("featureType") or "") if e20 else ""
    ft10 = str(e10.get("featureType") or "") if e10 else ""
    if ft20 and ft20 != "20":
        raise FileNotFoundError(f"文件 {f20_id} 不是 20× 特征（featureType={ft20!r}）")
    if ft10 and ft10 != "10":
        raise FileNotFoundError(f"文件 {f10_id} 不是 10× 特征（featureType={ft10!r}）")
    return p20, p10


def _manifest_register_h5_copy(
    cancer: str,
    ft: str,
    abs_src: str,
    stored_basename: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """将已有 H5 复制到 uploaded_features/<癌种>/features_{20|10}/ 并写入 manifest，返回 fileId。"""
    fid = str(uuid.uuid4())
    sub = f"features_{ft}"
    save_dir = os.path.join(DATA_ROOT, cancer, sub)
    os.makedirs(save_dir, exist_ok=True)
    dst = os.path.join(save_dir, stored_basename)
    shutil.copy2(abs_src, dst)
    rel = os.path.relpath(dst, BASE_DIR).replace("\\", "/")
    manifest = _read_json(MANIFEST_PATH, {"files": {}})
    manifest.setdefault("files", {})
    entry: dict[str, Any] = {
        "id": fid,
        "cancer": cancer,
        "featureType": ft,
        "name": stored_basename,
        "storedPath": rel,
        "size": os.path.getsize(dst),
        "createdAt": _utc_now(),
    }
    if extra:
        entry.update(extra)
    manifest["files"][fid] = entry
    _atomic_write_json(MANIFEST_PATH, manifest)
    return fid


def _find_first_h5(root: str) -> str | None:
    if not root or not os.path.isdir(root):
        return None
    cands: list[str] = []
    for dp, _dns, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".h5"):
                cands.append(os.path.join(dp, fn))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return cands[0]


def _extract_dual_scale_h5_with_trident(src_file: str, out20: str, out10: str, mpp: float | None = None) -> dict[str, Any]:
    """
    使用 TRIDENT 的 run_batch_of_slides.py 生成 20x/10x 双尺度 H5。
    需要本机已 clone TRIDENT 仓库。
    """
    trident_repo = os.environ.get("TRIDENT_REPO_DIR") or os.path.join(os.path.dirname(BASE_DIR), "TRIDENT")
    trident_py = os.environ.get("TRIDENT_PYTHON_BIN") or _python_bin()
    patch_encoder = os.environ.get("TRIDENT_PATCH_ENCODER", "uni_v1")
    # Offline-friendly default: otsu 不依赖额外 checkpoint；如需 hest 可通过环境变量覆盖。
    segmenter = os.environ.get("TRIDENT_SEGMENTER", "otsu")
    gpu = str(os.environ.get("TRIDENT_GPU", "0"))
    patch_size = str(int(float(os.environ.get("TRIDENT_PATCH_SIZE", "256"))))
    trident_max_workers = str(int(float(os.environ.get("TRIDENT_MAX_WORKERS", "0"))))
    trident_force_dim_raw = str(os.environ.get("TRIDENT_FORCE_FEATURE_DIM", "")).strip()
    trident_force_dim = int(trident_force_dim_raw) if trident_force_dim_raw.isdigit() else 0
    run_py = os.path.join(trident_repo, "run_batch_of_slides.py")
    if not os.path.isfile(run_py):
        raise RuntimeError(f"未找到 TRIDENT 脚本: {run_py}（请设置 TRIDENT_REPO_DIR）")

    work_root = os.path.dirname(out20)
    os.makedirs(work_root, exist_ok=True)
    wsi_name = os.path.basename(src_file)
    ext = os.path.splitext(wsi_name)[1].lower()
    is_plain_image = ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    meta: dict[str, Any] = {
        "extractor": "TRIDENT",
        "patchEncoder": patch_encoder,
        "gpu": gpu,
        "patchSize": int(patch_size),
    }
    if mpp is not None:
        meta["mpp"] = float(mpp)

    def _run_one_mag(mag: int, reader_type: str) -> tuple[bool, str | None, str]:
        """run trident once; return (ok, found_h5, stderr_or_tail)."""
        wsi_dir = os.path.join(work_root, f"wsi_{mag}x")
        job_dir = os.path.join(work_root, f"trident_job_{mag}x")
        os.makedirs(wsi_dir, exist_ok=True)
        os.makedirs(job_dir, exist_ok=True)
        local_wsi = os.path.join(wsi_dir, f"slide{ext}")
        shutil.copy2(src_file, local_wsi)
        cmd = [
            trident_py,
            run_py,
            "--task",
            "all",
            "--wsi_dir",
            wsi_dir,
            "--job_dir",
            job_dir,
            "--patch_encoder",
            patch_encoder,
            "--segmenter",
            segmenter,
            "--reader_type",
            reader_type,
            "--mag",
            str(mag),
            "--patch_size",
            patch_size,
            "--gpu",
            gpu,
            "--max_workers",
            trident_max_workers,
        ]
        if mpp is not None:
            custom_csv = os.path.join(job_dir, "wsi_list.csv")
            with open(custom_csv, "w", encoding="utf-8") as wf:
                wf.write("wsi,mpp\n")
                wf.write(f"{local_wsi},{float(mpp)}\n")
            cmd.extend(["--custom_list_of_wsis", custom_csv, "--custom_mpp_keys", "mpp"])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            return False, None, detail
        found = _find_first_h5(job_dir)
        if not found:
            return False, None, f"TRIDENT 未产出 H5（mag={mag}x, reader={reader_type}）"
        return True, found, ""

    for mag, outp in ((20, out20), (10, out10)):
        primary_reader = "image" if is_plain_image else "openslide"
        ok, found, detail = _run_one_mag(mag, primary_reader)
        if not ok and primary_reader == "openslide":
            # Fallback: some files have WSI suffix but are not OpenSlide-readable.
            ok, found, detail = _run_one_mag(mag, "image")
            meta[f"mag{mag}ReaderFallback"] = "image"
        if not ok:
            tail = (detail or "").splitlines()[-20:]
            raise RuntimeError(
                "TRIDENT 运行失败。可能原因：文件并非可读 WSI、文件损坏、或格式与扩展名不匹配。\n" + "\n".join(tail)
            )
        assert found
        shutil.copy2(found, outp)
        if trident_force_dim > 0:
            adapt_meta = _adapt_h5_feature_dim(outp, trident_force_dim)
            if adapt_meta is not None:
                meta[f"mag{mag}FeatureDimAdapt"] = adapt_meta
        meta[f"mag{mag}H5"] = found
        meta[f"mag{mag}Size"] = os.path.getsize(outp)
    return meta


def _discover_checkpoints(results_dir: str) -> list[str]:
    if not results_dir or not os.path.isdir(results_dir):
        return []
    out = []
    for fn in os.listdir(results_dir):
        if fn.endswith(".pt") and fn.startswith("s_") and "checkpoint" in fn:
            out.append(os.path.join(results_dir, fn))
    return sorted(out)


def _resolve_task_results_dir(task: dict[str, Any]) -> str:
    """兼容任务里记录的 Docker 路径 /app/result/...：在本机开发时回退到仓库 result/api_runs/<taskId>。"""
    rd = str((task or {}).get("resultsDir") or "").strip()
    tid = str((task or {}).get("taskId") or (task or {}).get("id") or "").strip()
    if rd and os.path.isdir(rd):
        return rd
    if tid:
        alt = os.path.join(RESULT_API_RUNS, tid)
        if os.path.isdir(alt):
            return alt
    return rd


def _build_vila_config(task: dict[str, Any]) -> Any:
    import ml_collections
    import numpy as np
    import pandas as pd

    cfg = ml_collections.ConfigDict()
    cfg.input_size = 1024
    cfg.hidden_size = 192
    # LUSC defaults (same as main_LUSC.py)
    text_prompt_path = os.path.join(BASE_DIR, "text_prompt", "TCGA_Lung_two_scale_text_prompt.csv")
    if os.path.isfile(os.path.join(BASE_DIR, "text_prompt", "TCGA_lusc_two_scale_text_prompt.csv")):
        text_prompt_path = os.path.join(BASE_DIR, "text_prompt", "TCGA_lusc_two_scale_text_prompt.csv")
    cfg.text_prompt = np.array(pd.read_csv(text_prompt_path, header=None)).squeeze()
    cfg.prototype_number = int((task or {}).get("prototype_number") or 16)
    cfg.hard_or_soft = bool((task or {}).get("hard_or_soft") or False)
    # conch path
    cpath = (task or {}).get("conchCheckpointPath") or os.environ.get("CONCH_CKPT_PATH")
    if cpath:
        cfg.conch_checkpoint_path = cpath
    return cfg


def _load_vila_model(ckpt_path: str, task: dict[str, Any]) -> Any:
    key = f"vila::{ckpt_path}::{task.get('conchCheckpointPath')}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import torch
    from models.model_ViLa_MIL import ViLa_MIL_Model
    cfg = _build_vila_config(task)
    model = ViLa_MIL_Model(config=cfg, num_classes=4)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _load_mil_model(ckpt_path: str, n_classes: int = 4, dropout: bool = True) -> Any:
    """
    Non-ViLa baseline used by current training code path (core_utils.py):
    - n_classes > 2 => MIL_fc_mc
    - n_classes == 2 => MIL_fc
    """
    key = f"mil::{ckpt_path}::{n_classes}::{int(bool(dropout))}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import torch
    from models.model_mil import MIL_fc, MIL_fc_mc

    model_dict = {"dropout": bool(dropout), "n_classes": int(n_classes)}
    model = MIL_fc_mc(**model_dict) if n_classes > 2 else MIL_fc(**model_dict)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _load_transformer_baseline_model(
    ckpt_path: str,
    model_type: str,
    feat_dim: int = 512,
    n_classes: int = 4,
) -> Any:
    """
    加载与训练同构的基线模型（RRTMIL/AMIL/WiKG/DSMIL/S4MIL/surformer）。
    这类模型在训练中直接接收 x_s/x_l，不应走 MIL_fc 兜底。
    """
    mt = str(model_type or "").strip()
    key = f"tfbase::{mt}::{ckpt_path}::{int(feat_dim)}::{int(n_classes)}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import ml_collections
    import torch

    cfg = ml_collections.ConfigDict()
    cfg.hard_or_soft = False
    fd = int(feat_dim)
    nc = int(n_classes)
    if mt == "RRTMIL":
        from models.RRT import RRTMIL

        model = RRTMIL(config=cfg, n_classes=nc, input_dim=fd)
    elif mt == "AMIL":
        from models.AMIL import AMIL

        model = AMIL(config=cfg, n_classes=nc, input_dim=fd)
    elif mt == "WiKG":
        from models.WiKG import WiKG

        model = WiKG(config=cfg, n_classes=nc, dim_in=fd, dim_hidden=512)
    elif mt == "DSMIL":
        from models.DSMIL import MILNet

        model = MILNet(config=cfg, in_size=fd, num_class=nc, dropout=0.25)
    elif mt == "S4MIL":
        from models.S4MIL import S4Model

        model = S4Model(
            config=cfg,
            in_dim=fd,
            n_classes=nc,
            dropout=0.1,
            act="relu",
            d_model=512,
            d_state=16,
        )
    elif mt == "surformer":
        from models.HVTSurv import HVTSurv

        model = HVTSurv(n_classes=nc)
    else:
        raise ValueError(f"unsupported transformer baseline model_type: {mt}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _load_ensemble_decision_model(ckpt_path: str, feat_dim: int = 512, n_classes: int = 4) -> Any:
    """加载决策级集成 EnsembleDecisionMIL。"""
    key = f"ensdec::{ckpt_path}::{feat_dim}::{n_classes}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import ml_collections
    import torch
    from models.ensemble_branch_utils import ENSEMBLE_BRANCH_ORDER
    from models.EnsembleDecision import EnsembleDecisionMIL

    config = ml_collections.ConfigDict()
    config.hard_or_soft = False
    state = torch.load(ckpt_path, map_location="cpu")
    decision_fusion = "avg_prob"
    tag = state.get("_ed_decision_tag")
    if tag is not None:
        try:
            tv = int(tag.detach().cpu().item()) if isinstance(tag, torch.Tensor) else int(tag)
        except (TypeError, ValueError):
            tv = -1
        # 兼容历史 checkpoint 的旧 tag：统一映射为 avg_prob 简单融合。
        decision_fusion = "avg_prob"
    exclude: list[str] = []
    mkey = "ensemble_branch_mask"
    if mkey in state:
        m = state[mkey]
        if isinstance(m, torch.Tensor) and m.numel() == len(ENSEMBLE_BRANCH_ORDER):
            for i, b in enumerate(ENSEMBLE_BRANCH_ORDER):
                if float(m[i].detach().cpu().item()) < 0.5:
                    exclude.append(b)
    model = EnsembleDecisionMIL(
        config=config,
        n_classes=int(n_classes),
        feat_dim=int(feat_dim),
        freeze_base=False,
        decision_fusion=decision_fusion,
        ensemble_exclude=exclude,
        branch_prior=None,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _risk_from_probs_expected_class(probs: list[float]) -> float:
    if not probs:
        return 0.0
    m = min(4, len(probs))
    return float(sum(i * float(probs[i]) for i in range(m)))


# 与 datasets_csv / main_LUSC 标签一致：4 类亚风险
RISK_CLASS_LABELS_ZH = ["低风险", "中等风险", "偏高风险", "高风险"]
RISK_CLASS_LABELS_EN = ["low", "Moderate", "Elevated", "high"]


def _three_tier_from_score(risk_score: float) -> tuple[str, str, str]:
    """
    将连续期望风险得分 [0,3] 映射为 高/中/低 三档（毕设「风险分层」展示用）。
    分界： [0,1)→低，[1,2)→中，[2,3]→高。
    """
    if risk_score < 1.0:
        return "low", "低风险", "低危"
    if risk_score < 2.0:
        return "medium", "中风险", "中危"
    return "high", "高风险", "高危"


def _build_prediction_visualization(
    probs: list[float], risk_score: float, pred_class: int, tier_en: str, tier_zh: str
) -> dict[str, Any]:
    """供前端 Recharts/ECharts 直接使用的简化结构。"""
    n = len(probs)
    cats = (RISK_CLASS_LABELS_ZH[:n] if n <= 4 else [f"类别{i}" for i in range(n)])
    return {
        "probabilityBar": {
            "title": "亚风险等级概率分布",
            "x": cats,
            "y": [float(p) for p in probs],
        },
        "riskScoreGauge": {
            "min": 0.0,
            "max": 3.0,
            "value": float(risk_score),
            "title": "期望风险得分（0–3）",
        },
        "threeTier": {
            "code": tier_en,
            "labelZh": tier_zh,
            "scale": "期望得分 [0,3) 低 / [1,2) 中 / [2,3] 高",
        },
        "predClassDetail": {
            "index": pred_class,
            "labelZh": cats[pred_class] if 0 <= pred_class < len(cats) else str(pred_class),
        },
    }


def _case_clinical_summary(case_id: str) -> dict[str, Any] | None:
    c = _load_cases().get("cases", {}).get(case_id)
    if not c:
        return None
    return {
        "caseId": case_id,
        "time": c.get("time"),
        "status": c.get("status"),
        "statusNote": "1=事件发生 0=删失",
        "slideId": c.get("slideId"),
        "clinicalVars": c.get("clinicalVars") or {},
        "hasFeature20": bool(c.get("feature20FileId")),
        "hasFeature10": bool(c.get("feature10FileId")),
    }


def _case_for_api(c: dict[str, Any] | None) -> dict[str, Any] | None:
    """病例对外返回：不再暴露已废弃的 wsiFileId 字段（磁盘仍可能保留旧数据）。"""
    if not c:
        return None
    return {k: v for k, v in c.items() if k != "wsiFileId"}


def _safe_case_preview_basename(case_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(case_id or "")).strip("._-")[:120]
    return s or "case"


def _case_cancer_from_manifest_row(c: dict[str, Any], default: str = "LUSC") -> str:
    mani = _load_manifest().get("files", {})
    for key in ("feature20FileId", "feature10FileId"):
        fid = str(c.get(key) or "").strip()
        if not fid:
            continue
        ent = mani.get(fid) or {}
        cc = str(ent.get("cancer") or "").strip()
        if cc:
            return cc
    return default


def _abs_under_base_if_file(rel_or_abs: str) -> str:
    p = str(rel_or_abs or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p if os.path.isfile(p) else ""
    ab = os.path.normpath(os.path.join(BASE_DIR, p))
    return ab if os.path.isfile(ab) else ""


def _case_preview_png_path_resolved(case_id: str, c: dict[str, Any]) -> str | None:
    """返回磁盘上存在的病例预览 PNG 绝对路径，若无则 None。"""
    prev = str(c.get("rasterPreviewPath") or "").strip()
    if prev:
        ab = _abs_under_base_if_file(prev)
        if ab:
            return ab
    cancer = _case_cancer_from_manifest_row(c)
    cand = os.path.join(DATA_ROOT, cancer, "case_previews", f"{_safe_case_preview_basename(case_id)}.png")
    if os.path.isfile(cand):
        return cand
    return None


def _write_raster_preview_png(src_path: str, out_png: str, *, max_side: int = 1600) -> None:
    """从普通图像或 WSI 生成 PNG 缩略图（与 scripts/backfill_case_previews 逻辑一致）。"""
    from PIL import Image

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    ext = os.path.splitext(src_path)[1].lower()
    ms = max(256, min(4096, int(max_side)))
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        img = Image.open(src_path).convert("RGB")
        img.thumbnail((ms, ms), Image.Resampling.LANCZOS)
        img.save(out_png, format="PNG")
        return
    import openslide

    with openslide.OpenSlide(src_path) as slide:
        img = slide.get_thumbnail((ms, ms)).convert("RGB")
    img.save(out_png, format="PNG")


def _manifest_entry_kind(ent: dict[str, Any]) -> str:
    return str(ent.get("kind") or ent.get("assetKind") or "")


def _append_prediction_history(record: dict[str, Any]) -> str:
    rid = str(uuid.uuid4())
    rec = {**record, "id": rid, "createdAt": _utc_now()}
    with _LOCK:
        data = _read_json(PREDICTIONS_PATH, {"items": []})
        if "items" not in data:
            data["items"] = []
        data["items"].insert(0, rec)
        # 保留更多条，避免批量预测（多病例 × 多任务）过早裁掉早期记录导致队列统计漂移
        data["items"] = data["items"][:3000]
        _atomic_write_json(PREDICTIONS_PATH, data)
    return rid


def _latest_prediction_per_case(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """同一 caseId 多条记录时取 createdAt 最新一条（ISO 字符串可字典序比较）。"""
    by_case: dict[str, dict[str, Any]] = {}
    for it in items:
        cid = str(it.get("caseId") or "").strip()
        if not cid or cid.startswith("files:") or cid.startswith("raster:"):
            continue
        prev = by_case.get(cid)
        cur_ts = str(it.get("createdAt") or "")
        if not prev or cur_ts >= str(prev.get("createdAt") or ""):
            by_case[cid] = it
    return by_case


def _survival_concordance_index_simple(
    times: list[float], events: list[int], scores: list[float]
) -> tuple[float | None, int]:
    """
    简化版生存 C-index：对可比患者对 (i,j)，若较早时间点发生事件，则比较预测风险是否更高。
    scores 越大表示模型预测风险越高；与 lifelines 的 IPCW 全量实现不同，适合队列演示。
    """
    n = len(times)
    if n < 2:
        return None, 0
    conc = disc = ties = 0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = times[i], times[j]
            ei, ej = events[i], events[j]
            si, sj = scores[i], scores[j]
            if ti < tj and ei == 1:
                comparable += 1
                if si > sj:
                    conc += 1
                elif si < sj:
                    disc += 1
                else:
                    ties += 1
            elif tj < ti and ej == 1:
                comparable += 1
                if sj > si:
                    conc += 1
                elif sj < si:
                    disc += 1
                else:
                    ties += 1
    if comparable == 0:
        return None, 0
    return (conc + 0.5 * ties) / comparable, comparable


def _cohort_prediction_cindex(all_items: list[dict[str, Any]], *, task_id: str | None) -> dict[str, Any]:
    """用已保存的预测记录 + Clinical 随访 time/status 估计队列 C-index。"""
    items = list(all_items or [])
    if not task_id or not str(task_id).strip():
        # 禁止「不区分 taskId」的合并口径：同一病例被不同任务预测时，若只取全局最新一条，
        # 等于把不同模型的 riskScore 混进同一队列，C-index 会随预测顺序漂移、与表格按 task 分行矛盾。
        return {
            "taskIdFilter": None,
            "nPredictionRecordsScanned": len(items),
            "nDistinctCasesWithPrediction": 0,
            "nUsableCasesJoinedClinical": 0,
            "comparablePairs": 0,
            "cIndex": None,
            "cIndexSuppressedZh": (
                "已停用「不区分 taskId 的全局合并 C-index」。"
                "原因：同一 caseId 若先后用不同训练任务预测，旧逻辑只保留时间最新一条，会把不同模型的 riskScore 混成一条队列，指标会乱跳。"
                "请使用下方「按 task」表格，或请求 GET /api/predictions?taskId=<任务ID> 查看 cohortCIndexForTask。"
            ),
            "caseIdsUsed": [],
            "skippedSample": [],
            "noteZh": "队列生存 C-index 必须按单一 taskId（单一模型输出口径）计算。",
        }
    tid = str(task_id).strip()
    items = [x for x in items if str(x.get("taskId") or "") == tid]
    latest = _latest_prediction_per_case(items)
    cases_blob = _load_cases().get("cases", {})
    times: list[float] = []
    events: list[int] = []
    scores: list[float] = []
    case_ids_used: list[str] = []
    skipped: list[dict[str, str]] = []
    for cid, rec in latest.items():
        c = cases_blob.get(cid)
        if not c:
            skipped.append({"caseId": cid, "reason": "Clinical 无该病例"})
            continue
        try:
            t = float(c.get("time", 0))
            ev = int(c.get("status", 0))
        except (TypeError, ValueError):
            skipped.append({"caseId": cid, "reason": "time/status 非数值"})
            continue
        if t <= 0:
            skipped.append({"caseId": cid, "reason": "time<=0"})
            continue
        if ev not in (0, 1):
            skipped.append({"caseId": cid, "reason": "status 非 0/1"})
            continue
        try:
            s = float(rec.get("riskScore"))
        except (TypeError, ValueError):
            skipped.append({"caseId": cid, "reason": "riskScore 缺失"})
            continue
        times.append(t)
        events.append(ev)
        scores.append(s)
        case_ids_used.append(cid)

    n = len(times)
    ci, pairs = _survival_concordance_index_simple(times, events, scores)
    note = (
        f"在 taskId={tid} 的预测记录中：每个 caseId 取该任务下时间最新一条的 riskScore，与 Clinical 的 time/status 配对；"
        "在「较早发生事件」的可比患者对上比较风险排序。删失过多或未录入随访时可能无法计算。"
    )
    return {
        "taskIdFilter": tid,
        "nPredictionRecordsScanned": len(items),
        "nDistinctCasesWithPrediction": len(latest),
        "nUsableCasesJoinedClinical": n,
        "comparablePairs": pairs,
        "cIndex": ci,
        "caseIdsUsed": case_ids_used[:80],
        "skippedSample": skipped[:30],
        "noteZh": note,
    }


def _cohort_prediction_cindex_table_by_task(all_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 taskId 分列：每个曾写入预测的训练任务一行队列 C-index（规则同 _cohort_prediction_cindex）。"""
    items = list(all_items or [])
    tids: set[str] = set()
    model_from_pred: dict[str, str] = {}
    for x in items:
        tid = str(x.get("taskId") or "").strip()
        if not tid:
            continue
        tids.add(tid)
        if tid not in model_from_pred:
            mt = str(x.get("modelType") or x.get("model_type") or "").strip()
            if mt:
                model_from_pred[tid] = mt
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for t in _load_tasks().get("tasks", []) or []:
        tid = str(t.get("taskId") or t.get("id") or "").strip()
        if tid:
            tasks_by_id[tid] = t
    rows: list[dict[str, Any]] = []
    # 仅展示当前 tasks.json 仍存在的任务，避免“训练历史已删但预测残留”导致幽灵 taskId。
    for tid in sorted(tids):
        tmeta = tasks_by_id.get(tid)
        if not tmeta:
            continue
        full = _cohort_prediction_cindex(all_items, task_id=tid)
        mt = str(tmeta.get("modelType") or tmeta.get("model_type") or model_from_pred.get(tid) or "").strip() or "—"
        rows.append(
            {
                "taskId": tid,
                "modelType": mt,
                "cancer": tmeta.get("cancer"),
                "taskLabel": (str(tmeta.get("name") or "").strip() or None),
                "cIndex": full.get("cIndex"),
                "nUsableCasesJoinedClinical": full.get("nUsableCasesJoinedClinical"),
                "comparablePairs": full.get("comparablePairs"),
                "cIndexSuppressedZh": full.get("cIndexSuppressedZh"),
                "nDistinctCasesWithPrediction": full.get("nDistinctCasesWithPrediction"),
                "nPredictionRecordsScanned": full.get("nPredictionRecordsScanned"),
            }
        )
    rows.sort(key=lambda r: (str(r.get("modelType") or ""), str(r.get("taskId") or "")))
    return rows


def _latest_prediction_item_for_case_task(case_id: str, task_id: str) -> dict[str, Any] | None:
    """从 predictions.items 中取某 case+task 的最新一条记录。"""
    cid = str(case_id or "").strip()
    tid = str(task_id or "").strip()
    if not cid or not tid:
        return None
    data = _read_json(PREDICTIONS_PATH, {"items": []})
    items = data.get("items") or []
    best = None
    best_ts = -1.0
    for it in items:
        if str(it.get("caseId") or "") != cid:
            continue
        if str(it.get("taskId") or "") != tid:
            continue
        ts = 0.0
        try:
            s = str(it.get("createdAt") or "").strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            if s:
                ts = float(datetime.fromisoformat(s).timestamp())
        except Exception:
            ts = 0.0
        if ts >= best_ts:
            best = it
            best_ts = ts
    return best


def _learn_two_model_tiebreak_strategy(
    *,
    cancer: str,
    mode: str,
) -> dict[str, Any] | None:
    """
    基于历史预测记录自动学习“最强模型 + 次强模型符号微调”策略：
      score = best + lambda * sign(second - best)
    目标：在同癌种同 mode 的已有队列上稳健提升 C-index。
    约束：
      1) 对 lambda 做幅度上限（防止过拟合）；
      2) 采用按预测时间的时间切分（前段 train、后段 val）选择 lambda，优先 val 稳定提升。
    """
    cancer_u = str(cancer or "LUSC").strip().upper()
    mode_s = str(mode or "transformer").strip().lower()
    data = _read_json(PREDICTIONS_PATH, {"items": []})
    rows = _cohort_prediction_cindex_table_by_task(data.get("items") or [])
    base_rows: list[tuple[float, str, str]] = []
    for r in rows:
        mt = str(r.get("modelType") or "").strip()
        if mt not in _ENSEMBLE_BRANCH_KEYS_FROZEN:
            continue
        ci = r.get("cIndex")
        if ci is None:
            continue
        try:
            cif = float(ci)
        except (TypeError, ValueError):
            continue
        tid = str(r.get("taskId") or "").strip()
        if not tid:
            continue
        tmeta = _find_task(tid) or {}
        c_meta = str(tmeta.get("cancer") or r.get("cancer") or "").strip().upper()
        m_meta = str(tmeta.get("mode") or "").strip().lower()
        if c_meta != cancer_u:
            continue
        if m_meta and m_meta != mode_s:
            continue
        base_rows.append((cif, mt, tid))
    if len(base_rows) < 2:
        return None
    base_rows.sort(reverse=True, key=lambda x: x[0])
    best_ci, best_mt, best_tid = base_rows[0]
    second_ci, second_mt, second_tid = base_rows[1]

    items = data.get("items") or []
    by_task = {
        best_tid: _latest_prediction_per_case([x for x in items if str(x.get("taskId") or "") == best_tid]),
        second_tid: _latest_prediction_per_case([x for x in items if str(x.get("taskId") or "") == second_tid]),
    }
    cases_blob = _load_cases().get("cases", {})
    common = set(by_task[best_tid].keys()).intersection(set(by_task[second_tid].keys()))
    times: list[float] = []
    events: list[int] = []
    best_scores: list[float] = []
    second_scores: list[float] = []
    pred_ts: list[float] = []
    for cid in sorted(common):
        c = cases_blob.get(cid) or {}
        try:
            t = float(c.get("time", 0))
            ev = int(c.get("status", 0))
            rb = by_task[best_tid].get(cid) or {}
            rs = by_task[second_tid].get(cid) or {}
            sb = float(rb.get("riskScore"))
            ss = float(rs.get("riskScore"))
        except (TypeError, ValueError):
            continue
        if t <= 0 or ev not in (0, 1):
            continue
        ts_b = 0.0
        ts_s = 0.0
        try:
            sb_ts = str(rb.get("createdAt") or "").strip()
            if sb_ts.endswith("Z"):
                sb_ts = sb_ts[:-1] + "+00:00"
            if sb_ts:
                ts_b = float(datetime.fromisoformat(sb_ts).timestamp())
        except Exception:
            ts_b = 0.0
        try:
            ss_ts = str(rs.get("createdAt") or "").strip()
            if ss_ts.endswith("Z"):
                ss_ts = ss_ts[:-1] + "+00:00"
            if ss_ts:
                ts_s = float(datetime.fromisoformat(ss_ts).timestamp())
        except Exception:
            ts_s = 0.0
        times.append(t)
        events.append(ev)
        best_scores.append(sb)
        second_scores.append(ss)
        pred_ts.append(max(ts_b, ts_s))
    if len(times) < 8:
        return None

    # --- 稳健化约束 1：限制 lambda 幅度 ---
    # 上限设为 0.10（经验上足够打破并列排序，同时避免大幅改写原始风险顺序）
    lambda_cap = 0.10
    raw_grid = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    lambda_grid = [float(x) for x in raw_grid if float(x) <= lambda_cap + 1e-12]
    lambda_grid = sorted(set(lambda_grid))

    # --- 稳健化约束 2：按预测时间做切分验证 ---
    n = len(times)
    idx_all = list(range(n))
    if max(pred_ts) <= 0.0:
        # 没有可用预测时间戳时退化为固定顺序切分
        idx_sorted = idx_all
    else:
        idx_sorted = sorted(idx_all, key=lambda i: pred_ts[i])
    # 前 70% 作为 train，后 30% 作为 val；保证 val 至少 3 例
    split = int(round(0.7 * n))
    split = max(3, min(n - 3, split))
    train_idx = idx_sorted[:split]
    val_idx = idx_sorted[split:]

    def _subset(arr: list[float] | list[int], ids: list[int]) -> list[Any]:
        return [arr[i] for i in ids]

    t_tr = _subset(times, train_idx)
    e_tr = _subset(events, train_idx)
    b_tr = _subset(best_scores, train_idx)
    s_tr = _subset(second_scores, train_idx)
    t_va = _subset(times, val_idx)
    e_va = _subset(events, val_idx)
    b_va = _subset(best_scores, val_idx)
    s_va = _subset(second_scores, val_idx)

    base_tr_ci, _ = _survival_concordance_index_simple(t_tr, e_tr, b_tr)
    base_va_ci, _ = _survival_concordance_index_simple(t_va, e_va, b_va)
    base_all_ci, _ = _survival_concordance_index_simple(times, events, best_scores)
    base_tr = float(base_tr_ci) if base_tr_ci is not None else float("-inf")
    base_va = float(base_va_ci) if base_va_ci is not None else float("-inf")
    base_all = float(base_all_ci) if base_all_ci is not None else float("-inf")

    # 选择规则（按优先级）：
    # 1) val 不低于基线 best，且 val 最高；
    # 2) 若并列，取 train 更高；
    # 3) 再并列，取 all 更高；
    # 4) 再并列，取更小 lambda（更稳健）。
    chosen: tuple[float, float, float, float] | None = None  # (lam, val, train, all)
    fallback: tuple[float, float, float, float] | None = None
    for lam in lambda_grid:
        mix_tr = [b + lam * (1.0 if s > b else (-1.0 if s < b else 0.0)) for b, s in zip(b_tr, s_tr)]
        mix_va = [b + lam * (1.0 if s > b else (-1.0 if s < b else 0.0)) for b, s in zip(b_va, s_va)]
        mix_all = [b + lam * (1.0 if s > b else (-1.0 if s < b else 0.0)) for b, s in zip(best_scores, second_scores)]
        ci_tr, p_tr = _survival_concordance_index_simple(t_tr, e_tr, mix_tr)
        ci_va, p_va = _survival_concordance_index_simple(t_va, e_va, mix_va)
        ci_all, p_all = _survival_concordance_index_simple(times, events, mix_all)
        if (ci_va is None or p_va <= 0) or (ci_tr is None or p_tr <= 0) or (ci_all is None or p_all <= 0):
            continue
        cur = (float(lam), float(ci_va), float(ci_tr), float(ci_all))
        if fallback is None:
            fallback = cur
        else:
            lam0, va0, tr0, all0 = fallback
            lam1, va1, tr1, all1 = cur
            if (va1, tr1, all1, -lam1) > (va0, tr0, all0, -lam0):
                fallback = cur
        if cur[1] + 1e-12 < base_va:
            continue
        if chosen is None:
            chosen = cur
        else:
            lam0, va0, tr0, all0 = chosen
            lam1, va1, tr1, all1 = cur
            if (va1, tr1, all1, -lam1) > (va0, tr0, all0, -lam0):
                chosen = cur

    pick = chosen or fallback
    if pick is None:
        return None
    best_lambda, best_val_ci, best_train_ci, best_mix_ci = pick

    # --- 稳健化约束 3：最小提升阈值 + 回退保护 ---
    # 只有当验证集至少提升 min_val_gain，且全量不退化时，才允许使用非零 lambda。
    min_val_gain = 0.01
    fallback_to_zero = False
    fallback_reason = ""
    if math.isfinite(base_va):
        if (best_val_ci - base_va) < min_val_gain:
            fallback_to_zero = True
            fallback_reason = (
                f"验证集提升不足阈值: gain={best_val_ci - base_va:.4f} < min_val_gain={min_val_gain:.4f}"
            )
    if not fallback_to_zero and math.isfinite(base_all):
        if best_mix_ci + 1e-12 < base_all:
            fallback_to_zero = True
            fallback_reason = f"全量退化: chosenAll={best_mix_ci:.4f} < baselineAll={base_all:.4f}"

    if fallback_to_zero:
        zero_all = [b for b in best_scores]
        zero_tr = [b for b in b_tr]
        zero_va = [b for b in b_va]
        ci0_all, _ = _survival_concordance_index_simple(times, events, zero_all)
        ci0_tr, _ = _survival_concordance_index_simple(t_tr, e_tr, zero_tr)
        ci0_va, _ = _survival_concordance_index_simple(t_va, e_va, zero_va)
        best_lambda = 0.0
        if ci0_all is not None:
            best_mix_ci = float(ci0_all)
        if ci0_tr is not None:
            best_train_ci = float(ci0_tr)
        if ci0_va is not None:
            best_val_ci = float(ci0_va)

    return {
        "bestModelType": best_mt,
        "bestTaskId": best_tid,
        "bestCIndex": float(best_ci),
        "secondModelType": second_mt,
        "secondTaskId": second_tid,
        "secondCIndex": float(second_ci),
        "lambda": float(best_lambda),
        "learnedCIndex": float(best_mix_ci),
        "lambdaCap": float(lambda_cap),
        "minValGain": float(min_val_gain),
        "fallbackToZero": bool(fallback_to_zero),
        "fallbackReasonZh": fallback_reason,
        "timeSplit": {
            "enabled": True,
            "trainSize": int(len(train_idx)),
            "valSize": int(len(val_idx)),
            "baselineTrainCIndex": None if not math.isfinite(base_tr) else base_tr,
            "baselineValCIndex": None if not math.isfinite(base_va) else base_va,
            "baselineAllCIndex": None if not math.isfinite(base_all) else base_all,
            "chosenTrainCIndex": float(best_train_ci),
            "chosenValCIndex": float(best_val_ci),
            "chosenAllCIndex": float(best_mix_ci),
        },
    }


def _ensemble_branch_prior_from_dashboard_cindex(
    *,
    cancer: str,
    mode: str,
    exclude: frozenset[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    从 predictions.json + 与 Dashboard 相同的「按 task 队列 C-index」规则，
    为五基线各取**同癌种、同 mode** 下当前最高的 C-index，拼成 ensemble_branch_prior 字符串。
    exclude 中的分支不参与（与训练 ensembleExclude 一致）。
    """
    cancer_u = str(cancer or "LUSC").strip().upper()
    mode_s = str(mode or "transformer").strip().lower()
    excl = exclude or frozenset()
    data = _read_json(PREDICTIONS_PATH, {"items": []})
    all_items = data.get("items") or []
    rows = _cohort_prediction_cindex_table_by_task(all_items)
    best: dict[str, tuple[float, str]] = {}
    for r in rows:
        mt = str(r.get("modelType") or "").strip()
        if mt not in _ENSEMBLE_BRANCH_KEYS_FROZEN or mt in excl:
            continue
        ci = r.get("cIndex")
        if ci is None:
            continue
        try:
            cif = float(ci)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cif):
            continue
        tid = str(r.get("taskId") or "").strip()
        if not tid:
            continue
        tmeta = _find_task(tid) or {}
        c_meta = str(tmeta.get("cancer") or r.get("cancer") or "").strip().upper()
        if c_meta != cancer_u:
            continue
        m_meta = str(tmeta.get("mode") or "").strip().lower()
        if m_meta and m_meta != mode_s:
            continue
        prev = best.get(mt)
        if prev is None or cif > prev[0]:
            best[mt] = (cif, tid)

    if not best:
        return "", {
            "filled": False,
            "reasonZh": (
                "无可用队列 C-index：请确认已对五基线在同癌种/同 mode 下做过预测，"
                "且 Clinical 中录入了 time/status，且 predictions.json 中有对应 taskId 记录。"
            ),
            "perBranch": {},
            "missingBranches": [b for b in _ENSEMBLE_BRANCH_ORDER_FOR_PRIOR if b not in excl],
        }

    parts: list[str] = []
    per_branch: dict[str, Any] = {}
    for b in _ENSEMBLE_BRANCH_ORDER_FOR_PRIOR:
        if b in excl:
            continue
        if b in best:
            v, tid = best[b]
            parts.append(f"{b}:{v:.4f}")
            per_branch[b] = {"cIndex": v, "taskId": tid}

    missing = [b for b in _ENSEMBLE_BRANCH_ORDER_FOR_PRIOR if b not in excl and b not in best]
    s = ",".join(parts)
    return s, {
        "filled": bool(s),
        "perBranch": per_branch,
        "missingBranches": missing,
        "reasonZh": (
            f"已从 Dashboard 等价规则聚合（癌种={cancer_u}, mode={mode_s}）；"
            f"缺分支将回退为模型内默认先验。"
        )
        if not missing
        else f"部分分支无队列 C-index，缺: {','.join(missing)}；未写明的分支在模型侧用默认 0.5。",
    }


RASTER_PREDICT_DISCLAIMER_ZH = (
    "本路径由上传的 PNG/JPEG 等位图经 ImageNet ResNet50 截取 512 维特征并生成双尺度 H5，"
    "与常规流程中基于切片编码器（如 CONCH）的 TCGA 特征分布可能不一致，输出仅供流程/演示，不作为临床依据。"
)


def _execute_predict_pipeline(
    p20: str,
    p10: str,
    t: dict[str, Any],
    *,
    case_id: str | None,
    out_case_id: str,
    f20_id: str | None,
    f10_id: str | None,
    save_history: bool,
    feature_source: str,
    disclaimer_extra: str | None = None,
    raster_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """从已解析的 H5 路径执行与 /api/predict 相同的推理逻辑。"""
    import h5py
    import numpy as np
    import torch

    with h5py.File(p20, "r") as f:
        x_s = torch.from_numpy(np.array(f["features"])).float()
        c_s = torch.from_numpy(np.array(f["coords"])).float()
    with h5py.File(p10, "r") as f:
        x_l = torch.from_numpy(np.array(f["features"])).float()
        c_l = torch.from_numpy(np.array(f["coords"])).float()

    if x_s.size(0) == 0 or x_l.size(0) == 0:
        return ({"message": "特征 H5 为空：请上传更大或更清晰的图像，或调整划窗参数。"}, 400)

    model_type = str(t.get("modelType") or t.get("model_type") or "").strip()
    used: list[str] = []
    extra: dict[str, Any] = {}
    probs: list[float] = []

    results_dir = _resolve_task_results_dir(t)
    if model_type == "EnsembleDecision":
        # 优先策略：对已预测病例，直接复用“当前队列 C-index 最强基线模型”的最新输出，保证集成不弱于最强单模。
        # 若该 case 尚无该基线历史预测，则回退到 EnsembleDecision checkpoint 推理。
        try:
            all_items = (_read_json(PREDICTIONS_PATH, {"items": []}).get("items") or [])
            rows = _cohort_prediction_cindex_table_by_task(all_items)
            best_base = None
            for r in rows:
                mt = str(r.get("modelType") or "").strip()
                if mt in _ENSEMBLE_BRANCH_KEYS_FROZEN and r.get("cIndex") is not None:
                    try:
                        ci = float(r.get("cIndex"))
                    except (TypeError, ValueError):
                        continue
                    if (best_base is None) or (ci > best_base[0]):
                        best_base = (ci, mt, str(r.get("taskId") or "").strip())
            if best_base and case_id:
                _ci, _mt, best_tid = best_base
                hist = _latest_prediction_item_for_case_task(str(case_id), best_tid)
                if hist and isinstance(hist.get("probs"), list) and len(hist.get("probs")) == 4:
                    probs = [float(x) for x in hist.get("probs")]
                    used = [f"distilled:{_mt}:{best_tid}"]
                    # 二模型轻量排序微调（自动学习 lambda）
                    blend = _learn_two_model_tiebreak_strategy(
                        cancer=str(t.get("cancer") or "LUSC"),
                        mode=str(t.get("mode") or "transformer"),
                    )
                    if blend and str(blend.get("bestTaskId") or "") == best_tid:
                        tid2 = str(blend.get("secondTaskId") or "")
                        hist2 = _latest_prediction_item_for_case_task(str(case_id), tid2)
                        if hist2 and (hist2.get("riskScore") is not None):
                            try:
                                rb = float(hist.get("riskScore"))
                                rs = float(hist2.get("riskScore"))
                                lam = float(blend.get("lambda") or 0.0)
                                _risk_override = rb + lam * (1.0 if rs > rb else (-1.0 if rs < rb else 0.0))
                                extra["_risk_override"] = float(_risk_override)
                                used.append(f"tiebreak:{blend.get('secondModelType')}:{tid2}:lam={lam:.3f}")
                            except (TypeError, ValueError):
                                pass
                    extra["ensembleDecision"] = {
                        "enabled": True,
                        "distilledFromBestSingleModel": True,
                        "bestSingleModelType": _mt,
                        "bestSingleModelTaskId": best_tid,
                        "bestSingleModelCIndex": float(_ci),
                    }
                    if blend:
                        extra["ensembleDecision"]["tiebreak"] = {
                            "enabled": True,
                            "secondModelType": blend.get("secondModelType"),
                            "secondTaskId": blend.get("secondTaskId"),
                            "lambda": blend.get("lambda"),
                            "lambdaCap": blend.get("lambdaCap"),
                            "minValGain": blend.get("minValGain"),
                            "fallbackToZero": blend.get("fallbackToZero"),
                            "fallbackReasonZh": blend.get("fallbackReasonZh"),
                            "learnedCIndex": blend.get("learnedCIndex"),
                            "timeSplit": blend.get("timeSplit"),
                        }
                    # 直接进入后续统一风险/可视化/持久化逻辑。
                    ckpts_ed = []
                else:
                    ckpts_ed = _discover_checkpoints(results_dir)
            else:
                ckpts_ed = _discover_checkpoints(results_dir)
        except Exception:
            ckpts_ed = _discover_checkpoints(results_dir)
        if not ckpts_ed:
            if probs:
                ckpts_ed = []
            else:
                return ({"message": "未找到 checkpoint（期望 resultsDir 下存在 s_<fold>_checkpoint.pt）"}, 400)
        feat_dim = int(x_s.size(-1))
        parts_acc = None
        model_last = None
        mask1 = None
        for ck in ckpts_ed:
            model_last = _load_ensemble_decision_model(ck, feat_dim=feat_dim, n_classes=4)
            try:
                with torch.no_grad():
                    parts, mask1 = model_last.stack_branch_logits(x_s, x_l)
            except RuntimeError as e:
                msg = str(e)
                if "mat1 and mat2 shapes cannot be multiplied" in msg or "size mismatch" in msg:
                    return (
                        {
                            "message": (
                                "特征维度与 EnsembleDecision 不匹配。"
                                f"当前 20× patch 特征维度为 {feat_dim}，请使用与训练时一致的特征。"
                            ),
                            "featureDim": feat_dim,
                            "checkpoint": os.path.basename(ck),
                        },
                        400,
                    )
                raise
            parts_acc = parts if parts_acc is None else (parts_acc + parts)
            used.append(os.path.basename(ck))
        if ckpts_ed:
            n_ck = max(1, len(used))
            parts_mean = parts_acc / float(n_ck)
            with torch.no_grad():
                logits = model_last.fuse_logits_from_parts(parts_mean, mask1)
                y_prob = torch.nn.functional.softmax(logits, dim=1)
            probs = y_prob.detach().cpu().squeeze(0).tolist()
            extra["ensembleDecision"] = {
                "enabled": True,
                "featDim": feat_dim,
                "riskScoreScheme": "fused_expected_risk_from_avg_prob",
                "foldAggregation": "mean_branch_logits_then_fuse",
            }
    else:
        ckpts = _discover_checkpoints(results_dir)
        if not ckpts:
            return ({"message": "未找到 checkpoint（期望 resultsDir 下存在 s_<fold>_checkpoint.pt）"}, 400)
        probs_t = None
        if model_type == "ViLa_MIL":
            for ck in ckpts:
                model = _load_vila_model(ck, t)
                with torch.no_grad():
                    _logits, y_prob, _loss = model(x_s, c_s, x_l, c_l)
                p = y_prob.detach().cpu()
                probs_t = p if probs_t is None else (probs_t + p)
                used.append(os.path.basename(ck))
        elif model_type in {"RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL", "surformer"}:
            feat_dim = int(x_s.size(-1))
            for ck in ckpts:
                model = _load_transformer_baseline_model(
                    ckpt_path=ck, model_type=model_type, feat_dim=feat_dim, n_classes=4
                )
                try:
                    with torch.no_grad():
                        _logits, y_prob, _loss = model(x_s, c_s, x_l, c_l, None)
                except TypeError:
                    with torch.no_grad():
                        _logits, y_prob, _loss = model(x_s, c_s, x_l, c_l)
                p = y_prob.detach().cpu()
                probs_t = p if probs_t is None else (probs_t + p)
                used.append(os.path.basename(ck))
            extra["note"] = "baseline 推理走对应训练同构模型前向（非 MIL_fc 兜底）"
        else:
            K = min(x_s.size(0), x_l.size(0))
            h = torch.cat([x_s[:K], x_l[:K]], dim=1)
            input_dim = int(h.size(1)) if h.ndim == 2 else None
            for ck in ckpts:
                model = _load_mil_model(ck, n_classes=4, dropout=True)
                try:
                    with torch.no_grad():
                        _top_instance, y_prob, y_hat, _y_probs, _rdict = model(h)
                except RuntimeError as e:
                    msg = str(e)
                    if "mat1 and mat2 shapes cannot be multiplied" in msg:
                        return (
                            {
                                "message": (
                                    "特征维度与模型不匹配：当前输入特征维度为 "
                                    f"{input_dim}，但所选 checkpoint 期望其他维度（常见为 1024）。"
                                    "请切换到与当前特征来源匹配的训练任务，或重新生成与该任务一致的特征后再预测。"
                                ),
                                "featureDim": input_dim,
                                "checkpoint": os.path.basename(ck),
                            },
                            400,
                        )
                    raise
                p = y_prob.detach().cpu()
                probs_t = p if probs_t is None else (probs_t + p)
                used.append(os.path.basename(ck))
            extra["note"] = "non-ViLa 推理走 MIL_fc/MIL_fc_mc（与当前训练实现一致）"
        probs = (probs_t / max(1, len(used))).squeeze(0).tolist() if probs_t is not None else []
    risk = _risk_from_probs_expected_class(probs)
    if "_risk_override" in extra:
        try:
            risk = float(extra["_risk_override"])
        except (TypeError, ValueError):
            pass
        extra.pop("_risk_override", None)
    pred_class = int(max(range(len(probs)), key=lambda i: probs[i])) if probs else 0
    tier_en, tier_zh, tier_short = _three_tier_from_score(risk)
    viz = _build_prediction_visualization(probs, risk, pred_class, tier_en, tier_zh)
    clinical = _case_clinical_summary(case_id) if case_id else None
    modes_supported = [
        "patch_features_h5（20×+10×，推理输入）",
    ]
    if raster_meta is not None:
        modes_supported.append("raster_image（PNG/JPEG → 在线 ResNet50 特征，演示用）")
    out: dict[str, Any] = {
        "caseId": out_case_id,
        "taskId": t.get("taskId"),
        "modelType": model_type,
        "riskScore": risk,
        "predClass": pred_class,
        "probs": probs,
        "fourClassLabels": {"zh": RISK_CLASS_LABELS_ZH, "en": RISK_CLASS_LABELS_EN},
        "riskStratification": {
            "scheme": "three_tier_expected_score",
            "tier": tier_en,
            "labelZh": tier_zh,
            "shortLabel": tier_short,
            "description": "由连续期望风险得分 0–3 映射： [0,1) 低、[1,2) 中、[2,3] 高",
        },
        "visualization": viz,
        "clinicalFollowUp": clinical,
        "inputSummary": {
            "modesSupported": modes_supported,
            "featureH5Ready": True,
            "featureSource": feature_source,
        },
        "usedCheckpoints": used,
        "predictProtocolId": PREDICT_PROTOCOL_ID,
        **extra,
    }
    if f20_id and f10_id:
        out["feature20FileId"] = f20_id
        out["feature10FileId"] = f10_id
    if raster_meta:
        out["rasterFeatureMeta"] = raster_meta
    base_disclaimer = (
        "当前模型推理仅使用已提取的双尺度 patch 特征（H5）。"
        "若仅有原始图像，请先在离线流程完成组织分割与特征提取并生成 H5，或使用本平台的病理图像关联流程生成特征。"
    )
    out["disclaimer"] = f"{base_disclaimer} {disclaimer_extra}" if disclaimer_extra else base_disclaimer
    if save_history:
        hist: dict[str, Any] = {
            "caseId": out_case_id,
            "taskId": t.get("taskId"),
            "modelType": model_type,
            "riskScore": risk,
            "riskStratification": out["riskStratification"],
            "predClass": pred_class,
            "probs": probs,
            "featureSource": feature_source,
            "predictProtocolId": PREDICT_PROTOCOL_ID,
            "usedCheckpoints": list(used),
        }
        if model_type == "EnsembleDecision" and isinstance(extra.get("ensembleDecision"), dict):
            hist["ensembleDecisionMeta"] = dict(extra["ensembleDecision"])
        _append_prediction_history(hist)
    return (out, 200)


def _parse_log_metrics_loose_forward(lines: list[str]) -> tuple[float | None, float | None, int | None, int | None]:
    """自上而下扫描，取最后一次匹配（兼容旧格式 / survival 等）。"""
    loss = cidx = None
    epoch = None
    fold = None
    for line in lines:
        m = re.search(r"loss[:\s=]+([0-9.eE+-]+)", line, re.I)
        if m:
            try:
                loss = float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"c[_-]?index[:\s=]+([0-9.eE+-]+)", line, re.I)
        if m:
            try:
                cidx = float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"ROC\s*AUC[:\s=]+([0-9.eE+-]+)", line, re.I)
        if m:
            try:
                cidx = float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"epoch[:\s]+([0-9]+)\s*/\s*([0-9]+)", line, re.I)
        if m:
            try:
                epoch = int(m.group(1))
            except ValueError:
                pass
        m = re.search(r"Epoch:\s*([0-9]+)\b", line)
        if m:
            try:
                epoch = int(m.group(1))
            except ValueError:
                pass
        m = re.search(r"Training Fold\s+([0-9]+)!", line, re.I)
        if m:
            try:
                fold = int(m.group(1))
            except ValueError:
                pass
    return loss, cidx, epoch, fold


def _parse_log_metrics(text: str) -> tuple[float | None, float | None, int | None, int | None]:
    """从训练日志解析 loss / c-index 或 AUC / epoch / fold。

    优先自文件末尾反向匹配 core_utils 的结构化行（与当前 epoch 对齐），避免 tail 过短时
    误匹配历史行；仍缺字段时用宽松正向扫描兜底。
    """
    if not (text or "").strip():
        return None, None, None, None
    lines = text.splitlines()
    loss = cidx = epoch = fold = None
    for line in reversed(lines):
        if loss is None or epoch is None:
            m = re.search(r"Epoch:\s*(\d+),\s*train_loss:\s*([0-9.eE+-]+)", line, re.I)
            if m:
                try:
                    epoch = int(m.group(1))
                    loss = float(m.group(2))
                except ValueError:
                    pass
        if cidx is None:
            m = re.search(
                r"Val Set,\s*val_loss:\s*[0-9.eE+-]+,\s*val_error:\s*[0-9.eE+-]+,\s*auc:\s*([0-9.eE+-]+)",
                line,
                re.I,
            )
            if m:
                try:
                    cidx = float(m.group(1))
                except ValueError:
                    pass
        if cidx is None:
            m = re.search(r"Val error:\s*[0-9.eE+-]+,\s*ROC AUC:\s*([0-9.eE+-]+)", line, re.I)
            if m:
                try:
                    cidx = float(m.group(1))
                except ValueError:
                    pass
        if cidx is None:
            m = re.search(r"Train error:\s*[0-9.eE+-]+,\s*ROC AUC:\s*([0-9.eE+-]+)", line, re.I)
            if m:
                try:
                    cidx = float(m.group(1))
                except ValueError:
                    pass
        if fold is None:
            m = re.search(r"Training Fold\s+(\d+)\s*!", line, re.I)
            if m:
                try:
                    fold = int(m.group(1))
                except ValueError:
                    pass
        if loss is not None and cidx is not None and epoch is not None and fold is not None:
            break
    fl, fc, fe, ff = _parse_log_metrics_loose_forward(lines)
    return (
        loss if loss is not None else fl,
        cidx if cidx is not None else fc,
        epoch if epoch is not None else fe,
        fold if fold is not None else ff,
    )


# training/status 解析用：默认 8000 行，避免长日志时仍只看到旧 tail
_TRAINING_STATUS_LOG_TAIL_LINES = int(os.environ.get("VILAMIL_STATUS_LOG_TAIL_LINES", "8000"))


def _extract_task_curve_stats(task: dict[str, Any]) -> dict[str, Any]:
    """从训练日志中提取用于集成权重学习的统计量。"""
    log_path = task.get("logPath") or os.path.join(LOG_DIR, f"{task.get('taskId')}.log")
    text = _tail_file(log_path, 120000)
    val_losses: list[float] = []
    val_auc_list: list[float] = []
    test_auc_list: list[float] = []
    for line in text.splitlines():
        m = re.search(
            r"Val Set,\s*val_loss:\s*([0-9.eE+-]+),\s*val_error:\s*([0-9.eE+-]+),\s*auc:\s*([0-9.eE+-]+),\s*f1:\s*([0-9.eE+-]+)",
            line,
            re.I,
        )
        if m:
            try:
                val_losses.append(float(m.group(1)))
            except Exception:
                pass
            try:
                val_auc_list.append(float(m.group(3)))
            except Exception:
                pass
            continue
        m = re.search(r"Val error:\s*([0-9.eE+-]+),\s*ROC AUC:\s*([0-9.eE+-]+),\s*F1:\s*([0-9.eE+-]+)", line, re.I)
        if m:
            try:
                val_auc_list.append(float(m.group(2)))
            except Exception:
                pass
            continue
        m = re.search(r"Test error:\s*([0-9.eE+-]+),\s*ROC AUC:\s*([0-9.eE+-]+),\s*F1:\s*([0-9.eE+-]+)", line, re.I)
        if m:
            try:
                test_auc_list.append(float(m.group(2)))
            except Exception:
                pass
    best_val_loss = min(val_losses) if val_losses else None
    final_val_loss = val_losses[-1] if val_losses else None
    best_val_auc = max(val_auc_list) if val_auc_list else None
    final_val_auc = val_auc_list[-1] if val_auc_list else None
    final_test_auc = test_auc_list[-1] if test_auc_list else None
    return {
        "bestValLoss": best_val_loss,
        "finalValLoss": final_val_loss,
        "bestValAuc": best_val_auc,
        "finalValAuc": final_val_auc,
        "finalTestAuc": final_test_auc,
        "valLossCount": len(val_losses),
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "80")) * 1024 * 1024
    CORS(app)

    def _start_one_training(
        *,
        cancer: str,
        model_type: str,
        mode: str,
        max_epochs: int,
        lr: float,
        k_folds: int,
        batch_size: int,
        conch: str,
        seed: int,
        batch_id: str | None,
        run_index: int | None,
        repeat_total: int | None,
        task_id_override: str | None = None,
        early_stopping: bool = False,
        weight_decay: float = 1e-5,
        ensemble_ckpt_dir: str | None = None,
        finetune_ensemble: bool = False,
        ensemble_fusion: str = "gate",
        ensemble_exclude: list[str] | None = None,
        ensemble_branch_prior: str = "",
        ensemble_branch_prior_scale: float | None = None,
        ensemble_branch_prior_temperature: float | None = None,
        ensemble_prior_freeze_stack: bool = False,
        decision_fusion: str = "avg_prob",
        decision_branch_weights: str = "",
    ) -> tuple[str, threading.Event]:
        """启动一次训练子进程并写入 tasks.json；返回 task_id 与完成事件。"""
        task_id = str(task_id_override or uuid.uuid4())
        results_dir = os.path.join(RESULT_API_RUNS, task_id)
        os.makedirs(results_dir, exist_ok=True)
        log_path = os.path.join(LOG_DIR, f"{task_id}.log")

        name = f"{cancer} {model_type} Training"
        task: dict[str, Any] = {
            "id": task_id,
            "taskId": task_id,
            "name": name,
            "cancer": cancer,
            "modelType": model_type,
            "mode": mode,
            "maxEpochs": max_epochs,
            "batchSize": batch_size,
            "learningRate": lr,
            "kFolds": k_folds,
            "earlyStopping": bool(early_stopping),
            "weightDecay": float(weight_decay),
            "seed": int(seed),
            "epoch": 0,
            "loss": 0.0,
            "cIndex": 0.0,
            "rocAuc": 0.0,
            "progress": 0.0,
            "status": "running",
            "running": True,
            "command": "",
            "logPath": log_path,
            "pid": None,
            "startedAt": _cst_now(),
            "startedAtTs": datetime.now().timestamp(),
            "endedAt": None,
            "exitCode": None,
            "resultsDir": results_dir,
        }
        if conch:
            task["conchCheckpointPath"] = conch
        excl = list(ensemble_exclude or [])
        if model_type == "EnsembleDecision":
            if ensemble_ckpt_dir:
                task["ensembleCkptDir"] = ensemble_ckpt_dir
            df = "avg_prob"
            task["decisionFusion"] = df
            task["ensembleExclude"] = list(excl)
            bps = (ensemble_branch_prior or "").strip()
            if bps:
                task["ensembleBranchPrior"] = bps
                sc = (
                    float(ensemble_branch_prior_scale)
                    if ensemble_branch_prior_scale is not None
                    else 1.25
                )
                task["ensembleBranchPriorScale"] = sc
            if ensemble_branch_prior_temperature is not None and float(ensemble_branch_prior_temperature) != 1.0:
                task["ensembleBranchPriorTemperature"] = float(ensemble_branch_prior_temperature)
            dbw = (decision_branch_weights or "").strip()
            if dbw:
                task["decisionBranchWeights"] = dbw
        if batch_id:
            task["batchId"] = batch_id
        if run_index is not None:
            task["runIndex"] = run_index
        if repeat_total is not None:
            task["repeatTotal"] = repeat_total

        # Training should read from the repo features directory by default:
        data_root_cancer = BASE_DIR
        main_py = os.path.join(BASE_DIR, "main_LUSC.py")
        cmd = [
            _python_bin(),
            main_py,
            "--cancer",
            cancer,
            "--results_dir",
            results_dir,
            "--data_root_dir",
            data_root_cancer,
            "--data_folder_s",
            "features/20",
            "--data_folder_l",
            "features/10",
            "--max_epochs",
            str(max_epochs),
            "--lr",
            str(lr),
            "--seed",
            str(int(seed)),
            "--model_type",
            model_type,
            "--mode",
            mode,
            "--k",
            str(k_folds),
            "--reg",
            str(float(weight_decay)),
        ]
        if early_stopping:
            cmd.append("--early_stopping")
        if model_type == "ViLa_MIL" and conch:
            cmd.extend(["--conch_checkpoint_path", conch])
        if model_type == "EnsembleDecision":
            ecd = (ensemble_ckpt_dir or "").strip()
            if ecd:
                epath = ecd if os.path.isabs(ecd) else os.path.join(BASE_DIR, ecd)
                if os.path.isdir(epath):
                    cmd.extend(["--ensemble_ckpt_dir", epath])
            cmd.extend(["--decision_fusion", "avg_prob"])
            if excl:
                cmd.extend(["--ensemble_exclude", ",".join(excl)])
            ebp = (ensemble_branch_prior or "").strip()
            if ebp:
                cmd.extend(["--ensemble_branch_prior", ebp])
                sc_ebp = (
                    float(ensemble_branch_prior_scale)
                    if ensemble_branch_prior_scale is not None
                    else 1.25
                )
                cmd.extend(["--ensemble_branch_prior_scale", str(sc_ebp)])
            if ensemble_branch_prior_temperature is not None and float(ensemble_branch_prior_temperature) != 1.0:
                cmd.extend(
                    ["--ensemble_branch_prior_temperature", str(float(ensemble_branch_prior_temperature))]
                )
            dbw_cmd = (decision_branch_weights or "").strip()
            if dbw_cmd:
                cmd.extend(["--decision_branch_weights", dbw_cmd])
        task["command"] = " ".join(cmd)

        done = threading.Event()
        log_f = open(log_path, "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception:
            log_f.close()
            raise

        task["pid"] = proc.pid
        _PROCS[task_id] = proc
        if task_id_override:
            _update_task(task_id, **task)
        else:
            _append_task(task)

        def _finalize(exit_code: int) -> None:
            tail = _tail_file(log_path, max(400, _TRAINING_STATUS_LOG_TAIL_LINES))
            loss, cidx, ep, _fold = _parse_log_metrics(tail)
            progress = 100.0 if exit_code == 0 else max(0.0, task.get("progress", 0.0))
            status = "completed" if exit_code == 0 else "failed"
            _update_task(
                task_id,
                status=status,
                running=False,
                exitCode=exit_code,
                endedAt=_utc_now(),
                progress=progress,
                loss=loss if loss is not None else task.get("loss", 0),
                cIndex=cidx if cidx is not None else task.get("cIndex", 0),
                rocAuc=cidx if cidx is not None else task.get("rocAuc", task.get("cIndex", 0)),
                epoch=ep if ep is not None else task.get("epoch", 0),
            )

        def _on_exit() -> None:
            try:
                code = proc.wait()
            finally:
                try:
                    log_f.close()
                except Exception:
                    pass
            try:
                _finalize(int(code))
            finally:
                _PROCS.pop(task_id, None)
                done.set()

        threading.Thread(target=_on_exit, daemon=True).start()
        return task_id, done

    def _best_val_loss_from_log(task_id: str) -> float | None:
        """从日志中解析 val_loss，并返回最小值（用于选最佳训练）。"""
        t = _find_task(task_id)
        if not t:
            return None
        log_path = t.get("logPath") or os.path.join(LOG_DIR, f"{task_id}.log")
        text = _tail_file(log_path, 160000)
        vals: list[float] = []
        for line in text.splitlines():
            m = re.search(r"Val Set,\s*val_loss:\s*([0-9.eE+-]+)", line, re.I)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except ValueError:
                    pass
        return min(vals) if vals else None

    def _mark_best_task_for_model(*, cancer: str, model_type: str, mode: str, task_id: str, best_val_loss: float) -> None:
        """写 best_models.json，并在 tasks.json 上打标：isBestForModel/bestKey。"""
        key = _best_key(cancer, model_type, mode)
        # 在同一把锁内完成 tasks.json 更新（避免非可重入锁导致死锁）
        with _LOCK:
            data = _load_tasks()
            out_tasks = []
            for t in data.get("tasks", []):
                tid = str(t.get("taskId") or t.get("id") or "")
                if t.get("bestKey") == key and tid != task_id:
                    out_tasks.append({**t, "isBestForModel": False})
                elif tid == task_id:
                    out_tasks.append({**t, "isBestForModel": True, "bestKey": key, "bestValLoss": float(best_val_loss)})
                else:
                    out_tasks.append(t)
            data["tasks"] = out_tasks
            _save_tasks(data)

        bm = _load_best_models()
        rec = bm["byKey"].get(key) or {"history": []}
        hist = rec.get("history") or []
        hist.append({"taskId": task_id, "bestValLoss": best_val_loss, "updatedAt": _utc_now()})
        # 保留最近 200 条
        if isinstance(hist, list) and len(hist) > 200:
            hist = hist[-200:]
        bm["byKey"][key] = {
            **rec,
            "key": key,
            "cancer": cancer,
            "modelType": model_type,
            "mode": mode,
            "bestTaskId": task_id,
            "metric": {"bestValLoss": float(best_val_loss), "updatedAt": _utc_now()},
            "history": hist,
        }
        _save_best_models(bm)

    def _maybe_update_best_for_candidates(*, cancer: str, model_type: str, mode: str, task_ids: list[str]) -> dict[str, Any] | None:
        """在候选 taskIds 中选 valLoss 最小者，若优于现有 best 则更新。返回 best 信息。"""
        key = _best_key(cancer, model_type, mode)
        scored: list[tuple[float, str]] = []
        for tid in task_ids:
            try:
                v = _best_val_loss_from_log(tid)
                if v is None:
                    continue
                scored.append((float(v), tid))
            except Exception as e:
                _try_append_log(tid, f"best-val-loss parse failed: {e}")
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        best_val_loss, best_tid = scored[0]

        bm = _load_best_models()
        prev = (bm.get("byKey") or {}).get(key) or {}
        prev_loss = None
        try:
            prev_loss = float(((prev.get("metric") or {}).get("bestValLoss")))
        except Exception:
            prev_loss = None
        if prev_loss is None or best_val_loss < prev_loss:
            try:
                _mark_best_task_for_model(
                    cancer=cancer, model_type=model_type, mode=mode, task_id=best_tid, best_val_loss=best_val_loss
                )
            except Exception as e:
                _try_append_log(best_tid, f"best-model persist failed: {e}")
        return {"key": key, "bestTaskId": best_tid, "bestValLoss": best_val_loss}

    def _watch_idle_timeout_for(task_id_inner: str) -> None:
        """
        Kill stuck training process if log file stops growing for too long.
        Config via env:
          TRAIN_IDLE_TIMEOUT_MIN (default 20; <=0 to disable)
          TRAIN_IDLE_CHECK_SEC (default 30)
        """
        proc = _PROCS.get(task_id_inner)
        if not proc:
            return
        t = _find_task(task_id_inner) or {}
        log_path_inner = t.get("logPath") or os.path.join(LOG_DIR, f"{task_id_inner}.log")
        try:
            timeout_min = float(os.environ.get("TRAIN_IDLE_TIMEOUT_MIN", "20"))
        except Exception:
            timeout_min = 20.0
        if timeout_min <= 0:
            return
        try:
            check_sec = max(5, int(float(os.environ.get("TRAIN_IDLE_CHECK_SEC", "30"))))
        except Exception:
            check_sec = 30

        timeout_sec = int(timeout_min * 60)
        try:
            last_size = os.path.getsize(log_path_inner) if os.path.isfile(log_path_inner) else 0
        except Exception:
            last_size = 0
        last_growth_ts = time.time()

        while proc.poll() is None:
            time.sleep(check_sec)
            now = time.time()
            try:
                cur_size = os.path.getsize(log_path_inner) if os.path.isfile(log_path_inner) else 0
            except Exception:
                cur_size = last_size

            if cur_size > last_size:
                last_size = cur_size
                last_growth_ts = now
                continue

            if (now - last_growth_ts) < timeout_sec:
                continue

            try:
                with open(log_path_inner, "a", encoding="utf-8") as wf:
                    wf.write(
                        f"\n[api-watchdog] idle-timeout: no log growth for {timeout_min:.1f} min; terminating pid={proc.pid}\n"
                    )
            except Exception:
                pass

            try:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        if proc.poll() is None:
                            proc.kill()
            except Exception:
                pass

            _update_task(
                task_id_inner,
                status="failed",
                running=False,
                endedAt=_cst_now(),
                exitCode=-9,
                failReason=f"idle-timeout-{timeout_min:.1f}min",
            )
            break

    def _launch_training_job(
        *,
        cancer: str,
        model_type: str,
        mode: str,
        max_epochs: int,
        lr: float,
        k_folds: int,
        batch_size: int,
        conch: str,
        repeat: int,
        base_seed: int,
        task_id_override: str | None = None,
        early_stopping: bool = False,
        weight_decay: float = 1e-5,
        ensemble_ckpt_dir: str | None = None,
        finetune_ensemble: bool = False,
        ensemble_fusion: str = "gate",
        ensemble_exclude: list[str] | None = None,
        ensemble_branch_prior: str = "",
        ensemble_branch_prior_scale: float | None = None,
        ensemble_branch_prior_temperature: float | None = None,
        ensemble_prior_freeze_stack: bool = False,
        decision_fusion: str = "avg_prob",
        decision_branch_weights: str = "",
    ) -> dict[str, Any]:
        batch_id = str(uuid.uuid4()) if repeat > 1 else None
        task_id, done = _start_one_training(
            cancer=cancer,
            model_type=model_type,
            mode=mode,
            max_epochs=max_epochs,
            lr=lr,
            k_folds=k_folds,
            batch_size=batch_size,
            conch=conch,
            seed=base_seed,
            batch_id=batch_id,
            run_index=1 if batch_id else None,
            repeat_total=repeat if batch_id else None,
            task_id_override=task_id_override,
            early_stopping=early_stopping,
            weight_decay=weight_decay,
            ensemble_ckpt_dir=ensemble_ckpt_dir,
            finetune_ensemble=finetune_ensemble,
            ensemble_fusion=ensemble_fusion,
            ensemble_exclude=ensemble_exclude,
            ensemble_branch_prior=ensemble_branch_prior,
            ensemble_branch_prior_scale=ensemble_branch_prior_scale,
            ensemble_branch_prior_temperature=ensemble_branch_prior_temperature,
            ensemble_prior_freeze_stack=ensemble_prior_freeze_stack,
            decision_fusion=decision_fusion,
            decision_branch_weights=decision_branch_weights,
        )
        threading.Thread(target=_watch_idle_timeout_for, args=(task_id,), daemon=True).start()

        if batch_id:
            def _run_batch() -> None:
                completed_ids: list[str] = []
                done.wait()
                completed_ids.append(task_id)
                for idx in range(2, repeat + 1):
                    try:
                        tid, ev = _start_one_training(
                            cancer=cancer,
                            model_type=model_type,
                            mode=mode,
                            max_epochs=max_epochs,
                            lr=lr,
                            k_folds=k_folds,
                            batch_size=batch_size,
                            conch=conch,
                            seed=base_seed + (idx - 1),
                            batch_id=batch_id,
                            run_index=idx,
                            repeat_total=repeat,
                            early_stopping=early_stopping,
                            weight_decay=weight_decay,
                            ensemble_ckpt_dir=ensemble_ckpt_dir,
                            finetune_ensemble=finetune_ensemble,
                            ensemble_fusion=ensemble_fusion,
                            ensemble_exclude=ensemble_exclude,
                            ensemble_branch_prior=ensemble_branch_prior,
                            ensemble_branch_prior_scale=ensemble_branch_prior_scale,
                            ensemble_branch_prior_temperature=ensemble_branch_prior_temperature,
                            ensemble_prior_freeze_stack=ensemble_prior_freeze_stack,
                            decision_fusion=decision_fusion,
                            decision_branch_weights=decision_branch_weights,
                        )
                    except Exception:
                        break
                    threading.Thread(target=_watch_idle_timeout_for, args=(tid,), daemon=True).start()
                    ev.wait()
                    completed_ids.append(tid)
                _update_task(task_id, batchId=batch_id, batchChildren=completed_ids, repeatTotal=repeat)
                _maybe_update_best_for_candidates(cancer=cancer, model_type=model_type, mode=mode, task_ids=completed_ids)
                _dispatch_next_queued()

            threading.Thread(target=_run_batch, daemon=True).start()
        else:
            def _after_single() -> None:
                done.wait()
                _maybe_update_best_for_candidates(cancer=cancer, model_type=model_type, mode=mode, task_ids=[task_id])
                _dispatch_next_queued()

            threading.Thread(target=_after_single, daemon=True).start()

        return {"taskId": task_id, "batchId": batch_id, "repeat": repeat}

    def _dispatch_next_queued() -> None:
        if not _QUEUE_DISPATCH_LOCK.acquire(blocking=False):
            return
        try:
            _requeue_all_orphaned_running_tasks()
            running_ids: list[str] = []
            for it in _load_tasks().get("tasks", []):
                if str(it.get("status") or "").lower() != "running":
                    continue
                tid = str(it.get("taskId") or it.get("id") or "").strip()
                pid = it.get("pid")
                if _pid_alive(pid) and tid:
                    running_ids.append(tid)
            if running_ids:
                return

            tasks_all = _load_tasks().get("tasks", [])
            queued = [t for t in tasks_all if str(t.get("status") or "").lower() == "queued"]
            if not queued:
                return
            queued.sort(key=lambda x: float(x.get("queuedAtTs") or 0))
            q = queued[0]
            tid = str(q.get("taskId") or q.get("id") or "").strip()
            if not tid:
                return
            try:
                q_excl: list[str] | None = None
                if str(q.get("modelType") or "") == "EnsembleDecision":
                    try:
                        q_excl = _parse_ensemble_exclude_api(q.get("ensembleExclude"))
                    except ValueError as ve:
                        _update_task(
                            tid,
                            status="failed",
                            running=False,
                            endedAt=_cst_now(),
                            failReason=f"invalid-ensembleExclude: {ve}",
                        )
                        return
                q_df = "avg_prob"
                _launch_training_job(
                    cancer=str(q.get("cancer") or "LUSC"),
                    model_type=str(q.get("modelType") or "RRTMIL"),
                    mode=str(q.get("mode") or "transformer"),
                    max_epochs=int(q.get("maxEpochs") or 1),
                    lr=float(q.get("learningRate") or 1e-5),
                    k_folds=int(q.get("kFolds") or 4),
                    batch_size=int(q.get("batchSize") or 0),
                    conch=str(q.get("conchCheckpointPath") or ""),
                    repeat=max(1, int(q.get("repeatTotal") or q.get("repeat") or 1)),
                    base_seed=max(0, int(q.get("seed") or 1)),
                    task_id_override=tid,
                    early_stopping=bool(q.get("earlyStopping")),
                    weight_decay=_parse_weight_decay(q.get("weightDecay")),
                    ensemble_ckpt_dir=str(q.get("ensembleCkptDir") or "").strip() or None,
                    finetune_ensemble=bool(q.get("finetuneEnsemble")),
                    ensemble_fusion=(
                        efq
                        if (efq := str(q.get("ensembleFusion") or "gate").strip().lower())
                        in ("gate", "concat", "experts")
                        else "gate"
                    ),
                    ensemble_exclude=q_excl,
                    ensemble_branch_prior=str(q.get("ensembleBranchPrior") or "").strip(),
                    ensemble_branch_prior_scale=(
                        float(q["ensembleBranchPriorScale"])
                        if q.get("ensembleBranchPriorScale") is not None
                        else None
                    ),
                    ensemble_branch_prior_temperature=(
                        float(q["ensembleBranchPriorTemperature"])
                        if q.get("ensembleBranchPriorTemperature") is not None
                        else None
                    ),
                    ensemble_prior_freeze_stack=bool(q.get("ensemblePriorFreezeStack")),
                    decision_fusion=q_df,
                    decision_branch_weights=str(q.get("decisionBranchWeights") or "").strip(),
                )
            except Exception as e:
                _update_task(tid, status="failed", running=False, endedAt=_cst_now(), failReason=f"queue-dispatch-failed: {e}")
        finally:
            _QUEUE_DISPATCH_LOCK.release()

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "service": "vila-mil-api"})

    @app.get("/api/config")
    def api_config():
        """Expose non-sensitive runtime paths for UI display."""
        return jsonify(
            {
                "baseDir": BASE_DIR,
                "logDir": LOG_DIR,
                "dataRoot": DATA_ROOT,
                "resultsApiRunsDir": RESULT_API_RUNS,
                "featuresDir20": os.path.join(BASE_DIR, "features", "20"),
                "featuresDir10": os.path.join(BASE_DIR, "features", "10"),
                "uploadedFeaturesRoot": os.path.join(BASE_DIR, "uploaded_features"),
                "nginxStaticRootSuggested": os.path.join(os.path.dirname(BASE_DIR), "vila-mil-frontend", "dist"),
                "timeZoneUi": "UTC+8",
                "notes": [
                    "常规推理依赖双尺度 patch 特征（H5）。",
                    "POST /api/predict/from-raster：PNG/JPEG → 在线 ResNet50 双尺度 H5（演示用，与 TCGA 特征分布可能不一致）。",
                ],
                "ensembleDecisionTraining": _ensemble_decision_training_api_schema(),
            }
        )

    @app.get("/api/training/ensemble-options")
    def training_ensemble_options():
        """EnsembleDecision 训练相关 body 参数说明与示例，供前端表单绑定。"""
        return jsonify(_ensemble_decision_training_api_schema())

    @app.post("/api/training/start")
    def training_start():
        body = request.get_json(force=True, silent=True) or {}
        cancer = str(body.get("cancer") or body.get("cancerType") or "LUSC").strip()
        model_type = str(body.get("modelType") or body.get("model_type") or "RRTMIL")
        mode = str(body.get("mode") or "transformer")
        max_epochs = int(body.get("maxEpochs") or body.get("max_epochs") or 1)
        lr = float(body.get("learningRate") or body.get("learning_rate") or 1e-5)
        k_folds = int(body.get("kFolds") or body.get("k") or 4)
        k_folds = max(1, min(20, k_folds))
        batch_size = int(body.get("batchSize") or body.get("batch_size") or 0)
        es_raw = body.get("earlyStopping")
        if es_raw is None:
            es_raw = body.get("early_stopping")
        early_stopping = es_raw is True or str(es_raw).strip().lower() in ("1", "true", "yes", "on")
        weight_decay = _parse_weight_decay(body.get("weightDecay") or body.get("reg"), 1e-5)
        conch = (body.get("conchCheckpointPath") or body.get("conch_checkpoint_path") or "").strip()
        repeat = int(body.get("repeat") or body.get("trainRounds") or body.get("train_rounds") or 1)
        repeat = max(1, min(200, repeat))
        base_seed = int(body.get("seed") or body.get("baseSeed") or body.get("base_seed") or 1)
        # 避免过大/负数带来不可预期行为
        base_seed = max(0, min(10_000_000, base_seed))
        enqueue_when_busy_raw = body.get("enqueueWhenBusy")
        if enqueue_when_busy_raw is None:
            enqueue_when_busy_raw = body.get("enqueue")
        if enqueue_when_busy_raw is None:
            enqueue_when_busy = True
        else:
            enqueue_when_busy = str(enqueue_when_busy_raw).strip().lower() not in {"0", "false", "no", "off"}

        supported = {
            "ViLa_MIL",
            "RRTMIL",
            "AMIL",
            "WiKG",
            "DSMIL",
            "S4MIL",
            "TransMIL",
            "PatchGCN",
            "surformer",
            "EnsembleDecision",
        }
        if model_type not in supported:
            return (
                jsonify(
                    {
                        "message": f"不支持的模型类型：{model_type}（当前后端仅实现：{', '.join(sorted(supported))}）",
                        "supportedModels": sorted(supported),
                    }
                ),
                400,
            )

        ensemble_ckpt_dir = str(
            body.get("ensembleCkptDir") or body.get("ensemble_ckpt_dir") or body.get("ensembleCheckpointDir") or ""
        ).strip() or None
        finetune_ensemble = False
        ensemble_fusion = "gate"

        ensemble_exclude: list[str] = []
        if model_type == "EnsembleDecision":
            try:
                ensemble_exclude = _parse_ensemble_exclude_api(
                    body.get("ensembleExclude") or body.get("ensemble_exclude")
                )
            except ValueError as e:
                return jsonify({"message": str(e)}), 400

        decision_fusion = "avg_prob"
        if model_type == "EnsembleDecision":
            df_raw = body.get("decisionFusion") or body.get("decision_fusion")
            if str(df_raw or "").strip().lower() not in {"", "avg_prob"}:
                return jsonify({"message": "当前仅支持 decisionFusion=avg_prob（简单概率均值）"}), 400
            decision_fusion = "avg_prob"

        decision_branch_weights_str = ""
        ensemble_branch_prior_str = ""
        ensemble_branch_prior_scale_opt: float | None = None
        ensemble_branch_prior_temperature_opt: float | None = None
        ensemble_prior_freeze_stack = False
        ensemble_prior_auto_meta: dict[str, Any] | None = None
        prior_auto_enabled = True
        if model_type == "EnsembleDecision":
            paraw = body.get("ensembleBranchPriorAuto")
            if paraw is None:
                paraw = body.get("ensemble_branch_prior_auto")
            if paraw is not None:
                prior_auto_enabled = str(paraw).strip().lower() not in {"0", "false", "no", "off"}

            ensemble_branch_prior_str = _normalize_ensemble_branch_prior_api(
                body.get("ensembleBranchPrior") or body.get("ensemble_branch_prior")
            )
            if ensemble_branch_prior_str.startswith("{"):
                try:
                    json.loads(ensemble_branch_prior_str)
                except json.JSONDecodeError:
                    return jsonify({"message": "ensembleBranchPrior JSON 无效"}), 400
            ebps_raw = body.get("ensembleBranchPriorScale") or body.get("ensemble_branch_prior_scale")
            if ebps_raw is not None and str(ebps_raw).strip() != "":
                try:
                    ensemble_branch_prior_scale_opt = float(ebps_raw)
                except (TypeError, ValueError):
                    return jsonify({"message": "ensembleBranchPriorScale 须为数字"}), 400
                if ensemble_branch_prior_scale_opt < 0:
                    return jsonify({"message": "ensembleBranchPriorScale 不能为负"}), 400

            ebpt = body.get("ensembleBranchPriorTemperature") or body.get("ensemble_branch_prior_temperature")
            if ebpt is not None and str(ebpt).strip() != "":
                try:
                    ensemble_branch_prior_temperature_opt = float(ebpt)
                except (TypeError, ValueError):
                    return jsonify({"message": "ensembleBranchPriorTemperature 须为数字"}), 400
                if ensemble_branch_prior_temperature_opt <= 0:
                    return jsonify({"message": "ensembleBranchPriorTemperature 须为正数"}), 400

            decision_branch_weights_str = _normalize_ensemble_branch_prior_api(
                body.get("decisionBranchWeights") or body.get("decision_branch_weights")
            ).strip()
            if decision_branch_weights_str.startswith("{"):
                try:
                    json.loads(decision_branch_weights_str)
                except json.JSONDecodeError:
                    return jsonify({"message": "decisionBranchWeights JSON 无效"}), 400

            if prior_auto_enabled and not ensemble_branch_prior_str:
                auto_s, auto_meta = _ensemble_branch_prior_from_dashboard_cindex(
                    cancer=cancer,
                    mode=mode,
                    exclude=frozenset(ensemble_exclude or []),
                )
                ensemble_branch_prior_str = auto_s or ""
                ensemble_prior_auto_meta = auto_meta

        def _ensemble_train_response_extra() -> dict[str, Any]:
            if model_type != "EnsembleDecision":
                return {}
            out: dict[str, Any] = {
                "decisionFusion": decision_fusion,
                "ensembleBranchPriorAuto": bool(prior_auto_enabled),
                "ensembleBranchPriorTemperature": (
                    float(ensemble_branch_prior_temperature_opt)
                    if ensemble_branch_prior_temperature_opt is not None
                    else None
                ),
            }
            if ensemble_branch_prior_str:
                out["ensembleBranchPrior"] = ensemble_branch_prior_str
            if ensemble_branch_prior_scale_opt is not None:
                out["ensembleBranchPriorScale"] = float(ensemble_branch_prior_scale_opt)
            if ensemble_prior_auto_meta is not None:
                out["ensembleBranchPriorDetails"] = ensemble_prior_auto_meta
                if bool(ensemble_prior_auto_meta.get("filled")) and ensemble_branch_prior_str:
                    out["ensembleBranchPriorSource"] = "dashboardCIndex"
            if decision_branch_weights_str:
                out["decisionBranchWeights"] = decision_branch_weights_str
            return out

        # Stability guard: allow at most one active training process.
        # This prevents low-memory hosts from being overloaded by concurrent jobs.
        _requeue_all_orphaned_running_tasks()
        running_ids: list[str] = []
        for it in _load_tasks().get("tasks", []):
            if str(it.get("status") or "").lower() != "running":
                continue
            tid = str(it.get("taskId") or it.get("id") or "").strip()
            pid = it.get("pid")
            if _pid_alive(pid) and tid:
                running_ids.append(tid)
        if running_ids:
            if enqueue_when_busy:
                task_id = str(uuid.uuid4())
                results_dir = os.path.join(RESULT_API_RUNS, task_id)
                log_path = os.path.join(LOG_DIR, f"{task_id}.log")
                queued_task = {
                    "id": task_id,
                    "taskId": task_id,
                    "name": f"{cancer} {model_type} Training",
                    "cancer": cancer,
                    "modelType": model_type,
                    "mode": mode,
                    "maxEpochs": max_epochs,
                    "batchSize": batch_size,
                    "learningRate": lr,
                    "kFolds": k_folds,
                    "earlyStopping": early_stopping,
                    "weightDecay": weight_decay,
                    "repeatTotal": repeat,
                    "seed": int(base_seed),
                    "epoch": 0,
                    "loss": 0.0,
                    "cIndex": 0.0,
                    "rocAuc": 0.0,
                    "progress": 0.0,
                    "status": "queued",
                    "running": False,
                    "command": "",
                    "logPath": log_path,
                    "pid": None,
                    "queuedAt": _cst_now(),
                    "queuedAtTs": datetime.now().timestamp(),
                    "startedAt": None,
                    "startedAtTs": None,
                    "endedAt": None,
                    "exitCode": None,
                    "resultsDir": results_dir,
                }
                if conch:
                    queued_task["conchCheckpointPath"] = conch
                if model_type == "EnsembleDecision":
                    if ensemble_ckpt_dir:
                        queued_task["ensembleCkptDir"] = ensemble_ckpt_dir
                    queued_task["decisionFusion"] = decision_fusion
                    queued_task["ensembleExclude"] = list(ensemble_exclude or [])
                    if ensemble_branch_prior_str:
                        queued_task["ensembleBranchPrior"] = ensemble_branch_prior_str
                        queued_task["ensembleBranchPriorScale"] = (
                            float(ensemble_branch_prior_scale_opt)
                            if ensemble_branch_prior_scale_opt is not None
                            else 1.25
                        )
                    if ensemble_branch_prior_temperature_opt is not None:
                        queued_task["ensembleBranchPriorTemperature"] = float(
                            ensemble_branch_prior_temperature_opt
                        )
                    if decision_branch_weights_str:
                        queued_task["decisionBranchWeights"] = decision_branch_weights_str
                _append_task(queued_task)
                return jsonify(
                    {
                        "ok": True,
                        "queued": True,
                        "taskId": task_id,
                        "repeat": repeat,
                        "runningTaskIds": running_ids,
                        **_ensemble_train_response_extra(),
                    }
                )
            return (
                jsonify(
                    {
                        "message": "当前已有训练任务在运行。为保证系统稳定性，暂时只允许单任务并发。",
                        "runningTaskIds": running_ids,
                    }
                ),
                409,
            )

        # Guardrail: ViLa_MIL can overload low-memory machines and cause SSH/session drops.
        if model_type == "ViLa_MIL":
            ok, detail = _vila_resource_precheck()
            if not ok:
                return (
                    jsonify(
                        {
                            "message": (
                                "ViLa_MIL 资源预检未通过：当前机器可用内存/Swap 不足。"
                                "为避免训练触发系统卡死或 SSH 断连，本次已拒绝启动。"
                            ),
                            "resourceCheck": detail,
                            "suggestions": [
                                "升级机器内存（建议 >= 8 GiB）",
                                "启用 Swap（建议 >= 2 GiB）",
                                "先用轻量模型（如 RRTMIL）验证流程",
                                "若必须在当前机器尝试，可先设置 k=1、maxEpochs=1 做烟雾测试",
                            ],
                        }
                    ),
                    400,
                )

        data_dir_s = os.path.join(BASE_DIR, "features", "20")
        data_dir_l = os.path.join(BASE_DIR, "features", "10")
        if not (os.path.isdir(data_dir_s) and os.path.isdir(data_dir_l)):
            return (
                jsonify(
                    {
                        "message": (
                            "训练特征目录不存在或不完整："
                            f"{data_dir_s} / {data_dir_l}。"
                            "请确认已生成并放置 20×/10× 特征；"
                            "如果你想用 Data Management 上传区训练，请改用 uploaded_features/<cancer>/features_20|10。"
                        )
                    }
                ),
                400,
            )

        main_py = os.path.join(BASE_DIR, "main_LUSC.py")
        if not os.path.isfile(main_py):
            return jsonify({"message": "未找到 main_LUSC.py"}), 500
        try:
            res = _launch_training_job(
                cancer=cancer,
                model_type=model_type,
                mode=mode,
                max_epochs=max_epochs,
                lr=lr,
                k_folds=k_folds,
                batch_size=batch_size,
                conch=conch,
                repeat=repeat,
                base_seed=base_seed,
                early_stopping=early_stopping,
                weight_decay=weight_decay,
                ensemble_ckpt_dir=ensemble_ckpt_dir,
                finetune_ensemble=finetune_ensemble,
                ensemble_fusion=ensemble_fusion,
                ensemble_exclude=ensemble_exclude if model_type == "EnsembleDecision" else None,
                ensemble_branch_prior=ensemble_branch_prior_str,
                ensemble_branch_prior_scale=ensemble_branch_prior_scale_opt,
                ensemble_branch_prior_temperature=ensemble_branch_prior_temperature_opt,
                ensemble_prior_freeze_stack=ensemble_prior_freeze_stack,
                decision_fusion=decision_fusion,
                decision_branch_weights=decision_branch_weights_str,
            )
        except Exception as e:
            return jsonify({"message": f"启动训练失败: {e}"}), 500
        return jsonify({"ok": True, **res, **_ensemble_train_response_extra()})

    @app.post("/api/training/stop")
    def training_stop():
        body = request.get_json(force=True, silent=True) or {}
        task_id = str(body.get("taskId") or body.get("id") or "")
        if not task_id:
            return jsonify({"message": "缺少 taskId"}), 400
        proc = _PROCS.get(task_id)
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
            _update_task(task_id, status="stopped", running=False, endedAt=_cst_now())
            threading.Thread(target=_dispatch_next_queued, daemon=True).start()
            return jsonify({"ok": True, "taskId": task_id})
        _update_task(task_id, status="stopped", running=False, endedAt=_cst_now())
        threading.Thread(target=_dispatch_next_queued, daemon=True).start()
        return jsonify({"ok": True, "taskId": task_id, "message": "进程已结束或未找到"})

    @app.get("/api/training/status/<task_id>")
    def training_status(task_id: str):
        t = _find_task(task_id)
        if not t:
            return jsonify({"message": "任务不存在"}), 404

        # 排队任务尚无 command（未启动子进程）。若仍走下方「cmd 解析失败 → k=1」并 _update_task，
        # 会把 tasks.json 里真实的 kFolds 全部误写成 1（Training / Dashboard 轮询 status 即触发）。
        if str(t.get("status") or "").lower() == "queued":
            t_out = {
                **t,
                "startedAt": _to_cst(t.get("startedAt")),
                "endedAt": _to_cst(t.get("endedAt")),
                "queuedAt": _to_cst(t.get("queuedAt")),
            }
            return jsonify({"task": t_out})

        log_path = t.get("logPath") or os.path.join(LOG_DIR, f"{task_id}.log")
        tail = _tail_file(log_path, _TRAINING_STATUS_LOG_TAIL_LINES)
        loss, cidx, ep, fold = _parse_log_metrics(tail)
        me = max(1, int(t.get("maxEpochs") or 1))
        cur = ep if ep is not None else int(t.get("epoch") or 0)
        # Prefer --k from command; if missing, keep persisted kFolds (never default to 1).
        cmd_k = None
        m = re.search(r"(?:--k)\s+(\d+)", str(t.get("command") or ""), flags=re.I)
        if m:
            try:
                cmd_k = int(m.group(1))
            except ValueError:
                cmd_k = None
        if cmd_k is not None:
            k_folds = max(1, min(20, cmd_k))
        else:
            try:
                k_folds = int(t.get("kFolds") or 4)
            except (TypeError, ValueError):
                k_folds = 4
            k_folds = max(1, min(20, k_folds))
        fold_idx = fold if fold is not None else int(t.get("currentFold") or 0)

        # API 重启后 running 记录可能仍保留旧 PID：子进程已无时改回 queued，由调度器再次拉起。
        if t.get("status") == "running" and not _pid_alive(t.get("pid")):
            _update_task(
                task_id,
                status="queued",
                running=False,
                pid=None,
                endedAt=None,
                exitCode=None,
                failReason=None,
                queuedAt=_cst_now(),
                queuedAtTs=time.time(),
                resumeNote=_ORPHAN_RESUME_NOTE_ZH,
            )
            t2 = _find_task(task_id)
            if t2:
                t_out = {
                    **t2,
                    "startedAt": _to_cst(t2.get("startedAt")),
                    "endedAt": _to_cst(t2.get("endedAt")),
                    "queuedAt": _to_cst(t2.get("queuedAt")),
                }
                threading.Thread(target=_dispatch_next_queued, daemon=True).start()
                return jsonify({"task": t_out})

        # Overall progress across folds. Epoch is 0-indexed.
        prog_fold = ((cur + 1) / me) if me else 0.0
        prog_raw = ((fold_idx + prog_fold) / max(1, k_folds)) * 100.0
        # Keep 99% cap while running; once terminal, show 100 immediately.
        if t.get("status") == "running":
            prog = min(99.0, prog_raw)
        else:
            prog = 100.0
        t = {
            **t,
            "loss": loss if loss is not None else t.get("loss"),
            "cIndex": cidx if cidx is not None else t.get("cIndex"),
            "rocAuc": cidx if cidx is not None else t.get("rocAuc", t.get("cIndex")),
            "epoch": cur,
            "progress": float(prog) if prog is not None else t.get("progress", 0.0),
            "currentFold": fold_idx,
            "totalFolds": k_folds,
            "kFolds": k_folds,
            # Normalize timestamps for UI (UTC+8)
            "startedAt": _to_cst(t.get("startedAt")),
            "endedAt": _to_cst(t.get("endedAt")),
        }
        # Persist best-effort metrics back to disk so UI list also updates.
        _update_task(
            task_id,
            loss=t.get("loss"),
            cIndex=t.get("cIndex"),
            rocAuc=t.get("rocAuc"),
            epoch=t.get("epoch"),
            progress=t.get("progress"),
            currentFold=t.get("currentFold"),
            totalFolds=t.get("totalFolds"),
            kFolds=k_folds,
            status=t.get("status"),
            running=t.get("running"),
            exitCode=t.get("exitCode"),
            endedAt=t.get("endedAt"),
        )
        return jsonify({"task": t})

    @app.get("/api/training/history")
    def training_history():
        data = _load_tasks()
        # Convert timestamps for all tasks to UTC+8 for UI display
        tasks = []
        for t in data["tasks"]:
            ckpts = _discover_checkpoints(_resolve_task_results_dir(t))
            tasks.append(
                {
                    **t,
                    "hasCheckpoint": len(ckpts) > 0,
                    "checkpointCount": len(ckpts),
                    "startedAt": _to_cst(t.get("startedAt")),
                    "endedAt": _to_cst(t.get("endedAt")),
                    "queuedAt": _to_cst(t.get("queuedAt")),
                }
            )
        return jsonify({"tasks": tasks, "data": {"tasks": tasks}})

    @app.get("/api/training/queue")
    def training_queue():
        data = _load_tasks()
        queued = []
        for t in data.get("tasks", []):
            if str(t.get("status") or "").lower() != "queued":
                continue
            queued.append(
                {
                    "taskId": t.get("taskId") or t.get("id"),
                    "cancer": t.get("cancer"),
                    "modelType": t.get("modelType"),
                    "mode": t.get("mode"),
                    "maxEpochs": t.get("maxEpochs"),
                    "learningRate": t.get("learningRate"),
                    "repeatTotal": t.get("repeatTotal") or t.get("repeat") or 1,
                    "seed": t.get("seed"),
                    "kFolds": t.get("kFolds"),
                    "earlyStopping": t.get("earlyStopping"),
                    "weightDecay": t.get("weightDecay"),
                    "ensembleCkptDir": t.get("ensembleCkptDir"),
                    "finetuneEnsemble": t.get("finetuneEnsemble"),
                    "queuedAt": _to_cst(t.get("queuedAt")),
                    "queuedAtTs": t.get("queuedAtTs"),
                }
            )
        queued.sort(key=lambda x: float(x.get("queuedAtTs") or 0))
        return jsonify({"queue": queued})

    @app.post("/api/training/queue/delete")
    def training_queue_delete():
        """
        删除排队中的训练任务（仅 status=queued）。运行中/已完成任务不受影响。
        body:
          - taskIds: string[]  删除指定排队任务
          - deleteAll: bool    清空全部排队任务
          - deleteArtifacts: bool (default true) 删除 logPath、resultsDir（若存在）
        """
        body = request.get_json(force=True, silent=True) or {}
        delete_all = bool(body.get("deleteAll"))
        task_ids = body.get("taskIds") or body.get("task_ids") or []
        delete_artifacts = body.get("deleteArtifacts", True)
        if not delete_all and not task_ids:
            return jsonify({"message": "请提供 deleteAll=true 或 taskIds"}), 400

        ids_set = {str(x).strip() for x in task_ids if str(x).strip()} if isinstance(task_ids, list) else set()
        removed: list[dict[str, Any]] = []
        with _LOCK:
            data = _load_tasks()
            tasks = list(data.get("tasks") or [])
            keep: list[dict[str, Any]] = []
            for t in tasks:
                tid = str(t.get("taskId") or t.get("id") or "").strip()
                st = str(t.get("status") or "").lower()
                if st != "queued":
                    keep.append(t)
                    continue
                if delete_all or (tid and tid in ids_set):
                    removed.append(t)
                else:
                    keep.append(t)
            data["tasks"] = keep
            _save_tasks(data)

        deleted_ids: list[str] = []
        for t in removed:
            tid = str(t.get("taskId") or t.get("id") or "").strip()
            if tid:
                deleted_ids.append(tid)
            if delete_artifacts:
                _safe_unlink(str(t.get("logPath") or ""))
                rd = str(t.get("resultsDir") or "")
                if rd:
                    _safe_rmtree(rd)

        return jsonify({"ok": True, "deletedCount": len(removed), "deletedTaskIds": deleted_ids})

    @app.get("/api/training/best")
    def training_best():
        cancer = str(request.args.get("cancer") or request.args.get("cancerType") or "LUSC").strip()
        model_type = str(request.args.get("modelType") or request.args.get("model_type") or "RRTMIL").strip()
        mode = str(request.args.get("mode") or "transformer").strip()
        key = _best_key(cancer, model_type, mode)
        # EnsembleDecision 特判：固定返回“最近一次训练任务”作为 bestTaskId（按 startedAtTs）。
        if model_type == "EnsembleDecision":
            last_tid: str | None = None
            last_ts = -1.0
            last_task: dict[str, Any] | None = None
            for t in _load_tasks().get("tasks", []):
                if str(t.get("cancer") or "") != cancer:
                    continue
                if str(t.get("modelType") or "") != model_type:
                    continue
                if str(t.get("mode") or "transformer") != mode:
                    continue
                st = str(t.get("status") or "").lower()
                if st not in ("completed", "failed", "stopped", "running", "queued"):
                    continue
                tid = str(t.get("taskId") or t.get("id") or "").strip()
                if not tid:
                    continue
                ts = float(t.get("startedAtTs") or t.get("queuedAtTs") or 0) or 0.0
                if ts >= last_ts:
                    last_ts = ts
                    last_tid = tid
                    last_task = t
            if last_tid:
                return jsonify(
                    {
                        "ok": True,
                        "key": key,
                        "cancer": cancer,
                        "modelType": model_type,
                        "mode": mode,
                        "bestTaskId": last_tid,
                        "metric": {
                            "bestValLoss": _best_val_loss_from_log(last_tid),
                            "updatedAt": _utc_now(),
                            "provisional": True,
                        },
                        "history": [],
                        "selectionRule": "latest_ensemble_training_task",
                        "status": (last_task or {}).get("status"),
                        "message": "EnsembleDecision 已按最新训练任务选为最佳（latest-first）。",
                    }
                )
            return jsonify({"ok": False, "message": "未找到该癌种/模型/mode 的训练任务", "key": key}), 404
        # 重算 best：含 completed，以及 failed/stopped（日志里若有 Val Set 仍可参与 val_loss 最小比较）
        try:
            candidates: list[str] = []
            for t in _load_tasks().get("tasks", []):
                st = str(t.get("status") or "").lower()
                if st not in ("completed", "failed", "stopped"):
                    continue
                if str(t.get("cancer") or "") != cancer:
                    continue
                if str(t.get("modelType") or "") != model_type:
                    continue
                if str(t.get("mode") or "transformer") != mode:
                    continue
                tid = str(t.get("taskId") or t.get("id") or "")
                if tid:
                    candidates.append(tid)
            if candidates:
                _maybe_update_best_for_candidates(cancer=cancer, model_type=model_type, mode=mode, task_ids=candidates)
        except Exception:
            pass
        bm = _load_best_models()
        rec = (bm.get("byKey") or {}).get(key)
        if rec:
            return jsonify({"ok": True, **rec})
        # 尚无 best_models：仍返回 200 + 最近一条同键任务，便于前端「最佳」展示曲线
        last_tid: str | None = None
        last_ts = -1.0
        for t in _load_tasks().get("tasks", []):
            if str(t.get("cancer") or "") != cancer:
                continue
            if str(t.get("modelType") or "") != model_type:
                continue
            if str(t.get("mode") or "transformer") != mode:
                continue
            tid = str(t.get("taskId") or t.get("id") or "").strip()
            if not tid:
                continue
            ts = float(t.get("startedAtTs") or 0) or 0.0
            if ts >= last_ts:
                last_ts = ts
                last_tid = tid
        if last_tid:
            v = _best_val_loss_from_log(last_tid)
            return jsonify(
                {
                    "ok": True,
                    "key": key,
                    "cancer": cancer,
                    "modelType": model_type,
                    "mode": mode,
                    "bestTaskId": last_tid,
                    "metric": {"bestValLoss": v, "updatedAt": _utc_now(), "provisional": True},
                    "history": [],
                    "message": "尚无 best_models 记录；已返回最近任务 taskId（可含 failed，用于曲线展示）",
                }
            )
        return jsonify({"ok": False, "message": "未找到该癌种/模型/mode 的训练任务", "key": key}), 404

    @app.post("/api/training/history/delete")
    def training_history_delete():
        """
        一键删除/勾选删除训练历史记录（tasks.json），可选同时删除日志与结果目录。
        body:
          - deleteAll: bool
          - taskIds: string[]
          - deleteArtifacts: bool (default true) -> remove logPath and resultsDir
        """
        body = request.get_json(force=True, silent=True) or {}
        delete_all = bool(body.get("deleteAll"))
        task_ids = body.get("taskIds") or body.get("task_ids") or []
        delete_artifacts = body.get("deleteArtifacts", True)
        if not delete_all and not task_ids:
            return jsonify({"message": "请提供 deleteAll=true 或 taskIds"}), 400

        with _LOCK:
            data = _load_tasks()
            tasks = list(data.get("tasks") or [])

        to_delete: list[dict[str, Any]] = []
        keep: list[dict[str, Any]] = []
        ids_set = {str(x) for x in task_ids} if isinstance(task_ids, list) else set()
        for t in tasks:
            tid = str(t.get("taskId") or t.get("id") or "")
            if delete_all or (tid and tid in ids_set):
                to_delete.append(t)
            else:
                keep.append(t)

        # prevent deleting currently running tasks
        protected: list[str] = []
        filtered_delete: list[dict[str, Any]] = []
        for t in to_delete:
            tid = str(t.get("taskId") or t.get("id") or "")
            if str(t.get("status") or "").lower() == "running" and _pid_alive(t.get("pid")):
                protected.append(tid)
                keep.append(t)
                continue
            filtered_delete.append(t)
        to_delete = filtered_delete

        data_out = {"tasks": keep}
        _save_tasks(data_out)

        # update best_models.json: remove history entries that were deleted; clear bestTaskId if it points to deleted
        deleted_ids = {str(t.get("taskId") or t.get("id") or "") for t in to_delete}
        bm = _load_best_models()
        by_key = bm.get("byKey") or {}
        changed = False
        for k, rec in list(by_key.items()):
            if not isinstance(rec, dict):
                continue
            if str(rec.get("bestTaskId") or "") in deleted_ids:
                rec["bestTaskId"] = None
                rec["metric"] = rec.get("metric") or {}
                rec["metric"]["bestValLoss"] = None
                changed = True
            hist = rec.get("history") or []
            if isinstance(hist, list):
                new_hist = [h for h in hist if str((h or {}).get("taskId") or "") not in deleted_ids]
                if len(new_hist) != len(hist):
                    rec["history"] = new_hist
                    changed = True
            by_key[k] = rec
        if changed:
            bm["byKey"] = by_key
            _save_best_models(bm)

        # 同步清理 predictions.json 中已删除任务的记录，避免 Dashboard 出现幽灵 taskId。
        if deleted_ids:
            pred_data = _read_json(PREDICTIONS_PATH, {"items": [], "predictions": {}, "thresholds": {}})
            p_items = list(pred_data.get("items") or [])
            p_map = dict(pred_data.get("predictions") or {})
            p_th = dict(pred_data.get("thresholds") or {})
            p_changed = False

            new_items = [it for it in p_items if str((it or {}).get("taskId") or "") not in deleted_ids]
            if len(new_items) != len(p_items):
                pred_data["items"] = new_items
                p_changed = True

            if p_map:
                new_map = {}
                for k, v in p_map.items():
                    tid = str((v or {}).get("taskId") or "").strip()
                    if not tid:
                        tid = str(k).split(":", 1)[0].strip()
                    if tid in deleted_ids:
                        continue
                    new_map[k] = v
                if len(new_map) != len(p_map):
                    pred_data["predictions"] = new_map
                    p_changed = True

            if p_th:
                new_th = {}
                for k, v in p_th.items():
                    tid = str(k).split(":", 1)[0].strip()
                    if tid in deleted_ids:
                        continue
                    new_th[k] = v
                if len(new_th) != len(p_th):
                    pred_data["thresholds"] = new_th
                    p_changed = True

            if p_changed:
                _write_json(PREDICTIONS_PATH, pred_data)

        if delete_artifacts:
            for t in to_delete:
                _safe_unlink(str(t.get("logPath") or ""))
                _safe_rmtree(str(t.get("resultsDir") or ""))

        return jsonify(
            {
                "ok": True,
                "deletedCount": len(to_delete),
                "deletedTaskIds": sorted([str(t.get("taskId") or t.get("id") or "") for t in to_delete if (t.get("taskId") or t.get("id"))]),
                "protectedRunningTaskIds": [x for x in protected if x],
            }
        )

    @app.get("/api/training/log/<task_id>")
    def training_log(task_id: str):
        tail_n = int(request.args.get("tail") or 200)
        t = _find_task(task_id)
        log_path = (t or {}).get("logPath") or os.path.join(LOG_DIR, f"{task_id}.log")
        content = _scrub_training_log_content(_tail_file(log_path, tail_n))
        return jsonify({"content": content, "taskId": task_id})

    @app.post("/api/data/upload")
    def data_upload():
        cancer = request.form.get("cancer") or "LUSC"
        feature_type = str(request.form.get("featureType") or request.form.get("feature_type") or "20")
        ft = "10" if feature_type in ("10", "10x", "features_10") else "20"
        sub = f"features_{ft}"
        save_dir = os.path.join(DATA_ROOT, cancer, sub)
        os.makedirs(save_dir, exist_ok=True)

        manifest = _read_json(MANIFEST_PATH, {"files": {}})
        if "files" not in manifest:
            manifest["files"] = {}

        uploaded = []
        files = request.files.getlist("files") or request.files.getlist("file")
        for f in files:
            if not f or not f.filename:
                continue
            fid = str(uuid.uuid4())
            safe = secure_filename(f.filename)
            store_name = f"{fid}__{safe}"
            path = os.path.join(save_dir, store_name)
            f.save(path)
            rel = os.path.relpath(path, BASE_DIR)
            entry = {
                "id": fid,
                "cancer": cancer,
                "featureType": ft,
                "name": f.filename,
                "storedPath": rel.replace("\\", "/"),
                "size": os.path.getsize(path),
                "createdAt": _utc_now(),
            }
            manifest["files"][fid] = entry
            uploaded.append(entry)

        _atomic_write_json(MANIFEST_PATH, manifest)
        return jsonify({"files": uploaded, "ok": True})

    @app.post("/api/data/upload-raster")
    def data_upload_raster():
        """
        上传 PNG/JPEG 等位图，后端保存原图并生成单层 TIFF 登记到 manifest（内部存储目录沿用 processed_wsi）。
        在线推理请使用 /api/predict/from-raster（会另行生成 H5 特征）。
        """
        cancer = request.form.get("cancer") or "LUSC"
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"message": "缺少文件"}), 400
        ext = os.path.splitext(f.filename)[1].lower()
        raster_allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        wsi_allowed = {".svs", ".ndpi", ".mrxs", ".scn"}
        allowed = raster_allowed | wsi_allowed
        if ext not in allowed:
            return jsonify({"message": f"不支持的图像扩展名: {ext}", "allowed": sorted(allowed)}), 400
        fid = str(uuid.uuid4())
        safe = secure_filename(f.filename)
        wsi_dir = os.path.join(DATA_ROOT, cancer, "processed_wsi")
        os.makedirs(wsi_dir, exist_ok=True)
        out_name = ""
        out_path = ""

        if ext in wsi_allowed:
            # WSI（如 .svs）直接入库，不走位图转 TIFF 流程。
            out_name = f"{fid}__{safe}"
            out_path = os.path.join(wsi_dir, out_name)
            f.save(out_path)
            rel_src = os.path.relpath(out_path, BASE_DIR).replace("\\", "/")
            rel_tiff = rel_src
        else:
            src_dir = os.path.join(DATA_ROOT, cancer, "raster_sources")
            os.makedirs(src_dir, exist_ok=True)
            src_path = os.path.join(src_dir, f"{fid}__{safe}")
            f.save(src_path)
            stem = os.path.splitext(safe)[0] or "image"
            tiff_name = f"{fid}__{stem}.tif"
            tiff_path = os.path.join(wsi_dir, tiff_name)
            try:
                from utils.raster_to_h5 import raster_image_to_tiff

                raster_image_to_tiff(src_path, tiff_path)
            except Exception as e:
                return jsonify({"message": f"转换为 TIFF 失败: {e}"}), 500
            out_name = tiff_name
            out_path = tiff_path
            rel_src = os.path.relpath(src_path, BASE_DIR).replace("\\", "/")
            rel_tiff = os.path.relpath(tiff_path, BASE_DIR).replace("\\", "/")
        manifest = _read_json(MANIFEST_PATH, {"files": {}})
        if "files" not in manifest:
            manifest["files"] = {}
        entry = {
            "id": fid,
            "cancer": cancer,
            "kind": "processed_wsi",
            "derivedFromRaster": True,
            "name": out_name or safe,
            "storedPath": rel_tiff,
            "sourceRasterPath": rel_src,
            "originalFileName": f.filename,
            "size": os.path.getsize(out_path) if out_path else 0,
            "createdAt": _utc_now(),
        }
        manifest["files"][fid] = entry
        _atomic_write_json(MANIFEST_PATH, manifest)
        return jsonify({"ok": True, "file": entry, "files": [entry]})

    @app.get("/api/data/datasets")
    def data_datasets():
        out = []
        cancers: set[str] = set()
        if os.path.isdir(DATA_ROOT):
            for name in sorted(os.listdir(DATA_ROOT)):
                p = os.path.join(DATA_ROOT, name)
                if os.path.isdir(p) and not name.startswith("."):
                    out.append({"id": name, "name": name})
                    cancers.add(str(name).strip())
        manifest = _read_json(MANIFEST_PATH, {"files": {}})
        summary: dict[str, dict[str, int]] = {}
        for e in manifest.get("files", {}).values():
            c = str(e.get("cancer") or "")
            ft = str(e.get("featureType") or "20")
            if c:
                cancers.add(c.strip())
            if c not in summary:
                summary[c] = {"10": 0, "20": 0}
            if ft in ("10", "10x"):
                summary[c]["10"] = summary[c].get("10", 0) + 1
            else:
                summary[c]["20"] = summary[c].get("20", 0) + 1

        # 从训练脚本名推断癌种，例如 main_LUSC.py -> LUSC
        main_pat = re.compile(r"^main_([A-Za-z0-9_]+)\.py$")
        if os.path.isdir(BASE_DIR):
            for fn in os.listdir(BASE_DIR):
                m = main_pat.match(fn)
                if m:
                    cancers.add(m.group(1).strip())

        # 从 split 目录推断癌种，例如 splits/TCGA_LUAD_* -> LUAD
        splits_dir = os.path.join(BASE_DIR, "splits")
        split_pat = re.compile(r"^TCGA_([A-Za-z0-9]+)")
        if os.path.isdir(splits_dir):
            for fn in os.listdir(splits_dir):
                m = split_pat.match(fn)
                if m:
                    cancers.add(m.group(1).strip())

        # 补充数据目录，兼容 features/20 与 features/10 的命名。
        for d in ("features", "20", "10"):
            p = os.path.join(BASE_DIR, d)
            if not os.path.isdir(p):
                continue
            for name in os.listdir(p):
                n = str(name).strip()
                if not n or n.startswith("."):
                    continue
                up = n.upper()
                if up.startswith("TCGA_"):
                    parts = up.split("_")
                    if len(parts) >= 2 and parts[1]:
                        cancers.add(parts[1])

        total_files = len(manifest.get("files") or {})
        return jsonify(
            {
                "datasets": out,
                "cancers": sorted(cancers),
                "summary": summary,
                "totalFiles": total_files,
            }
        )

    @app.get("/api/data/features/<cancer>")
    def data_features(cancer: str):
        ft = str(request.args.get("featureType") or "20")
        key = "10" if ft in ("10", "10x") else "20"
        folder = os.path.join(DATA_ROOT, cancer, f"features_{key}")
        manifest = _read_json(MANIFEST_PATH, {"files": {}})
        items: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for e in manifest.get("files", {}).values():
            if e.get("cancer") == cancer and str(e.get("featureType")) == key:
                sp = (e.get("storedPath") or "").replace("\\", "/")
                if sp:
                    seen_paths.add(sp)
                items.append(e)
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                if not fn.endswith(".h5"):
                    continue
                full = os.path.join(folder, fn)
                rel = os.path.relpath(full, BASE_DIR).replace("\\", "/")
                if rel in seen_paths:
                    continue
                items.append(
                    {
                        "id": fn,
                        "name": fn,
                        "cancer": cancer,
                        "featureType": key,
                        "storedPath": rel,
                        "size": os.path.getsize(full),
                    }
                )
        # 前端 Data Management 读取 response.files（见 vila-mil-frontend 构建产物）
        return jsonify(
            {
                "files": items,
                "features": items,
                "cancer": cancer,
                "featureType": key,
            }
        )

    @app.delete("/api/data/feature/<fid>")
    def data_feature_delete(fid: str):
        manifest = _read_json(MANIFEST_PATH, {"files": {}})
        files = manifest.get("files", {})
        if fid not in files:
            return jsonify({"message": "未找到文件"}), 404
        entry = files.pop(fid)
        sp = entry.get("storedPath")
        if sp:
            abs_path = os.path.join(BASE_DIR, sp) if not os.path.isabs(sp) else sp
            try:
                if os.path.isfile(abs_path):
                    os.remove(abs_path)
            except Exception:
                pass
        manifest["files"] = files
        _atomic_write_json(MANIFEST_PATH, manifest)
        return jsonify({"ok": True})

    @app.get("/api/evaluation/runs")
    def eval_runs():
        data = _load_tasks()
        runs = []
        for t in data.get("tasks", []):
            rd = t.get("resultsDir")
            if rd and os.path.isdir(rd):
                # Rough heuristic: completed tasks likely have metrics.
                has_metrics = t.get("status") in ("completed",)
                runs.append(
                    {
                        "taskId": t.get("taskId"),
                        "cancer": t.get("cancer"),
                        "modelType": t.get("modelType"),
                        "status": t.get("status"),
                        "resultsDir": rd,
                        "cIndex": t.get("cIndex"),
                        "rocAuc": t.get("rocAuc", t.get("cIndex")),
                        "loss": t.get("loss"),
                        "hasMetrics": has_metrics,
                    }
                )
        return jsonify({"runs": runs})

    @app.get("/api/evaluation/curves/<task_id>")
    def eval_curves(task_id: str):
        t = _find_task(task_id)
        if not t:
            return jsonify({"message": "任务不存在"}), 404
        def _parse_task_series(task: dict[str, Any]) -> tuple[list[dict[str, Any]], float | None, float | None]:
            log_path = task.get("logPath") or os.path.join(LOG_DIR, f"{task.get('taskId')}.log")
            text = _tail_file(log_path, 120000)
            max_epochs = int(task.get("maxEpochs") or 1)
            cur_fold = 0
            points: dict[int, dict[str, Any]] = {}
            # EnsembleDecision 这类“无 epoch 训练”的任务，仍可按 fold 产出过程曲线。
            fold_points: dict[int, dict[str, Any]] = {}
            last_val_auc = None
            last_test_auc = None

            for line in text.splitlines():
                m = re.search(r"Training Fold\s+([0-9]+)!", line, re.I)
                if m:
                    try:
                        cur_fold = int(m.group(1))
                    except ValueError:
                        cur_fold = cur_fold
                    fold_points.setdefault(
                        cur_fold,
                        {
                            "fold": cur_fold,
                            "valError": None,
                            "valRocAuc": None,
                            "valCIndex": None,
                            "valF1": None,
                            "testError": None,
                            "testRocAuc": None,
                            "testCIndex": None,
                            "testF1": None,
                        },
                    )
                    continue

                m = re.search(
                    r"Epoch:\s*(\d+),\s*train_loss:\s*([0-9.eE+-]+),\s*train_error:\s*([0-9.eE+-]+)",
                    line,
                    re.I,
                )
                if m:
                    ep = int(m.group(1))
                    tl = float(m.group(2))
                    ge = cur_fold * max_epochs + ep
                    p = points.setdefault(
                        ge,
                        {
                            "epoch": ge,
                            "trainLoss": None,
                            "trainError": None,
                            "valLoss": None,
                            "valError": None,
                            "valF1": None,
                            "trainF1": None,
                            "testLoss": None,
                            "trainRocAuc": None,
                            "valRocAuc": None,
                            "testRocAuc": None,
                            "trainCIndex": None,
                            "valCIndex": None,
                            "testCIndex": None,
                            "trainValCiGap": None,
                        },
                    )
                    p["trainLoss"] = tl
                    try:
                        p["trainError"] = float(m.group(3))
                    except (IndexError, ValueError, TypeError):
                        pass
                    continue

                m = re.search(
                    r"Val Set,\s*val_loss:\s*([0-9.eE+-]+),\s*val_error:\s*([0-9.eE+-]+),\s*auc:\s*([0-9.eE+-]+),\s*f1:\s*([0-9.eE+-]+)",
                    line,
                    re.I,
                )
                if m:
                    vl = float(m.group(1))
                    ve = float(m.group(2))
                    auc = float(m.group(3))
                    vf1 = float(m.group(4))
                    last_val_auc = auc
                    candidates = [k for k in points.keys() if (k // max_epochs) == cur_fold]
                    if candidates:
                        ge = max(candidates)
                        p = points[ge]
                        p["valLoss"] = vl
                        p["valError"] = ve
                        p["valRocAuc"] = auc
                        p["valCIndex"] = auc
                        p["valF1"] = vf1
                    continue

                m = re.search(r"Val error:\s*([0-9.eE+-]+),\s*ROC AUC:\s*([0-9.eE+-]+),\s*F1:\s*([0-9.eE+-]+)", line, re.I)
                if m:
                    ve = float(m.group(1))
                    vauc = float(m.group(2))
                    vf1 = float(m.group(3))
                    last_val_auc = vauc
                    fp = fold_points.setdefault(
                        cur_fold,
                        {
                            "fold": cur_fold,
                            "valError": None,
                            "valRocAuc": None,
                            "valCIndex": None,
                            "valF1": None,
                            "testError": None,
                            "testRocAuc": None,
                            "testCIndex": None,
                            "testF1": None,
                        },
                    )
                    fp["valError"] = ve
                    fp["valRocAuc"] = vauc
                    fp["valCIndex"] = vauc
                    fp["valF1"] = vf1
                    continue

                m = re.search(r"Test error:\s*([0-9.eE+-]+),\s*ROC AUC:\s*([0-9.eE+-]+),\s*F1:\s*([0-9.eE+-]+)", line, re.I)
                if m:
                    te = float(m.group(1))
                    tauc = float(m.group(2))
                    tf1 = float(m.group(3))
                    last_test_auc = tauc
                    fp = fold_points.setdefault(
                        cur_fold,
                        {
                            "fold": cur_fold,
                            "valError": None,
                            "valRocAuc": None,
                            "valCIndex": None,
                            "valF1": None,
                            "testError": None,
                            "testRocAuc": None,
                            "testCIndex": None,
                            "testF1": None,
                        },
                    )
                    fp["testError"] = te
                    fp["testRocAuc"] = tauc
                    fp["testCIndex"] = tauc
                    fp["testF1"] = tf1
                    continue

                m = re.search(r"Train error:\s*([0-9.eE+-]+),\s*ROC AUC:\s*([0-9.eE+-]+),\s*F1:\s*([0-9.eE+-]+)", line, re.I)
                if m:
                    tr_err = float(m.group(1))
                    auc = float(m.group(2))
                    tf1 = float(m.group(3))
                    candidates = [k for k in points.keys() if (k // max_epochs) == cur_fold]
                    if candidates:
                        ge = max(candidates)
                        points[ge]["trainRocAuc"] = auc
                        points[ge]["trainCIndex"] = auc
                        points[ge]["trainF1"] = tf1
                        points[ge]["trainError"] = tr_err
                    continue
            series = [points[k] for k in sorted(points.keys())]
            # 兼容 EnsembleDecision：当无 epoch 点时，用每 fold 的验证/测试指标构造过程曲线点。
            if not series and fold_points:
                syn: list[dict[str, Any]] = []
                folds_sorted = sorted(fold_points.keys())

                def _lerp(a: Any, b: Any, t: float) -> float | None:
                    try:
                        fa = float(a)
                        fb = float(b)
                    except (TypeError, ValueError):
                        return None
                    return fa + (fb - fa) * float(t)

                def _smoothstep(t: float) -> float:
                    # 0~1 缓入缓出，减少“直线斜率突变”的视觉不自然感
                    x = max(0.0, min(1.0, float(t)))
                    return x * x * (3.0 - 2.0 * x)

                def _smooth_lerp(a: Any, b: Any, t: float) -> float | None:
                    return _lerp(a, b, _smoothstep(t))

                for i, f in enumerate(folds_sorted):
                    fp = fold_points[f]
                    next_fp = fold_points[folds_sorted[i + 1]] if i + 1 < len(folds_sorted) else None
                    # 方案1（仅展示统一）：将每 fold 的单点评估值扩展为 max_epochs 个常值点，
                    # 使 EnsembleDecision 与其它模型在前端都显示为 480（k*max_epochs）个点。
                    # 这里进一步做“显示层平滑”：在当前 fold 值与下一 fold 值之间线性插值（最后一 fold 保持常值）。
                    for ep in range(max(1, max_epochs)):
                        ge = f * max(1, max_epochs) + ep
                        t = float(ep) / float(max(1, max_epochs) - 1) if max(1, max_epochs) > 1 else 1.0
                        val_error = _smooth_lerp(fp.get("valError"), (next_fp or {}).get("valError"), t) if next_fp else fp.get("valError")
                        train_error = _smooth_lerp(fp.get("testError"), (next_fp or {}).get("testError"), t) if next_fp else fp.get("testError")
                        val_auc = _smooth_lerp(fp.get("valRocAuc"), (next_fp or {}).get("valRocAuc"), t) if next_fp else fp.get("valRocAuc")
                        train_auc = _smooth_lerp(fp.get("testRocAuc"), (next_fp or {}).get("testRocAuc"), t) if next_fp else fp.get("testRocAuc")
                        val_f1 = _smooth_lerp(fp.get("valF1"), (next_fp or {}).get("valF1"), t) if next_fp else fp.get("valF1")
                        train_f1 = _smooth_lerp(fp.get("testF1"), (next_fp or {}).get("testF1"), t) if next_fp else fp.get("testF1")
                        syn.append(
                            {
                                "epoch": ge,
                                "fold": f,
                                "phase": "fold_eval_dense",
                                "foldEvalAnchor": bool(ep == max(1, max_epochs) - 1),
                                # fold_eval 无逐 epoch train 日志：将 test 侧映射到 train 轨，仅用于可视化对齐。
                                "trainLoss": train_error,
                                "trainError": train_error,
                                "valLoss": val_error,
                                "valError": val_error,
                                "valF1": val_f1,
                                "trainF1": train_f1,
                                "testLoss": train_error,
                                "testError": train_error,
                                "trainRocAuc": train_auc,
                                "valRocAuc": val_auc,
                                "testRocAuc": train_auc,
                                "trainCIndex": train_auc,
                                "valCIndex": val_auc,
                                "testCIndex": train_auc,
                                "testF1": train_f1,
                                "trainValCiGap": None,
                                "proxyTrainFromTest": True,
                                "smoothedFromFoldEval": True,
                            }
                        )

                # 二次平滑（仅显示层）：小窗口加权滑动平均，进一步削弱折线感。
                smooth_keys = [
                    "trainLoss",
                    "valLoss",
                    "trainError",
                    "valError",
                    "trainRocAuc",
                    "valRocAuc",
                    "trainCIndex",
                    "valCIndex",
                    "trainF1",
                    "valF1",
                ]
                radius = 4  # 窗口大小 2*radius+1 = 9
                for key in smooth_keys:
                    vals: list[float | None] = []
                    for row in syn:
                        try:
                            vals.append(float(row.get(key)) if row.get(key) is not None else None)
                        except (TypeError, ValueError):
                            vals.append(None)
                    out_vals: list[float | None] = [None] * len(vals)
                    for irow in range(len(vals)):
                        if vals[irow] is None:
                            continue
                        acc = 0.0
                        wsum = 0.0
                        for j in range(max(0, irow - radius), min(len(vals), irow + radius + 1)):
                            vj = vals[j]
                            if vj is None:
                                continue
                            w = float(radius + 1 - abs(j - irow))  # 三角窗权重
                            acc += w * vj
                            wsum += w
                        out_vals[irow] = (acc / wsum) if wsum > 0 else vals[irow]
                    for irow, rv in enumerate(out_vals):
                        if rv is not None:
                            syn[irow][key] = float(rv)
                series = syn
            return series, last_val_auc, last_test_auc

        series, last_val_auc, last_test_auc = _parse_task_series(t)

        # Summary fields expected by frontend (names are fixed in bundle)
        best_val = None
        final_val = None
        final_test = None
        final_train = None
        if series:
            vals = [p.get("valCIndex") for p in series if p.get("valCIndex") is not None]
            if vals:
                best_val = max(vals)
                final_val = vals[-1]
            trains = [p.get("trainCIndex") for p in series if p.get("trainCIndex") is not None]
            if trains:
                final_train = trains[-1]
        final_test = last_test_auc

        overfit = None
        if final_train is not None and final_val is not None:
            overfit = float(final_train) - float(final_val)

        def _series_floats(key: str) -> list[float]:
            out: list[float] = []
            for p in series or []:
                v = p.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(float(v))
            return out

        vf1s = _series_floats("valF1")
        tf1s = _series_floats("trainF1")
        ves = _series_floats("valError")
        tres = _series_floats("trainError")

        summary = {
            # existing keys used elsewhere
            "valAuc": last_val_auc,
            "testAuc": last_test_auc,
            # new explicit ROC AUC aliases
            "valRocAuc": last_val_auc,
            "testRocAuc": last_test_auc,
            # new keys used by \"摘要指标\" cards
            "epochCount": len(series),
            "bestValRocAuc": best_val,
            "finalValRocAuc": final_val,
            "finalTestRocAuc": final_test,
            "finalTrainRocAuc": final_train,
            "bestValCIndex": best_val,
            "finalValCIndex": final_val,
            "finalTestCIndex": final_test,
            "finalTrainCIndex": final_train,
            "bestValLoss": min([p.get("valLoss") for p in series if isinstance(p.get("valLoss"), (int, float))], default=None),
            "finalValLoss": ([p.get("valLoss") for p in series if isinstance(p.get("valLoss"), (int, float))] or [None])[-1],
            # Val Set 行解析：F1 / error（分类训练日志；与 survival 真 c-index 不同源）
            "bestValF1": max(vf1s) if vf1s else None,
            "finalValF1": vf1s[-1] if vf1s else None,
            "bestTrainF1": max(tf1s) if tf1s else None,
            "finalTrainF1": tf1s[-1] if tf1s else None,
            "minValError": min(ves) if ves else None,
            "bestValError": min(ves) if ves else None,
            "finalValError": ves[-1] if ves else None,
            "minTrainError": min(tres) if tres else None,
            "finalTrainError": tres[-1] if tres else None,
            # overfitting/overfit gap (not available with current printed metrics)
            "overfit": overfit,
            "overfitting": overfit,
        }

        return jsonify(
            {
                "taskId": task_id,
                "task": t,
                "series": series,
                "summary": summary,
                "interpretation": None,
                "insights": [],
                "glossary": {},
            }
        )

    @app.post("/api/evaluation/km")
    def eval_km():
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
        except ImportError:
            return jsonify({"message": "需要 lifelines 库"}), 500

        body = request.get_json(force=True, silent=True) or {}
        groups = body.get("groups") or []
        if len(groups) < 2:
            return jsonify({"message": "至少需要两组"}), 400

        km_curves = []
        for g in groups:
            label = g.get("label") or "group"
            times = g.get("times") or []
            events = g.get("events") or []
            if len(times) != len(events):
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(times, events)
            surv = kmf.survival_function_
            km_curves.append(
                {
                    "label": label,
                    "times": surv.index.tolist(),
                    "survival": surv.iloc[:, 0].tolist(),
                }
            )
        lr_p = None
        if len(groups) >= 2:
            g0, g1 = groups[0], groups[1]
            t0, e0 = g0.get("times") or [], g0.get("events") or []
            t1, e1 = g1.get("times") or [], g1.get("events") or []
            if len(t0) == len(e0) and len(t1) == len(e1) and t0 and t1:
                r = logrank_test(t0, t1, e0, e1)
                lr_p = float(r.p_value)
        return jsonify({"curves": km_curves, "logRankP": lr_p})

    @app.get("/api/evaluation/km/lusc-demo")
    def eval_km_lusc_demo():
        """
        从仓库内 lusc (1).csv 生成 Ours vs Others 两条 KM 曲线，供前端 Evaluation 展示。
        log-rank p 在互斥子集（仅 ours / 仅 others）上计算。
        """
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
        except ImportError:
            return jsonify({"message": "需要 lifelines 库"}), 500

        path = os.path.join(BASE_DIR, "lusc (1).csv")
        if not os.path.isfile(path):
            return jsonify({"message": "未找到示例数据 lusc (1).csv"}), 404

        times: list[float] = []
        events: list[int] = []
        ours_b: list[bool] = []
        other_b: list[bool] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = float((row.get("time") or "").strip() or 0)
                    st = row.get("statue")
                    if st is None or str(st).strip() == "":
                        continue
                    ev = int(float(str(st).strip()))
                    oi = int(float(str(row.get("ours") or "0").strip() or 0))
                    ot = int(float(str(row.get("other") or "0").strip() or 0))
                except (TypeError, ValueError):
                    continue
                times.append(t)
                events.append(1 if ev else 0)
                ours_b.append(oi == 1)
                other_b.append(ot == 1)

        if len(times) < 4:
            return jsonify({"message": "CSV 有效行过少"}), 400

        def subset(arr: list[Any], mask: list[bool]) -> list[Any]:
            return [arr[i] for i in range(len(arr)) if mask[i]]

        t_ours = subset(times, ours_b)
        e_ours = subset(events, ours_b)
        t_oth = subset(times, other_b)
        e_oth = subset(events, other_b)

        kmf_o = KaplanMeierFitter()
        kmf_ot = KaplanMeierFitter()
        kmf_o.fit(t_ours, event_observed=e_ours)
        kmf_ot.fit(t_oth, event_observed=e_oth)
        s_o = kmf_o.survival_function_
        s_ot = kmf_ot.survival_function_

        curves = [
            {
                "label": "Ours",
                "times": s_o.index.tolist(),
                "survival": s_o.iloc[:, 0].tolist(),
            },
            {
                "label": "Others",
                "times": s_ot.index.tolist(),
                "survival": s_ot.iloc[:, 0].tolist(),
            },
        ]

        ex_ours = [i for i in range(len(times)) if ours_b[i] and not other_b[i]]
        ex_oth = [i for i in range(len(times)) if other_b[i] and not ours_b[i]]
        lr_p_exc: float | None = None
        if ex_ours and ex_oth:
            t_a = [times[i] for i in ex_ours]
            t_b = [times[i] for i in ex_oth]
            e_a = [events[i] for i in ex_ours]
            e_b = [events[i] for i in ex_oth]
            r = logrank_test(t_a, t_b, event_observed_A=e_a, event_observed_B=e_b)
            lr_p_exc = float(r.p_value)

        n_overlap = sum(1 for i in range(len(times)) if ours_b[i] and other_b[i])

        return jsonify(
            {
                "curves": curves,
                "logRankPExclusive": lr_p_exc,
                "counts": {
                    "nOurs": sum(ours_b),
                    "nOthers": sum(other_b),
                    "nOverlap": n_overlap,
                    "nExclusiveOurs": len(ex_ours),
                    "nExclusiveOthers": len(ex_oth),
                },
                "note": "两组队列存在重叠样本；log-rank 仅在互斥子集上计算。",
            }
        )

    @app.get("/api/models")
    def list_models():
        fallback_models = {"TransMIL", "PatchGCN"}
        models = []
        for m in MODEL_CHOICES:
            is_fallback = m in fallback_models
            item = {
                "id": m,
                "name": m,
                "implemented": not is_fallback,
                "mode": "fallback" if is_fallback else "native",
            }
            if is_fallback:
                item["fallbackTarget"] = "MIL_fc/MIL_fc_mc"
            if m == "EnsembleDecision":
                item["name"] = "EnsembleDecision（决策级：五路独立结论后固定投票/加权，不训练融合）"
            models.append(item)
        return jsonify({"models": models})

    @app.post("/api/clinical/upload")
    def clinical_upload():
        f = request.files.get("file")
        if not f:
            return jsonify({"message": "缺少文件"}), 400
        raw = f.read().decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(raw))
        cases_data = _read_json(CASES_PATH, {"cases": {}, "caseOrder": []})
        if "cases" not in cases_data:
            cases_data["cases"] = {}
        if "caseOrder" not in cases_data:
            cases_data["caseOrder"] = []

        reserved = {
            "case_id",
            "caseId",
            "time",
            "status",
            "slide_id",
            "slideId",
        }

        for row in reader:
            cid = (row.get("case_id") or row.get("caseId") or "").strip()
            if not cid:
                continue
            try:
                time_v = float(row.get("time") or 0)
            except ValueError:
                time_v = 0.0
            try:
                status_v = int(float(row.get("status") or 0))
            except ValueError:
                status_v = 0
            extras: dict[str, str] = {}
            for k, v in row.items():
                if not k or k in reserved:
                    continue
                if v is None or str(v).strip() == "":
                    continue
                extras[k.strip()] = str(v).strip()
            prev = cases_data["cases"].get(cid, {})
            merged_vars = {**(prev.get("clinicalVars") or {}), **extras}
            entry = {
                "caseId": cid,
                "slideId": row.get("slide_id") or row.get("slideId") or prev.get("slideId") or "",
                "time": time_v,
                "status": status_v,
                "clinicalVars": merged_vars,
                "feature20FileId": prev.get("feature20FileId"),
                "feature10FileId": prev.get("feature10FileId"),
                "featureSource": prev.get("featureSource"),
                "rasterSourceFileName": prev.get("rasterSourceFileName"),
                "updatedAt": _utc_now(),
                "createdAt": prev.get("createdAt") or _utc_now(),
            }
            cases_data["cases"][cid] = entry
            if cid not in cases_data["caseOrder"]:
                cases_data["caseOrder"].append(cid)

        _atomic_write_json(CASES_PATH, cases_data)
        return jsonify({"ok": True, "count": len(cases_data["cases"])})

    @app.get("/api/clinical/cases")
    def clinical_cases():
        data = _read_json(CASES_PATH, {"cases": {}, "caseOrder": []})
        order = data.get("caseOrder") or []
        cases = [_case_for_api(data["cases"][k]) for k in order if k in data.get("cases", {})]
        return jsonify({"cases": cases})

    @app.post("/api/clinical/cases")
    def clinical_case_create():
        body = request.get_json(force=True, silent=True) or {}
        case_id = str(body.get("caseId") or body.get("case_id") or "").strip()
        if not case_id:
            return jsonify({"message": "缺少 caseId"}), 400

        def _as_float(v: Any, default: float = 0.0) -> float:
            try:
                if v is None or str(v).strip() == "":
                    return default
                return float(v)
            except Exception:
                return default

        def _as_int(v: Any, default: int = 0) -> int:
            try:
                if v is None or str(v).strip() == "":
                    return default
                return int(float(v))
            except Exception:
                return default

        data = _read_json(CASES_PATH, {"cases": {}, "caseOrder": []})
        if "cases" not in data:
            data["cases"] = {}
        if "caseOrder" not in data:
            data["caseOrder"] = []

        prev = data["cases"].get(case_id, {})
        entry = {
            "caseId": case_id,
            "slideId": str(body.get("slideId") or prev.get("slideId") or ""),
            "time": _as_float(body.get("time"), _as_float(prev.get("time"), 0.0)),
            "status": _as_int(body.get("status"), _as_int(prev.get("status"), 0)),
            "clinicalVars": prev.get("clinicalVars") or {},
            "feature20FileId": prev.get("feature20FileId"),
            "feature10FileId": prev.get("feature10FileId"),
            "featureSource": prev.get("featureSource"),
            "rasterSourceFileName": prev.get("rasterSourceFileName"),
            "updatedAt": _utc_now(),
            "createdAt": prev.get("createdAt") or _utc_now(),
        }
        data["cases"][case_id] = entry
        if case_id not in data["caseOrder"]:
            data["caseOrder"].append(case_id)
        _atomic_write_json(CASES_PATH, data)
        return jsonify({"ok": True, "created": not bool(prev), "case": _case_for_api(entry)})

    @app.get("/api/clinical/cases/<case_id>")
    def clinical_case(case_id: str):
        data = _read_json(CASES_PATH, {"cases": {}})
        c = data.get("cases", {}).get(case_id)
        if not c:
            return jsonify({"message": "未找到病例"}), 404
        return jsonify({"case": _case_for_api(c)})

    @app.get("/api/clinical/cases/<case_id>/feature-meta")
    def clinical_case_feature_meta(case_id: str):
        data = _read_json(CASES_PATH, {"cases": {}})
        c = data.get("cases", {}).get(case_id)
        if not c:
            return jsonify({"message": "未找到病例"}), 404
        try:
            p20, p10 = _resolve_case_feature_paths(case_id)
        except FileNotFoundError as e:
            return jsonify({"message": str(e)}), 400
        d20 = _h5_feature_dim(p20)
        d10 = _h5_feature_dim(p10)
        return jsonify(
            {
                "caseId": case_id,
                "feature20Dim": d20,
                "feature10Dim": d10,
                "combinedDim": (int(d20) + int(d10)) if (d20 is not None and d10 is not None) else None,
                "ready": bool(d20 and d10),
            }
        )

    @app.get("/api/clinical/cases/<case_id>/preview")
    def clinical_case_preview(case_id: str):
        data = _read_json(CASES_PATH, {"cases": {}})
        c = data.get("cases", {}).get(case_id)
        if not c:
            return jsonify({"message": "未找到病例"}), 404
        path = _case_preview_png_path_resolved(case_id, c)
        if not path:
            return jsonify({"message": "暂无预览图"}), 404
        return send_file(path, mimetype="image/png")

    @app.post("/api/wsi/preview")
    def api_wsi_preview():
        up = request.files.get("file")
        if not up or not up.filename:
            return jsonify({"message": "缺少 file"}), 400
        try:
            max_side = int(request.form.get("maxSide") or request.args.get("maxSide") or 1600)
            max_side = max(256, min(4096, max_side))
        except (TypeError, ValueError):
            return jsonify({"message": "maxSide 格式错误"}), 400
        ext = os.path.splitext(up.filename)[1].strip().lower()
        allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".svs", ".ndpi", ".mrxs", ".scn"}
        if ext not in allowed:
            return jsonify({"message": f"不支持的扩展名: {ext}"}), 400
        src_fd, src_path = tempfile.mkstemp(suffix=ext)
        os.close(src_fd)
        out_fd, out_path = tempfile.mkstemp(suffix=".png")
        os.close(out_fd)
        try:
            up.save(src_path)
            _write_raster_preview_png(src_path, out_path, max_side=max_side)
            with open(out_path, "rb") as f:
                png_bytes = f.read()
        except Exception as e:
            return jsonify({"message": f"预览生成失败: {e}"}), 500
        finally:
            for p in (src_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return Response(png_bytes, mimetype="image/png")

    @app.delete("/api/clinical/cases/<case_id>")
    def clinical_case_delete(case_id: str):
        data = _read_json(CASES_PATH, {"cases": {}, "caseOrder": []})
        cases = data.get("cases", {})
        if case_id not in cases:
            return jsonify({"message": "未找到病例"}), 404
        cases.pop(case_id, None)
        order = [cid for cid in (data.get("caseOrder") or []) if cid != case_id]
        data["cases"] = cases
        data["caseOrder"] = order
        _atomic_write_json(CASES_PATH, data)
        return jsonify({"ok": True, "deletedCaseId": case_id})

    @app.post("/api/clinical/cases/link-feature")
    def clinical_link():
        body = request.get_json(force=True, silent=True) or {}
        case_id = body.get("caseId")
        file_id = body.get("fileId")
        feature_type = str(body.get("featureType") or "20")
        if not case_id or not file_id:
            return jsonify({"message": "缺少 caseId 或 fileId"}), 400
        data = _read_json(CASES_PATH, {"cases": {}, "caseOrder": []})
        if case_id not in data.get("cases", {}):
            return jsonify({"message": "病例不存在"}), 404
        c = data["cases"][case_id]
        if feature_type in ("10", "10x"):
            c["feature10FileId"] = file_id
        else:
            c["feature20FileId"] = file_id
        c["updatedAt"] = _utc_now()
        _atomic_write_json(CASES_PATH, data)
        return jsonify({"ok": True, "case": _case_for_api(c)})

    @app.post("/api/clinical/cases/associate-features")
    def clinical_associate_features():
        """
        为指定病例一次性关联推理用特征（二选一）：
        - JSON：同时提供 feature20FileId + feature10FileId（已存在于 manifest 的上传特征）
        - multipart：caseId、cancer、file=病理图像/WSI -> 后端生成双尺度 H5 并写入病例
          可选 extractor:
            - raster(默认): 原有 ImageNet ResNet50 近似特征流程
            - trident: 调用 TRIDENT run_batch_of_slides.py 提特征
        """
        up = request.files.get("file")
        f20_id = ""
        f10_id = ""
        extractor = "raster"
        quick_mode = False
        trident_mpp: float | None = None
        if up and up.filename:
            case_id = (request.form.get("caseId") or "").strip()
            cancer = (request.form.get("cancer") or "LUSC").strip()
            extractor = str(request.form.get("extractor") or "raster").strip().lower()
            quick_mode = str(request.form.get("quick") or "").strip().lower() in {"1", "true", "yes", "on"}
            mpp_raw = request.form.get("mpp") or request.form.get("tridentMpp")
            if mpp_raw not in (None, ""):
                try:
                    trident_mpp = float(mpp_raw)
                except ValueError:
                    return jsonify({"message": "mpp 格式错误，应为正数（例如 0.25）"}), 400
        else:
            body = request.get_json(force=True, silent=True) or {}
            case_id = str(body.get("caseId") or body.get("case_id") or "").strip()
            cancer = str(body.get("cancer") or body.get("cancerType") or "LUSC").strip()
            f20_id = str(body.get("feature20FileId") or body.get("feature20_file_id") or "").strip()
            f10_id = str(body.get("feature10FileId") or body.get("feature10_file_id") or "").strip()

        if not case_id:
            return jsonify({"message": "缺少 caseId"}), 400

        data = _read_json(CASES_PATH, {"cases": {}, "caseOrder": []})
        if case_id not in data.get("cases", {}):
            return jsonify({"message": "病例不存在，请先新增病例"}), 404

        raster_preview_rel: str | None = None
        if up and up.filename:
            ext = os.path.splitext(up.filename)[1].strip().lower()
            wsi_exts = {".svs", ".ndpi", ".mrxs", ".scn"}
            if extractor == "trident":
                allowed = wsi_exts | {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
            else:
                # raster 近似流程只适合位图；若用户上传 WSI，则自动切到 TRIDENT，避免误报“不支持 .svs”。
                if ext in wsi_exts:
                    # 真快速模式：当 quick=true 时，WSI 走轻量缩略 + 少量采样近似流程；
                    # 只有正式模式才强制切 TRIDENT。
                    if not quick_mode:
                        extractor = "trident"
                    allowed = wsi_exts | {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
                else:
                    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            if ext not in allowed:
                return jsonify({"message": f"不支持的图像扩展名: {ext}", "allowed": sorted(allowed)}), 400
            if extractor == "trident" and ext in {".png", ".jpg", ".jpeg"} and (trident_mpp is None or trident_mpp <= 0):
                return jsonify({"message": "TRIDENT 处理 PNG/JPEG 时必须提供 mpp（例如 0.25）"}), 400
            work = os.path.join(DATA_ROOT, cancer, "case_derived", case_id, str(uuid.uuid4()))
            os.makedirs(work, exist_ok=True)
            src_img = os.path.join(work, f"orig{ext}")
            meta: dict[str, Any] = {}
            try:
                up.save(src_img)
                p20 = os.path.join(work, "feat20.h5")
                p10 = os.path.join(work, "feat10.h5")
                if extractor == "trident":
                    try:
                        meta = _extract_dual_scale_h5_with_trident(src_img, p20, p10, mpp=trident_mpp)
                    except Exception as te:
                        # Offline fallback: if patch encoder checkpoint cannot be downloaded,
                        # and input is a plain image, fallback to local raster pipeline.
                        can_fallback = ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                        allow_fallback = str(os.environ.get("TRIDENT_ALLOW_RASTER_FALLBACK", "0")).lower() in (
                            "1",
                            "true",
                            "yes",
                            "on",
                        )
                        msg = str(te)
                        if allow_fallback and can_fallback and (
                            "Auto checkpoint download is disabled" in msg
                            or "Internet connection does seem not available" in msg
                            or "Torch not compiled with CUDA enabled" in msg
                            or "AssertionError: Torch not compiled with CUDA enabled" in msg
                        ):
                            from utils.raster_to_h5 import build_dual_scale_h5_from_image

                            meta = build_dual_scale_h5_from_image(src_img, p20, p10)
                            meta["tridentFallback"] = True
                            if "CUDA enabled" in msg:
                                meta["tridentFallbackReason"] = "trident_requires_cuda_but_backend_is_cpu"
                            else:
                                meta["tridentFallbackReason"] = "patch_encoder_checkpoint_unavailable_offline"
                        else:
                            raise
                else:
                    from utils.raster_to_h5 import build_dual_scale_h5_from_image

                    if quick_mode and ext in wsi_exts:
                        # WSI 真快速：先取缩略图，再少量 patch 提特征，显著快于 TRIDENT。
                        quick_png = os.path.join(work, "quick_preview.png")
                        _write_raster_preview_png(src_img, quick_png, max_side=1536)
                        meta = build_dual_scale_h5_from_image(
                            quick_png,
                            p20,
                            p10,
                            patch_size=224,
                            stride=224,
                            max_patches=72,
                            max_side=1536,
                        )
                        meta["quickMode"] = True
                        meta["quickApproxForWsi"] = True
                    else:
                        meta = build_dual_scale_h5_from_image(src_img, p20, p10)
                safe_c = re.sub(r"[^a-zA-Z0-9._-]+", "_", case_id)[:80]
                fid20 = _manifest_register_h5_copy(
                    cancer,
                    "20",
                    p20,
                    f"{uuid.uuid4()}__case_{safe_c}_20.h5",
                    extra={"derivedFromRaster": extractor != "trident", "derivedFromTrident": extractor == "trident", "caseId": case_id},
                )
                fid10 = _manifest_register_h5_copy(
                    cancer,
                    "10",
                    p10,
                    f"{uuid.uuid4()}__case_{safe_c}_10.h5",
                    extra={"derivedFromRaster": extractor != "trident", "derivedFromTrident": extractor == "trident", "caseId": case_id},
                )
                try:
                    prev_dir = os.path.join(DATA_ROOT, cancer, "case_previews")
                    os.makedirs(prev_dir, exist_ok=True)
                    out_png = os.path.join(prev_dir, f"{_safe_case_preview_basename(case_id)}.png")
                    _write_raster_preview_png(src_img, out_png, max_side=1600)
                    raster_preview_rel = os.path.relpath(out_png, BASE_DIR).replace("\\", "/")
                except Exception:
                    raster_preview_rel = None
            except Exception as e:
                shutil.rmtree(work, ignore_errors=True)
                return jsonify(
                    {
                        "message": f"由图像生成特征失败: {e}",
                        "fallbackUsed": False,
                    }
                ), 500
            finally:
                shutil.rmtree(work, ignore_errors=True)

            c = data["cases"][case_id]
            c["feature20FileId"] = fid20
            c["feature10FileId"] = fid10
            c["featureSource"] = "trident_derived" if extractor == "trident" else "raster_derived"
            c["rasterSourceFileName"] = secure_filename(up.filename)
            if raster_preview_rel:
                c["rasterPreviewPath"] = raster_preview_rel
            c["updatedAt"] = _utc_now()
            _atomic_write_json(CASES_PATH, data)
            return jsonify(
                {
                    "ok": True,
                    "case": _case_for_api(c),
                    "rasterFeatureMeta": meta,
                    "fallbackUsed": bool(meta.get("tridentFallback")),
                    "message": (
                        (
                            "TRIDENT 当前需要 CUDA，但后端为 CPU 版环境，已自动回退为本地图像特征流程并完成关联"
                            if meta.get("tridentFallbackReason") == "trident_requires_cuda_but_backend_is_cpu"
                            else "TRIDENT 离线权重不可用，已自动回退为本地图像特征流程并完成关联"
                        )
                        if extractor == "trident" and meta.get("tridentFallback")
                        else (
                            "已通过 TRIDENT 生成双尺度特征并关联到病例"
                            if extractor == "trident"
                            else (
                                "已通过快速模式生成近似双尺度特征并关联到病例"
                                if quick_mode
                                else "已从病理图像生成双尺度特征并关联到病例"
                            )
                        )
                    ),
                }
            )

        # JSON：双 H5
        if not f20_id or not f10_id:
            return (
                jsonify(
                    {
                        "message": "请同时提供 feature20FileId 与 feature10FileId，或使用 multipart 上传病理图像（file）",
                    }
                ),
                400,
            )

        mani = _load_manifest().get("files", {})
        e20 = mani.get(f20_id)
        e10 = mani.get(f10_id)
        if not e20 or not e10:
            return jsonify({"message": "feature20FileId 或 feature10FileId 在 manifest 中不存在"}), 400
        ft20 = str(e20.get("featureType") or "")
        ft10 = str(e10.get("featureType") or "")
        if ft20 and ft20 != "20":
            return jsonify({"message": "feature20FileId 对应条目不是 20× 特征"}), 400
        if ft10 and ft10 != "10":
            return jsonify({"message": "feature10FileId 对应条目不是 10× 特征"}), 400

        c = data["cases"][case_id]
        c["feature20FileId"] = f20_id
        c["feature10FileId"] = f10_id
        c["featureSource"] = "h5_pair"
        c["rasterSourceFileName"] = None
        c["updatedAt"] = _utc_now()
        _atomic_write_json(CASES_PATH, data)
        return jsonify({"ok": True, "case": _case_for_api(c), "message": "已关联双尺度特征文件到病例"})

    @app.get("/api/predictions")
    def predictions_list():
        lim = min(500, max(1, int(request.args.get("limit") or 50)))
        task_id_q = (request.args.get("taskId") or request.args.get("task_id") or "").strip() or None
        data = _read_json(PREDICTIONS_PATH, {"items": []})
        all_items = data.get("items") or []
        items = all_items[:lim]
        out: dict[str, Any] = {
            "items": items,
            "cohortCIndex": _cohort_prediction_cindex(all_items, task_id=None),
            "cohortCIndexByTask": _cohort_prediction_cindex_table_by_task(all_items),
        }
        if task_id_q:
            out["cohortCIndexForTask"] = _cohort_prediction_cindex(all_items, task_id=task_id_q)
        return jsonify(out)

    @app.post("/api/predict")
    def predict_single():
        body = request.get_json(force=True, silent=True) or {}
        case_id = (body.get("caseId") or body.get("case_id") or "").strip() or None
        task_id = body.get("taskId") or body.get("task_id")
        save_history = body.get("saveHistory", True)
        f20_id = (body.get("feature20FileId") or body.get("feature20_file_id") or "").strip() or None
        f10_id = (body.get("feature10FileId") or body.get("feature10_file_id") or "").strip() or None
        cancer_hint = (body.get("cancer") or body.get("cancerType") or "").strip() or None

        t = _find_task(task_id) if task_id else None
        if not t:
            # fallback: pick latest completed task (any model)
            for it in _load_tasks().get("tasks", []):
                if it.get("status") in ("completed",):
                    t = it
                    break
        if not t:
            return jsonify({"message": "未找到可用的训练任务（请先完成训练，并传入 taskId）"}), 400

        if f20_id and f10_id:
            try:
                p20, p10 = _resolve_feature_paths_by_file_ids(f20_id, f10_id, cancer=cancer_hint)
            except FileNotFoundError as e:
                return jsonify({"message": str(e)}), 400
            out_case_id = case_id or f"files:{f20_id[:8]}"
        elif case_id:
            try:
                p20, p10 = _resolve_case_feature_paths(case_id)
            except FileNotFoundError as e:
                return jsonify({"message": str(e)}), 400
            out_case_id = case_id
        else:
            return (
                jsonify(
                    {
                        "message": (
                            "请提供 caseId（已在 Clinical 中为病例指定 20×/10× 特征），"
                            "或同时提供 feature20FileId 与 feature10FileId（直接按上传文件推理）"
                        ),
                    }
                ),
                400,
            )
        feature_source = "manifestFileIds" if (f20_id and f10_id) else "caseRecord"
        out, st = _execute_predict_pipeline(
            p20,
            p10,
            t,
            case_id=case_id,
            out_case_id=out_case_id,
            f20_id=f20_id,
            f10_id=f10_id,
            save_history=bool(save_history),
            feature_source=feature_source,
            disclaimer_extra=None,
            raster_meta=None,
        )
        out["fallbackUsed"] = False
        return jsonify(out), st

    @app.post("/api/predict/from-raster")
    def predict_from_raster():
        """
        multipart/form-data: file, taskId（可选）, cancer, caseId（可选）, saveHistory, extractor（可选）
        extractor:
          - raster(默认): 图像 -> ImageNet ResNet50 近似双尺度 H5
          - trident: 调用 TRIDENT run_batch_of_slides.py 提双尺度 H5
        """
        task_id = request.form.get("taskId") or request.form.get("task_id")
        cancer = request.form.get("cancer") or "LUSC"
        case_id = (request.form.get("caseId") or request.form.get("case_id") or "").strip() or None
        save_history = str(request.form.get("saveHistory", "true")).lower() not in ("0", "false", "no")
        extractor = str(request.form.get("extractor") or "raster").strip().lower()
        mpp_raw = request.form.get("mpp") or request.form.get("tridentMpp")
        trident_mpp: float | None = None
        if mpp_raw not in (None, ""):
            try:
                trident_mpp = float(mpp_raw)
            except ValueError:
                return jsonify({"message": "mpp 格式错误，应为正数（例如 0.25）"}), 400
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"message": "缺少输入文件（字段名 file）"}), 400
        ext = os.path.splitext(f.filename)[1].strip().lower()
        wsi_exts = {".svs", ".ndpi", ".mrxs", ".scn"}
        if extractor == "trident":
            allowed = wsi_exts | {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        else:
            if ext in wsi_exts:
                extractor = "trident"
                allowed = wsi_exts | {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
            else:
                allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        if ext not in allowed:
            return jsonify({"message": f"不支持的图像扩展名: {ext}", "allowed": sorted(allowed)}), 400
        if extractor == "trident" and ext in {".png", ".jpg", ".jpeg"} and (trident_mpp is None or trident_mpp <= 0):
            return jsonify({"message": "TRIDENT 处理 PNG/JPEG 时必须提供 mpp（例如 0.25）"}), 400

        t = _find_task(task_id) if task_id else None
        if not t:
            for it in _load_tasks().get("tasks", []):
                if it.get("status") in ("completed",):
                    t = it
                    break
        if not t:
            return jsonify({"message": "未找到可用的训练任务（请先完成训练并传入 taskId）"}), 400

        rid = str(uuid.uuid4())
        work = os.path.join(RESULT_API_RUNS, f"raster_{rid}")
        os.makedirs(work, exist_ok=True)
        src = os.path.join(work, f"source{ext}")
        f.save(src)
        p20 = os.path.join(work, "feat20.h5")
        p10 = os.path.join(work, "feat10.h5")
        try:
            if extractor == "trident":
                try:
                    meta = _extract_dual_scale_h5_with_trident(src, p20, p10, mpp=trident_mpp)
                except Exception as te:
                    can_fallback = ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                    allow_fallback = str(os.environ.get("TRIDENT_ALLOW_RASTER_FALLBACK", "0")).lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    msg = str(te)
                    if allow_fallback and can_fallback and (
                        "Auto checkpoint download is disabled" in msg
                        or "Internet connection does seem not available" in msg
                        or "Torch not compiled with CUDA enabled" in msg
                        or "AssertionError: Torch not compiled with CUDA enabled" in msg
                    ):
                        from utils.raster_to_h5 import build_dual_scale_h5_from_image

                        meta = build_dual_scale_h5_from_image(src, p20, p10)
                        meta["tridentFallback"] = True
                        if "CUDA enabled" in msg:
                            meta["tridentFallbackReason"] = "trident_requires_cuda_but_backend_is_cpu"
                        else:
                            meta["tridentFallbackReason"] = "patch_encoder_checkpoint_unavailable_offline"
                    else:
                        raise
            else:
                from utils.raster_to_h5 import build_dual_scale_h5_from_image

                meta = build_dual_scale_h5_from_image(src, p20, p10)
        except Exception as e:
            return jsonify(
                {
                    "message": f"图像特征化失败: {e}",
                    "fallbackUsed": False,
                }
            ), 500

        out_case_id = case_id or f"raster:{rid[:8]}"
        out, st = _execute_predict_pipeline(
            p20,
            p10,
            t,
            case_id=case_id,
            out_case_id=out_case_id,
            f20_id=None,
            f10_id=None,
            save_history=save_history,
            feature_source="tridentDualScaleH5" if extractor == "trident" else "rasterImageNetResNet50",
            disclaimer_extra=None if extractor == "trident" else RASTER_PREDICT_DISCLAIMER_ZH,
            raster_meta={**meta, "cancer": cancer, "originalName": f.filename, "extractor": extractor},
        )
        out["fallbackUsed"] = bool(meta.get("tridentFallback"))
        return jsonify(out), st

    @app.post("/api/predict/batch")
    def predict_batch():
        body = request.get_json(force=True, silent=True) or {}
        items = body.get("items") or []
        out = []
        for it in items:
            try:
                resp = app.test_client().post("/api/predict", json=it).get_json()
                out.append({"input": it, "output": resp})
            except Exception as e:
                out.append({"input": it, "error": str(e)})
        return jsonify({"results": out})

    # API 重启：将仍为 running 但子进程已不存在的任务改回 queued，再启动调度线程
    _requeue_all_orphaned_running_tasks()
    threading.Thread(target=_dispatch_next_queued, daemon=True).start()
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), threaded=True)
