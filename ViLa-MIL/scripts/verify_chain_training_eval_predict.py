#!/usr/bin/env python3
"""
全链路验证：best 任务 → 评估曲线摘要 → 批量预测 → 各 task 队列 C-index。
用法: python3 scripts/verify_chain_training_eval_predict.py [API根，默认 http://127.0.0.1]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import quote

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1").rstrip("/")


def get(path: str, timeout: int = 120) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post(path: str, obj: dict, timeout: int = 1200) -> dict:
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def usable_cases() -> list[str]:
    d = get("/api/clinical/cases")
    out: list[str] = []
    for c in d.get("cases") or []:
        cid = str(c.get("caseId") or "").strip()
        if not cid or not c.get("feature20FileId") or not c.get("feature10FileId"):
            continue
        try:
            t = float(c.get("time") or 0)
            ev = int(c.get("status") if c.get("status") is not None else -1)
        except (TypeError, ValueError):
            continue
        if t > 0 and ev in (0, 1):
            out.append(cid)
    return out


def best_task_id(model: str) -> str | None:
    d = get(f"/api/training/best?cancer=LUSC&modelType={model}&mode=transformer")
    tid = d.get("bestTaskId")
    return str(tid).strip() if tid else None


def main() -> int:
    print("=== 1) 病例 ===")
    cases = usable_cases()
    print(f"可用: {len(cases)}")
    if len(cases) < 2:
        return 1
    use = cases[:31]

    models = ["RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL", "EnsembleDecision"]
    print("\n=== 2) bestTaskId ===")
    task_by_model: dict[str, str] = {}
    for m in models:
        tid = best_task_id(m)
        if not tid and m == "EnsembleDecision":
            hist = get("/api/training/history")
            cand = [
                t
                for t in hist.get("tasks", [])
                if str(t.get("modelType") or "") == "EnsembleDecision"
                and str(t.get("status") or "").lower() == "completed"
            ]
            best_tid, best_auc = None, -1.0
            for t in cand[:12]:
                xt = str(t.get("taskId") or "").strip()
                if not xt:
                    continue
                try:
                    ev = get(f"/api/evaluation/curves/{xt}")
                    s = ev.get("summary") or {}
                    v = s.get("bestValRocAuc") or s.get("bestValCIndex")
                    v = float(v) if v is not None else -1.0
                except (urllib.error.HTTPError, urllib.error.URLError, ValueError, TypeError):
                    v = -1.0
                if v > best_auc:
                    best_auc, best_tid = v, xt
            if best_tid:
                tid = best_tid
                print(f"{m}: API 无 bestTaskId，曲线 bestValAuc 回退 {tid[:12]}… (~{best_auc:.4f})")
        if not tid:
            print(f"{m}: 跳过")
            continue
        task_by_model[m] = tid
        print(f"  {m}: {tid}")

    print("\n=== 3) 评估 bestValAUC ===")
    for m, tid in task_by_model.items():
        try:
            ev = get(f"/api/evaluation/curves/{tid}")
            s = ev.get("summary") or {}
            b = s.get("bestValRocAuc") or s.get("bestValCIndex")
            print(f"  {m}: {b}")

    print(f"\n=== 4) 批量预测 ×{len(use)} ===")
    for m, tid in task_by_model.items():
        items = [{"caseId": cid, "taskId": tid, "saveHistory": True} for cid in use]
        t0 = time.time()
        try:
            post("/api/predict/batch", {"items": items}, timeout=2400)
        except Exception as e:
            print(f"  {m}: {e}")
            continue
        print(f"  {m}: ok {time.time() - t0:.1f}s")

    print("\n=== 5) 队列 C-index ===")
    for m, tid in task_by_model.items():
        try:
            pr = get(f"/api/predictions?limit=50&taskId={quote(tid, safe='')}")
            c = (pr.get("cohortCIndexForTask") or {}).get("cIndex")
            print(f"  {m}: {c}")
        except Exception as e:
            print(f"  {m}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
