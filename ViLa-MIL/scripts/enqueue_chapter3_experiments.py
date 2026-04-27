#!/usr/bin/env python3
"""
按第 3 章实验思路批量提交训练任务（经 POST /api/training/start）。

默认：LUSC、mode=transformer、kFolds=4、早停开。
- 五个单模基线：lr=1e-5，maxEpochs=120
- EnsembleDecision（决策级固定融合，不训练融合层）：maxEpochs=1，lr 仅占位

用法：
  cd ViLa-MIL
  python scripts/enqueue_chapter3_experiments.py
  python scripts/enqueue_chapter3_experiments.py --base-url http://127.0.0.1:8000/api

仅打印计划：
  python scripts/enqueue_chapter3_experiments.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1/api",
        help="API 根路径（含 /api，不含末尾斜杠）",
    )
    ap.add_argument("--cancer", default="LUSC")
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--k-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    start_url = f"{base}/training/start"

    common = {
        "cancer": args.cancer,
        "mode": "transformer",
        "maxEpochs": args.max_epochs,
        "kFolds": args.k_folds,
        "weightDecay": 1e-5,
        "earlyStopping": True,
        "seed": args.seed,
        "enqueueWhenBusy": True,
    }

    baselines = ["RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL"]
    jobs: list[tuple[str, dict[str, Any]]] = []

    for m in baselines:
        jobs.append(
            (
                f"基线 {m}",
                {**common, "modelType": m, "learningRate": 1e-5, "repeat": 1},
            )
        )

    ed_common = {
        **common,
        "modelType": "EnsembleDecision",
        "maxEpochs": 1,
        "learningRate": 1e-5,
        "repeat": 1,
        "ensembleBranchPriorAuto": True,
        "ensembleBranchPriorTemperature": 0.55,
    }

    jobs.append(("决策级 weighted（主）", {**ed_common, "decisionFusion": "weighted"}))
    jobs.append(("决策级 mean（消融）", {**ed_common, "decisionFusion": "mean"}))
    jobs.append(("决策级 soft_vote（消融）", {**ed_common, "decisionFusion": "soft_vote"}))

    jobs.append(
        (
            "决策级 weighted · 无 Dashboard 自动先验",
            {**ed_common, "decisionFusion": "weighted", "ensembleBranchPriorAuto": False},
        )
    )

    branches = ["RRTMIL", "AMIL", "WiKG", "DSMIL", "S4MIL"]
    for b in branches:
        jobs.append(
            (
                f"决策级 weighted · 留一排除 {b}",
                {**ed_common, "decisionFusion": "weighted", "ensembleExclude": [b]},
            )
        )

    print(f"共 {len(jobs)} 个任务将提交到 {start_url}（基线 maxEpochs={args.max_epochs}，决策级 maxEpochs=1）\n")
    for i, (name, _) in enumerate(jobs, 1):
        print(f"  {i:2}. {name}")

    if args.dry_run:
        print("\n--dry-run：未发送请求。")
        return 0

    ok = 0
    for name, payload in jobs:
        try:
            r = post_json(start_url, payload)
            tid = r.get("taskId") or r.get("id")
            q = r.get("queued")
            print(f"\n[{name}] taskId={tid} queued={q}")
            ok += 1
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            print(f"\n[{name}] HTTP {e.code}: {err}", file=sys.stderr)
        except Exception as e:
            print(f"\n[{name}] {e}", file=sys.stderr)

    print(f"\n完成：成功提交 {ok}/{len(jobs)}")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
