#!/usr/bin/env python3
"""在本地 API 上批量创建演示病例（随访 + 双尺度特征）并执行 Predict。需已启动 api_server（默认 PORT=8000）。"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API = os.environ.get("VILAMIL_API", "http://127.0.0.1:8000/api").rstrip("/")

# 与现有 CASE_001 相同的 LUSC 双尺度特征（manifest 已登记）
F20 = "1eeae5e8-4329-4f86-be12-a0fe924e24b0"
F10 = "b2d017f2-5bc8-4c1e-a2c8-9a60a3acf07e"

# 本地 result/api_runs 下存在 checkpoint 的 LUSC 已完成任务（勿选依赖过重或未装全的单模）
TASK_ID = os.environ.get("VILAMIL_PREDICT_TASK_ID", "2461292b-fec9-4193-a244-83b4d8ec7b58")  # DSMIL LUSC

CASES: list[tuple[str, float, int]] = [
    ("DEMO_LUSC_A01", 18.0, 1),
    ("DEMO_LUSC_A02", 44.0, 0),
    ("DEMO_LUSC_A03", 9.0, 1),
    ("DEMO_LUSC_A04", 52.0, 0),
    ("DEMO_LUSC_A05", 28.0, 1),
    ("DEMO_LUSC_A06", 36.0, 0),
]


def req(method: str, path: str, body: dict | None = None, timeout: int = 600) -> tuple[int, dict]:
    url = f"{API}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"message": raw or str(e)}


def main() -> int:
    print("API:", API)
    st, h = req("GET", "/health", None, timeout=10)
    if st != 200 or not h.get("ok"):
        print("health 失败:", st, h, file=sys.stderr)
        return 1
    print("health OK")

    for case_id, time_v, status_v in CASES:
        st, out = req(
            "POST",
            "/clinical/cases",
            {"caseId": case_id, "slideId": f"{case_id}.svs", "time": time_v, "status": status_v},
            timeout=30,
        )
        if st not in (200, 201):
            print("create case 失败", case_id, st, out, file=sys.stderr)
            return 1
        st, out = req(
            "POST",
            "/clinical/cases/associate-features",
            {
                "caseId": case_id,
                "cancer": "LUSC",
                "feature20FileId": F20,
                "feature10FileId": F10,
            },
            timeout=30,
        )
        if st != 200:
            print("associate-features 失败", case_id, st, out, file=sys.stderr)
            return 1
        print("病例就绪:", case_id, "time=", time_v, "status=", status_v)

    for case_id, _t, _s in CASES:
        print("预测:", case_id, "...")
        st, out = req(
            "POST",
            "/predict",
            {"caseId": case_id, "taskId": TASK_ID, "saveHistory": True},
            timeout=600,
        )
        if st != 200:
            print("predict 失败", case_id, st, out, file=sys.stderr)
            return 1
        rs = out.get("riskScore")
        print("  OK riskScore=", rs, "taskId=", out.get("taskId"))

    print("全部完成。可在 Prediction 页查看历史预测与队列 C-index。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
