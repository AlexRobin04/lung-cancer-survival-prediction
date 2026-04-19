#!/usr/bin/env python3
"""
ViLa-MIL 后端 HTTP 全量接口探测（经网关 http://host/ 的 /api/*）。

用法：
  python3 scripts/api_full_test.py
  python3 scripts/api_full_test.py --base http://127.0.0.1:8080

说明：
  - 默认只做「可安全自动化」的用例：校验状态码与 JSON 形态，避免默认启动真实训练。
  - 加 --allow-training-smoke 时，若仓库内存在 features/20 与 features/10，会尝试启动 1 epoch 的 RRTMIL
    烟雾训练（仍可能因资源预检失败而 400，属预期）。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# 1x1 PNG
_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


@dataclass
class Case:
    name: str
    method: str
    path: str
    expect_status: int | tuple[int, ...]
    body: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str | None = None
    note: str = ""


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"----vilaBoundary{uuid.uuid4().hex}"
    crlf = b"\r\n"
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}".encode() + crlf)
        parts.append(f'Content-Disposition: form-data; name="{k}"'.encode() + crlf + crlf)
        parts.append(str(v).encode("utf-8") + crlf)
    for k, (filename, data, ctype) in files.items():
        parts.append(f"--{boundary}".encode() + crlf)
        disp = f'Content-Disposition: form-data; name="{k}"; filename="{filename}"'
        parts.append(disp.encode() + crlf)
        parts.append(f"Content-Type: {ctype}".encode() + crlf + crlf)
        parts.append(data + crlf)
    parts.append(f"--{boundary}--".encode() + crlf)
    return b"".join(parts), boundary


def _request(
    base: str,
    method: str,
    path: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str, bytes]:
    url = base.rstrip("/") + path
    h = dict(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.getcode(), resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def _json_body(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def run_cases(base: str, cases: list[Case]) -> tuple[int, int]:
    ok = 0
    fail = 0
    for c in cases:
        hdrs = dict(c.headers)
        data = c.body
        if c.content_type:
            hdrs["Content-Type"] = c.content_type
        code, ctype, raw = _request(base, c.method, c.path, data=data, headers=hdrs)
        exp = c.expect_status if isinstance(c.expect_status, tuple) else (c.expect_status,)
        passed = code in exp
        snippet = raw[:240].decode("utf-8", errors="replace").replace("\n", " ")
        tag = "OK " if passed else "FAIL"
        if not passed:
            fail += 1
        else:
            ok += 1
        extra = f"  ({c.note})" if c.note else ""
        print(f"[{tag}] {c.method} {c.path} -> {code} (expect {exp}){extra}")
        if not passed:
            print(f"       body: {snippet}")
        elif "application/json" in (ctype or "") and raw:
            try:
                json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                print(f"       WARN: non-JSON body: {snippet}")
    return ok, fail


def main() -> int:
    ap = argparse.ArgumentParser(description="ViLa-MIL API full smoke test (via gateway)")
    ap.add_argument("--base", default="http://127.0.0.1", help="网关根 URL（无尾斜杠）")
    ap.add_argument(
        "--allow-training-smoke",
        action="store_true",
        help="允许在具备特征目录时尝试启动极短 RRTMIL 训练（有副作用）",
    )
    args = ap.parse_args()
    base = args.base.rstrip("/")

    fake_tid = str(uuid.uuid4())
    case_id = f"api-smoke-{uuid.uuid4().hex[:8]}"

    mp_upload, b_upload = _multipart(
        {"cancer": "LUSC", "featureType": "20"},
        {"files": ("tiny.png", _MIN_PNG, "image/png")},
    )
    mp_raster, b_raster = _multipart(
        {"cancer": "LUSC"},
        {"file": ("tiny.png", _MIN_PNG, "image/png")},
    )
    csv_clinical = "case_id,time,status,age\n" + f"{case_id},12,1,55\n"
    mp_clin, b_clin = _multipart({}, {"file": ("c.csv", csv_clinical.encode("utf-8"), "text/csv")})
    mp_empty, b_empty = _multipart({}, {})
    mp_cancer_only, b_cancer_only = _multipart({"cancer": "LUSC"}, {})

    km_body = _json_body(
        {
            "groups": [
                {"label": "A", "times": [1, 2, 5, 8], "events": [1, 1, 0, 1]},
                {"label": "B", "times": [2, 4, 6, 9], "events": [1, 0, 1, 1]},
            ]
        }
    )

    cases: list[Case] = [
        Case("health", "GET", "/api/health", 200),
        Case("config", "GET", "/api/config", 200),
        Case("models", "GET", "/api/models", 200),
        Case("training start unsupported model", "POST", "/api/training/start", 400, _json_body({"modelType": "NotARealModel"}), content_type="application/json"),
        Case("training stop no id", "POST", "/api/training/stop", 400, _json_body({}), content_type="application/json"),
        Case("training status 404", "GET", f"/api/training/status/{fake_tid}", 404),
        Case("training history", "GET", "/api/training/history", 200),
        Case("training queue", "GET", "/api/training/queue", 200),
        Case("queue delete no body", "POST", "/api/training/queue/delete", 400, _json_body({}), content_type="application/json"),
        Case("training best (may 404)", "GET", "/api/training/best?cancer=LUSC&modelType=RRTMIL&mode=transformer", (200, 404)),
        Case("history delete no body", "POST", "/api/training/history/delete", 400, _json_body({}), content_type="application/json"),
        Case("training log (no task)", "GET", f"/api/training/log/{fake_tid}?tail=5", 200),
        Case("data upload empty", "POST", "/api/data/upload", 200, mp_empty, content_type=f"multipart/form-data; boundary={b_empty}"),
        Case("data upload tiny png", "POST", "/api/data/upload", 200, mp_upload, content_type=f"multipart/form-data; boundary={b_upload}"),
        Case("data upload-raster no file", "POST", "/api/data/upload-raster", 400, mp_cancer_only, content_type=f"multipart/form-data; boundary={b_cancer_only}"),
        Case("data upload-raster tiny", "POST", "/api/data/upload-raster", (200, 500), mp_raster, content_type=f"multipart/form-data; boundary={b_raster}", note="500 若容器内缺图像依赖"),
        Case("datasets", "GET", "/api/data/datasets", 200),
        Case("features LUSC", "GET", "/api/data/features/LUSC", 200),
        Case("delete feature missing", "DELETE", "/api/data/feature/no-such-fid-000", 404),
        Case("eval runs", "GET", "/api/evaluation/runs", 200),
        Case("eval curves 404", "GET", f"/api/evaluation/curves/{fake_tid}", 404),
        Case("eval km bad body", "POST", "/api/evaluation/km", 400, _json_body({"groups": []}), content_type="application/json"),
        Case("eval km ok", "POST", "/api/evaluation/km", (200, 500), km_body, content_type="application/json", note="500 若未装 lifelines"),
        Case("eval km lusc demo", "GET", "/api/evaluation/km/lusc-demo", (200, 400, 404, 500), note="依赖镜像内 CSV / lifelines"),
        Case("clinical upload no file", "POST", "/api/clinical/upload", 400, mp_empty, content_type=f"multipart/form-data; boundary={b_empty}"),
        Case("clinical cases list", "GET", "/api/clinical/cases", 200),
        Case("clinical create no id", "POST", "/api/clinical/cases", 400, _json_body({}), content_type="application/json"),
        Case("clinical create", "POST", "/api/clinical/cases", 200, _json_body({"caseId": case_id, "time": 10, "status": 1}), content_type="application/json"),
        Case("clinical get", "GET", f"/api/clinical/cases/{case_id}", 200),
        Case("clinical feature-meta", "GET", f"/api/clinical/cases/{case_id}/feature-meta", 400),
        Case("clinical upload csv", "POST", "/api/clinical/upload", 200, mp_clin, content_type=f"multipart/form-data; boundary={b_clin}"),
        Case("clinical link bad", "POST", "/api/clinical/cases/link-feature", 400, _json_body({}), content_type="application/json"),
        Case("clinical associate no ids", "POST", "/api/clinical/cases/associate-features", 400, _json_body({"caseId": case_id}), content_type="application/json"),
        Case("predictions", "GET", "/api/predictions", 200),
        Case("predict no input", "POST", "/api/predict", 400, _json_body({}), content_type="application/json"),
        Case("predict from-raster no file", "POST", "/api/predict/from-raster", 400, mp_cancer_only, content_type=f"multipart/form-data; boundary={b_cancer_only}"),
        Case("predict batch empty", "POST", "/api/predict/batch", 200, _json_body({"items": []}), content_type="application/json"),
        Case("clinical delete", "DELETE", f"/api/clinical/cases/{case_id}", 200),
    ]

    print(f"Base URL: {base}\n")
    ok, fail = run_cases(base, cases)

    if args.allow_training_smoke:
        print("\n--allow-training-smoke: POST /api/training/start (RRTMIL, k=1, maxEpochs=1)")
        body = _json_body(
            {
                "cancer": "LUSC",
                "modelType": "RRTMIL",
                "mode": "transformer",
                "maxEpochs": 1,
                "kFolds": 1,
                "enqueueWhenBusy": False,
            }
        )
        code, _, raw = _request(base, "POST", "/api/training/start", data=body, headers={"Content-Type": "application/json"})
        exp = (200, 400, 409, 500)
        passed = code in exp
        print(f"[{'OK ' if passed else 'FAIL'}] POST /api/training/start -> {code} (expect one of {exp})")
        print(f"       {raw[:400].decode('utf-8', errors='replace')}")
        if passed:
            ok += 1
        else:
            fail += 1

    print(f"\n合计: 通过 {ok}，失败 {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
