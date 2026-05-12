#!/usr/bin/env python3
"""
将目录中的 SVS 批量导入 Clinical：为每张切片创建病例并调用
POST /api/clinical/cases/associate-features（与前端一致）。

用法（在 ViLa-MIL 目录下）:
  python3 scripts/clinical_import_svs_dir.py /path/to/svs_dir

环境: 需能 import api_server（依赖 ViLa-MIL 的 Python 环境）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# 仓库内 api_server 与 scripts 同级
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

os.chdir(BASE)

TCGA_BARCODE = re.compile(r"(TCGA-[0-9]{2}-[0-9]{4})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("wsi_dir", help="含 .svs 的目录")
    p.add_argument("--cancer", default="LUSC", help="与训练/预测任务一致的癌种键（默认 LUSC）")
    p.add_argument("--extractor", default="raster", choices=("raster", "trident"))
    p.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="WSI + raster 时使用快速近似（默认开启）",
    )
    p.add_argument("--no-quick", action="store_false", dest="quick")
    p.add_argument(
        "--demo-fake",
        action="store_true",
        help="multipart 带 demoFakeExtraction：复制模板 H5 绑定（需环境变量 VILA_ALLOW_CLINICAL_DEMO_FAKE_EXTRACTION=1；无 torch 时可用）",
    )
    args = p.parse_args()

    if args.demo_fake:
        os.environ.setdefault("VILA_ALLOW_CLINICAL_DEMO_FAKE_EXTRACTION", "1")

    from api_server import create_app

    app = create_app()
    client = app.test_client()

    wsi_dir = os.path.expanduser(args.wsi_dir)
    if not os.path.isdir(wsi_dir):
        print("not a directory:", wsi_dir, file=sys.stderr)
        return 2

    files = sorted(
        f for f in os.listdir(wsi_dir) if f.lower().endswith(".svs") and not f.startswith(".")
    )
    if not files:
        print("no .svs in", wsi_dir, file=sys.stderr)
        return 2

    ok = 0
    for name in files:
        path = os.path.join(wsi_dir, name)
        m = TCGA_BARCODE.search(name)
        case_id = f"LUAD_IMPORT_{m.group(1)}" if m else f"LUAD_IMPORT_{os.path.splitext(name)[0][:64]}"

        rv = client.post(
            "/api/clinical/cases",
            json={"caseId": case_id, "time": 1.0, "status": 1, "slideId": ""},
            content_type="application/json",
        )
        if rv.status_code not in (200, 201):
            print("create case failed", case_id, rv.status_code, rv.data[:500], file=sys.stderr)
            continue

        with open(path, "rb") as fp:
            data = {
                "caseId": case_id,
                "cancer": args.cancer,
                "extractor": args.extractor,
                "quick": "true" if args.quick else "false",
                "file": (fp, name),
            }
            if args.demo_fake:
                data["demoFakeExtraction"] = "true"
            rv2 = client.post(
                "/api/clinical/cases/associate-features",
                data=data,
                content_type="multipart/form-data",
            )

        if rv2.status_code != 200:
            print("associate failed", case_id, rv2.status_code, rv2.data[:800], file=sys.stderr)
            continue

        body = rv2.get_json(silent=True) or {}
        print(json.dumps({"caseId": case_id, "ok": body.get("ok"), "message": body.get("message")}, ensure_ascii=False))
        ok += 1

    print("imported", ok, "/", len(files))
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
