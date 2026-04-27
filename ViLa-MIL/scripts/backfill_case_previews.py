#!/usr/bin/env python3
"""
一键补齐历史病例预览图（rasterPreviewPath）。

用途：
1) 扫描 cases.json 中缺失预览图的病例；
2) 在给定目录中按文件名匹配原始图像/WSI；
3) 生成 PNG 缩略图并写入 uploaded_features/<cancer>/case_previews/；
4) 回写 cases.json 的 rasterPreviewPath 字段。

示例：
python scripts/backfill_case_previews.py \
  --search-dir "/Users/zzfly/Desktop/TCGA_LUSC_batch_svs" \
  --search-dir "/Users/zzfly/Desktop"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


SUPPORTED_EXTS = {
    ".svs",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _safe_case_name(case_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(case_id or "")).strip("._-")[:120] or "case"


def _build_name_index(search_dirs: list[Path]) -> dict[str, Path]:
    """
    以 basename 建索引：filename -> absolute path
    同名冲突时保留先扫描到的路径。
    """
    idx: dict[str, Path] = {}
    for d in search_dirs:
        if not d.is_dir():
            continue
        for root, _dirs, files in os.walk(d):
            for fn in files:
                ext = Path(fn).suffix.lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                if fn not in idx:
                    idx[fn] = Path(root) / fn
    return idx


def _resolve_case_cancer(case_row: dict[str, Any], manifest_files: dict[str, Any], default: str) -> str:
    for key in ("feature20FileId", "feature10FileId"):
        fid = str(case_row.get(key) or "").strip()
        if not fid:
            continue
        ent = manifest_files.get(fid) or {}
        c = str(ent.get("cancer") or "").strip()
        if c:
            return c
    return default


def _create_preview_png(src: Path, out_png: Path, max_side: int = 1600) -> None:
    from PIL import Image

    out_png.parent.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        img = Image.open(src).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        img.save(out_png, format="PNG")
        return

    # WSI: openslide
    import openslide

    with openslide.OpenSlide(str(src)) as slide:
        img = slide.get_thumbnail((max_side, max_side)).convert("RGB")
    img.save(out_png, format="PNG")


def main() -> int:
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Backfill clinical case preview PNGs.")
    parser.add_argument("--base-dir", default=str(repo_dir), help="ViLa-MIL 根目录")
    parser.add_argument("--cases-json", default=None, help="cases.json 路径（默认 base/uploaded_features/cases.json）")
    parser.add_argument("--manifest-json", default=None, help="manifest.json 路径（默认 base/uploaded_features/manifest.json）")
    parser.add_argument("--search-dir", action="append", default=[], help="按文件名匹配原图的搜索目录，可重复传")
    parser.add_argument("--default-cancer", default="LUSC", help="无法从 manifest 解析时的默认癌种")
    parser.add_argument("--max-side", type=int, default=1600, help="预览图最长边")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在预览")
    parser.add_argument("--dry-run", action="store_true", help="仅打印，不写文件")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    cases_path = Path(args.cases_json).resolve() if args.cases_json else (base_dir / "uploaded_features" / "cases.json")
    manifest_path = (
        Path(args.manifest_json).resolve() if args.manifest_json else (base_dir / "uploaded_features" / "manifest.json")
    )

    cases_data = _read_json(cases_path, {"cases": {}, "caseOrder": []})
    manifest_data = _read_json(manifest_path, {"files": {}})
    cases_map: dict[str, Any] = cases_data.get("cases") or {}
    mani_files: dict[str, Any] = manifest_data.get("files") or {}

    if not cases_map:
        print("未找到病例数据，退出。")
        return 0

    search_dirs = [Path(p).resolve() for p in args.search_dir]
    if not search_dirs:
        # 给一个兜底：桌面 + 项目目录
        search_dirs = [Path.home() / "Desktop", base_dir]

    print("构建文件名索引中...")
    name_index = _build_name_index(search_dirs)
    print(f"可匹配文件数: {len(name_index)}")

    touched = 0
    skipped_has_preview = 0
    skipped_no_name = 0
    skipped_not_found = 0
    failed = 0

    for case_id, row in cases_map.items():
        row = dict(row or {})
        prev_rel = str(row.get("rasterPreviewPath") or "").strip()
        if prev_rel and not args.overwrite:
            prev_abs = Path(prev_rel) if os.path.isabs(prev_rel) else (base_dir / prev_rel)
            if prev_abs.is_file():
                skipped_has_preview += 1
                continue

        src_name = str(row.get("rasterSourceFileName") or "").strip()
        if not src_name:
            skipped_no_name += 1
            continue

        src_path = name_index.get(src_name)
        if not src_path:
            skipped_not_found += 1
            continue

        cancer = _resolve_case_cancer(row, mani_files, args.default_cancer)
        out_png = base_dir / "uploaded_features" / cancer / "case_previews" / f"{_safe_case_name(case_id)}.png"
        rel = os.path.relpath(out_png, base_dir).replace("\\", "/")

        try:
            if not args.dry_run:
                _create_preview_png(src_path, out_png, max_side=max(256, min(4096, int(args.max_side))))
                row["rasterPreviewPath"] = rel
                cases_map[case_id] = row
            touched += 1
            print(f"[OK] {case_id} <- {src_path.name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {case_id}: {e}")

    if not args.dry_run:
        cases_data["cases"] = cases_map
        _write_json(cases_path, cases_data)

    print(
        json.dumps(
            {
                "totalCases": len(cases_map),
                "updated": touched,
                "skippedHasPreview": skipped_has_preview,
                "skippedNoRasterSourceFileName": skipped_no_name,
                "skippedSourceNotFound": skipped_not_found,
                "failed": failed,
                "dryRun": bool(args.dry_run),
                "casesPath": str(cases_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

