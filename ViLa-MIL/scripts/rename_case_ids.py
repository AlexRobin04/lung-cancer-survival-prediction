"""
批量重命名 cases.json 中所有 caseId：按 caseOrder 顺序，
第 n 个末尾加上 _n（即第 1 个加 _1，第 2 个加 _2 ...）。

同步更新 predictions.json 和 manifest.json 中的 caseId 引用。
"""
import json
import os
import sys

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploaded_features")

def update_cases(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases") or {}
    order = data.get("caseOrder") or []

    old_to_new = {}
    new_cases = {}
    new_order = []

    for idx, cid in enumerate(order):
        if cid not in cases:
            new_order.append(cid)
            continue
        suffix = idx + 1  # 1-based
        new_id = f"{cid}_{suffix}"
        old_to_new[cid] = new_id
        entry = cases[cid]
        entry["caseId"] = new_id
        new_cases[new_id] = entry
        new_order.append(new_id)

    # Keep cases not in order but in the dict
    for cid in cases:
        if cid not in old_to_new:
            new_cases[cid] = cases[cid]
            # append to order if not already
            if cid not in new_order:
                new_order.append(cid)

    data["cases"] = new_cases
    data["caseOrder"] = new_order

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return old_to_new


def update_predictions(path, old_to_new):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    changed = False

    # items array
    items = data.get("items") or []
    for item in items:
        cid = item.get("caseId") or ""
        if cid in old_to_new:
            item["caseId"] = old_to_new[cid]
            changed = True

    # old predictions dict (keys like taskId:fold:caseId)
    preds = data.get("predictions") or {}
    new_preds = {}
    for key, val in preds.items():
        cid = val.get("caseId") or ""
        if cid in old_to_new:
            val["caseId"] = old_to_new[cid]
            new_key_parts = key.rsplit(":", 1)
            if len(new_key_parts) == 2:
                new_key = f"{new_key_parts[0]}:{old_to_new[cid]}"
            else:
                new_key = key
            new_preds[new_key] = val
            changed = True
        else:
            new_preds[key] = val
    if changed:
        data["predictions"] = new_preds

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def update_manifest(path, old_to_new):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    files = data.get("files") or {}
    changed = False
    for key, entry in files.items():
        cid = entry.get("caseId") or ""
        if cid in old_to_new:
            entry["caseId"] = old_to_new[cid]
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    if not os.path.isdir(BASE):
        print(f"uploaded_features 目录不存在: {BASE}")
        sys.exit(1)

    cases_path = os.path.join(BASE, "cases.json")
    if not os.path.isfile(cases_path):
        print("cases.json 不存在，退出")
        sys.exit(1)

    print(f"读取 {cases_path} ...")
    old_to_new = update_cases(cases_path)
    print(f"共重命名 {len(old_to_new)} 个病例：")
    for old, new in old_to_new.items():
        print(f"  {old} → {new}")

    pred_path = os.path.join(BASE, "predictions.json")
    update_predictions(pred_path, old_to_new)
    print(f"已同步更新 {pred_path}")

    man_path = os.path.join(BASE, "manifest.json")
    update_manifest(man_path, old_to_new)
    print(f"已同步更新 {man_path}")

    print("\n完成！")


if __name__ == "__main__":
    main()
