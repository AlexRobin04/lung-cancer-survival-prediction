"""
Standalone batch prediction script.
Loads model from checkpoint, predicts all TCGA cases, compares risk scores with clinical data.
Usage: /Users/zzfly/miniconda3/envs/ViLa-MIL/bin/python standalone_predict.py
"""
import json, os, sys, warnings
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import model loading from api_server
from api_server import (
    _load_transformer_baseline_model,
    _discover_checkpoints,
    _resolve_task_results_dir,
    RESULT_API_RUNS,
    DATA_ROOT,
    TASKS_PATH,
)

warnings.filterwarnings("ignore")

TASK_ID = "9b043b7e-16a7-4c1e-a520-89b15b93cb14"
MAX_CASES = 0  # 0 = predict all available cases

def load_task(task_id):
    with open(TASKS_PATH) as f:
        data = json.load(f)
    for t in data.get("tasks", []):
        if (t.get("taskId") or t.get("id") or "") == task_id:
            return t
    return None

def resolve_case_features(case_id):
    """Resolve feature h5 paths from caseId, mirroring _resolve_case_feature_paths."""
    with open(os.path.join(DATA_ROOT, "cases.json")) as f:
        cases_data = json.load(f)
    c = cases_data["cases"].get(case_id)
    if not c:
        return None, None, None
    f20_id = c.get("feature20FileId")
    f10_id = c.get("feature10FileId")
    if not f20_id or not f10_id:
        return None, None, None
    with open(os.path.join(DATA_ROOT, "manifest.json")) as f:
        manifest = json.load(f)
    e20 = manifest["files"].get(f20_id)
    e10 = manifest["files"].get(f10_id)
    if not e20 or not e10:
        return None, None, None
    p20 = os.path.join(os.path.dirname(DATA_ROOT), e20["storedPath"])
    p10 = os.path.join(os.path.dirname(DATA_ROOT), e10["storedPath"])
    if not os.path.isfile(p20) or not os.path.isfile(p10):
        return None, None, None
    time_val = c.get("time", 0)
    status_val = c.get("status", 0)
    return p20, p10, {"time": time_val, "status": status_val}

def main():
    task = load_task(TASK_ID)
    if not task:
        print(f"Task {TASK_ID} not found")
        return

    model_type = str(task.get("modelType") or "").strip()
    print(f"Model: {model_type}, taskId: {TASK_ID}")

    results_dir = _resolve_task_results_dir(task)
    ckpts = _discover_checkpoints(results_dir)
    print(f"Checkpoints: {len(ckpts)}")
    if not ckpts:
        print("No checkpoints found!")
        return

    # Load cases
    with open(os.path.join(DATA_ROOT, "cases.json")) as f:
        cases_data = json.load(f)

    tcga_cases = []
    for k, v in cases_data["cases"].items():
        cid = v.get("caseId", "")
        if cid.startswith("TCGA-") and "_" not in cid:
            tcga_cases.append(cid)

    print(f"TCGA cases to predict: {len(tcga_cases)}")

    # Load model once (use first checkpoint)
    feat_dim = 512  # RRTMIL uses 512-dim features (20x only or combined?)
    # Actually for RRTMIL: the features are 512-dim (20x), not combined 1024
    # Let me check by reading one feature file first

    sample_cid = tcga_cases[0]
    p20, p10, clin = resolve_case_features(sample_cid)
    if p20:
        with h5py.File(p20, "r") as f:
            sample_feat = np.array(f["features"])
        feat_dim = sample_feat.shape[-1]
        print(f"Feature dimension: {feat_dim} (from 20x)")
    else:
        print(f"Cannot resolve features for {sample_cid}")
        return

    model = _load_transformer_baseline_model(
        ckpt_path=ckpts[0], model_type=model_type, feat_dim=feat_dim, n_classes=4
    )
    model.eval()

    # Limit to MAX_CASES (0 = all)
    if MAX_CASES > 0:
        tcga_cases = tcga_cases[:MAX_CASES]
        print(f"Predicting first {MAX_CASES} cases")

    results = []
    errors = []
    for i, cid in enumerate(tcga_cases):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(tcga_cases)}")
        p20, p10, clin = resolve_case_features(cid)
        if p20 is None:
            errors.append({"caseId": cid, "error": "feature_not_found"})
            continue
        try:
            with h5py.File(p20, "r") as f:
                x_s = torch.from_numpy(np.array(f["features"])).float()
                c_s = torch.from_numpy(np.array(f["coords"])).float()
            with h5py.File(p10, "r") as f:
                x_l = torch.from_numpy(np.array(f["features"])).float()
                c_l = torch.from_numpy(np.array(f["coords"])).float()

            if x_s.size(0) == 0 or x_l.size(0) == 0:
                errors.append({"caseId": cid, "error": "empty_features"})
                continue

            with torch.no_grad():
                _logits, y_prob, _loss = model(x_s, c_s, x_l, c_l, None)

            p = y_prob.detach().cpu().squeeze(0).tolist()
            if isinstance(p, float):
                probs = [p]
            else:
                probs = list(p)

            # Invert: class 0 (shortest survival) = highest mortality risk
            n = len(probs)
            risk_score = sum((n - 1 - i) * prob for i, prob in enumerate(probs)) / max(n, 1)

            results.append({
                "caseId": cid,
                "riskScore": round(float(risk_score), 6),
                "probs": [round(float(x), 6) for x in probs],
                "time": clin["time"],
                "status": clin["status"],
            })
        except Exception as e:
            errors.append({"caseId": cid, "error": str(e)[:200]})

    print(f"\nDone: {len(results)} successes, {len(errors)} errors")

    # Save results
    output = {
        "taskId": TASK_ID,
        "modelType": model_type,
        "predictions": results,
        "errors": errors,
    }
    out_path = "/tmp/prediction_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")

    # Summary statistics
    if results:
        scores = [r["riskScore"] for r in results]
        times = [r["time"] for r in results]
        print(f"\n=== Summary ===")
        print(f"Risk score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"Risk score mean: {sum(scores)/len(scores):.4f}")
        print(f"Time range: [{min(times):.4f}, {max(times):.4f}]")
        print(f"Time mean: {sum(times)/len(times):.4f}")

        # Compute C-index
        def _cindex(t, e, s):
            n = int(len(t))
            conc = ties = disc = 0
            comp = 0
            for i in range(n):
                for j in range(i + 1, n):
                    ti, tj = float(t[i]), float(t[j])
                    ei, ej = int(e[i]), int(e[j])
                    si, sj = float(s[i]), float(s[j])
                    if ti < tj and ei == 1:
                        comp += 1
                        if si > sj: conc += 1
                        elif si < sj: disc += 1
                        else: ties += 1
                    elif tj < ti and ej == 1:
                        comp += 1
                        if sj > si: conc += 1
                        elif sj < si: disc += 1
                        else: ties += 1
            if comp == 0: return float("nan"), 0
            return (conc + 0.5 * ties) / comp, comp

        cidx, pairs = _cindex(times, [r["status"] for r in results], scores)
        print(f"\n=== C-index ===")
        print(f"Harrell C-index: {cidx:.4f} (comparable pairs: {pairs})")

if __name__ == "__main__":
    main()
