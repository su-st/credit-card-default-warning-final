"""L0 only: reproduce M0 without changing any research settings.

Run: python experiments/00_reproduce_midterm.py
Optional independent historical evidence: --midterm-indices JSON_FILE
(a JSON array of zero-based test indices from the midterm).
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from _common import CV, data, feature_cols, make_model, metrics, np, pd, train, test, y_train, y_test
import joblib
from sklearn.model_selection import cross_val_predict, cross_val_score
from src.config import DATA_DIR, ROOT, XGB_BASELINE
from src.threshold import select_threshold, threshold_table


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midterm-indices", type=Path)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-l0-" + uuid4().hex[:8]
    model_id = "xgb-baseline-sensitive-s42train-l0"
    out = ROOT / "outputs" / "runs" / run_id
    result = ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    result.mkdir(parents=True, exist_ok=False)
    cols = feature_cols(lag=0, sensitive=True)
    model = make_model("XGBoost", cols, y_train, xgb_params=XGB_BASELINE)
    # Intentionally retain full-training-label weights for every CV fold (L0).
    print(f"L0 {run_id}: fitting original M0 and five-fold CV", flush=True)
    cv = cross_val_score(model, train[cols], y_train, cv=CV, scoring="roc_auc", n_jobs=1)
    oof = cross_val_predict(model, train[cols], y_train, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
    table = threshold_table(y_train, oof, loss=0, contact=0, rescue=0)
    threshold = select_threshold(table, "f1")
    policy = {"kind": "single", "threshold": 0.5, "max_list_rate": None}
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    policy_id = "l0-common-0.5-" + policy_hash[:12]
    model.fit(train[cols], y_train)
    joblib.dump(model, out / "model.joblib")
    # Reload the exact artifact whose hash is recorded.
    model = joblib.load(out / "model.joblib")
    observed = metrics(y_test, model.predict_proba(test[cols])[:, 1], threshold=0.5)
    pd.DataFrame({"customer_id": train.index, "oof_score": oof, "y": y_train.to_numpy()}).to_csv(out / "oof.csv", index=False)
    fold_assignment = np.empty(len(train), dtype=int)
    for fold, (_, valid) in enumerate(CV.split(train[cols], y_train)):
        fold_assignment[valid] = fold
    pd.DataFrame({"customer_id": train.index, "fold": fold_assignment}).to_csv(out / "folds.csv", index=False)
    historical_match = None
    if args.midterm_indices:
        historical = np.asarray(json.loads(args.midterm_indices.read_text()), dtype=int)
        historical_match = bool(np.array_equal(test.index.to_numpy(), historical + 1))
    checks = [
        ("test_rows", 6000, len(test), len(test) == 6000),
        ("test_defaults", 1327, int(y_test.sum()), int(y_test.sum()) == 1327),
        ("test_ID_equals_midterm_index_plus_1", True, historical_match, historical_match),
        ("test_AUC", .775, observed["AUC"], abs(observed["AUC"] - .775) <= .001),
        ("CV_mean_AUC", .781, float(cv.mean()), abs(cv.mean() - .781) <= .001),
        ("test_F1_at_0.5", .532, observed["F1-score"], round(observed["F1-score"], 3) == .532),
        ("test_Recall_at_0.5", .606, observed["Recall"], round(observed["Recall"], 3) == .606),
        ("test_list_at_0.5", 1696, observed["名單人數"], observed["名單人數"] == 1696),
        ("OOF_F1_threshold", .58, threshold, threshold == .58),
    ]
    rows = []
    for name, expected, actual, passed in checks:
        rows.append(dict(run_id=run_id, model_id=model_id, policy_id=policy_id,
                         metric=name, midterm=expected, reproduced=actual,
                         difference=None if actual is None else float(actual) - float(expected),
                         status="unverified" if passed is None else "pass" if passed else "mismatch"))
    frame = pd.DataFrame(rows)
    frame.to_csv(result / "reproduce.csv", index=False)
    params_hash = hashlib.sha256(json.dumps({"params": model.get_params(deep=False).__repr__(), "features": cols}, sort_keys=True).encode()).hexdigest()
    evidence = dict(run_id=run_id, model_id=model_id, model_hash=sha(out / "model.joblib"),
                    params_hash=params_hash, features=cols, level="L0", policy_id=policy_id,
                    policy_hash=policy_hash, policy=policy, oof_f1_threshold=threshold,
                    oof_sha256=sha(out / "oof.csv"), folds_sha256=sha(out / "folds.csv"))
    (out / "threshold.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    packages = ["pandas", "numpy", "scikit-learn", "xgboost", "xlrd", "joblib", "streamlit", "python-dotenv", "requests", "line-bot-sdk", "shap", "matplotlib", "pytest"]
    manifest = dict(evidence, status="complete", acceptance="pass" if all(r["status"] == "pass" for r in rows) else "not_passed",
                    created_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in packages},
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True)),
                    data_sha256=sha(DATA_DIR / "default_of_credit_card_clients.csv"),
                    class_weight_algorithm="Original make_model: negative/positive over all 24000 training labels before CV",
                    cv_auc_by_fold=cv.tolist(), historical_indices_sha256=sha(args.midterm_indices) if args.midterm_indices else None,
                    limitations=["Historical test indices not supplied" ] if historical_match is None else [],
                    usage="Historical reproduction only; must not inform L1 design")
    for folder in (out, result):
        (folder / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(frame[["metric", "midterm", "reproduced", "difference", "status"]].to_string(index=False))
    print(f"Results: results/{run_id}; acceptance={manifest['acceptance']}")
    return 0 if manifest["acceptance"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
