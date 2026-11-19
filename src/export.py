"""Export a registered L1-H model/policy/run as one verified bundle.

python -m src.export --run-id <running-L1-H-run>
No model is trained here and no threshold fallback is allowed. Official test
export belongs to the frozen final_eval entry point after G2b (C2).
"""
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from .data import age_group
from .registry import (GroupedModel, IntegrityError, Registry, Run, content_hash,
                       file_hash, identifier, read_json, validate_ids)
from .threshold import policy_list

MONTH = {1: "9 月", 2: "8 月", 3: "7 月", 4: "6 月", 5: "5 月", 6: "4 月"}
SEX = {1: "男", 2: "女"}
EDU = {1: "研究所", 2: "大學", 3: "高中", 4: "其他"}
MAR = {1: "已婚", 2: "單身", 3: "其他"}


def describe(col: str, v) -> str:
    """把特徵和數值轉成行員看得懂的一句話。"""
    if col.startswith("PAY_AMT"):
        return f"{MONTH[int(col[7:])]}繳款 {v:,.0f} 元"
    if col.startswith("PAY_") and col[4:].isdigit():
        if isinstance(v, str):
            return f"{MONTH[int(col[4:])]}繳款狀態（{v}）"
        m = MONTH[int(col[4:])]
        return f"{m}延遲繳款 {int(v)} 個月" if v > 0 else f"{m}繳款狀態（代碼 {int(v)}）"
    if col.startswith("BILL_AMT"):
        return f"{MONTH[int(col[8:])]}帳單 {v:,.0f} 元"
    return {"LIMIT_BAL": f"信用額度 {v:,.0f} 元", "AGE": f"年齡 {int(v)} 歲", "SEX": f"性別{SEX.get(v, v)}",
            "EDUCATION": f"教育程度{EDU.get(v, v)}", "MARRIAGE": f"婚姻狀況{MAR.get(v, v)}"}.get(col, f"{col} = {v}")


def top_factors(model, X: pd.DataFrame, k: int = 3) -> list:
    """用 XGBoost 內建的 TreeSHAP 貢獻值，找出每位客戶推高風險的前 k 個原始特徵。
    （SHAP 負責人可改用 shap 套件，但輸出格式維持不變。）"""
    from .frozen_methods import FrozenGroupedClassifier, delay_partition
    if isinstance(model, FrozenGroupedClassifier):
        labels = delay_partition(X, rules_freeze_id=model.rules_freeze_id)
        rows = [[""] * k for _ in range(len(X))]
        for label, submodel in model.models_.items():
            mask = labels == label
            if mask.any():
                for position, factors in zip(np.flatnonzero(mask), top_factors(submodel, X.loc[mask], k)):
                    rows[position] = factors
        return rows
    if isinstance(model, GroupedModel):
        delayed = (X[list(model.pay_columns)] > 0).any(axis=1).to_numpy()
        rows = [[""] * k for _ in range(len(X))]
        for mask, submodel in ((~delayed, model.no_delay_model), (delayed, model.delay_model)):
            if mask.any():
                for position, factors in zip(np.flatnonzero(mask), top_factors(submodel, X.loc[mask, list(model.columns)], k)):
                    rows[position] = factors
        return rows
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    Z = pre.transform(X)
    contrib = clf.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
    names = pre.get_feature_names_out()
    orig = [n.split("__", 1)[1] for n in names]
    orig = [o.rsplit("_", 1)[0] if n.startswith("cat__") else o for o, n in zip(orig, names)]
    c = pd.DataFrame(contrib, columns=names, index=X.index).T.groupby(orig).sum().T
    out = []
    for cid, row in c.iterrows():
        pos = row[row > 0].sort_values(ascending=False).head(k)
        out.append([describe(col, X.at[cid, col]) for col in pos.index] + [""] * (k - len(pos)))
    return out


def verify_predictions(registry, run_id, X):
    """Reload the hashed model and verify every exported customer's score."""
    bundle = registry.load_run(run_id, allow_running=True)
    predictions = bundle["predictions"]
    if predictions is None:
        raise IntegrityError("Predictions are missing")
    ids = validate_ids(X.index)
    if set(ids) != set(predictions["customer_id"]) or len(ids) != len(predictions):
        raise IntegrityError("Verification must include every exported customer")
    ordered = X.loc[predictions["customer_id"], bundle["model_record"]["columns"]]
    actual = bundle["model"].predict_proba(ordered)[:, 1]
    if not np.allclose(actual, predictions["risk_score"], atol=1e-9, rtol=0):
        raise IntegrityError("Reloaded model does not reproduce all exported scores")
    return len(predictions)


def export_run(run, frame, *, factors_fn=None):
    """Write once, verify all 4,800 development rows, then finish the run.

    The model, feature list and policy are read exclusively from this run.
    Accepts an already obtained L1-H frame so C2 can replace data access without
    changing the registry or the handoff contract.
    """
    # Reject formal/test export before inspecting data or loading any model.
    if run.meta["level"] != "L1-H":
        raise PermissionError("src.export only accepts L1-H; use frozen final_eval for L2")
    run._running()
    try:
        return _export_run(run, frame, factors_fn=factors_fn or top_factors)
    except Exception as exc:
        run.fail(f"{type(exc).__name__}: {exc}")
        raise


def _export_run(run, frame, *, factors_fn):
    bundle = run.registry.load_run(run.run_id, allow_running=True)
    record, policy, model = bundle["model_record"], bundle["threshold"], bundle["model"]
    ids = validate_ids(frame.index)
    if len(frame) != 4800 or content_hash(ids.tolist()) != record["prediction_ids_hash"]:
        raise IntegrityError("L1-H must contain the registered 4,800 holdout IDs in order")
    cols = record["columns"]
    X = frame[cols]
    prob = model.predict_proba(X)[:, 1]
    score = pd.Series(prob, index=frame.index)
    selected = policy_list(score, policy)
    out = pd.DataFrame({
        "customer_id": ids,
        "risk_score": prob,
        "in_list": selected.to_numpy(),
        "sex": frame["SEX"].map(SEX).values,
        "age": frame["AGE"].values,
        "age_group": age_group(frame["AGE"]).astype(str).values,
        "education": frame["EDUCATION"].map(EDU).values,
        "marriage": frame["MARRIAGE"].map(MAR).values,
        "limit_bal": frame["LIMIT_BAL"].values,
        "bill_amt_latest": frame["BILL_AMT1"].values,
        "actual_default": frame["default"].values,
        "data_split": "dev_holdout",
        "model_id": record["model_id"],
        "run_id": run.run_id,
        "policy_id": policy["policy_id"],
    })
    f = factors_fn(model, X)
    if len(f) != len(frame) or any(len(row) != 3 for row in f):
        raise IntegrityError("Factors must contain exactly three values per customer")
    for i in range(3):
        out[f"factor_{i+1}"] = [r[i] for r in f]
    out = out.sort_values(["risk_score", "customer_id"], ascending=[False, True])
    out.insert(2, "risk_rank", range(1, len(out) + 1))
    with (run.path / "predictions.csv").open("x", encoding="utf-8-sig", newline="") as f:
        out.to_csv(f, index=False, float_format="%.17g")
    run.update(predictions_sha256=file_hash(run.path / "predictions.csv"))
    count = verify_predictions(run.registry, run.run_id, X)
    run.update(predictions_verified=True, verified_customers=count)
    run.finish()
    return run.registry.load_run(run.run_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--level", default="L1-H")
    args = parser.parse_args(argv)
    if args.level != "L1-H":
        raise PermissionError("Before G2b, export accepts only L1-H")
    registry = Registry()
    meta = read_json(registry.runs / identifier(args.run_id) / "run.json")
    if meta["level"] != "L1-H" or meta["status"] != "running":
        raise PermissionError("A new running L1-H run is required")
    from .config import DATA_DIR
    data_file = DATA_DIR / "default_of_credit_card_clients.csv"
    if file_hash(data_file) != meta["data_sha256"]:
        raise IntegrityError("Current data file hash differs from the registered model")
    from .protocol import dev_holdout
    _, holdout = dev_holdout()
    bundle = export_run(Run(registry, meta), holdout)
    print(f"已驗證 {bundle['run']['verified_customers']} 位 L1-H 開發客戶；run_id={args.run_id}（非正式結果）")


if __name__ == "__main__":
    main()
