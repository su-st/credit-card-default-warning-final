"""產生研究組交給平台的交接檔（outputs/predictions.csv）與模型檔。

執行：python -m src.export
欄位定義見 docs/handoff_schema.md。
TODO（余旻恩）：另外輸出移除敏感屬性的版本 outputs/predictions_no_sensitive.csv，欄位相同，給公平性監控頁比較用。研究組更新模型或門檻後重跑這支程式，平台就會讀到新結果。
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import LIST_THRESHOLD, OUT_DIR, PLATFORM_USE_SENSITIVE, XGB_TUNED
from .data import age_group, feature_cols, load_data, split
from .evaluate import metrics
from .models import make_model

MONTH = {1: "9 月", 2: "8 月", 3: "7 月", 4: "6 月", 5: "5 月", 6: "4 月"}
SEX = {1: "男", 2: "女"}
EDU = {1: "研究所", 2: "大學", 3: "高中", 4: "其他"}
MAR = {1: "已婚", 2: "單身", 3: "其他"}


def describe(col: str, v) -> str:
    """把特徵和數值轉成行員看得懂的一句話。"""
    if col.startswith("PAY_AMT"):
        return f"{MONTH[int(col[7:])]}繳款 {v:,.0f} 元"
    if col.startswith("PAY_"):
        m = MONTH[int(col[4:])]
        return f"{m}延遲繳款 {int(v)} 個月" if v > 0 else f"{m}繳款狀態（代碼 {int(v)}）"
    if col.startswith("BILL_AMT"):
        return f"{MONTH[int(col[8:])]}帳單 {v:,.0f} 元"
    return {"LIMIT_BAL": f"信用額度 {v:,.0f} 元", "AGE": f"年齡 {int(v)} 歲", "SEX": f"性別{SEX.get(v, v)}",
            "EDUCATION": f"教育程度{EDU.get(v, v)}", "MARRIAGE": f"婚姻狀況{MAR.get(v, v)}"}.get(col, f"{col} = {v}")


def top_factors(model, X: pd.DataFrame, k: int = 3) -> list:
    """用 XGBoost 內建的 TreeSHAP 貢獻值，找出每位客戶推高風險的前 k 個原始特徵。
    （SHAP 負責人可改用 shap 套件，但輸出格式維持不變。）"""
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


def main():
    df = load_data()
    train, test = split(df)
    cols = feature_cols(sensitive=PLATFORM_USE_SENSITIVE)
    model = make_model("XGBoost", cols, train["default"], XGB_TUNED).fit(train[cols], train["default"])
    prob = model.predict_proba(test[cols])[:, 1]

    out = pd.DataFrame({
        "customer_id": test.index,
        "risk_score": np.round(prob, 4),
        "in_list": (prob >= LIST_THRESHOLD).astype(int),
        "sex": test["SEX"].map(SEX).values,
        "age": test["AGE"].values,
        "age_group": age_group(test["AGE"]).astype(str).values,
        "education": test["EDUCATION"].map(EDU).values,
        "marriage": test["MARRIAGE"].map(MAR).values,
        "limit_bal": test["LIMIT_BAL"].values,
        "bill_amt_latest": test["BILL_AMT1"].values,
        "actual_default": test["default"].values,
        "data_split": "test",
    })
    f = top_factors(model, test[cols])
    for i in range(3):
        out[f"factor_{i+1}"] = [r[i] for r in f]
    out = out.sort_values("risk_score", ascending=False)
    out.insert(2, "risk_rank", range(1, len(out) + 1))

    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"model": model, "cols": cols}, OUT_DIR / "model.joblib")
    m = {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in metrics(test["default"], prob, LIST_THRESHOLD).items()}
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "threshold": LIST_THRESHOLD,
            "use_sensitive": PLATFORM_USE_SENSITIVE, "xgb_params": XGB_TUNED, "test_metrics": m}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已輸出 {len(out):,} 位客戶到 outputs/predictions.csv")
    print("測試集表現：", m)


if __name__ == "__main__":
    main()
