"""負責人：余旻恩｜分支：exp/tuning-log
F1 調參紀錄重現（資料層級 L1：24,000 筆訓練集，seed 42，五折 CV，完全不碰測試集）。

做的事：
1. 隨機搜尋 XGBoost 參數，把每一組參數與 CV AUC 存成 tuning_search.csv
2. 用 XGB_BASELINE 與 XGB_TUNED，分別在「含敏感屬性」與「不含敏感屬性」兩種特徵組算五折 CV AUC，存成 tuning_summary.csv
3. 依更新規則（提升至少 MIN_GAIN 才換參數）印出結論，結果照實記錄，不改程式湊數字

更新規則：搜尋到的最佳參數，CV AUC 要比目前 src/config.py 的 XGB_TUNED 高出至少 MIN_GAIN（0.002），
才更新 XGB_TUNED。理由：五折 CV AUC 的標準差約 0.008，差距小於 0.002 很可能只是隨機波動，
換參數不一定真的比較好，卻會讓門檻、SHAP、公平性等後續結果全部需要重跑。

用法：
    python experiments/01_tuning.py              # 正式執行，預設 200 組
    python experiments/01_tuning.py --n-iter 2   # 先試跑 2 組，估計時間
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import XGB_BASELINE, XGB_TUNED
from src.protocol import cv_evaluate, dev_data, make_model, oof_predict, prepare, start_run, tune_model

MIN_GAIN = 0.002        # 至少要提升這麼多，才更新 XGB_TUNED
SEED = 42
SEARCH_SENSITIVE = True  # 搜尋用含敏感屬性的特徵組，與原本 01 的搜尋相同，方便重現 0.786
# make_model 沒給的參數會用 XGB_TUNED 補上，所以 XGB_BASELINE 必須把其餘參數明確寫成 XGBoost 預設值，
# 才是期中真正的基準模型（期中的 XGBClassifier 只設了 n_estimators、learning_rate、max_depth）
BASELINE_FULL = dict(XGB_BASELINE, min_child_weight=1, subsample=1.0, colsample_bytree=1.0, gamma=0.0, reg_lambda=1.0)

class _Candidates:
    """候選值清單。寫成 .choice([...]) 是為了讓 tests/test_frozen_rules.py 能用 AST 讀出清單，並與凍結規則比對。"""

    @staticmethod
    def choice(values):
        return list(values)


cand = _Candidates()
# 候選值清單（與凍結規則 g2a-20261116 的 search_space 相同）；只沿用清單，抽樣交給 tune_model
p = dict(n_estimators=cand.choice([200, 300, 500, 800, 1000, 1500]),
         learning_rate=cand.choice([0.01, 0.02, 0.03, 0.05, 0.1]),
         max_depth=cand.choice([3, 4, 5, 6]), min_child_weight=cand.choice([1, 3, 5, 10]),
         subsample=cand.choice([0.6, 0.8, 1.0]), colsample_bytree=cand.choice([0.5, 0.7, 0.9, 1.0]),
         gamma=cand.choice([0, 0.5, 1, 2]), reg_lambda=cand.choice([1, 3, 5, 10]))
SEARCH_SPACE = p

parser = argparse.ArgumentParser(description="F1 調參紀錄重現（L1）")
parser.add_argument("--n-iter", type=int, default=200, help="隨機搜尋組數，預設 200")
args = parser.parse_args()
N_ITER = args.n_iter

df = dev_data()  # L1：訓練集 24,000 筆
y = df["default"]
print(f"資料層級 L1，{len(df):,} 筆，違約率 {y.mean():.1%}")

run = start_run("L1", f"F1 調參紀錄重現：隨機搜尋 {N_ITER} 組，並比較 XGB_BASELINE 與 XGB_TUNED",
                model_id="xgb-tuned-ns-s42train")
print(f"run_id = {run.run_id}")

# ---------- 1. 隨機搜尋 ----------
X_search, _ = prepare(df, lag=0, sensitive=SEARCH_SENSITIVE, engineered=False)
result = tune_model(X_search, y, SEARCH_SPACE, n_iter=N_ITER, folds=5, seed=SEED)
history = result["完整搜尋紀錄"].drop(columns=["params"]).sort_values("cv_auc_mean", ascending=False)
history = history.reset_index(drop=True)
assert len(history) == N_ITER == result["實際搜尋組數"], "搜尋紀錄組數與程式設定不一致"
run.save_csv("tuning_search", history)
best_auc = result["最佳 CV AUC"]
print(history.head())

# ---------- 2. XGB_BASELINE 與 XGB_TUNED，含與不含敏感屬性 ----------
rows = []
for sensitive in (True, False):
    X, cols = prepare(df, lag=0, sensitive=sensitive, engineered=False)
    for name, params in (("XGB_BASELINE", BASELINE_FULL), ("XGB_TUNED", XGB_TUNED)):
        cv = cv_evaluate(lambda p=params: make_model("XGBoost", params=dict(p)), X, y, folds=5, seed=SEED)
        rows.append({"特徵組": "含敏感屬性" if sensitive else "不含敏感屬性", "參數": name,
                     "特徵數": len(cols), "CV AUC 平均": cv["平均"].iloc[0], "CV AUC 標準差": cv["標準差"].iloc[0]})
summary = pd.DataFrame(rows)
run.save_csv("tuning_summary", summary)
print(summary.round(4).to_string(index=False))

# ---------- 3. 更新規則：搜尋最佳 vs. 目前 XGB_TUNED（同為含敏感屬性的特徵組）----------
cur = summary[(summary["特徵組"] == "含敏感屬性") & (summary["參數"] == "XGB_TUNED")]["CV AUC 平均"].iloc[0]
gain = best_auc - cur
print(f"\n目前 XGB_TUNED 的 CV AUC：{cur:.4f}（期中紀錄 0.786）")
print(f"搜尋到的最佳 CV AUC：{best_auc:.4f}，差距 {gain:+.4f}")
if gain >= MIN_GAIN:
    print("建議更新 XGB_TUNED 為搜尋紀錄第一列的參數，並另外發 PR，通知組員重跑門檻與其他實驗")
else:
    print(f"差距未達 {MIN_GAIN}，維持目前的 XGB_TUNED")

# ---------- 4. 保存 M1-NS（XGB_TUNED、不含敏感屬性）的訓練紀錄，run 需要綁定一個模型才能完成 ----------
X_ns, _ = prepare(df, lag=0, sensitive=False, engineered=False)
oof = oof_predict(lambda: make_model("XGBoost", params=dict(XGB_TUNED)), X_ns, y, folds=5, seed=SEED)
model = make_model("XGBoost", params=dict(XGB_TUNED)).fit(X_ns, y)
run.attach_model(model, X_ns, y, oof)
run.finish()
print(f"完成。結果在 results/{run.run_id}/")