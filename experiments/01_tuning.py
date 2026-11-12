"""負責人：余旻恩｜分支：exp/tuning
XGBoost 調參。只用訓練集五折 CV 挑參數，測試集最後才評估一次。

更新規則：搜尋到的最佳參數，CV AUC 要比目前 src/config.py 的 XGB_TUNED 高出至少 MIN_GAIN（0.002），
才更新 XGB_TUNED，程式最後會自動比較並印出建議。
理由：五折 CV AUC 的標準差約 0.008，差距小於 0.002 很可能只是隨機波動，換參數不一定真的比較好，
卻會讓門檻、SHAP、公平性等後續結果全部需要重跑。

紀錄：
- 60 組隨機搜尋：CV AUC 0.786（期中 0.781），更新為目前的 XGB_TUNED
- 200 組隨機搜尋（n_estimators 加入 1000、1500）：提升未達 0.002，維持原本的 XGB_TUNED
"""
from _common import *

MIN_GAIN = 0.002  # 至少要提升這麼多，才更新 XGB_TUNED

cols = feature_cols()
rng = np.random.default_rng(RANDOM_STATE)
rows = []
for i in range(200):  # TODO 可加大搜尋次數或改用 Optuna
    p = dict(n_estimators=int(rng.choice([200, 300, 500, 800, 1000, 1500])),
             learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.1])),
             max_depth=int(rng.choice([3, 4, 5, 6])), min_child_weight=int(rng.choice([1, 3, 5, 10])),
             subsample=float(rng.choice([0.6, 0.8, 1.0])), colsample_bytree=float(rng.choice([0.5, 0.7, 0.9, 1.0])),
             gamma=float(rng.choice([0, 0.5, 1, 2])), reg_lambda=float(rng.choice([1, 3, 5, 10])))
    cv = cross_val_score(make_model("XGBoost", cols, y_train, p), train[cols], y_train, cv=CV, scoring="roc_auc")
    rows.append({**p, "cv_auc_mean": cv.mean(), "cv_auc_std": cv.std()})
res = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
res.to_csv(RESULTS / "tuning_search.csv", index=False, encoding="utf-8-sig")
print(res.head())

# 和目前的 XGB_TUNED 比較，決定要不要更新
cur = cross_val_score(make_model("XGBoost", cols, y_train, XGB_TUNED), train[cols], y_train, cv=CV, scoring="roc_auc")
best = res.iloc[0]["cv_auc_mean"]
print(f"\n目前 XGB_TUNED 的 CV AUC：{cur.mean():.4f}（標準差 {cur.std():.4f}）")
print(f"搜尋到的最佳 CV AUC：{best:.4f}，差距 {best - cur.mean():+.4f}")
if best - cur.mean() >= MIN_GAIN:
    print("建議更新 XGB_TUNED 為第一列的參數，並另外發 PR，通知組員重跑門檻與其他實驗")
else:
    print(f"差距未達 {MIN_GAIN}，維持目前的 XGB_TUNED")
