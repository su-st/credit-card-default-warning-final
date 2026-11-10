"""負責人：余旻恩｜分支：exp/tuning
XGBoost 調參。只用訓練集五折 CV 挑參數，測試集最後才評估一次。
完成後把最佳參數更新到 src/config.py 的 XGB_TUNED。
初步測試：60 組隨機搜尋，CV AUC 0.786（期中 0.781）。
"""
from _common import *

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
