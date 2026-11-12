"""負責人：余旻恩｜分支：exp/calibration
模型的風險分數是不是真正的違約機率？
XGBoost 用 scale_pos_weight 加重違約樣本，輸出的分數通常會比實際違約機率偏高。
平台目前稱為「風險分數」並以排名呈現，這支程式檢查偏差有多大，以及機率校準能修正多少。

流程：
1. 原始模型：訓練集訓練，測試集計算 Brier 分數與分組校準表（預測平均 vs 實際違約率）
2. 校準模型：CalibratedClassifierCV（isotonic，訓練集內五折），一樣在測試集評估
3. 比較兩者的 AUC（排序能力應幾乎不變）、Brier 分數、平均預測值與實際違約率

TODO（余旻恩）
1. 執行一次，看原始分數偏高多少
2. 試試 method="sigmoid"（Platt scaling），比較兩種校準方法
3. 畫校準曲線（x：預測機率，y：實際違約率，對角線代表完美校準），存到 outputs/results/
4. 跟組長討論平台要不要改用校準後的機率。注意：若改用，分數的尺度會改變，
   02_threshold.py 選出的門檻必須重跑，不能沿用
"""
from _common import *
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

cols = feature_cols()
raw = make_model("XGBoost", cols, y_train, XGB_TUNED).fit(train[cols], y_train)
cal = CalibratedClassifierCV(make_model("XGBoost", cols, y_train, XGB_TUNED), method="isotonic", cv=CV)
cal.fit(train[cols], y_train)

probs = {"原始分數": raw.predict_proba(test[cols])[:, 1], "校準後": cal.predict_proba(test[cols])[:, 1]}

summary, bins = [], []
edges = np.linspace(0, 1, 11)
for name, p in probs.items():
    summary.append({"模型": name, "AUC": roc_auc_score(y_test, p), "Brier 分數（越低越好）": brier_score_loss(y_test, p),
                    "平均預測值": p.mean(), "實際違約率": y_test.mean()})
    b = pd.cut(p, edges, include_lowest=True)
    t = pd.DataFrame({"分數區間": b, "p": p, "y": y_test.values}).groupby("分數區間", observed=True)
    t = t.agg(人數=("y", "size"), 平均預測值=("p", "mean"), 實際違約率=("y", "mean")).reset_index()
    t.insert(0, "模型", name)
    bins.append(t)

summary = pd.DataFrame(summary)
bins = pd.concat(bins, ignore_index=True)
bins["分數區間"] = bins["分數區間"].astype(str)
summary.round(4).to_csv(RESULTS / "08_calibration_summary.csv", index=False, encoding="utf-8-sig")
bins.round(4).to_csv(RESULTS / "08_calibration_bins.csv", index=False, encoding="utf-8-sig")
print(summary.round(4).to_string(index=False))
print("\n分組校準表（平均預測值和實際違約率越接近，代表分數越接近真正的機率）：")
print(bins.round(3).to_string(index=False))
