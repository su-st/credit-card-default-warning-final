"""負責人：陳品婕｜分支：exp/shap
SHAP 分析：整體重要特徵（summary plot）與個別客戶的主要風險原因。
平台目前用 XGBoost 內建 TreeSHAP 算前 3 個原因（src/export.py 的 top_factors），
這裡做完整分析與圖表，若要改平台的原因顯示方式，修改 top_factors 但維持輸出欄位不變。
需要：pip install shap

目前完成：整體 SHAP summary plot（beeswarm）、平均 |SHAP| 長條圖、重要性數值表。
輸出（outputs/results/）：
    04_shap_summary_plot.png
    04_shap_importance_bar.png
    04_shap_importance.csv
"""
import matplotlib

matplotlib.use("Agg")  # 只存檔，不跳出視窗
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from _common import *

# 圖上的中文（標題）在 Windows 用微軟正黑體，找不到就退回預設字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 訓練模型：與組內共用設定相同（調參後的 XGBoost，全部特徵）
cols = feature_cols()
model = make_model("XGBoost", cols, y_train, XGB_TUNED).fit(train[cols], y_train)

# 2. 取出前處理後的測試集（Pipeline 的 pre 步驟會做 one-hot 與標準化）
pre, clf = model.named_steps["pre"], model.named_steps["clf"]
X_test = pre.transform(test[cols])
if hasattr(X_test, "toarray"):  # 若是稀疏矩陣就轉成一般陣列
    X_test = X_test.toarray()
names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]  # 去掉 cat__ / num__ 前綴

# 3. 計算 SHAP 值（針對「違約=1」的 log-odds）
try:
    sv = shap.TreeExplainer(clf)(X_test)
    values, base = sv.values, sv.base_values
except Exception as e:  # 備案：用 XGBoost 內建 TreeSHAP（平台 top_factors 也是這個做法）
    print("shap.TreeExplainer 失敗，改用 XGBoost 內建 pred_contribs：", e)
    contrib = clf.get_booster().predict(xgb.DMatrix(X_test, feature_names=names), pred_contribs=True)
    values, base = contrib[:, :-1], contrib[:, -1]
explanation = shap.Explanation(values=values, base_values=base, data=X_test, feature_names=names)

# 4. 圖一：整體 summary plot（每個點是一位客戶，顏色是該特徵的值高低）
plt.figure()
shap.summary_plot(values, X_test, feature_names=names, max_display=15, show=False)
plt.title("SHAP summary plot：各特徵對違約預測的影響")
plt.tight_layout()
plt.savefig(RESULTS / "04_shap_summary_plot.png", dpi=200, bbox_inches="tight")
plt.close()

# 5. 圖二：平均 |SHAP| 長條圖（報告好讀的版本）
plt.figure()
shap.plots.bar(explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(RESULTS / "04_shap_importance_bar.png", dpi=200, bbox_inches="tight")
plt.close()

# 6. 重要性數值表（報告可直接引用）
importance = (pd.Series(np.abs(values).mean(axis=0), index=names, name="mean_abs_shap")
              .sort_values(ascending=False))
importance.to_csv(RESULTS / "04_shap_importance.csv", encoding="utf-8-sig")

print("測試集客戶數：", len(test))
print("平均 |SHAP| 前 10 名：")
print(importance.head(10).round(4))
print("圖與表已存到", RESULTS)