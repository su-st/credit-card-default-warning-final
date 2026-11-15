"""負責人：陳品婕｜分支：exp/shap
SHAP 分析：整體重要特徵（summary plot）與個別客戶的主要風險原因。
平台目前用 XGBoost 內建 TreeSHAP 算前 3 個原因（src/export.py 的 top_factors），
這裡做完整分析與圖表，若要改平台的原因顯示方式，修改 top_factors 但維持輸出欄位不變。
需要：pip install shap

目前完成：整體 SHAP summary plot（beeswarm）、平均 |SHAP| 長條圖、重要性數值表。
2026-11 更新（依組長 review 建議）：EDUCATION、MARRIAGE 的 One-Hot 欄位 SHAP 值加總回原本的特徵，
與期中報告做法一致；圖上的顏色改用原始特徵值（不是標準化後的值）。
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
from src.config import CAT_COLS

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

# 4. 把 One-Hot 欄位（EDUCATION_1、EDUCATION_2…）的 SHAP 值加總回原本的特徵
#    SHAP 值有可加性，同一個特徵拆成多個欄位的貢獻相加，就是該特徵的總貢獻
agg_values = []
for c in cols:
    idx = [i for i, n in enumerate(names) if n == c or (c in CAT_COLS and n.startswith(c + "_"))]
    assert idx, f"找不到 {c} 對應的欄位"
    agg_values.append(values[:, idx].sum(axis=1))
agg_values = np.column_stack(agg_values)
assert np.allclose(agg_values.sum(axis=1), values.sum(axis=1), atol=1e-3)  # 加總前後總貢獻應相同

X_show = test[cols]  # 圖上的顏色與數值用原始特徵值
explanation = shap.Explanation(values=agg_values, base_values=base, data=X_show.values, feature_names=cols)

# 5. 圖一：整體 summary plot（每個點是一位客戶，顏色是該特徵的值高低）
plt.figure()
shap.summary_plot(agg_values, X_show, max_display=15, show=False)
plt.title("SHAP summary plot：各特徵對違約預測的影響")
plt.tight_layout()
plt.savefig(RESULTS / "04_shap_summary_plot.png", dpi=200, bbox_inches="tight")
plt.close()

# 6. 圖二：平均 |SHAP| 長條圖（報告好讀的版本）
plt.figure()
shap.plots.bar(explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(RESULTS / "04_shap_importance_bar.png", dpi=200, bbox_inches="tight")
plt.close()

# 7. 重要性數值表（報告可直接引用）
importance = (pd.Series(np.abs(agg_values).mean(axis=0), index=cols, name="mean_abs_shap")
              .sort_values(ascending=False))
importance.to_csv(RESULTS / "04_shap_importance.csv", encoding="utf-8-sig")

print("測試集客戶數：", len(test))
print("平均 |SHAP| 前 10 名（類別欄位已合併）：")
print(importance.head(10).round(4))
for c in CAT_COLS:
    if c in importance.index:
        print(f"{c} 合併後平均 |SHAP|：{importance[c]:.4f}（排名第 {list(importance.index).index(c) + 1}）")
print("圖與表已存到", RESULTS)
