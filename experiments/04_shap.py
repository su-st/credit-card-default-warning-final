"""負責人：陳品婕｜分支：exp/shap
SHAP 分析：整體重要特徵（summary plot）與個別客戶的主要風險原因。
平台目前用 XGBoost 內建 TreeSHAP 算前 3 個原因（src/export.py 的 top_factors），
這裡做完整分析與圖表，若要改平台的原因顯示方式，修改 top_factors 但維持輸出欄位不變。
需要：pip install shap
"""
from _common import *

# TODO
# import shap
# model = make_model("XGBoost", feature_cols(), y_train, XGB_TUNED).fit(train[feature_cols()], y_train)
# 用 shap.TreeExplainer 計算測試集 SHAP 值，畫 summary plot 存到 outputs/results/
