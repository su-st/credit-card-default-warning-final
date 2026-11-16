"""負責人：張佩芩｜分支：exp/fairness
RQ1 延伸：公平性分析加入教育程度與婚姻狀況分組（期中只有性別、年齡層）。
"""
from _common import *

train, evaluation = dev_holdout()
y_train, y_evaluation = train["default"], evaluation["default"]
# L1-H functional development only; do not select settings or report performance.
print("L1-H 開發資料，非正式效能；僅供功能確認")


EDU = {1: "研究所", 2: "大學", 3: "高中", 4: "其他"}
MAR = {1: "已婚", 2: "單身", 3: "其他"}
groups = {"教育程度": evaluation["EDUCATION"].map(EDU), "婚姻狀況": evaluation["MARRIAGE"].map(MAR)}
for use_s in [True, False]:
    cols = feature_cols(sensitive=use_s)
    m = make_model("XGBoost", cols, y_train, XGB_TUNED).fit(train[cols], y_train)
    pred = (m.predict_proba(evaluation[cols])[:, 1] >= 0.5).astype(int)
    for name, g in groups.items():
        print("使用敏感屬性" if use_s else "移除敏感屬性", name)
        print(group_errors(y_evaluation, pred, g).round(3))
# TODO 搭配 03 的重複交叉驗證，報告族群差距的平均與標準差
