"""負責人：余旻恩｜分支：exp/repeated-cv
重複交叉驗證：確認 RQ1 至 RQ3 的結果在不同切分下是否穩定，特別是人數少的交叉族群。
TODO
1. 用 RepeatedStratifiedKFold（例如 5 折 × 10 次）重做 RQ1 的族群漏判率、誤判率
2. 重做 RQ2（邏輯迴歸加衍生特徵 vs. XGBoost）與 RQ3（提早 1 至 5 個月）的 AUC
3. 報告平均與標準差
"""
from _common import *
from sklearn.model_selection import RepeatedStratifiedKFold

RCV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
