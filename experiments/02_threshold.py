"""負責人：蘇韻蓉｜分支：exp/threshold
門檻與成本分析：比較不同門檻下的漏判、誤報與名單人數，並加入成本假設選出門檻。
請在余旻恩完成調參、更新 XGB_TUNED 之後再跑正式結果。
注意：門檻要用訓練集的交叉驗證預測來決定，不能看測試集挑。
完成後把選定的門檻更新到 src/config.py 的 LIST_THRESHOLD。
"""
from _common import *
from sklearn.model_selection import cross_val_predict

cols = feature_cols()
oof = cross_val_predict(make_model("XGBoost", cols, y_train, XGB_TUNED), train[cols], y_train,
                        cv=CV, method="predict_proba")[:, 1]
rows = []
for t in np.arange(0.2, 0.81, 0.05):
    m = metrics(y_train, oof, t)
    rows.append({"門檻": round(t, 2), **{k: round(v, 3) for k, v in m.items()}})
# TODO 加入成本假設（例如漏判一位的損失 vs. 聯絡一位的成本），找出成本最低的門檻
print(pd.DataFrame(rows))
