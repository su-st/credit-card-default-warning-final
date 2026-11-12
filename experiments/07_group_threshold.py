"""負責人：張佩芩｜分支：exp/group-threshold
RQ1 延伸：公平性從「檢查」到「改善」。參考 Hardt et al.（2016）的 Equal Opportunity 做法，
各族群使用不同門檻，讓各族群違約者的 Recall 接近，再比較付出的代價。

流程：
1. 只用訓練集五折交叉驗證的預測（OOF）決定門檻，完全不碰測試集
2. 單一門檻：讀 outputs/threshold.json（02_threshold.py 選出的門檻，沒有就用 config 備用值）
3. 族群門檻：以單一門檻在 OOF 上的整體 Recall 為目標，每個族群各自找 Recall 最接近目標的門檻
4. 在測試集比較兩種做法：各族群 Recall、FPR、Equal Opportunity 差距、FPR 差距，以及整體名單人數與誤報

注意：模型本身可以不用敏感屬性，但族群門檻在決定名單時一定要知道客戶屬於哪個族群，
報告需要討論這樣做是否符合金管會指引與反歧視的要求。

TODO（張佩芩）
1. 執行一次，確認性別分組的結果
2. 把 GROUP 改成 "年齡層"，以及加入性別 × 年齡層，人數少的族群門檻可能不穩定，要註明
3. 畫圖：兩種做法各族群 Recall 的長條圖，存到 outputs/results/
4. 報告討論：差距縮小多少、整體 Recall 與名單人數的代價、是否值得，以及合規疑慮
5. 06_fairness_groups.py 目前門檻寫死 0.5，可改用 load_threshold() 讀自動選出的門檻
"""
from _common import *
from sklearn.model_selection import cross_val_predict

from src.threshold import GRID, load_threshold

USE_SENSITIVE = False  # 模型本身不使用敏感屬性，只在決定門檻時用族群資訊
GROUP = "性別"          # 可改成 "年齡層"
SEX = {1: "男性", 2: "女性"}


def groups_of(d: pd.DataFrame) -> pd.Series:
    if GROUP == "性別":
        return d["SEX"].map(SEX)
    if GROUP == "年齡層":
        return age_group(d["AGE"]).astype(str)
    raise ValueError(GROUP)


def recall(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return (pred & (y == 1)).sum() / (y == 1).sum()


def fpr(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return (pred & (y == 0)).sum() / (y == 0).sum()


cols = feature_cols(sensitive=USE_SENSITIVE)
oof = cross_val_predict(make_model("XGBoost", cols, y_train, XGB_TUNED), train[cols], y_train,
                        cv=CV, method="predict_proba")[:, 1]
model = make_model("XGBoost", cols, y_train, XGB_TUNED).fit(train[cols], y_train)
p_test = model.predict_proba(test[cols])[:, 1]

t0, source = load_threshold()
g_train, g_test = groups_of(train), groups_of(test)
target = recall(y_train, oof >= t0)

# 每個族群在 OOF 上找 Recall 最接近目標的門檻（同樣接近時取較高的門檻，誤報較少）
thr = {}
for g in sorted(g_train.unique()):
    idx = (g_train == g).values
    gaps = [abs(recall(y_train[idx], oof[idx] >= t) - target) for t in GRID]
    best = min(range(len(GRID)), key=lambda i: (round(gaps[i], 4), -GRID[i]))
    thr[g] = float(GRID[best])

print(f"分組：{GROUP}　模型使用敏感屬性：{USE_SENSITIVE}")
print(f"單一門檻：{t0}（{source}）　OOF 整體 Recall 目標：{target:.3f}")
print("族群門檻：", thr)

pred_single = p_test >= t0
pred_group = p_test >= g_test.map(thr).values

rows, summary = [], []
for name, pred, tmap in [("單一門檻", pred_single, None), ("族群門檻", pred_group, thr)]:
    per = []
    for g in sorted(g_test.unique()):
        idx = (g_test == g).values
        per.append({"做法": name, "族群": g, "門檻": tmap[g] if tmap else t0,
                    "人數": int(idx.sum()), "違約人數": int(y_test[idx].sum()),
                    "Recall": recall(y_test[idx], pred[idx]), "FPR": fpr(y_test[idx], pred[idx])})
    per = pd.DataFrame(per)
    rows.append(per)
    n_list = int(pred.sum())
    tp = int((pred & (y_test.values == 1)).sum())
    summary.append({"做法": name,
                    "Equal Opportunity 差距": per["Recall"].max() - per["Recall"].min(),
                    "FPR 差距": per["FPR"].max() - per["FPR"].min(),
                    "整體 Recall": recall(y_test, pred), "整體 Precision": tp / n_list if n_list else 0.0,
                    "名單人數": n_list, "誤報人數": n_list - tp})

detail = pd.concat(rows, ignore_index=True)
summary = pd.DataFrame(summary)
detail.round(3).to_csv(RESULTS / f"07_group_threshold_{GROUP}_detail.csv", index=False, encoding="utf-8-sig")
summary.round(4).to_csv(RESULTS / f"07_group_threshold_{GROUP}_summary.csv", index=False, encoding="utf-8-sig")
print("\n測試集各族群：")
print(detail.round(3).to_string(index=False))
print("\n測試集整體：")
print(summary.round(3).to_string(index=False))
