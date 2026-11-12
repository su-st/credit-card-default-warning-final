"""負責人：陳品婕｜分支：exp/missed-analysis
模型最大的弱點：期中發現漏判的違約者有 84.5% 近六個月沒有延遲紀錄。
這支程式找出這些人和其他客戶有什麼不同，作為新增特徵或在報告說明限制的依據。

為了不讓測試集影響模型設計，分析用的是訓練集五折交叉驗證的預測（OOF），
門檻讀 outputs/threshold.json（02_threshold.py 選出的門檻）。

分析一：違約者中，成功辨識與漏判的人各項特徵比較
分析二：近六個月都沒有延遲的客戶中，違約與未違約的人各項特徵比較（漏判者大多屬於這群）
        如果兩者在某些特徵上差很多，那些特徵就是辨識這群人的線索
分析三：比較改善方法，一樣只用訓練集五折交叉驗證
  A 基準：目前的特徵
  B 加入繳款狀態次數：代碼 0、−1、−2 各出現幾個月，以及代碼 0 最長連續幾個月
    （UCI 說明文件沒有定義 0 與 −2，常見解讀是 0 為只繳最低應繳、−2 為無消費，報告引用時要註明是推測）
  C 再加入金額變化：最近繳款金額相對過去的變化、繳款金額小於上期帳單的月數、繳款金額的波動程度
  D 分群建模：有延遲紀錄與沒有延遲紀錄的客戶分開訓練（使用 C 的特徵）
    注意：兩個模型的分數尺度不同，直接合併排序會讓名單偏向某一群，結果要謹慎解讀
  E 分群門檻：用基準模型，但沒有延遲紀錄的客戶改用較低的門檻
  A 到 D 在「名單人數相同」的條件下比較，才公平（新特徵會改變分數的尺度，不能用同一個門檻比）
  主要看「近六個月無延遲的違約者被辨識出的比例」，也要看整體 Recall 與 AUC
  E 會讓名單變大，所以另外列出多抓到幾位違約者、多了幾位誤報

TODO（陳品婕）
1. 執行一次，看分析二哪些特徵差距最大，以及分析三哪個方法對無延遲違約者改善最多
2. 用 KMeans 把漏判者分成 2 到 4 群，描述每群的特徵（例如額度使用率高、還款比例低）
3. 方法 D 可改成「兩群各自依人數比例分配名單名額」，或先校準兩個模型的分數再合併，看結果是否不同
4. 根據分析二與分群結果，可再設計新特徵加進 extra_features()，看是否進一步改善
5. 改善明顯的特徵要加進 src/features.py 時，先跟組長討論（會影響平台與其他人的實驗）
6. 搭配 05_pay_categorical.py 的結果，在報告說明哪些做法有改善、哪些沒有，以及剩下的漏判為什麼資料抓不到
"""
from _common import *
from sklearn.model_selection import cross_val_predict

from src.threshold import load_threshold

FEATS = {"N_DELAY": "近六個月延遲次數", "MAX_DELAY": "最長延遲月數", "UTIL_1": "9 月額度使用率",
         "UTIL_TREND": "使用率變化（9 月 − 4 月）", "PAYRATIO_1": "9 月還款比例", "LIMIT_BAL": "信用額度",
         "BILL_AMT1": "9 月帳單金額", "PAY_AMT1": "9 月繳款金額", "AGE": "年齡"}

cols = feature_cols()
oof = cross_val_predict(make_model("XGBoost", cols, y_train, XGB_TUNED), train[cols], y_train,
                        cv=CV, method="predict_proba")[:, 1]
t0, source = load_threshold()
d = add_features(train).assign(prob=oof, pred=(oof >= t0).astype(int))


def profile(df: pd.DataFrame) -> pd.Series:
    s = df[list(FEATS)].median().rename(FEATS)
    s["人數"] = len(df)
    s["近六個月無延遲比例"] = (df["N_DELAY"] == 0).mean()
    s["模型分數平均"] = df["prob"].mean()
    return s


bad = d[d["default"] == 1]
a1 = pd.DataFrame({"成功辨識的違約者": profile(bad[bad["pred"] == 1]), "漏判的違約者": profile(bad[bad["pred"] == 0])})

clean = d[d["N_DELAY"] == 0]
a2 = pd.DataFrame({"無延遲且違約": profile(clean[clean["default"] == 1]),
                   "無延遲且未違約": profile(clean[clean["default"] == 0])})
a2["差異比例"] = (a2["無延遲且違約"] - a2["無延遲且未違約"]) / a2["無延遲且未違約"].abs().replace(0, np.nan)
a2.loc[["人數", "近六個月無延遲比例"], "差異比例"] = np.nan  # 這兩列比較沒有意義

a1.round(3).to_csv(RESULTS / "09_missed_vs_caught.csv", encoding="utf-8-sig")
a2.round(3).to_csv(RESULTS / "09_no_delay_default_vs_not.csv", encoding="utf-8-sig")
print(f"門檻：{t0}（{source}），訓練集 OOF 預測")
print(f"違約者 {len(bad):,} 人，漏判 {(bad['pred'] == 0).sum():,} 人，"
      f"其中近六個月無延遲 {((bad['pred'] == 0) & (bad['N_DELAY'] == 0)).sum():,} 人")
print("\n分析一：成功辨識 vs 漏判（中位數）")
print(a1.round(3).to_string())
print("\n分析二：近六個月無延遲的客戶，違約 vs 未違約（中位數，差異比例依絕對值大小看）")
print(a2.round(3).to_string())


# ================= 分析三：改善方法比較 =================
from src.data import PAY_COLS  # noqa: E402

PAY_AMT = [f"PAY_AMT{m}" for m in range(1, 7)]


def max_run(row) -> int:
    best = cur = 0
    for v in row:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def extra_features(d: pd.DataFrame, money: bool) -> pd.DataFrame:
    """只用 4 至 9 月的資料，不使用違約標籤，不會造成資料洩漏。"""
    d = d.copy()
    d["N_CODE0"] = (d[PAY_COLS] == 0).sum(axis=1)
    d["N_CODE_M1"] = (d[PAY_COLS] == -1).sum(axis=1)
    d["N_CODE_M2"] = (d[PAY_COLS] == -2).sum(axis=1)
    d["RUN_CODE0"] = d[PAY_COLS].apply(max_run, axis=1)
    if money:
        past = d[PAY_AMT[1:]].mean(axis=1)
        d["PAYAMT_CHANGE"] = (d["PAY_AMT1"] - past) / (past + 1)                       # 最近繳款相對過去的變化
        d["N_UNDERPAY"] = sum((d[f"PAY_AMT{m}"] < d[f"BILL_AMT{m+1}"]).astype(int) for m in range(1, 6))
        d["PAYAMT_CV"] = d[PAY_AMT].std(axis=1) / (d[PAY_AMT].mean(axis=1) + 1)        # 繳款金額波動
    return d


def oof_scores(d: pd.DataFrame, cols: list) -> np.ndarray:
    m = make_model("XGBoost", cols, d["default"], XGB_TUNED)
    return cross_val_predict(m, d[cols], d["default"], cv=CV, method="predict_proba")[:, 1]


def evaluate(name: str, score: np.ndarray, n_list: int) -> dict:
    """名單人數固定為 n_list：取分數最高的 n_list 人。"""
    flag = np.zeros(len(score), dtype=bool)
    flag[np.argsort(-score)[:n_list]] = True
    y = train["default"].values
    nodelay = (base["N_DELAY"] == 0).values
    return {"方法": name, "AUC": roc_auc_score(y, score), "名單人數": int(flag.sum()),
            "整體 Recall": (flag & (y == 1)).sum() / (y == 1).sum(),
            "無延遲違約者 Recall": (flag & (y == 1) & nodelay).sum() / ((y == 1) & nodelay).sum(),
            "誤報人數": int((flag & (y == 0)).sum())}


from sklearn.metrics import roc_auc_score  # noqa: E402

base = add_features(train)
n_list = int((oof >= t0).sum())  # 和基準模型在目前門檻下的名單人數相同
dB, dC = extra_features(base, money=False), extra_features(base, money=True)
colsB = cols + ["N_CODE0", "N_CODE_M1", "N_CODE_M2", "RUN_CODE0"]
colsC = colsB + ["PAYAMT_CHANGE", "N_UNDERPAY", "PAYAMT_CV"]

scoreB, scoreC = oof_scores(dB, colsB), oof_scores(dC, colsC)
# D 分群建模：兩群各自做五折交叉驗證
scoreD = np.zeros(len(dC))
for has_delay in [False, True]:
    idx = ((dC["N_DELAY"] > 0) == has_delay).values
    scoreD[idx] = oof_scores(dC[idx], colsC)

res = pd.DataFrame([evaluate("A 基準", oof, n_list), evaluate("B 加入繳款狀態次數", scoreB, n_list),
                    evaluate("C 再加入金額變化", scoreC, n_list), evaluate("D 分群建模", scoreD, n_list)])

# E 分群門檻：無延遲的客戶門檻往下調，看多抓到幾位違約者、多了幾位誤報
y = train["default"].values
nodelay = (base["N_DELAY"] == 0).values
rows_e = []
for t1 in [t0, t0 - 0.05, t0 - 0.10, t0 - 0.15]:
    flag = np.where(nodelay, oof >= t1, oof >= t0)
    rows_e.append({"無延遲客戶的門檻": round(t1, 2), "名單人數": int(flag.sum()),
                   "整體 Recall": (flag & (y == 1)).sum() / (y == 1).sum(),
                   "無延遲違約者 Recall": (flag & (y == 1) & nodelay).sum() / ((y == 1) & nodelay).sum(),
                   "誤報人數": int((flag & (y == 0)).sum())})
res_e = pd.DataFrame(rows_e)

res.round(4).to_csv(RESULTS / "09_methods_same_list_size.csv", index=False, encoding="utf-8-sig")
res_e.round(4).to_csv(RESULTS / "09_group_threshold_nodelay.csv", index=False, encoding="utf-8-sig")
print(f"\n分析三：改善方法比較（名單人數固定為 {n_list:,} 人，訓練集五折交叉驗證）")
print(res.round(3).to_string(index=False))
print("\n方法 E：無延遲客戶改用較低門檻（其他客戶維持原門檻）")
print(res_e.round(3).to_string(index=False))
