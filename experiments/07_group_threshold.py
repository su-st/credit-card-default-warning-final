"""負責人：張佩芩｜分支：exp/group-threshold
RQ1 延伸：公平性從「檢查」到「改善」。參考 Hardt et al.（2016）的 Equal Opportunity 做法，
各族群使用不同門檻，讓各族群違約者的 Recall 接近，再比較付出的代價。

流程：
1. 只用訓練集五折交叉驗證的預測（OOF）決定門檻，完全不碰測試集
2. 單一門檻：以 --run-id 指定同一個模型的 L1-H run，不讀全域門檻或默用備用值
3. 族群門檻：以單一門檻在 OOF 上的整體 Recall 為目標，每個族群各自找 Recall 最接近目標的門檻
4. 在 L1-H 開發保留集做功能確認（非正式結果），比較兩種做法：各族群 Recall、FPR、Equal Opportunity 差距、FPR 差距，以及整體名單人數與誤報

注意：模型本身可以不用敏感屬性，但族群門檻在決定名單時一定要知道客戶屬於哪個族群，
報告需要討論這樣做是否符合金管會指引與反歧視的要求。

TODO（張佩芩）
1. 執行一次，確認性別分組的結果
2. 把 GROUP 改成 "年齡層"，以及加入性別 × 年齡層，人數少的族群門檻可能不穩定，要註明
3. 畫圖：兩種做法各族群 Recall 的長條圖，存到 outputs/results/
4. 報告討論：差距縮小多少、整體 Recall 與名單人數的代價、是否值得，以及合規疑慮
5. 06_fairness_groups.py 目前門檻寫死 0.5，可改用 load_threshold() 讀自動選出的門檻
"""
import argparse
from _common import *
from sklearn.model_selection import cross_val_predict

from src.threshold import GRID
from src.registry import Registry, IntegrityError, content_hash, file_hash, json_value
from src.config import DATA_DIR

USE_SENSITIVE = False  # 模型本身不使用敏感屬性，只在決定門檻時用族群資訊
parser = argparse.ArgumentParser(description="L1-H 族群門檻功能確認，非正式結果")
parser.add_argument("group", nargs="?", default="性別", choices=["性別", "年齡層", "性別×年齡層"])
parser.add_argument("--run-id", required=True, help="同一個不含敏感屬性模型的已完成 L1-H run")
args = parser.parse_args()
GROUP = args.group
bundle = Registry().load_run(args.run_id)
if bundle["run"]["level"] != "L1-H":
    raise PermissionError("07 功能開發只能使用 L1-H run")
record = bundle["model_record"]
if file_hash(DATA_DIR / "default_of_credit_card_clients.csv") != record["data_sha256"]:
    raise IntegrityError("目前資料與來源模型的資料檔雜湊不符")
train, evaluation = dev_holdout()
y_train, y_evaluation = train["default"], evaluation["default"]
if (content_hash(train.index.tolist()) != record["training_ids_hash"]
        or content_hash(evaluation.index.tolist()) != record["prediction_ids_hash"]):
    raise IntegrityError("L1-H 客戶範圍與來源模型不符")
print(f"L1-H 開發資料，非正式結果；model_id={record['model_id']} run_id={args.run_id}")
SEX = {1: "男性", 2: "女性"}


def groups_of(d: pd.DataFrame) -> pd.Series:
    if GROUP == "性別":
        return d["SEX"].map(SEX)
    if GROUP == "年齡層":
        return age_group(d["AGE"]).astype(str)
    if GROUP == "性別×年齡層":
        return d["SEX"].map(SEX) + "×" + age_group(d["AGE"]).astype(str)
    raise ValueError(GROUP)


def recall(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return (pred & (y == 1)).sum() / (y == 1).sum()


def fpr(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return (pred & (y == 0)).sum() / (y == 0).sum()


cols = feature_cols(sensitive=USE_SENSITIVE)
if cols != record["columns"]:
    raise IntegrityError("07 特徵與來源模型不符")
oof = cross_val_predict(make_model("XGBoost", cols, y_train, XGB_TUNED), train[cols], y_train,
                        cv=CV, method="predict_proba")[:, 1]
model = make_model("XGBoost", cols, y_train, XGB_TUNED).fit(train[cols], y_train)
if content_hash({"params": json_value(model), "columns": cols}) != record["params_hash"]:
    raise IntegrityError("07 模型參數與來源門檻不符")
saved_oof = pd.read_csv(Registry().path(record["model_path"]).parent / "oof.csv", converters={"oof_score": float})
if not np.allclose(oof, saved_oof["oof_score"].to_numpy(), atol=1e-9, rtol=0):
    raise IntegrityError("07 OOF 與來源門檻證據不符")
p_evaluation = model.predict_proba(evaluation[cols])[:, 1]

t0 = float(bundle["threshold"]["threshold"])
source = f"L1-H run_id={args.run_id} policy_id={bundle['threshold']['policy_id']}"
g_train, g_evaluation = groups_of(train), groups_of(evaluation)
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

pred_single = p_evaluation >= t0
pred_group = p_evaluation >= g_evaluation.map(thr).values

rows, summary = [], []
for name, pred, tmap in [("單一門檻", pred_single, None), ("族群門檻", pred_group, thr)]:
    per = []
    for g in sorted(g_evaluation.unique()):
        idx = (g_evaluation == g).values
        per.append({"做法": name, "族群": g, "門檻": tmap[g] if tmap else t0,
                    "人數": int(idx.sum()), "違約人數": int(y_evaluation[idx].sum()),
                    "Recall": recall(y_evaluation[idx], pred[idx]), "FPR": fpr(y_evaluation[idx], pred[idx])})
    per = pd.DataFrame(per)
    rows.append(per)
    n_list = int(pred.sum())
    tp = int((pred & (y_evaluation.values == 1)).sum())
    summary.append({"做法": name,
                    "Equal Opportunity 差距": per["Recall"].max() - per["Recall"].min(),
                    "FPR 差距": per["FPR"].max() - per["FPR"].min(),
                    "整體 Recall": recall(y_evaluation, pred), "整體 Precision": tp / n_list if n_list else 0.0,
                    "名單人數": n_list, "誤報人數": n_list - tp})

detail = pd.concat(rows, ignore_index=True)
summary = pd.DataFrame(summary)
detail.round(3).to_csv(RESULTS / f"07_group_threshold_{GROUP}_detail.csv", index=False, encoding="utf-8-sig")
summary.round(4).to_csv(RESULTS / f"07_group_threshold_{GROUP}_summary.csv", index=False, encoding="utf-8-sig")
print("\nL1-H 開發資料各族群（非正式結果）：")
print(detail.round(3).to_string(index=False))
print("\nL1-H 開發資料整體（非正式結果）：")
print(summary.round(3).to_string(index=False))

# ===== 畫圖：兩種做法各族群 Recall =====
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 找得到中文字型就用；找不到就改用預設字型並提醒，程式不中斷
FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/msjh.ttc",    # WSL 借用 Windows 字型
    "C:/Windows/Fonts/msjh.ttc",        # 直接在 Windows 執行
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 部分 Linux
]
font_name = None
for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        try:
            font_manager.fontManager.addfont(fp)
            font_name = font_manager.FontProperties(fname=fp).get_name()
            break
        except Exception as e:
            print(f"提醒：字型 {fp} 載入失敗（{e}），改試下一個")
if font_name:
    plt.rcParams["font.family"] = font_name
else:
    print("提醒：找不到中文字型，改用預設字型，圖上的中文可能顯示成方塊（數據不受影響）")
plt.rcParams["axes.unicode_minus"] = False

pivot = detail.pivot(index="族群", columns="做法", values="Recall")[["單一門檻", "族群門檻"]]
ax = pivot.plot(kind="bar", figsize=(max(6, len(pivot) * 1.2), 4.5), rot=30)
ax.set_ylabel("Recall（違約者被抓到的比例）")
ax.set_title(f"各族群 Recall：單一門檻 vs 族群門檻（分組：{GROUP}）")
ax.set_ylim(0, 1)
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=8)
plt.tight_layout()
png = RESULTS / f"07_group_threshold_{GROUP}_recall.png"
plt.savefig(png, dpi=200)
print("圖已存到", png)
