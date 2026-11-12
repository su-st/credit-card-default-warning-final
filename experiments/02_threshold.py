"""負責人：蘇韻蓉｜分支：exp/threshold
門檻與成本分析：由程式依規則自動選出門檻（回應期中評語「門檻應在驗證集決定，再於測試集報告」
與「未量化預警名單成本」）。

流程：
1. 只用訓練集做五折交叉驗證，得到每位訓練客戶的驗證預測（OOF），完全不碰測試集
2. 在 OOF 上計算每個門檻的名單人數、誤報人數與總成本
3. 依 src/config.py 的 THRESHOLD_RULE 自動選出門檻，存到 outputs/threshold.json
4. 重跑 python -m src.export，平台與測試集報告就會自動使用這個門檻

成本假設（LOSS_PER_DEFAULT、COST_PER_CONTACT、RESCUE_RATE）在 src/config.py，需自行訂定並說明依據。
請在余旻恩更新 XGB_TUNED 之後重跑，才是正式結果。
"""
from _common import *
from sklearn.model_selection import cross_val_predict

from src.config import COST_PER_CONTACT, LOSS_PER_DEFAULT, MAX_LIST_RATE, RESCUE_RATE, THRESHOLD_RULE, TOP_N_RATE
from src.threshold import save_threshold, select_threshold, threshold_table

cols = feature_cols()
oof = cross_val_predict(make_model("XGBoost", cols, y_train, XGB_TUNED), train[cols], y_train,
                        cv=CV, method="predict_proba")[:, 1]

# 漏判損失：沒有指定時，用訓練集違約者最近一期帳單的平均（只用訓練集，不看測試集）
loss = LOSS_PER_DEFAULT if LOSS_PER_DEFAULT is not None else float(train.loc[y_train == 1, "BILL_AMT1"].clip(lower=0).mean())

table = threshold_table(y_train, oof, loss, COST_PER_CONTACT, RESCUE_RATE)
best = select_threshold(table, THRESHOLD_RULE, TOP_N_RATE, len(train), MAX_LIST_RATE)
row = table[table["門檻"] == best].iloc[0]
table.round(4).to_csv(RESULTS / "threshold_table.csv", index=False, encoding="utf-8-sig")

save_threshold({"threshold": best, "rule": THRESHOLD_RULE, "selected_on": "訓練集五折交叉驗證預測（OOF）",
                "assumptions": {"loss_per_default": round(loss), "cost_per_contact": COST_PER_CONTACT, "rescue_rate": RESCUE_RATE},
                "validation_metrics": {k: (round(float(v), 4)) for k, v in row.items()}})

print(f"規則：{THRESHOLD_RULE}　漏判損失假設：{loss:,.0f} 元　聯絡成本：{COST_PER_CONTACT} 元　挽回比例：{RESCUE_RATE}")
print(f"自動選出的門檻：{best}")
for k, v in row.items():  # 人數與成本用千分位整數，比例取到小數第三位
    print(f"  {k}：{v:,.0f}" if k in ("名單人數", "誤報人數", "漏判人數", "總成本") else f"  {k}：{v:.3f}")

# 敏感度分析：成本假設改變時，選出的門檻會怎麼變（報告用）
sens = []
for contact in [50, 500, 2000, 5000]:
    for rescue in [0.1, 0.2, 0.3]:
        t = threshold_table(y_train, oof, loss, contact, rescue)
        b = select_threshold(t, "cost"); r = t[t["門檻"] == b].iloc[0]
        sens.append({"聯絡成本": contact, "挽回比例": rescue, "最佳門檻": b, "名單人數": int(r["名單人數"]),
                     "Precision": round(r["Precision"], 3), "Recall": round(r["Recall"], 3)})
sens = pd.DataFrame(sens)
sens.to_csv(RESULTS / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
print("\n敏感度分析（總成本最低規則，不設人力上限）：")
print(sens.to_string(index=False))
# TODO 畫出門檻對總成本的曲線圖，存到 outputs/results/
# TODO 在報告說明三個成本假設的依據，並討論敏感度分析的結果
