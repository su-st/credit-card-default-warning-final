"""02 門檻與成本：C1 只遷移輸入／輸出，不更改原有平均損失分析公式。

使用 C2 產生的 L1 已登錄模型與 OOF。設定從規則凍結紀錄的
legacy_threshold_analysis 讀取。本入口是舊算法對照，不是 C3 的正式 D2
逐客戶帳單成本選擇；C3 完成前不產生或宣稱正式成本政策。

python experiments/02_threshold.py --model-record outputs/models/.../model.json --rules-freeze-id <id>
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.registry import IntegrityError, Registry, content_hash, file_hash, read_json
from src.threshold import save_threshold, select_threshold, threshold_table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-record", required=True)
    parser.add_argument("--rules-freeze-id", required=True)
    args = parser.parse_args(argv)
    registry = Registry()
    rules = registry.load_rules(args.rules_freeze_id)
    settings = rules["legacy_threshold_analysis"]
    record = read_json(registry.path(args.model_record))
    registry.load_model(record)
    if record["level"] != "L1":
        raise IntegrityError("02 requires this model's L1 OOF; L1-H cannot select research settings")
    from src.config import DATA_DIR
    if file_hash(DATA_DIR / "default_of_credit_card_clients.csv") != record["data_sha256"]:
        raise IntegrityError("Current data hash differs from the registered model")
    # C2 owns data access. Refuse the old _common import-time test-set access.
    from src.protocol import dev_data
    train = dev_data()
    if content_hash(train.index.tolist()) != record["training_ids_hash"]:
        raise IntegrityError("Development training IDs differ from the registered model")
    saved_oof = pd.read_csv(registry.path(record["model_path"]).parent / "oof.csv", converters={"oof_score": float})
    y_train = train["default"]
    if not np.array_equal(saved_oof["y"], y_train.to_numpy()):
        raise IntegrityError("OOF labels differ from current training data")
    oof = saved_oof["oof_score"].to_numpy()
    LOSS_PER_DEFAULT = settings["loss_per_default"]
    COST_PER_CONTACT = settings["cost_per_contact"]
    RESCUE_RATE = settings["rescue_rate"]
    THRESHOLD_RULE = settings["rule"]
    TOP_N_RATE = settings["top_n_rate"]
    MAX_LIST_RATE = settings["max_list_rate"]
    run = registry.start_run("L1", "02 legacy-average-loss comparison", model_id=record["model_id"], rules_freeze_id=args.rules_freeze_id)
    run.bind_model(record)
    try:
        analyze(registry, run, record, args, train, y_train, oof, LOSS_PER_DEFAULT,
                COST_PER_CONTACT, RESCUE_RATE, THRESHOLD_RULE, TOP_N_RATE, MAX_LIST_RATE,
                settings["sensitivity_contact_costs"], settings["sensitivity_rescue_rates"])
    except Exception as exc:
        run.fail(f"{type(exc).__name__}: {exc}")
        raise


def analyze(registry, run, record, args, train, y_train, oof, LOSS_PER_DEFAULT,
            COST_PER_CONTACT, RESCUE_RATE, THRESHOLD_RULE, TOP_N_RATE, MAX_LIST_RATE,
            sensitivity_contact_costs, sensitivity_rescue_rates):
    # 漏判損失：沒有指定時，用訓練集違約者最近一期帳單的平均（只用訓練集，不看測試集）
    loss = LOSS_PER_DEFAULT if LOSS_PER_DEFAULT is not None else float(train.loc[y_train == 1, "BILL_AMT1"].clip(lower=0).mean())

    table = threshold_table(y_train, oof, loss, COST_PER_CONTACT, RESCUE_RATE)
    best = select_threshold(table, THRESHOLD_RULE, TOP_N_RATE, len(train), MAX_LIST_RATE)
    row = table[table["門檻"] == best].iloc[0]

    save_threshold({"threshold": best, "rule": THRESHOLD_RULE, "selected_on": "訓練集五折交叉驗證預測（OOF）",
                    "assumptions": {"loss_per_default": loss, "cost_per_contact": COST_PER_CONTACT, "rescue_rate": RESCUE_RATE},
                    "validation_metrics": {k: (round(float(v), 4)) for k, v in row.items()},
                    "max_list_rate": MAX_LIST_RATE}, run=run, model_record=record)
    run.save_csv("threshold_table", table.round(4))
    print(f"model_id={record['model_id']} policy_id={run.meta['policy_id']} run_id={run.run_id}")

    print(f"規則：{THRESHOLD_RULE}　漏判損失假設：{loss:,.0f} 元　聯絡成本：{COST_PER_CONTACT} 元　挽回比例：{RESCUE_RATE}")
    print(f"自動選出的門檻：{best}")
    for k, v in row.items():  # 人數與成本用千分位整數，比例取到小數第三位
        print(f"  {k}：{v:,.0f}" if k in ("名單人數", "誤報人數", "漏判人數", "總成本") else f"  {k}：{v:.3f}")

    # 敏感度分析：成本假設改變時，選出的門檻會怎麼變（報告用）
    sens = []
    for contact in sensitivity_contact_costs:
        for rescue in sensitivity_rescue_rates:
            t = threshold_table(y_train, oof, loss, contact, rescue)
            b = select_threshold(t, "cost"); r = t[t["門檻"] == b].iloc[0]
            sensitivity = registry.start_run("L1", "02 legacy-average-loss sensitivity", model_id=record["model_id"], rules_freeze_id=args.rules_freeze_id)
            sensitivity.bind_model(record)
            selected = save_threshold({"threshold": b, "rule": "cost", "selected_on": "L1 five-fold OOF; legacy average loss",
                "assumptions": {"loss_per_default": loss, "cost_per_contact": contact, "rescue_rate": rescue},
                "max_list_rate": None}, run=sensitivity, model_record=record)
            sensitivity.finish()
            sens.append({"run_id": sensitivity.run_id, "model_id": record["model_id"], "policy_id": selected["policy_id"],"聯絡成本": contact, "挽回比例": rescue, "最佳門檻": b, "名單人數": int(r["名單人數"]),
                         "Precision": round(r["Precision"], 3), "Recall": round(r["Recall"], 3)})
    sens = pd.DataFrame(sens)
    run.save_csv("threshold_sensitivity", sens)
    run.finish()
    print("\n敏感度分析（總成本最低規則，不設人力上限）：")
    print(sens.to_string(index=False))
    # TODO 畫出門檻對總成本的曲線圖，存到 results/<run_id>/
    # TODO 在報告說明三個成本假設的依據，並討論敏感度分析的結果
    # TODO 設定 config.py 的 MAX_LIST_RATE（名單人力上限）：目前聯絡成本低時，總成本最低會選擇聯絡幾乎所有人，實務上不可行


if __name__ == "__main__":
    main()
