"""門檻自動選擇。門檻只用訓練集的交叉驗證預測（驗證資料）決定，測試集只用來報告最後結果。"""
import json

import numpy as np
import pandas as pd

from .config import LIST_THRESHOLD, OUT_DIR

THRESHOLD_FILE = OUT_DIR / "threshold.json"
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def confusion(y_true, prob, t):
    y = np.asarray(y_true); pred = np.asarray(prob) >= t
    return dict(TP=int((pred & (y == 1)).sum()), FP=int((pred & (y == 0)).sum()),
                FN=int((~pred & (y == 1)).sum()), TN=int((~pred & (y == 0)).sum()))


def total_cost(c, loss, contact, rescue):
    """總成本 = 漏判損失 + 聯絡了但沒救回的損失 + 聯絡成本。"""
    return c["FN"] * loss + c["TP"] * (1 - rescue) * loss + (c["TP"] + c["FP"]) * contact


def threshold_table(y_true, prob, loss, contact, rescue) -> pd.DataFrame:
    """每個門檻的名單人數、Precision、Recall、F1、誤報人數與總成本。"""
    rows = []
    for t in GRID:
        c = confusion(y_true, prob, t); n = c["TP"] + c["FP"]
        prec = c["TP"] / n if n else 0.0
        rec = c["TP"] / (c["TP"] + c["FN"])
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"門檻": t, "名單人數": n, "Precision": prec, "Recall": rec, "F1-score": f1,
                     "誤報人數": c["FP"], "漏判人數": c["FN"], "總成本": total_cost(c, loss, contact, rescue)})
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame, rule: str, top_n_rate: float = 0.2, n_total: int | None = None,
                     max_list_rate: float | None = None) -> float:
    """依規則從表中選出門檻。cost 規則可加上人力上限 max_list_rate。"""
    if rule == "cost":
        ok = table if max_list_rate is None else table[table["名單人數"] <= max_list_rate * n_total]
        return float(ok.loc[ok["總成本"].idxmin(), "門檻"])
    if rule == "f1":
        return float(table.loc[table["F1-score"].idxmax(), "門檻"])
    if rule == "top_n":
        target = top_n_rate * n_total
        return float(table.loc[(table["名單人數"] - target).abs().idxmin(), "門檻"])
    raise ValueError(f"未知的規則：{rule}")


def save_threshold(info: dict):
    OUT_DIR.mkdir(exist_ok=True)
    THRESHOLD_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def load_threshold() -> tuple[float, str]:
    """回傳（門檻, 來源說明）。還沒執行 02_threshold.py 時用 config 的備用值。"""
    if THRESHOLD_FILE.exists():
        info = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
        return float(info["threshold"]), f"自動選擇（規則：{info['rule']}）"
    return LIST_THRESHOLD, "config 備用值（尚未執行 experiments/02_threshold.py）"
