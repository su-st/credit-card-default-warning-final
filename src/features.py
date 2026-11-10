"""14 個衍生特徵，與期中筆記本第 22 格相同。只使用 4 至 9 月的資料，不使用違約標籤。"""
import pandas as pd

from .data import PAY_COLS

ENG_PREFIX = ("UTIL_", "PAYRATIO_", "N_DELAY", "MAX_DELAY")


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for m in range(1, 7):
        d[f"UTIL_{m}"] = d[f"BILL_AMT{m}"] / d["LIMIT_BAL"]          # 額度使用率
    for m in range(1, 6):
        d[f"PAYRATIO_{m}"] = d[f"PAY_AMT{m}"] / d[f"BILL_AMT{m+1}"].clip(lower=1)  # 還款比例
    d["N_DELAY"] = (d[PAY_COLS] > 0).sum(axis=1)                      # 近六個月延遲次數
    d["MAX_DELAY"] = d[PAY_COLS].max(axis=1)                          # 最長延遲月數
    d["UTIL_TREND"] = d["UTIL_1"] - d["UTIL_6"]                       # 使用率變化
    return d


def engineered_cols(d: pd.DataFrame) -> list:
    return [c for c in d.columns if c.startswith(ENG_PREFIX)]
