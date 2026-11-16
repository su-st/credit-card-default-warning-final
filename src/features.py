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


def add_available_features(d: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Deterministic per-customer features using months lag+1 through 6 only.

    At lag=0 the values and column order equal add_features exactly. Trend uses
    the most recent available utilization minus month 6 utilization.
    """
    if not isinstance(lag, int) or isinstance(lag, bool) or not 0 <= lag <= 5:
        raise ValueError("lag must be an integer from 0 through 5")
    out = d.copy()
    months = range(lag + 1, 7)
    for m in months:
        out[f"UTIL_{m}"] = out[f"BILL_AMT{m}"] / out["LIMIT_BAL"]
    for m in range(lag + 1, 6):
        out[f"PAYRATIO_{m}"] = out[f"PAY_AMT{m}"] / out[f"BILL_AMT{m+1}"].clip(lower=1)
    pay = [f"PAY_{m}" for m in months]
    out["N_DELAY"] = (out[pay] > 0).sum(axis=1)
    out["MAX_DELAY"] = out[pay].max(axis=1)
    out["UTIL_TREND"] = out[f"UTIL_{lag+1}"] - out["UTIL_6"]
    return out
