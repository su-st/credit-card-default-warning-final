"""資料讀取、清理與切分，與期中筆記本第 4、13 格相同。"""
import os
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, RANDOM_STATE, TEST_SIZE, SENSITIVE

UCI_ZIP = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
PAY_COLS = [f"PAY_{m}" for m in range(1, 7)]


def load_raw() -> pd.DataFrame:
    """讀取 UCI 原始資料。data/ 內已有 CSV 或 XLS 就直接讀，否則從 UCI 下載。"""
    csv = DATA_DIR / "default_of_credit_card_clients.csv"
    xls = DATA_DIR / "default of credit card clients.xls"
    if csv.exists():
        return pd.read_csv(csv)
    if xls.exists():
        return pd.read_excel(xls, header=1)
    DATA_DIR.mkdir(exist_ok=True)
    zpath = DATA_DIR / "credit.zip"
    urllib.request.urlretrieve(UCI_ZIP, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA_DIR)
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".xls"):
                return pd.read_excel(os.path.join(root, f), header=1)
    raise FileNotFoundError("找不到資料檔，請手動下載 UCI 資料放到 data/")


def load_data() -> pd.DataFrame:
    """回傳清理後的資料，以 ID 為索引（平台用 ID 識別客戶）。"""
    raw = load_raw()
    df = raw.rename(columns={"PAY_0": "PAY_1", "default payment next month": "default"})
    df = df.set_index("ID")
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def feature_cols(lag: int = 0, sensitive: bool = True) -> list:
    """lag = 0 使用全部 6 個月，lag = k 代表拿掉最近 k 個月（提早 k+1 個月預警）。"""
    months = range(1 + lag, 7)
    cols = (["LIMIT_BAL"] + [f"PAY_{m}" for m in months]
            + [f"BILL_AMT{m}" for m in months] + [f"PAY_AMT{m}" for m in months])
    return cols + (SENSITIVE if sensitive else [])


def split(df: pd.DataFrame):
    """80/20 分層切分，random_state 固定為 42。"""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["default"])


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 29, 39, 49, 200], labels=["29 歲以下", "30-39 歲", "40-49 歲", "50 歲以上"])
