"""違約預警平台。執行：streamlit run app/main.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # 先讀 .env，TAIDE 與 LINE 設定才會生效

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.views import customer, fairness, risk_list  # noqa: E402

st.set_page_config(page_title="信用卡違約預警平台", layout="wide")


@st.cache_data
def load():
    pred = ROOT / "outputs" / "predictions.csv"
    if not pred.exists():
        st.error("找不到 outputs/predictions.csv，請先執行 python -m src.export")
        st.stop()
    meta_path = ROOT / "outputs" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return pd.read_csv(pred), meta


df, meta = load()
PAGES = {"每月風險名單": risk_list, "單一客戶查詢": customer, "公平性監控": fairness}
st.sidebar.title("違約預警平台")
page = st.sidebar.radio("頁面", list(PAGES), key="page")
st.sidebar.caption("模式：歷史資料回測（2005 年 UCI 公開資料，測試集 6,000 人）")
if meta:
    st.sidebar.caption(f"模型更新：{meta.get('created', '')}　門檻：{meta.get('threshold')}")
PAGES[page].render(df, meta)
