"""單一客戶查詢：風險分數、主要原因、TAIDE 說明、關懷訊息草稿，行員確認後才發送 LINE。
負責人：蔡翌佳｜分支：feat/customer-page
TAIDE 與 LINE 的功能寫在 llm/ 與 notify/（蘇韻蓉負責），這頁只要呼叫，不用修改那兩個資料夾。
TODO
1. 用折線圖顯示這位客戶近六個月的帳單、繳款金額與延遲狀態（資料可從 src.data.load_data() 依客戶 ID 取得）
2. 用長條圖呈現主要風險因素，取代目前的三行文字
3. 顯示風險分數在全體客戶中的位置，例如前百分之幾
4. 整理版面，讓行員一眼看到重點
"""
import streamlit as st

from llm import explain
from notify import line_push


def render(df, meta):
    st.header("單一客戶查詢")
    default_id = int(df.iloc[0]["customer_id"])
    cid = st.number_input("客戶 ID", min_value=1, value=default_id, step=1)
    row = df[df["customer_id"] == cid]
    if row.empty:
        st.warning("測試集中沒有這位客戶。")
        return
    c = row.iloc[0].to_dict()

    a, b, d = st.columns(3)
    a.metric("風險分數", f"{c['risk_score']:.2f}")
    b.metric("風險排名", f"{int(c['risk_rank']):,} / {len(df):,}")
    d.metric("是否列入名單", "是" if c["in_list"] else "否")

    st.subheader("主要風險原因")
    for i in (1, 2, 3):
        if isinstance(c.get(f"factor_{i}"), str) and c[f"factor_{i}"]:
            st.write(f"{i}. {c[f'factor_{i}']}")

    st.subheader("風險說明（給行員）")
    if st.button("產生說明"):
        st.session_state["exp"] = explain.explain(c)
    if "exp" in st.session_state:
        text, how = st.session_state["exp"]
        st.info(text)
        st.caption(f"產生方式：{how}")

    st.subheader("關懷訊息草稿（給客戶）")
    if st.button("產生訊息草稿"):
        st.session_state["msg"] = explain.care_message(c)[0]
    msg = st.text_area("行員可直接修改內容", st.session_state.get("msg", ""), height=120)
    ok = st.checkbox("我已確認訊息內容正確")
    if st.button("發送 LINE（測試帳號）", disabled=not (ok and msg.strip())):
        st.success(line_push.push_text(msg))
