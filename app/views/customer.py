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


def _invalidate_confirmation():
    st.session_state["customer_confirmed"] = False
    st.session_state.pop("customer_approval", None)


def _confirm_message():
    if st.session_state.get("customer_confirmed"):
        st.session_state["customer_approval"] = (
            st.session_state["customer_content_id"], st.session_state["customer_message"]
        )
    else:
        st.session_state.pop("customer_approval", None)


def render(df, meta):
    # 在 widget 建立前重設確認，避免發送後畫面仍顯示已核准。
    outcome = st.session_state.pop("customer_send_outcome", None)
    if outcome is not None:
        _invalidate_confirmation()
    st.header("單一客戶查詢")
    default_id = int(df.iloc[0]["customer_id"])
    cid = st.number_input("客戶 ID", min_value=1, value=default_id, step=1)
    if st.session_state.get("customer_content_id") != cid:
        for key in ("exp", "msg", "customer_explanation", "customer_message",
                    "customer_approval", "customer_confirmed"):
            st.session_state.pop(key, None)
        st.session_state["customer_content_id"] = cid
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
        st.session_state["customer_explanation"] = (cid, explain.explain(c))
    if "customer_explanation" in st.session_state:
        owner, (text, how) = st.session_state["customer_explanation"]
        assert owner == cid
        st.info(text)
        st.caption(f"產生方式：{how}")

    st.subheader("關懷訊息草稿（給客戶）")
    if st.button("產生訊息草稿"):
        st.session_state["customer_message"] = explain.care_message(c)[0]
        _invalidate_confirmation()
    msg = st.text_area("行員可直接修改內容", key="customer_message", height=120,
                       on_change=_invalidate_confirmation)
    reason = explain.check_care(msg)
    st.caption("目前僅允許固定模板內容；其他改寫須另行審核。修改後請重新確認。")
    if msg and reason:
        st.warning(f"訊息未通過檢查：{reason}。請重新產生訊息草稿以使用固定模板。")
    ok = st.checkbox("我已確認訊息內容正確", key="customer_confirmed",
                     on_change=_confirm_message)
    approved = st.session_state.get("customer_approval") == (cid, msg)
    simulation = line_push.is_simulation()
    st.caption("LINE 模式：模擬，不會對外發送。" if simulation else "LINE 模式：實際發送至設定的測試帳號。")
    if st.button("發送 LINE（測試帳號）", disabled=not (ok and approved and not reason)):
        try:
            result = line_push.push_text(msg, approved_text=st.session_state["customer_approval"][1])
        except Exception:
            outcome = ("error", "LINE 發送失敗，請檢查設定；請勿直接重複發送。")
        else:
            if simulation:
                outcome = ("info", result)
            else:
                outcome = ("success", result)
        st.session_state.pop("customer_approval", None)
        st.session_state["customer_send_outcome"] = outcome
        st.rerun()
    if outcome is not None:
        level, message = outcome
        getattr(st, level)(message)
        st.caption("本次確認已失效；再次操作前請重新檢查訊息並勾選確認。")
