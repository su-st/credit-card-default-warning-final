"""T-蔡3：單一客戶查詢（C6 交接資料、C5 草稿與 PR #8 發送安全）。

只呈現平台傳入的客戶資料，不另外讀取資料檔、重算分數或調整名單。
TAIDE 品質檢查與客戶訊息發送白名單是不同的規則。
"""
import streamlit as st

from llm import explain, templates
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


def _show_generation(record):
    """顯示 C5 產生紀錄，不將原始草稿用作發送授權。"""
    how, reason = record["how"], record["reason"]
    st.caption(f"產生方式：{how}")
    if reason:
        # 呼叫例外可能帶有本機連線資訊，不直接顯示例外字串。
        detail = "TAIDE 呼叫失敗（請確認本機服務或逾時）" if "呼叫失敗" in how else reason
        st.warning(f"未採用 TAIDE 原始輸出的原因：{detail}")
    else:
        st.caption("已通過草稿品質檢查，仍需行員檢查內容。")
    # 未通過的原始輸出不呈現，避免不合格內容被誤用；顯示採用文字與原因即可。


def _prepare_care(record):
    """保留 C5 草稿，但送出欄位只預填經現有白名單核可的固定模板。"""
    reason = explain.check_care(record["text"])
    if reason:
        return templates.care_message({}), reason
    return record["text"], ""


def _show_customer_summary(df, c, meta):
    rank = int(c["risk_rank"])
    selected = df["in_list"].eq(1)
    count = int(selected.sum())
    a, b, d = st.columns(3)
    a.metric("風險分數", f"{c['risk_score']:.2f}")
    b.metric("全體風險排名", f"{rank:,} / {len(df):,}")
    d.metric("預警名單狀態", "名單內" if c["in_list"] == 1 else "名單外")
    if c["in_list"] == 1:
        # 依交接檔既有排名呈現名單內位置，不重算分數或名單。
        list_position = int((selected & df["risk_rank"].le(rank)).sum())
        st.caption(f"名單內第 {list_position:,} 名（共 {count:,} 人），全體第 {rank:,} 名。")
    else:
        st.caption(f"名單外，本期預警名單共 {count:,} 人。")
    if meta.get("threshold") is not None:
        cap = meta.get("max_list_rate")
        cap_text = "不限制" if cap is None else f"{cap:.0%}"
        st.caption(f"本期政策門檻：{meta['threshold']}｜名單上限：{cap_text}，以交接檔最終名單判定為準。")
    if meta.get("mode") == "歷史回測" and c.get("actual_default") in (0, 1):
        st.caption("歷史回測實際結果：" + ("違約" if c["actual_default"] == 1 else "未違約"))


def render(df, meta):
    # 在 widget 建立前重設確認，避免發送後畫面仍顯示已核准。
    outcome = st.session_state.pop("customer_send_outcome", None)
    if outcome is not None:
        _invalidate_confirmation()
    st.header("單一客戶查詢")
    if meta.get("mode") == "新資料預警（模擬）":
        st.caption("新資料預警（模擬）｜未串接真實銀行系統，不顯示實際結果。")
    elif meta.get("mode") == "歷史回測":
        st.caption("歷史回測｜開發資料，非正式結果" if meta.get("level") == "L1-H" else "歷史回測")
    if df.empty:
        st.warning("目前沒有可供查詢的客戶資料。")
        return
    default_id = int(df.iloc[0]["customer_id"])
    cid = st.number_input("客戶 ID", min_value=1, value=default_id, step=1)
    if st.session_state.get("customer_content_id") != cid:
        for key in ("exp", "msg", "customer_explanation", "customer_message",
                    "customer_approval", "customer_confirmed", "customer_care_generation"):
            st.session_state.pop(key, None)
        st.session_state["customer_content_id"] = cid
        outcome = None
    row = df[df["customer_id"] == cid]
    if row.empty:
        st.warning("目前資料中沒有這位客戶。")
        return
    c = row.iloc[0].to_dict()

    _show_customer_summary(df, c, meta)

    st.subheader("主要風險因素")
    st.caption("模型判斷依據，不代表原因（不代表違約原因）。")
    factors = [c[f"factor_{i}"] for i in (1, 2, 3)
               if isinstance(c.get(f"factor_{i}"), str) and c[f"factor_{i}"].strip()]
    if factors:
        for i, factor in enumerate(factors, 1):
            st.write(f"{i}. {factor}")
    else:
        st.caption("目前無可顯示的主要風險因素。")

    # 只提供共用產生函式需要的欄位，不傳入實際標籤、ID 或其他資料。
    facts = {key: c[key] for key in ("risk_score", "risk_rank", "in_list",
                                     "factor_1", "factor_2", "factor_3") if key in c}
    st.subheader("風險說明（給行員）")
    if st.button("產生說明"):
        st.session_state["customer_explanation"] = (
             cid, explain.generate("explain", {**facts, "customer_id": cid})
        )
    if "customer_explanation" in st.session_state:
        owner, record = st.session_state["customer_explanation"]
        assert owner == cid
        st.info(record["text"])
        _show_generation(record)

    st.subheader("關懷訊息草稿（給客戶）")
    st.caption("草稿品質檢查與 LINE 發送檢查分開，只有固定模板通過發送檢查後才可發送。")
    if st.button("產生訊息草稿"):
        record = explain.generate("care", facts)
        message, send_reason = _prepare_care(record)
        st.session_state["customer_care_generation"] = (cid, record, send_reason)
        st.session_state["customer_message"] = message
        _invalidate_confirmation()
    if "customer_care_generation" in st.session_state:
        owner, record, send_reason = st.session_state["customer_care_generation"]
        assert owner == cid
        st.caption("本次產生紀錄（不代表下方目前編輯內容已核准）")
        _show_generation(record)
        if send_reason:
            st.caption("TAIDE 草稿（僅供行員參考，不可直接發送）：")
            st.text(record["text"])
            st.warning(f"草稿不符合發送白名單：{send_reason}。產生時已改用固定模板，仍需行員確認。")
    st.caption("下方為待確認的客戶訊息，不是上方的研究草稿。")
    msg = st.text_area("行員可直接修改內容", key="customer_message", height=120,
                       on_change=_invalidate_confirmation)
    reason = explain.check_care(msg)
    st.caption("目前僅允許固定模板內容，其他改寫須另行審核。修改後請重新確認。")
    if msg and reason:
        st.warning(f"訊息未通過檢查：{reason}。請重新產生訊息草稿以使用固定模板。")
    ok = st.checkbox("我已確認訊息內容正確", key="customer_confirmed",
                     on_change=_confirm_message)
    approved = st.session_state.get("customer_approval") == (cid, msg)
    simulation = line_push.is_simulation()
    st.caption("LINE 模式：模擬，不會對外發送。" if simulation else "LINE 模式：實際發送至設定的測試帳號。")
    if st.button("發送 LINE（測試帳號）", disabled=not (ok and approved and not reason)):
        try:
            result = line_push.send_customer_care(msg, approved_text=st.session_state["customer_approval"][1])
        except Exception:
            outcome = ("error", "LINE 發送失敗，請檢查設定。請勿直接重複發送。")
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
        st.caption("本次確認已失效，再次操作前請重新檢查訊息並勾選確認。")
