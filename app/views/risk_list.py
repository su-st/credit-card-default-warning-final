"""每月風險名單。"""
import streamlit as st


def render(df, meta):
    st.header("每月風險名單")
    st.caption("依模型風險分數排序。名單人數由門檻決定，門檻情境分析完成後會更新。")

    n = st.slider("顯示前幾名", 50, 2000, int(df["in_list"].sum()), step=50)
    top = df.head(n)
    c1, c2, c3 = st.columns(3)
    c1.metric("名單人數", f"{n:,}")
    c2.metric("名單中實際違約（回測）", f"{int(top['actual_default'].sum()):,}")
    c3.metric("涵蓋全部違約者比例（回測）", f"{top['actual_default'].sum() / df['actual_default'].sum():.1%}")

    show = top[["risk_rank", "customer_id", "risk_score", "factor_1", "factor_2", "factor_3", "limit_bal"]]
    show = show.rename(columns={"risk_rank": "排名", "customer_id": "客戶 ID", "risk_score": "風險分數",
                                "factor_1": "主要原因 1", "factor_2": "主要原因 2", "factor_3": "主要原因 3",
                                "limit_bal": "信用額度"})
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.download_button("下載名單 CSV", show.to_csv(index=False).encode("utf-8-sig"), "risk_list.csv")
