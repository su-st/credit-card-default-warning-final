"""公平性監控：各族群的漏判率與誤判率（歷史回測）。
負責人：蔡翌佳｜分支：feat/fairness-page
TODO
1. 加入「性別 × 年齡層」交叉族群
2. 差距超過設定值（例如 0.05）時顯示警示
3. 人數少於 100 的族群加註「人數少，結果僅供參考」
4. 並排比較「使用」與「移除」敏感屬性兩個模型（需要余旻恩的第二份交接檔）
5. 加入白話說明，解釋漏判率、誤判率與差距的意義
"""
import pandas as pd
import streamlit as st

from src.evaluate import group_errors

GROUPS = {"性別": "sex", "年齡層": "age_group", "教育程度": "education", "婚姻狀況": "marriage"}


def render(df, meta):
    st.header("公平性監控")
    st.caption("比較各族群被漏判（違約卻沒列入名單）與誤判（沒違約卻列入名單）的比例。")
    g = st.selectbox("分組方式", list(GROUPS))
    y = df["actual_default"]
    t = group_errors(y, df["in_list"].values, df[GROUPS[g]])
    gap_fnr = t["漏判率"].max() - t["漏判率"].min()
    gap_fpr = t["誤判率"].max() - t["誤判率"].min()
    a, b = st.columns(2)
    a.metric("漏判率最大差距", f"{gap_fnr:.3f}")
    b.metric("誤判率最大差距", f"{gap_fpr:.3f}")
    st.dataframe(t.round(3), hide_index=True, use_container_width=True)
    st.bar_chart(t.set_index("族群")[["漏判率", "誤判率"]], stack=False)
