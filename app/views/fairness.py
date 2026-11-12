
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

GROUPS = {
    "性別": "sex",
    "年齡層": "age_group",
    "教育程度": "education",
    "婚姻狀況": "marriage",
}


def render(df, meta):
    st.header("公平性監控")
    st.caption("比較各族群被漏判（違約卻沒列入名單）與誤判（沒違約卻列入名單）的比例。")

    # 新增：白話說明
    with st.expander("這些指標代表什麼？", expanded=True):
        st.markdown("""
        - **漏判率（FNR）**：實際上會違約，卻沒有被系統列入風險名單的比例。漏判率越高，代表越容易漏掉真正有風險的客戶。
        - **誤判率（FPR）**：實際上沒有違約，卻被系統列入風險名單的比例。誤判率越高，代表越容易把沒有違約的客戶判成有風險。
        - **族群差距**：不同族群之間，最高與最低錯誤率的差距。差距越大，表示模型對不同族群的預測表現越不一致，值得進一步檢查。
        """)

        st.caption(
            "例如差距為 0.05，代表最高與最低的錯誤率相差 5 個百分點。"
            "但有差距不一定代表模型存在歧視，仍需要搭配樣本數和其他分析判斷。"
        )

    g = st.selectbox("分組方式", list(GROUPS))

    y = df["actual_default"]
    t = group_errors(y, df["in_list"].values, df[GROUPS[g]])

    gap_fnr = t["漏判率"].max() - t["漏判率"].min()
    gap_fpr = t["誤判率"].max() - t["誤判率"].min()

    a, b = st.columns(2)
    a.metric("漏判率最大差距", f"{gap_fnr:.3f}")
    b.metric("誤判率最大差距", f"{gap_fpr:.3f}")

    # 新增：確認各族群人數
    count_col = next(
        (col for col in ["人數", "樣本數", "總人數"] if col in t.columns),
        None,
    )

    if count_col is None:
        # 若評估函式沒有提供人數，就從原始資料計算
        counts = (
            df[GROUPS[g]]
            .astype("string")
            .fillna("缺失值")
            .value_counts()
        )
        t["人數"] = (
            t["族群"]
            .astype("string")
            .fillna("缺失值")
            .map(counts)
        )
        count_col = "人數"

    # 新增：小樣本族群提醒
    t["樣本提醒"] = t[count_col].apply(
        lambda n: "⚠️ 人數少，結果僅供參考"
        if pd.notna(n) and n < 100
        else ""
    )

    small_groups = t[t[count_col] < 100]

    if not small_groups.empty:
        names = "、".join(small_groups["族群"].astype(str))
        st.warning(
            f"以下族群人數少於 100 人：{names}。"
            "由於樣本較少，錯誤率可能較不穩定，結果僅供參考。"
        )

    if t[count_col].isna().any():
        st.warning("部分族群無法確認人數，請檢查族群名稱與原始資料是否一致。")

    st.dataframe(t.round(3), hide_index=True, use_container_width=True)

    st.caption(
        "提醒：漏判率是以實際違約人數為分母，"
        "誤判率是以實際未違約人數為分母。"
        "即使族群總人數達 100 人，仍需留意各類實際樣本是否足夠。"
    )

    st.bar_chart(
        t.set_index("族群")[["漏判率", "誤判率"]],
        stack=False,
    )
