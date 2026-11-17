"""公平性監控（T-蔡1：版面與說明文字）。

負責人：蔡翌佳｜分支：feat/fairness-page-v2
本頁只呈現共用評估結果，不自行計算公平性指標或讀取資料檔。
C6 串接後，將已驗證的交接資料、run metadata 與共用評估結果傳入 render。
T-蔡2 的正式圖表及雙模型比較另行完成。
"""

import pandas as pd
import streamlit as st


GROUPS = ("性別", "年齡層", "性別×年齡層", "教育程度", "婚姻狀況")
GROUP_COLUMNS = (
    "分組方式", "族群", "總人數", "違約人數", "未違約人數",
    "Recall", "Recall 下限", "Recall 上限", "FNR", "FPR",
    "FPR 下限", "FPR 上限", "小樣本",
)
GAP_COLUMNS = ("分組方式", "EO 差距", "FPR 差距")
DISPLAY_COLUMNS = (
    "族群", "總人數", "違約人數", "未違約人數", "Recall",
    "Recall 95% 信賴區間", "FNR", "FPR", "FPR 95% 信賴區間", "樣本提醒",
)


def _format_rate(value):
    """只調整顯示格式；空值不當成 0。"""
    return "無法計算" if pd.isna(value) else f"{value:.3f}"


def _format_interval(lower, upper):
    if pd.isna(lower) or pd.isna(upper):
        return "無法計算"
    return f"{lower:.3f} ～ {upper:.3f}"


def _group_mask(table, group):
    # 容許分組名稱中乘號前後有空白，只影響畫面篩選。
    return table["分組方式"].astype("string").str.replace(" ", "", regex=False).eq(group)


def _show_explanation():
    with st.expander("這些指標代表什麼？", expanded=True):
        st.markdown("""
- **Recall（召回率）**：實際違約的客戶中，有多少人被列入風險名單。數值越高，表示越能找出實際違約的客戶。
- **FNR（漏判率）**：實際違約的客戶中，有多少人沒有被列入名單。
- **FPR（誤報率）**：實際未違約的客戶中，有多少人被列入名單。
- **EO 差距**＝各族群 Recall 的最大差距，也就是最高 Recall 減去最低 Recall。
- **FPR 差距**：各族群最高與最低誤報率之間的差距。
- **Wilson 95% 信賴區間**：呈現各族群 Recall 與 FPR 估計值的不確定性。區間越寬，表示估計越不精確，應搭配樣本數解讀。
""")
        st.caption("例如差距為 0.05，代表相差 5 個百分點。差距縮小不代表符合公平性要求。")
        st.caption("違約人數或未違約人數少於 93 人時，標示為小樣本，解讀以信賴區間為準。分母為 0 時，對應指標無法計算。")


def _show_results(groups, gaps, group):
    """呈現 src.protocol.evaluate() 的 groups 與 gaps，不重算統計量。"""
    if not set(GROUP_COLUMNS).issubset(groups.columns) or not set(GAP_COLUMNS).issubset(gaps.columns):
        st.error("公平性資料欄位尚未符合共用評估介面，請確認資料串接。")
        return

    t = groups.loc[_group_mask(groups, group)].copy()
    g = gaps.loc[_group_mask(gaps, group)]
    if t.empty or len(g) != 1:
        st.info("此分組尚無可顯示的共用評估結果。")
        return

    if t["小樣本"].isna().any() or not t["小樣本"].isin([True, False]).all():
        st.error("小樣本欄位格式不符，請確認共用評估結果。")
        return

    a, b = st.columns(2)
    a.metric("EO 差距（Recall 最大差距）", _format_rate(g.iloc[0]["EO 差距"]))
    b.metric("FPR 最大差距", _format_rate(g.iloc[0]["FPR 差距"]))

    small = t["小樣本"].eq(True)
    if small.any():
        names = "、".join(t.loc[small, "族群"].astype(str))
        st.warning(f"小樣本族群：{names}。違約人數或未違約人數少於 93 人，解讀以信賴區間為準。")

    display = t[["族群", "總人數", "違約人數", "未違約人數"]].copy()
    display["Recall"] = t["Recall"].map(_format_rate)
    display["Recall 95% 信賴區間"] = [
        _format_interval(lo, hi) for lo, hi in zip(t["Recall 下限"], t["Recall 上限"])
    ]
    display["FNR"] = t["FNR"].map(_format_rate)
    display["FPR"] = t["FPR"].map(_format_rate)
    display["FPR 95% 信賴區間"] = [
        _format_interval(lo, hi) for lo, hi in zip(t["FPR 下限"], t["FPR 上限"])
    ]
    display["樣本提醒"] = small.map({True: "小樣本，解讀以區間為準", False: ""})
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption("Recall 與 FNR 的分母是實際違約人數；FPR 的分母是實際未違約人數。分母為 0 的族群不納入對應差距計算。")
    st.bar_chart(t.set_index("族群")[["Recall", "FPR"]], stack=False)


def render(df, meta, *, evaluation=None):
    """T-蔡1 的暫時呈現介面；C6 尚未接上時不顯示舊資料指標。

    evaluation 應為共用 evaluate() 回傳的 dict，由整合層提供；
    不在本頁載入交接檔、推測政策或重算指標。
    """
    st.header("公平性監控")
    # C6 在模擬模式會移除標籤。沒有完整實際標籤就提早返回，
    # 即使呼叫端傳入回測結果，也不能在這個模式顯示。
    label = next((col for col in ("actual_default", "default") if col in df.columns), None)
    if label is None or df.empty or df[label].isna().any():
        st.info("新資料預警（模擬）或尚無完整實際違約標籤：無法計算公平性回測指標。待取得實際結果後，才能進行族群比較。")
        return

    # 此為 T-蔡1 開發版。正式資料的載入與一致性驗證由 C6 負責。
    if meta.get("level") != "L1-H":
        st.info("尚未接入 L1-H 開發資料。為避免顯示舊版或未確認來源的結果，暫不顯示回測指標。")
        return

    st.caption("歷史回測｜L1-H 開發資料，非正式結果；僅供介面與功能測試。")
    _show_explanation()
    group = st.selectbox("分組方式", GROUPS)
    if evaluation is None:
        st.info("尚未接入共用評估結果；開發資料數字將在資料串接完成後顯示。")
        st.dataframe(pd.DataFrame(columns=DISPLAY_COLUMNS), hide_index=True, width="stretch")
        return
    _show_results(evaluation["groups"], evaluation["gaps"], group)
