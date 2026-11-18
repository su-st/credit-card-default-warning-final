"""公平性監控（T-蔡1 與 T-蔡2 比較版面）。

負責人：蔡翌佳
本頁只呈現共用評估結果，不自行計算公平性指標或讀取資料檔。
透過 C6 共用資料介面取得已驗證的政策，再由 protocol.evaluate 計算。
T-蔡2 先提供比較版面，F3 結果與共用讀取介面確認後再串接。
"""

import pandas as pd
import streamlit as st

from app.data import BACKTEST, load_platform_run, page_data
from src import protocol as p


GROUPS = ("性別", "年齡層", "性別×年齡層", "教育程度", "婚姻狀況")
GROUP_COLUMNS = (
    "分組方式", "族群", "總人數", "違約人數", "未違約人數",
    "Recall", "Recall 下限", "Recall 上限", "FNR", "FPR",
    "FPR 下限", "FPR 上限", "小樣本",
)
GAP_COLUMNS = ("分組方式", "EO 差距", "FPR 差距")



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


def _evaluate_backtest(df, meta):
    """使用 C6 已驗證的同一 run 與完整政策，不直接讀取交接檔。"""
    # page_data 目前只傳遞政策摘要；完整政策必須透過共用介面取得。
    bundle = load_platform_run(meta["run_id"])
    verified_df, verified_meta = page_data(bundle, BACKTEST)
    keys = ("run_id", "model_id", "policy_id", "level", "predictions_sha256")
    if any(meta.get(key) != verified_meta[key] for key in keys):
        raise ValueError("公平性頁面與已驗證 run 的識別不一致")
    if not df.equals(verified_df):
        raise ValueError("公平性頁面資料與已驗證交接資料不一致")

    threshold = bundle["policy"]
    run = bundle["run"]
    if threshold.get("source_level") != meta["level"]:
        raise ValueError("政策來源資料層級不一致")
    if any(threshold[key] != run[key] for key in ("run_id", "model_id", "policy_id", "policy_hash")):
        raise ValueError("政策與已驗證 run 的識別或雜湊不一致")

    if "protocol_policy" in threshold:
        policy = p.Policy(**threshold["protocol_policy"])
        if policy.policy_id != meta["policy_id"] or policy.policy_hash != threshold["policy_hash"]:
            raise ValueError("政策識別或雜湊不一致")
    else:
        # C1 格式的政策 ID/hash 包含訓練來源證據，已由
        # load_platform_run -> Registry.load_run -> validate_threshold 驗證。
        # 使用其原有單一門檻及上限建立 C2 評估物件，不重新選擇政策，
        # 也不將轉換後的 C2 ID 誤認為登錄的 C1 政策 ID。
        legacy = threshold["policy"]
        if legacy["kind"] != "single" or legacy["threshold"] != threshold["threshold"]:
            raise ValueError("不支援或不一致的 C1 政策")
        policy = p.Policy.single(legacy["threshold"], legacy["max_list_rate"])

    frame = verified_df.set_index("customer_id", verify_integrity=True)
    groups = {
        "性別": frame["sex"],
        "年齡層": frame["age_group"],
        "性別×年齡層": frame["sex"].astype("string") + " × " + frame["age_group"].astype("string"),
        "教育程度": frame["education"],
        "婚姻狀況": frame["marriage"],
    }
    if any(series.isna().any() for series in groups.values()):
        raise ValueError("交接資料存在缺失族群標籤，請確認共用分組規則")
    evaluation = p.evaluate(frame["actual_default"], score=frame["risk_score"],
                            pred=frame["in_list"], policy=policy, groups=groups)
    # evaluate 會驗證既有名單與政策一致。頁面只呈現族群表和差距；
    # 原始 model_id / run_id / policy_id 仍由 C6 側欄提供。
    return {"groups": evaluation["groups"], "gaps": evaluation["gaps"]}


# 規格 §4.5.4 與 D2 的比較名稱及文字說明，並非實際政策紀錄。
COMPARISON_METHODS = {
    "共同數值門檻 0.5": "兩個模型皆使用 0.5 門檻，不設名單上限。名單人數可能不同，用於對照期中結果。",
    "固定名單人數（前 20%）": "兩個模型皆依分數取前 20%，控制名單規模，比較排序表現。",
    "各模型正式政策": "各自使用訓練資料 OOF 選出的政策，主要情境名單上限為 20%。實際門檻須以各模型的政策紀錄為準。",
}


def _show_model_comparison(group):
    """T-蔡2 版面預備：不讀檔、不重算、不以 T-蔡1 結果代替 F3。"""
    st.divider()
    st.subheader("含與不含敏感屬性模型比較")
    st.caption("RQ1｜F3 分析結果比較（尚未串接）")
    method = st.selectbox("比較方式", tuple(COMPARISON_METHODS), key="fairness_comparison_method")
    st.caption(COMPARISON_METHODS[method])
    st.caption(f"目前分組：{group}（沿用上方分組方式）")
    a, b = st.columns(2)
    with a:
        st.markdown("**M1｜含敏感屬性**")
        st.caption("尚未取得 F3 比較結果")
    with b:
        st.markdown("**M1-NS｜不含敏感屬性**")
        st.caption("尚未取得 F3 比較結果")
    st.info("尚未載入 F3 公平性比較結果。完成資料介面串接與來源驗證後，才會顯示 EO 差距、FPR 差距、族群表格及圖表。")
    st.caption("結果來源：F3 的 results/<run_id>/ 分析結果。兩模型各自的資料層級、model_id、policy_id、run_id 與門檻來源尚待確認。")
    st.caption("本區不使用上方 L1-H 單模型評估結果代替 F3，也不在頁面重算比較指標。差距縮小不代表符合公平性要求。")


def render(df, meta):
    """C6 平台入口：模擬模式不讀取政策或計算回測指標。"""
    if meta.get("mode") != BACKTEST:
        st.info("公平性監控僅在歷史回測模式顯示；模擬資料沒有實際標籤，無法計算族群錯誤率。")
        return
    st.header("公平性監控")
    if df.empty or "actual_default" not in df or not df["actual_default"].isin([0, 1]).all():
        st.info("尚無完整實際違約標籤，無法計算公平性回測指標。")
        return
    if meta.get("level") != "L1-H":
        st.info("目前僅支援 L1-H 開發資料，正式資料尚未開放。")
        return

    st.caption("歷史回測｜L1-H 開發資料，非正式結果；僅供介面與功能測試。")
    try:
        evaluation = _evaluate_backtest(df, meta)
    except Exception as exc:
        st.error(f"公平性資料驗證或評估失敗，請回報組長：{exc}")
        return
    _show_explanation()
    group = st.selectbox("分組方式", GROUPS)
    _show_results(evaluation["groups"], evaluation["gaps"], group)
    _show_model_comparison(group)
