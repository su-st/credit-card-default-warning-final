"""公平性監控（T-蔡1 與 T-蔡2 比較版面）。

負責人：蔡翌佳
本頁只呈現共用評估結果，不自行計算公平性指標或讀取資料檔。
透過 C6 共用資料介面取得已驗證的政策，再由 protocol.evaluate 計算。
T-蔡2 透過 load_fairness_comparison 讀取 F3 已存檔的彙總結果。
"""

import pandas as pd
import streamlit as st

from app.data import (BACKTEST, FAIRNESS_COMPARISONS, load_fairness_comparison,
                      load_platform_run, load_platform_fairness, page_data)
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

    _show_group_table(t)
    st.bar_chart(t.set_index("族群")[["Recall", "FPR"]], stack=False)


def _show_group_table(t):
    """共用族群表的顯示，不重算人數或指標。"""
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
    st.caption("Recall 與 FNR 的分母是實際違約人數，FPR 的分母是實際未違約人數。分母為 0 的族群不納入對應差距計算。")


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


# 使用 F3 的原始比較名稱，說明只描述政策規則，不是實際門檻紀錄。
COMPARISON_METHODS = {
    "共同門檻0.5": "共同數值門檻 0.5，不設名單上限。名單人數可能不同，用於對照期中結果。",
    "固定名單前20%": "固定名單人數，兩個模型皆依分數取前 20%，控制名單規模，比較排序表現。",
    "正式政策": "各模型使用自己的 OOF 門檻及主要情境 20% 名單上限。實際門檻須以各模型對應的政策紀錄為準。",
}
MODEL_LABELS = {"M1": "M1｜含敏感屬性", "M1-NS": "M1-NS｜不含敏感屬性"}
GAP_METRICS = ("EO 差距", "FPR 差距")


def _format_count(value):
    return "無法計算" if pd.isna(value) else f"{value:,.0f}"


def _format_ci_table(table):
    """僅格式化共用模組已計算的 Bootstrap 區間，不補值或重算。"""
    display = table[["指標", "有效重抽", "無效重抽"]].copy()
    display["估計值"] = table["估計值"].map(_format_rate)
    reliable = table["可靠"].eq(True).fillna(False)
    display["Bootstrap 95% 信賴區間"] = [
        _format_interval(lo, hi) if ok else "不可靠，不顯示區間"
        for lo, hi, ok in zip(table["下限"], table["上限"], reliable)
    ]
    display["可靠性"] = reliable.map({True: "可靠", False: "不可靠"})
    return display[["指標", "估計值", "Bootstrap 95% 信賴區間", "有效重抽", "無效重抽", "可靠性"]]


def _check_numeric_columns(table, columns):
    """防止欄位異常時顯示部分結果，不轉換來源值。"""
    for key in columns:
        values = table[key]
        if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                or values.isin([float("inf"), -float("inf")]).any()):
            raise ValueError(f"公平性結果的數值欄位格式不符：{key}")


def _selected_comparison(bundle, method, group):
    """只篩選已驗證的回傳表，先確認整個選項可用，才開始顯示。"""
    def select(table, grouped=True):
        mask = table["政策比較"].eq(method)
        if grouped:
            mask &= _group_mask(table, group)
        return table.loc[mask].copy()

    selected = {}
    for model in MODEL_LABELS:
        data = bundle["models"][model]
        overall = select(data["overall"], grouped=False)
        groups = select(data["groups"])
        gaps = select(data["gaps_point"])
        ci = select(data["gaps_ci"])
        if len(overall) != 1 or len(gaps) != 1 or groups.empty:
            raise ValueError("選取的模型、政策或族群結果不完整")
        if len(ci) != 2 or set(ci["指標"]) != set(GAP_METRICS):
            raise ValueError("選取的差距信賴區間不完整")
        _check_numeric_columns(overall, ("AUC", "Precision", "Recall", "F1", "名單人數", "名單比例"))
        _check_numeric_columns(groups, tuple(k for k in GROUP_COLUMNS if k not in ("分組方式", "族群", "小樣本")))
        _check_numeric_columns(gaps, GAP_METRICS)
        _check_numeric_columns(ci, ("估計值", "下限", "上限", "有效重抽", "無效重抽"))
        if pd.isna(overall.iloc[0]["policy_id"]) or not str(overall.iloc[0]["policy_id"]).strip():
            raise ValueError("公平性結果缺少政策識別")
        if groups["小樣本"].isna().any() or not groups["小樣本"].isin([True, False]).all():
            raise ValueError("小樣本欄位格式不符")
        if ci["可靠"].isna().any() or not ci["可靠"].isin([True, False]).all():
            raise ValueError("差距區間可靠性欄位格式不符")
        selected[model] = {"overall": overall.iloc[0], "groups": groups,
                           "gaps": gaps.iloc[0], "ci": ci, "source": data["source"]}
    paired = select(bundle["paired"])
    if len(paired) != 2 or set(paired["指標"]) != set(GAP_METRICS):
        raise ValueError("選取的成對比較不完整")
    _check_numeric_columns(paired, ("估計值", "下限", "上限", "有效重抽", "無效重抽"))
    if paired["可靠"].isna().any() or not paired["可靠"].isin([True, False]).all():
        raise ValueError("成對比較區間可靠性欄位格式不符")
    return selected, paired


def _show_model_result(model, result):
    st.markdown(f"**{MODEL_LABELS[model]}**")
    source = result["source"]
    overall = result["overall"]
    st.caption(f"資料層級：{source['level']}｜model_id：{source['model_id']}")
    st.caption(f"run_id：{source['run_id']}")
    policy_id = overall["policy_id"]
    st.caption(f"policy_id：{policy_id if pd.notna(policy_id) else '未提供'}")
    st.metric("EO 差距（Recall 最大差距）", _format_rate(result["gaps"]["EO 差距"]))
    st.metric("FPR 差距", _format_rate(result["gaps"]["FPR 差距"]))
    st.dataframe(pd.DataFrame({
        "整體指標": ["AUC", "Precision", "Recall", "F1", "名單人數", "名單比例"],
        "數值": [_format_rate(overall[k]) for k in ("AUC", "Precision", "Recall", "F1")]
                + [_format_count(overall["名單人數"]), _format_rate(overall["名單比例"])],
    }), hide_index=True, width="stretch")
    st.markdown("**各族群錯誤率與 Wilson 95% 信賴區間**")
    _show_group_table(result["groups"])
    st.markdown("**差距的 Bootstrap 95% 信賴區間**")
    st.dataframe(_format_ci_table(result["ci"]), hide_index=True, width="stretch")
    if not result["ci"]["可靠"].eq(True).all():
        st.warning("部分差距區間的有效重抽不足，不顯示不可靠的區間。")


def _show_comparison_charts(selected):
    st.markdown("**兩模型差距比較**")
    # 僅重排來源點估計供繪圖，空值不補成 0。
    gap_chart = pd.DataFrame({model: [result["gaps"][m] for m in GAP_METRICS]
                              for model, result in selected.items()}, index=list(GAP_METRICS))
    if gap_chart.notna().any().any():
        st.bar_chart(gap_chart, stack=False)
    else:
        st.info("本分組沒有可繪製的差距數值。")
    st.caption("長條圖使用原始結果的數值，缺值不畫成 0。請搭配上方區間解讀。")
    for metric in ("Recall", "FPR"):
        st.markdown(f"**各族群 {metric} 比較**")
        chart = pd.concat({model: result["groups"].set_index("族群")[metric]
                           for model, result in selected.items()}, axis=1)
        if chart.notna().any().any():
            if metric == "Recall":
                # Use the same grouped-bar layout with an explicit 0–1 scale.
                recall_data = (
                    chart.rename_axis("group")
                    .reset_index()
                    .melt(id_vars="group", var_name="model", value_name="recall")
                )
                st.vega_lite_chart(
                    recall_data,
                    {
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "group", "type": "ordinal",
                                  "sort": list(chart.index)},
                            "xOffset": {"field": "model", "type": "nominal",
                                        "sort": ["M1", "M1-NS"]},
                            "y": {"field": "recall", "type": "quantitative",
                                  "scale": {"domain": [0, 1]},
                                  "axis": {"title": "Recall"}},
                            "color": {"field": "model", "type": "nominal",
                                      "scale": {"domain": ["M1", "M1-NS"]},
                                      "legend": {"title": None}},
                        },
                    },
                )
            else:
                st.bar_chart(chart, stack=False)
        else:
            st.info("本分組沒有可繪製的數值。")


def _show_model_comparison(group, platform_level="L1-H", platform_meta=None):
    """T-蔡2：只透過 C6 讀 F3 彙總結果，不計算指標或讀客戶資料。"""
    st.divider()
    st.subheader("含與不含敏感屬性模型比較")
    st.caption("RQ1｜F3 分析結果比較")
    method = st.selectbox("比較方式", FAIRNESS_COMPARISONS, key="fairness_comparison_method")
    st.caption(COMPARISON_METHODS[method])
    st.caption(f"目前分組：{group}（沿用上方分組方式）")
    try:
        bundle = load_fairness_comparison(platform_run=platform_meta)
        source = bundle["source"]
        expected_level = "L1" if platform_level == "L1-H" else "L2"
        if source["level"] != expected_level:
            raise ValueError(f"目前平台須搭配 {expected_level} 的 F3 比較結果，不能混用資料層級")
        if tuple(source["comparison_names"]) != FAIRNESS_COMPARISONS:
            raise ValueError("F3 比較名稱與共用介面不一致")
        selected, paired = _selected_comparison(bundle, method, group)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"尚未載入 F3 公平性比較結果：{exc}")
        st.caption("請確認 PLATFORM_FAIRNESS_RUN_ID 及已完成的 F3 結果，或回報組長。不使用其他結果替代。")
        return

    if source["level"] == "L1":
        st.info("本區使用 L1 訓練集 OOF 開發結果，非正式效能。上方是 L1-H 單模型開發監控，兩區資料來源不同。")
    else:
        st.caption("本區使用 L2 正式 F3 比較結果，與平台選定模型的單模型監控分開呈現。")
    st.caption(f"規則凍結紀錄：{source['rules_freeze_id']}｜來源：F3 的 results/<run_id>/ 結果")
    if method == "正式政策":
        st.caption("共用比較表提供各模型 policy_id，未提供實際門檻數值。門檻須回查各自政策來源，不以平台門檻代替。")
    a, b = st.columns(2)
    with a:
        _show_model_result("M1", selected["M1"])
    with b:
        _show_model_result("M1-NS", selected["M1-NS"])
    _show_comparison_charts(selected)
    st.markdown("**成對比較：M1-NS 減 M1**")
    st.dataframe(_format_ci_table(paired), hide_index=True, width="stretch")
    if not paired["可靠"].eq(True).all():
        st.warning("部分成對比較區間的有效重抽不足，請保守解讀。")
    st.caption("估計值為不含敏感屬性模型減去含敏感屬性模型的差距，負值表示差距較小，正值表示差距較大。區間包含 0 時，不能僅憑點估計認定差距有明確變化。")
    st.caption("本區不重算 F3 指標。差距縮小不代表符合公平性要求。")


def render(df, meta):
    """C6 平台入口：模擬模式不讀取政策或計算回測指標。"""
    if meta.get("mode") != BACKTEST:
        st.info("公平性監控僅在歷史回測模式顯示。模擬資料沒有實際標籤，無法計算族群錯誤率。")
        return
    st.header("公平性監控")
    if df.empty or "actual_default" not in df or not df["actual_default"].isin([0, 1]).all():
        st.info("尚無完整實際違約標籤，無法計算公平性回測指標。")
        return
    level = meta.get("level")
    if level not in ("L1-H", "L2"):
        st.info("目前沒有可顯示的公平性資料層級。")
        return

    if level == "L1-H":
        st.caption("歷史回測｜L1-H 開發資料，非正式結果，僅供介面與功能測試。")
        try:
            evaluation = _evaluate_backtest(df, meta)
        except Exception as exc:
            st.error(f"公平性資料驗證或評估失敗，請回報組長：{exc}")
            return
    else:
        st.caption("正式歷史回測（L2 測試集），單模型結果來自平台同一 run 的已存檔結果。")
        try:
            evaluation = load_platform_fairness(meta['run_id'])
            if any(evaluation['source'][key] != meta[key] for key in
                   ('run_id', 'model_id', 'policy_id', 'model_freeze_id', 'rules_freeze_id', 'summary_id')):
                raise ValueError("公平性單模型結果與平台來源不一致")
        except Exception as exc:
            st.error(f"正式公平性結果驗證失敗：{exc}")
            return

    _show_explanation()
    group = st.selectbox("分組方式", GROUPS)
    _show_results(evaluation["groups"], evaluation["gaps"], group)
    _show_model_comparison(group, level, meta)
