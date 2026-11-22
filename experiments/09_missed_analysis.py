"""負責人：陳品婕｜分支：exp/missed-analysis
漏判分析（T-陳3 / F6）：模型最大的弱點是漏判「近六個月沒有延遲紀錄」的違約者。
這支程式（1）描述漏判者與其他客戶的差異，（2）比較方法 A～E 能否多抓到這群人。

資料：L1（dev_data() 的 24,000 筆訓練集），全部用 OOF，所有方法使用相同資料與折。
公式、分群與門檻搜尋全部呼叫 src/frozen_methods.py 的共用函式，不在這裡重寫，也不先分群再分折；
不讀 outputs/threshold.json，門檻由 choose_threshold 在各方法自己的 OOF 上產生。
  A 基準：M1-NS（原始 19 欄）
  B 加入繳款狀態次數（prepare_missed_features method='B'）
  C 再加入金額變化（method='C'）
  D 分群建模（method='D' + make_grouped_model）；兩群分數直接合併、未校準
  E 無延遲群門檻搜尋（search_no_delay_threshold，沿用 M1-NS 分數；不參加模型替換、不直接套平台）
三種比較分開呈現：共同門檻 0.5、固定名單前 20%、正式政策。

輸出（results/<run_id>/）：missed_profile.csv、missed_summary.csv、missed_methods_compare.csv、
    missed_E_candidates.csv、missed_E_cost.png
執行（在專案根目錄）：python experiments/09_missed_analysis.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.features import add_features
from src.frozen_methods import (
    delay_partition,
    make_grouped_model,
    prepare_missed_features,
    search_no_delay_threshold,
)
from src.protocol import (
    Policy,
    apply_policy,
    bill_at,
    choose_threshold,
    cv_evaluate,
    dev_data,
    evaluate,
    load_rules,
    make_model,
    oof_predict,
    prepare,
    start_run,
)

try:
    from src.data import age_group
except ImportError:
    raise SystemExit(
        "找不到 src.data.age_group：請問組長年齡層的共用函式在哪裡，不要自己定義分組。"
    )

FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
font_name = None
for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        try:
            font_manager.fontManager.addfont(fp)
            font_name = font_manager.FontProperties(fname=fp).get_name()
            break
        except Exception as e:
            print(f"提醒：字型 {fp} 載入失敗（{e}），改試下一個")
if font_name:
    plt.rcParams["font.family"] = font_name
else:
    print("提醒：找不到中文字型，改用預設字型，圖上的中文可能顯示成方塊（數據不受影響）")
plt.rcParams["axes.unicode_minus"] = False

RULES_FREEZE_ID = "g2a-20261116"
NO_DELAY, HAS_DELAY = "無延遲", "有延遲"

# 本實驗以 A（M1-NS）作為主要註冊模型；B～D 是同一實驗中的比較方法。
# 使用獨立 ID，避免覆寫或混淆先前已註冊的正式 M1-NS。
RUN_MODEL_ID = "xgb-missed-analysis-a"

FEATS = {
    "N_DELAY": "近六個月延遲次數",
    "MAX_DELAY": "最長延遲月數",
    "UTIL_1": "9 月額度使用率",
    "UTIL_TREND": "使用率變化（9 月 − 4 月）",
    "PAYRATIO_1": "9 月還款比例",
    "LIMIT_BAL": "信用額度",
    "BILL_AMT1": "9 月帳單金額",
    "PAY_AMT1": "9 月繳款金額",
    "AGE": "年齡",
}


def flatten(prefix, obj):
    """把 evaluate／cv_evaluate 回傳的資料攤成一列欄位。"""
    if obj is None:
        return {}
    if isinstance(obj, pd.DataFrame):
        return {
            f"{prefix}|{i}|{c}": v
            for i, r in obj.iterrows()
            for c, v in r.items()
        }
    if isinstance(obj, pd.Series):
        return {f"{prefix}|{k}": v for k, v in obj.items()}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(f"{prefix}|{k}", v))
        return out
    return {prefix: obj}


def as_series(x, index):
    # 若 x 原本是 Series，直接回傳以保留 OOF attrs（fold_assignment 等）。
    return x if isinstance(x, pd.Series) else pd.Series(np.asarray(x), index=index)


def as_flag(final_list, index):
    """將 apply_policy 回傳的 0/1 預測 Series 轉成布林 Series。"""
    if isinstance(final_list, pd.Series):
        if not final_list.index.equals(index):
            raise ValueError("apply_policy 回傳的索引與原始資料不一致")
        values = final_list.to_numpy()
    else:
        values = np.asarray(final_list)
        if values.ndim != 1 or len(values) != len(index):
            raise ValueError("apply_policy 回傳長度或維度不符")

    unique_values = set(pd.unique(values).tolist())
    if unique_values.issubset({False, True, 0, 1}):
        return pd.Series(values.astype(bool), index=index)

    raise TypeError("apply_policy 回傳值不是布林或 0/1 預測結果")


def profile(d, prob, delay):
    s = d[list(FEATS)].median().rename(FEATS)
    s["人數"] = len(d)
    s["近六個月無延遲比例"] = (delay.loc[d.index] == NO_DELAY).mean()
    s["模型分數平均"] = prob.loc[d.index].mean()
    return s


def main():
    rules = load_rules(RULES_FREEZE_ID)
    cv = dict(folds=rules.cv["n_splits"], seed=rules.cv["random_state"])
    df = dev_data()
    y = df["default"]
    bill = bill_at(df, 0)
    groups = {
        "性別": df["SEX"].map({1: "男性", 2: "女性"}),
        "年齡層": age_group(df["AGE"]).astype(str),
    }
    partition = delay_partition(df)
    nodelay = partition == NO_DELAY
    nodelay_default = nodelay & (y == 1)

    run = start_run(
        "L1",
        "09 漏判分析：漏判者特徵與方法 A～E 的 L1 比較",
        model_id=RUN_MODEL_ID,
        rules_freeze_id=RULES_FREEZE_ID,
    )

    # ---------- 方法 A：M1-NS，並取得 E 的搜尋結果 ----------
    print("[A] M1-NS OOF…", flush=True)
    X_ns, _ = prepare(df, lag=0, sensitive=False, engineered=False)
    factory = lambda: make_model("XGBoost", params=rules.dev_base["params"])
    oof_ns = as_series(oof_predict(factory, X_ns, y, **cv), df.index)

    # 註冊本次實驗的主要模型 A。oof_ns 保留 oof_predict 的 fold_assignment attrs。
    print("[A] 註冊主要模型與 OOF 證據…", flush=True)
    model_A = factory()
    model_A.fit(X_ns, y)
    run.attach_model(model_A, X_ns, y, oof_ns, seed=cv["seed"])

    print("[E] 無延遲群門檻搜尋…", flush=True)
    result_E = search_no_delay_threshold(df, oof_ns)
    base_policy, t0 = result_E["base_policy"], result_E["t0"]

    # ---------- 分析一、二：漏判者特徵 ----------
    caught = as_flag(apply_policy(oof_ns, base_policy)[0], df.index)
    feat = add_features(df).assign(score=oof_ns)
    bad = feat[y == 1]
    a1 = pd.DataFrame(
        {
            "成功辨識的違約者": profile(
                bad[caught[bad.index]], oof_ns, partition
            ),
            "漏判的違約者": profile(
                bad[~caught[bad.index]], oof_ns, partition
            ),
        }
    )
    clean = feat[nodelay]
    a2 = pd.DataFrame(
        {
            "無延遲且違約": profile(
                clean[y[clean.index] == 1], oof_ns, partition
            ),
            "無延遲且未違約": profile(
                clean[y[clean.index] == 0], oof_ns, partition
            ),
        }
    )
    a2["差異比例"] = (
        a2["無延遲且違約"] - a2["無延遲且未違約"]
    ) / a2["無延遲且未違約"].abs().replace(0, np.nan)
    a2.loc[["人數", "近六個月無延遲比例"], "差異比例"] = np.nan
    profile_table = pd.concat(
        {
            "分析一：成功辨識 vs 漏判": a1,
            "分析二：無延遲客戶，違約 vs 未違約": a2,
        },
        names=["分析", "特徵"],
    ).reset_index()
    run.save_csv("missed_profile.csv", profile_table)

    missed = bad[~caught[bad.index]]
    missed_nodelay = int((partition.loc[missed.index] == NO_DELAY).sum())
    summary = pd.DataFrame(
        [
            {
                "基準門檻 t0（來源：choose_threshold 於 M1-NS OOF）": t0,
                "違約者人數": len(bad),
                "漏判人數": len(missed),
                "漏判者中近六個月無延遲": missed_nodelay,
                "漏判者中無延遲比例": missed_nodelay / max(len(missed), 1),
                "無延遲違約者人數": int(nodelay_default.sum()),
                "無延遲違約者被辨識比例（基準）": (
                    (caught & nodelay_default).sum()
                    / max(int(nodelay_default.sum()), 1)
                ),
            }
        ]
    )
    run.save_csv("missed_summary.csv", summary)

    # ---------- 分析三：方法 A～D ----------
    X_B, _ = prepare_missed_features(df, method="B")
    X_C, _ = prepare_missed_features(df, method="C")
    X_D, _ = prepare_missed_features(df, method="D")
    methods = {
        "A 基準 M1-NS": (X_ns, factory),
        "B 加入繳款狀態次數": (X_B, factory),
        "C 再加入金額變化": (X_C, factory),
        "D 分群建模": (X_D, make_grouped_model),
    }
    notes = {
        "D 分群建模": "兩群分數直接合併、未校準；前 20% 名單可能偏向某一群"
    }

    rows = []

    def add_row(name, pname, source, out, flag, cv_res=None, note=""):
        row = {
            "方法": name,
            "比較": pname,
            "門檻來源": source,
            "備註": note,
            "名單人數（自算，對照用）": int(flag.sum()),
            "無延遲違約者被辨識比例（自算，描述用）": (
                (flag & nodelay_default).sum()
                / max(int(nodelay_default.sum()), 1)
            ),
        }
        row.update(flatten("CV", cv_res))
        row.update(flatten("整體", out["overall"]))
        row.update(flatten("族群", out["groups"]))
        row.update(flatten("EO差距", out["gaps"]))
        rows.append(row)

    for name, (X, fac) in methods.items():
        print(f"[{name}] 交叉驗證與 OOF 分數…", flush=True)
        cv_res = cv_evaluate(fac, X, y, **cv)
        score = (
            oof_ns
            if name.startswith("A")
            else as_series(oof_predict(fac, X, y, **cv), df.index)
        )
        formal = choose_threshold(y, score, bill, rules.scenario)["政策物件"]
        policies = {
            "共同門檻 0.5": (
                "Policy.single(0.5, None)",
                Policy.single(0.5, None),
            ),
            "固定名單前 20%": (
                "Policy.top_n(0.2)",
                Policy.top_n(0.2),
            ),
            "正式政策": (
                "choose_threshold 在同一方法的 OOF 上選出",
                formal,
            ),
        }
        for pname, (source, policy) in policies.items():
            out = evaluate(
                y,
                score=score,
                policy=policy,
                groups=groups,
                bill=bill,
                scenario=rules.scenario,
            )
            flag = as_flag(apply_policy(score, policy)[0], df.index)
            add_row(
                name,
                pname,
                source,
                out,
                flag,
                cv_res,
                notes.get(name, ""),
            )

    # ---------- 方法 E：沿用 M1-NS 分數，不參加模型替換 ----------
    policy_E = result_E["policy"]
    out_E = evaluate(
        y,
        score=oof_ns,
        policy=policy_E,
        partition=partition,
        groups=groups,
        bill=bill,
        scenario=rules.scenario,
    )
    flag_E = as_flag(
        apply_policy(oof_ns, policy_E, partition=partition)[0], df.index
    )
    add_row(
        "E 無延遲群門檻",
        "正式政策（E）",
        "search_no_delay_threshold（t0 來自 M1-NS OOF 的 choose_threshold）",
        out_E,
        flag_E,
        note=(
            "E 不產生新分數，不參加模型替換，不直接套用平台；"
            "名單人數與基準不同，不可當成模型改善"
        ),
    )

    compare = pd.DataFrame(rows)
    run.save_csv("missed_methods_compare.csv", compare)
    # ---------- 方法 A～E：無延遲違約者被辨識比例比較 ----------
    metric = "無延遲違約者被辨識比例（自算，描述用）"

    categories = [
        "共同門檻 0.5",
        "固定名單前 20%",
        "正式政策",
    ]

    method_order = [
        "A 基準 M1-NS",
        "B 加入繳款狀態次數",
        "C 再加入金額變化",
        "D 分群建模",
        "E 無延遲群門檻",
    ]

    chart_data = compare.copy()
    chart_data["圖表比較方式"] = chart_data["比較"].replace(
        {"正式政策（E）": "正式政策"}
    )

    chart_data = (
        chart_data.pivot(
            index="方法",
            columns="圖表比較方式",
            values=metric,
        )
        .reindex(index=method_order, columns=categories)
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(method_order))
    width = 0.25

    for i, category in enumerate(categories):
        values = chart_data[category].to_numpy(dtype=float) * 100

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=category,
        )

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{value:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(method_order)
    ax.set_ylabel("無延遲違約者被辨識比例（%）")
    ax.set_xlabel("比較方法")
    ax.set_title("方法 A～E：無延遲違約者被辨識比例比較")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    max_value = np.nanmax(chart_data.to_numpy(dtype=float)) * 100
    ax.set_ylim(0, max_value * 1.18)

    fig.tight_layout()
    run.save_fig("missed_methods_compare.png", fig)
    plt.close(fig)
    cand = result_E["candidates"].copy()
    # 依 final_cost 最低選擇；成本相同時，選較高的無延遲群門檻。
    best_index = cand.sort_values(
        ["final_cost", "threshold"],
        ascending=[True, False],
        kind="stable",
    ).index[0]
    cand["selected"] = False
    cand.loc[best_index, "selected"] = True
    run.save_csv("missed_E_candidates.csv", cand)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        cand["threshold"],
        cand["raw_cost"],
        marker="o",
        label="raw_cost（未截上限）",
    )
    ax.plot(
        cand["threshold"],
        cand["final_cost"],
        marker="s",
        label="final_cost（截上限後，選擇依據）",
    )
    ax.set_xlabel("無延遲群門檻")
    ax.set_ylabel("成本")
    ax.set_title("E：無延遲群門檻搜尋（L1，M1-NS 分數）")
    ax.legend()
    fig.tight_layout()
    run.save_fig("missed_E_cost.png", fig)
    plt.close(fig)

    run.finish()

    print(
        f"\nrun_id：{getattr(run, 'run_id', '（請看 results/ 底下最新的資料夾）')}"
    )
    print(summary.T.to_string(header=False))
    print(compare.dropna(axis=1, how="all").T.to_string())


if __name__ == "__main__":
    main()
