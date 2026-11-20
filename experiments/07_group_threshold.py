"""負責人：張佩芩｜分支：exp/group-threshold
RQ1 延伸（F3b）：參考 Hardt et al.（2016）的簡化做法，在 L1 以 M1-NS OOF 分數為基礎，
各族群各自搜尋門檻讓 Recall 接近單一正式政策的原始整體 Recall，並比較原始與最終名單表現。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from src.config import XGB_TUNED
from src.data import age_group
from src.frozen_methods import search_group_thresholds
from src.protocol import (
    apply_policy,
    bill_at,
    dev_data,
    evaluate,
    gap_ci,
    load_rules,
    make_model,
    oof_predict,
    prepare,
    start_run,
)

RULES_FREEZE_ID = "g2a-20261116"
MODEL_ID = "xgb-tuned-ns-s42train"
GROUPS = ["性別", "年齡層", "性別×年齡層"]

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


def make_partition(df: pd.DataFrame, group_name: str) -> pd.Series:
    sex = df["SEX"].map({1: "男性", 2: "女性"})
    age = age_group(df["AGE"]).astype(str)
    if group_name == "性別":
        part = sex
    elif group_name == "年齡層":
        part = age
    elif group_name == "性別×年齡層":
        part = sex + "×" + age
    else:
        raise ValueError(f"不支援的分組：{group_name}")
    part = part.rename(group_name)
    assert part.notna().all(), f"分組「{group_name}」存在空值"
    return part


def summarize_pred(y: pd.Series, pred: pd.Series, part: pd.Series, policy_label: str, stage_label: str,
                   cap_triggered: bool, raw_count: int, n_boot: int) -> tuple[dict, pd.DataFrame]:
    pred_s = pd.Series(np.asarray(pred, dtype=int), index=y.index, name="pred")
    y_arr = y.to_numpy(dtype=int)
    p_arr = pred_s.to_numpy(dtype=int)
    n_list = int(p_arr.sum())
    tp = int(((p_arr == 1) & (y_arr == 1)).sum())
    fn = int(((p_arr == 0) & (y_arr == 1)).sum())
    fp = int(((p_arr == 1) & (y_arr == 0)).sum())
    recall_overall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision_overall = tp / n_list if n_list > 0 else 0.0

    ci_df = gap_ci(y, pred_s, part, n_boot=n_boot)
    eo_row = ci_df[ci_df["指標"] == "EO 差距"]
    eo_gap = float(eo_row["估計值"].iloc[0]) if not eo_row.empty else np.nan
    eo_lo = float(eo_row["下限"].iloc[0]) if not eo_row.empty else np.nan
    eo_hi = float(eo_row["上限"].iloc[0]) if not eo_row.empty else np.nan

    per_rows = []
    for g in sorted(part.astype(str).unique()):
        m = (part.astype(str) == g).to_numpy()
        n_tot = int(m.sum())
        n_pos = int((m & (y_arr == 1)).sum())
        n_neg = int((m & (y_arr == 0)).sum())
        g_tp = int((m & (p_arr == 1) & (y_arr == 1)).sum())
        g_fp = int((m & (p_arr == 1) & (y_arr == 0)).sum())
        g_rec = g_tp / n_pos if n_pos > 0 else np.nan
        g_fpr = g_fp / n_neg if n_neg > 0 else np.nan
        small_sample = (n_pos < 93) or (n_neg < 93)
        per_rows.append({
            "政策": policy_label,
            "名單階段": stage_label,
            "族群": g,
            "總人數": n_tot,
            "違約人數": n_pos,
            "未違約人數": n_neg,
            "小樣本(<93)": small_sample,
            "名單人數": int((m & (p_arr == 1)).sum()),
            "誤報人數": g_fp,
            "Recall": g_rec,
            "FPR": g_fpr,
            "上限觸發註記": (
                "名單上限已觸發：截取後各族群 Recall 接近目標的設計不保證成立，以最終名單為準"
                if (cap_triggered and stage_label == "最終名單") else ""
            ),
        })

    per_df = pd.DataFrame(per_rows)
    summary_row = {
        "政策": policy_label,
        "名單階段": stage_label,
        "原始名單人數": int(raw_count),
        "最終名單人數": n_list,
        "誤報人數": fp,
        "整體 Recall": recall_overall,
        "整體 Precision": precision_overall,
        "EO 差距": eo_gap,
        "EO 差距 95% CI 下限": eo_lo,
        "EO 差距 95% CI 上限": eo_hi,
        "名單上限是否觸發": bool(cap_triggered),
        "截取後提醒": (
            "名單上限已觸發：各族群 Recall 接近的設計在截取後不保證成立，實際數字以最終名單為準"
            if (cap_triggered and stage_label == "最終名單") else ""
        ),
    }
    return summary_row, per_df


def run_one_group(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, fitted_model,
                  oof_score: pd.Series, bill: pd.Series, group_name: str, n_boot: int) -> None:
    rules = load_rules(RULES_FREEZE_ID)
    part = make_partition(df, group_name)
    res = search_group_thresholds(y, oof_score, bill, part, partition_name=group_name,
                                 rules_freeze_id=RULES_FREEZE_ID)
    base_pol = res["base_policy"]
    grp_pol = res["policy"]

    base_final, base_info = apply_policy(oof_score, base_pol, partition=part)
    grp_final, grp_info = apply_policy(oof_score, grp_pol, partition=part)

    # 確認 evaluate 可正常驗證（含名單上限與分組）
    _ = evaluate(y, score=oof_score, policy=grp_pol, partition=part,
                 groups={group_name: part}, bill=bill, scenario=rules.scenario)

    sum_rows, detail_dfs = [], []
    for pol_label, final_pred, info in [
        ("單一正式政策", base_final, base_info),
        ("族群門檻政策", grp_final, grp_info),
    ]:
        for stage_label, pred_series in [
            ("原始名單", info["pred_raw"]),
            ("最終名單", final_pred),
        ]:
            s_row, d_df = summarize_pred(
                y=y,
                pred=pred_series,
                part=part,
                policy_label=pol_label,
                stage_label=stage_label,
                cap_triggered=bool(info["名單上限是否觸發"]),
                raw_count=int(info["原始名單人數"]),
                n_boot=n_boot,
            )
            sum_rows.append(s_row)
            detail_dfs.append(d_df)

    summary_df = pd.DataFrame(sum_rows)
    detail_df = pd.concat(detail_dfs, ignore_index=True)
    cand_df = res["candidates"].copy()
    cand_df["t0"] = res["t0"]
    cand_df["target_recall"] = res["target_recall"]

    run = start_run(
        "L1",
        f"T-張4 F3b 族群門檻比較（分組：{group_name}，模型：{MODEL_ID}）",
        model_id=MODEL_ID,
        rules_freeze_id=RULES_FREEZE_ID,
    )
    run.attach_model(fitted_model, X, y, oof_score)
    slug_map = {"性別": "sex", "年齡層": "age", "性別×年齡層": "sex_age"}
    suffix = slug_map.get(group_name, group_name)

    run.save_csv(f"group_threshold_{suffix}_candidates", cand_df)
    run.save_csv(f"group_threshold_{suffix}_summary", summary_df)
    run.save_csv(f"group_threshold_{suffix}_groups", detail_df)

    # 畫圖：最終名單各族群 Recall 比較（單一正式政策 vs 族群門檻政策）
    plot_df = detail_df[detail_df["名單階段"] == "最終名單"].pivot(
        index="族群", columns="政策", values="Recall"
    )
    cols_order = [c for c in ["單一正式政策", "族群門檻政策"] if c in plot_df.columns]
    plot_df = plot_df[cols_order]

    fig, ax = plt.subplots(figsize=(max(6.5, len(plot_df) * 1.15), 4.8))
    plot_df.plot(kind="bar", ax=ax, rot=30)
    ax.set_ylabel("Recall（違約者被納入名單比例）")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"各族群 Recall 比較（最終名單｜分組：{group_name}）")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    fig.tight_layout()
    run.save_fig(f"group_threshold_{suffix}_recall", fig)
    plt.close(fig)

    run.finish()
    print(f"\n=== 分組：{group_name} 完成（t0={res['t0']:.4f}, target_recall={res['target_recall']:.4f}）===")
    print("選定族群門檻：", grp_pol.thresholds)
    print(summary_df.to_string(index=False))
    if bool(grp_info["名單上限是否觸發"]):
        print("注意：族群門檻已觸發名單上限；各族群 Recall 接近目標的設計在截取後不保證成立，實際數字以最終名單為準。")


def main() -> None:
    parser = argparse.ArgumentParser(description="T-張4 F3b 族群門檻研究比較（L1）")
    parser.add_argument("--group", choices=GROUPS, default=None, help="指定單一分組；未指定則依序跑全部三種分組")
    parser.add_argument("--n-boot", type=int, default=2000, help="EO 差距 bootstrap 重抽次數（預設 2000）")
    args = parser.parse_args()

    rules = load_rules(RULES_FREEZE_ID)
    df = dev_data()
    y = df["default"]
    bill = bill_at(df, 0)
    X, _ = prepare(df, lag=0, sensitive=False, engineered=False)
    cv = dict(folds=rules.cv["n_splits"], seed=rules.cv["random_state"])
    oof_score = oof_predict(lambda: make_model("XGBoost", params=dict(XGB_TUNED)), X, y, **cv)
    fitted_model = make_model("XGBoost", params=dict(XGB_TUNED)).fit(X, y)

    target_groups = [args.group] if args.group else GROUPS
    for g in target_groups:
        run_one_group(df, X, y, fitted_model, oof_score, bill, g, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
