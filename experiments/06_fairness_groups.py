"""負責人：張佩芩｜分支：exp/fairness-groups
RQ1（F3）：M1（含敏感屬性）與 M1-NS（不含）在各族群的漏判率、誤報率與差距。
資料層級：L1（dev_data，24,000 筆訓練集），不讀測試集。

依照組長最新指示：
1. M1 與 M1-NS 各自獨立跑一個 run，模型 ID 直接使用 00 階段印出的原生 model_id。
2. 族群表各自存放在自己的 run 資料夾中。
3. 兩模型的成對比較（gap_ci）統一存在 M1-NS 的 run 中，並在 CSV 額外增加欄位記錄 M1 的 run_id 與 model_id。

用法：
    python experiments/06_fairness_groups.py [n_boot]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import XGB_TUNED
from src.data import age_group
from src.protocol import (Policy, apply_policy, bill_at, choose_threshold, dev_data,
                          evaluate, gap_ci, load_rules, make_model, oof_predict,
                          prepare, start_run)

RULES_FREEZE_ID = "g2a-20261116"
COMPARISONS = ["共同門檻0.5", "固定名單前20%", "正式政策"]
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000   # 預設 2000 次正式版重抽

# 組長要求：模型不用自己取名，直接使用跑 00_build_dev_base 時印出的 M1 和 M1-NS 原生 model_id
# 請確保這裡的 ID 與你當時 00 階段印出的一致
MODEL_IDS = {
    "M1": "xgb-tuned-s42train",       # M1 原生 model_id（含敏感屬性）
    "M1-NS": "xgb-tuned-ns-s42train"  # 不含敏感屬性
}


def factory():
    return make_model("XGBoost", params=dict(XGB_TUNED))


def make_partitions(df: pd.DataFrame) -> dict:
    """每種分組都必須互斥且完整，每次呼叫只用一種（規格 §4.5.6）。"""
    sex = df["SEX"].map({1: "男性", 2: "女性"})
    age = age_group(df["AGE"]).astype(str)
    # 教育 0、5、6 與婚姻 0 併入「其他」，要與期中的處理一致，並寫進報告
    edu = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4).map(
        {1: "研究所", 2: "大學", 3: "高中", 4: "其他"})
    mar = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3).map(
        {1: "已婚", 2: "單身", 3: "其他"})
    parts = {"性別": sex, "年齡層": age, "性別×年齡層": sex + "×" + age,
             "教育程度": edu, "婚姻狀況": mar}
    for name, p in parts.items():
        assert p.notna().all(), f"分組「{name}」有空值，分組必須完整"
    return parts


import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

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


import re


def _safe_slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_-")
    return slug if slug else fallback


def plot_group_metrics_with_ci(run, groups_df: pd.DataFrame, mname: str) -> None:
    if groups_df is None or groups_df.empty:
        return
    policies = groups_df["政策比較"].dropna().unique().tolist() if "政策比較" in groups_df.columns else [None]
    specs = [
        ("Recall", "Recall 下限", "Recall 上限", "Recall", "fairness_groups_recall_ci"),
        ("FPR", "FPR 下限", "FPR 上限", "FPR（誤報率）", "fairness_groups_fpr_ci"),
    ]
    for idx, policy in enumerate(policies):
        sub = groups_df[groups_df["政策比較"] == policy].copy() if policy is not None else groups_df.copy()
        if sub.empty:
            continue
        policy_code_map = {
            "共同門檻0.5": "s",
            "固定名單前20%": "n",
            "正式政策": "c",
        }
        if policy is None:
            suffix = ""
        else:
            code = policy_code_map.get(str(policy)) or _safe_slug(policy, f"p{idx}")
            suffix = f"_{code}"
        title_p = f"（{policy}）" if policy is not None else ""
        labels = (
            sub["分組方式"].astype(str) + "｜" + sub["族群"].astype(str)
            if {"分組方式", "族群"}.issubset(sub.columns)
            else sub.iloc[:, 0].astype(str)
        )
        for col, lo_c, hi_c, xlabel, base in specs:
            if col not in sub.columns:
                continue
            vals = sub[col].to_numpy(dtype=float)
            y_pos = np.arange(len(sub))
            fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(sub) * 0.32 + 1.5)))
            if lo_c in sub.columns and hi_c in sub.columns:
                lo_e = np.clip(vals - sub[lo_c].to_numpy(dtype=float), 0.0, None)
                hi_e = np.clip(sub[hi_c].to_numpy(dtype=float) - vals, 0.0, None)
                ax.errorbar(vals, y_pos, xerr=[lo_e, hi_e], fmt="o", capsize=3, markersize=4.5, label="95% 信賴區間")
            else:
                ax.scatter(vals, y_pos, s=25, label="點估計")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{mname}{title_p} - 各族群 {xlabel} 附信賴區間")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            run.save_fig(f"{base}{suffix}", fig)
            plt.close(fig)

def run_single_model(mname: str, sensitive: bool, df: pd.DataFrame, bill, scenario, parts, y) -> dict:
    """單獨執行一個模型的完整評估流程，並回傳相關結果供後續使用。"""
    print(f"\n--- 開始執行模型 {mname} (sensitive={sensitive}) ---")
    X, _ = prepare(df, lag=0, sensitive=sensitive, engineered=False)
    score = oof_predict(factory, X, y)
    
    # 選擇門檻
    official = choose_threshold(y, score, bill, scenario)["政策物件"]
    pols = {
        "共同門檻0.5": Policy.single(0.5, None),
        "固定名單前20%": Policy.top_n(0.2),
        "正式政策": official
    }

    overall_rows, group_rows, gap_rows = [], [], []
    preds = {}
    for cname in COMPARISONS:
        policy = pols[cname]
        tag = {"模型": mname, "政策比較": cname}
        out = evaluate(y, score=score, policy=policy, groups=parts,
                       bill=bill, scenario=scenario)
        overall_rows.append(out["overall"].assign(**tag))
        group_rows.append(out["groups"].assign(**tag))
        gap_rows.append(out["gaps"].assign(**tag))
        pred, _ = apply_policy(score, policy)
        preds[cname] = pd.Series(pred, index=y.index).astype(int)

    # 自助法信賴區間（單模型各族群指標差距）
    ci_rows = []
    for cname in COMPARISONS:
        for pname, part in parts.items():
            ci = gap_ci(y, preds[cname], part, n_boot=N_BOOT)
            ci_rows.append(ci.assign(模型=mname, 政策比較=cname, 分組方式=pname))

    return {
        "model": factory().fit(X, y),
        "X": X,
        "score": score,
        "overall": pd.concat(overall_rows, ignore_index=True),
        "groups": pd.concat(group_rows, ignore_index=True),
        "gaps_point": pd.concat(gap_rows, ignore_index=True),
        "gaps_ci": pd.concat(ci_rows, ignore_index=True),
        "preds": preds
    }


if __name__ == "__main__":
    data = dev_data()
    y = data["default"]
    scenario = load_rules(RULES_FREEZE_ID).scenario
    bill = bill_at(data, 0)
    parts = make_partitions(data)

    # 1. 跑 M1 模型（含敏感屬性）
    m1_res = run_single_model("M1", sensitive=True, df=data, bill=bill, scenario=scenario, parts=parts, y=y)
    run_m1 = start_run("L1", "F3 族群公平性：M1 模型獨立評估", model_id=MODEL_IDS["M1"], rules_freeze_id=RULES_FREEZE_ID)
    print(f"M1 run_id = {run_m1.run_id}")
    run_m1.attach_model(m1_res["model"], m1_res["X"], y, m1_res["score"])
    run_m1.save_csv("fairness_overall", m1_res["overall"])
    run_m1.save_csv("fairness_groups", m1_res["groups"])
    plot_group_metrics_with_ci(run_m1, m1_res["groups"], "M1")
    run_m1.save_csv("fairness_gaps_point", m1_res["gaps_point"])
    run_m1.save_csv("fairness_gaps", m1_res["gaps_ci"])
    run_m1.finish()
    print(f"M1 完成，結果在 results/{run_m1.run_id}/")

    # 2. 跑 M1-NS 模型（不含敏感屬性），並計算與 M1 的成對比較（存在 M1-NS 的 run 中）
    m1ns_res = run_single_model("M1-NS", sensitive=False, df=data, bill=bill, scenario=scenario, parts=parts, y=y)
    
    # 計算 M1-NS 相對 M1 的成對比較（gap_ci 用 pred_b），並加入 M1 溯源欄位
    pair_ci_dfs = []
    for cname in COMPARISONS:
        for pname, part in parts.items():
            diff = gap_ci(y, m1_res["preds"][cname], part, n_boot=N_BOOT,
                          pred_b=m1ns_res["preds"][cname])
            diff_assigned = diff.assign(
                模型="M1-NS 減 M1", 
                政策比較=cname, 
                分組方式=pname,
                m1_run_id=run_m1.run_id,
                m1_model_id=MODEL_IDS["M1"]
            )
            pair_ci_dfs.append(diff_assigned)
    
    # 合併 M1-NS 自身的信賴區間與成對比較
    pair_ci_combined = pd.concat(pair_ci_dfs, ignore_index=True)
    final_m1ns_gaps = pd.concat([m1ns_res["gaps_ci"], pair_ci_combined], ignore_index=True)

    run_m1ns = start_run("L1", "F3 族群公平性：M1-NS 模型評估與 M1 成對比較", model_id=MODEL_IDS["M1-NS"], rules_freeze_id=RULES_FREEZE_ID)
    print(f"M1-NS run_id = {run_m1ns.run_id}")
    run_m1ns.attach_model(m1ns_res["model"], m1ns_res["X"], y, m1ns_res["score"])
    run_m1ns.save_csv("fairness_overall", m1ns_res["overall"])
    run_m1ns.save_csv("fairness_groups", m1ns_res["groups"])
    plot_group_metrics_with_ci(run_m1ns, m1ns_res["groups"], "M1-NS")
    run_m1ns.save_csv("fairness_gaps_point", m1ns_res["gaps_point"])
    run_m1ns.save_csv("fairness_gaps", final_m1ns_gaps)   # 包含自身區間與帶有 M1 溯源的成對比較
    run_m1ns.finish()
    print(f"M1-NS 完成，結果在 results/{run_m1ns.run_id}/")
    print("\n全部執行完畢！請記錄 M1 與 M1-NS 的兩個 run_id 供後續分析與報告使用。")