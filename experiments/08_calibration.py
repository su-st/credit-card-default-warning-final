"""負責人：余旻恩｜分支：exp/calibration
F7 機率校準評估（資料層級 L1：24,000 筆訓練集，seed 42，完全不碰測試集）。

模型的風險分數是不是接近真正的違約比例？XGBoost 用 scale_pos_weight 加重違約樣本，
輸出的分數通常比實際違約率偏高。平台以「風險分數」與排名呈現，這支程式只做研究評估：
校準後分數不取代平台的風險分數。

做的事：
1. 用 M1-NS（XGB_TUNED、不含敏感屬性）取得 24,000 筆的 OOF 風險分數
2. crossfit_calibrate：5 折交叉擬合 isotonic 校準，每折用其他 4 折擬合校準器、再套用到該折，
   擬合資料與評估資料分開，並記錄每折校準器的來源（calibration_folds.csv）
3. calibration_report：比較校準前後的 Brier 分數與校準曲線，同時列 AUC 與分數的不同值個數
4. rank_change：isotonic 校準會產生同分，檢查校準後各名單比例與原本名單的重疊比例
5. 這是 L1 的開發評估，正式數字以凍結後的 L2 為準

用法：
    python experiments/08_calibration.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.calibration import calibration_report, crossfit_calibrate, rank_change
from src.training_time import timed_call, execution_environment
from src.config import XGB_TUNED
from src.protocol import dev_data, make_model, oof_predict, prepare, start_run

SEED = 42
FOLDS = 5
BINS = 10
RATES = [0.05, 0.10, 0.20, 0.30, 0.50]  # 名單比例；0.20 是平台使用的前 20%

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

def run_analysis(df):
    """原L1校準分析，新增計時，資料不在import時讀取。"""
    y = df["default"]
    X, cols = prepare(df, lag=0, sensitive=False, engineered=False)  # M1-NS：不含敏感屬性
    print(f"資料層級 L1，{len(df):,} 筆，違約率 {y.mean():.1%}，特徵數 {len(cols)}")
    print("L1 開發評估；正式數字以凍結後的 L2 為準，校準後分數不取代平台的風險分數")

    run = start_run("L1", "F7 校準評估：M1-NS 的 OOF 風險分數，交叉擬合 isotonic 校準前後比較",
                    model_id="xgb-tuned-ns-s42train")
    print(f"run_id = {run.run_id}")

    # ---------- 1. M1-NS 的 OOF 風險分數 ----------
    factory = lambda: make_model("XGBoost", params=dict(XGB_TUNED))
    oof, oof_seconds = timed_call(oof_predict, factory, X, y, folds=FOLDS, seed=SEED)

    # ---------- 2. 交叉擬合校準：擬合與評估用不同的折 ----------
    (calibrated, provenance), crossfit_seconds = timed_call(crossfit_calibrate, y, oof, folds=FOLDS, seed=SEED)
    run.save_csv("calibration_folds", provenance)  # 規格 4.5.7：記錄每折校準器的來源

    # ---------- 3. 校準前後比較 ----------
    versions = {"原始風險分數": oof, "校準後分數（交叉擬合 isotonic）": calibrated}
    summary_rows, curves = [], []
    for name, score in versions.items():
        report = calibration_report(y, score, bins=BINS)
        summary_rows.append({"版本": name, "AUC": roc_auc_score(y, score),
                             "Brier 分數（越低越好）": report["Brier分數"],
                             "平均分數": score.mean(), "實際違約率": y.mean(),
                             "不同分數的個數": score.nunique()})
        curve = report["校準曲線"].copy()
        curve["分數區間"] = curve["分數區間"].astype(str)
        curve.insert(0, "版本", name)
        curves.append(curve)
    summary = pd.DataFrame(summary_rows)
    bins = pd.concat(curves, ignore_index=True)
    run.save_csv("calibration_report", summary.round(4))
    run.save_csv("calibration_bins", bins.round(4))
    print(summary.round(4).to_string(index=False))
    print("\n分組校準表（平均預測值和實際違約率越接近，分數越接近實際違約比例）：")
    print(bins.round(3).to_string(index=False))

    # ---------- 4. 排名變化：校準後的前 N% 名單與原本的重疊比例 ----------
    ranks = pd.DataFrame([{"名單比例": rate, "與原始名單的重疊比例": rank_change(oof, calibrated, rate)}
                          for rate in RATES])
    run.save_csv("rank_change", ranks.round(4))
    print("\n校準後名單與原始名單的重疊比例（isotonic 同分時，同分者依客戶編號排序）：")
    print(ranks.round(4).to_string(index=False))

    # ---------- 5. 校準曲線圖 ----------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="完美校準")
    for name, curve in zip(versions, curves):
        ax.plot(curve["平均預測值"], curve["實際違約率"], marker="o", label=name)
    ax.set_xlabel("平均分數")
    ax.set_ylabel("實際違約率")
    ax.set_title("校準曲線（M1-NS，L1 的 OOF 分數）")
    ax.legend()
    fig.tight_layout()
    run.save_fig("calibration_curve", fig)

    # ---------- 6. run 需要綁定一個模型才能完成：保存 M1-NS 的訓練紀錄 ----------
    model, fit_seconds = timed_call(make_model("XGBoost", params=dict(XGB_TUNED)).fit, X, y)
    run.update(training_timing=dict(fit_seconds=fit_seconds, oof_seconds=oof_seconds,
                                   crossfit_seconds=crossfit_seconds, environment=execution_environment()))
    run.save_csv("training_timing", pd.DataFrame([dict(fit_seconds=fit_seconds, oof_seconds=oof_seconds, crossfit_seconds=crossfit_seconds)]))
    run.attach_model(model, X, y, oof)
    run.finish()
    print(f"完成。結果在 results/{run.run_id}/")

    return dict(run=run, summary=summary, bins=bins, ranks=ranks, calibrated=calibrated)


def main():
    return run_analysis(dev_data())


if __name__ == "__main__":
    main()
