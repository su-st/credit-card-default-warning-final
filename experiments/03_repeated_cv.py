"""負責人：余旻恩｜分支：exp/repeated-cv
F5 RQ3 提早預警與涵蓋率（資料層級 L1：24,000 筆訓練集，seed 42，完全不碰測試集），
以及 F8 穩定性分析所需的「單一切分評估」函式（L3 的 10 個 seed 由 Codex 在凍結後呼叫）。

做的事：
1. RQ3：對 lag=0～4（提早 1～5 個月）分別取得 M1-NS 的特徵（不含敏感屬性），
   用 cv_evaluate 算 5 折 CV AUC，用 oof_predict 加 coverage_curve 算前 5%～50% 名單的涵蓋率。
   lag=k 時只使用第 k+1～6 個月的資料，特徵不含最近 k 個月。只比較 AUC 與固定比例名單的涵蓋率，不套用成本門檻。
2. evaluate_one_split：把「訓練、選門檻、評估」寫成一個函式，輸入是訓練資料與評估資料。
   L1 時 evaluation=None，直接在訓練資料的 OOF 分數上評估；L3 時由 outer_splits 提供每個 seed 的訓練與測試資料。
   固定 0.5 與前 20% 的政策不重新選擇，只有正式政策在該次訓練資料的 OOF 重新選門檻（規格第 4.5.1 節）。

注意：L3 的 10 次切分，測試集彼此重疊，也與調參資料重疊，只是穩定性分析，不是獨立驗證。

用法：
    python experiments/03_repeated_cv.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import XGB_TUNED
from src.protocol import (Policy, bill_at, choose_threshold, coverage_curve, cv_evaluate, dev_data, evaluate,
                          load_rules, make_model, oof_predict, prepare, start_run)

RULES_FREEZE_ID = "g2a-20261116"
SEED = 42
FOLDS = 5
LAGS = [0, 1, 2, 3, 4]  # lag=k：只用第 k+1～6 個月，提早 k+1 個月預警
RATES = [round(0.05 * i, 2) for i in range(1, 11)]  # 名單比例 5%～50%


def factory():
    return make_model("XGBoost", params=dict(XGB_TUNED))


def rq3_lead_time(df, *, lags=LAGS, rates=RATES, folds=FOLDS, seed=SEED):
    """RQ3：各 lag 的 CV AUC，以及 OOF 風險分數的前 N% 名單涵蓋率。"""
    y = df["default"]
    lead_rows, coverage_frames, oof_scores = [], [], {}
    for lag in lags:
        X, cols = prepare(df, lag=lag, sensitive=False, engineered=False)
        cv = cv_evaluate(factory, X, y, folds=folds, seed=seed)
        oof = oof_predict(factory, X, y, folds=folds, seed=seed)
        oof_scores[lag] = (X, oof)
        lead_rows.append({"lag": lag, "提早月數": lag + 1, "特徵數": len(cols),
                          "CV AUC 平均": cv["平均"].iloc[0], "CV AUC 標準差": cv["標準差"].iloc[0]})
        coverage = coverage_curve(y, oof, rates, bill=bill_at(df, 0))  # 事後評估一律用 9 月帳單（規格第 4.4 節）
        coverage.insert(0, "提早月數", lag + 1)
        coverage.insert(0, "lag", lag)
        coverage_frames.append(coverage)
        print(f"lag={lag}（提早 {lag + 1} 個月）：CV AUC {lead_rows[-1]['CV AUC 平均']:.4f}")
    return pd.DataFrame(lead_rows), pd.concat(coverage_frames, ignore_index=True), oof_scores


def evaluate_one_split(train, evaluation, *, rules, lag=0, seed=SEED):
    """訓練、選門檻、評估。回傳三種標準政策在評估資料上的整體結果。

    train：訓練資料（含 default 欄）。evaluation：評估資料；L1 給 None，改在 train 的 OOF 分數上評估。
    共同數值門檻 0.5 與固定名單前 20% 不重新選擇；正式政策在 train 的 OOF 重新選門檻。
    帳單：選門檻用預警當時的帳單 bill_at(train, lag)；事後評估（涵蓋率、總成本）一律用 9 月帳單 bill_at(·, 0)。
    """
    X, _ = prepare(train, lag=lag, sensitive=False, engineered=False)
    y = train["default"]
    oof = oof_predict(factory, X, y, folds=FOLDS, seed=seed)
    formal = choose_threshold(y, oof, bill_at(train, lag), rules.scenario)["政策物件"]
    policies = {"共同數值門檻 0.5": Policy.single(0.5, None),
                "固定名單前 20%": Policy.top_n(0.2),
                "正式政策（成本門檻）": formal}
    if evaluation is None:
        y_eval, score_eval, bill_eval = y, oof, bill_at(train, 0)  # 評估用 9 月帳單
    else:
        model = factory().fit(X, y)
        Z, _ = prepare(evaluation, lag=lag, sensitive=False, engineered=False)
        score_eval = pd.Series(model.predict_proba(Z)[:, 1], index=evaluation.index)
        y_eval, bill_eval = evaluation["default"], bill_at(evaluation, 0)  # 評估用 9 月帳單
    rows = []
    for name, policy in policies.items():
        overall = evaluate(y_eval, score=score_eval, policy=policy, bill=bill_eval, scenario=rules.scenario)["overall"]
        overall.insert(0, "政策", name)
        rows.append(overall)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    df = dev_data()  # L1：訓練集 24,000 筆
    y = df["default"]
    rules = load_rules(RULES_FREEZE_ID)
    print(f"資料層級 L1，{len(df):,} 筆，違約率 {y.mean():.1%}")
    print("L1 開發評估；正式數字以凍結後的 L2、L3 為準")

    run = start_run("L1", "F5 RQ3 提早預警：M1-NS 在 lag 0～4 的 CV AUC 與前 5%～50% 名單涵蓋率",
                    model_id="xgb-tuned-ns-s42train")
    print(f"run_id = {run.run_id}")

    # ---------- 1. RQ3 ----------
    lead, coverage, scores = rq3_lead_time(df)
    run.save_csv("rq3_lead_time", lead.round(4))
    run.save_csv("rq3_coverage", coverage.round(4))
    print(lead.round(4).to_string(index=False))
    print(coverage[coverage["名單比例"].isin([0.1, 0.2, 0.3])].round(4).to_string(index=False))

    # ---------- 2. 單一切分評估函式：L1 版（在 OOF 上評估），確認程式能跑 ----------
    check = evaluate_one_split(df, None, rules=rules)
    print("\n單一切分評估（L1，以 OOF 分數評估；L3 由 outer_splits 提供各 seed 的訓練與測試資料）：")
    print(check[["政策", "AUC", "Precision", "Recall", "名單人數", "違約者涵蓋率", "總成本"]].round(4).to_string(index=False))

    # ---------- 3. run 需要綁定一個模型才能完成：保存 lag=0 的 M1-NS 訓練紀錄 ----------
    X0, oof0 = scores[0]
    model = factory().fit(X0, y)
    run.attach_model(model, X0, y, oof0)
    run.finish()
    print(f"完成。結果在 results/{run.run_id}/")
