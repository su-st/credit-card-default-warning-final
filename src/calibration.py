"""F7 shared cross-fitted calibration. Does not replace platform risk scores."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from .registry import content_hash


def crossfit_calibrate(y, oof_score, *, folds=5, seed=42):
    from .protocol import _aligned, _labels
    _aligned(y, oof_score)
    _labels(y)
    if not np.isfinite(oof_score).all() or not oof_score.between(0, 1).all():
        raise ValueError("Invalid OOF scores")
    if y.value_counts().min() < folds:
        raise ValueError("Too few examples per class")
    calibrated = pd.Series(np.nan, index=y.index, name="calibrated_score")
    assignment = pd.Series(-1, index=y.index, dtype=int)
    provenance = []
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(oof_score, y)):
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_score.iloc[tr], y.iloc[tr])
        calibrated.iloc[va] = calibrator.predict(oof_score.iloc[va])
        assignment.iloc[va] = fold
        provenance.append({"折編號": fold, "訓練ID雜湊": content_hash(y.index[tr].tolist()),
            "驗證ID雜湊": content_hash(y.index[va].tolist()), "訓練人數": len(tr), "驗證人數": len(va), "seed": seed})
    calibrated.attrs["fold_assignment"] = assignment
    return calibrated, pd.DataFrame(provenance)


def calibration_report(y, score, *, bins=10):
    from .protocol import _aligned, _labels
    _aligned(y, score)
    _labels(y)
    if not isinstance(bins, int) or bins < 1 or not np.isfinite(score).all() or not score.between(0, 1).all():
        raise ValueError("Invalid bins/scores")
    intervals = pd.cut(score, np.linspace(0, 1, bins+1), include_lowest=True)
    frame = pd.DataFrame({"分數區間": intervals, "score": score, "y": y})
    curve = frame.groupby("分數區間", observed=True).agg(人數=("y", "size"), 平均預測值=("score", "mean"), 實際違約率=("y", "mean")).reset_index()
    return {"Brier分數": float(brier_score_loss(y, score)), "校準曲線": curve}


def rank_change(score_a, score_b, rate):
    from .protocol import _aligned, apply_policy, Policy
    _aligned(score_a, score_b)
    a, _ = apply_policy(score_a, Policy.top_n(rate))
    b, _ = apply_policy(score_b, Policy.top_n(rate))
    return float(((a == 1) & (b == 1)).sum()/a.sum()) if a.sum() else np.nan
