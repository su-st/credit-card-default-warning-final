"""評估指標與族群錯誤率。"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def metrics(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred),
            "Recall": recall_score(y_true, pred), "F1-score": f1_score(y_true, pred),
            "AUC": roc_auc_score(y_true, prob), "名單人數": int(pred.sum())}


def group_errors(y_true: pd.Series, pred, groups: pd.Series) -> pd.DataFrame:
    """各族群的漏判率（FNR）與誤判率（FPR）。"""
    pred = pd.Series(np.asarray(pred), index=y_true.index)
    rows = []
    for g, idx in groups.groupby(groups, observed=True).groups.items():
        yt, yp = y_true.loc[idx], pred.loc[idx]
        pos, neg = (yt == 1).sum(), (yt == 0).sum()
        rows.append([g, len(yt), yt.mean(),
                     ((yp == 0) & (yt == 1)).sum() / pos if pos else np.nan,
                     ((yp == 1) & (yt == 0)).sum() / neg if neg else np.nan])
    return pd.DataFrame(rows, columns=["族群", "人數", "實際違約率", "漏判率", "誤判率"])
