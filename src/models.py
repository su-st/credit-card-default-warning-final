"""模型建立，與期中筆記本 make_model 相同，另外可傳入調參後的 XGBoost 參數。"""
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator

from .config import CAT_COLS, RANDOM_STATE, XGB_BASELINE

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def pos_weight(y) -> float:
    return (y == 0).sum() / (y == 1).sum()


def make_model(name: str, cols: list, y_train, xgb_params: dict | None = None) -> Pipeline:
    """name 為「邏輯迴歸」「決策樹」「XGBoost」。三個模型都加類別權重。"""
    cat = [c for c in CAT_COLS if c in cols]
    num = [c for c in cols if c not in cat]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("num", StandardScaler(), num)])
    if name == "邏輯迴歸":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    elif name == "決策樹":
        clf = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "XGBoost":
        params = dict(XGB_BASELINE if xgb_params is None else xgb_params)
        clf = XGBClassifier(**params, scale_pos_weight=pos_weight(y_train), eval_metric="logloss",
                            random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])


class FitWeightedXGBClassifier(XGBClassifier):
    """C2: compute scale_pos_weight from the labels actually passed to fit."""
    def fit(self, X, y, **kwargs):
        self.fit_label_count_ = len(y)
        self.fit_positive_count_ = int((y == 1).sum())
        if not 0 < self.fit_positive_count_ < len(y):
            raise ValueError("Both classes are required to compute class weights")
        self.set_params(scale_pos_weight=float(pos_weight(y)))
        return super().fit(X, y, **kwargs)


class FeatureSelector(BaseEstimator):
    """Serializable ColumnTransformer selector fitted inside each pipeline."""
    def __init__(self, categorical):
        self.categorical = categorical

    def __call__(self, X):
        return [c for c in X.columns if (c in CAT_COLS) == self.categorical]


def make_fold_model(name, *, params=None):
    """New pipeline; the historical make_model above remains unchanged."""
    from .config import XGB_TUNED

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FeatureSelector(True)),
        ("num", StandardScaler(), FeatureSelector(False)),
    ])
    params = {} if params is None else dict(params)
    if name != "XGBoost" and params.get("class_weight", "balanced") != "balanced":
        raise ValueError("Class weights must be computed from each fit's labels")
    if name == "XGBoost":
        if "scale_pos_weight" in params:
            raise ValueError("Class weight is computed in fit, not supplied by the caller")
        configured = dict(XGB_TUNED, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
        configured.update(params)
        clf = FitWeightedXGBClassifier(**configured)
    elif name == "邏輯迴歸":
        clf = LogisticRegression(**dict(dict(max_iter=3000, class_weight="balanced"), **params))
    elif name == "決策樹":
        clf = DecisionTreeClassifier(**dict(dict(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE), **params))
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])
