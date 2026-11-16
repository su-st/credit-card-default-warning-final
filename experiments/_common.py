"""Development-only common imports; L0 must explicitly use the legacy modules.

No test set or labels exist at module scope. Legacy experiment call signatures
are bridged to C2 pipelines; the old src.models.make_model is reserved for L0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as _pandas
from sklearn.model_selection import cross_val_score
from src.config import OUT_DIR, RANDOM_STATE, XGB_TUNED
from src.data import age_group, feature_cols
from src.evaluate import group_errors, metrics
from src.features import add_features, engineered_cols
from src.models import CV
from src.protocol import dev_data, dev_holdout, prepare, make_model as _make_model
from src.registry import _guard_sealed


class _Pandas:
    def __getattr__(self, name):
        if name.startswith('read_'):
            def guarded(*args, **kwargs):
                path = args[0] if args else kwargs.get('path_or_buf', kwargs.get('filepath_or_buffer'))
                if isinstance(path, (str, Path)):
                    _guard_sealed(path)
                return getattr(_pandas, name)(*args, **kwargs)
            return guarded
        return getattr(_pandas, name)

    def read_csv(self, path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            _guard_sealed(path)
            if Path(path).resolve() == (OUT_DIR / "predictions.csv").resolve():
                raise PermissionError("Legacy test handoff is retired; load a registered L1-H run")
        return _pandas.read_csv(path, *args, **kwargs)


pd = _Pandas()
_train = None
RESULTS = OUT_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def __getattr__(name):
    global _train
    if name not in {"train", "data", "y_train"}:
        raise AttributeError(name)
    if _train is None:
        _train = dev_data()
    return _train["default"] if name == "y_train" else _train


def make_model(name, cols=None, y_train=None, xgb_params=None):
    return _make_model(name, params=xgb_params)


__all__ = ["np", "pd", "cross_val_score", "OUT_DIR", "RANDOM_STATE", "XGB_TUNED", "age_group", "feature_cols",
           "group_errors", "metrics", "add_features", "engineered_cols", "CV", "make_model", "RESULTS",
           "train", "data", "y_train", "dev_data", "dev_holdout", "prepare"]
