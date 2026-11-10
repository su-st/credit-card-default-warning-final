"""實驗共用：讀資料、切分，與期中完全相同。每支實驗檔開頭 from _common import * 即可。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402,F401
import pandas as pd  # noqa: E402,F401
from sklearn.model_selection import cross_val_score  # noqa: E402,F401

from src.config import OUT_DIR, RANDOM_STATE, XGB_TUNED  # noqa: E402,F401
from src.data import age_group, feature_cols, load_data, split  # noqa: E402
from src.evaluate import group_errors, metrics  # noqa: E402,F401
from src.features import add_features, engineered_cols  # noqa: E402,F401
from src.models import CV, make_model  # noqa: E402,F401

data = load_data()
train, test = split(data)
y_train, y_test = train["default"], test["default"]
RESULTS = OUT_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
