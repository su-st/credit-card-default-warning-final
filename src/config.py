"""專案共用設定。改這裡的數字，所有實驗與平台都會一起套用。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42          # 與期中相同，確保可重現
TEST_SIZE = 0.2            # 80/20 分層切分

CAT_COLS = ["EDUCATION", "MARRIAGE"]
SENSITIVE = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

# 期中可行性測試使用的 XGBoost 參數
XGB_BASELINE = dict(n_estimators=300, learning_rate=0.05, max_depth=4)

# 調參初步測試（60 組隨機搜尋，五折 CV AUC 0.786）選出的參數
# 負責調參的組員完成正式調參後，更新這裡
XGB_TUNED = dict(n_estimators=500, learning_rate=0.01, max_depth=6, min_child_weight=3,
                 subsample=0.8, colsample_bytree=0.7, gamma=0.0, reg_lambda=10.0)

# 平台使用的模型設定
PLATFORM_USE_SENSITIVE = True   # RQ1 正式結果出來後再決定是否移除敏感屬性
LIST_THRESHOLD = 0.5            # 門檻情境分析完成後更新
