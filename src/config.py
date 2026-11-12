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
LIST_THRESHOLD = 0.5            # 備用值：還沒執行 experiments/02_threshold.py 時才會用到

# ---------- 門檻自動選擇（experiments/02_threshold.py）----------
# 規則：「cost」總成本最低、「f1」F1 最高、「top_n」固定名單人數
THRESHOLD_RULE = "f1"
# 以下三個是成本假設，數字需要自己訂定並在報告說明依據
LOSS_PER_DEFAULT = None    # 漏判一位違約者的損失（元）。None 代表用訓練集違約者最近一期帳單的平均
COST_PER_CONTACT = 50      # 【假設】聯絡一位客戶的成本（元）
RESCUE_RATE = 0.2          # 【假設】聯絡後成功避免違約的比例
TOP_N_RATE = 0.2           # 規則為 top_n 時，名單佔全部客戶的比例
MAX_LIST_RATE = None       # 規則為 cost 時的人力上限：名單最多佔全部客戶的比例，None 代表不限制
