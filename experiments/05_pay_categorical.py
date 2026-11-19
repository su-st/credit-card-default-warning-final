"""負責人：陳品婕｜分支：exp/pay-categorical
繳款狀態改用類別編碼：PAY_1 至 PAY_6 的代碼目前被當成數字。
UCI 官方說明只定義 -1（準時繳清）與 1～9（延遲月數），沒有定義 0 與 -2；
本研究不假設這兩個代碼的意義，只把它們當成不同類別處理。
TODO 依規格，用事先寫好的固定對照表把 PAY_1 至 PAY_6 的代碼轉成類別，再交給 Pipeline 內的 OneHotEncoder 處理
（不使用目標編碼），比較 CV AUC 與 F1。
"""
from _common import *
