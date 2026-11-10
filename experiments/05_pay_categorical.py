"""負責人：陳品婕｜分支：exp/pay-categorical
繳款狀態改用類別編碼：PAY_1 至 PAY_6 的 -2、-1、0 都代表沒有延遲，但目前被當成數字。
TODO 把 PAY 欄位加入 OneHotEncoder（或另外做「是否延遲」「延遲月數」兩個欄位），比較 CV AUC 與 F1。
"""
from _common import *
