"""呼叫本機部署的 TAIDE 產生說明。沒有設定模型或呼叫失敗時，自動改用固定句型。

本機用 Ollama 執行 TAIDE，設定兩個環境變數（寫在 .env）：
  OLLAMA_URL   預設 http://localhost:11434
  TAIDE_MODEL  在 Ollama 裡的模型名稱，例如 ollama list 顯示的名稱
"""
import os

import requests

from . import templates

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TAIDE_MODEL = os.getenv("TAIDE_MODEL", "")

RULES = ("請用繁體中文、口語且客氣的語氣回答，不超過 80 字。"
         "只能根據下面提供的資料說明，不可以自行編造數字或原因。")


def _ask(prompt: str) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": TAIDE_MODEL, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.2}}, timeout=90)
    r.raise_for_status()
    return r.json()["response"].strip()


def _facts(c: dict) -> str:
    factors = "、".join(f for f in [c.get("factor_1"), c.get("factor_2"), c.get("factor_3")] if isinstance(f, str) and f)
    return f"風險分數 {c['risk_score']:.2f}（0 到 1，越高越可能違約），主要風險因素：{factors}。"


def explain(c: dict) -> tuple[str, str]:
    """回傳（行員看的風險說明, 產生方式）。"""
    if TAIDE_MODEL:
        try:
            return _ask(f"你是銀行信用卡風控助理。{RULES}\n請向行員說明這位客戶為何被列入關懷名單。\n資料：{_facts(c)}"), "TAIDE"
        except Exception as e:  # 模型沒開或逾時
            print("TAIDE 呼叫失敗，改用固定句型：", e)
    return templates.risk_explanation(c), "固定句型"


def care_message(c: dict) -> tuple[str, str]:
    """回傳（給客戶的關懷訊息草稿, 產生方式）。訊息不得透露分數或「違約」字眼。"""
    if TAIDE_MODEL:
        try:
            return _ask(f"你是銀行信用卡客服。{RULES}\n請寫一則給持卡人的 LINE 關懷訊息，提醒繳款期限並表示可協助還款安排。"
                        "不要提到風險分數、模型或違約。"), "TAIDE"
        except Exception as e:
            print("TAIDE 呼叫失敗，改用固定句型：", e)
    return templates.care_message(c), "固定句型"
