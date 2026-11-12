"""呼叫本機部署的 TAIDE 產生說明。沒有設定模型、呼叫失敗或輸出沒通過檢查時，自動改用固定句型。
負責人：蘇韻蓉｜分支：feat/taide

本機用 Ollama 執行 TAIDE，設定兩個環境變數（寫在 .env）：
  OLLAMA_URL   預設 http://localhost:11434
  TAIDE_MODEL  在 Ollama 裡的模型名稱，例如 ollama list 顯示的名稱

為什麼要檢查輸出：
  初次測試時，TAIDE 在關懷訊息裡自行編出繳款日期與客服電話，
  因此每次產生後都由程式檢查，不合格就改用固定句型，並記錄原因。
  檢查只能抓到數字、字數與禁用字，語意錯誤仍需行員確認後才發送。
"""
import os
import re

import requests

from . import templates

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TAIDE_MODEL = os.getenv("TAIDE_MODEL", "")
MAX_CHARS = 120  # 提示詞要求 80 字，檢查時保留一點彈性

# ---------- 提示詞 ----------
EXPLAIN_SYSTEM = (
    "你是銀行信用卡風控助理，負責向行員說明客戶為何被列入關懷名單。"
    "規則：用繁體中文直接說明，不要問候、道歉或客套。最多兩句，80 字以內。"
    "只能使用資料裡的數字與事實，不可以加入其他數字、日期或自行推測。"
    "資料裡列出的因素都是讓風險升高的原因，請照原文引用，不要改寫成其他意思。"
    "不要自行評論金額高低或補充解釋，例如「金額較高」「紀錄不佳」，只要列出因素。"
)
EXPLAIN_EXAMPLE = (
    "資料：風險等級高（分數 0.85）。推高風險的主要因素：9 月延遲繳款 2 個月、8 月延遲繳款 2 個月、信用額度 50,000 元。",
    "這位客戶風險等級高（分數 0.85），主要原因是 9 月延遲繳款 2 個月、8 月延遲繳款 2 個月，"
    "另外信用額度 50,000 元也推高了風險，建議優先關懷。",
)

CARE_SYSTEM = (
    "你是銀行信用卡客服，要寫一則給持卡人的 LINE 關懷訊息。"
    "規則：用繁體中文，語氣溫和客氣，80 字以內。內容是提醒對方留意本期帳單的繳款期限，"
    "並表示若有困難可以協助安排還款方式。"
    "不可以出現任何數字、日期、金額或電話號碼，需要時請寫「帳單上的繳款期限」或「客服專線」。"
    "不可以提到風險、分數、模型、名單或違約。不要署名，也不要提到 TAIDE 或人工智慧。"
)
CARE_EXAMPLE = (
    "請寫一則關懷訊息。",
    "您好，提醒您留意本期帳單上的繳款期限。若近期繳款有困難，歡迎透過客服專線與我們聯繫，"
    "我們會協助您了解可行的還款方式。",
)
CARE_BANNED = ["違約", "風險", "分數", "模型", "名單", "TAIDE", "人工智慧"]


# ---------- 呼叫 TAIDE ----------
def _chat(system: str, example: tuple, user: str) -> str:
    """用 Ollama 的 chat API 呼叫 TAIDE，附一組範例讓輸出格式穩定。"""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": example[0]},
                {"role": "assistant", "content": example[1]},
                {"role": "user", "content": user}]
    r = requests.post(f"{OLLAMA_URL}/api/chat",
                      json={"model": TAIDE_MODEL, "messages": messages, "stream": False,
                            "options": {"temperature": 0.2, "num_predict": 200}},
                      timeout=120)
    r.raise_for_status()
    return _clean(r.json()["message"]["content"])


def _clean(text: str) -> str:
    """去掉 TAIDE 有時會加的標題（例如「關懷訊息：」）、引號與換行。"""
    text = re.sub(r"^\s*(關懷訊息|訊息草稿|風險說明)\s*[:：]?\s*", "", text.strip())
    text = re.sub(r"^\s*(訊息|說明)\s*[:：]\s*", "", text)
    text = re.sub(r"\s*\n+\s*", "", text)
    return text.strip().strip("「」\"'“” ")


# ---------- 檢查 ----------
def _numbers(text: str) -> set:
    """取出文字裡的數字（含全形），去掉千分位逗號。"""
    text = text.translate(str.maketrans("０１２３４５６７８９．", "0123456789."))
    return {n.replace(",", "") for n in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)}


def check_explain(text: str, facts: str) -> str:
    """回傳不合格的原因，空字串代表通過。"""
    if not text:
        return "沒有輸出"
    if len(text) > MAX_CHARS:
        return f"超過字數（{len(text)} 字）"
    extra = _numbers(text) - _numbers(facts)
    if extra:
        return "出現資料以外的數字：" + "、".join(sorted(extra))
    return ""


def check_care(text: str) -> str:
    if not text:
        return "沒有輸出"
    if len(text) > MAX_CHARS:
        return f"超過字數（{len(text)} 字）"
    if _numbers(text):
        return "出現數字：" + "、".join(sorted(_numbers(text)))
    bad = [w for w in CARE_BANNED if w in text]
    if bad:
        return "出現禁用字：" + "、".join(bad)
    return ""


# ---------- 對外提供的函式 ----------
def _factors(c: dict) -> list:
    return [f for f in [c.get("factor_1"), c.get("factor_2"), c.get("factor_3")] if isinstance(f, str) and f]


def _facts(c: dict) -> str:
    s = c["risk_score"]
    level = "高" if s >= 0.7 else ("中" if s >= 0.5 else "低")
    factors = "、".join(_factors(c)) or "無明顯單一因素"
    return f"風險等級{level}（分數 {s:.2f}）。推高風險的主要因素：{factors}。"


def generate(kind: str, c: dict) -> dict:
    """產生說明或關懷訊息，回傳完整紀錄，方便做改善前後的比較實驗。
    kind 為 "explain" 或 "care"。
    回傳 {"text": 最後使用的文字, "how": 產生方式, "raw": TAIDE 原始輸出, "reason": 不合格原因}
    """
    fallback = templates.risk_explanation(c) if kind == "explain" else templates.care_message(c)
    if not TAIDE_MODEL:
        return {"text": fallback, "how": "固定句型（未設定 TAIDE_MODEL）", "raw": "", "reason": "未設定模型"}
    try:
        if kind == "explain":
            facts = _facts(c)
            raw = _chat(EXPLAIN_SYSTEM, EXPLAIN_EXAMPLE, "資料：" + facts)
            reason = check_explain(raw, facts)
        else:
            raw = _chat(CARE_SYSTEM, CARE_EXAMPLE, "請寫一則關懷訊息。")
            reason = check_care(raw)
    except Exception as e:  # 模型沒開或逾時
        print("TAIDE 呼叫失敗，改用固定句型：", e)
        return {"text": fallback, "how": "固定句型（TAIDE 呼叫失敗）", "raw": "", "reason": str(e)}
    if reason:
        print(f"TAIDE 輸出未通過檢查（{reason}），改用固定句型。原始輸出：{raw}")
        return {"text": fallback, "how": f"固定句型（TAIDE 輸出{reason}）", "raw": raw, "reason": reason}
    return {"text": raw, "how": "TAIDE", "raw": raw, "reason": ""}


def explain(c: dict) -> tuple[str, str]:
    """回傳（行員看的風險說明, 產生方式）。"""
    r = generate("explain", c)
    return r["text"], r["how"]


def care_message(c: dict) -> tuple[str, str]:
    """回傳（給客戶的關懷訊息草稿, 產生方式）。訊息不得透露分數或「違約」字眼。"""
    r = generate("care", c)
    return r["text"], r["how"]
