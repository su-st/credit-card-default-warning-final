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
    "你是銀行信用卡風控助理，負責向行員說明本期預警名單與模型風險因素。"
    "規則：用繁體中文直接說明，不要問候、道歉或客套。最多兩句，80 字以內。"
    "只能使用資料裡的名單狀態、排名、分數與事實，不可以加入其他數字、日期或自行推測。"
    "不得用分數高低另分高中低等級，不得更改資料提供的名單狀態。"
    "因素是模型判斷依據，不代表違約原因；只稱為風險因素，不把關聯描述成因果。"
    "說明必須註明：模型判斷依據，不代表違約原因。"
    "不要自行評論金額高低或補充因果解釋，只列出資料提供的因素。"
)
EXPLAIN_EXAMPLE = (
    "資料：列入本期預警名單（風險排名第 12 名）。風險分數 0.85。風險因素：9 月延遲繳款 2 個月。",
    "這位客戶列入本期預警名單（風險排名第 12 名），風險因素為9 月延遲繳款 2 個月。"
    "模型判斷依據，不代表違約原因。",
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

# C5 leader-approved draft quality vocabulary, separate from the send whitelist.
CARE_QUALITY_DATES = ('今天', '明天', '後天', '今晚', '明早', '本週', '下週', '本月', '月底', '月初', '中午')
CARE_QUALITY_PROMISES = ('保證', '免息', '核准', '凍結', '停卡')
CARE_QUALITY_LINK = re.compile(r'https?://|www\.', re.IGNORECASE)
CARE_CHINESE_NUMBER = re.compile(
    r'[零〇一二兩三四五六七八九十百千萬億壹貳參肆伍陸柒捌玖拾佰仟]{2,}'
    r'|[零〇一二兩三四五六七八九十百千萬億]+(?:年|月|日|天|點|時|分|元|塊)')
EXPLAIN_UNSUPPORTED_WORDING = re.compile(
    r'風險\s*(?:等級|程度)?\s*(?:為|是|偏|較)?\s*[高中低]|違約機率|主要原因')
FACTOR_NOTE = '模型判斷依據，不代表違約原因'


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


def quality_check(kind: str, text: str, facts: str) -> str:
    """Check draft quality only; an empty reason never authorizes sending."""
    if kind not in ('explain', 'care'):
        raise ValueError('kind must be explain or care')
    if not text or not text.strip():
        return '沒有輸出'
    if len(text) > MAX_CHARS:
        return f'超過字數（{len(text)} 字）'
    if kind == 'care':
        if _numbers(text):
            return '出現數字：' + '、'.join(sorted(_numbers(text)))
        if CARE_CHINESE_NUMBER.search(text):
            return '出現中文數字'
        if CARE_QUALITY_LINK.search(text):
            return '出現未核實的連結'
        dates = [word for word in CARE_QUALITY_DATES if word in text]
        if dates:
            return '出現未核實的日期用語：' + '、'.join(dates)
        promises = [word for word in CARE_QUALITY_PROMISES if word in text]
        if promises:
            return '出現未核實的承諾或威嚇用語：' + '、'.join(promises)
        bad = [word for word in CARE_BANNED if word in text]
        if bad:
            return '出現禁用字：' + '、'.join(bad)
    else:
        if EXPLAIN_UNSUPPORTED_WORDING.search(text):
            return '使用固定風險分級或不適當用詞'
        extra = _numbers(text) - _numbers(facts)
        if extra:
            return '出現資料以外的數字：' + '、'.join(sorted(extra))
    return ''


def check_explain(text: str, facts: str) -> str:
    """Compatibility entry point for staff draft quality checks."""
    return quality_check('explain', text, facts)


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
    # 沒有權威期限、客服聯絡資料或核准條款可供核對，因此拒絕自由生成內容。
    # 此處是完整模板白名單，不是通用語意驗證器；新增模板須人工審核。
    if text.strip() != templates.care_message({}):
        return "非固定模板內容，無法核實期限、聯絡資訊或承諾"
    return ""


# ---------- 對外提供的函式 ----------
def _factors(c: dict) -> list:
    return [f for f in [c.get("factor_1"), c.get("factor_2"), c.get("factor_3")] if isinstance(f, str) and f]


def _facts(c: dict) -> str:
    status = templates.list_status(c)
    factors = '、'.join(_factors(c)) or '無明顯單一因素'
    return (f"{status}。風險分數 {c['risk_score']:.2f}。風險因素：{factors}。"
            '模型判斷依據，不代表違約原因。')


def generate(kind: str, c: dict) -> dict:
    """產生說明或關懷訊息，回傳完整紀錄，方便做改善前後的比較實驗。
    kind 為 "explain" 或 "care"。
    回傳 {"text": 最後使用的文字, "how": 產生方式, "raw": TAIDE 原始輸出, "reason": 不合格原因}
    """
    if kind not in ("explain", "care"):
        raise ValueError("kind must be explain or care")
    fallback = templates.risk_explanation(c) if kind == "explain" else templates.care_message(c)
    if not TAIDE_MODEL:
        return {"text": fallback, "how": "固定句型（未設定 TAIDE_MODEL）", "raw": "", "reason": "未設定模型"}
    try:
        if kind == "explain":
            facts = _facts(c)
            raw = _chat(EXPLAIN_SYSTEM, EXPLAIN_EXAMPLE, "資料：" + facts)
            reason = quality_check("explain", raw, facts)
        else:
            raw = _chat(CARE_SYSTEM, CARE_EXAMPLE, "請寫一則關懷訊息。")
            reason = quality_check("care", raw, "")
    except Exception as e:  # 模型沒開或逾時
        print("TAIDE 呼叫失敗，改用固定句型：", e)
        return {"text": fallback, "how": "固定句型（TAIDE 呼叫失敗）", "raw": "", "reason": str(e)}
    text = raw
    if not reason and kind == "explain" and FACTOR_NOTE not in text:
        text = text.rstrip("。") + "。" + FACTOR_NOTE + "。"
        reason = quality_check(kind, text, facts)
    if reason:
        print(f"TAIDE 輸出未通過檢查（{reason}），改用固定句型。原始輸出：{raw}")
        return {"text": fallback, "how": f"固定句型（TAIDE 輸出{reason}）", "raw": raw, "reason": reason}
    return {"text": text, "how": "TAIDE", "raw": raw, "reason": ""}


def explain(c: dict) -> tuple[str, str]:
    """回傳（行員看的風險說明, 產生方式）。"""
    r = generate("explain", c)
    return r["text"], r["how"]


def care_message(c: dict) -> tuple[str, str]:
    """回傳（給客戶的關懷訊息草稿, 產生方式）。訊息不得透露分數或「違約」字眼。"""
    r = generate("care", c)
    return r["text"], r["how"]
