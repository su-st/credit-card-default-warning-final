"""負責人：蘇韻蓉｜分支：exp/taide-eval
TAIDE 輸出改善前後比較（期末報告用）。
改善前：期中框架原本的提示詞，沒有範例、沒有檢查
改善後：llm/explain.py 目前的版本（範例、禁止數字與署名、程式檢查）
兩者都用同一套檢查規則（llm/explain.py 的 check_explain、check_care）判斷是否合格。

需要：
1. Ollama 已開啟，.env 已填 TAIDE_MODEL
2. 已執行 python -m src.export，產生 outputs/predictions.csv
3. feat/taide 已合併到 main（需要 explain.py 的 generate 與檢查函式）

輸出 outputs/results/10_taide_eval.csv：每位客戶兩種版本的原始輸出與不合格原因。
程式只能抓到數字、字數與禁用字，語意錯誤（例如把 2,400 元說成金額偏高）
請在「人工判讀」欄自己標註，再一起統計。

TODO（蘇韻蓉）
1. 執行一次（30 位客戶 × 2 版本 × 2 種內容，約需 10 到 20 分鐘）
2. 逐筆閱讀輸出，在 CSV 的人工判讀欄標註語意錯誤
3. 把不合格比例整理成表格放進期末報告
"""
import time

from dotenv import load_dotenv

from _common import *

load_dotenv(OUT_DIR.parent / ".env")  # 要在匯入 llm.explain 之前讀，TAIDE_MODEL 才會生效

import requests  # noqa: E402

from llm import explain  # noqa: E402

N = 30  # 測試的客戶人數

# ---------- 改善前：期中框架原本的提示詞 ----------
OLD_RULES = ("請用繁體中文、口語且客氣的語氣回答，不超過 80 字。"
             "只能根據下面提供的資料說明，不可以自行編造數字或原因。")


def old_ask(prompt: str) -> str:
    r = requests.post(f"{explain.OLLAMA_URL}/api/generate",
                      json={"model": explain.TAIDE_MODEL, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.2}}, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


def old_explain(c: dict) -> str:
    factors = "、".join(f for f in [c.get("factor_1"), c.get("factor_2"), c.get("factor_3")] if isinstance(f, str) and f)
    facts = f"風險分數 {c['risk_score']:.2f}（0 到 1，越高越可能違約），主要風險因素：{factors}。"
    return old_ask(f"你是銀行信用卡風控助理。{OLD_RULES}\n請向行員說明這位客戶為何被列入關懷名單。\n資料：{facts}")


def old_care(c: dict) -> str:
    return old_ask(f"你是銀行信用卡客服。{OLD_RULES}\n請寫一則給持卡人的 LINE 關懷訊息，提醒繳款期限並表示可協助還款安排。"
                   "不要提到風險分數、模型或違約。")


# ---------- 執行 ----------
if not explain.TAIDE_MODEL:
    raise SystemExit("沒有設定 TAIDE_MODEL，請先在 .env 填入（ollama list 顯示的名稱）")
pred_file = OUT_DIR / "predictions.csv"
if not pred_file.exists():
    raise SystemExit("找不到 outputs/predictions.csv，請先執行 python -m src.export")

df = pd.read_csv(pred_file)
sample = df[df["in_list"] == 1].sample(n=N, random_state=RANDOM_STATE)
rows = []
start = time.time()
for i, (_, s) in enumerate(sample.iterrows(), 1):
    c = s.to_dict()
    facts = explain._facts(c)
    old_e, old_m = old_explain(c), old_care(c)
    new_e, new_m = explain.generate("explain", c), explain.generate("care", c)
    rows.append({
        "customer_id": c["customer_id"],
        "改善前_說明": old_e, "改善前_說明_不合格原因": explain.check_explain(old_e, facts),
        "改善前_訊息": old_m, "改善前_訊息_不合格原因": explain.check_care(old_m),
        "改善後_說明": new_e["raw"], "改善後_說明_不合格原因": new_e["reason"],
        "改善後_訊息": new_m["raw"], "改善後_訊息_不合格原因": new_m["reason"],
        "人工判讀": "",
    })
    print(f"{i}/{N} 完成（{time.time() - start:.0f} 秒）")

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "10_taide_eval.csv", index=False, encoding="utf-8-sig")
summary = pd.DataFrame({
    "內容": ["行員說明", "關懷訊息"],
    "改善前不合格比例": [(out["改善前_說明_不合格原因"] != "").mean(), (out["改善前_訊息_不合格原因"] != "").mean()],
    "改善後不合格比例": [(out["改善後_說明_不合格原因"] != "").mean(), (out["改善後_訊息_不合格原因"] != "").mean()],
})
summary.to_csv(RESULTS / "10_taide_eval_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n{N} 位客戶，程式檢查的不合格比例：")
print(summary.round(3).to_string(index=False))
print("\n請打開 outputs/results/10_taide_eval.csv 逐筆閱讀，在人工判讀欄標註語意錯誤")
