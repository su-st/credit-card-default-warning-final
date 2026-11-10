"""固定句型（TAIDE 無法使用時的備案），也作為 TAIDE 提示詞的範例。"""


def risk_explanation(c: dict) -> str:
    """給行員看的風險說明。c 為 predictions.csv 的一列。"""
    level = "高" if c["risk_score"] >= 0.7 else ("中" if c["risk_score"] >= 0.5 else "低")
    factors = "、".join(f for f in [c.get("factor_1"), c.get("factor_2"), c.get("factor_3")] if isinstance(f, str) and f)
    return (f"客戶 {c['customer_id']} 的違約風險為{level}（分數 {c['risk_score']:.2f}）。"
            f"模型判斷的主要原因是{factors or '無明顯單一因素'}。建議優先關懷並確認近期繳款安排。")


def care_message(c: dict) -> str:
    """給客戶的關懷訊息。不提模型、分數或「違約」等字眼，避免造成壓力。"""
    return ("您好，這裡是信用卡客服。提醒您留意本期帳單的繳款期限，"
            "若近期繳款有任何困難，歡迎與我們聯繫，我們可以協助您了解分期或其他還款方式。")
