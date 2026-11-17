"""固定句型（TAIDE 無法使用時的備案），也作為 TAIDE 提示詞的範例。"""


def list_status(c: dict) -> str:
    """Describe the handoff policy, never infer a grade from risk_score.

    A row does not contain the list size. Show it only when the caller supplies
    list_count; never invent a population total from the customer's rank.
    """
    import math
    from numbers import Real

    selected, rank = c['in_list'], c['risk_rank']
    if selected not in (0, 1):
        raise ValueError('in_list must be 0 or 1')
    if isinstance(rank, bool) or not isinstance(rank, Real) or not math.isfinite(rank) or rank < 1 or int(rank) != rank:
        raise ValueError('risk_rank must be a positive integer')
    if not selected:
        return '未列入本期預警名單'
    total = c.get('list_count')
    if total is None:
        return f'列入本期預警名單（風險排名第 {int(rank)} 名）'
    if isinstance(total, bool) or not isinstance(total, Real) or not math.isfinite(total) or total < 1 or int(total) != total:
        raise ValueError('list_count must be a positive integer when supplied')
    return f'列入本期預警名單（風險排名第 {int(rank)} 名，名單共 {int(total)} 人）'


def risk_explanation(c: dict) -> str:
    """給行員看的風險說明。c 為 predictions.csv 的一列。"""
    status = list_status(c)
    factors = '、'.join(f for f in [c.get('factor_1'), c.get('factor_2'), c.get('factor_3')] if isinstance(f, str) and f)
    return (f"客戶 {c['customer_id']}：{status}；風險分數 {c['risk_score']:.2f}。"
            f"風險因素：{factors or '無明顯單一因素'}。模型判斷依據，不代表違約原因。")


def care_message(c: dict) -> str:
    """給客戶的關懷訊息。不提模型、分數或「違約」等字眼，避免造成壓力。"""
    return ("您好，這裡是信用卡客服。提醒您留意本期帳單的繳款期限，"
            "若近期繳款有任何困難，歡迎與我們聯繫，我們可以協助您了解分期或其他還款方式。")
