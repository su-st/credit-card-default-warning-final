"""LINE 推播。只發給測試用帳號（組員自己的 LINE），不會發給真實客戶。

在 .env 設定：
  LINE_CHANNEL_ACCESS_TOKEN  LINE Developers 後台 Messaging API 分頁的 Channel access token
  LINE_TEST_USER_ID          收訊息的測試帳號 User ID（後台 Basic settings 的 Your user ID）
  LINE_SEND_ENABLED          預設 false；明確設 true 且具備憑證才實際發送
CLI 僅顯示操作指引，無論環境設定為何都不發送。
"""
import os

from linebot.v3.messaging import ApiClient, Configuration, MessagingApi, PushMessageRequest, TextMessage

TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
TEST_USER = os.getenv("LINE_TEST_USER_ID", "")
SEND_ENABLED = os.getenv("LINE_SEND_ENABLED", "").lower() == "true"


def ready() -> bool:
    return bool(TOKEN and TEST_USER)


def is_simulation() -> bool:
    """必須明確啟用且具備設定，才可能真正發送。"""
    return not (SEND_ENABLED and ready())


def push_text(text: str, user_id: str | None = None, *, approved_text: str | None = None) -> str:
    """推送一則文字訊息，回傳結果說明。未明確啟用或缺少憑證時只模擬。"""
    from llm.explain import check_care

    if user_id is not None and user_id != TEST_USER:
        raise ValueError("僅允許設定的 LINE 測試帳號")
    reason = check_care(text)
    if reason:
        raise ValueError(f"關懷訊息未通過檢查：{reason}")
    if is_simulation():
        return "（模擬發送）未呼叫 LINE API，訊息沒有真的送出。"
    if approved_text != text:
        raise PermissionError("實際發送需要目前訊息的人工核准")
    with ApiClient(Configuration(access_token=TOKEN)) as client:
        MessagingApi(client).push_message(PushMessageRequest(to=TEST_USER,
                                                             messages=[TextMessage(text=text)]))
    return "已送出到測試帳號。"


if __name__ == "__main__":
    print("CLI 不提供發送功能。請在 Streamlit 客戶頁檢查訊息並人工確認後操作。")
