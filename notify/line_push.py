"""LINE 推播。只發給測試用帳號（組員自己的 LINE），不會發給真實客戶。

在 .env 設定：
  LINE_CHANNEL_ACCESS_TOKEN  LINE Developers 後台 Messaging API 分頁的 Channel access token
  LINE_TEST_USER_ID          收訊息的測試帳號 User ID（後台 Basic settings 的 Your user ID）
"""
import os

from linebot.v3.messaging import ApiClient, Configuration, MessagingApi, PushMessageRequest, TextMessage

TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
TEST_USER = os.getenv("LINE_TEST_USER_ID", "")


def ready() -> bool:
    return bool(TOKEN and TEST_USER)


def push_text(text: str, user_id: str | None = None) -> str:
    """推送一則文字訊息，回傳結果說明。未設定 token 時只模擬，不會真的發送。"""
    if not ready():
        return "（模擬發送）尚未設定 LINE token，訊息沒有真的送出。"
    with ApiClient(Configuration(access_token=TOKEN)) as client:
        MessagingApi(client).push_message(PushMessageRequest(to=user_id or TEST_USER,
                                                             messages=[TextMessage(text=text)]))
    return "已送出到測試帳號。"


if __name__ == "__main__":
    print(push_text("違約預警平台測試訊息"))
