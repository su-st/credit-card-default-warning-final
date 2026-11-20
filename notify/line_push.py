"""Separate LINE customer/staff routes. Never load .env or print settings.

New customer care requires an exact approval in both modes. Staff accepts only
StaffDigest's count/ID/rank/test-link schema. Neither API accepts a recipient.
CLI never sends, regardless of process settings.
"""
import os

from linebot.v3.messaging import ApiClient, Configuration, MessagingApi, PushMessageRequest, TextMessage

from .staff_digest import DEFAULT_PLATFORM_URL, render_digest

TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
_LEGACY_CUSTOMER = os.getenv("LINE_TEST_USER_ID", "")
if _LEGACY_CUSTOMER:
    print("LINE_TEST_USER_ID 已改名為 LINE_TEST_CUSTOMER_ID，請更新設定名稱。")
# TEST_USER remains the internal compatibility name used by PR #8 callers/tests.
TEST_USER = os.getenv("LINE_TEST_CUSTOMER_ID", "") or _LEGACY_CUSTOMER
TEST_STAFF = os.getenv("LINE_TEST_STAFF_ID", "")
PLATFORM_URL = os.getenv("LINE_PLATFORM_URL", DEFAULT_PLATFORM_URL)
SEND_ENABLED = os.getenv("LINE_SEND_ENABLED", "").lower() == "true"


def _recipient(kind):
    if kind not in ('customer', 'staff'):
        raise ValueError('未知 LINE 訊息類型')
    value = TEST_USER if kind == 'customer' else TEST_STAFF
    if not isinstance(value, str) or not value.strip():
        setting = 'LINE_TEST_CUSTOMER_ID' if kind == 'customer' else 'LINE_TEST_STAFF_ID'
        raise ValueError('請設定 ' + setting)
    return value


def recipient_configured(kind='customer'):
    """Expose only a boolean to pages, never the actual account identifier."""
    try:
        _recipient(kind)
    except ValueError:
        return False
    return True


def ready(kind='customer') -> bool:
    return bool(TOKEN and recipient_configured(kind))


def is_simulation(kind='customer') -> bool:
    return not (SEND_ENABLED and ready(kind))


def same_test_account() -> bool:
    """Only a display flag, no account value is returned to the platform."""
    return bool(recipient_configured('customer') and recipient_configured('staff') and TEST_USER == TEST_STAFF)


def _deliver(text, kind):
    target = _recipient(kind)
    if not SEND_ENABLED:
        return '（模擬發送）未呼叫 LINE API，訊息沒有真的送出。'
    if not TOKEN:
        raise ValueError('LINE 發送設定不完整')
    try:
        with ApiClient(Configuration(access_token=TOKEN)) as client:
            MessagingApi(client).push_message(PushMessageRequest(to=target, messages=[TextMessage(text=text)]))
    except Exception:
        # SDK exceptions can contain the Authorization header or recipient.
        # Never echo them to stdout, a UI or the caller's traceback.
        raise RuntimeError('LINE 發送失敗，請檢查設定，請勿直接重複發送。') from None
    return '已送出到測試帳號。'


def _validate_customer_text(text):
    from llm.explain import check_care

    if type(text) is not str or check_care(text):
        raise ValueError('關懷訊息未通過檢查')


def send_customer_care(text, approved_text=None):
    """Customer-only route, fixed template and matching approval in both modes.

    There is no recipient parameter. Every invocation requires approved_text
    equal to the current text, pages must discard approvals after each attempt.
    """
    _validate_customer_text(text)
    _recipient('customer')
    if type(approved_text) is not str or approved_text != text:
        raise PermissionError('發送需要目前訊息的人工核准')
    return _deliver(text, 'customer')


def send_staff_digest(digest):
    """Staff-only route with no recipient or per-message approval parameter.

    digest must be StaffDigest from notify.staff_digest, not free-form text.
    Content/link/account checks also run with LINE_SEND_ENABLED=false.
    """
    text = render_digest(digest)
    _recipient('staff')
    return _deliver(text, 'staff')


def push_text(text: str, user_id: str | None = None, *, approved_text: str | None = None) -> str:
    """Legacy customer-only adapter, never accepts a staff digest.

    Leader-approved PR #8 exception: only SEND_ENABLED=false and an omitted
    approval may return a compatibility dry-run. It never enters the network
    delivery path. All actual sends delegate to the strict customer API.
    user_id is retained solely for old callers and cannot alter the recipient.
    """
    if user_id is not None and user_id != _recipient('customer'):
        raise ValueError('僅允許設定的 LINE 測試客戶帳號')
    if not SEND_ENABLED and approved_text is None:
        _validate_customer_text(text)
        _recipient('customer')
        return '（相容模擬發送）未呼叫 LINE API，訊息沒有真的送出。'
    return send_customer_care(text, approved_text=approved_text)


if __name__ == "__main__":
    print("CLI 不提供發送功能。請在平台檢查訊息後操作。")
