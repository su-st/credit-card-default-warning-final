"""信用卡預警平台。執行：streamlit run app/main.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / '.env')

import streamlit as st  # noqa: E402
from app.data import BACKTEST, SIMULATION, load_platform_run, page_data  # noqa: E402
from app.views import customer, fairness, risk_list  # noqa: E402

st.set_page_config(page_title='信用卡違約預警平台', layout='wide')

try:
    bundle = load_platform_run()
except Exception as exc:
    st.error(f'平台資料驗證失敗：{exc}')
    st.stop()

st.sidebar.title('違約預警平台')
mode = st.sidebar.radio('模式', [BACKTEST, SIMULATION], key='mode')
df, meta = page_data(bundle, mode)
context = (meta['run_id'], meta['predictions_sha256'], mode)
if st.session_state.get('platform_context') != context:
    for key in list(st.session_state):
        if key.startswith('customer_') or key in ('exp', 'msg'):
            st.session_state.pop(key, None)
    st.session_state['platform_context'] = context

if meta['level'] == 'L1-H':
    st.sidebar.warning('開發資料，非正式結果')
for label, value in [('model_id', meta['model_id']), ('run_id', meta['run_id']),
                     ('policy_id', meta['policy_id']), ('資料層級', meta['level']),
                     ('模式', mode), ('門檻', meta['threshold']), ('規則', meta['rule'])]:
    st.sidebar.caption(f'{label}：{value}')
cap = meta['max_list_rate']
st.sidebar.caption('名單上限：不限制' if cap is None else f'名單上限：{cap:.0%}')
if mode == SIMULATION:
    st.warning('模擬，未串接真實銀行系統')
elif meta['level'] == 'L1-H':
    st.info('L1-H 歷史回測僅供功能確認，不是正式效能結果。')

PAGES = {'每月風險名單': risk_list, '單一客戶查詢': customer, '公平性監控': fairness}
page = st.sidebar.radio('頁面', list(PAGES), key='page')
PAGES[page].render(df, meta)
