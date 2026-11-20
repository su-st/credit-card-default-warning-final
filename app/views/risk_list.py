"""每月風險名單：交接政策決定名單，瀏覽範圍不改變名單。"""
import streamlit as st

from app.data import BACKTEST, list_summary
from notify import line_push
from notify.staff_digest import PREVIEW_LIMIT, build_digest, render_digest


def _table(frame):
    return frame[['risk_rank', 'customer_id', 'risk_score', 'factor_1', 'factor_2', 'factor_3', 'limit_bal']].rename(
        columns={'risk_rank': '排名', 'customer_id': '客戶 ID', 'risk_score': '風險分數',
                 'factor_1': '風險因素 1', 'factor_2': '風險因素 2', 'factor_3': '風險因素 3',
                 'limit_bal': '信用額度'})


def _show(frame):
    st.dataframe(frame, hide_index=True, width='stretch',
                 column_config={'風險分數': st.column_config.NumberColumn(format='%.4f')})


def _staff_panel(frame):
    st.subheader('行員名單摘要')
    if line_push.same_test_account():
        st.warning('行員與客戶由同一測試帳號模擬')
    try:
        digest = build_digest(frame, platform_url=line_push.PLATFORM_URL)
        preview = render_digest(digest)
    except (ValueError, TypeError, OverflowError):
        st.error('無法建立行員摘要，請檢查交接名單與本機或測試連結設定。')
        return
    st.caption(f'摘要最多列出前 {PREVIEW_LIMIT} 位，名單總人數依交接政策，完整名單請在本機或測試平台查看。')
    st.code(preview, language=None)
    simulation = line_push.is_simulation('staff')
    st.caption('LINE 行員摘要：模擬，不會對外發送。' if simulation else 'LINE 行員摘要：實際發送至設定的測試行員。')
    configured = line_push.recipient_configured('staff')
    enabled = configured and (not line_push.SEND_ENABLED or line_push.ready('staff'))
    if not enabled:
        st.caption('行員通知設定尚未完成，目前僅供預覽。')
    if st.button('發送行員名單摘要', key='staff_digest_send', disabled=not enabled):
        try:
            result = line_push.send_staff_digest(digest)
        except Exception:
            st.error('行員摘要發送失敗，請檢查設定，請勿直接重複發送。')
        else:
            if simulation:
                st.info(result)
            else:
                st.success(result)


def render(df, meta):
    st.header('每月風險名單')
    st.caption('預警名單依本次政策決定並按風險排名顯示，瀏覽範圍不會改變預警名單。')
    selected = df.loc[df['in_list'].eq(1)]
    if meta['mode'] == BACKTEST:
        summary = list_summary(df)
        a, b, c, d = st.columns(4)
        a.metric('預警名單人數', f"{summary['list_count']:,}")
        b.metric('名單中實際違約人數', f"{summary['actual_defaults_in_list']:,}")
        c.metric('涵蓋違約者比例', '無法計算' if summary['default_coverage'] is None else f"{summary['default_coverage']:.1%}")
        d.metric('涵蓋違約者帳單金額比例', '無法計算' if summary['bill_coverage'] is None else f"{summary['bill_coverage']:.1%}")
    else:
        st.metric('預警名單人數', f'{len(selected):,}')
    st.caption('模型判斷依據，不代表違約原因')
    st.subheader('預警名單')
    show = _table(selected)
    _show(show)
    st.download_button('下載預警名單 CSV', show.to_csv(index=False).encode('utf-8-sig'), 'risk_list.csv')
    with st.expander('瀏覽範圍（不改變預警名單）'):
        n = st.slider('瀏覽前幾名', min_value=1, max_value=len(df),
                      value=min(max(len(selected), 50), len(df)), step=1, key='browse_count')
        st.caption(f'瀏覽排名前 {n:,} 人，政策預警名單仍為 {len(selected):,} 人。')
        _show(_table(df.head(n)))
    _staff_panel(df)
