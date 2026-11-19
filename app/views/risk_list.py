"""每月風險名單：交接政策決定名單，瀏覽範圍不改變名單。"""
import streamlit as st

from app.data import BACKTEST, list_summary


def _table(frame):
    return frame[['risk_rank', 'customer_id', 'risk_score', 'factor_1', 'factor_2', 'factor_3', 'limit_bal']].rename(
        columns={'risk_rank': '排名', 'customer_id': '客戶 ID', 'risk_score': '風險分數',
                 'factor_1': '風險因素 1', 'factor_2': '風險因素 2', 'factor_3': '風險因素 3',
                 'limit_bal': '信用額度'})


def _show(frame):
    st.dataframe(frame, hide_index=True, width='stretch',
                 column_config={'風險分數': st.column_config.NumberColumn(format='%.4f')})


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
