"""Presentation checks on synthetic aggregates, no real records or calculations."""
from copy import deepcopy
import json

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from app.views import fairness as f


@pytest.mark.parametrize('group,labels,expected', [
    ('年齡層', ['50 歲以上','30-39 歲','29 歲以下','40-49 歲'], list(f.AGE_GROUP_ORDER)),
    ('性別', ['女性','男性'], ['男性','女性']),
    ('教育程度', ['高中','其他','研究所','大學'], ['高中','其他','研究所','大學']),
    ('婚姻狀況', ['單身','已婚','其他'], ['單身','已婚','其他']),
    ('性別×年齡層', ['女性 × 50 歲以上','男性 × 29 歲以下'], ['男性 × 29 歲以下','女性 × 50 歲以上']),
    ('年齡層', ['未知甲','50 歲以上','未知乙','29 歲以下'], ['29 歲以下','50 歲以上','未知甲','未知乙']),
    ('性別', ['未知甲','女性','未知乙','男性'], ['男性','女性','未知甲','未知乙']),
    ('年齡層', ['30-39歲','29歲以下'], ['29歲以下','30-39歲']),
    ('年齡層', [], []),
])
def test_group_order(group,labels,expected):
    original=labels.copy()
    assert f._group_order(group,labels)==expected
    assert labels==original


@pytest.mark.parametrize('labels', [
    ['女性×50 歲以上','男性×40-49 歲','女性×29 歲以下','男性×50 歲以上',
     '男性×29 歲以下','女性×40-49 歲','男性×30-39 歲','女性×30-39 歲'],
    ['女性 × 50 歲以上','男性 × 40-49 歲','女性 × 29 歲以下','男性 × 50 歲以上',
     '男性 × 29 歲以下','女性 × 40-49 歲','男性 × 30-39 歲','女性 × 30-39 歲'],
])
def test_cross_group_full_order_and_unknown_last(labels):
    unknown=['未識別×40-49 歲','男性×未知年齡','其他標籤']
    incoming=[unknown[0],*labels,unknown[1],unknown[2]]
    expected=[next(label for label in labels if label.replace(' ','')==(sex+'×'+age).replace(' ',''))
              for sex in ('男性','女性') for age in f.AGE_GROUP_ORDER]
    assert f._group_order('性別×年齡層',incoming)==expected+unknown


def groups(partition='年齡層'):
    rows=[]
    labels=f.AGE_GROUP_ORDER if partition=='年齡層' else [sex+'×'+age for sex in ('男性','女性') for age in f.AGE_GROUP_ORDER]
    for i,label in enumerate(reversed(labels)):
        r=dict.fromkeys(f.GROUP_COLUMNS,.123456789012345)
        r.update(分組方式=partition,族群=label,總人數=100+i,違約人數=40+i,未違約人數=60,小樣本=True)
        r['Recall']=float('nan') if i==1 else .4+i*.01
        rows.append(r)
    return pd.DataFrame(rows)


@pytest.mark.parametrize('partition',['年齡層','性別×年齡層'])
def test_all_three_group_tables_share_order_without_changing_values(monkeypatch,partition):
    frame=groups(partition);original=frame.copy(deep=True);shown=[]
    monkeypatch.setattr(f.st,'dataframe',lambda df,**kwargs:shown.append(df.copy()))
    monkeypatch.setattr(f.st,'vega_lite_chart',lambda *args,**kwargs:None)
    gaps=pd.DataFrame([{'分組方式':partition,'EO 差距':.1,'FPR 差距':.2}])
    f._show_results(frame,gaps,partition)
    result={'groups':frame,'source':{'level':'L2','model_id':'fake-model','run_id':'fake-run'},
            'overall':pd.Series(dict(AUC=.8,Precision=.5,Recall=.4,F1=.45,名單人數=20,名單比例=.2,policy_id='fake-policy')),
            'gaps':gaps.iloc[0],
            'ci':pd.DataFrame([dict(指標=m,估計值=.1,下限=0,上限=.2,有效重抽=2000,無效重抽=0,可靠=True) for m in f.GAP_METRICS])}
    f._show_model_result('M1',result);f._show_model_result('M1-NS',result)
    tables=[t for t in shown if '族群' in t]
    assert len(tables)==3
    for table in tables:
        order=f._group_order(partition,frame['族群'])
        assert table['族群'].tolist()==order
        expected=frame.set_index('族群').loc[order]
        assert table['總人數'].tolist()==expected['總人數'].tolist()
        assert table['Recall'].tolist()==expected['Recall'].map(f._format_rate).tolist()
    assert_frame_equal(frame,original)


@pytest.mark.parametrize('partition',['年齡層','性別×年齡層'])
def test_chart_specs_colors_order_axes_and_missing_values(monkeypatch,partition):
    captured=[]
    monkeypatch.setattr(f.st,'vega_lite_chart',lambda data,spec,**kw:captured.append((data.copy(),deepcopy(spec))))
    frame=groups(partition);before=frame.copy(deep=True)
    gaps=pd.DataFrame([{'分組方式':partition,'EO 差距':float('nan'),'FPR 差距':.2}])
    f._show_results(frame,gaps,partition)
    selected={m:{'groups':frame,'gaps':gaps.iloc[0]} for m in reversed(f.COMPARISON_MODEL_ORDER)}
    f._show_comparison_charts(selected,partition)
    assert len(captured)==4
    for i,(data,spec) in enumerate(captured):
        enc=spec['encoding']
        axis=enc['x']['axis']
        assert axis['labelAngle']==0 and axis['labelLimit']==0 and axis['labelOverlap'] is False
        assert axis['labelExpr']=="indexof(datum.label, '×') >= 0 ? split(datum.label, '×') : datum.label"
        if i!=1:assert enc['x']['sort']==f._group_order(partition,frame['族群'])
        assert enc['color']['legend']=={'title':None,'orient':'top'}
        assert enc['x']['axis']['title']==('指標' if i==1 else '族群')
        order=f.PLATFORM_METRIC_ORDER if i==0 else f.COMPARISON_MODEL_ORDER
        colors=f.PLATFORM_METRIC_COLORS if i==0 else f.COMPARISON_MODEL_COLORS
        assert enc['xOffset']['sort']==list(order)
        assert enc['color']['scale']=={'domain':list(order),'range':list(colors)}
        assert data['value'].isna().any() if i!=3 else data['value'].notna().all()
    assert not set(f.PLATFORM_METRIC_COLORS)&set(f.COMPARISON_MODEL_COLORS)
    assert captured[2][1]['encoding']['y']['scale']['domain']==[0,1]
    assert 'scale' not in captured[3][1]['encoding']['y']
    assert_frame_equal(frame,before)


def test_synthetic_chart_renders_in_apptest():
    def script():
        import pandas as pd
        from app.views.fairness import _show_grouped_metric_chart,PLATFORM_METRIC_ORDER,PLATFORM_METRIC_COLORS
        _show_grouped_metric_chart(pd.DataFrame({'Recall':[.4,None],'FPR':[.1,.2]},index=['男性','女性']),
            '比例',x_title='族群',series_order=PLATFORM_METRIC_ORDER,series_colors=PLATFORM_METRIC_COLORS)
    ui=AppTest.from_function(script).run()
    assert not ui.exception
    charts=ui.get('vega_lite_chart');assert len(charts)==1
    spec=json.loads(charts[0].proto.spec)
    assert spec['encoding']['x']['axis']['title']=='族群'
