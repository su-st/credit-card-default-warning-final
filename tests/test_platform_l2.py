"""Synthetic L2 handoff, saved fairness and page switching. No real data or sends."""
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import data
from src.registry import IntegrityError, Registry, file_hash
from test_platform_fairness_data import comparison


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def formal(comparison, monkeypatch):
    root = comparison
    freeze_path = root/'outputs/frozen/model-g2b-synthetic.json'
    save(freeze_path, {'synthetic': True})
    shared = dict(level='L2', status='complete', git_dirty=False, rules_freeze_id='g2a-20261116',
                  model_freeze_id='g2b-synthetic', model_freeze_sha256=file_hash(freeze_path),
                  summary_id='summary-synthetic', model_hash='synthetic-model-hash', policy_hash='synthetic-policy-hash')
    for rid in ('m1-run', 'ns-run'):
        path=root/'results'/rid/'run.json'
        old=json.loads(path.read_text());save(path, {**old, **shared})
    frame=pd.DataFrame({key:['']*6000 for key in data.SIMULATION_COLUMNS})
    frame=frame.assign(customer_id=range(1,6001), risk_rank=range(1,6001),
        risk_score=[.8]*1200+[.1]*4800, in_list=[1]*1200+[0]*4800,
        actual_default=[1,0]*3000, bill_amt_latest=100, limit_bal=1000,
        data_split='test', model_id='ns-model', run_id='platform-run', policy_id='c-policy',
        factor_1='人工風險因素', sex='男', age=30, age_group='30–39', education='大學', marriage='單身')
    folder=root/'outputs/runs/platform-run';folder.mkdir(parents=True)
    frame.to_csv(folder/'predictions.csv',index=False)
    meta=dict(shared, run_id='platform-run',model_id='ns-model',policy_id='c-policy',entry='platform-c',
              predictions_verified=True,verified_customers=6000,predictions_sha256=file_hash(folder/'predictions.csv'))
    save(folder/'run.json',meta);save(root/'results/platform-run/run.json',meta)
    policy=dict(threshold=.61,source_level='L1',protocol_policy=dict(kind='single',selection='cost',max_list_rate=.2))
    bundle=dict(run=meta,model_record=dict(model_id='ns-model'),threshold=policy,predictions=frame)
    loader=Mock(return_value=bundle);monkeypatch.setattr(Registry,'load_run',loader)
    # Saved synthetic formal-policy rows, not recomputed by any reader or page.
    for name,src in [('groups','groups'),('gaps','gaps_point'),('gap_ci','gaps')]:
        t=pd.read_csv(root/'results/ns-run'/f'fairness_{src}.csv')
        t=t.loc[t['模型'].eq('M1-NS') & t['政策比較'].eq('正式政策')].copy()
        t=t.assign(run_id='platform-run',model_id='ns-model',policy_id='c-policy',entry_id='platform-c',
                   level='L2',model_freeze_id='g2b-synthetic',rules_freeze_id='g2a-20261116')
        t.to_csv(root/'results/platform-run'/f'{name}.csv',index=False)
    manifest=dict(shared,summary_id='summary-synthetic',tasks={})
    for task,rid in [('platform-c','platform-run'),('fairness-M1','m1-run'),('fairness-M1-NS','ns-run')]:
        folder=root/'results'/rid
        manifest['tasks'][task]=dict(status='complete',run_id=rid,model_hash=shared['model_hash'],
            policy_hash=shared['policy_hash'],run_sha256=file_hash(folder/'run.json'),
            result_sha256={p.name:file_hash(p) for p in folder.glob('*.csv')})
    save(root/'results/summary-synthetic/evaluation_manifest.json',manifest)
    monkeypatch.setattr(data,'ROOT',root)
    monkeypatch.setenv('PLATFORM_RUN_ID','platform-run')
    monkeypatch.setenv('PLATFORM_FAIRNESS_RUN_ID','ns-run')
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED','1')
    from notify import line_push
    monkeypatch.setattr(line_push,'SEND_ENABLED',False)
    for name in ('send_staff_digest','send_customer_care','push_text'):
        monkeypatch.setattr(line_push,name,Mock(side_effect=AssertionError('No send permitted')))
    data._cached_predictions.clear()
    return root,meta,frame,loader


def test_formal_bundle_saved_fairness_and_no_fallback(formal, monkeypatch):
    root,meta,frame,loader=formal
    b=data.load_platform_run();pd.testing.assert_frame_equal(b['predictions'],frame)
    assert b['policy']['source_level']=='L1'
    assert len(b['predictions'])==6000 and b['predictions'].in_list.sum()==1200
    f=data.load_platform_fairness();assert f['groups'].shape[0]==5 and f['gaps'].shape[0]==5
    compare=data.load_fairness_comparison(platform_run=meta)
    assert compare['source']['m1_run_id']=='m1-run'
    simulated,_=data.page_data(b,data.SIMULATION)
    assert set(simulated)==set(data.SIMULATION_COLUMNS) and 'actual_default' not in simulated
    assert simulated.factor_1.tolist()==frame.factor_1.tolist()
    monkeypatch.delenv('PLATFORM_RUN_ID')
    with pytest.raises(ValueError):data.load_platform_run()


@pytest.mark.parametrize('key,value', [('entry','platform-n'),('git_dirty',True),('status','failed'),
    ('predictions_verified',False),('verified_customers',5999),('model_freeze_sha256','wrong'),
    ('model_freeze_id','other'),('summary_id','missing'),('predictions_sha256','wrong')])
def test_invalid_formal_never_loads_handoff_or_falls_back(formal,key,value):
    root,meta,_,loader=formal
    save(root/'outputs/runs/platform-run/run.json',{**meta,key:value})
    with pytest.raises((IntegrityError,ValueError,OSError)):data.load_platform_run()
    loader.assert_not_called()


@pytest.mark.parametrize('change',['count','split','model','policy','run','source_level'])
def test_mixed_handoff_or_policy_rejected(formal,change):
    _,_,_,loader=formal;b=loader.return_value
    if change=='count':b['predictions']=b['predictions'].iloc[:-1]
    elif change=='split':b['predictions']=b['predictions'].assign(data_split='dev_holdout')
    elif change=='source_level':b['threshold']['source_level']='L1-H'
    else:b['predictions']=b['predictions'].assign(**{change+'_id':'development'})
    with pytest.raises(IntegrityError):data.load_platform_run()


@pytest.mark.parametrize('filename',['groups.csv','gaps.csv','gap_ci.csv','run.json'])
def test_saved_fairness_corruption_rejected(formal,filename):
    root,_,_,_=formal;p=root/'results/platform-run'/filename
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises((IntegrityError,ValueError)):data.load_platform_fairness()


def test_f3_must_match_platform_summary_and_freeze(formal):
    _,meta,_,_=formal
    for key in ('summary_id','model_freeze_id','rules_freeze_id'):
        with pytest.raises(IntegrityError):data.load_fairness_comparison(platform_run={**meta,key:'other'})


def test_formal_pages_and_switch_clear_drafts(formal,monkeypatch):
    root,meta,_,_=formal
    ui=AppTest.from_file(str(Path(__file__).parents[1]/'app/main.py'),default_timeout=30).run()
    assert not ui.exception and not ui.error
    assert any('正式歷史回測（L2 測試集）' in x.value for x in ui.info)
    assert any('6,000' in x.value for x in ui.caption)
    assert ui.metric[0].value=='1,200'
    ui.radio(key='page').set_value('公平性監控').run()
    assert not ui.exception and not ui.error and len(ui.metric)>=6 and len(ui.dataframe)>=4
    ui.session_state['customer_message']='old draft';ui.session_state['customer_approval']=(1,'old draft')
    ui.radio(key='mode').set_value(data.SIMULATION).run()
    assert not ui.exception and not ui.metric and not ui.dataframe
    assert 'customer_approval' not in ui.session_state and 'customer_message' not in ui.session_state
    ui.radio(key='page').set_value('單一客戶查詢').run()
    assert not ui.exception
    assert all('實際' not in x.label and '涵蓋' not in x.label for x in ui.metric)
    monkeypatch.setenv('PLATFORM_RUN_ID','nonexistent-formal')
    ui.run();assert ui.error and not ui.metric and not ui.dataframe

from test_batch_run import setup_batch


def test_batch_current_run_separated_without_resetting_duplicate_guard(setup_batch, monkeypatch):
    from app import batch_run as batch
    b,_,sender=setup_batch
    first=batch.run_batch('2026-11');assert first['notified']
    b['run'].update(run_id='formal-run',level='L2')
    b['predictions']=b['predictions'].assign(run_id='formal-run')
    assert batch.latest_batch(run_id='formal-run') is None
    assert batch.latest_batch(run_id='fake-run')['notified']
    script="from app.views.risk_list import _batch_panel\n_batch_panel({'run_id':'formal-run','level':'L2'})"
    ui=AppTest.from_string(script).run()
    assert not ui.exception
    assert any('不代表目前資料已通知' in x.value for x in ui.info)
    second=batch.run_batch('2026-11')
    assert second['status']=='已推播，略過' and sender.call_count==1
    assert second['notified'] is False  # Another run's notification does not notify this run.
    assert batch.latest_batch(run_id='formal-run')['run_id']=='formal-run'


def test_formal_batch_simulates_only_staff_and_preserves_handoff(setup_batch, monkeypatch):
    from app import batch_run as batch
    from notify import line_push
    b,_,sender=setup_batch
    b['run'].update(run_id='formal-run',level='L2')
    b['predictions']=b['predictions'].assign(run_id='formal-run')
    monkeypatch.setattr(line_push,'SEND_ENABLED',False)
    r=batch.run_batch('2026-11')
    assert r['level']=='L2' and not r['notified'] and not r['delivery_attempted']
    assert sender.call_count==1 and r['list_count']==2
