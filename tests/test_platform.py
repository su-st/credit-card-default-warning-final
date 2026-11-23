"""C6 bundle/mode/UI tests use synthetic customers and never open real test data."""
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from app import data as platform
from app.views.risk_list import _table
from src import protocol as p
from src.registry import IntegrityError, file_hash, read_json, write_json
from test_registry import synthetic, registered, completed, rewrite, pending
from src.export import export_run


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    monkeypatch.setenv('LINE_SEND_ENABLED', 'false')
    platform._cached_predictions.clear()


def load(completed):
    registry, run, _, _ = completed
    return platform.load_platform_run(run.run_id, root=registry.root)


@pytest.mark.parametrize('key,value', [
    ('run_id','other-run'), ('model_id','other-model'), ('policy_id','other-policy'),
    ('status','running'), ('predictions_verified',False), ('predictions_verified','true'),
    ('predictions_sha256','0'*64), ('model_hash','0'*64), ('level','L2'),
])
def test_reject_inconsistent_or_unverified_run(completed,key,value):
    registry, run, _, _ = completed
    path=run.path/'run.json'
    meta=read_json(path);meta[key]=value;rewrite(path,meta)
    with pytest.raises((IntegrityError,ValueError)):
        load(completed)


@pytest.mark.parametrize('key', ['run_id','model_id','policy_id'])
def test_policy_identifiers_must_match_run(completed,key):
    _, run, _, _ = completed
    path=run.path/'threshold.json'
    policy=read_json(path);policy[key]='other-identity';rewrite(path,policy)
    with pytest.raises(IntegrityError):load(completed)


@pytest.mark.parametrize('key', ['run_id','model_id','policy_id'])
def test_handoff_identity_mismatch_even_when_checksum_is_updated(completed,key):
    _, run, _, _ = completed
    path=run.path/'predictions.csv'
    frame=pd.read_csv(path,converters={'risk_score':float},keep_default_na=False)
    frame.loc[0,key]='other-identity';frame.to_csv(path,index=False,float_format='%.17g')
    meta=read_json(run.path/'run.json');meta['predictions_sha256']=file_hash(path);rewrite(run.path/'run.json',meta)
    with pytest.raises(IntegrityError):load(completed)


@pytest.mark.parametrize("change", ["checksum", "status"])
def test_cached_data_cannot_bypass_checksum_or_completion_checks(completed,change):
    load(completed)  # Populate the cache first.
    _, run, _, _ = completed
    if change == 'checksum':
        with (run.path/'predictions.csv').open('a') as f:f.write('\n')
    else:
        meta=read_json(run.path/'run.json');meta['status']='running';rewrite(run.path/'run.json',meta)
    with pytest.raises(IntegrityError):load(completed)


def test_rank_and_final_list_match_handoff_at_full_precision(completed):
    bundle=load(completed)
    _, _, original, _ = completed
    assert_frame_equal(bundle['predictions'],original['predictions'])
    expected=original['predictions'].loc[original['predictions']['in_list'].eq(1)]
    shown=_table(bundle['predictions'].loc[bundle['predictions']['in_list'].eq(1)])
    assert shown['排名'].tolist()==expected['risk_rank'].tolist()
    assert shown['客戶 ID'].tolist()==expected['customer_id'].tolist()
    np.testing.assert_array_equal(shown['風險分數'],expected['risk_score'])


def test_simulation_allowlist_drops_all_extra_labels_without_mutating_bundle(completed):
    bundle=load(completed)
    for key in ['default','y','target','label','actual_default_future','default payment next month']:
        bundle['predictions'][key]=1
    simulated,meta=platform.page_data(bundle,platform.SIMULATION)
    assert set(simulated)==set(platform.SIMULATION_COLUMNS)
    assert meta['mode']==platform.SIMULATION
    assert 'actual_default' not in simulated
    assert 'actual_default' in bundle['predictions']
    historical,_=platform.page_data(bundle,platform.BACKTEST)
    assert 'actual_default' in historical


def test_list_summary_matches_common_evaluation_and_has_zero_denominator_marker(completed):
    bundle=load(completed);df=bundle['predictions']
    summary=platform.list_summary(df)
    threshold=bundle['policy']
    policy=p.Policy.single(threshold['threshold'],threshold['policy']['max_list_rate'])
    ids=pd.Index(df['customer_id'])
    y=pd.Series(df['actual_default'].to_numpy(),index=ids)
    score=pd.Series(df['risk_score'].to_numpy(),index=ids)
    bill=pd.Series(df['bill_amt_latest'].clip(lower=0).to_numpy(),index=ids)
    expected=p.evaluate(y,score=score,policy=policy,bill=bill,scenario=p.Scenario(1,.5,1,.2))['overall'].iloc[0]
    assert summary['list_count']==expected['名單人數']
    assert summary['actual_defaults_in_list']==int((df['actual_default'].eq(1)&df['in_list'].eq(1)).sum())
    assert summary['default_coverage']==expected['違約者涵蓋率']
    assert summary['bill_coverage']==expected['帳單金額涵蓋率']
    no_defaults=df.assign(actual_default=0)
    assert platform.list_summary(no_defaults)['default_coverage'] is None
    assert platform.list_summary(no_defaults)['bill_coverage'] is None


def app(completed,monkeypatch):
    registry,run,_,_=completed
    monkeypatch.setattr(platform,'ROOT',registry.root)
    monkeypatch.setenv('PLATFORM_RUN_ID',run.run_id)
    return AppTest.from_file(str(Path(__file__).parents[1]/'app/main.py'),default_timeout=30).run()


def test_ui_modes_browsing_and_fairness_gate(completed,monkeypatch):
    ui=app(completed,monkeypatch)
    assert not ui.exception
    labels=[m.label for m in ui.metric]
    assert '名單中實際違約人數' in labels
    captions=' '.join(c.value for c in ui.sidebar.caption)
    for key in ['model_id','run_id','policy_id','門檻','規則','資料層級','模式','名單上限']:
        assert key in captions
    expected_count=ui.metric[0].value
    expected_ids=ui.dataframe[0].value['客戶 ID'].tolist()
    ui.slider(key='browse_count').set_value(17).run()
    assert not ui.exception
    assert ui.metric[0].value==expected_count
    assert ui.dataframe[0].value['客戶 ID'].tolist()==expected_ids
    ui.radio(key='mode').set_value(platform.SIMULATION).run()
    assert not ui.exception
    assert [m.label for m in ui.metric]==['預警名單人數']
    assert any('模擬，未串接真實銀行系統' in w.value for w in ui.warning)
    ui.radio(key='page').set_value('公平性監控').run()
    assert not ui.exception and not ui.metric and not ui.dataframe
    assert any('僅在歷史回測' in info.value for info in ui.info)
    ui.radio(key='page').set_value('單一客戶查詢').run()
    assert not ui.exception
    assert all('實際違約' not in m.label and '涵蓋' not in m.label for m in ui.metric)
    ui.radio(key='mode').set_value(platform.BACKTEST).run()
    ui.radio(key='page').set_value('公平性監控').run()
    assert not ui.exception and ui.metric and ui.dataframe


def test_ui_mode_switch_invalidates_customer_approval(completed,monkeypatch):
    ui=app(completed,monkeypatch)
    ui.session_state['customer_approval']=(1,'synthetic stale approval')
    ui.session_state['customer_message']='synthetic stale text'
    ui.radio(key='mode').set_value(platform.SIMULATION).run()
    assert not ui.exception
    assert 'customer_approval' not in ui.session_state
    assert 'customer_message' not in ui.session_state


def test_ui_invalid_run_stops_before_rendering_data(completed,monkeypatch):
    _,run,_,_=completed
    meta=read_json(run.path/'run.json');meta['status']='running';rewrite(run.path/'run.json',meta)
    ui=app(completed,monkeypatch)
    assert not ui.exception
    assert ui.error and '驗證失敗' in ui.error[0].value
    assert not ui.dataframe and not ui.metric


def test_platform_wording_and_localhost_config():
    root=Path(__file__).parents[1]
    prohibited='違約'+'機率'
    for path in (root/'app').rglob('*'):
        if path.is_file():
            assert prohibited.encode() not in path.read_bytes(),path
    config=tomllib.loads((root/'.streamlit/config.toml').read_text())
    assert config['server']['address']=='localhost'
    assert config['browser']['gatherUsageStats'] is False


def test_formal_level_rejected_before_handoff_access(tmp_path,monkeypatch):
    folder=tmp_path/'outputs/runs/synthetic-blocked';folder.mkdir(parents=True)
    write_json(folder/'run.json',{'run_id':'synthetic-blocked','level':'L2','status':'complete',
                                 'predictions_verified':True,'predictions_sha256':'a'*64})
    monkeypatch.setattr(platform,'file_hash',lambda *a:pytest.fail('accessed formal handoff'))
    with pytest.raises(IntegrityError,match='platform-c'):
        platform.load_platform_run('synthetic-blocked',root=tmp_path)


def test_run_switch_refreshes_cache_and_customer_approval(completed,monkeypatch):
    registry,first,original,fixture=completed
    old=load(completed)
    ui=app(completed,monkeypatch)
    ui.session_state['customer_approval']=(1,'stale')
    second,_=pending(registry,original['model_record'],threshold=.9,cap=.1)
    export_run(second,fixture.hold)
    new=platform.load_platform_run(second.run_id,root=registry.root)
    assert set(old['predictions']['run_id'])=={first.run_id}
    assert set(new['predictions']['run_id'])=={second.run_id}
    assert old['run']['predictions_sha256']!=new['run']['predictions_sha256']
    monkeypatch.setenv('PLATFORM_RUN_ID',second.run_id)
    ui.run()
    assert not ui.exception and 'customer_approval' not in ui.session_state
    assert any(second.run_id in c.value for c in ui.sidebar.caption)


@pytest.mark.parametrize('column',['risk_rank','in_list'])
def test_reject_forged_ranking_or_list_even_with_updated_checksum(completed,column):
    _,run,_,_=completed
    path=run.path/'predictions.csv'
    frame=pd.read_csv(path,converters={'risk_score':float},keep_default_na=False)
    frame.loc[0,column]=99 if column=='risk_rank' else 1-int(frame.loc[0,column])
    frame.to_csv(path,index=False,float_format='%.17g')
    meta=read_json(run.path/'run.json');meta['predictions_sha256']=file_hash(path);rewrite(run.path/'run.json',meta)
    with pytest.raises(IntegrityError):load(completed)
