"""PR #42 identifier and bootstrap-cache regressions, synthetic summaries only."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src import _evaluation as ev, protocol as p
from src.registry import Registry, read_json, write_json


def test_identifier_reader_preserves_text_and_numeric_metrics(tmp_path):
    path=tmp_path/'table.csv'
    values=dict(model_key='05',entry_id='0005',comparison_id='0001',model_id='0002',
        policy_id='0003',run_id='0004',model_freeze_id='0006',source_run_id='NA',
        custom_source_id='0007',model_hash='0008',AUC=.8123456789012345,Precision=np.nan,seed=42)
    pd.DataFrame([values]).to_csv(path,index=False)
    frame=ev.read_result_csv(path)
    for name,value in values.items():
        if isinstance(value,str):
            assert frame.at[0,name]==value and isinstance(frame.at[0,name],str)
    assert pd.api.types.is_numeric_dtype(frame.AUC)
    assert np.isclose(frame.at[0,'AUC'],values['AUC'])
    assert pd.isna(frame.at[0,'Precision']) and frame.at[0,'seed']==42


@pytest.mark.parametrize('level',['L2','L3'])
def test_summary_files_preserve_05_and_other_identifiers(tmp_path,monkeypatch,level):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    seeds=list(range(42,52)) if level=='L3' else [42]
    monkeypatch.setattr(p,'load_rules',lambda *a:SimpleNamespace(data_protocol={'l3_seeds':seeds}))
    registry=Registry(tmp_path)
    monkeypatch.setattr(Registry,'load_run',lambda self,run_id:dict(run=read_json(self.runs/run_id/'run.json')))
    tasks=[f'{seed}/{kind}' for seed in seeds for kind in ('05','rq3')]
    batch=ev.Batch(level,dict(model_freeze_id='model-freeze',rules_freeze_id='rules'),'a'*64,tasks)
    for seed in seeds:
        for kind in ('05','rq3'):
            run_id=f'000{seed}{kind}'
            folder=registry.results/run_id;folder.mkdir(parents=True)
            private=registry.runs/run_id;private.mkdir(parents=True)
            entry_id='0005' if kind=='05' else 'rq3'
            meta=dict(run_id=run_id,model_id='0002',model_hash='b'*64,policy_id='0003',
                status='complete',seed=seed,entry=entry_id,comparison_id='0001',level=level,
                model_freeze_id='model-freeze',rules_freeze_id='rules')
            write_json(private/'run.json',meta);write_json(folder/'run.json',meta)
            tags=dict(entry_id=entry_id,model_key=kind,comparison_id='0001',model_id='0002',
                policy_id='0003',run_id=run_id,model_freeze_id='model-freeze',rules_freeze_id='rules',
                score_version='raw',policy_kind='c',lag=0,seed=seed,is_primary=True)
            if kind=='05':
                row=dict(tags,AUC=.8,Accuracy=.7,Precision=np.nan,Recall=.6,F1=.5,fit_seconds=1.2)
                pd.DataFrame([row]).to_csv(folder/'table_4_1_row.csv',index=False)
                pd.DataFrame([dict(row,label='1',support=10)]).to_csv(folder/'classification_report.csv',index=False)
            else:
                pd.DataFrame([dict(tags,AUC=.75,fit_seconds=1.2)]).to_csv(folder/'rq3_lead_time.csv',index=False)
                pd.DataFrame([dict(tags,名單比例=.2,違約者涵蓋率=.6,帳單金額涵蓋率=.5)]).to_csv(folder/'rq3_coverage.csv',index=False)
            batch.meta['tasks'][f'{seed}/{kind}'].update(status='running',run_id=run_id)
            batch.done(f'{seed}/{kind}',SimpleNamespace(meta=meta,run_id=run_id,path=private,result_path=folder))
    batch.finish()
    filenames=('table_4_1_l2.csv','table_4_3_l2.csv') if level=='L2' else ('stability_by_seed.csv','stability_summary.csv')
    for name in filenames:
        # Read as text to inspect what was actually written, independently of
        # the new reader. Casting an already-written 5 to str cannot hide the bug.
        frame=pd.read_csv(batch.path/name,dtype=str)
        assert not frame.empty and frame.model_key.eq('05').all()
        assert frame.entry_id.eq('0005').all() and frame.comparison_id.eq('0001').all()
        for column,value in [('model_id','0002'),('policy_id','0003')]:
            if column in frame:assert frame[column].eq(value).all()
        if 'run_id' in frame:assert frame.run_id.str.startswith('000').all()


@pytest.fixture
def gap_inputs():
    ids=pd.Index(range(1,13))
    y=pd.Series([0,1,0,1]*3,index=ids)
    pred=pd.Series([0,0,1,1,0,1,0,0,1,1,0,1],index=ids)
    groups={'sex':pd.Series(['A']*6+['B']*6,index=ids),
            'age':pd.Series(['X','X','Y','Y']*3,index=ids)}
    rules=SimpleNamespace(d8={'bootstrap':{'n_boot':20,'seed':42}})
    return y,pred,groups,rules


@pytest.mark.parametrize('paired',[False,True])
def test_gap_cache_is_exact_and_immune_to_caller_mutation(gap_inputs,monkeypatch,paired):
    y,pred,groups,rules=gap_inputs
    pred_b=1-pred if paired else None
    expected=ev.gap_tables(y,pred,groups,rules,pred_b=pred_b)
    spy=Mock(wraps=p.gap_ci);monkeypatch.setattr(p,'gap_ci',spy)
    cache={}
    first=ev.gap_tables(y,pred,groups,rules,pred_b=pred_b,cache=cache)
    pd.testing.assert_frame_equal(first,expected,check_exact=True)
    first.loc[0,'估計值']=999
    again=ev.gap_tables(y.copy(),pred.copy(),{k:g.copy() for k,g in groups.items()},
        deepcopy(rules),pred_b=None if pred_b is None else pred_b.copy(),cache=cache)
    pd.testing.assert_frame_equal(again,expected,check_exact=True)
    again.loc[0,'下限']=999
    pd.testing.assert_frame_equal(ev.gap_tables(y,pred,groups,rules,pred_b=pred_b,cache=cache),expected,check_exact=True)
    assert spy.call_count==len(groups) and len(cache)==1


@pytest.mark.parametrize('change',['y','pred','pred_b','partition','group_name','group_order','ids','n_boot','seed'])
def test_gap_cache_keys_all_content_and_never_reuses_different_inputs(gap_inputs,monkeypatch,change):
    y,pred,groups,rules=gap_inputs
    cache={};ev.gap_tables(y,pred,groups,rules,cache=cache)
    y=y.copy();pred=pred.copy();groups={k:g.copy() for k,g in groups.items()};rules=deepcopy(rules);pred_b=None
    if change=='y':y.iloc[0]=1-y.iloc[0]
    elif change=='pred':pred.iloc[1]=1-pred.iloc[1]
    elif change=='pred_b':pred_b=1-pred
    elif change=='partition':groups['sex'].iloc[0]='B'
    elif change=='group_name':groups={'other':groups['sex'],'age':groups['age']}
    elif change=='group_order':groups=dict(reversed(list(groups.items())))
    elif change=='ids':
        y.index=pred.index=pd.Index(range(101,113))
        for group in groups.values():group.index=y.index
    elif change=='n_boot':rules.d8['bootstrap']['n_boot']+=1
    else:rules.d8['bootstrap']['seed']+=1
    expected=ev.gap_tables(y,pred,groups,rules,pred_b=pred_b)
    spy=Mock(wraps=p.gap_ci);monkeypatch.setattr(p,'gap_ci',spy)
    actual=ev.gap_tables(y,pred,groups,rules,pred_b=pred_b,cache=cache)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    assert spy.call_count==len(groups) and len(cache)==2


def test_cache_is_scoped_to_a_batch(gap_inputs,tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    freeze=dict(model_freeze_id='freeze',rules_freeze_id='rules')
    first=ev.Batch('L2',freeze,'a'*64,[]);second=ev.Batch('L2',freeze,'a'*64,[])
    spy=Mock(wraps=p.gap_ci);monkeypatch.setattr(p,'gap_ci',spy)
    y,pred,groups,rules=gap_inputs
    for cache in (first.gap_cache,first.gap_cache,second.gap_cache):
        ev.gap_tables(y,pred,groups,rules,cache=cache)
    assert first.gap_cache is not second.gap_cache
    assert spy.call_count==2*len(groups)
