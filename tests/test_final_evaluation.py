"""C9b-2 synthetic releases only. Real customer/test/sealed access is forbidden."""
from copy import deepcopy
import importlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src import model_freeze as mf, protocol as p, final_eval as final, stability_eval as stability
from src import _evaluation as ev, _protocol_runs as pr, registry as registry_module
from src.registry import Registry, IntegrityError, content_hash, file_hash, read_json, write_json
from test_model_freeze import sandbox


@pytest.fixture
def release(sandbox, monkeypatch):
    (root, train, rules)=sandbox
    monkeypatch.setattr(registry_module,'git_dirty',lambda *args:False)
    monkeypatch.setattr(pr,'git_dirty',lambda *args:False)
    # The C6b reader requires this literal rules ID. Only the temporary copy is
    # synthetic, no file in the real project frozen directory is changed.
    rules=deepcopy(rules);rules['rules_freeze_id']='g2a-20261116'
    write_json(root/'outputs/frozen/rules-g2a-20261116.json',rules)
    draft=mf.prepare('g2a-20261116')
    frozen=mf.freeze(draft,model_freeze_id='g2b-20261123',approved_at='2026-11-23T10:00:00+08:00',platform_model='M1-NS')
    frame=train.iloc[np.arange(6000)%len(train)].copy()
    frame.index=pd.Index(range(24001,30001),name='ID')
    frame.attrs.update(level='L2',model_freeze_id='g2b-20261123',seed=42)
    loader=Mock(return_value=frame)
    monkeypatch.setattr(p,'final_test',loader)
    def fake_ci(y,pred,partition,*,n_boot,seed,pred_b=None):
        assert n_boot==2000 and seed==42
        assert y.index.equals(pred.index) and y.index.equals(partition.index)
        if pred_b is not None:assert y.index.equals(pred_b.index)
        return pd.DataFrame([dict(指標=name,估計值=.1,下限=-.1,上限=.2,有效重抽=n_boot,
                                 無效重抽=0,可靠=True,比較='B-A' if pred_b is not None else '單一名單')
                             for name in ('EO 差距','FPR 差距')])
    monkeypatch.setattr(p,'gap_ci',fake_ci)
    monkeypatch.setattr(final,'top_factors',lambda model,X:[['factor A','factor B','factor C'] for _ in range(len(X))])
    return root,train,frame,frozen,loader


def test_l2_complete_batch_reports_handoff_and_fairness_reader(release,monkeypatch):
    root,train,frame,path,loader=release
    def forbidden(*args,**kwargs):pytest.fail('L2 fitted or selected a setting')
    for name in ('choose_threshold','oof_predict','make_model'):monkeypatch.setattr(p,name,forbidden)
    for name in ('search_group_thresholds','search_no_delay_threshold'):monkeypatch.setattr(mf.fm,name,forbidden)
    monkeypatch.setattr(mf,'_save_calibrator',forbidden)
    score_calls=[];original=ev.score_model
    def counted(frame,entry,source):
        score_calls.append(entry['model_hash']);return original(frame,entry,source)
    monkeypatch.setattr(final,'score_model',counted)
    verify_calls=[];verify=final.verify_predictions
    def counted_verify(registry,run_id,X):
        assert len(X)==6000
        verify_calls.append(run_id);return verify(registry,run_id,X)
    monkeypatch.setattr(final,'verify_predictions',counted_verify)
    summary=final.run_final_eval('g2b-20261123')
    loader.assert_called_once_with('g2b-20261123')
    manifest=read_json(root/'results'/summary/'evaluation_manifest.json')
    assert manifest['status']=='complete' and len(manifest['source_run_ids'])==54
    assert len(score_calls)==len(set(score_calls))==13
    assert len(verify_calls)==1
    tasks=manifest['tasks'];reg=Registry(root)
    platform_run=reg.load_run(tasks['platform-c']['run_id'])
    assert platform_run['run']['verified_customers']==6000 and platform_run['run']['predictions_verified'] is True
    out=platform_run['predictions']
    assert len(out)==6000 and not out.customer_id.duplicated().any()
    assert out.risk_rank.tolist()==list(range(1,6001))
    assert set(out.data_split)=={'test'}
    ordered=out.sort_values(['risk_score','customer_id'],ascending=[False,True])
    pd.testing.assert_frame_equal(out,ordered)
    assert (reg.results/platform_run['run']['run_id']/'confusion_matrix.png').is_file()
    for key,row in tasks.items():
        bundle=reg.load_run(row['run_id'])
        folder=reg.results/row['run_id']
        assert not list(folder.glob('predictions*'))
        if key!='platform-c':assert not (reg.runs/row['run_id']/'predictions.csv').exists()
        if key.startswith('fairness-'):continue
        if key.startswith('rq3-'):
            assert 'policy_id' not in bundle['run']
            assert (folder/'rq3_lead_time.csv').is_file() and (folder/'rq3_coverage.csv').is_file()
            assert not (folder/'overall.csv').exists()
            assert len(pd.read_csv(folder/'rq3_coverage.csv'))==10
            continue
        overall=pd.read_csv(folder/'overall.csv').iloc[0]
        table=pd.read_csv(folder/'table_4_1_row.csv').iloc[0]
        classes=pd.read_csv(folder/'classification_report.csv')
        assert table.Accuracy==(table.TN+table.TP)/6000
        for metric in ('AUC','Precision','Recall','F1'):assert np.isclose(table[metric],overall[metric],equal_nan=True)
        pos=classes[classes.label.eq('1')].iloc[0]
        for metric in ('Precision','Recall','F1'):assert np.isclose(pos[metric],overall[metric],equal_nan=True)
        assert classes.support.sum()==12000  # Two class supports plus macro total support.
        assert (folder/'raw_policy_overall.csv').exists()
        if key.startswith('f7-iso'):
            for name in ('calibration_report.csv','calibration_bins.csv','calibration_curve.png','rank_change.csv'):
                assert (folder/name).is_file()
    from app.data import load_fairness_comparison
    fairness=load_fairness_comparison(tasks['fairness-M1-NS']['run_id'],root=root)
    assert fairness['source']['level']=='L2'
    assert fairness['source']['m1_run_id']==tasks['fairness-M1']['run_id']
    assert len(fairness['paired'])==30
    assert set(fairness['models'])=={'M1','M1-NS'}
    for filename in ('table_4_1_l2.csv','table_4_3_l2.csv'):
        rows=ev.read_result_csv(root/'results'/summary/filename)
        candidate=rows[rows.entry_id.str.startswith('f6-05-')]
        assert not candidate.empty and candidate.model_key.eq('05').all()
    assert list(fairness['models']['M1']['groups'])==[*p.GROUP_COLUMNS,'模型','政策比較','run_id','model_id']
    assert list(fairness['models']['M1-NS']['gaps_ci'])==[
        '指標','估計值','下限','上限','有效重抽','無效重抽','可靠','比較','模型','政策比較','分組方式',
        'm1_run_id','m1_model_id','run_id','model_id']
    for public in (root/'results'/summary).glob('*.csv'):
        assert not {'customer_id','risk_score','actual_default'} & set(pd.read_csv(public))


def test_invalid_release_never_touches_either_loader(release,monkeypatch):
    root,_,_,path,loader=release
    outer=Mock(side_effect=AssertionError('outer data accessed'))
    monkeypatch.setattr(p,'outer_splits',outer)
    original=path.read_bytes();freeze=read_json(path)
    cases=[]
    for key,value in [('status','prepared'),('approved_at',None),('approved_by','unknown'),('rules_freeze_sha256','0'*64),('data_sha256','0'*64)]:
        changed=deepcopy(freeze);changed[key]=value;cases.append(changed)
    changed=deepcopy(freeze);changed['entries'].pop();cases.append(changed)
    changed=deepcopy(freeze);changed['entries'][-1]['model_hash']='0'*64;cases.append(changed)
    for field in ('model_hash','params_hash','policy_hash'):
        changed=deepcopy(freeze);changed['entries'][0][field]='0'*64;cases.append(changed)
    changed=deepcopy(freeze)
    entry=next(e for e in changed['entries'] if e['entry_id']=='f7-iso-c')
    entry['calibrator_hash']='0'*64;cases.append(changed)
    changed=deepcopy(freeze);changed['entries'][0]['source_run_id']='missing';cases.append(changed)
    for data in cases:
        write_json(path,data,exclusive=False)
        for function in (final.run_final_eval,stability.run_stability_eval):
            with pytest.raises((IntegrityError,ValueError,KeyError,FileNotFoundError)):function('g2b-20261123')
    path.unlink()
    for function in (final.run_final_eval,stability.run_stability_eval):
        with pytest.raises(FileNotFoundError):function('g2b-20261123')
    path.write_bytes(original)
    loader.assert_not_called();outer.assert_not_called()
    assert not list((root/'results').glob('summary-*'))


def test_corrupt_actual_artifacts_stop_before_data(release):
    root,_,_,path,loader=release
    freeze=read_json(path);reg=Registry(root)
    entry=freeze['entries'][0];source=reg.load_run(entry['source_run_id'])
    cal=next(e for e in freeze['entries'] if e['entry_id']=='f7-iso-c')
    targets=[reg.path(source['model_record']['model_path']),
             reg.path(source['model_record']['model_path']).parent/'oof.csv',
             reg.path(cal['calibrator']['calibrator_path']),
             root/'data/default_of_credit_card_clients.csv']
    for target in targets:
        original=target.read_bytes()
        try:
            target.write_bytes(original+b'bad')
            with pytest.raises(IntegrityError):final.run_final_eval('g2b-20261123')
        finally:target.write_bytes(original)
    loader.assert_not_called()


def test_failed_l2_batch_and_resume_do_not_overwrite_completed_runs(release,monkeypatch):
    root,_,_,_,_=release
    original=final.evaluate_entry;calls=[]
    def fail_second(run,*args,**kwargs):
        calls.append(run)
        if len(calls)==2:raise RuntimeError('synthetic report failure')
        return original(run,*args,**kwargs)
    monkeypatch.setattr(final,'evaluate_entry',fail_second)
    with pytest.raises(RuntimeError,match='synthetic'):final.run_final_eval('g2b-20261123')
    manifest_path=next((root/'results').glob('summary-*/evaluation_manifest.json'))
    before=manifest_path.read_bytes();manifest=read_json(manifest_path)
    assert manifest['status']=='failed'
    assert calls[0].meta['status']=='complete' and calls[1].meta['status']=='failed'
    assert any(row['status']=='pending' for row in manifest['tasks'].values())
    completed=(calls[0].path/'run.json').read_bytes()
    saved=calls[0].result_path/'overall.csv';saved_bytes=saved.read_bytes()
    saved.write_bytes(saved_bytes+b'\n')
    with pytest.raises(IntegrityError,match='result missing or changed'):
        final.run_final_eval('g2b-20261123',resume=manifest['summary_id'])
    saved.write_bytes(saved_bytes)
    monkeypatch.setattr(final,'evaluate_entry',original)
    summary=final.run_final_eval('g2b-20261123',resume=manifest['summary_id'])
    after=read_json(root/'results'/summary/'evaluation_manifest.json')
    assert after['status']=='complete'
    assert after['tasks']['rq1-m1-s']['run_id']==calls[0].run_id
    assert after['tasks']['rq1-m1-n']['run_id']!=calls[1].run_id
    assert manifest_path.read_bytes()==before and (calls[0].path/'run.json').read_bytes()==completed


def test_reload_verification_failure_cannot_finish_platform(release,monkeypatch):
    root,_,_,_,_=release
    # Run only the platform evaluation action after validating the full synthetic release.
    freeze,sources,rules,_=ev.validate_release('g2b-20261123')
    entry=next(e for e in freeze['entries'] if e['entry_id']=='platform-c')
    frame=release[2];score,X=ev.score_model(frame,entry,sources['platform-c'])
    run=p.start_run('L2','synthetic failure',model_freeze_id='g2b-20261123',entry='platform-c')
    monkeypatch.setattr(final,'verify_predictions',Mock(side_effect=IntegrityError('bad reload')))
    with pytest.raises(IntegrityError,match='bad reload'):
        try:final.export_platform(run,entry,frame,score,X,sources['platform-c']['model'])
        except Exception as exc:run.fail(str(exc));raise
    assert read_json(run.path/'run.json')['status']=='failed'
    assert not run.meta.get('predictions_verified')


def test_q7_zero_denominators_and_strict_macro():
    table=ev.classification_table(dict(TN=10,FP=0,FN=4,TP=0)).set_index('label')
    assert pd.isna(table.at[1,'Precision']) and table.at[1,'Precision_undefined']
    assert pd.isna(table.at['macro','Precision'])
    assert table.at['macro','support']==14
    assert table.at[1,'F1']==0
    empty=ev.classification_table(dict(TN=0,FP=0,FN=0,TP=0))
    assert empty[['Precision','Recall','F1']].isna().all().all()


def test_tied_scores_use_customer_id_order_and_capacity():
    score=pd.Series([.8,.8,.8,.3,.1],index=[5,2,4,1,3])
    pred,info=p.apply_policy(score,p.Policy.single(.1,.4))
    assert set(pred[pred==1].index)=={2,4}
    assert int(info['pred_raw'].sum())==5 and info['名單上限是否觸發']


def test_imports_and_cli_are_inert(monkeypatch):
    for name in ('final_test','outer_splits','_full_data'):
        monkeypatch.setattr(p,name,Mock(side_effect=AssertionError('import read data')))
    importlib.reload(final);importlib.reload(stability)
    for module in (final,stability):
        with pytest.raises(SystemExit):module.main([])


def test_new_loaders_are_included_in_source_gate(tmp_path):
    (tmp_path/'src').mkdir();(tmp_path/'experiments').mkdir()
    file=tmp_path/'src/unsafe.py'
    file.write_text('from src.protocol import final_test as fetch\nfetch("id")\n')
    assert 'guarded evaluation' in p.audit_sources(tmp_path)[0]['reason']
    file.write_text('from src.protocol import outer_splits as fetch\nfetch("id")\n')
    assert 'guarded evaluation' in p.audit_sources(tmp_path)[0]['reason']
    assert p.audit_sources()==[]


def _mock_reports(run,entry,frame,score,rules,timing,*,raw_score=None,gap_cache=None):
    """Mock costly report rendering only, keep actual L3 fit/OOF/policies/registry."""
    assert len(frame)==6000 and score.index.equals(frame.index)
    if entry['evaluation_kind']=='score_only':
        ev.save_table(run,entry,'rq3_lead_time',pd.DataFrame([dict(AUC=.7,fit_seconds=timing['fit_seconds'])]))
        ev.save_table(run,entry,'rq3_coverage',pd.DataFrame([{'名單比例':r,'違約者涵蓋率':.5,'帳單金額涵蓋率':.4} for r in ev.REPORT_RATES]))
    else:
        ev.save_table(run,entry,'table_4_1_row',pd.DataFrame([dict(AUC=.7,Accuracy=.8,Precision=.6,
            Recall=.5,F1=.55,fit_seconds=timing['fit_seconds'])]))
        ev.save_table(run,entry,'classification_report',ev.classification_table(dict(TN=3000,FP=1000,FN=1000,TP=1000)))
    ev.save_table(run,entry,'training_timing',pd.DataFrame([timing]))


def test_l3_ten_seeds_retrain_and_reselect_with_shared_fit_evidence(release,monkeypatch):
    root,train,test,path,loader=release
    # Hash rejection is exercised with real validation in the L2/gate tests.
    # Here cache already-verified L1 sources only, avoiding thousands of repeated
    # L1 deserializations while keeping all L3 artifact/source checks real.
    registry=Registry(root)
    approved=read_json(path)
    validated={entry['entry_id']:registry.frozen_entry('g2b-20261123',entry['entry_id'])
               for entry in approved['entries']}
    monkeypatch.setattr(Registry,'frozen_entry',lambda self,freeze_id,entry:validated[entry])
    original_load=Registry.load_model;model_cache={}
    def verified_cache(self,record):
        key=record['model_path'];folder=self.path(key).parent
        if key not in model_cache:
            model_cache[key]=(original_load(self,record),file_hash(folder/'model.json'))
        result,digest=model_cache[key]
        assert result[1]==record and file_hash(folder/'model.json')==digest
        for name,field in [('model.joblib','model_hash'),('oof.csv','oof_sha256'),('folds.csv','folds_sha256')]:
            assert file_hash(folder/name)==record[field]
        return result
    monkeypatch.setattr(Registry,'load_model',verified_cache)
    calls=[];original_train=stability._train
    def counted_train(run,entry,frame,seed,rules):
        calls.append((seed,entry['model_key']))
        return original_train(run,entry,frame,seed,rules)
    monkeypatch.setattr(stability,'_train',counted_train)
    outer_calls=[]
    def outer(freeze_id,*,seeds):
        outer_calls.append((freeze_id,list(seeds)))
        for seed in seeds:
            a=train.copy();b=test.copy()
            a.attrs.update(level='L3',seed=seed);b.attrs.update(level='L3',seed=seed)
            yield seed,a,b
    monkeypatch.setattr(p,'outer_splits',outer)
    monkeypatch.setattr(stability,'evaluate_entry',_mock_reports)
    summary=stability.run_stability_eval('g2b-20261123')
    loader.assert_not_called()
    assert outer_calls==[('g2b-20261123',list(range(42,52)))]
    assert len(calls)==130 and len(set(calls))==130
    manifest=read_json(root/'results'/summary/'evaluation_manifest.json')
    assert manifest['status']=='complete' and len(manifest['source_run_ids'])==520
    chosen={e['entry_id']:e for e in read_json(path)['entries']}
    reg=Registry(root);hash_matches=0;source_ids={}
    for key,row in manifest['tasks'].items():
        bundle=reg.load_run(row['run_id']);meta=bundle['run'];record=bundle['model_record']
        seed,entry_id=key.split('/')
        assert meta['seed']==int(seed) and record['level']=='L3'
        assert meta['model_id']!=chosen[entry_id]['model_id']
        assert meta['retraining']['independent_fit'] is True
        assert meta['cv_seed']==42
        mf._check_timing(meta['training_timing'])
        assert meta['retraining']['training_ids_hash']==record['training_ids_hash']
        source_ids.setdefault((seed,chosen[entry_id]['model_key']),set()).add(meta['retraining']['fit_run_id'])
        hash_matches+=record['model_hash']==chosen[entry_id]['model_hash']
        assert not (reg.runs/meta['run_id']/'predictions.csv').exists()
    assert hash_matches>0
    assert all(len(value)==1 for value in source_ids.values())
    table=pd.read_csv(root/'results'/summary/'stability_summary.csv')
    assert table.n_valid.eq(10).all()
    raw=pd.read_csv(root/'results'/summary/'stability_by_seed.csv')
    row=table[table.metric.eq('fit_seconds')].iloc[0]
    frame=raw[raw.entry_id.eq(row.entry_id)]
    assert np.isclose(row['std'],frame.fit_seconds.std(ddof=1))
    assert not {'policy_id','model_id'} & set(table.columns)
    for filename in ('stability_by_seed.csv','stability_summary.csv'):
        rows=ev.read_result_csv(root/'results'/summary/filename)
        candidate=rows[rows.entry_id.str.startswith('f6-05-')]
        assert not candidate.empty and candidate.model_key.eq('05').all()
    for file in (root/'results'/summary).glob('*.csv'):
        assert not {'customer_id','risk_score','actual_default'} & set(pd.read_csv(file))


def test_missing_l3_seed_cannot_publish_complete_summary(release,monkeypatch):
    root,train,test,_,loader=release
    # Mock entry work so only the missing-seed guard is exercised.
    monkeypatch.setattr(p,'outer_splits',lambda *a,**k:iter([]))
    with pytest.raises(IntegrityError,match='Missing L3 seed'):
        stability.run_stability_eval('g2b-20261123')
    loader.assert_not_called()
    manifest=read_json(next((root/'results').glob('summary-*/evaluation_manifest.json')))
    assert manifest['status']=='failed' and all(r['status']=='pending' for r in manifest['tasks'].values())
    assert not list((root/'results'/manifest['summary_id']).glob('table_4_1_l3.csv'))


def test_q11_rejects_borrowed_l2_record_even_with_declared_fit(release):
    root,train,_,path,_=release
    freeze=read_json(path);chosen=freeze['entries'][0]
    source=Registry(root).load_run(chosen['source_run_id'])
    run=p.start_run('L3','synthetic borrowed model',model_freeze_id='g2b-20261123',entry=chosen['entry_id'])
    run.update(seed=42,retraining=dict(independent_fit=True,fit_seconds=1.))
    with pytest.raises(IntegrityError):run.bind_model(source['model_record'])
    run.fail('synthetic rejected borrowing')


def test_test_ids_must_be_unique_disjoint_and_complete(release):
    _,train,frame,path,_=release
    freeze,sources,rules,_=ev.validate_release('g2b-20261123')
    with pytest.raises(IntegrityError):ev.check_test(frame.iloc[:-1],sources,rules)
    bad=frame.copy();bad.index=pd.Index([1,*frame.index[1:]])
    with pytest.raises(IntegrityError):ev.check_test(bad,sources,rules)
    bad=frame.copy();bad.index=pd.Index([frame.index[1],*frame.index[1:]])
    with pytest.raises((IntegrityError,ValueError)):ev.check_test(bad,sources,rules)


@pytest.mark.parametrize('prefix',['rq1-m1','f7-iso'])
def test_l3_resume_reuses_completed_fit_and_never_overwrites_it(release,monkeypatch,prefix):
    root,train,test,_,_=release
    if prefix=='f7-iso':
        freeze,sources,rules,digest=ev.validate_release('g2b-20261123')
        subset=deepcopy(freeze)
        subset['entries']=[entry for entry in subset['entries'] if entry['entry_id'].startswith(prefix+'-')]
        # Isolate retry of calibration after exercising the real full gate.
        monkeypatch.setattr(stability,'validate_release',lambda *a:(subset,sources,rules,digest))
    def one_split(*args,**kwargs):
        a=train.copy();b=test.copy()
        a.attrs.update(level='L3',seed=42);b.attrs.update(level='L3',seed=42)
        yield 42,a,b
    monkeypatch.setattr(p,'outer_splits',one_split)
    def first_attempt(run,entry,*args,**kwargs):
        if entry['entry_id']==prefix+'-n':raise RuntimeError('first policy failure')
        return _mock_reports(run,entry,*args,**kwargs)
    monkeypatch.setattr(stability,'evaluate_entry',first_attempt)
    with pytest.raises(RuntimeError,match='first policy'):
        stability.run_stability_eval('g2b-20261123')
    prior_path=next((root/'results').glob('summary-*/evaluation_manifest.json'))
    prior=read_json(prior_path);prior_bytes=prior_path.read_bytes()
    first=prior['tasks']['42/'+prefix+'-s']['run_id']
    source_path=root/'outputs/runs'/first/'run.json';source_bytes=source_path.read_bytes()
    def second_attempt(run,entry,*args,**kwargs):
        if entry['entry_id']==prefix+'-c':raise RuntimeError('second intentional stop')
        return _mock_reports(run,entry,*args,**kwargs)
    monkeypatch.setattr(stability,'evaluate_entry',second_attempt)
    monkeypatch.setattr(stability,'_train',Mock(side_effect=AssertionError('completed model was refitted')))
    if prefix=='f7-iso':
        monkeypatch.setattr(mf,'_save_calibrator',Mock(side_effect=AssertionError('completed calibrator was refitted')))
    with pytest.raises(RuntimeError,match='second intentional'):
        stability.run_stability_eval('g2b-20261123',resume=prior['summary_id'])
    manifests=[read_json(path) for path in (root/'results').glob('summary-*/evaluation_manifest.json')]
    latest=next(m for m in manifests if m['summary_id']!=prior['summary_id'])
    new_policy=latest['tasks']['42/'+prefix+'-n']
    assert new_policy['status']=='complete'
    meta=read_json(root/'outputs/runs'/new_policy['run_id']/'run.json')
    assert meta['retraining']['fit_run_id']==first
    if prefix=='f7-iso':
        assert meta['calibrator']==read_json(source_path)['calibrator']
        assert meta['calibration_source_run_id']==first
    assert source_path.read_bytes()==source_bytes and prior_path.read_bytes()==prior_bytes
