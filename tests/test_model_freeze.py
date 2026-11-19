"""C9b-1 synthetic end-to-end preparation. Real data and test entry points fail."""
from copy import deepcopy
from datetime import datetime
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import model_freeze as mf, protocol as p
from src.registry import Registry, IntegrityError, content_hash, file_hash, read_json, write_json


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    root=Path(__file__).parents[1]
    (tmp_path/'docs').mkdir();(tmp_path/'docs/frozen_rules.md').write_bytes((root/'docs/frozen_rules.md').read_bytes())
    rules=read_json(root/'outputs/frozen/rules-g2a-20261116.json')
    rules['rules_freeze_id']='synthetic-rules'
    rules['dev_base']['params'].update(n_estimators=2,max_depth=2,n_jobs=1)
    n=160
    frame=pd.DataFrame(index=pd.Index(range(1,n+1),name='ID'))
    frame['LIMIT_BAL']=1000.+np.arange(n)
    for month in range(1,7):
        frame[f'PAY_{month}']=np.where(np.arange(n)%4<2,0,1)
        frame[f'PAY_AMT{month}']=10.+np.arange(n)%7+month
        frame[f'BILL_AMT{month}']=100.+np.arange(n)%13+month
    frame['SEX']=np.tile([1,2],n//2)
    frame['AGE']=np.tile([25,35,45,55],n//4)
    frame['EDUCATION']=2;frame['MARRIAGE']=1
    frame['default']=(np.arange(n)%3==0).astype(int)
    frame.attrs.update(level='L1',seed=42)
    data=tmp_path/'data';data.mkdir();frame.to_csv(data/'default_of_credit_card_clients.csv')
    rules['data_protocol']['selection']['train_rows']=n
    rules['pay_categorical']['l1_verification'].update(training_ids_hash=content_hash(frame.index.tolist()),
        data_sha256=file_hash(data/'default_of_credit_card_clients.csv'))
    (tmp_path/'outputs/frozen').mkdir(parents=True)
    write_json(tmp_path/'outputs/frozen/rules-synthetic-rules.json',rules)
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'DATA_DIR',data)
    monkeypatch.setattr(p,'dev_data',lambda:frame.copy())
    def forbidden(*args,**kwargs):raise AssertionError('Test/holdout/raw/seal access forbidden')
    for name in ('_full_data','final_test','outer_splits','dev_holdout','nested_eval','open_sealed'):
        monkeypatch.setattr(p,name,forbidden)
    return tmp_path,frame,rules


@pytest.fixture
def prepared(sandbox):
    path=mf.prepare('synthetic-rules')
    return sandbox,path


def test_prepare_builds_real_sources_no_model_freeze(prepared):
    (root,frame,rules),path=prepared
    assert not list((root/'outputs/frozen').glob('model-*.json'))
    draft=read_json(path);registry=Registry(root)
    assert draft['platform_model'] is None and draft['status']=='prepared'
    assert len(draft['entries'])==len(mf.entry_specs(rules['g2b']['eligible_candidates']))
    records={};policies={}
    for entry in draft['entries']:
        bundle=registry.load_run(entry['source_run_id'])
        mf._validate_source(registry,draft,entry,bundle)
        record=bundle['model_record'];records[record['model_id']]=record
        assert record['training_count']==len(frame)
        assert bundle['run']['status']=='complete'
        timing=entry['training_timing'];mf._check_timing(timing)
        assert timing['wall_seconds']>=timing['artifact_seconds']>0
        assert datetime.fromisoformat(timing['started_at']).utcoffset() is not None
        assert datetime.fromisoformat(timing['finished_at'])>=datetime.fromisoformat(timing['started_at'])
        assert timing['environment']['logical_cpus']
        assert 'threadpools' in timing['environment'] and 'model_threads' in timing
        if entry['evaluation_kind']=='score_only':
            assert 'policy_id' not in entry and bundle['threshold'] is None
            assert not (registry.runs/entry['source_run_id']/'threshold.json').exists()
        else:policies[entry['entry_id']]=entry['policy_id']
    assert len(records)==13
    assert len({r['folds_sha256'] for r in records.values()})==1
    lr0=next(e for e in draft['entries'] if e['entry_id']=='rq2-lr0-c')
    lr1=next(e for e in draft['entries'] if e['entry_id']=='rq2-lr1-c')
    assert len(lr0['columns'])==19 and len(lr1['columns'])==33
    d=next(e for e in draft['entries'] if e['entry_id']=='f6-d-c')
    assert set(d['training_timing']['submodel_fit_seconds'])=={'有延遲','無延遲'}
    assert d['training_timing']['submodel_total_fit_seconds']>0
    e=next(e for e in draft['entries'] if e['entry_id']=='f6-e')
    ns=next(e for e in draft['entries'] if e['entry_id']=='rq2-ns-c')
    assert e['model_hash']==ns['model_hash'] and e['training_timing']['fit_seconds']==ns['training_timing']['fit_seconds']
    cal=next(e for e in draft['entries'] if e['entry_id']=='f7-iso-c')
    bundle=registry.load_run(cal['source_run_id'])
    _,cross=mf.load_calibrator(registry,cal['calibrator'],bundle['model_record'])
    expected=p.choose_threshold(frame['default'],pd.Series(cross.oof_score.to_numpy(),index=frame.index),p.bill_at(frame,0),p.load_rules('synthetic-rules').scenario)['門檻']
    assert bundle['threshold']['protocol_policy']['threshold']==expected
    assert bundle['threshold']['protocol_policy']['evidence']['oof_scores_hash']==content_hash(cross.oof_score.tolist())
    assert cal['calibrator_hash']==cal['calibrator']['calibrator_hash']
    assert cal['calibrator']['calibrator_hash']==file_hash(registry.path(cal['calibrator']['calibrator_path']))
    report=pd.read_csv(root/'results'/draft['build_run_id']/'g2b_candidates.csv')
    assert set(report.candidate)==set(rules['g2b']['eligible_candidates'])
    assert report[['reload_ok','features_ok','shap_ok']].all().all(),report.engineering_error


def test_search_evidence_is_saved_and_required(prepared):
    (root,frame,_),path=prepared
    draft=read_json(path);registry=Registry(root)
    for entry in draft['entries']:
        if entry['entry_id'] not in {'f6-e','f3b-sex','f3b-age','f3b-sex-age'}:
            continue
        source=registry.load_run(entry['source_run_id'])
        evidence=entry['search_evidence']
        assert evidence==source['run']['search_evidence']
        assert content_hash(evidence)==entry['search_evidence_hash']
        private=registry.path(evidence['candidates_path'])
        public=root/'results'/entry['source_run_id']/'search_candidates.csv'
        assert private.read_bytes()==public.read_bytes()
        table=pd.read_csv(private,converters={key:float for key in ('threshold','target_recall','raw_cost','final_cost')})
        assert table.selected.any()
        policy=source['threshold']['protocol_policy']
        from src._protocol_runs import _partition
        partition=(_partition(frame,policy['partition_name']) if entry['entry_id']!='f6-e'
                   else mf.fm.delay_partition(frame,rules_freeze_id='synthetic-rules'))
        oof=pd.read_csv(registry.path(source['model_record']['model_path']).parent/'oof.csv',converters={'oof_score':float})
        score=pd.Series(oof['oof_score'].to_numpy(),index=frame.index)
        final,info=p.apply_policy(score,p.Policy(**policy),partition=partition)
        assert evidence['cap_triggered']==info['名單上限是否觸發']
        assert evidence['raw_results']['overall'] and evidence['final_results']['overall']
        if entry['entry_id']=='f6-e':
            assert evidence['target_recall'] is None
            selected=table[table.selected].iloc[0]
            assert selected.raw_count==int(info['pred_raw'].sum())
            assert selected.final_count==int(final.sum())
            assert selected.raw_cost==evidence['raw_results']['overall'][0]['總成本']
            assert selected.final_cost==evidence['final_results']['overall'][0]['總成本']
        else:
            assert evidence['target_recall']==float((score[frame.default==1]>=evidence['t0']).mean())
        for target in (private,public):
            original=target.read_bytes()
            try:
                for corrupt in (False,True):
                    if corrupt:target.write_bytes(original+b'corrupt')
                    else:target.unlink()
                    with pytest.raises(IntegrityError,match='candidates'):
                        mf.validate_entry(registry,draft,entry,source)
                    target.write_bytes(original)
            finally:target.write_bytes(original)
        altered=deepcopy(entry);altered.pop('search_evidence')
        with pytest.raises(IntegrityError,match='search evidence'):
            mf.validate_entry(registry,draft,altered,source)


def test_timing_requires_complete_valid_measurements():
    timing=dict(fit_seconds=1.,oof_seconds=2.,policy_seconds=0.,calibrator_fit_seconds=0.,
                artifact_seconds=.1,wall_seconds=3.1,started_at='2026-11-19T10:00:00+08:00',
                finished_at='2026-11-19T10:00:03.100000+08:00',
                environment={'packages':{'pytest':'fixed'},'logical_cpus':1})
    mf._check_timing(timing)
    for key in ('artifact_seconds','wall_seconds','started_at','finished_at'):
        bad=deepcopy(timing);bad.pop(key)
        with pytest.raises(IntegrityError):mf._check_timing(bad)
    for key in ('artifact_seconds','wall_seconds'):
        for value in (-1,float('nan'),float('inf'),True):
            with pytest.raises(IntegrityError):mf._check_timing(dict(timing,**{key:value}))
    for value in ('invalid','2026-11-19T10:00:03','2026-11-19T09:59:59+08:00'):
        with pytest.raises(IntegrityError):mf._check_timing(dict(timing,finished_at=value))
    with pytest.raises(IntegrityError):mf._check_timing(dict(timing,artifact_seconds=4.))


def test_freeze_explicit_approval_readback_and_immutable(prepared,monkeypatch):
    (root,frame,rules),path=prepared
    monkeypatch.setattr(p,'dev_data',lambda:pytest.fail('freeze read data'))
    monkeypatch.setattr(mf,'model_factory',lambda *a:pytest.fail('freeze trained model'))
    monkeypatch.setattr(p,'make_model',lambda *a,**k:pytest.fail('freeze created model'))
    monkeypatch.setattr(p,'oof_predict',lambda *a,**k:pytest.fail('freeze regenerated OOF'))
    monkeypatch.setattr(mf,'_save_calibrator',lambda *a,**k:pytest.fail('freeze calibrated model'))
    dest=mf.freeze(path,model_freeze_id='g2b-20261123',approved_at='2026-11-23T10:00:00+08:00',platform_model='M1-NS')
    freeze=read_json(dest);registry=Registry(root)
    assert freeze['approved_by']=='蘇韻蓉' and freeze['platform_model']=='M1-NS'
    assert sum(e['role']=='platform' for e in freeze['entries'])==1
    for entry in freeze['entries']:registry.frozen_entry('g2b-20261123',entry['entry_id'])
    # The existing public run interface also accepts Q3 with no dummy threshold.
    run=p.start_run('L2','synthetic no-threshold readback',model_freeze_id='g2b-20261123',entry='rq3-lag1')
    run.finish()
    assert registry.load_run(run.run_id)['threshold'] is None
    assert 'policy_id' not in run.meta
    with pytest.raises(FileExistsError):
        mf.freeze(path,model_freeze_id='g2b-20261123',approved_at='2026-11-23T10:00:00+08:00',platform_model='M1-NS')


@pytest.mark.parametrize('case',['missing','duplicate','model_hash','calibrator_hash','policy','timing','report'])
def test_freeze_rejects_incomplete_or_tampered_draft(prepared,case):
    (root,_,_),path=prepared
    draft=read_json(path)
    if case=='missing':draft.pop('entries')
    elif case=='duplicate':draft['entries'][1]=deepcopy(draft['entries'][0])
    elif case=='model_hash':draft['entries'][0]['model_hash']='a'*64
    elif case=='calibrator_hash':
        next(e for e in draft['entries'] if e['score_version']=='calibrated')['calibrator']['calibrator_hash']='a'*64
    elif case=='policy':draft['entries'][0]['protocol_policy']['threshold']=.123
    elif case=='timing':draft['entries'][0]['training_timing'].pop('fit_seconds')
    elif case=='report':
        Registry(root).path(draft['candidate_report']).write_text('tampered')
    write_json(path,draft,exclusive=False)
    # Deliberately re-seal synthetic metadata to exercise inner validators too.
    meta_path=root/'outputs/runs'/read_json(path).get('build_run_id','missing')/'run.json'
    if meta_path.exists():
        meta=read_json(meta_path);meta['draft_sha256']=file_hash(path)
        write_json(meta_path,meta,exclusive=False)
    with pytest.raises((IntegrityError,KeyError)):
        mf.freeze(path,model_freeze_id='g2b-20261123',approved_at='2026-11-23T10:00:00+08:00',platform_model='M1-NS')
    assert not list((root/'outputs/frozen').glob('model-*.json'))


@pytest.mark.parametrize('field',['model','oof','folds','calibrator','crossfit'])
def test_freeze_validates_actual_artifact_hashes(prepared,field):
    (root,_,_),path=prepared;draft=read_json(path);registry=Registry(root)
    entry=next(e for e in draft['entries'] if e['entry_id']=='f7-iso-c')
    record=registry.load_run(entry['source_run_id'])['model_record']
    base=registry.path(record['model_path'])
    if field=='model':target=base
    elif field in ('oof','folds'):target=base.parent/(field+'.csv')
    else:
        target=registry.path(entry['calibrator']['calibrator_path'])
        if field=='crossfit':target=target.parent/'crossfit_oof.csv'
    with target.open('ab') as out:out.write(b'corrupt')
    with pytest.raises(IntegrityError):
        mf.freeze(path,model_freeze_id='g2b-20261123',approved_at='2026-11-23T10:00:00+08:00',platform_model='M1-NS')


def test_d7_boundaries_and_all_five_ranking_levels(sandbox):
    rules=p.load_rules('synthetic-rules')
    def row(name,recall=.02,gender=.01,age=.01,count=26,auc=0.):
        return dict(candidate=name,recall_gain=recall,auc_gain=auc,gender_eo_increase=gender,
                    age_eo_increase=age,input_feature_count_before_onehot=count,
                    reload_ok=True,features_ok=True,shap_ok=True)
    names=['05','09 B','09 C','09 D']
    tables=[
        [row(n,recall=.02000000001 if n=='09 D' else .02) for n in names],
        [row(n,gender=.005 if n=='09 D' else .01,age=.005 if n=='09 D' else .01) for n in names],
        [row(n,gender=0 if n=='09 D' else .01,age=.01) for n in names],
        [row(n,count=19 if n=='09 D' else 26) for n in names],
        [row(n) for n in names],
    ]
    for i,rows in enumerate(tables):
        result=mf.rank_candidates(pd.DataFrame(rows),rules)
        assert result.loc[result['rank']==1,'candidate'].iloc[0]==('05' if i==4 else '09 D')
    rows=[row('05',recall=.01,gender=.02,age=.02),row('09 B',recall=.009999999999),
          row('09 C',auc=-1e-12),row('09 D',age=.020000000001)]
    result=mf.rank_candidates(pd.DataFrame(rows),rules)
    assert result.d7_eligible.tolist()==[True,False,False,False]


def test_entrypoints_import_does_not_obtain_data(sandbox):
    root,_,_=sandbox
    for filename in ['00_build_model_freeze.py','03_repeated_cv.py','08_calibration.py']:
        path=Path(__file__).parents[1]/'experiments'/filename
        spec=importlib.util.spec_from_file_location('safe_import_'+filename[:-3],path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert not list((root/'outputs').glob('runs/*'))


def test_builder_audit_detects_test_access(tmp_path):
    for folder in ('src','experiments'):(tmp_path/folder).mkdir()
    for rel in ['src/model_freeze.py','experiments/00_build_model_freeze.py']:
        (tmp_path/rel).write_text('from src.protocol import final_test as forbidden_alias\nforbidden_alias("fake")\n')
    findings=p.audit_sources(tmp_path)
    assert {r['file'] for r in findings}=={'src/model_freeze.py','experiments/00_build_model_freeze.py'}


def test_prepare_failure_marks_batch_and_current_model_failed(sandbox,monkeypatch):
    root,_,_=sandbox
    def broken(*args):raise RuntimeError('synthetic fit failure')
    monkeypatch.setattr(mf,'model_factory',broken)
    with pytest.raises(RuntimeError,match='synthetic fit failure'):
        mf.prepare('synthetic-rules')
    metas=[read_json(path) for path in (root/'outputs/runs').glob('*/run.json')]
    assert len(metas)==2 and all(m['status']=='failed' for m in metas)
    assert not list((root/'outputs/frozen').glob('model-*.json'))
    assert not list((root/'outputs').glob('drafts/*/manifest.json'))


def test_invalid_request_and_missing_rules_stop_before_data(sandbox,monkeypatch):
    root,_,_=sandbox
    monkeypatch.setattr(p,'dev_data',lambda:pytest.fail('invalid preparation accessed data'))
    with pytest.raises(FileNotFoundError):mf.prepare('missing')
    request=root/'request.json';write_json(request,{'rules_freeze_id':'synthetic-rules','params':{'max_depth':99}})
    with pytest.raises(ValueError,match='no design overrides'):mf.prepare('synthetic-rules',manifest=request)
    assert not (root/'outputs/runs').exists()


def test_cli_missing_approval_or_platform_does_not_prepare(sandbox,monkeypatch):
    monkeypatch.setattr(mf,'prepare',lambda *a,**k:pytest.fail('freeze dispatched prepare'))
    with pytest.raises(SystemExit):mf.main(['freeze','--manifest','missing'])


def test_manifest_paths_cannot_read_data_or_non_json(sandbox,monkeypatch):
    root,_,_=sandbox
    monkeypatch.setattr(mf,'read_json',lambda *a:pytest.fail('unsafe manifest was opened'))
    for path in [root/'data/default_of_credit_card_clients.csv', root/'.env', root/'data/test.json']:
        with pytest.raises(ValueError):
            mf.freeze(path,model_freeze_id='g2b-20261123',approved_at='2026-11-23T10:00:00+08:00',platform_model='M1-NS')


def test_unapproved_or_invalid_platform_never_publishes(prepared):
    (root,_,_),path=prepared
    for freeze_id,approved,model in [('invalid','2026-11-23T10:00:00+08:00','M1-NS'),
        ('g2b-20261123','2026-11-23T10:00:00','M1-NS'),
        ('g2b-20261123','2026-11-24T10:00:00+08:00','M1-NS'),
        ('g2b-20261123','','M1-NS'),
        ('g2b-20261123','2026-11-23T10:00:00+08:00','09 E')]:
        with pytest.raises(ValueError):mf.freeze(path,model_freeze_id=freeze_id,approved_at=approved,platform_model=model)
    assert not list((root/'outputs/frozen').glob('model-*.json'))


def test_engineering_shap_failure_is_reported_without_selection(prepared,monkeypatch):
    (root,frame,_),path=prepared;registry=Registry(root)
    entry=next(e for e in read_json(path)['entries'] if e['entry_id']=='f6-05-c')
    bundle=registry.load_run(entry['source_run_id'])
    X,_=mf.features(frame,entry['configuration'],'synthetic-rules')
    model_bundle=dict(run=type('Source',(),{'registry':registry})(),record=bundle['model_record'],
                      configuration=entry['configuration'],X=X,model=bundle['model'])
    def unsupported(*args):raise ValueError('synthetic unsupported SHAP')
    monkeypatch.setattr(mf,'top_factors',unsupported)
    result=mf.engineering_check(model_bundle,frame,'synthetic-rules')
    assert result['reload_ok'] and result['features_ok'] and not result['shap_ok']
    assert 'synthetic unsupported SHAP' in result['engineering_error']
    assert read_json(path)['platform_model'] is None


def test_group_timing_is_external_to_model_bytes(sandbox):
    import io,joblib
    from src.frozen_methods import prepare_missed_features,make_grouped_model
    _,frame,_=sandbox
    X,_=prepare_missed_features(frame,method='D',rules_freeze_id='synthetic-rules')
    model=make_grouped_model(rules_freeze_id='synthetic-rules').fit(X,frame['default'])
    first=io.BytesIO();joblib.dump(model,first)
    model.fit_seconds_by_group_={'有延遲':999.,'無延遲':1000.}
    second=io.BytesIO();joblib.dump(model,second)
    assert first.getvalue()==second.getvalue()
