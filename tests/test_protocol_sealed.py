"""L4 uses tiny synthetic data and mocked search, never real test customers."""
from dataclasses import asdict
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src import protocol as p
from src import _sealed as sealed
from src.registry import _SEALED_ACCESS, IntegrityError, content_hash, file_hash, read_json, write_json
from test_protocol import small, rule_fixture


@pytest.fixture
def environment(tmp_path,monkeypatch,small):
    rules=rule_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(p,'_full_data',lambda:small.copy())
    def search(X,y,space,*,n_iter,folds=5,seed=42):
        return {'最佳參數':{'n_estimators':2,'max_depth':2,'n_jobs':1},'最佳 CV AUC':.5,
            '完整搜尋紀錄':pd.DataFrame(),'實際搜尋組數':n_iter,'搜尋範圍雜湊':content_hash(space)}
    monkeypatch.setattr(p,'tune_model',search)
    return tmp_path,rules,small


def paths(root):
    return sorted((root/'outputs/sealed').glob('l4-*'))


def edit_json(path,fn):
    with sealed._access():
        value=read_json(path)
        fn(value)
        path.write_text(__import__('json').dumps(value))


def resign_test_fixture(folder,filename):
    # Synthetic adversarial fixtures simulate a file written in a different row
    # order before sealing. Production code never rewrites seal.json.
    with sealed._access():
        edit_json(folder/'seal.json',lambda s:s['files'].__setitem__(filename,file_hash(folder/filename)))


def test_seal_no_run_no_outer_metrics_then_open_once_immutable(environment,capsys):
    root,rules,data=environment
    seal_id=p.nested_eval('fixture')
    output=capsys.readouterr().out
    assert output.count('outer fold')==5
    for forbidden in ['AUC','Recall','Precision','F1','score']:
        assert forbidden not in output
    assert not list((root/'outputs/runs').glob('*'))
    folder=paths(root)[0]
    with sealed._access():
        before={str(f.relative_to(folder)):file_hash(f) for f in folder.rglob('*') if f.is_file()}
    run=p.open_sealed('fixture','fixture')
    assert run.meta['status']=='complete' and run.meta['level']=='L4'
    assert not {'model_id','model_hash','params_hash','policy_id','policy_hash'}&run.meta.keys()
    assert 'fold_policies_hash' in run.meta and len(run.meta['folds'])==5
    assert len(pd.read_csv(run.result_path/'nested_summary.csv'))==5
    assert all((run.path/'folds'/str(k)/'model-policy.json').exists() for k in range(5))
    with sealed._access():
        after={str(f.relative_to(folder)):file_hash(f) for f in folder.rglob('*') if f.is_file()}
    assert before==after
    with pytest.raises(FileExistsError):p.open_sealed('fixture','fixture')
    with pytest.raises(FileExistsError):p.nested_eval('fixture')


def test_unfrozen_open_checks_before_sealed_read(environment,monkeypatch):
    root,_,_=environment
    p.nested_eval('fixture')
    monkeypatch.setattr(sealed,'_sealed_folders',lambda *a:pytest.fail('read sealed files before freeze check'))
    with pytest.raises(FileNotFoundError):p.open_sealed('fixture','absent')


@pytest.mark.parametrize('key,value',[('outer_folds',4),('n_iter',31),('search_space',{'n_estimators':[1]})])
def test_l4_configuration_tamper_stops_before_data_load(environment,monkeypatch,key,value):
    root,_,_=environment
    edit_json(root/'outputs/frozen/rules-fixture.json',lambda r:r['l4'].__setitem__(key,value))
    monkeypatch.setattr(p,'_full_data',lambda:pytest.fail('loaded before config validation'))
    with pytest.raises(IntegrityError):p.nested_eval('fixture')
    assert not paths(root)


def test_markdown_hash_mismatch(environment):
    root,_,_=environment
    (root/'docs/frozen_rules.md').write_text('tampered')
    with pytest.raises(IntegrityError):p.nested_eval('fixture')


def test_sealed_file_hash_failure_no_open_run(environment):
    root,_,_=environment
    p.nested_eval('fixture')
    folder=paths(root)[0]
    with (folder/'fold-0/outer.csv').open('a') as f:f.write('corrupt')
    with pytest.raises(IntegrityError,match='hash'):p.open_sealed('fixture','fixture')
    assert not list((root/'outputs/runs').glob('*'))


def test_model_freeze_references_wrong_rules(environment):
    root,_,_=environment
    p.nested_eval('fixture')
    rules=__import__('json').loads((root/'outputs/frozen/rules-fixture.json').read_text())
    rules['rules_freeze_id']='different'
    write_json(root/'outputs/frozen/rules-different.json',rules)
    edit_json(root/'outputs/frozen/model-fixture.json',lambda r:r.__setitem__('rules_freeze_id','different'))
    with pytest.raises(IntegrityError,match='differ'):p.open_sealed('fixture','fixture')


@pytest.mark.parametrize('damage',['missing','duplicate','extra','wrong-fold'])
def test_open_rejects_invalid_ids_even_with_matching_file_hash(environment,damage):
    root,_,_=environment
    p.nested_eval('fixture')
    folder=paths(root)[0]
    file=folder/'fold-0/outer.csv'
    rows=pd.read_csv(file)
    if damage=='missing':rows=rows.iloc[1:]
    elif damage=='duplicate':rows.loc[1,'customer_id']=rows.loc[0,'customer_id']
    elif damage=='extra':rows.loc[0,'customer_id']=999999
    else:rows.loc[0,'fold']=1
    rows.to_csv(file,index=False)
    resign_test_fixture(folder,'fold-0/outer.csv')
    with pytest.raises((IntegrityError,ValueError)):p.open_sealed('fixture','fixture')


def test_open_aligns_permuted_rows_by_id(environment):
    root,_,_=environment
    p.nested_eval('fixture')
    folder=paths(root)[0]
    for k in range(5):
        file=folder/f'fold-{k}/outer.csv'
        pd.read_csv(file,converters={'score':float}).iloc[::-1].to_csv(file,index=False,float_format='%.17g')
        resign_test_fixture(folder,f'fold-{k}/outer.csv')
    run=p.open_sealed('fixture','fixture')
    assert run.meta['status']=='complete'
    from sklearn.metrics import roc_auc_score
    with sealed._access():
        rows=pd.read_csv(folder/'fold-0/outer.csv',converters={'score':float}).set_index('customer_id')
    expected=roc_auc_score(environment[2]['default'].loc[rows.index],rows['score'])
    assert pd.read_csv(run.result_path/'fold-0.csv')['AUC'].iloc[0]==pytest.approx(expected)


def test_seal_failure_retry_uses_new_folder_and_no_completed_record_is_overwritten(environment,monkeypatch):
    root,_,_=environment
    search=p.tune_model
    monkeypatch.setattr(p,'tune_model',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('synthetic failure')))
    with pytest.raises(RuntimeError):p.nested_eval('fixture')
    failed=paths(root)[0]
    with sealed._access():
        before=file_hash(failed/'seal.json')
        assert read_json(failed/'seal.json')['status']=='failed'
    monkeypatch.setattr(p,'tune_model',search)
    p.nested_eval('fixture')
    assert len(paths(root))==2
    with sealed._access():assert file_hash(failed/'seal.json')==before
