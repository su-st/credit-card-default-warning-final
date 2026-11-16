"""Integration gates enforce protocol-only data access after PR #9 migration."""
import ast
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import protocol as p
from src._protocol_runs import _OPENING, summarize_l3
from src.registry import IntegrityError, Registry, content_hash
from test_protocol import small, rule_fixture


def test_g2b_migration_gate_has_no_bypasses():
    assert p.audit_sources() == []


def test_new_direct_bypass_is_found_even_with_import_alias(tmp_path):
    (tmp_path/'experiments').mkdir()
    (tmp_path/'src').mkdir()
    (tmp_path/'experiments/bypass.py').write_text('from src.data import split as bypass\ntrain, held = bypass(data)\n')
    assert p.audit_sources(tmp_path)[0]['reason']=='direct data/test split bypasses protocol'
    (tmp_path/'experiments/bypass.py').write_text('from pandas import read_csv as r\np = OUT_DIR / "predictions.csv"\ndata = r(p)\n')
    assert p.audit_sources(tmp_path)[0]['reason']=='direct sealed/legacy handoff read'


def test_feature_contracts_of_all_migrated_experiments(small):
    from src.data import feature_cols
    for script, sensitive_values in [('04_shap.py',[True]),('06_fairness_groups.py',[True,False]),('07_group_threshold.py',[False]),('08_calibration.py',[True])]:
        source=(Path(__file__).parents[1]/'experiments'/script).read_text()
        tree=ast.parse(source)
        assert not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in {'test','y_test'} for n in ast.walk(tree))
        for sensitive in sensitive_values:
            _, cols=p.prepare(small,lag=0,sensitive=sensitive,engineered=False)
            assert cols==feature_cols(lag=0,sensitive=sensitive)
    source=(Path(__file__).parents[1]/'src/export.py').read_text()
    assert 'from .protocol import dev_holdout' in source
    assert 'split(load_data())' not in source


def test_legacy_l0_builder_body_is_unchanged():
    # C0's historical model is kept verbatim, not replaced by a compatibility shim.
    import hashlib
    import inspect
    import src.models
    source=inspect.getsource(src.models.make_model).strip()
    # Historical source-text SHA256 from main 83ff6d5. Unlike ast.dump, source
    # text is independent of Python's AST schema and shallow CI history.
    assert hashlib.sha256(source.encode('utf-8')).hexdigest()=='467245b148016022075a3d70face4801ae2fa0b99aa4f20576aa32f822c899dc'


def test_l3_separate_seed_runs_real_retraining_fixed_policy_and_summary(tmp_path,monkeypatch,small):
    rule_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(p,'_full_data',lambda:small.copy())
    X,cols=p.prepare(small,lag=0,sensitive=False,engineered=False)
    cfg={'name':'XGBoost','params':{'n_estimators':2,'n_jobs':1},'lag':0,'sensitive':False,'engineered':False}
    chosen={'entry_id':'fixed','model_id':'synthetic-frozen-model','model_hash':'f'*64,
            'columns':cols,'configuration':cfg,'protocol_policy':asdict(p.Policy.top_n(.2))}
    freeze={'rules_freeze_id':'fixture'}
    monkeypatch.setattr(Registry,'frozen_entry',lambda self,*a:(freeze,deepcopy(chosen),{},'e'*64))
    runs=[]
    for seed,train,test in p.outer_splits('fixture',seeds=[42,43]):
        run=p.start_run('L3','synthetic L3',model_freeze_id='fixture',entry='fixed')
        with pytest.raises(IntegrityError):run.update(model_hash=chosen['model_hash'])
        report=run.evaluate_split(seed,train,test)
        assert report['overall'].iloc[0]['名單人數']==4
        assert run.meta['seed']==seed and run.meta['model_hash']!=chosen['model_hash']
        assert run.meta['policy_id']==p.Policy.top_n(.2).policy_id
        runs.append(run)
    assert runs[0].run_id!=runs[1].run_id
    assert runs[0].meta['model_id']!=runs[1].meta['model_id']
    assert runs[0].meta['model_hash']!=runs[1].meta['model_hash']
    summary=summarize_l3(runs)
    assert summary['source_run_ids']==[r.run_id for r in runs]
    assert len(summary['by_seed'])==2
    assert set(summary['summary']['統計'])=={'mean','std','min','max'}
    assert not {'model_hash','policy_hash'}&set(summary['summary'].columns)


def test_l4_rejects_top_level_frozen_hash_even_with_valid_open_context(tmp_path,monkeypatch):
    rule_fixture(tmp_path,monkeypatch)
    token=_OPENING.set({'seal_id':'synthetic','seal_sha256':'a'*64,'data_sha256':'b'*64,'folds':[]})
    try:
        run=p.start_run('L4','synthetic opening guard',rules_freeze_id='fixture',model_freeze_id='fixture')
    finally:
        _OPENING.reset(token)
    with pytest.raises(IntegrityError):run.update(model_hash='f'*64)
    with pytest.raises(IntegrityError):run.update(policy_hash='f'*64)


def test_protocol_development_policy_exports_all_customers(tmp_path,monkeypatch):
    # End-to-end new protocol policy + existing C1 exporter, 4,800 synthetic IDs.
    from test_registry import synthetic as fixture
    from src.export import export_run
    data=fixture.__wrapped__()
    rule_fixture(tmp_path,monkeypatch)
    X=data.train[data.cols]
    factory=lambda:p.make_model('XGBoost',params={'n_estimators':2,'n_jobs':1})
    oof=p.oof_predict(factory,X,data.train['default'])
    run=p.start_run('L1-H','new policy export',model_id='synthetic-protocol-dev19200')
    run.attach_model(factory().fit(X,data.train['default']),X,data.train['default'],oof,prediction_ids=data.hold.index)
    policy=run.record_policy(p.Policy.single(.5,.2))
    bundle=export_run(run,data.hold)
    assert bundle['run']['verified_customers']==4800
    assert bundle['predictions']['in_list'].sum()<=960
    assert bundle['run']['policy_id']==policy.policy_id
