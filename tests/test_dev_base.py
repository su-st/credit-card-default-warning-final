"""G2a preparation: real protocol/registry/export on synthetic L1 customers."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src import protocol as p
from src.registry import Registry, file_hash, write_json
from test_protocol import rule_fixture
from test_registry import synthetic


def script():
    path = Path(__file__).parents[1]/'experiments/00_build_dev_base.py'
    spec = importlib.util.spec_from_file_location('build_dev_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_freeze_stops_before_data_or_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'dev_data', lambda: pytest.fail('loaded data before freeze'))
    with pytest.raises(FileNotFoundError):
        script().build_dev_base('missing')
    assert not (tmp_path/'outputs').exists()


def test_dev_base_three_models_same_l1_folds_and_verified_handoff(tmp_path, monkeypatch, synthetic):
    rules = rule_fixture(tmp_path, monkeypatch)
    rules['dev_base'] = {'seed': 42, 'folds': 5, 'params': {
        'n_estimators': 1, 'learning_rate': .1, 'max_depth': 1, 'min_child_weight': 1,
        'subsample': 1., 'colsample_bytree': 1., 'gamma': 0., 'reg_lambda': 1., 'n_jobs': 1}}
    freeze = tmp_path/'outputs/frozen/rules-fixture.json'
    write_json(freeze, rules, exclusive=False)
    frozen_digest = file_hash(freeze)
    # Extend an existing 24,000-row synthetic fixture to all original features.
    import pandas as pd
    frame = pd.concat([synthetic.train, synthetic.hold]).copy()
    for month in range(2, 7):
        for prefix in ['PAY_', 'BILL_AMT', 'PAY_AMT']:
            frame[f'{prefix}{month}'] = month
    frame['PAY_AMT1'] = 1
    frame.attrs['level'] = 'L1'
    a, b = frame.iloc[:19200].copy(), frame.iloc[19200:].copy()
    a.attrs.update(level='L1-H', role='train')
    b.attrs.update(level='L1-H', role='holdout')
    monkeypatch.setattr(p, 'dev_data', lambda: frame.copy())
    monkeypatch.setattr(p, 'dev_holdout', lambda: (a.copy(), b.copy()))
    monkeypatch.setattr(p, '_full_data', lambda: pytest.fail('bypassed injected development interfaces'))
    module = script()
    first = module.main(['--rules-freeze-id', 'fixture'])
    registry = Registry(tmp_path)
    assert first['baseline_role'].tolist() == ['M1', 'M1-NS', 'M1-NS-dev19200']
    assert first['training_count'].tolist() == [24000,24000,19200]
    assert first['feature_count'].tolist() == [23,19,19]
    assert first['model_id'].nunique() == first['run_id'].nunique() == first['policy_id'].nunique() == 3
    assert first.iloc[0]['folds_sha256'] == first.iloc[1]['folds_sha256']
    old = {}
    for i, row in first.iterrows():
        bundle = registry.load_run(row['run_id'])
        record = bundle['model_record']
        assert bundle['run']['status'] == 'complete'
        assert bundle['run']['rules_freeze_id'] == 'fixture'
        assert bundle['threshold']['source_level'] == row['level']
        assert bundle['threshold']['policy_id'] == row['policy_id']
        folder = registry.path(record['model_path']).parent
        assert (folder/'oof.csv').exists() and (folder/'folds.csv').exists()
        assert record['seed'] == 42 and record['n_splits'] == 5
        if i == 2:
            assert bundle['run']['verified_customers'] == 4800
            assert set(bundle['predictions']['data_split']) == {'dev_holdout'}
            assert not set(bundle['predictions']['customer_id']) & set(a.index)
            assert bundle['predictions']['in_list'].sum() <= np.floor(rules['scenario']['max_list_rate'] * 4800)
        else:
            assert bundle['predictions'] is None
        path = registry.runs/row['run_id']/'threshold.json'
        old[path] = file_hash(path)
    second = module.build_dev_base('fixture')
    assert not set(first['run_id']) & set(second['run_id'])
    assert not set(first['model_id']) & set(second['model_id'])
    assert (first['model_hash'].to_numpy() == second['model_hash'].to_numpy()).all()
    assert all(file_hash(path) == digest for path, digest in old.items())
    assert file_hash(freeze) == frozen_digest
    assert sorted(path.name for path in freeze.parent.iterdir()) == ['model-fixture.json', 'rules-fixture.json']


def test_bad_baseline_settings_stop_before_data(tmp_path, monkeypatch):
    rules = rule_fixture(tmp_path, monkeypatch)
    rules['dev_base'] = {'seed': 43, 'folds': 5, 'params': {}}
    path = tmp_path/'outputs/frozen/rules-fixture.json'
    write_json(path, rules, exclusive=False)
    monkeypatch.setattr(p, 'dev_data', lambda: pytest.fail('loaded data with invalid settings'))
    with pytest.raises(ValueError, match='seed 42'):
        script().build_dev_base('fixture')
    rules['dev_base']['seed'] = 42
    write_json(path, rules, exclusive=False)
    with pytest.raises(ValueError, match='explicitly supply'):
        script().build_dev_base('fixture')
