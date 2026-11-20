"""T-Chen2: actual synthetic pipelines, shared folds and policy provenance."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import protocol as p
from src.registry import IntegrityError, Registry, read_json
from test_model_freeze import sandbox


def module():
    path = Path(__file__).parents[1] / 'experiments/05_pay_categorical.py'
    spec = importlib.util.spec_from_file_location('pay_analysis', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pay_complete_analysis_has_six_comparisons_and_bound_thresholds(sandbox):
    root, frame, _ = sandbox
    run, compare, eligibility = module().run_analysis('synthetic-rules')
    assert run.meta['status'] == 'complete'
    assert len(compare) == 6
    assert set(compare.method) == {'M1-NS', '05'}
    assert set(compare.comparison) == {'single_0.5', 'top_20pct', 'formal_cost'}
    assert compare.folds_sha256.nunique() == 1
    assert compare.model_id.nunique() == 2
    assert len(set(compare.loc[compare.comparison == 'top_20pct', '名單人數'])) == 1
    assert compare.loc[compare.comparison == 'top_20pct', '名單人數'].eq(len(frame) // 5).all()
    assert eligibility.candidate.tolist() == ['05']
    registry = Registry(root)
    for row in compare.to_dict('records'):
        source = registry.load_run(row['run_id'])
        threshold = source['threshold']
        record = source['model_record']
        assert threshold['model_hash'] == row['model_hash'] == record['model_hash']
        assert threshold['oof_sha256'] == record['oof_sha256']
        assert record['training_count'] == len(frame)
        groups = pd.read_csv(root / 'results' / row['run_id'] / 'groups.csv')
        for _, group in groups.groupby('分組方式'):
            assert group['總人數'].sum() == len(frame)
            assert group['違約人數'].sum() == frame['default'].sum()
            assert group['未違約人數'].sum() == (frame['default'] == 0).sum()
        assert {'Recall 下限', 'Recall 上限', 'FPR 下限', 'FPR 上限', '小樣本'} <= set(groups)
    assert not list((root / 'outputs/frozen').glob('model-*.json'))
    assert (run.result_path / 'pay_categorical_compare.png').exists()


def test_pay_rejects_non_l1_before_training(sandbox, monkeypatch):
    _, frame, _ = sandbox
    bad = frame.copy()
    bad.attrs['level'] = 'L2'
    monkeypatch.setattr(p, 'dev_data', lambda: bad)
    monkeypatch.setattr(p, 'oof_predict', lambda *a, **k: pytest.fail('must reject before training'))
    with pytest.raises(IntegrityError, match='frozen seed-42 L1'):
        module().run_analysis('synthetic-rules')


def test_pay_detects_changed_candidate_fold_assignment(sandbox, monkeypatch):
    real = p.oof_predict
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        result = real(*args, **kwargs)
        calls += 1
        if calls == 2:
            fold = result.attrs['fold_assignment'].copy()
            fold.iloc[0] = (fold.iloc[0] + 1) % kwargs['folds']
            result.attrs['fold_assignment'] = fold
        return result

    monkeypatch.setattr(p, 'oof_predict', changed)
    with pytest.raises(IntegrityError, match='identical ordered L1 customers and folds'):
        module().run_analysis('synthetic-rules')


def test_pay_import_does_not_load_data_or_start_a_run(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('import must be inert')
    monkeypatch.setattr(p, 'dev_data', forbidden)
    monkeypatch.setattr(p, 'start_run', forbidden)
    assert module().TASK == 'T-陳2'
