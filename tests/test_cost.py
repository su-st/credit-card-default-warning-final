"""C3: hand-calculated costs, capacity, frozen L1 evidence and persistence."""
from copy import deepcopy
from dataclasses import asdict
from itertools import product

import numpy as np
import pandas as pd
import pytest

from src import config, protocol as p
from src.registry import IntegrityError, Registry, file_hash, read_json, write_json
from src.threshold import analyze_cost_scenarios, run_cost_analysis, total_cost
from test_protocol import small, rule_fixture


def hand_data():
    df = pd.DataFrame({'default': [1, 1, 0, 1, 0],
                       'BILL_AMT1': [1000, 2000, 9000, -200, 500]},
                      index=pd.Index([5, 4, 3, 2, 1], name='ID'))
    return df, pd.Series([.9, .8, .8, .2, .1], index=df.index)


def test_hand_cost_after_cap_and_negative_bill_clipping(monkeypatch):
    df, score = hand_data()
    # floor(.4*5)=2; .8 tie goes to ID 3 (non-default), so selected IDs 5 and 3.
    scenario = p.Scenario(.4, .5, 50, .2)
    calls = []
    original = p.apply_policy
    def spy(score, policy, **kwargs):
        result = original(score, policy, **kwargs)
        calls.append((policy.threshold, int(result[0].sum())))
        return result
    monkeypatch.setattr(p, 'apply_policy', spy)
    table, policies = analyze_cost_scenarios(df['default'], score, p.bill_at(df, 0), [scenario])
    row = table.iloc[0]
    # At the optimum: FN=1000 + TP=400 + one contact=50 => 1450.
    assert row['總成本'] == 1450
    assert row['門檻'] == .9
    assert {t for t, _ in calls} == set(np.round(np.arange(.05, .951, .01), 2))
    assert all(n <= 2 for _, n in calls)
    assert row['Recall'] == pytest.approx(1/3)
    assert row['F1'] == .5
    assert row['名單人數'] == 1
    assert policies[0].scenario == asdict(scenario)


def test_all_candidates_over_cap_are_valid_and_highest_tie_wins():
    df, _ = hand_data()
    score = pd.Series(.99, index=df.index)
    table, policies = analyze_cost_scenarios(df['default'], score, p.bill_at(df, 0),
                                           [p.Scenario(.4, .5, 50, .2)])
    # All 91 thresholds require truncation, selecting IDs 1 and 2 on tied scores.
    row = table.iloc[0]
    assert row['門檻'] == .95
    assert row['名單上限是否觸發']
    assert row['原始名單人數'] == 5 and row['名單人數'] == 2
    assert row['總成本'] == 1600  # 500+1000 FN + zero-loss TP + 2*50
    assert p.apply_policy(score, policies[0])[0].loc[[1, 2]].eq(1).all()


@pytest.mark.parametrize('cap', [0, .1, .2, .3, 1])
def test_capacity_on_selection_and_new_evaluation_batch(cap):
    df, score = hand_data()
    table, policies = analyze_cost_scenarios(df['default'], score, p.bill_at(df, 0),
                                           [p.Scenario(cap, .5, 50, .2)])
    assert table.iloc[0]['名單人數'] <= np.floor(cap * len(df))
    larger = pd.Series(.99, index=range(1, 18))
    pred, _ = p.apply_policy(larger, policies[0])
    assert pred.sum() <= np.floor(cap * len(larger))


@pytest.mark.parametrize('field,value', [(f, -1) for f in ('max_list_rate','loss_ratio','contact_cost','rescue_rate')]
                         + [('rescue_rate', 1.01), ('contact_cost', np.inf), ('loss_ratio', np.nan)])
def test_invalid_assumptions_raise(field, value):
    values = asdict(p.Scenario(.2, .5, 50, .2))
    values[field] = value
    with pytest.raises(ValueError):
        p.Scenario(**values)


def test_d2_four_axes_each_respects_own_capacity_and_policy_identity():
    assert (config.LOSS_RATIO, config.MAX_LIST_RATE, config.COST_PER_CONTACT,
            config.RESCUE_RATE, config.THRESHOLD_RULE) == (.5, .2, 50, .2, 'cost')
    axes = config.COST_SENSITIVITY
    assert axes == {'max_list_rate': (.1,.2,.3), 'loss_ratio': (.5,.75,1.),
                    'contact_cost': (50,500,2000), 'rescue_rate': (.1,.2,.3)}
    scenarios = [p.Scenario(*values) for values in product(*axes.values())]
    df, score = hand_data()
    table, policies = analyze_cost_scenarios(df['default'], score, p.bill_at(df, 0), scenarios)
    assert len(table) == 81 and len({pol.policy_id for pol in policies}) == 81
    assert (table['名單人數'] <= np.floor(table['max_list_rate'] * len(df))).all()
    assert table['名單上限是否觸發'].notna().all()
    # Legacy average loss remains available with the original formula.
    assert total_cost({'FN': 2, 'TP': 1, 'FP': 1}, 1000, 50, .2) == 2900


@pytest.fixture
def registered(tmp_path, monkeypatch, small):
    rules = rule_fixture(tmp_path, monkeypatch)
    # Deliberately unlike config: the executable JSON, not config, must win.
    rules['scenario'] = asdict(p.Scenario(.3, .75, 500, .1))
    rules['cost_sensitivity'] = {'max_list_rate': [.1, .2, .3], 'loss_ratio': [.5],
                                 'contact_cost': [50], 'rescue_rate': [.2]}
    write_json(tmp_path/'outputs/frozen/rules-fixture.json', rules, exclusive=False)
    small.attrs['level'] = 'L1'
    monkeypatch.setattr(p, 'dev_data', lambda: small.copy())
    X, _ = p.prepare(small, lag=0, sensitive=False, engineered=False)
    factory = lambda: p.make_model('XGBoost', params={'n_estimators': 1, 'n_jobs': 1})
    oof = p.oof_predict(factory, X, small['default'])
    run = p.start_run('L1', 'synthetic C3 model source', model_id='synthetic-c3')
    record = run.attach_model(factory().fit(X, small['default']), X, small['default'], oof)
    run.record_policy(p.Policy.single(.5, None))
    run.finish()
    return record, rules


def test_frozen_l1_end_to_end_all_ids_immutable_and_comparisons(registered, tmp_path):
    record, rules = registered
    result = run_cost_analysis(record, rules_freeze_id='fixture')
    assert len(result) == 6  # main + unbounded cost + 3 sensitivity + F1
    assert result['run_id'].nunique() == 6
    registry = Registry(tmp_path)
    assert set(result['比較']) == {'主要成本政策','無上限成本對照','敏感度情境','無上限F1對照'}
    assert result.iloc[0]['max_list_rate'] == .3
    assert result.iloc[0]['contact_cost'] == 500
    assert result.iloc[0]['loss_ratio'] == .75
    assert (result['名單人數'] <= np.floor(result['max_list_rate'] * 100)).all()
    hashes = {}
    for _, row in result.iterrows():
        bundle = registry.load_run(row['run_id'])
        assert bundle['model_record']['model_hash'] == record['model_hash']
        assert bundle['threshold']['policy_id'] == row['policy_id']
        assert bundle['run']['level'] == 'L1'
        path = registry.runs/row['run_id']/'threshold.json'
        hashes[path] = file_hash(path)
        assert (registry.results/row['run_id']/'cost_summary.csv').exists()
    # Retry gets new run IDs without touching completed records.
    retry = run_cost_analysis(record, rules_freeze_id='fixture')
    assert not set(result['run_id']) & set(retry['run_id'])
    assert all(file_hash(path) == digest for path, digest in hashes.items())


def test_reject_holdout_before_loading_development_data(registered, monkeypatch):
    record, _ = registered
    record = deepcopy(record)
    record['level'] = 'L1-H'
    # Model hash/record content are normally verified by load_model.
    monkeypatch.setattr(Registry, 'load_model', lambda *args: None)
    monkeypatch.setattr(p, 'dev_data', lambda: pytest.fail('loaded data for rejected level'))
    with pytest.raises(IntegrityError, match='L1 OOF'):
        run_cost_analysis(record, rules_freeze_id='fixture')


def test_reject_changed_training_ids_labels_or_data(registered, monkeypatch, small):
    record, _ = registered
    altered = small.copy()
    altered.iloc[0, altered.columns.get_loc('default')] ^= 1
    monkeypatch.setattr(p, 'dev_data', lambda: altered)
    with pytest.raises(IntegrityError, match='IDs/labels'):
        run_cost_analysis(record, rules_freeze_id='fixture')
    monkeypatch.setattr(p, 'dev_data', lambda: small.iloc[::-1])
    with pytest.raises(IntegrityError, match='IDs/order'):
        run_cost_analysis(record, rules_freeze_id='fixture')
    (p.DATA_DIR/'default_of_credit_card_clients.csv').write_text('changed synthetic bytes')
    with pytest.raises(IntegrityError, match='data differs'):
        run_cost_analysis(record, rules_freeze_id='fixture')


def test_missing_frozen_sensitivity_never_falls_back(registered, tmp_path):
    record, rules = registered
    del rules['cost_sensitivity']
    write_json(tmp_path/'outputs/frozen/rules-fixture.json', rules, exclusive=False)
    with pytest.raises(AttributeError, match='cost_sensitivity'):
        run_cost_analysis(record, rules_freeze_id='fixture')
