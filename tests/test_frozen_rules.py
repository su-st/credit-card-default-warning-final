"""Validate the approved G2a artifact without data access, model fits or experiments."""
import ast
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import ParameterSampler

from src import protocol as p
from src.config import XGB_TUNED
from src.registry import content_hash, file_hash
from src._sealed import _configuration

ROOT = Path(__file__).parents[1]


def test_approved_freeze_loads_exact_markdown_and_existing_l4_contract():
    rules = p.load_rules('g2a-20261116')
    assert rules.status == 'approved' and rules.approved_at == '2026-11-16'
    assert rules.markdown_sha256 == file_hash(ROOT/'docs/frozen_rules.md')
    assert rules.schedule['previous_g2a_date'] == '2026-11-18'
    assert rules.schedule['g2a_date'] == '2026-11-16'
    assert _configuration(rules) == rules.l4
    assert rules.l4_configuration_hash == content_hash(rules.l4)
    assert (rules.l4['seed'], rules.l4['n_iter'], rules.l4['outer_folds'], rules.l4['inner_folds']) == (42,60,5,5)
    assert rules.l4['scenario'] == asdict(rules.scenario)
    assert rules.dev_base == {'params': {**XGB_TUNED, 'random_state':42, 'eval_metric':'logloss'}, 'seed':42, 'folds':5}
    text = (ROOT/'docs/frozen_rules.md').read_text()
    assert not any(x in text for x in ['待組長確認','待確認','尚未凍結','準備稿'])


def test_complete_01_choice_lists_and_c2_sampler_tie_rule_without_training(monkeypatch):
    rules = p.load_rules('g2a-20261116')
    tree = ast.parse((ROOT/'experiments/01_tuning.py').read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='p' for t in n.targets))
    choices = {kw.arg: ast.literal_eval(next(n.args[0] for n in ast.walk(kw.value)
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='choice'))
               for kw in assignment.value.keywords}
    assert rules.search_space == choices
    expected = list(ParameterSampler(choices, n_iter=60, random_state=42))
    # Invoke actual C2 sampler/selection, but stub all CV: no fit, no real data.
    def no_training_cv(factory, X, y, *, folds, seed):
        assert (folds,seed)==(5,42)
        return pd.DataFrame({'AUC':[.75]*5})
    monkeypatch.setattr(p, 'cv_evaluate', no_training_cv)
    monkeypatch.setattr(p, 'make_model', lambda *a,**k: (_ for _ in ()).throw(AssertionError('model creation forbidden')))
    out = p.tune_model(pd.DataFrame(),pd.Series(dtype=int),choices,n_iter=60,folds=5,seed=42)
    assert out['實際搜尋組數']==60
    assert out['最佳參數']=={**expected[0],'random_state':42}
    assert out['完整搜尋紀錄']['params'].tolist()==[{**v,'random_state':42} for v in expected]


def test_candidate_rules_pay_evidence_legacy_and_decision_scopes():
    rules = p.load_rules('g2a-20261116')
    candidates = {c['name']:c for c in rules.candidates}
    assert list(candidates)==['M1','M1-NS','05','09 B','09 C','09 D','09 E']
    assert [candidates[n]['input_feature_count_before_onehot'] for n in candidates]==[23,19,19,23,26,26,19]
    assert candidates['09 D']['features']==candidates['09 C']['features']
    assert candidates['09 D']['score_combination']=='direct_merge_without_calibration'
    assert candidates['09 E']['g2b_replacement_eligible'] is False
    assert candidates['09 E']['platform_use'] is False
    assert rules.g2b['eligible_candidates']==['05','09 B','09 C','09 D']
    assert [v['metric'] for v in rules.g2b['ranking']]==['recall_gain','max(gender_eo_increase, age_eo_increase)','gender_eo_increase + age_eo_increase','input_feature_count_before_onehot','fixed_candidate_order']
    mapping=rules.pay_categorical['mapping']
    assert len({mapping[str(v)] for v in [-2,-1,0,1,2]})==5
    assert {mapping[str(v)] for v in range(3,10)}=={'延遲三個月以上'}
    evidence=rules.pay_categorical['l1_verification']
    assert (evidence['level'],evidence['seed'],evidence['rows'],evidence['month_values'])==('L1',42,24000,144000)
    assert evidence['columns']==[f'PAY_{m}' for m in range(1,7)]
    assert sum(evidence['counts'].values())==144000
    assert sum(evidence['counts'].get(str(v),0) for v in range(3,9))==1821
    assert evidence['counts'].get('9',0)==0
    assert all(evidence['counts'][str(v)]/144000<.001 for v in [5,6,8])
    legacy=rules.legacy_threshold_analysis
    assert legacy['loss_per_default'] is None and legacy['max_list_rate'] is None
    assert (legacy['cost_per_contact'],legacy['rescue_rate'],legacy['rule'],legacy['top_n_rate'])==(50,.2,'cost',.2)
    assert legacy['sensitivity_contact_costs']==[50,500,2000] and legacy['sensitivity_rescue_rates']==[.1,.2,.3]
    assert rules.group_threshold['target_definition']=='single_policy_oof_recall'
    assert rules.group_threshold['gap_round_decimals']==4 and rules.group_threshold['tie_break']=='higher_threshold'
    assert all(c['implementation_acceptance']=='separate_G2b_verification' for c in candidates.values())
