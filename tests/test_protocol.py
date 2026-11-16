"""C2 contract, statistical hand calculations and data-layer safety tests."""
from copy import deepcopy
from dataclasses import asdict
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from src import protocol as p
from src.calibration import crossfit_calibrate, calibration_report, rank_change
from src.features import add_features, engineered_cols
from src.data import feature_cols
from src.models import make_model as historical_model
from src.registry import IntegrityError, Registry, content_hash, file_hash, read_json, write_json
from src._protocol_runs import _OPENING, _policy_for_split


@pytest.fixture
def small():
    rng = np.random.default_rng(4)
    n = 100
    df = pd.DataFrame({"LIMIT_BAL": rng.integers(20, 100, n), "SEX": rng.integers(1,3,n),
        "AGE": rng.integers(22,65,n), "EDUCATION": rng.integers(1,5,n), "MARRIAGE": rng.integers(1,4,n),
        "default": np.tile([0, 0, 1, 0, 1], n//5)}, index=pd.Index(range(101,201),name="ID"))
    for month in range(1,7):
        df[f"PAY_{month}"] = rng.integers(-2,4,n)
        df[f"BILL_AMT{month}"] = rng.integers(-10,100,n)
        df[f"PAY_AMT{month}"] = rng.integers(0,100,n)
    return df


def rule_fixture(tmp_path, monkeypatch, *, space=None):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    (data_dir/'default_of_credit_card_clients.csv').write_text('Synthetic fixture bytes, not UCI data')
    monkeypatch.setattr(p, 'DATA_DIR', data_dir)
    frozen = tmp_path / 'outputs/frozen'
    frozen.mkdir(parents=True,exist_ok=True)
    (tmp_path/'docs').mkdir(exist_ok=True)
    md=tmp_path/'docs/frozen_rules.md'
    md.write_text('Synthetic fixture only, no project rules')
    scenario=asdict(p.Scenario(.2,.5,1,.2))
    space=space or {'n_estimators':[1,2], 'max_depth':list(range(1,16)), 'n_jobs':[1]}
    cfg={'model':'M1-NS','outer_folds':5,'inner_folds':5,'seed':42,'n_iter':30,'search_space':space,'scenario':scenario}
    rules={'rules_freeze_id':'fixture','markdown_sha256':file_hash(md), 'scenario':scenario,
        'n_iter':30,'search_space':space,'l4':cfg,'l4_configuration_hash':content_hash(cfg),
        'group_threshold':{'target_definition':'single_policy_oof_recall'}}
    write_json(frozen/'rules-fixture.json',rules)
    write_json(frozen/'model-fixture.json',{'model_freeze_id':'fixture','rules_freeze_id':'fixture',
        'approved_at':'synthetic-fixture','entries':[]})
    return rules


def test_exact_public_signatures():
    expected={
        'dev_data':'()', 'dev_holdout':'()', 'final_test':'(model_freeze_id)',
        'bill_at':'(df, lag)', 'prepare':'(df, *, lag, sensitive, engineered)',
        'make_model':'(name, *, params=None)', 'oof_predict':'(model_fn, X, y, *, folds=5, seed=42)',
        'cv_evaluate':'(model_fn, X, y, *, folds=5, seed=42)',
        'load_rules':'(rules_freeze_id)', 'tune_model':'(X, y, search_space, *, n_iter, folds=5, seed=42)',
        'nested_eval':'(rules_freeze_id)', 'open_sealed':'(rules_freeze_id, model_freeze_id)',
        'apply_policy':'(score, policy, *, partition=None)', 'choose_threshold':'(y, score, bill, scenario)',
        'coverage_curve':'(y, score, rates, bill=None)',
        'evaluate':'(y, *, score=None, policy=None, pred=None, partition=None, groups=None, bill=None, scenario=None)',
        'gap_ci':'(y, pred, partition, *, n_boot=2000, seed=42, pred_b=None)',
        'start_run':'(level, description, *, model_id=None, rules_freeze_id=None, model_freeze_id=None, entry=None)',
    }
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(p,name)))==signature
    assert str(inspect.signature(p.outer_splits))=='(model_freeze_id, seeds=range(42, 52))'
    assert str(inspect.signature(p.Policy.single))=='(threshold, max_list_rate)'
    assert str(inspect.signature(p.Policy.group))=='(partition_name, thresholds, max_list_rate)'
    assert str(inspect.signature(p.Policy.top_n))=='(rate)'
    assert str(inspect.signature(crossfit_calibrate))=='(y, oof_score, *, folds=5, seed=42)'
    assert str(inspect.signature(calibration_report))=='(y, score, *, bins=10)'
    assert str(inspect.signature(rank_change))=='(score_a, score_b, rate)'
    from src.explain_shap import shap_values, global_importance
    assert str(inspect.signature(shap_values))=='(model, X, *, merge_onehot=True)'
    assert str(inspect.signature(global_importance))=='(shap_df)'
    from src._protocol_runs import ProtocolRun
    assert str(inspect.signature(ProtocolRun.save_csv))=='(self, name, df)'
    assert str(inspect.signature(ProtocolRun.save_fig))=='(self, name, fig)'
    assert str(inspect.signature(ProtocolRun.finish))=='(self)'
    assert list(inspect.signature(p.Scenario).parameters)==['max_list_rate','loss_ratio','contact_cost','rescue_rate']


def test_data_layers_gate_before_loading_and_repeat_splits(tmp_path,monkeypatch,small):
    rule_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(p,'_full_data',lambda: pytest.fail('loaded data before checking freeze'))
    with pytest.raises(FileNotFoundError): p.final_test('absent')
    with pytest.raises(FileNotFoundError): next(p.outer_splits('absent'))
    monkeypatch.setattr(p,'_full_data',lambda:small.copy())
    dev=p.dev_data()
    a,b=p.dev_holdout()
    final=p.final_test('fixture')
    assert len(dev)==80 and len(a)==64 and len(b)==16 and len(final)==20
    assert not set(dev.index)&set(final.index)
    assert not set(a.index)&set(b.index)
    splits=list(p.outer_splits('fixture',seeds=[42,43]))
    repeat=list(p.outer_splits('fixture',seeds=[42,43]))
    assert splits[0][1].index.equals(repeat[0][1].index)
    assert not splits[0][2].index.equals(splits[1][2].index)
    assert splits[0][1].index.equals(dev.index)


@pytest.mark.parametrize('lag',range(5))
@pytest.mark.parametrize('sensitive',[True,False])
@pytest.mark.parametrize('engineered',[True,False])
def test_prepare_available_months_and_invariance(small,lag,sensitive,engineered):
    X,cols=p.prepare(small,lag=lag,sensitive=sensitive,engineered=engineered)
    assert cols==X.columns.tolist()
    assert not {'default','ID','customer_id'}&set(cols)
    for month in range(1,lag+1):
        assert not {f'PAY_{month}',f'BILL_AMT{month}',f'PAY_AMT{month}',f'UTIL_{month}',f'PAYRATIO_{month}'}&set(cols)
    changed=small.copy()
    for month in range(1,lag+1):
        for prefix in ('PAY_','BILL_AMT','PAY_AMT'):
            changed[f'{prefix}{month}']=9999999
    assert_frame_equal(X,p.prepare(changed,lag=lag,sensitive=sensitive,engineered=engineered)[0])
    original=feature_cols(lag=lag,sensitive=sensitive)
    assert cols[:len(original)]==original
    if lag==0 and engineered:
        old=add_features(small)
        assert cols==original+engineered_cols(old)
        assert_frame_equal(X,old[cols])


def test_each_fit_uses_only_its_labels_and_legacy_is_unchanged(small):
    X,_=p.prepare(small,lag=0,sensitive=True,engineered=False)
    model=p.make_model('XGBoost',params={'n_estimators':2,'n_jobs':1})
    a=clone(model).fit(X.iloc[:50],small['default'].iloc[:50])
    idx=np.r_[np.flatnonzero(small['default'].to_numpy()==0)[:30],np.flatnonzero(small['default'].to_numpy()==1)[:5]]
    b=clone(model).fit(X.iloc[idx],small['default'].iloc[idx])
    assert a['clf'].get_params()['scale_pos_weight']==1.5
    assert b['clf'].get_params()['scale_pos_weight']==6
    legacy=historical_model('XGBoost',X.columns,small['default'])
    assert legacy['clf'].get_params()['scale_pos_weight']==1.5
    assert type(legacy['clf']) is not type(model['clf'])
    for name in ['邏輯迴歸','決策樹']:
        assert p.make_model(name)['clf'].get_params()['class_weight']=='balanced'
    assert p.make_model('邏輯迴歸')['clf'].get_params()['max_iter']==3000


def test_oof_cv_tune_schemas_provenance_and_reproducibility(small):
    X,_=p.prepare(small,lag=0,sensitive=False,engineered=False)
    factory=lambda:p.make_model('XGBoost',params={'n_estimators':2,'n_jobs':1})
    score=p.oof_predict(factory,X,small['default'])
    assert score.index.equals(X.index) and not score.isna().any()
    assert set(score.attrs['fold_assignment'])==set(range(5))
    for row in score.attrs['fold_sources']:
        assert row['training_ids_hash']!=row['validation_ids_hash']
    cv=p.cv_evaluate(factory,X,small['default'])
    assert cv.columns.tolist()==['折編號','AUC','平均','標準差']
    assert cv['平均'].nunique()==1
    space={'n_estimators':[1,2],'n_jobs':[1]}
    result=p.tune_model(X,small['default'],space,n_iter=2)
    other=p.tune_model(X,small['default'],space,n_iter=2)
    assert set(result)=={'最佳參數','最佳 CV AUC','完整搜尋紀錄','實際搜尋組數','搜尋範圍雜湊'}
    assert result['實際搜尋組數']==2 and result['搜尋範圍雜湊']==content_hash(space)
    assert_frame_equal(result['完整搜尋紀錄'],other['完整搜尋紀錄'])
    with pytest.raises(ValueError):p.tune_model(X,small['default'],space,n_iter=3)


def examples():
    ids=pd.Index([8,2,5,3])
    return (pd.Series([1,0,1,0],index=ids),pd.Series([.9,.9,.6,.1],index=ids),
            pd.Series([100.,50.,200.,0.],index=ids),pd.Series(['a','a','b','b'],index=ids))


def test_policy_lists_cost_tie_grid_and_coverage_hand_calculation():
    y,score,bill,group=examples()
    pred,info=p.apply_policy(score,p.Policy.single(.5,.25))
    assert pred.to_dict()=={8:0,2:1,5:0,3:0}
    assert info['原始名單人數']==3 and info['名單上限是否觸發']
    assert p.apply_policy(score,p.Policy.top_n(.5))[0].to_dict()=={8:1,2:1,5:0,3:0}
    assert p.apply_policy(score,p.Policy.top_n(0))[0].sum()==0
    assert p.apply_policy(score,p.Policy.top_n(1))[0].sum()==4
    with pytest.raises(ValueError):p.apply_policy(score,p.Policy.group('g',{'a':.5,'b':.5},.5))
    with pytest.raises(ValueError):p.apply_policy(score,p.Policy.group('g',{'a':.5},.5),partition=group)
    scenario=p.Scenario(.25,.5,1,.2)
    chosen=p.choose_threshold(y,score,bill,scenario)
    # Any threshold <=.9 takes the highest tied ID=2 (a false positive):
    # FN loss 150 plus contact 1. Above .9 selects none: 150. Tie picks .95.
    assert chosen['門檻']==.95 and chosen['總成本']==150
    assert chosen['最終名單人數']==0
    zero=p.choose_threshold(y,score,bill,p.Scenario(.25,0,0,0))
    assert zero['門檻']==.95
    curve=p.coverage_curve(y,score,[.5],bill=bill)
    assert curve.iloc[0]['違約者涵蓋率']==.5
    assert curve.iloc[0]['帳單金額涵蓋率']==pytest.approx(1/3)
    for args in [(-.1,.5,1,.2),(.2,-1,1,.2),(.2,.5,-1,.2),(.2,.5,1,1.1)]:
        with pytest.raises(ValueError):p.Scenario(*args)


def test_evaluate_schema_three_forms_zero_denominators_and_wilson():
    y,score,bill,group=examples()
    policy=p.Policy.single(.5,None)
    pred,_=p.apply_policy(score,policy)
    a=p.evaluate(y,score=score,policy=policy,groups={'g':group},bill=bill,scenario=p.Scenario(1,.5,1,.2))
    b=p.evaluate(y,score=score,pred=pred,policy=policy)
    c=p.evaluate(y,pred=pred,policy=policy)
    assert a['overall'].columns.tolist()==p.OVERALL_COLUMNS
    assert a['groups'].columns.tolist()==p.GROUP_COLUMNS
    assert a['gaps'].columns.tolist()==p.GAP_COLUMNS
    row=a['overall'].iloc[0]
    assert row['Precision']==pytest.approx(2/3) and row['Recall']==1 and row['F1']==pytest.approx(.8)
    assert row['總成本']==123
    assert b['overall']['AUC'].iloc[0]==a['overall']['AUC'].iloc[0]
    assert np.isnan(c['overall']['AUC'].iloc[0]) and np.isnan(c['overall']['原始名單人數'].iloc[0])
    assert np.isnan(p.evaluate(y,pred=pred,policy=policy,bill=bill)['overall']['帳單金額涵蓋率'].iloc[0])
    with pytest.raises(ValueError):p.evaluate(y,score=score,pred=1-pred,policy=policy)
    with pytest.raises(ValueError):p.evaluate(y,score=score)
    with pytest.raises(ValueError):p.evaluate(y,pred=pred,policy=p.Policy.group('g',{'a':.5,'b':.5},None))
    only_positive=pd.Series(['positive' if v else 'negative' for v in y],index=y.index)
    groups=p.evaluate(y,score=score,policy=policy,groups={'kind':only_positive})['groups'].set_index('族群')
    assert np.isnan(groups.loc['negative','Recall'])
    assert np.isnan(groups.loc['positive','FPR'])
    assert groups['小樣本'].all()
    lo,hi=p._wilson(10,20)
    assert lo==pytest.approx(.299298008,abs=1e-8) and hi==pytest.approx(.700701992,abs=1e-8)


def test_bootstrap_stratified_paired_differences_and_zero_denominator():
    y,score,_,group=examples()
    pred=p.apply_policy(score,p.Policy.single(.5,None))[0]
    result=p.gap_ci(y,pred,group,n_boot=40,seed=42,pred_b=pred.copy())
    assert (result[['估計值','下限','上限']]==0).all().all()
    assert (result['有效重抽']==40).all() and result['可靠'].all()
    # Single positive/negative per stratum => every resample is unchanged.
    one=p.gap_ci(y,pred,group,n_boot=40)
    assert (one['下限']==one['上限']).all()
    no_positive=pd.Series(0,index=y.index)
    assert np.isnan(p.gap_ci(no_positive,pred,group,n_boot=10).iloc[0]['下限'])


def test_calibration_crossfit_sources_brier_and_rank_change():
    y=pd.Series(np.tile([0,1],10),index=range(1,21))
    score=pd.Series(np.linspace(.05,.95,20),index=y.index)
    calibrated,source=crossfit_calibrate(y,score)
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    for fold,(tr,va) in enumerate(cv.split(score,y)):
        expected=IsotonicRegression(out_of_bounds='clip').fit(score.iloc[tr],y.iloc[tr]).predict(score.iloc[va])
        np.testing.assert_array_equal(calibrated.iloc[va],expected)
        assert source.iloc[fold]['訓練ID雜湊']==content_hash(y.index[tr].tolist())
        assert not set(y.index[tr])&set(y.index[va])
    report=calibration_report(y,score)
    assert report['Brier分數']==pytest.approx(float(np.mean((y-score)**2)))
    assert set(report)=={'Brier分數','校準曲線'}
    assert rank_change(score,score,.2)==1
    assert rank_change(score,1-score,.2)==0


def test_signed_shap_merge_before_absolute_value(small,monkeypatch):
    from src.explain_shap import shap_values,global_importance
    X=small[['EDUCATION','MARRIAGE','LIMIT_BAL']]
    model=p.make_model('XGBoost',params={'n_estimators':1,'n_jobs':1}).fit(X,small['default'])
    width=len(model['pre'].get_feature_names_out())
    values=np.zeros((len(X),width+1))
    education=[i for i,n in enumerate(model['pre'].get_feature_names_out()) if n.startswith('cat__EDUCATION_')]
    values[:,education[0]]=3
    values[:,education[1]]=-2
    monkeypatch.setattr(type(model['clf'].get_booster()),'predict',lambda *a,**kw:values)
    merged=shap_values(model,X)
    assert (merged['EDUCATION']==1).all()
    assert global_importance(merged)['EDUCATION']==1
    assert shap_values(model,X,merge_onehot=False).shape[1]==width


def test_common_has_no_test_globals_or_import_data_side_effect(monkeypatch,tmp_path):
    import importlib.util
    monkeypatch.setattr(p,'dev_data',lambda:pytest.fail('loaded data on import'))
    spec=importlib.util.spec_from_file_location('common_isolation',Path(__file__).parents[1]/'experiments/_common.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not hasattr(module,'test') and not hasattr(module,'y_test')
    sealed=tmp_path/'outputs/sealed/fixture.csv'
    sealed.parent.mkdir(parents=True)
    sealed.write_text('x\n1\n')
    with pytest.raises(PermissionError):module.pd.read_csv(sealed)
    with pytest.raises(PermissionError):read_json(sealed)
    with pytest.raises(PermissionError):file_hash(sealed)


def test_start_run_guards_and_protocol_policy_model_identity(tmp_path,monkeypatch,small):
    rule_fixture(tmp_path,monkeypatch)
    for level in ['L0','L1','L1-H']:
        with pytest.raises(ValueError):p.start_run(level,'missing model')
    for level in ['L2','L3','L4']:
        with pytest.raises((ValueError,FileNotFoundError)):
            p.start_run(level,'missing freeze')
    with pytest.raises(PermissionError):p.start_run('L4','cannot open',rules_freeze_id='fixture',model_freeze_id='fixture')
    X,_=p.prepare(small,lag=0,sensitive=False,engineered=False)
    factory=lambda:p.make_model('XGBoost',params={'n_estimators':2,'n_jobs':1})
    oof=p.oof_predict(factory,X,small['default'])
    run=p.start_run('L1','fixture',model_id='synthetic-l1')
    run.attach_model(factory().fit(X,small['default']),X,small['default'],oof)
    policy=run.record_policy(p.Policy.single(.5,None))
    run.save_csv('metrics',p.evaluate(small['default'],score=oof,policy=policy)['overall'])
    run.finish()
    record=p._registry().load_run(run.run_id)
    assert record['run']['policy_id']==policy.policy_id
    assert file_hash(run.result_path/'run.json')==file_hash(run.path/'run.json')
    with pytest.raises(IntegrityError):run.save_csv('again',pd.DataFrame())


def test_l3_fixed_policies_and_recomputed_cost_group_policy():
    y,score,bill,group=examples()
    rules=SimpleNamespace(scenario=p.Scenario(.5,.5,1,.2),group_threshold={'target_definition':'single_policy_oof_recall'})
    for policy in [p.Policy.single(.5,None),p.Policy.top_n(.2)]:
        assert _policy_for_split(y,score,bill,policy,rules).policy_id==policy.policy_id
    cost=p.choose_threshold(y,score,bill,rules.scenario)['政策物件']
    changed=_policy_for_split(y,score*.9,bill,cost,rules)
    assert changed.evidence['oof_scores_hash']!=cost.evidence['oof_scores_hash']
    grouped=_policy_for_split(y,score,bill,p.Policy.group('g',{'a':.5,'b':.5},.5),rules,group)
    assert set(grouped.thresholds)=={'a','b'}
    assert p.apply_policy(score,grouped,partition=group)[0].sum()<=2
    assert grouped.scenario==asdict(rules.scenario)
