"""C9a: synthetic customers only; never load the real credit data."""
from copy import deepcopy
import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import frozen_methods as fm, protocol as p
from src.registry import IntegrityError


@pytest.fixture
def rules():
    return p.load_rules('g2a-20261116')


@pytest.fixture
def frame(rules):
    n = 40
    df = pd.DataFrame(100., index=pd.Index(range(101, 101+n)), columns=rules.candidates[1]['features'])
    for col in rules.pay_categorical['columns']:
        df[col] = 0
    # Each group has both classes in sufficient numbers for shared five-fold OOF.
    df.loc[df.index[n//2:], 'PAY_1'] = 2
    df['default'] = np.tile([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], n//10)
    df.attrs['level'] = 'L1'
    return df


@pytest.fixture(autouse=True)
def forbid_real_data(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Real data forbidden in C9a tests')
    monkeypatch.setattr(p, '_full_data', forbidden)


def test_pay_mapping_and_unknown_codes(frame, rules):
    codes = list(map(int, rules.pay_categorical['mapping']))
    for col in rules.pay_categorical['columns']:
        frame.loc[frame.index[:len(codes)], col] = codes
    X, columns = fm.prepare_pay_categorical(frame)
    assert columns == rules.candidates[2]['features']
    for col in rules.pay_categorical['columns']:
        assert isinstance(X[col].dtype, pd.CategoricalDtype)
        assert X[col].astype(str).iloc[:len(codes)].tolist() == [rules.pay_categorical['mapping'][str(c)] for c in codes]
    for invalid in [10, -3, 1.5, np.nan]:
        bad = frame.copy()
        bad['PAY_2'] = bad['PAY_2'].astype(float)
        bad.loc[bad.index[0], 'PAY_2'] = invalid
        with pytest.raises(ValueError):
            fm.prepare_pay_categorical(bad)
    assert all(c not in columns for c in ['SEX', 'AGE', 'EDUCATION', 'MARRIAGE', 'default'])


def test_pay_encoder_fits_training_fold_only_and_is_serializable(frame):
    X, _ = fm.prepare_pay_categorical(frame)
    # Only validation customers have a mapped, legitimate category not seen at fit.
    frame.loc[frame.index[-2:], 'PAY_6'] = 9
    Z, _ = fm.prepare_pay_categorical(frame)
    model = fm.make_pay_categorical_model()
    assert not hasattr(model['pre'], 'transformers_')
    model.fit(X.iloc[:-2], frame['default'].iloc[:-2])
    encoder = model['pre'].named_transformers_['cat']
    assert '延遲三個月以上' not in encoder.categories_[-1]
    scores = model.predict_proba(Z.iloc[-2:])
    assert np.isfinite(scores).all()
    stream = io.BytesIO(); joblib.dump(model, stream); stream.seek(0)
    np.testing.assert_array_equal(joblib.load(stream).predict_proba(Z.iloc[-2:]), scores)
    assert model['clf'].fit_label_count_ == len(frame)-2


def test_feature_formulas_by_hand(frame, rules):
    row = frame.index[0]
    frame.loc[row, rules.pay_categorical['columns']] = [0, 0, -1, -2, 0, 2]
    frame.loc[row, [f'PAY_AMT{i}' for i in range(1, 7)]] = [10, 20, 30, 40, 50, 60]
    frame.loc[row, [f'BILL_AMT{i}' for i in range(1, 7)]] = [999, 15, 20, 35, 45, 50]
    B, bcols = fm.prepare_missed_features(frame, method='B')
    C, ccols = fm.prepare_missed_features(frame, method='C')
    D, dcols = fm.prepare_missed_features(frame, method='D')
    assert bcols == next(c for c in rules.candidates if c['name']=='09 B')['features']
    assert ccols == dcols == next(c for c in rules.candidates if c['name']=='09 C')['features']
    assert len(bcols)==23 and len(ccols)==26
    assert B.loc[row, ['N_CODE0','N_CODE_M1','N_CODE_M2','RUN_CODE0']].tolist()==[3,1,1,2]
    assert C.loc[row,'PAYAMT_CHANGE'] == pytest.approx((10-40)/41)
    assert C.loc[row,'N_UNDERPAY'] == 3  # 10<15, 20<20 false, 30<35, 40<45, 50<50 false
    assert C.loc[row,'PAYAMT_CV'] == pytest.approx(np.sqrt(350)/36)
    pd.testing.assert_frame_equal(C, D)
    assert rules.feature_formulas == fm._FORMULAS
    assert frame.loc[row, 'PAY_1']==0  # Input was not mutated.


def test_changed_formula_contract_fails_closed(frame, rules, monkeypatch):
    changed=deepcopy(rules)
    changed.feature_formulas['PAYAMT_CV']='std(..., ddof=0)'
    monkeypatch.setattr(p,'load_rules',lambda _:changed)
    with pytest.raises(IntegrityError,match='feature_formulas'):
        fm.prepare_missed_features(frame,method='C')


def test_frozen_settings_are_read_not_local_defaults(frame, rules, monkeypatch):
    changed=deepcopy(rules)
    changed.dev_base['params']['n_estimators']=7
    changed.pay_categorical['mapping']['9']='測試分類'
    monkeypatch.setattr(p,'load_rules',lambda _:changed)
    frame.loc[frame.index[0],'PAY_1']=9
    X,_=fm.prepare_pay_categorical(frame)
    assert X.iloc[0]['PAY_1']=='測試分類'
    assert fm.make_pay_categorical_model()['clf'].get_params()['n_estimators']==7


def test_d_clone_fit_route_weights_reload(frame, rules):
    X,_=fm.prepare_missed_features(frame,method='D')
    y=frame['default']
    # Imbalance differs between groups; each submodel must use its own labels.
    y=y.copy();y.iloc[:20]=[0]*5+[1]*15;y.iloc[20:]=[0]*12+[1]*8
    model=clone(fm.make_grouped_model()).fit(X,y)
    partition=fm.delay_partition(frame)
    scores=model.predict_proba(X)
    for label,submodel in model.models_.items():
        mask=partition==label
        np.testing.assert_array_equal(scores[mask],submodel.predict_proba(X.loc[mask]))
        clf=submodel['clf']
        assert clf.fit_label_count_==int(mask.sum())
        assert clf.get_params()['scale_pos_weight']==pytest.approx((y[mask]==0).sum()/(y[mask]==1).sum())
        for key,value in rules.dev_base['params'].items():
            assert clf.get_params()[key]==value
    stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
    restored=joblib.load(stream)
    np.testing.assert_array_equal(restored.predict_proba(X.iloc[::-1]),scores[::-1])
    np.testing.assert_array_equal(restored.predict(X),model.predict(X))
    # One group at prediction time is legal; no calibration or cross-group mixing.
    mask=partition=='無延遲'
    np.testing.assert_array_equal(restored.predict_proba(X.loc[mask]),scores[mask])


def test_d_shared_oof_folds(frame, rules):
    X,_=fm.prepare_missed_features(frame,method='D')
    seen=[]
    def factory():
        m=fm.make_grouped_model()
        original=m.fit
        def checked(Xtr,ytr):
            seen.append(Xtr.index.copy())
            original(Xtr,ytr)
            for sub in m.models_.values():
                assert sub['clf'].fit_label_count_ < len(Xtr)
            return m
        m.fit=checked
        return m
    score=p.oof_predict(factory,X,frame['default'],folds=rules.cv['n_splits'],seed=rules.cv['random_state'])
    assert np.isfinite(score).all() and score.index.equals(frame.index)
    assignment=score.attrs['fold_assignment']
    for fold,ids in enumerate(seen):
        assert set(ids)==set(frame.index[assignment!=fold])


def test_d_invalid_training_group_fails(frame):
    X,_=fm.prepare_missed_features(frame,method='D')
    y=frame['default'].copy();y.iloc[:20]=0
    with pytest.raises(ValueError,match='requires both classes'):
        fm.make_grouped_model().fit(X,y)
    with pytest.raises(ValueError,match='column order'):
        fm.make_grouped_model().fit(X[X.columns[::-1]],frame['default'])


def test_e_grid_costs_capacity_and_tie(frame, rules):
    score=pd.Series(.99,index=frame.index)
    frame['BILL_AMT1']=0  # All candidates have same final count/cost; high threshold wins.
    result=fm.search_no_delay_threshold(frame,score)
    table=result['candidates'];policy=result['policy']
    cfg=next(c for c in rules.candidates if c['name']=='09 E')
    assert table['threshold'].tolist()==[t for t in cfg['no_delay_grid'] if t<=result['t0']]
    assert table['threshold'].max()<=result['t0']
    assert policy.thresholds['有延遲']==result['t0']
    assert policy.thresholds['無延遲']==table['threshold'].max()
    assert table['selected'].sum()==1 and table['final_cost'].nunique()==1
    assert table['cap_triggered'].all() and (table['raw_count']==len(frame)).all()
    assert (table['final_count']==int(len(frame)*cfg['max_list_rate'])).all()
    assert (table['raw_cost']==len(frame)*rules.scenario.contact_cost).all()
    assert (table['final_cost']==table['final_count']*rules.scenario.contact_cost).all()
    pred,_=p.apply_policy(score,policy,partition=result['partition'])
    assert frame.index[pred.astype(bool)].tolist()==frame.index[:int(pred.sum())].tolist()


def test_e_only_below_t0_and_best_cost(frame, rules):
    score=pd.Series(np.tile([.15,.35,.55,.75,.99],8),index=frame.index)
    result=fm.search_no_delay_threshold(frame,score)
    table=result['candidates']
    chosen=table.loc[table['selected']].iloc[0]
    expected=table.sort_values(['final_cost','threshold'],ascending=[True,False]).iloc[0]
    assert chosen['threshold']==expected['threshold']
    assert chosen['final_cost']==expected['final_cost']
    for row in table.itertuples():
        assert row.threshold<=result['t0']
        policy=p.Policy.group('有無延遲',{'有延遲':result['t0'],'無延遲':row.threshold},rules.scenario.max_list_rate)
        pred,info=p.apply_policy(score,policy,partition=result['partition'])
        cost=p.evaluate(frame['default'],score=score,policy=policy,bill=p.bill_at(frame,0),
                        scenario=rules.scenario,partition=result['partition'])['overall']['總成本'].iloc[0]
        assert row.final_cost==cost and row.final_count==pred.sum()
        assert row.raw_count==info['pred_raw'].sum()


def test_f3b_target_raw_recall_ties_and_capacity(frame, rules):
    y=frame['default'];score=pd.Series(.99,index=frame.index);bill=p.bill_at(frame,0)
    partition=pd.Series(np.tile(['女性','男性'],20),index=frame.index)
    result=fm.search_group_thresholds(y,score,bill,partition,partition_name='性別')
    policy=result['policy'];table=result['candidates']
    assert result['target_recall']==1  # Raw single list includes all; capped recall is smaller.
    for label in partition.unique():
        assert policy.thresholds[label]==max(rules.group_threshold['grid'])
        per=table[table['group']==label]
        assert len(per)==len(rules.group_threshold['grid']) and per['selected'].sum()==1
        assert (per['total_count']==int((partition==label).sum())).all()
        assert (per['default_count']==int(((partition==label) & (y==1)).sum())).all()
        assert (per['nondefault_count']==per['total_count']-per['default_count']).all()
    pred,info=p.apply_policy(score,policy,partition=partition)
    assert pred.sum()==int(len(frame)*rules.group_threshold['max_list_rate'])
    assert info['pred_raw'].sum()==len(frame) and info['名單上限是否觸發']


def test_f3b_rounding_is_from_freeze(frame,rules,monkeypatch):
    changed=deepcopy(rules)
    changed.group_threshold['gap_round_decimals']=0
    monkeypatch.setattr(p,'load_rules',lambda _:changed)
    y=frame['default'];score=pd.Series(np.tile([.2,.6,.9,.7],10),index=frame.index)
    partition=pd.Series(np.tile(['女性','男性'],20),index=frame.index)
    result=fm.search_group_thresholds(y,score,p.bill_at(frame,0),partition,partition_name='性別')
    for label,per in result['candidates'].groupby('group'):
        assert per['rounded_gap'].tolist()==[round(v,0) for v in per['gap']]
        expected=per.sort_values(['rounded_gap','threshold'],ascending=[True,False]).iloc[0]
        assert result['policy'].thresholds[label]==expected['threshold']


@pytest.mark.parametrize('bad',['missing','reorder','multi','empty_positive','unknown_name','L2'])
def test_f3b_rejects_invalid_partition_or_data(frame,bad):
    y=frame['default'];score=pd.Series(.5,index=frame.index);bill=p.bill_at(frame,0)
    partition=pd.Series('全體',index=frame.index);name='性別'
    if bad=='missing':partition.iloc[0]=None
    elif bad=='reorder':partition=partition.iloc[::-1]
    elif bad=='multi':partition=pd.DataFrame({'a':partition,'b':partition})
    elif bad=='empty_positive':partition.iloc[np.flatnonzero(y.to_numpy()==0)]='無違約群'
    elif bad=='unknown_name':name='任意分組'
    elif bad=='L2':score.attrs['level']='L2'
    with pytest.raises((ValueError,PermissionError)):
        fm.search_group_thresholds(y,score,bill,partition,partition_name=name)


def test_zero_payments_and_longest_run(frame):
    row=frame.index[0]
    frame.loc[row,[f'PAY_AMT{i}' for i in range(1,7)]]=0
    X,_=fm.prepare_missed_features(frame,method='C')
    assert X.loc[row,'RUN_CODE0']==6
    assert X.loc[row,'PAYAMT_CHANGE']==0 and X.loc[row,'PAYAMT_CV']==0
    assert X.loc[row,'N_UNDERPAY']==5


def test_search_settings_grid_capacity_and_rounding_are_loaded(frame,rules,monkeypatch):
    changed=deepcopy(rules)
    changed.group_threshold.update(grid=[.2,.4],max_list_rate=.1)
    cfg=next(c for c in changed.candidates if c['name']=='09 E')
    cfg.update(no_delay_grid=[.2,.4],max_list_rate=.1)
    monkeypatch.setattr(p,'load_rules',lambda _:changed)
    score=pd.Series(.99,index=frame.index)
    e=fm.search_no_delay_threshold(frame,score)
    assert e['candidates']['threshold'].tolist()==[.2,.4]
    assert e['policy'].max_list_rate==.1
    partition=pd.Series('全體',index=frame.index)
    f=fm.search_group_thresholds(frame['default'],score,p.bill_at(frame,0),partition,partition_name='性別')
    assert f['candidates']['threshold'].tolist()==[.2,.4]
    assert f['policy'].max_list_rate==.1


@pytest.mark.parametrize('bad',['alignment','L1-H','no_defaults','missing_group','nan_score','negative_bill'])
def test_e_invalid_inputs(frame,bad):
    score=pd.Series(.5,index=frame.index)
    if bad=='alignment':score=score.iloc[::-1]
    elif bad=='L1-H':frame.attrs['level']='L1-H'
    elif bad=='no_defaults':frame['default']=0
    elif bad=='missing_group':frame['PAY_1']=0
    elif bad=='nan_score':score.iloc[0]=np.nan
    elif bad=='negative_bill':
        # Bills enter through bill_at and must be clipped before cost calculation.
        frame['BILL_AMT1']=-10
        assert np.isfinite(fm.search_no_delay_threshold(frame,score)['candidates']['final_cost']).all()
        return
    with pytest.raises((ValueError,PermissionError)):
        fm.search_no_delay_threshold(frame,score)
