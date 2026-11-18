"""Shared G2a methods; no data loading, training or rule reads on import."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from . import protocol as p
from .registry import IntegrityError, content_hash

RULES_FREEZE_ID = 'g2a-20261116'
# Formula contracts, not tunable settings. Never eval strings from a JSON file.
_FORMULAS = {
    'N_CODE0': 'sum(PAY_m == 0 for m=1..6)',
    'N_CODE_M1': 'sum(PAY_m == -1 for m=1..6)',
    'N_CODE_M2': 'sum(PAY_m == -2 for m=1..6)',
    'RUN_CODE0': 'longest consecutive run of PAY_m == 0 in PAY_1..PAY_6 order',
    'PAYAMT_CHANGE': '(PAY_AMT1 - mean(PAY_AMT2..PAY_AMT6)) / (mean(PAY_AMT2..PAY_AMT6) + 1)',
    'N_UNDERPAY': 'sum(PAY_AMTm < BILL_AMT(m+1) for m=1..5)',
    'PAYAMT_CV': 'std(PAY_AMT1..PAY_AMT6, ddof=1) / (mean(PAY_AMT1..PAY_AMT6) + 1)',
}


def _rules(rules_freeze_id):
    rules = p.load_rules(rules_freeze_id)
    if rules.feature_formulas != _FORMULAS:
        raise IntegrityError('Unsupported frozen feature_formulas; implementation review required')
    if (rules.pay_categorical['target_encoding']
            or rules.group_threshold['tie_break'] != 'higher_threshold'
            or rules.group_threshold['target_definition'] != 'single_policy_oof_recall'
            or _candidate(rules, '09 E')['tie_break'] != 'higher_no_delay_threshold'):
        raise IntegrityError('Unsupported frozen method selection/encoding rule')
    return rules


def _candidate(rules, name):
    return next(c for c in rules.candidates if c['name'] == name)


def _base(df, rules):
    cfg = _candidate(rules, 'M1-NS')
    X, _ = p.prepare(df, lag=cfg['lag'], sensitive=cfg['sensitive'], engineered=cfg['engineered'])
    if list(X) != cfg['features']:
        raise IntegrityError('M1-NS features differ from frozen list')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Features must be finite and non-null')
    return X


def prepare_pay_categorical(df, *, rules_freeze_id=RULES_FREEZE_ID):
    """05: deterministic fixed mapping only; One-Hot is learned in the pipeline."""
    rules = _rules(rules_freeze_id)
    X = _base(df, rules)
    mapping = {int(k): v for k, v in rules.pay_categorical['mapping'].items()}
    for col in rules.pay_categorical['columns']:
        if not X[col].isin(mapping).all():
            raise ValueError(f'{col}: PAY code outside frozen mapping')
        X[col] = X[col].map(mapping).astype('category')
    return X, list(X)


def make_pay_categorical_model(*, rules_freeze_id=RULES_FREEZE_ID):
    """05 pipeline with fold-fitted OneHotEncoder and per-fit XGB weights."""
    rules = _rules(rules_freeze_id)
    pay = rules.pay_categorical['columns']
    numeric = [c for c in _candidate(rules, '05')['features'] if c not in pay]
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), pay),
                             ('num', StandardScaler(), numeric)])
    clf = p.make_model('XGBoost', params=rules.dev_base['params']).named_steps['clf']
    return Pipeline([('pre', pre), ('clf', clf)])


def prepare_missed_features(df, *, method, rules_freeze_id=RULES_FREEZE_ID):
    """09 B/C/D features in the exact frozen input order; no learned transform."""
    if method not in {'B', 'C', 'D'}:
        raise ValueError('method must be B, C or D')
    rules = _rules(rules_freeze_id)
    X = _base(df, rules)
    pay = X[rules.pay_categorical['columns']]
    # Constants below are part of the approved formulas, not experiment settings.
    for name, code in [('N_CODE0', 0), ('N_CODE_M1', -1), ('N_CODE_M2', -2)]:
        X[name] = (pay == code).sum(axis=1)
    current = pd.Series(0, index=X.index)
    longest = current.copy()
    for col in pay:
        current = (current + 1).where(pay[col] == 0, 0)
        longest = pd.concat([longest, current], axis=1).max(axis=1)
    X['RUN_CODE0'] = longest
    if method in {'C', 'D'}:
        amounts = [c for c in _candidate(rules, 'M1-NS')['features'] if c.startswith('PAY_AMT')]
        bills = [c for c in _candidate(rules, 'M1-NS')['features'] if c.startswith('BILL_AMT')]
        past = X[amounts[1:]].mean(axis=1)
        X['PAYAMT_CHANGE'] = (X[amounts[0]] - past) / (past + 1)
        X['N_UNDERPAY'] = sum((X[a] < X[b]).astype(int) for a, b in zip(amounts[:-1], bills[1:]))
        X['PAYAMT_CV'] = X[amounts].std(axis=1, ddof=1) / (X[amounts].mean(axis=1) + 1)
    columns = _candidate(rules, '09 ' + method)['features']
    X = X.loc[:, columns].copy()
    X.attrs.update(df.attrs)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Frozen formula produced non-finite features')
    return X, list(X)


def delay_partition(df, *, rules_freeze_id=RULES_FREEZE_ID):
    """One label per customer: any positive PAY code means delayed."""
    rules = _rules(rules_freeze_id)
    if _candidate(rules, '09 D')['group_definition'] != 'any(PAY_1..PAY_6 > 0)':
        raise IntegrityError('Unsupported frozen delay definition')
    pay = df[rules.pay_categorical['columns']]
    if not np.isfinite(pay.to_numpy(dtype=float)).all():
        raise ValueError('PAY values must be finite and non-null')
    result = pd.Series(np.where((pay > 0).any(axis=1), '有延遲', '無延遲'), index=df.index, name='有無延遲')
    result.attrs.update(df.attrs)
    return result


class FrozenGroupedClassifier(ClassifierMixin, BaseEstimator):
    """D: oof_predict splits first; fit trains both groups inside that fold only."""
    def __init__(self, *, rules_freeze_id=RULES_FREEZE_ID):
        self.rules_freeze_id = rules_freeze_id

    def fit(self, X, y):
        rules = _rules(self.rules_freeze_id)
        if list(X) != _candidate(rules, '09 D')['features']:
            raise ValueError('D requires prepare_missed_features(method="D") column order')
        p._aligned(y)
        p._labels(y)
        if not X.index.equals(y.index):
            raise ValueError('X/y customer IDs differ')
        partition = delay_partition(X, rules_freeze_id=self.rules_freeze_id)
        models = {}
        for label in ('無延遲', '有延遲'):
            mask = partition == label
            if y.loc[mask].nunique() != 2:
                raise ValueError(f'D training group {label} requires both classes')
            models[label] = p.make_model('XGBoost', params=rules.dev_base['params']).fit(X.loc[mask], y.loc[mask])
        self.models_ = models
        self.columns_ = list(X)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'models_')
        if list(X) != self.columns_:
            raise ValueError('D prediction columns differ from fitted columns')
        partition = delay_partition(X, rules_freeze_id=self.rules_freeze_id)
        result = np.empty((len(X), len(self.classes_)), dtype=float)
        for label, model in self.models_.items():
            mask = partition == label
            if mask.any():
                result[mask.to_numpy()] = model.predict_proba(X.loc[mask])
        return result

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_grouped_model(*, rules_freeze_id=RULES_FREEZE_ID):
    """Return an unfitted, cloneable D estimator with frozen XGB parameters."""
    return FrozenGroupedClassifier(rules_freeze_id=rules_freeze_id)


def _selection_inputs(y, score, bill, partition):
    p._aligned(y, score, bill, partition)
    p._labels(y)
    for obj in (y, score, bill, partition):
        if obj.attrs.get('level') not in (None, 'L1', 'L3'):
            raise PermissionError('Threshold selection accepts training OOF only (L1 or L3)')
    if not len(y) or not (y == 1).any():
        raise ValueError('Threshold selection requires default observations')
    if (not partition.map(pd.api.types.is_scalar).all() or partition.isna().any()
            or partition.astype(str).nunique() != partition.nunique()):
        raise ValueError('Partition must be complete and mutually exclusive')


def _evidence(y, score, rules_freeze_id, base_policy):
    return dict(rules_freeze_id=rules_freeze_id, training_ids_hash=content_hash(y.index.tolist()),
                oof_scores_hash=content_hash(score.tolist()), base_policy_hash=base_policy.policy_hash)


def search_no_delay_threshold(df, oof_score, *, rules_freeze_id=RULES_FREEZE_ID):
    """E: return policy, all candidate raw/final costs/counts, t0 and partition."""
    rules = _rules(rules_freeze_id)
    cfg = _candidate(rules, '09 E')
    y, bill = df['default'], p.bill_at(df, _candidate(rules, 'M1-NS')['lag'])
    partition = delay_partition(df, rules_freeze_id=rules_freeze_id)
    _selection_inputs(y, oof_score, bill, partition)
    if set(partition) != {'有延遲', '無延遲'}:
        raise ValueError('E requires both delay groups')
    base = p.choose_threshold(y, oof_score, bill, rules.scenario)['政策物件']
    rows, policies = [], []
    for threshold in cfg['no_delay_grid']:
        if threshold > base.threshold:
            continue
        policy = replace(p.Policy.group(partition.name, {'有延遲': base.threshold, '無延遲': threshold}, cfg['max_list_rate']),
                         selection='group-cost', scenario=asdict(rules.scenario),
                         evidence=_evidence(y, oof_score, rules_freeze_id, base))
        final, info = p.apply_policy(oof_score, policy, partition=partition)
        raw_cost = p.evaluate(y, pred=info['pred_raw'], policy=policy, partition=partition,
                              bill=bill, scenario=rules.scenario)['overall']['總成本'].iloc[0]
        final_cost = p.evaluate(y, score=oof_score, pred=final, policy=policy, partition=partition,
                                bill=bill, scenario=rules.scenario)['overall']['總成本'].iloc[0]
        rows.append(dict(threshold=threshold, t0=base.threshold, raw_cost=raw_cost, final_cost=final_cost,
                         raw_count=info['原始名單人數'], final_count=int(final.sum()),
                         cap_triggered=info['名單上限是否觸發'], policy_id=policy.policy_id))
        policies.append(policy)
    if not rows:
        raise ValueError('Frozen E grid contains no threshold <= t0')
    best = min(range(len(rows)), key=lambda i: (rows[i]['final_cost'], -rows[i]['threshold']))
    table = pd.DataFrame(rows)
    table['selected'] = np.arange(len(table)) == best
    return dict(policy=policies[best], candidates=table, t0=base.threshold, partition=partition, base_policy=base)


def search_group_thresholds(y, oof_score, bill, partition, *, partition_name, rules_freeze_id=RULES_FREEZE_ID):
    """F3b: one complete partition; raw single-policy recall is the target."""
    rules = _rules(rules_freeze_id)
    cfg = rules.group_threshold
    if partition_name not in cfg['partitions']:
        raise ValueError('partition_name is not a frozen F3b partition')
    _selection_inputs(y, oof_score, bill, partition)
    base = p.choose_threshold(y, oof_score, bill, rules.scenario)['政策物件']
    target = float((oof_score[y == 1] >= base.threshold).mean())
    thresholds, rows = {}, []
    labels = partition.astype(str)
    for group in sorted(labels.unique()):
        members = labels == group
        positive = (labels == group) & (y == 1)
        if not positive.any():
            raise ValueError(f'F3b group {group} has no default observations')
        candidates = []
        for threshold in cfg['grid']:
            recall = float((oof_score[positive] >= threshold).mean())
            gap = abs(recall - target)
            candidates.append(dict(group=group, threshold=threshold, recall=recall, target_recall=target,
                                   gap=gap, rounded_gap=round(gap, cfg['gap_round_decimals']),
                                   total_count=int(members.sum()), default_count=int(positive.sum()),
                                   nondefault_count=int((members & (y == 0)).sum())))
        best = min(range(len(candidates)), key=lambda i: (candidates[i]['rounded_gap'], -candidates[i]['threshold']))
        thresholds[group] = candidates[best]['threshold']
        for i, row in enumerate(candidates):
            rows.append(dict(row, selected=i == best))
    policy = replace(p.Policy.group(partition_name, thresholds, cfg['max_list_rate']), scenario=asdict(rules.scenario),
                     evidence=_evidence(y, oof_score, rules_freeze_id, base))
    return dict(policy=policy, candidates=pd.DataFrame(rows), target_recall=target, t0=base.threshold, base_policy=base)
