"""Build local M1/M1-NS baselines after the leader publishes G2a rules.

    python experiments/00_build_dev_base.py --rules-freeze-id <approved-id>

Only protocol.dev_data/dev_holdout provide data. No tuning or test evaluation.
The frozen JSON must contain dev_base={params: <XGBoost parameters>, seed: 42,
folds: 5} and the C2 scenario. Missing rules never fall back to config/Markdown.
Each invocation creates new model/run identities; existing artifacts stay intact.
Models, OOF, folds and the L1-H handoff remain local and must not be committed.
"""
import argparse
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import protocol as p
from src.export import export_run
from src.registry import IntegrityError


def build_dev_base(rules_freeze_id):
    rules = p.load_rules(rules_freeze_id)  # Must precede all data access/writes.
    settings = rules.dev_base
    if settings['seed'] != 42 or settings['folds'] != 5:
        raise ValueError('Development baselines require seed 42 and five OOF folds')
    params = settings['params']
    required = {'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight',
                'subsample', 'colsample_bytree', 'gamma', 'reg_lambda'}
    if not isinstance(params, dict) or not required <= params.keys():
        raise ValueError('dev_base.params must explicitly supply the baseline XGBoost parameters')
    factory = lambda: p.make_model('XGBoost', params=params)
    factory()  # Validate parameters before loading data or creating any run.
    train = p.dev_data()
    dev_train, holdout = p.dev_holdout()
    if (train.attrs.get('level') != 'L1' or len(train) != 24000
            or any(df.attrs.get('level') != 'L1-H' for df in (dev_train, holdout))
            or len(dev_train) != 19200 or len(holdout) != 4800
            or set(dev_train.index) & set(holdout.index)
            or set(dev_train.index) | set(holdout.index) != set(train.index)):
        raise IntegrityError('Expected L1 24,000 and disjoint L1-H 19,200/4,800 customers')
    batch = uuid4().hex[:12]
    results = []
    for role, sensitive, level, frame, evaluation in [
            ('M1', True, 'L1', train, None),
            ('M1-NS', False, 'L1', train, None),
            ('M1-NS-dev19200', False, 'L1-H', dev_train, holdout)]:
        model_id = f'xgb-{role.lower()}-{batch}'
        run = p.start_run(level, f'Local development baseline {role}',
                          model_id=model_id, rules_freeze_id=rules_freeze_id)
        try:
            X, cols = p.prepare(frame, lag=0, sensitive=sensitive, engineered=False)
            y = frame['default']
            oof = p.oof_predict(factory, X, y, folds=settings['folds'], seed=settings['seed'])
            model = factory().fit(X, y)
            record = run.attach_model(model, X, y, oof, seed=settings['seed'],
                                      prediction_ids=None if evaluation is None else evaluation.index)
            policy = run.record_policy(p.choose_threshold(y, oof, p.bill_at(frame, 0),
                                                          rules.scenario)['政策物件'])
            run.update(baseline_role=role, build_batch_id=batch,
                       configuration={'name': 'XGBoost', 'params': params, 'lag': 0,
                                      'sensitive': sensitive, 'engineered': False})
            row = dict(baseline_role=role, level=level, model_id=model_id,
                       policy_id=policy.policy_id, run_id=run.run_id,
                       model_record=record['model_record'], model_hash=record['model_hash'],
                       features_hash=record['features_hash'], oof_sha256=record['oof_sha256'],
                       folds_sha256=record['folds_sha256'], training_count=len(X),
                       feature_count=len(cols), rules_freeze_id=rules_freeze_id,
                       predictions_path='' if evaluation is None else
                           f'outputs/runs/{run.run_id}/predictions.csv')
            run.save_csv('dev_base', pd.DataFrame([row]))
            if evaluation is None:
                run.finish()
            else:
                export_run(run, evaluation)  # Reload verifies every one of 4,800 scores.
            results.append(row)
            print(f"{role}: level={level} model_id={model_id} policy_id={policy.policy_id} "
                  f"run_id={run.run_id} model_record={record['model_record']}", flush=True)
        except Exception as exc:
            if run.meta['status'] == 'running':
                run.fail(f'{type(exc).__name__}: {exc}')
            raise
    print('L1-H 僅供功能開發；不得依其指標調整設定或當作正式效能。', flush=True)
    return pd.DataFrame(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules-freeze-id', required=True)
    args = parser.parse_args(argv)
    return build_dev_base(args.rules_freeze_id)


if __name__ == '__main__':
    main()
