"""C9b-1: build L1 evidence, then publish an explicitly approved model freeze.

No data is loaded on import. Only prepare() calls dev_data(); freeze() only
validates saved artifacts. Neither entry point obtains test data.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import re
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import frozen_methods as fm, protocol as p
from .calibration import crossfit_calibrate
from .export import top_factors
from .registry import IntegrityError, content_hash, file_hash, identifier, json_value, read_json, write_json
from .training_time import execution_environment, timed_call

APPROVER = '蘇韻蓉'


def configurations(rules):
    """Approved model definitions, never an unconstrained parameter override."""
    params = dict(rules.dev_base['params'])
    configs = {
        'M1': dict(name='XGBoost', params=params, lag=0, sensitive=True, engineered=False, method='raw'),
        'M1-NS': dict(name='XGBoost', params=params, lag=0, sensitive=False, engineered=False, method='raw'),
        'LR0': dict(name='邏輯迴歸', params={'max_iter': 3000, 'class_weight': 'balanced'}, lag=0, sensitive=False, engineered=False, method='raw'),
        'LR1': dict(name='邏輯迴歸', params={'max_iter': 3000, 'class_weight': 'balanced'}, lag=0, sensitive=False, engineered=True, method='raw'),
        'DT0': dict(name='決策樹', params={'max_depth': 4, 'class_weight': 'balanced', 'random_state': rules.seed}, lag=0, sensitive=False, engineered=False, method='raw'),
    }
    for lag in range(1, 5):
        configs[f'lag{lag}'] = dict(configs['M1-NS'], lag=lag)
    for method in rules.g2b['eligible_candidates']:
        configs[method] = dict(configs['M1-NS'], method=method)
    return configs


def features(frame, config, rules_freeze_id):
    method = config['method']
    if method == '05':
        return fm.prepare_pay_categorical(frame, rules_freeze_id=rules_freeze_id)
    if method in ('09 B', '09 C', '09 D'):
        return fm.prepare_missed_features(frame, method=method[-1], rules_freeze_id=rules_freeze_id)
    return p.prepare(frame, lag=config['lag'], sensitive=config['sensitive'], engineered=config['engineered'])


def model_factory(config, rules_freeze_id):
    if config['method'] == '05':
        return fm.make_pay_categorical_model(rules_freeze_id=rules_freeze_id)
    if config['method'] == '09 D':
        return fm.make_grouped_model(rules_freeze_id=rules_freeze_id)
    return p.make_model(config['name'], params=config['params'])


def _model_id(name, config, token, seed):
    algorithm={'XGBoost':'xgb', '邏輯迴歸':'lr', '決策樹':'dt'}[config['name']]
    parameters='tuned' if algorithm=='xgb' else 'baseline'
    feature_key=re.sub(r'[^a-z0-9-]','-',name.lower())
    return identifier(f'{algorithm}-{parameters}-{feature_key}-s{seed}train-{token}')


def entry_specs(candidates):
    """One entry per comparison/model/policy; platform is added only at approval."""
    specs = []
    def standard(prefix, comparison, model, version='raw'):
        for policy in ('s', 'n', 'c'):
            specs.append(dict(entry_id=f'{prefix}-{policy}', role='research', comparison_id=comparison,
                              model_key=model, score_version=version, policy_key=policy, evaluation_kind='policy'))
    for suffix, model in [('m1', 'M1'), ('ns', 'M1-NS')]:
        standard('rq1-'+suffix, 'rq1-sensitive', model)
    for suffix, model in [('lr0','LR0'), ('lr1','LR1'), ('dt0','DT0'), ('ns','M1-NS')]:
        standard('rq2-'+suffix, 'rq2-baselines', model)
    for lag in range(5):
        specs.append(dict(entry_id=f'rq3-lag{lag}', role='research', comparison_id='rq3-lead-time',
                          model_key='M1-NS' if lag == 0 else f'lag{lag}', score_version='raw',
                          policy_key=None, evaluation_kind='score_only'))
    for suffix, partition in [('sex','性別'), ('age','年齡層'), ('sex-age','性別×年齡層')]:
        specs.append(dict(entry_id='f3b-'+suffix, role='research', comparison_id='rq1-group-threshold',
                          model_key='M1-NS', score_version='raw', policy_key=partition, evaluation_kind='policy'))
    specs.append(dict(specs[-1], entry_id='f3b-single-c', policy_key='c'))
    standard('f6-a', 'rq2-f6', 'M1-NS')
    for candidate in candidates:
        suffix = '05' if candidate == '05' else candidate[-1].lower()
        standard('f6-'+suffix, 'rq2-f6', candidate)
    specs.append(dict(entry_id='f6-e', role='research', comparison_id='rq2-f6', model_key='M1-NS',
                      score_version='raw', policy_key='e', evaluation_kind='policy'))
    standard('f7-iso', 'f7-calibration', 'M1-NS', 'calibrated')
    standard('f7-raw', 'f7-calibration', 'M1-NS')
    return specs


def _timing(record, fit_seconds, oof_seconds, model, environment):
    return dict(training_timing_id=record['model_id'], training_count=record['training_count'],
                fit_seconds=fit_seconds, oof_seconds=oof_seconds, policy_seconds=0.,
                calibrator_fit_seconds=0., environment=environment,
                model_threads=(model.named_steps['clf'].get_params().get('n_jobs') if hasattr(model, 'named_steps') else
                               {k: v.named_steps['clf'].get_params().get('n_jobs') for k,v in model.models_.items()}),
                submodel_fit_seconds=getattr(model, 'fit_seconds_by_group_', {}),
                submodel_total_fit_seconds=sum(getattr(model, 'fit_seconds_by_group_', {}).values()))


def _check_timing(timing):
    for key in ('fit_seconds','oof_seconds','policy_seconds','calibrator_fit_seconds'):
        if not isinstance(timing[key], (int,float)) or not np.isfinite(timing[key]) or timing[key] < 0:
            raise IntegrityError('Invalid timing field: '+key)
    if (any(not np.isfinite(v) or v < 0 for v in timing.get('submodel_fit_seconds',{}).values())
            or timing.get('submodel_total_fit_seconds',0) != sum(timing.get('submodel_fit_seconds',{}).values())):
        raise IntegrityError('Invalid submodel timing')
    if not timing['environment']['packages'] or not timing['environment']['logical_cpus']:
        raise IntegrityError('Timing environment is missing')


def _save_calibrator(registry, record, y, oof, rules):
    (calibrated, provenance), crossfit_seconds = timed_call(crossfit_calibrate, y, oof,
        folds=rules.cv['n_splits'], seed=rules.cv['random_state'])
    calibrator, seconds = timed_call(IsotonicRegression(out_of_bounds='clip').fit, oof, y)
    folder = registry.outputs/'calibrators'/uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    joblib.dump(calibrator, folder/'calibrator.joblib', compress=3)
    pd.DataFrame(dict(customer_id=y.index, oof_score=calibrated.to_numpy(), y=y.to_numpy())).to_csv(folder/'crossfit_oof.csv', index=False)
    provenance.to_csv(folder/'crossfit_folds.csv', index=False)
    meta = dict(calibrator_path=registry.relative(folder/'calibrator.joblib'),
                calibrator_record=registry.relative(folder/'calibration.json'),
                calibrator_hash=file_hash(folder/'calibrator.joblib'),
                crossfit_oof_sha256=file_hash(folder/'crossfit_oof.csv'),
                crossfit_folds_sha256=file_hash(folder/'crossfit_folds.csv'),
                model_hash=record['model_hash'], source_oof_sha256=record['oof_sha256'],
                training_ids_hash=record['training_ids_hash'], seed=rules.cv['random_state'],
                folds=rules.cv['n_splits'], calibrator_fit_seconds=seconds, crossfit_seconds=crossfit_seconds)
    write_json(folder/'calibration.json', meta)
    return meta, calibrated


def load_calibrator(registry, meta, record):
    """Validate both final calibrator and cross-fitted selection evidence."""
    stored = read_json(registry.path(meta['calibrator_record']))
    if stored != meta:
        raise IntegrityError('Calibrator metadata mismatch')
    path = registry.path(meta['calibrator_path'])
    for name, key in [('calibrator.joblib','calibrator_hash'), ('crossfit_oof.csv','crossfit_oof_sha256'),
                      ('crossfit_folds.csv','crossfit_folds_sha256')]:
        if file_hash(path.parent/name) != meta[key]:
            raise IntegrityError('Calibrator artifact hash mismatch: '+name)
    for field, expected in [('model_hash', record['model_hash']), ('source_oof_sha256',record['oof_sha256']),
                            ('training_ids_hash',record['training_ids_hash'])]:
        if meta[field] != expected:
            raise IntegrityError('Calibrator source mismatch: '+field)
    cross = pd.read_csv(path.parent/'crossfit_oof.csv', converters={'oof_score': float})
    original = pd.read_csv(registry.path(record['model_path']).parent/'oof.csv', converters={'oof_score': float})
    if (not np.array_equal(cross['customer_id'], original['customer_id'])
            or not np.array_equal(cross['y'],original['y']) or not cross['oof_score'].between(0,1).all()):
        raise IntegrityError('Calibrated OOF IDs/labels/scores mismatch')
    return joblib.load(path), cross


def _source_run(bundle, policy, *, calibrated=None, policy_seconds=0.):
    registry, rules_id = bundle['run'].registry, bundle['run'].meta['rules_freeze_id']
    run = p.start_run('L1', 'G2b policy source', model_id=bundle['record']['model_id'], rules_freeze_id=rules_id)
    try:
        run.bind_model(bundle['record'])
        timing = dict(bundle['timing'], policy_seconds=policy_seconds)
        if calibrated:
            run.update(calibrator=calibrated, calibrator_hash=calibrated['calibrator_hash'],
                       calibrator_record=calibrated['calibrator_record'])
            timing.update(calibrator_fit_seconds=calibrated['calibrator_fit_seconds'],
                          crossfit_seconds=calibrated['crossfit_seconds'])
            policy = replace(policy, evidence={**policy.evidence, 'calibrator_hash':calibrated['calibrator_hash'],
                                               'crossfit_oof_sha256':calibrated['crossfit_oof_sha256']})
        policy = run.record_policy(policy)
        run.update(training_timing=timing, configuration=bundle['configuration'],
                   model_source_run_id=bundle['run'].run_id, evaluation_kind='policy')
        run.save_csv('training_timing', pd.DataFrame([json_value(timing)]))
        run.finish()
        return run.run_id, policy
    except Exception as exc:
        if run.meta['status'] == 'running':
            run.fail(f'{type(exc).__name__}: {exc}')
        raise


def engineering_check(bundle, frame, rules_id):
    """Real reload, deterministic feature transformation, and platform factors on L1."""
    checks = dict(reload_ok=False, features_ok=False, shap_ok=False, engineering_error='')
    try:
        model, _ = bundle['run'].registry.load_model(bundle['record'])
        transformed, _ = features(frame, bundle['configuration'], rules_id)
        pd.testing.assert_frame_equal(transformed, bundle['X'])
        checks['features_ok'] = True
        np.testing.assert_array_equal(model.predict_proba(transformed), bundle['model'].predict_proba(bundle['X']))
        checks['reload_ok'] = True
        # Cover both routing branches for D without describing all L1 customers.
        labels = fm.delay_partition(frame, rules_freeze_id=rules_id)
        ids = [labels.index[labels==group][0] for group in labels.unique()]
        factors = top_factors(model, transformed.loc[ids])
        checks['shap_ok'] = len(factors)==len(ids) and all(len(row)==3 and all(isinstance(v,str) for v in row) for row in factors)
    except Exception as exc:
        checks['engineering_error'] = f'{type(exc).__name__}: {exc}'
    return checks


def rank_candidates(frame, rules):
    """Five frozen ranking criteria, no rounding and no platform selection."""
    table = frame.copy()
    table['d7_eligible'] = ((table.recall_gain >= rules.g2b['min_recall_gain'])
        & (table.auc_gain >= rules.g2b['min_auc_gain'])
        & (table.gender_eo_increase <= rules.g2b['max_gender_eo_increase'])
        & (table.age_eo_increase <= rules.g2b['max_age_eo_increase']))
    table['max_eo_increase'] = table[['gender_eo_increase','age_eo_increase']].max(axis=1)
    table['sum_eo_increase'] = table.gender_eo_increase + table.age_eo_increase
    order = rules.g2b['ranking'][-1]['order']
    table['_order'] = table.candidate.map({name:i for i,name in enumerate(order)})
    ranked = table[table.d7_eligible].sort_values(
        ['recall_gain','max_eo_increase','sum_eo_increase','input_feature_count_before_onehot','_order'],
        ascending=[False,True,True,True,True])
    table['rank'] = pd.Series({index:i+1 for i,index in enumerate(ranked.index)},dtype='Int64')
    table['engineering_ready'] = table[['reload_ok','features_ok','shap_ok']].all(axis=1)
    return table.drop(columns='_order')


def _candidates_report(bundles, frame, rules):
    from ._protocol_runs import _partition
    groups = {'性別':_partition(frame,'性別'), '年齡層':_partition(frame,'年齡層')}
    policy = p.Policy.top_n(rules.g2b['policy']['rate'])
    def measures(bundle):
        report = p.evaluate(frame['default'],score=bundle['oof'],policy=policy,groups=groups)
        overall = report['overall'].iloc[0]
        gaps = report['gaps'].set_index('分組方式')['EO 差距']
        return dict(recall=overall['Recall'],auc=overall['AUC'],gender=gaps['性別'],age=gaps['年齡層'])
    base = measures(bundles['M1-NS'])
    rows=[]
    for name in rules.g2b['eligible_candidates']:
        b=bundles[name];m=measures(b)
        row = dict(candidate=name, model_id=b['record']['model_id'], source_run_id=b['policies']['n'][0],
                   policy_id=b['policies']['n'][1].policy_id, baseline_run_id=bundles['M1-NS']['policies']['n'][0],
                   baseline_model_id=bundles['M1-NS']['record']['model_id'],
                   recall_gain=m['recall']-base['recall'],auc_gain=m['auc']-base['auc'],
                   gender_eo_increase=m['gender']-base['gender'],age_eo_increase=m['age']-base['age'],
                   input_feature_count_before_onehot=len(b['record']['columns']), **b['engineering'])
        rows.append(row)
    return rank_candidates(pd.DataFrame(rows),rules)


def _entry(spec, bundle, source):
    run_id, policy = source
    record=bundle['record']
    meta=read_json(bundle['run'].registry.runs/run_id/'run.json')
    entry = dict(spec, source_run_id=run_id, model_id=record['model_id'], model_hash=record['model_hash'],
                 params_hash=record['params_hash'], columns=record['columns'], features_hash=record['features_hash'],
                 configuration=bundle['configuration'], training_timing=meta['training_timing'],
                 model_source_run_id=bundle['run'].run_id, implementation_commit=bundle['run'].meta['git_commit'])
    if policy is not None:
        entry.update(policy_id=policy.policy_id, policy_hash=policy.policy_hash, protocol_policy=asdict(policy))
    return entry


def _manifest_path(registry, path, *, draft):
    result=Path(path).resolve()
    if (result.suffix != '.json' or result.is_relative_to(p.DATA_DIR.resolve())
            or result.is_relative_to((registry.outputs/'sealed').resolve())):
        raise ValueError('Manifest must be JSON outside data and sealed artifacts')
    if draft and (result.name != 'manifest.json' or not result.is_relative_to((registry.outputs/'drafts').resolve())):
        raise ValueError('Freeze accepts only a completed preparation draft in outputs/drafts')
    return result


def prepare(rules_freeze_id=fm.RULES_FREEZE_ID, *, manifest=None):
    """Build a fresh, measured L1 evidence set. Never publish model-*.json."""
    registry=p._registry();rules=p.load_rules(rules_freeze_id)
    if p.audit_sources():
        raise IntegrityError('G2b source audit must pass before preparation')
    if manifest is not None:
        request=read_json(_manifest_path(registry,manifest,draft=False))
        if request != {'rules_freeze_id':rules_freeze_id}:
            raise ValueError('Preparation request only declares rules_freeze_id, no design overrides')
    df=p.dev_data()
    if (df.attrs.get('level')!='L1' or len(df)!=rules.data_protocol['selection']['train_rows']
            or content_hash(df.index.tolist())!=rules.pay_categorical['l1_verification']['training_ids_hash']):
        raise IntegrityError('Expected frozen seed-42 L1 customers')
    data_hash=file_hash(p.DATA_DIR/'default_of_credit_card_clients.csv')
    if data_hash!=rules.pay_categorical['l1_verification']['data_sha256']:
        raise IntegrityError('Training data hash differs from frozen rules')
    configs=configurations(rules);bundles={};token=uuid4().hex[:12]
    environment=execution_environment()
    report_run=p.start_run('L1','G2b candidate recommendation only',
                           model_id=_model_id('M1-NS',configs['M1-NS'],token,rules.seed),rules_freeze_id=rules_freeze_id)
    try:
        for name, config in configs.items():
            model_id=_model_id(name,config,token,rules.seed)
            run=p.start_run('L1','G2b measured model source: '+name,model_id=model_id,rules_freeze_id=rules_freeze_id)
            try:
                X,_=features(df,config,rules_freeze_id)
                factory=lambda cfg=config:model_factory(cfg,rules_freeze_id)
                oof,oof_seconds=timed_call(p.oof_predict,factory,X,df['default'],folds=rules.cv['n_splits'],seed=rules.cv['random_state'])
                model,fit_seconds=timed_call(factory().fit,X,df['default'])
                record=run.attach_model(model,X,df['default'],oof,seed=rules.cv['random_state'])
                timing=_timing(record,fit_seconds,oof_seconds,model,environment)
                run.update(training_timing=timing,configuration=config,evaluation_kind='score_only')
                run.save_csv('training_timing',pd.DataFrame([json_value(timing)]));run.finish()
                bundles[name]=dict(run=run,record=record,timing=timing,configuration=config,model=model,X=X,oof=oof,policies={})
            except Exception as exc:
                if run.meta['status']=='running':run.fail(f'{type(exc).__name__}: {exc}')
                raise
        # Shared folds/IDs are checked from persisted evidence, not just seeds.
        base=bundles['M1-NS']
        if any(b['record']['folds_sha256']!=base['record']['folds_sha256'] for b in bundles.values()):
            raise IntegrityError('Candidate fold assignments differ')
        for name,b in bundles.items():
            if name.startswith('lag'):continue
            formal,seconds=timed_call(p.choose_threshold,df['default'],b['oof'],p.bill_at(df,0),rules.scenario)
            policies={'s':p.Policy.single(.5,None),'n':p.Policy.top_n(rules.g2b['policy']['rate']),'c':formal['政策物件']}
            for key, policy in policies.items():
                b['policies'][key]=_source_run(b,policy,policy_seconds=seconds if key=='c' else 0.)
        for partition_name in rules.group_threshold['partitions']:
            from ._protocol_runs import _partition
            result,seconds=timed_call(fm.search_group_thresholds,df['default'],base['oof'],p.bill_at(df,0),
                _partition(df,partition_name),partition_name=partition_name,rules_freeze_id=rules_freeze_id)
            base['policies'][partition_name]=_source_run(base,result['policy'],policy_seconds=seconds)
        result,seconds=timed_call(fm.search_no_delay_threshold,df,base['oof'],rules_freeze_id=rules_freeze_id)
        base['policies']['e']=_source_run(base,result['policy'],policy_seconds=seconds)
        calibration,calibrated=_save_calibrator(registry,base['record'],df['default'],base['oof'],rules)
        cal_policies={}
        formal,seconds=timed_call(p.choose_threshold,df['default'],calibrated,p.bill_at(df,0),rules.scenario)
        for key,policy in {'s':p.Policy.single(.5,None),'n':p.Policy.top_n(rules.g2b['policy']['rate']),'c':formal['政策物件']}.items():
            cal_policies[key]=_source_run(base,policy,calibrated=calibration,policy_seconds=seconds if key=='c' else 0.)
        entries=[]
        for spec in entry_specs(rules.g2b['eligible_candidates']):
            b=bundles[spec['model_key']]
            source=(b['run'].run_id,None) if spec['policy_key'] is None else (
                cal_policies[spec['policy_key']] if spec['score_version']=='calibrated' else b['policies'][spec['policy_key']])
            entry=_entry(spec,b,source)
            if spec['score_version']=='calibrated':
                entry.update(calibrator=calibration, calibrator_hash=calibration['calibrator_hash'],
                             calibrator_record=calibration['calibrator_record'])
            entries.append(entry)
        for name in rules.g2b['eligible_candidates']:
            bundles[name]['engineering']=engineering_check(bundles[name],df,rules_freeze_id)
        report=_candidates_report(bundles,df,rules)
        try:
            report_run.bind_model(base['record']);report_run.record_policy(p.Policy.top_n(rules.g2b['policy']['rate']))
            report_run.save_csv('g2b_candidates',report)
            draft=dict(schema_version=2,status='prepared',rules_freeze_id=rules_freeze_id,
                       rules_freeze_sha256=file_hash(registry.outputs/'frozen'/f'rules-{rules_freeze_id}.json'),
                       build_run_id=report_run.run_id, training_ids_hash=base['record']['training_ids_hash'],
                       data_sha256=data_hash,entries=entries,platform_model=None,
                       candidate_report=registry.relative(report_run.path/'g2b_candidates.csv'))
            # Keep the machine-readable report in private outputs as well as aggregate results.
            report.to_csv(report_run.path/'g2b_candidates.csv',index=False)
            draft['candidate_report_sha256']=file_hash(report_run.path/'g2b_candidates.csv')
            path=registry.outputs/'drafts'/report_run.run_id/'manifest.json'
            path.parent.mkdir(parents=True,exist_ok=False)
            write_json(path,draft)
            report_run.update(draft_path=registry.relative(path),draft_sha256=file_hash(path),training_timing=base['timing'])
            report_run.finish()
            return path
        except Exception as exc:
            if report_run.meta['status']=='running':report_run.fail(f'{type(exc).__name__}: {exc}')
            raise
    except Exception as exc:
        if report_run.meta['status']=='running':report_run.fail(f'{type(exc).__name__}: {exc}')
        raise


def validate_entry(registry, freeze, entry, source):
    """Version-2 entry contract, shared by freeze and normal registry loading."""
    required={'entry_id','role','comparison_id','source_run_id','model_key','score_version','evaluation_kind',
              'model_id','model_hash','params_hash','columns','configuration','training_timing','model_source_run_id','implementation_commit'}
    if not required<=entry.keys():raise IntegrityError('Missing frozen entry fields')
    record=source['model_record'];meta=source['run']
    rules=registry.load_rules(freeze['rules_freeze_id'])
    if record['training_count']!=rules['data_protocol']['selection']['train_rows']:
        raise IntegrityError('Frozen training count mismatch')
    if (record['training_ids_hash']!=rules['pay_categorical']['l1_verification']['training_ids_hash']
            or record['data_sha256']!=rules['pay_categorical']['l1_verification']['data_sha256']):
        raise IntegrityError('Frozen L1 data evidence mismatch')
    _check_timing(entry['training_timing'])
    if (entry['training_timing']['training_count']!=record['training_count']
            or entry['training_timing']['training_timing_id']!=record['model_id']):
        raise IntegrityError('Timing/model identity mismatch')
    if entry['configuration']!=configurations(SimpleNamespace(**rules))[entry['model_key']]:
        raise IntegrityError('Configuration differs from approved method')
    if entry['configuration']!=meta.get('configuration') or entry['training_timing']!=meta.get('training_timing'):
        raise IntegrityError('Measured source configuration/timing mismatch')
    model_source=registry.load_run(entry['model_source_run_id'])['run']
    if entry['implementation_commit']!=model_source['git_commit']:
        raise IntegrityError('Implementation commit mismatch')
    models = (source['model'].models_.values() if hasattr(source['model'],'models_') else [source['model']])
    for model in models:
        actual=model.named_steps['clf'].get_params()
        if any(content_hash(actual[key])!=content_hash(value) for key,value in entry['configuration']['params'].items()):
            raise IntegrityError('Fitted estimator parameters differ from configuration')
    if (model_source['model_hash']!=record['model_hash'] or model_source['level']!='L1'
            or model_source['configuration']!=entry['configuration']
            or model_source['training_timing']['fit_seconds']!=entry['training_timing']['fit_seconds']):
        raise IntegrityError('Measured model source mismatch')
    if entry['evaluation_kind']=='score_only':
        if source['threshold'] is not None or {'policy_id','policy_hash','protocol_policy'} & entry.keys():
            raise IntegrityError('Score-only entry must not invent a policy')
    elif entry['evaluation_kind']=='policy':
        if source['threshold'] is None:raise IntegrityError('Policy entry lacks threshold evidence')
        for key in ('policy_id','policy_hash'):
            if entry[key]!=meta[key]:raise IntegrityError('Policy source mismatch')
        if entry['protocol_policy']!=source['threshold']['protocol_policy']:
            raise IntegrityError('Frozen policy contents mismatch')
    else:raise IntegrityError('Unknown frozen evaluation kind')
    if entry['score_version']=='calibrated':
        if entry.get('calibrator')!=meta.get('calibrator'):
            raise IntegrityError('Frozen calibrator source mismatch')
        if (entry.get('calibrator_hash')!=entry['calibrator']['calibrator_hash']
                or entry.get('calibrator_record')!=entry['calibrator']['calibrator_record']
                or entry['calibrator_hash']!=meta.get('calibrator_hash')):
            raise IntegrityError('Calibrator identity mismatch')
        load_calibrator(registry,entry['calibrator'],record)
    elif entry['score_version']!='raw' or {'calibrator','calibrator_hash','calibrator_record'} & entry.keys() or 'calibrator' in meta:
        raise IntegrityError('Invalid score version/calibrator')


def freeze(manifest, *, model_freeze_id, approved_at, platform_model, rules_freeze_id=None):
    """No training or data access. Publish once after explicit leader approval."""
    registry=p._registry()
    if not isinstance(model_freeze_id,str) or not re.fullmatch(r'g2b-\d{8}',model_freeze_id):raise ValueError('Expected g2b-YYYYMMDD')
    if not isinstance(approved_at,str) or not approved_at.strip():raise ValueError('Leader approval time required')
    approved=datetime.fromisoformat(approved_at.replace('Z','+00:00'))
    if approved.tzinfo is None or approved.astimezone(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d')!=model_freeze_id[4:]:
        raise ValueError('Approval time must include timezone and match the freeze date')
    dest=registry.outputs/'frozen'/f'model-{model_freeze_id}.json'
    if dest.exists():raise FileExistsError(dest)
    manifest=_manifest_path(registry,manifest,draft=True)
    draft=read_json(manifest)
    required={'status','schema_version','rules_freeze_id','rules_freeze_sha256','build_run_id','entries','candidate_report','candidate_report_sha256'}
    if not required<=draft.keys():raise IntegrityError('Missing draft fields')
    if rules_freeze_id is not None and rules_freeze_id!=draft['rules_freeze_id']:raise IntegrityError('Rules identity mismatch')
    if draft['status']!='prepared' or draft['schema_version']!=2:raise IntegrityError('Prepared draft required')
    rules=p.load_rules(draft['rules_freeze_id'])
    build=registry.load_run(draft['build_run_id'])['run']
    if registry.path(build['draft_path'])!=Path(manifest).resolve() or build['draft_sha256']!=file_hash(manifest):
        raise IntegrityError('Draft differs from completed preparation source')
    if draft['rules_freeze_sha256']!=file_hash(registry.outputs/'frozen'/f"rules-{draft['rules_freeze_id']}.json"):
        raise IntegrityError('Rules hash changed')
    entries=deepcopy(draft['entries'])
    expected={row['entry_id']:row for row in entry_specs(rules.g2b['eligible_candidates'])}
    if len(entries)!=len(expected) or {r['entry_id'] for r in entries}!=set(expected):
        raise IntegrityError('Missing, duplicate or unexpected entries')
    for entry in entries:
        if any(entry.get(k)!=v for k,v in expected[entry['entry_id']].items()):
            raise IntegrityError('Frozen comparison specification mismatch')
        source=registry.load_run(entry['source_run_id'])
        # Generic provenance checks are identical before and after publication.
        _validate_source(registry,draft,entry,source)
    if platform_model not in ['M1-NS',*rules.g2b['eligible_candidates']]:
        raise ValueError('Leader must explicitly select an approved platform candidate')
    if file_hash(registry.path(draft['candidate_report']))!=draft['candidate_report_sha256']:
        raise IntegrityError('Candidate recommendation report hash mismatch')
    report=pd.read_csv(registry.path(draft['candidate_report']))
    if platform_model!='M1-NS':
        selected=report.loc[report.candidate==platform_model]
        if len(selected)!=1 or not bool(selected.iloc[0].d7_eligible) or not bool(selected.iloc[0].engineering_ready):
            raise IntegrityError('Selected candidate fails D7 or engineering checks')
    for key in ('s','n','c'):
        source=next(e for e in entries if e['model_key']==platform_model and e['policy_key']==key and e['score_version']=='raw')
        entries.append(dict(source,entry_id='platform-'+key,comparison_id='platform-release',role='platform' if key=='c' else 'research'))
    frozen=dict(schema_version=2,status='approved',model_freeze_id=model_freeze_id,
                rules_freeze_id=draft['rules_freeze_id'],rules_freeze_sha256=draft['rules_freeze_sha256'],
                approved_by=APPROVER,approved_at=approved_at,platform_model=platform_model,
                source_draft_sha256=file_hash(manifest),build_run_id=draft['build_run_id'],
                git_commit=build['git_commit'],git_dirty=build['git_dirty'],python=build['python'],
                packages=build['packages'],data_sha256=draft['data_sha256'],entries=entries)
    write_json(dest,frozen)
    for entry in entries:registry.frozen_entry(model_freeze_id,entry['entry_id'])
    return dest


def _validate_source(registry, freeze, entry, source):
    if source['run']['level']!='L1' or source['run']['rules_freeze_id']!=freeze['rules_freeze_id']:
        raise IntegrityError('Only matching L1 source runs can be frozen')
    for key in ('model_id','model_hash','params_hash'):
        if entry[key]!=source['run'][key]:raise IntegrityError('Model source mismatch: '+key)
    if entry['columns']!=source['model_record']['columns']:raise IntegrityError('Feature source mismatch')
    validate_entry(registry,freeze,entry,source)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--rules-freeze-id',default=fm.RULES_FREEZE_ID)
    prep.add_argument('--manifest',help='Optional JSON request containing only rules_freeze_id')
    final=commands.add_parser('freeze');final.add_argument('--manifest',required=True)
    final.add_argument('--model-freeze-id',required=True);final.add_argument('--approved-at',required=True)
    final.add_argument('--platform-model',required=True);final.add_argument('--rules-freeze-id')
    args=parser.parse_args(argv)
    if args.command=='prepare':return prepare(args.rules_freeze_id,manifest=args.manifest)
    return freeze(args.manifest,model_freeze_id=args.model_freeze_id,approved_at=args.approved_at,platform_model=args.platform_model,rules_freeze_id=args.rules_freeze_id)
