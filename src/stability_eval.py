"""L3: frozen-design split sensitivity, never ten independent external validations."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd

from . import model_freeze as mf, protocol as p
from ._evaluation import Batch, check_test, cli, evaluate_entry, score_model, validate_release
from ._protocol_runs import ProtocolRun, _L3_SOURCE, training_evidence
from .registry import IntegrityError, content_hash
from .training_time import execution_environment, timed_call


def _resumed_bundle(batch, seed, entry, train):
    for key,row in batch.meta['tasks'].items():
        if not key.startswith(str(seed)+'/') or row['status']!='complete':continue
        if row.get('model_key') not in (None,entry['model_key']):continue
        source=batch.registry.load_run(row['run_id'])
        if source['run'].get('model_key')!=entry['model_key']:continue
        # Always reference the run that actually performed the fit.
        fit_id=source['run']['retraining']['fit_run_id']
        source=batch.registry.load_run(fit_id)
        record=source['model_record'];folder=batch.registry.path(record['model_path']).parent
        original=pd.read_csv(folder/'oof.csv',converters={'oof_score':float})
        if content_hash(train.index.tolist())!=record['training_ids_hash']:
            raise IntegrityError('Resume training split differs from completed fit')
        oof=pd.Series(original.oof_score.to_numpy(),index=train.index)
        assignment=pd.read_csv(folder/'folds.csv')
        oof.attrs.update(fold_assignment=pd.Series(assignment['fold'].to_numpy(),index=train.index),
                         folds=record['n_splits'],seed=source['run']['cv_seed'])
        X,_=mf.features(train,entry['configuration'],source['run']['rules_freeze_id'])
        bundle=dict(run=ProtocolRun(batch.registry,source['run']),record=record,model=source['model'],X=X,oof=oof,
                    timing=source['run']['training_timing'],configuration=entry['configuration'],test_score=None,policies={})
        for task,row in batch.meta['tasks'].items():
            if not task.startswith(str(seed)+'/') or row['status']!='complete':continue
            if row.get('model_key') not in (None,entry['model_key']):continue
            completed=batch.registry.load_run(row['run_id'])
            if completed['run'].get('model_key')!=entry['model_key'] or 'calibrator' not in completed['run']:continue
            calibration=completed['run']['calibrator']
            _,cross=mf.load_calibrator(batch.registry,calibration,record)
            bundle['calibration']=(calibration,pd.Series(cross.oof_score.to_numpy(),index=train.index))
            bundle['calibration_source_run_id']=completed['run'].get('calibration_source_run_id',row['run_id'])
            break
        return bundle
    return None


def _train(run, entry, train, seed, rules):
    clock=mf._RunClock();artifact=0.
    X,_=mf.features(train,entry['configuration'],run.meta['rules_freeze_id'])
    factory=lambda:mf.model_factory(entry['configuration'],run.meta['rules_freeze_id'])
    oof,oof_seconds=timed_call(p.oof_predict,factory,X,train['default'],folds=rules.cv['n_splits'],seed=rules.cv['random_state'])
    model,fit_seconds=timed_call(factory().fit,X,train['default'])
    run.update(seed=int(seed),retraining=training_evidence(run,X,oof,entry['configuration'],seed,fit_seconds))
    # record.seed is the outer split seed, folds keep their explicit frozen CV seed.
    record,seconds=timed_call(run.attach_model,model,X,train['default'],oof,seed=seed)
    artifact+=seconds
    timing=mf._timing(record,fit_seconds,oof_seconds,model,execution_environment())
    timing.update(split_seed=seed,cv_seed=rules.cv['random_state'])
    return dict(run=run,record=record,model=model,X=X,oof=oof,timing=timing,
                configuration=entry['configuration'],clock=clock,artifact=artifact,test_score=None,policies={})


def _policy(bundle, entry, train, rules):
    key=(entry['score_version'],entry['policy_key'])
    if key in bundle['policies']:return bundle['policies'][key]
    calibration=None;search=None
    if entry['score_version']=='calibrated':
        calibration,oof=bundle['calibration']
    else:oof=bundle['oof']
    policy_key=entry['policy_key']
    if policy_key=='s':policy=p.Policy.single(.5,None)
    elif policy_key=='n':policy=p.Policy.top_n(rules.g2b['policy']['rate'])
    elif policy_key=='c':policy=p.choose_threshold(train['default'],oof,p.bill_at(train,0),rules.scenario)['政策物件']
    elif policy_key=='e':
        search=mf.fm.search_no_delay_threshold(train,oof,rules_freeze_id=rules.rules_freeze_id)
        policy=search['policy']
    else:
        from ._protocol_runs import _partition
        search=mf.fm.search_group_thresholds(train['default'],oof,p.bill_at(train,0),_partition(train,policy_key),
            partition_name=policy_key,rules_freeze_id=rules.rules_freeze_id)
        policy=search['policy']
    if calibration:
        policy=replace(policy,evidence={**policy.evidence,'calibrator_hash':calibration['calibrator_hash'],
                                        'crossfit_oof_sha256':calibration['crossfit_oof_sha256']})
    result=(policy,calibration,search)
    bundle['policies'][key]=result
    return result


def run_stability_eval(model_freeze_id, *, resume=None):
    """Validate first, then evaluate every frozen entry on each frozen outer split."""
    freeze,sources,rules,digest=validate_release(model_freeze_id)
    entries=freeze['entries'];seeds=rules.data_protocol['l3_seeds']
    batch=Batch('L3',freeze,digest,[f'{seed}/{entry["entry_id"]}' for seed in seeds for entry in entries],resume)
    try:
        seen=[]
        for seed,train,test in p.outer_splits(model_freeze_id,seeds=seeds):
            if seed not in seeds or seed in seen:raise IntegrityError('Unexpected/duplicate L3 split seed')
            seen.append(seed)
            if train.attrs.get('seed')!=seed or test.attrs.get('seed')!=seed:
                raise IntegrityError('L3 frame/split seed mismatch')
            check_test(test,sources,rules,train=train)
            bundles={}
            for entry in entries:
                key=f'{seed}/{entry["entry_id"]}'
                if batch.completed(key):continue
                name=entry['model_key'];bundle=bundles.get(name)
                if bundle is None:
                    bundle=_resumed_bundle(batch,seed,entry,train)
                if bundle is None:
                    run=p.start_run('L3','L3 independent fit '+key,model_freeze_id=model_freeze_id,entry=entry['entry_id'])
                else:
                    token=_L3_SOURCE.set(bundle['run'].run_id)
                    try:run=p.start_run('L3','L3 policy '+key,model_freeze_id=model_freeze_id,entry=entry['entry_id'])
                    finally:_L3_SOURCE.reset(token)
                try:
                    batch.begin(key,run)
                    if bundle is None:
                        bundle=_train(run,entry,train,seed,rules)
                    else:run.bind_model(bundle['record'])
                    bundles[name]=bundle
                    clock=bundle['clock'] if bundle['run'].run_id==run.run_id and 'clock' in bundle else mf._RunClock()
                    timing=deepcopy(bundle['timing']);artifact=0.
                    run.update(seed=int(seed),model_key=name,configuration=entry['configuration'],comparison_id=entry['comparison_id'],
                               model_source_run_id=bundle['run'].run_id,cv_seed=rules.cv['random_state'])
                    calibration=None
                    if entry['evaluation_kind']!='score_only':
                        if entry['score_version']=='calibrated' and 'calibration' not in bundle:
                            bundle['calibration'],elapsed=timed_call(mf._save_calibrator,run.registry,
                                bundle['record'],train['default'],bundle['oof'],rules)
                            calibration=bundle['calibration'][0]
                            artifact+=max(0.,elapsed-calibration['calibrator_fit_seconds']-calibration['crossfit_seconds'])
                            bundle['calibration_source_run_id']=run.run_id
                        (policy,calibration,search),policy_seconds=timed_call(_policy,bundle,entry,train,rules)
                        timing['policy_seconds']=policy_seconds
                        if calibration:
                            run.update(calibrator=calibration,calibrator_hash=calibration['calibrator_hash'],
                                       calibrator_record=calibration['calibrator_record'],
                                       calibration_source_run_id=bundle['calibration_source_run_id'])
                            timing.update(calibrator_fit_seconds=calibration['calibrator_fit_seconds'],
                                crossfit_seconds=calibration['crossfit_seconds'],
                                calibrator_source_run_id=bundle['calibration_source_run_id'])
                        policy,seconds=timed_call(run.record_policy,policy);artifact+=seconds
                        if search:
                            summary,seconds=timed_call(mf._search_results,bundle,policy,search,train)
                            timing['policy_seconds']+=seconds
                            _,seconds=timed_call(mf._save_search_evidence,run,search,summary);artifact+=seconds
                    else:run.update(evaluation_kind='score_only')
                    timing=clock.finish(timing,artifact+bundle.get('artifact',0.) if bundle['run'].run_id==run.run_id else artifact,'L3 training/policy')
                    mf._check_timing(timing)
                    run.update(training_timing=timing,policy_source_run_id=run.run_id)
                    if bundle['run'].run_id==run.run_id:bundle['timing']=timing
                    if bundle['test_score'] is None:
                        bundle['test_score'],_=score_model(test,entry,dict(run=run.meta,model=bundle['model']))
                    score=bundle['test_score']
                    if calibration:
                        if 'calibrated_test' not in bundle:
                            calibrator,_=mf.load_calibrator(run.registry,calibration,bundle['record'])
                            bundle['calibrated_test']=pd.Series(calibrator.predict(score),index=score.index)
                        score=bundle['calibrated_test']
                    evaluate_entry(run,entry,test,score,rules,timing,raw_score=bundle['test_score'])
                    run.finish();batch.done(key,run)
                except Exception as exc:
                    if run.meta['status']=='running':run.fail(f'{type(exc).__name__}: {exc}')
                    raise
        if sorted(seen)!=sorted(seeds):raise IntegrityError('Missing L3 seed, refusing ten-seed summary')
        return batch.finish()
    except Exception as exc:
        batch.fail(exc)
        raise


def main(argv=None):cli(run_stability_eval,argv)


if __name__=='__main__':main()
