"""L2: evaluate an approved release once, without fitting or selecting settings."""
import pandas as pd

from . import model_freeze as mf, protocol as p
from ._evaluation import (Batch, COMPARISONS, check_test, cli, evaluate_entry,
                          gap_tables, partitions, policy_partition, read_result_csv, score_model, validate_release)
from .data import age_group
from .export import EDU, MAR, SEX, top_factors, verify_predictions
from .registry import IntegrityError, content_hash, file_hash, read_json


def export_platform(run, entry, frame, score, X, model):
    """Only platform-c emits customer rows, privately, with complete reload verification."""
    if run.meta['level']!='L2' or entry['entry_id']!='platform-c':
        raise PermissionError('Only frozen platform-c can export the L2 handoff')
    threshold=read_json(run.path/'threshold.json');policy=p.Policy(**threshold['protocol_policy'])
    pred,_=p.apply_policy(score,policy,partition=policy_partition(frame,policy,run.meta['rules_freeze_id']))
    factors=top_factors(model,X)
    if len(factors)!=len(frame) or any(len(row)!=3 for row in factors):
        raise IntegrityError('Missing platform SHAP factors')
    output=pd.DataFrame(dict(customer_id=frame.index,risk_score=score.to_numpy(),in_list=pred.to_numpy(),
        sex=frame.SEX.map(SEX).to_numpy(),age=frame.AGE.to_numpy(),age_group=age_group(frame.AGE).astype(str).to_numpy(),
        education=frame.EDUCATION.map(EDU).to_numpy(),marriage=frame.MARRIAGE.map(MAR).to_numpy(),
        limit_bal=frame.LIMIT_BAL.to_numpy(),bill_amt_latest=frame.BILL_AMT1.to_numpy(),
        actual_default=frame.default.to_numpy(),data_split='test',model_id=run.meta['model_id'],
        policy_id=run.meta['policy_id'],run_id=run.run_id))
    for i in range(3):output[f'factor_{i+1}']=[row[i] for row in factors]
    output=output.sort_values(['risk_score','customer_id'],ascending=[False,True])
    output.insert(2,'risk_rank',range(1,len(output)+1))
    with (run.path/'predictions.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        output.to_csv(handle,index=False,float_format='%.17g')
    run.update(predictions_sha256=file_hash(run.path/'predictions.csv'),test_ids_hash=content_hash(sorted(frame.index.tolist())))
    count=verify_predictions(run.registry,run.run_id,X)
    if count!=len(frame) or count!=6000:
        raise IntegrityError('Official platform export requires all 6,000 reload verifications')
    run.update(predictions_verified=True,verified_customers=count)


def fairness_comparison(batch, freeze, entries, frame, scores, rules, label):
    """The 06 double-run CSV contract, directly consumable by C6b."""
    key='fairness-'+label
    if batch.completed(key):return
    prefix='rq1-m1-' if label=='M1' else 'rq1-ns-'
    chosen=entries[prefix+'c']
    run=p.start_run('L2','F3 comparison '+label,model_freeze_id=freeze['model_freeze_id'],entry=chosen['entry_id'])
    try:
        batch.begin(key,run)
        overall=[];groups=[];points=[];intervals=[]
        flags={}
        base_columns={'overall':p.OVERALL_COLUMNS,'groups':p.GROUP_COLUMNS,'gaps':p.GAP_COLUMNS,
                      'gap_ci':['指標','估計值','下限','上限','有效重抽','無效重抽','可靠','比較','分組方式']}
        for policy_key,cname in COMPARISONS.items():
            entry=entries[prefix+policy_key]
            source_id=batch.meta['tasks'][entry['entry_id']]['run_id']
            folder=run.registry.results/source_id
            assign=dict(模型=label,政策比較=cname,model_id=run.meta['model_id'],run_id=run.run_id)
            for filename,target in [('overall',overall),('groups',groups),('gaps',points),('gap_ci',intervals)]:
                target.append(read_result_csv(folder/(filename+'.csv')).loc[:,base_columns[filename]].assign(**assign))
            policy=p.Policy(**entry['protocol_policy'])
            flags[policy_key]=p.apply_policy(scores[entry['model_hash']],policy)[0]
        if label=='M1-NS':
            m1_id=batch.meta['tasks']['fairness-M1']['run_id']
            m1_meta=run.registry.load_run(m1_id)['run']
            for policy_key,cname in COMPARISONS.items():
                m1_entry=entries['rq1-m1-'+policy_key]
                m1_pred=p.apply_policy(scores[m1_entry['model_hash']],p.Policy(**m1_entry['protocol_policy']))[0]
                pair=gap_tables(frame['default'],m1_pred,partitions(frame),rules,
                                pred_b=flags[policy_key],cache=batch.gap_cache)
                intervals.append(pair.assign(模型='M1-NS 減 M1',政策比較=cname,m1_run_id=m1_id,
                    m1_model_id=m1_meta['model_id'],model_id=run.meta['model_id'],run_id=run.run_id))
        for filename,tables in [('fairness_overall',overall),('fairness_groups',groups),
                                ('fairness_gaps_point',points),('fairness_gaps',intervals)]:
            table=pd.concat(tables,ignore_index=True)
            if filename=='fairness_gaps':
                columns=['指標','估計值','下限','上限','有效重抽','無效重抽','可靠','比較','模型','政策比較','分組方式']
                if label=='M1-NS':columns+=['m1_run_id','m1_model_id']
            else:
                source_key={'fairness_overall':'overall','fairness_groups':'groups','fairness_gaps_point':'gaps'}[filename]
                columns=[*base_columns[source_key],'模型','政策比較']
            table=table.loc[:,[*columns,'run_id','model_id']]
            # The 06 producer has no top-level policy on its comparison run.
            # Write its exact CSV schema without Run.save_csv injecting C's ID
            # into the groups/CI tables. Per-policy sources remain in metadata.
            run.result_path.mkdir(parents=True,exist_ok=True)
            with (run.result_path/(filename+'.csv')).open('x',encoding='utf-8',newline='') as handle:
                table.to_csv(handle,index=False)
        run.update(comparison_name='RQ1 F3',comparison_id='rq1-sensitive',
                   training_timing=chosen['training_timing'],model_source_run_id=chosen['model_source_run_id'],
                   policy_source_run_id=chosen['source_run_id'],
                   comparison_policy_sources={k:batch.meta['tasks'][prefix+k]['run_id'] for k in COMPARISONS})
        run.finish();batch.done(key,run)
    except Exception as exc:
        if run.meta['status']=='running':run.fail(f'{type(exc).__name__}: {exc}')
        raise


def run_final_eval(model_freeze_id, *, resume=None):
    """Validate all approved entries, then obtain L2 once and publish a complete batch."""
    freeze,sources,rules,digest=validate_release(model_freeze_id)
    entries={entry['entry_id']:entry for entry in freeze['entries']}
    batch=Batch('L2',freeze,digest,[*entries,'fairness-M1','fairness-M1-NS'],resume)
    try:
        frame=p.final_test(model_freeze_id)
        check_test(frame,sources,rules)
        scores={};features={};calibrated={}
        for key,entry in entries.items():
            if batch.completed(key):continue
            run=p.start_run('L2','Frozen evaluation '+key,model_freeze_id=model_freeze_id,entry=key)
            try:
                batch.begin(key,run)
                h=entry['model_hash']
                if h not in scores:
                    scores[h],features[h]=score_model(frame,entry,sources[key])
                score=scores[h]
                if entry['score_version']=='calibrated':
                    cal_hash=(h,entry['calibrator_hash'])
                    if cal_hash not in calibrated:
                        calibrator,_=mf.load_calibrator(run.registry,entry['calibrator'],sources[key]['model_record'])
                        calibrated[cal_hash]=pd.Series(calibrator.predict(score),index=score.index)
                    score=calibrated[cal_hash]
                run.update(comparison_id=entry['comparison_id'],training_timing=entry['training_timing'],
                    model_source_run_id=entry['model_source_run_id'],
                    policy_source_run_id=entry['source_run_id'],test_ids_hash=content_hash(sorted(frame.index.tolist())))
                evaluate_entry(run,entry,frame,score,rules,entry['training_timing'],
                               raw_score=scores[h],gap_cache=batch.gap_cache)
                if key=='platform-c':export_platform(run,entry,frame,score,features[h],sources[key]['model'])
                run.finish();batch.done(key,run)
            except Exception as exc:
                if run.meta['status']=='running':run.fail(f'{type(exc).__name__}: {exc}')
                raise
        # Resume may skip every entry, but F3 still needs the two raw score vectors.
        for key in ('rq1-m1-c','rq1-ns-c'):
            entry=entries[key]
            if entry['model_hash'] not in scores:
                scores[entry['model_hash']],_=score_model(frame,entry,sources[key])
        for label in ('M1','M1-NS'):fairness_comparison(batch,freeze,entries,frame,scores,rules,label)
        return batch.finish()
    except Exception as exc:
        batch.fail(exc)
        raise


def main(argv=None):cli(run_final_eval,argv)


if __name__=='__main__':main()
