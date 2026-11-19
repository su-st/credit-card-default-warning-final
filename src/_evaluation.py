"""Private, shared frozen-evaluation reports and auditable batch manifests.

No customer data is obtained on import. Only the guarded CLI orchestrators obtain
frames through protocol. Public result files contain aggregates, never rows of scores.
"""
import argparse
from datetime import datetime, timezone
import re
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import model_freeze as mf, protocol as p
from .calibration import calibration_report, rank_change
from .data import age_group
from .registry import IntegrityError, content_hash, file_hash, identifier, read_json, write_json

COMPARISONS = {'s': '共同門檻0.5', 'n': '固定名單前20%', 'c': '正式政策'}
REPORT_RATES = tuple(i/100 for i in range(5, 51, 5))


def read_result_csv(path):
    """Preserve identifiers verbatim, including leading zeros and literal NA.

    Numeric metrics retain pandas' normal numeric/null parsing. Customer-level
    OOF/fold files use their existing readers, not this aggregate-table reader.
    """
    frame = pd.read_csv(path)
    text = {'model_key', 'candidate', 'level', 'role', 'score_version', 'policy_kind'}
    identifiers = [name for name in frame if name in text or
                   name.endswith(('_id', '_hash', '_sha256'))]
    if identifiers:
        # Use a separate text read so NA-like identifiers stay literal without
        # disabling the existing null/numeric parsing of undefined metrics.
        literal = pd.read_csv(path, usecols=identifiers, dtype=str, keep_default_na=False)
        for name in identifiers:
            frame[name] = literal[name]
    return frame


def validate_release(model_freeze_id):
    """Validate the whole approved inventory before either data loader is called."""
    registry = p._registry()
    if p.audit_sources():
        raise IntegrityError('G2b source audit must pass before formal data access')
    path = registry.outputs/'frozen'/f'model-{identifier(model_freeze_id)}.json'
    freeze = read_json(path)
    if (freeze.get('schema_version') != 2 or freeze.get('status') != 'approved'
            or freeze.get('approved_by') != mf.APPROVER or not freeze.get('approved_at')
            or freeze.get('model_freeze_id') != model_freeze_id):
        raise IntegrityError('Leader-approved version-2 model freeze required')
    stamp = datetime.fromisoformat(freeze['approved_at'].replace('Z', '+00:00'))
    if (stamp.utcoffset() is None or not re.fullmatch(r'g2b-\d{8}', model_freeze_id)
            or stamp.astimezone(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d')!=model_freeze_id[4:]):
        raise IntegrityError('Approval timezone/date must match model freeze identity')
    rules = p.load_rules(freeze['rules_freeze_id'])
    expected = {row['entry_id']: row for row in mf.entry_specs(rules.g2b['eligible_candidates'])}
    platform = freeze.get('platform_model')
    if platform not in ['M1-NS', *rules.g2b['eligible_candidates']]:
        raise IntegrityError('Invalid approved platform model')
    for key in COMPARISONS:
        original = next(row for row in expected.values() if row['model_key'] == platform
                        and row['policy_key'] == key and row['score_version'] == 'raw')
        expected['platform-'+key] = dict(original, entry_id='platform-'+key,
            comparison_id='platform-release', role='platform' if key == 'c' else 'research')
    entries = freeze.get('entries', [])
    if len(entries) != len(expected) or {e['entry_id'] for e in entries} != set(expected):
        raise IntegrityError('Incomplete or duplicate approved evaluation inventory')
    digest = file_hash(path)
    sources = {}
    for entry in entries:
        if any(entry.get(k) != v for k, v in expected[entry['entry_id']].items()):
            raise IntegrityError('Entry differs from approved comparison design')
        _, _, source, actual = registry.frozen_entry(model_freeze_id, entry['entry_id'])
        if actual != digest:
            raise IntegrityError('Model freeze changed during validation')
        sources[entry['entry_id']] = source
    build=registry.load_run(freeze['build_run_id'])['run']
    draft_path=registry.path(build['draft_path'])
    if (file_hash(draft_path)!=freeze['source_draft_sha256']
            or file_hash(draft_path)!=build['draft_sha256']):
        raise IntegrityError('Approved draft source hash mismatch')
    draft=read_json(draft_path)
    research=[row for row in entries if not row['entry_id'].startswith('platform-')]
    if research!=draft['entries'] or file_hash(registry.path(draft['candidate_report']))!=draft['candidate_report_sha256']:
        raise IntegrityError('Approved research inventory/candidate evidence changed')
    if platform!='M1-NS':
        report=read_result_csv(registry.path(draft['candidate_report']))
        selected=report.loc[report.candidate==platform]
        if len(selected)!=1 or not bool(selected.iloc[0].d7_eligible) or not bool(selected.iloc[0].engineering_ready):
            raise IntegrityError('Approved platform candidate lacks eligibility/engineering evidence')
    if file_hash(p.DATA_DIR/'default_of_credit_card_clients.csv')!=freeze['data_sha256']:
        raise IntegrityError('Current data file hash differs from the approved release')
    return freeze, sources, rules, digest


def partitions(frame):
    """The five F3 partitions, with the same labels as the 06 CSV producer."""
    sex = frame['SEX'].map({1: '男性', 2: '女性'})
    age = age_group(frame['AGE']).astype(str)
    edu = frame['EDUCATION'].where(frame['EDUCATION'].isin([1, 2, 3]), 4).map(
        {1: '研究所', 2: '大學', 3: '高中', 4: '其他'})
    mar = frame['MARRIAGE'].where(frame['MARRIAGE'].isin([1, 2]), 3).map({1: '已婚', 2: '單身', 3: '其他'})
    groups = {'性別': sex, '年齡層': age, '性別×年齡層': sex+'×'+age, '教育程度': edu, '婚姻狀況': mar}
    if any(group.isna().any() for group in groups.values()):
        raise IntegrityError('Incomplete evaluation partition')
    return groups


def policy_partition(frame, policy, rules_id):
    if policy.kind != 'group':
        return None
    if policy.selection == 'group-cost':
        return mf.fm.delay_partition(frame, rules_freeze_id=rules_id)
    # Frozen F3b uses the common partition helper, including its intersection labels.
    from ._protocol_runs import _partition
    return _partition(frame, policy.partition_name)


def check_test(frame, sources, rules, *, train=None):
    if frame.attrs.get('level')!=('L3' if train is not None else 'L2'):
        raise IntegrityError('Evaluation frame has the wrong data level')
    ids = p.validate_ids(frame.index)
    expected = 6000  # Official L2/L3 split size, also enforced by the registry handoff contract.
    if len(ids) != expected:
        raise IntegrityError('Expected one complete 20% evaluation split')
    p._labels(frame['default'])
    if train is not None:
        training = p.validate_ids(train.index)
        if len(training) != rules.data_protocol['selection']['train_rows'] or set(training) & set(ids):
            raise IntegrityError('Invalid/overlapping training and evaluation IDs')
    else:
        for record in {b['model_record']['model_id']: b['model_record'] for b in sources.values()}.values():
            oof = pd.read_csv(p._registry().path(record['model_path']).parent/'oof.csv', usecols=['customer_id'])
            if set(ids) & set(oof.customer_id):
                raise IntegrityError('Evaluation IDs overlap frozen training IDs')


def score_model(frame, entry, source):
    X, columns = mf.features(frame, entry['configuration'], source['run']['rules_freeze_id'])
    if list(columns) != entry['columns']:
        raise IntegrityError('Evaluation feature columns differ from frozen model')
    p.validate_ids(X.index)
    if set(X.index)!=set(frame.index):
        raise IntegrityError('Feature/customer IDs differ')
    X=X.loc[frame.index]
    values = source['model'].predict_proba(X)[:, 1]
    score = pd.Series(values, index=X.index).reindex(frame.index)
    if not np.isfinite(score).all() or not score.between(0, 1).all():
        raise IntegrityError('Invalid evaluation scores')
    return score, X


def tag(run, entry):
    return dict(level=run.meta['level'], entry_id=entry['entry_id'], comparison_id=entry['comparison_id'],
        model_id=run.meta['model_id'], run_id=run.run_id, model_freeze_id=run.meta['model_freeze_id'],
        rules_freeze_id=run.meta['rules_freeze_id'], score_version=entry['score_version'],
        model_key=entry['model_key'], policy_kind=entry['policy_key'], lag=entry['configuration']['lag'],
        is_primary=entry['policy_key']=='c',
        threshold_source_run_id=run.meta.get('policy_source_run_id', entry['source_run_id']),
        **({'policy_id': run.meta['policy_id']} if 'policy_id' in run.meta else {}),
        **({'seed': run.meta['seed']} if 'seed' in run.meta else {}))


def save_table(run, entry, name, frame):
    frame = frame.copy()
    for key, value in tag(run, entry).items():
        frame[key] = value
    run.save_csv(name, frame)
    return frame


def classification_counts(y, pred):
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    return dict(TN=tn, FP=fp, FN=fn, TP=tp)


def classification_table(counts):
    """Q7: no silent zero_division=0 or skip-null macro averaging."""
    tn, fp, fn, tp = (counts[k] for k in ('TN', 'FP', 'FN', 'TP'))
    rows = []
    for label, a, b, c in [(0, tn, fn, fp), (1, tp, fp, fn)]:
        row = dict(label=label, support=a+c, support_scope='class', **counts)
        for name, numerator, denominator in [('Precision', a, a+b), ('Recall', a, a+c), ('F1', 2*a, 2*a+b+c)]:
            row[name] = numerator/denominator if denominator else np.nan
            row[name+'_undefined'] = denominator == 0
        rows.append(row)
    macro = dict(label='macro', support=tn+fp+fn+tp, support_scope='total', **counts)
    for name in ('Precision', 'Recall', 'F1'):
        macro[name+'_undefined'] = any(row[name+'_undefined'] for row in rows)
        macro[name] = np.nan if macro[name+'_undefined'] else sum(row[name] for row in rows)/2
    return pd.DataFrame([*rows, macro])


def gap_tables(y, pred, groups, rules, *, pred_b=None, cache=None):
    """Exact bootstrap memoization in a caller-owned, single-batch cache.

    Hash ordered customer IDs and values, all partition names/order/content,
    both lists (including the absence of pred_b), n_boot and seed. Never cache
    across batches or expose the stored DataFrame to mutation by a caller.
    """
    bootstrap = rules.d8['bootstrap']
    key = None
    if cache is not None:
        p._aligned(y, pred, *groups.values(), *([] if pred_b is None else [pred_b]))
        p._labels(y)
        if (not pred.isin([0, 1]).all() or any(group.isna().any() for group in groups.values())
                or pred_b is not None and not pred_b.isin([0, 1]).all()):
            raise ValueError('Complete partitions and binary predictions required')
        if not isinstance(bootstrap['n_boot'], int) or bootstrap['n_boot'] < 1:
            raise ValueError('n_boot must be positive')
        def series_content(series):
            return dict(ids=series.index.tolist(), values=series.tolist())
        key = content_hash(dict(y=series_content(y), pred=series_content(pred),
            pred_b=None if pred_b is None else series_content(pred_b),
            groups=[dict(name=name, content=series_content(group)) for name, group in groups.items()],
            n_boot=bootstrap['n_boot'], seed=bootstrap['seed']))
        if key in cache:
            return cache[key].copy(deep=True)
    result = pd.concat([p.gap_ci(y, pred, group, n_boot=bootstrap['n_boot'], seed=bootstrap['seed'],
                       pred_b=pred_b).assign(分組方式=name) for name, group in groups.items()], ignore_index=True)
    if cache is not None:
        cache[key] = result.copy(deep=True)
    return result


def evaluate_entry(run, entry, frame, score, rules, timing, *, raw_score=None, gap_cache=None):
    y = frame['default']; bill = p.bill_at(frame, 0)
    if entry['evaluation_kind'] == 'score_only':
        save_table(run, entry, 'rq3_lead_time', pd.DataFrame([dict(AUC=roc_auc_score(y, score) if y.nunique()==2 else np.nan,
            fit_seconds=timing['fit_seconds'])]))
        save_table(run, entry, 'rq3_coverage', p.coverage_curve(y, score, bill=bill, rates=REPORT_RATES))
        save_table(run, entry, 'training_timing', pd.DataFrame([timing]))
        return
    threshold = read_json(run.path/'threshold.json')
    policy = p.Policy(**threshold['protocol_policy'])
    part = policy_partition(frame, policy, run.meta['rules_freeze_id'])
    final, info = p.apply_policy(score, policy, partition=part)
    groups = partitions(frame)
    report = p.evaluate(y, score=score, policy=policy, groups=groups, partition=part, bill=bill, scenario=rules.scenario)
    raw = p.evaluate(y, pred=info['pred_raw'], policy=policy, groups=groups, partition=part, bill=bill, scenario=rules.scenario)
    # Raw-list AUC is the same score statistic, not recomputed from binary flags.
    raw['overall']['AUC'] = report['overall']['AUC'].iloc[0]
    raw['overall']['原始名單人數'] = info['原始名單人數']
    raw['overall']['名單上限是否觸發'] = info['名單上限是否觸發']
    for name, table in report.items():save_table(run, entry, name, table)
    for name, table in raw.items():save_table(run, entry, 'raw_policy_'+name, table)
    save_table(run, entry, 'gap_ci', gap_tables(y, final, groups, rules, cache=gap_cache))
    counts = classification_counts(y, final)
    classes = classification_table(counts)
    classes['max_list_rate'] = policy.max_list_rate
    classes['cap_triggered'] = info['名單上限是否觸發']
    save_table(run, entry, 'classification_report', classes)
    matrix = pd.DataFrame([[counts['TN'], counts['FP']], [counts['FN'], counts['TP']]],
                          columns=['predicted_0', 'predicted_1']).rename_axis('actual').reset_index()
    save_table(run, entry, 'confusion_matrix', matrix)
    row = report['overall'].iloc[0].to_dict()
    row.update(Accuracy=(counts['TN']+counts['TP'])/len(y), fit_seconds=timing['fit_seconds'],
               calibrator_fit_seconds=timing['calibrator_fit_seconds'], **counts)
    save_table(run, entry, 'table_4_1_row', pd.DataFrame([row]))
    save_table(run, entry, 'training_timing', pd.DataFrame([timing]))
    if entry['score_version'] == 'calibrated':
        cal = calibration_report(y, score, bins=10)
        save_table(run, entry, 'calibration_report', pd.DataFrame([{'Brier分數':cal['Brier分數']}]))
        save_table(run, entry, 'calibration_bins', cal['校準曲線'])
        save_table(run, entry, 'rank_change', pd.DataFrame([dict(top_n_overlap=rank_change(raw_score, score, rules.g2b['policy']['rate']))]))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); curve = cal['校準曲線']
        ax.plot(curve['平均預測值'], curve['實際違約率'], marker='o'); ax.plot([0,1],[0,1], '--')
        ax.set(xlabel='Mean score', ylabel='Observed default rate')
        run.save_fig('calibration_curve', fig); plt.close(fig)
    if entry['entry_id'] == 'platform-c' and run.meta['level'] == 'L2':
        import matplotlib.pyplot as plt
        values = matrix[['predicted_0','predicted_1']].to_numpy()
        fig, ax = plt.subplots(); ax.imshow(values)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(values[i,j]),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted [non-default(0), default(1)]',
               ylabel='Actual [non-default(0), default(1)]')
        ax.set_title('\n'.join(str(run.meta[k]) for k in ('level','model_id','policy_id','run_id','model_freeze_id')), fontsize=8)
        fig.tight_layout(); run.save_fig('confusion_matrix',fig); plt.close(fig)


class Batch:
    """Summary identities are not model run identities. Failed snapshots stay immutable."""
    def __init__(self, level, freeze, digest, tasks, resume=None):
        self.registry = p._registry()
        self.gap_cache = {}
        prior = None
        reused = {}
        if resume:
            prior = read_json(self.registry.results/identifier(resume)/'evaluation_manifest.json')
            if (prior['status'] != 'failed' or prior['level'] != level
                    or prior['model_freeze_sha256'] != digest or set(prior['tasks']) != set(tasks)):
                raise IntegrityError('Resume requires the same failed evaluation batch/freeze')
            for key,row in prior['tasks'].items():
                if row['status']=='complete':
                    loaded=self.registry.load_run(row['run_id'])['run']
                    if (loaded['level']!=level or loaded['model_freeze_sha256']!=digest
                            or loaded.get('evaluation_task')!=key):
                        raise IntegrityError('Completed resume source mismatch')
                    self.verify_files(row)
                    reused[key]=dict(row,reused_from=resume)
        self.summary_id = 'summary-'+level.lower()+'-'+uuid4().hex
        self.path = self.registry.results/self.summary_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.meta = dict(summary_id=self.summary_id, level=level, model_freeze_id=freeze['model_freeze_id'],
            model_freeze_sha256=digest, rules_freeze_id=freeze['rules_freeze_id'],status='running',
            created_at=datetime.now(timezone.utc).isoformat(), previous_summary_id=resume,
            tasks={key:dict(status='pending',run_id=None) for key in tasks})
        self.meta['tasks'].update(reused)
        self.write()

    def write(self):
        write_json(self.path/'evaluation_manifest.json',self.meta,exclusive=False)

    def begin(self, key, run):
        self.meta['tasks'][key]=dict(status='running',run_id=run.run_id,
            model_hash=run.meta.get('model_hash'),policy_hash=run.meta.get('policy_hash'))
        run.update(evaluation_task=key,summary_id=self.summary_id)
        self.write()

    def done(self, key, run):
        self.meta['tasks'][key].update(status='complete',model_hash=run.meta['model_hash'],
            model_key=run.meta.get('model_key'),
            policy_hash=run.meta.get('policy_hash'),run_sha256=file_hash(run.path/'run.json'),
            result_sha256={file.name:file_hash(file) for file in run.result_path.iterdir() if file.is_file()})
        self.write()

    def verify_files(self, row):
        if file_hash(self.registry.runs/row['run_id']/'run.json')!=row['run_sha256']:
            raise IntegrityError('Completed evaluation run changed')
        for name,digest in row['result_sha256'].items():
            path=self.registry.results/row['run_id']/identifier(name)
            if not path.is_file() or file_hash(path)!=digest:
                raise IntegrityError('Completed evaluation result missing or changed')

    def fail(self, exc):
        self.meta.update(status='failed',failure=f'{type(exc).__name__}: {exc}',finished_at=datetime.now(timezone.utc).isoformat())
        for row in self.meta['tasks'].values():
            if row['status']=='running':
                row.update(status='failed',failure=self.meta['failure'])
        self.write()

    def completed(self, key):return self.meta['tasks'][key]['status']=='complete'

    def finish(self):
        if not all(row['status']=='complete' for row in self.meta['tasks'].values()):
            raise IntegrityError('Incomplete batch cannot publish a complete summary')
        sources=[];policy=[];classification=[];rq3=[];coverage=[]
        for key,row in self.meta['tasks'].items():
            self.verify_files(row)
            meta=self.registry.load_run(row['run_id'])['run']
            sources.append(dict(task=key,run_id=row['run_id'],model_id=meta['model_id'],model_hash=meta['model_hash'],
                policy_id=meta.get('policy_id'),seed=meta.get('seed'),entry_id=meta['entry'],
                level=meta['level'],rules_freeze_id=meta['rules_freeze_id'],model_freeze_id=meta['model_freeze_id'],
                comparison_id=meta['comparison_id'],model_source_run_id=meta.get('model_source_run_id'),
                threshold_source_run_id=meta.get('policy_source_run_id'),summary_id=self.summary_id))
            for filename, target in [('table_4_1_row.csv',policy),('classification_report.csv',classification),
                                     ('rq3_lead_time.csv',rq3),('rq3_coverage.csv',coverage)]:
                path=self.registry.results/row['run_id']/filename
                if path.is_file():target.append(read_result_csv(path))
        pd.DataFrame(sources).to_csv(self.path/'source_runs.csv',index=False)
        if self.meta['level']=='L2':
            combined=pd.concat(policy)
            combined.to_csv(self.path/'table_4_1_l2.csv',index=False)
            combined.loc[combined.is_primary].to_csv(self.path/'table_4_1_l2_main.csv',index=False)
            pd.concat(classification).to_csv(self.path/'table_4_3_l2.csv',index=False)
            pd.concat(rq3).to_csv(self.path/'rq3_summary.csv',index=False)
        else:
            by_seed=pd.concat(policy,ignore_index=True)
            by_seed.to_csv(self.path/'stability_by_seed.csv',index=False)
            keys=['comparison_id','entry_id','model_key','score_version','policy_kind','lag','rules_freeze_id']
            wanted=p.load_rules(self.meta['rules_freeze_id']).data_protocol['l3_seeds']
            records=[]
            for group,frame in by_seed.groupby(keys,dropna=False):
                if sorted(frame.seed.tolist())!=sorted(wanted):raise IntegrityError('Missing/duplicate L3 seed')
                for metric in ('AUC','Accuracy','Precision','Recall','F1','fit_seconds'):
                    values=frame[metric]
                    records.append(dict(zip(keys,group),metric=metric,mean=values.mean(),std=values.std(ddof=1),
                        min=values.min(),max=values.max(),n_valid=int(values.notna().sum())))
            summary=pd.DataFrame(records)
            summary.to_csv(self.path/'stability_summary.csv',index=False)
            summary['mean_plus_minus_std']=summary.apply(lambda row:f'{row["mean"]} ± {row["std"]}',axis=1)
            summary.to_csv(self.path/'table_4_1_l3.csv',index=False)
            summary.loc[summary.policy_kind.eq('c')].to_csv(self.path/'table_4_1_l3_main.csv',index=False)
            for frame in (pd.concat(rq3,ignore_index=True),pd.concat(coverage,ignore_index=True)):
                for _,part in frame.groupby('entry_id'):
                    if set(part.seed)!=set(wanted):raise IntegrityError('Missing RQ3 seed')
            combined=pd.concat(rq3,ignore_index=True)
            rq=combined.groupby(['entry_id','lag','rules_freeze_id'])[['AUC','fit_seconds']].agg(['mean','std','min','max','count'])
            rq.columns=['_'.join(col).replace('_count','_n_valid') for col in rq.columns]
            rq.reset_index().to_csv(self.path/'rq3_summary.csv',index=False)
            curves=pd.concat(coverage,ignore_index=True).groupby(['entry_id','lag','名單比例'])[['違約者涵蓋率','帳單金額涵蓋率']].agg(['mean','std','min','max','count'])
            curves.columns=['_'.join(col).replace('_count','_n_valid') for col in curves.columns]
            curves.reset_index().to_csv(self.path/'rq3_coverage_summary.csv',index=False)
        self.meta.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
                         source_run_ids=[row['run_id'] for row in self.meta['tasks'].values()])
        self.write()
        return self.summary_id


def cli(function, argv=None):
    parser=argparse.ArgumentParser(description=function.__doc__)
    parser.add_argument('model_freeze_id');parser.add_argument('--resume',help='Failed summary_id to resume without overwriting completed runs')
    args=parser.parse_args(argv)
    print(function(args.model_freeze_id,resume=args.resume))
