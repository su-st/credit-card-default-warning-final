"""C6b reads synthetic saved summaries only, never models or customer data."""
import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app import data as platform
from src import protocol as p
from src.registry import IntegrityError, Registry


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    def forbidden(*args, **kwargs):
        pytest.fail('Reader attempted to load customer data, models or recalculate metrics')
    for name in ('dev_data', 'dev_holdout', 'final_test', '_full_data', 'evaluate', 'gap_ci', 'open_sealed'):
        monkeypatch.setattr(p, name, forbidden)
    monkeypatch.setattr(Registry, 'load_run', forbidden)
    for role, run_id, model_id in [('M1', 'm1-run', 'm1-model'), ('M1-NS', 'ns-run', 'ns-model')]:
        folder = tmp_path/'results'/run_id
        folder.mkdir(parents=True)
        meta = dict(run_id=run_id, model_id=model_id, status='complete', git_dirty=False,
                    level='L1', rules_freeze_id='g2a-20261116')
        (folder/'run.json').write_text(json.dumps(meta))
        tables = {key: [] for key in ('overall', 'groups', 'gaps_point', 'gaps')}
        for comp in platform.FAIRNESS_COMPARISONS:
            tag = dict(run_id=run_id, model_id=model_id, 模型=role, 政策比較=comp)
            tables['overall'].append(dict.fromkeys(p.OVERALL_COLUMNS, .123456789012345))
            tables['overall'][-1].update(tag, policy_id=run_id+'-policy')
            for part in platform.FAIRNESS_PARTITIONS:
                group = dict.fromkeys(p.GROUP_COLUMNS, .234567890123456)
                group.update(tag, 分組方式=part, 族群='人工族群', 總人數=10, 違約人數=4, 未違約人數=6, 小樣本=True)
                tables['groups'].append(group)
                tables['gaps_point'].append(dict(tag, 分組方式=part, **{'EO 差距':.123456789012345, 'FPR 差距':.2}))
                for metric in ('EO 差距', 'FPR 差距'):
                    ci = dict(tag, 分組方式=part, 指標=metric, 估計值=.123456789012345,
                              下限=-.1, 上限=.3, 有效重抽=2000, 無效重抽=0, 可靠=True, 比較='單一名單')
                    tables['gaps'].append(ci)
                    if role=='M1-NS':
                        tables['gaps'].append(dict(ci, 模型='M1-NS 減 M1', 比較='B-A',
                                                   m1_run_id='m1-run', m1_model_id='m1-model'))
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(folder/f'fairness_{name}.csv', index=False)
    return tmp_path


def load(root):
    return platform.load_fairness_comparison('ns-run', root=root)


def edit_meta(root, run, key, value):
    path = root/'results'/run/'run.json'
    meta=json.loads(path.read_text());meta[key]=value
    path.write_text(json.dumps(meta))


def edit_csv(root, run, name, change):
    path=root/'results'/run/f'fairness_{name}.csv'
    frame=pd.read_csv(path);change(frame).to_csv(path,index=False)


@pytest.mark.parametrize('level', ['L1','L2'])
def test_reads_saved_tables_and_provenance_without_recalculation(comparison, level):
    root=comparison
    for run in ('m1-run','ns-run'):edit_meta(root,run,'level',level)
    bundle=load(root)
    assert bundle['source']==dict(level=level,rules_freeze_id='g2a-20261116',m1_run_id='m1-run',
        m1_ns_run_id='ns-run',comparison_names=list(platform.FAIRNESS_COMPARISONS),comparison_direction='M1-NS 減 M1')
    for role,run in [('M1','m1-run'),('M1-NS','ns-run')]:
        result=bundle['models'][role]
        for name in ('overall','groups','gaps_point'):
            assert_frame_equal(result[name],pd.read_csv(root/'results'/run/f'fairness_{name}.csv'))
        expected=pd.read_csv(root/'results'/run/'fairness_gaps.csv')
        assert_frame_equal(result['gaps_ci'],expected[expected['模型'].eq(role)])
        assert result['source']['run_id']==run
        assert result['source']['model_id']==('m1-model' if role=='M1' else 'ns-model')
        assert len(result['source']['files_sha256'])==4
    assert len(bundle['paired'])==30
    assert set(bundle['paired']['m1_run_id'])=={'m1-run'}


@pytest.mark.parametrize('run', ['m1-run','ns-run'])
@pytest.mark.parametrize('key,value', [('status','running'),('status','failed'),('git_dirty',True),
    ('git_dirty','false'),('level','L1-H'),('level','L3'),('rules_freeze_id','other-rules'),
    ('run_id','other-run'),('model_id','wrong-model')])
def test_rejects_invalid_run_metadata(comparison,run,key,value):
    edit_meta(comparison,run,key,value)
    with pytest.raises(IntegrityError):load(comparison)


@pytest.mark.parametrize('run', ['m1-run','ns-run'])
@pytest.mark.parametrize('name', ['overall','groups','gaps_point','gaps'])
def test_requires_each_file_and_all_columns(comparison,run,name):
    path=comparison/'results'/run/f'fairness_{name}.csv'
    original=path.read_bytes()
    path.unlink()
    with pytest.raises(IntegrityError):load(comparison)
    path.write_bytes(original)
    frame=pd.read_csv(path)
    # Check every required column, not just a single example.
    optional={'m1_run_id','m1_model_id'} if run=='m1-run' else set()
    for col in set(frame)-optional:
        frame.drop(columns=col).to_csv(path,index=False)
        with pytest.raises(IntegrityError):load(comparison)
    path.write_bytes(original)


@pytest.mark.parametrize('run', ['m1-run','ns-run'])
@pytest.mark.parametrize('name', ['overall','groups','gaps_point','gaps'])
@pytest.mark.parametrize('key', ['model_id','run_id','模型','政策比較'])
def test_each_result_row_must_match_its_source(comparison,run,name,key):
    def change(frame):
        frame.loc[0,key]='wrong';return frame
    edit_csv(comparison,run,name,change)
    with pytest.raises(IntegrityError):load(comparison)


@pytest.mark.parametrize('key,value', [('m1_run_id','other-run'),('m1_run_id','ns-run'),
    ('m1_model_id','other-model'),('m1_model_id',None),('比較','A-B')])
def test_rejects_invalid_paired_sources_and_direction(comparison,key,value):
    def change(frame):
        frame.loc[frame['模型'].eq('M1-NS 減 M1'),key]=value
        return frame
    edit_csv(comparison,'ns-run','gaps',change)
    with pytest.raises((IntegrityError,ValueError)):load(comparison)


def test_rejects_mixed_data_levels(comparison):
    edit_meta(comparison,'m1-run','level','L2')
    with pytest.raises(IntegrityError):load(comparison)


@pytest.mark.parametrize('name', ['overall','groups','gaps_point','gaps'])
def test_rejects_duplicate_or_missing_comparisons(comparison,name):
    path=comparison/'results/ns-run'/f'fairness_{name}.csv'
    frame=pd.read_csv(path)
    pd.concat([frame,frame.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(IntegrityError):load(comparison)
    frame.loc[frame['政策比較'].ne('正式政策')].to_csv(path,index=False)
    with pytest.raises(IntegrityError):load(comparison)


def test_rejects_missing_pair_and_bad_ci_metric(comparison):
    path=comparison/'results/ns-run/fairness_gaps.csv'
    frame=pd.read_csv(path)
    frame[frame['模型'].eq('M1-NS')].to_csv(path,index=False)
    with pytest.raises(IntegrityError):load(comparison)
    frame.loc[frame['模型'].eq('M1-NS 減 M1'),'指標']='unknown'
    frame.to_csv(path,index=False)
    with pytest.raises(IntegrityError):load(comparison)


def test_setting_is_independent_and_missing_setting_fails(comparison,monkeypatch):
    monkeypatch.delenv('PLATFORM_FAIRNESS_RUN_ID',raising=False)
    monkeypatch.setenv('PLATFORM_RUN_ID','unrelated-platform-run')
    with pytest.raises(ValueError,match='PLATFORM_FAIRNESS_RUN_ID'):
        platform.load_fairness_comparison(root=comparison)
    monkeypatch.setenv('PLATFORM_FAIRNESS_RUN_ID','ns-run')
    assert platform.load_fairness_comparison(root=comparison)['source']['m1_ns_run_id']=='ns-run'
    # A valid L1 F3 run still cannot serve as the platform's L1-H handoff.
    output=comparison/'outputs/runs/ns-run';output.mkdir(parents=True)
    (output/'run.json').write_bytes((comparison/'results/ns-run/run.json').read_bytes())
    with pytest.raises(IntegrityError,match='L1-H'):
        platform.load_platform_run('ns-run',root=comparison)


def test_bad_ids_and_invalid_json_never_load(comparison):
    with pytest.raises(ValueError):platform.load_fairness_comparison('../sealed',root=comparison)
    (comparison/'results/ns-run/run.json').write_text('not json')
    with pytest.raises(IntegrityError):load(comparison)


def test_rejects_mismatched_group_labels(comparison):
    def change(frame):
        frame.loc[0,'族群']='another group';return frame
    edit_csv(comparison,'m1-run','groups',change)
    with pytest.raises(IntegrityError,match='族群比較'):load(comparison)


@pytest.mark.parametrize('filename',['run.json','fairness_groups.csv'])
def test_rejects_files_changing_during_load(comparison,monkeypatch,filename):
    original=platform._fairness_bytes;calls={}
    def changed(registry,run_id,name):
        key=(run_id,name);calls[key]=calls.get(key,0)+1
        raw=original(registry,run_id,name)
        return raw+b' ' if run_id=='ns-run' and name==filename and calls[key]>1 else raw
    monkeypatch.setattr(platform,'_fairness_bytes',changed)
    with pytest.raises(IntegrityError,match='發生變動'):load(comparison)


def test_undefined_confidence_intervals_are_preserved(comparison):
    def change(frame):
        frame.loc[0,['估計值','下限','上限']]=float('nan')
        frame.loc[0,'可靠']=False
        return frame
    edit_csv(comparison,'ns-run','gaps',change)
    row=load(comparison)['models']['M1-NS']['gaps_ci'].iloc[0]
    assert pd.isna(row['下限']) and not row['可靠']
