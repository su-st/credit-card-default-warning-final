"""Preserve 03/08 analysis tables on synthetic data while adding timing."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import roc_auc_score

from src import protocol as p
from src.calibration import crossfit_calibrate, calibration_report, rank_change
from test_model_freeze import sandbox


class MemoryRun:
    run_id='synthetic-analysis'
    def __init__(self):self.tables={};self.metadata={}
    def save_csv(self,name,frame):self.tables[name]=frame.copy()
    def save_fig(self,*args):pass
    def attach_model(self,*args):pass
    def update(self,**kwargs):self.metadata.update(kwargs)
    def finish(self):pass


def module(name):
    path=Path(__file__).parents[1]/'experiments'/name
    spec=importlib.util.spec_from_file_location(name[:-3],path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_08_legacy_tables_identical_and_timing_separate(sandbox,monkeypatch):
    _,frame,rules=sandbox
    m=module('08_calibration.py');run=MemoryRun()
    monkeypatch.setattr(m,'start_run',lambda *a,**k:run)
    monkeypatch.setattr(m,'XGB_TUNED',rules['dev_base']['params'])
    score=pd.Series(np.linspace(.05,.95,len(frame)),index=frame.index)
    monkeypatch.setattr(m,'oof_predict',lambda *a,**k:score.copy())
    monkeypatch.setattr(m.plt.Figure,'tight_layout',lambda self:None)
    actual=m.run_analysis(frame)
    # Reference is the original 08 summary/bin/rank calculation, with the same
    # common calibration functions and same input, not rounded expected guesses.
    calibrated,_=crossfit_calibrate(frame['default'],score,folds=m.FOLDS,seed=m.SEED)
    rows,curves=[],[]
    for name,values in {'原始風險分數':score,'校準後分數（交叉擬合 isotonic）':calibrated}.items():
        report=calibration_report(frame['default'],values,bins=m.BINS)
        rows.append({'版本':name,'AUC':roc_auc_score(frame['default'],values),
            'Brier 分數（越低越好）':report['Brier分數'],'平均分數':values.mean(),
            '實際違約率':frame['default'].mean(),'不同分數的個數':values.nunique()})
        curve=report['校準曲線'].copy();curve['分數區間']=curve['分數區間'].astype(str)
        curve.insert(0,'版本',name);curves.append(curve)
    ranks=pd.DataFrame([{'名單比例':rate,'與原始名單的重疊比例':rank_change(score,calibrated,rate)} for rate in m.RATES])
    assert_frame_equal(run.tables['calibration_report'],pd.DataFrame(rows).round(4))
    assert_frame_equal(run.tables['calibration_bins'],pd.concat(curves,ignore_index=True).round(4))
    assert_frame_equal(run.tables['rank_change'],ranks.round(4))
    assert all(v>=0 for k,v in run.metadata['training_timing'].items() if k.endswith('seconds'))
    assert 'training_timing' in run.tables
    import matplotlib.pyplot as plt
    plt.close('all')


def test_03_single_split_legacy_metrics_and_return_contract(sandbox,monkeypatch):
    _,frame,_=sandbox;m=module('03_repeated_cv.py')
    rules=p.load_rules('synthetic-rules')
    score=pd.Series(np.linspace(.05,.95,len(frame)),index=frame.index)
    monkeypatch.setattr(m,'oof_predict',lambda *a,**k:score.copy())
    actual=m.evaluate_one_split(frame,None,rules=rules)
    formal=p.choose_threshold(frame['default'],score,p.bill_at(frame,0),rules.scenario)['政策物件']
    rows=[]
    for name,policy in {'共同數值門檻 0.5':p.Policy.single(.5,None),'固定名單前 20%':p.Policy.top_n(.2),'正式政策（成本門檻）':formal}.items():
        overall=p.evaluate(frame['default'],score=score,policy=policy,bill=p.bill_at(frame,0),scenario=rules.scenario)['overall']
        overall.insert(0,'政策',name);rows.append(overall)
    expected=pd.concat(rows,ignore_index=True)
    actual.attrs={}
    assert_frame_equal(actual,expected)


def test_03_rq3_computation_source_is_unchanged():
    import ast,hashlib
    source=(Path(__file__).parents[1]/'experiments/03_repeated_cv.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='rq3_lead_time')
    assert hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()=='253966d28c2044c5c0c846051a23a5dd4ac24cc1bea41fdfb6eb736eeba107f2'
