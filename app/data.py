"""Verified run-backed platform data; no legacy/global handoff fallback."""
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.config import ROOT
from src.registry import IntegrityError, Registry, file_hash, identifier, read_json

BACKTEST = '歷史回測'
SIMULATION = '新資料預警（模擬）'
SIMULATION_COLUMNS = (
    'customer_id', 'risk_score', 'risk_rank', 'in_list', 'sex', 'age', 'age_group',
    'education', 'marriage', 'limit_bal', 'bill_amt_latest', 'factor_1', 'factor_2',
    'factor_3', 'data_split', 'model_id', 'run_id', 'policy_id',
)


@st.cache_data(show_spinner=False)
def _cached_predictions(run_id, predictions_sha256, _verified_frame):
    # The two public arguments are the cache key. Every caller first performs
    # full registry/file verification; a cache hit never bypasses those checks.
    return _verified_frame.copy(deep=True)


def load_platform_run(run_id=None, *, root=None):
    """Load a complete, verified bundle; reject non-L1-H before reading scores.

    Pages receive the same interface when formal support is added later. C6
    deliberately does not enable L2 or load an unapproved test handoff.
    """
    run_id = run_id if run_id is not None else os.getenv('PLATFORM_RUN_ID', '')
    if not run_id:
        raise ValueError('請設定 PLATFORM_RUN_ID，填入已完成的 M1-NS-dev19200 開發 run_id。')
    run_id = identifier(run_id)
    registry = Registry(ROOT if root is None else root)
    folder = registry.runs / run_id
    meta = read_json(folder / 'run.json')
    if meta.get('run_id') != run_id or meta.get('status') != 'complete':
        raise IntegrityError('平台只接受識別一致且狀態為 complete 的 run')
    if meta.get('level') != 'L1-H':
        raise IntegrityError('目前平台只接受 L1-H 開發 run；正式版本尚未開放')
    if meta.get('predictions_verified') is not True:
        raise IntegrityError('交接檔尚未通過完整客戶分數驗證')
    checksum = meta.get('predictions_sha256')
    if not checksum or file_hash(folder / 'predictions.csv') != checksum:
        raise IntegrityError('交接檔 SHA256 不符')
    # Registry verifies the actual model/OOF/policy hashes, every identifier,
    # final list, full-precision ranking, holdout membership and data split.
    bundle = registry.load_run(run_id)
    frame = bundle['predictions']
    if bundle['run'] != meta or file_hash(folder / 'predictions.csv') != checksum:
        raise IntegrityError('載入期間執行紀錄或交接檔發生變動')
    if frame is None or bundle['threshold'] is None:
        raise IntegrityError('缺少政策或交接資料')
    for key in ('model_id', 'policy_id', 'run_id'):
        if key not in frame or set(frame[key]) != {meta[key]}:
            raise IntegrityError(f'交接識別不符：{key}')
    if bundle['model_record']['model_id'] != meta['model_id']:
        raise IntegrityError('模型與執行紀錄不一致')
    if not set((*SIMULATION_COLUMNS, 'actual_default')) <= set(frame.columns):
        raise IntegrityError('交接檔缺少平台必要欄位')
    if not frame['actual_default'].isin([0, 1]).all():
        raise IntegrityError('歷史標籤必須為完整的二元值')
    if not np.isfinite(frame['bill_amt_latest'].to_numpy(dtype=float)).all():
        raise IntegrityError('帳單金額必須為有限數值')
    return {
        'run': meta,
        'model_record': bundle['model_record'],
        'policy': bundle['threshold'],
        'predictions': _cached_predictions(run_id, checksum, frame),
    }


def page_data(bundle, mode):
    """Return only page-safe data/metadata; simulation uses an explicit allowlist."""
    if mode not in (BACKTEST, SIMULATION):
        raise ValueError('未知平台模式')
    frame = bundle['predictions'].copy(deep=True)
    if mode == SIMULATION:
        frame = frame.loc[:, list(SIMULATION_COLUMNS)].copy()
    run, threshold = bundle['run'], bundle['policy']
    policy = threshold.get('protocol_policy', threshold.get('policy', {}))
    meta = {key: run[key] for key in ('model_id', 'run_id', 'policy_id', 'level', 'predictions_sha256')}
    meta.update(mode=mode, threshold=threshold['threshold'],
                rule=policy.get('selection', policy.get('rule', 'fixed')),
                max_list_rate=policy.get('max_list_rate'), scenario=threshold.get('scenario'))
    return frame, meta


def list_summary(frame):
    """Summarize the handoff's final list; bill coverage clips negative bills to zero."""
    selected = frame['in_list'].eq(1)
    actual = frame['actual_default'].eq(1)
    bill = frame['bill_amt_latest'].clip(lower=0)
    n_default = int(actual.sum())
    default_bill = float(bill.loc[actual].sum())
    return {
        'list_count': int(selected.sum()),
        'actual_defaults_in_list': int((selected & actual).sum()),
        'default_coverage': float((selected & actual).sum() / n_default) if n_default else None,
        'bill_coverage': float(bill.loc[selected & actual].sum() / default_bill) if default_bill else None,
    }
