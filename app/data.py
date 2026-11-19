"""Verified run-backed platform data; no legacy/global handoff fallback."""
import hashlib
import json
import os
from io import BytesIO

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
        raise IntegrityError('目前平台只接受 L1-H 開發 run，正式版本尚未開放')
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


FAIRNESS_COMPARISONS = ('共同門檻0.5', '固定名單前20%', '正式政策')
FAIRNESS_PARTITIONS = ('性別', '年齡層', '性別×年齡層', '教育程度', '婚姻狀況')
FAIRNESS_RULES_ID = 'g2a-20261116'


def _fairness_bytes(registry, run_id, filename):
    path = registry.results / run_id / filename
    if not path.resolve().is_relative_to(registry.results.resolve()):
        raise IntegrityError('公平性結果路徑超出 results 目錄')
    try:
        return path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f'缺少或無法讀取公平性結果檔：{run_id}/{filename}') from exc


def _fairness_meta(registry, run_id):
    raw = _fairness_bytes(registry, run_id, 'run.json')
    try:
        meta = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise IntegrityError('公平性 run.json 格式錯誤') from exc
    if (not isinstance(meta, dict) or meta.get('run_id') != run_id
            or meta.get('status') != 'complete' or meta.get('git_dirty') is not False
            or meta.get('level') not in ('L1', 'L2')
            or meta.get('rules_freeze_id') != FAIRNESS_RULES_ID):
        raise IntegrityError('公平性 run 必須完成、來源一致、git_dirty=false，且為指定規則的 L1 或 L2')
    try:
        identifier(meta['model_id'])
    except (KeyError, ValueError, TypeError) as exc:
        raise IntegrityError('公平性 run 缺少有效 model_id') from exc
    return meta, raw


def _fairness_tables(registry, meta, label):
    from src.protocol import GROUP_COLUMNS, GAP_COLUMNS, OVERALL_COLUMNS

    ci_columns = {'指標', '估計值', '下限', '上限', '有效重抽', '無效重抽', '可靠', '比較', '分組方式'}
    schemas = {'overall': set(OVERALL_COLUMNS), 'groups': set(GROUP_COLUMNS),
               'gaps_point': set(GAP_COLUMNS), 'gaps': ci_columns}
    tables, checksums = {}, {}
    for name, columns in schemas.items():
        filename = 'fairness_' + name + '.csv'
        raw = _fairness_bytes(registry, meta['run_id'], filename)
        try:
            table = pd.read_csv(BytesIO(raw), keep_default_na=True)
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            raise IntegrityError(f'公平性 CSV 格式錯誤：{filename}') from exc
        required = columns | {'模型', '政策比較', 'run_id', 'model_id'}
        if table.empty or not required <= set(table):
            raise IntegrityError(f'公平性結果缺少資料或必要欄位：{filename}')
        if any(not table[key].eq(meta[key]).all() for key in ('run_id', 'model_id')):
            raise IntegrityError(f'公平性結果識別不符：{filename}')
        allowed = {label, 'M1-NS 減 M1'} if label == 'M1-NS' and name == 'gaps' else {label}
        if not set(table['模型']) <= allowed or label not in set(table['模型']):
            raise IntegrityError(f'公平性模型角色不符：{filename}')
        if set(table['政策比較']) != set(FAIRNESS_COMPARISONS):
            raise IntegrityError(f'公平性比較名稱不完整或無法識別：{filename}')
        tables[name] = table
        checksums[filename] = hashlib.sha256(raw).hexdigest()
    return tables, checksums


def _fairness_structure(tables, label):
    expected = {(c, g) for c in FAIRNESS_COMPARISONS for g in FAIRNESS_PARTITIONS}
    for name, frame in tables.items():
        own = frame[frame['模型'].eq(label)]
        keys = ['政策比較'] if name == 'overall' else ['政策比較', '分組方式']
        unique = keys + (['族群'] if name == 'groups' else ['指標'] if name == 'gaps' else [])
        if own.duplicated(unique).any() or own[unique].isna().any().any():
            raise IntegrityError('公平性結果有重複或空白比較項目')
        if name == 'overall':
            if len(own) != len(FAIRNESS_COMPARISONS):
                raise IntegrityError('公平性整體比較不完整')
        elif set(zip(own['政策比較'], own['分組方式'])) != expected:
            raise IntegrityError('公平性分組比較不完整')
        if name == 'gaps':
            _fairness_ci_structure(own, expected, '單一名單')


def _fairness_ci_structure(frame, expected, direction):
    keys = ['政策比較', '分組方式', '指標']
    desired = {(c, g, metric) for c, g in expected for metric in ('EO 差距', 'FPR 差距')}
    if (frame.empty or frame.duplicated(keys).any()
            or set(frame[keys].itertuples(index=False, name=None)) != desired
            or not frame['比較'].eq(direction).all()):
        raise IntegrityError('公平性信賴區間或成對比較不完整，或比較方向不符')


def load_fairness_comparison(run_id=None, *, root=None):
    """讀取 F3 已存檔的 M1 與 M1-NS 比較，不重算指標或載入模型及客戶資料。

    參數：run_id 是 M1-NS 的 F3 run_id，省略時讀取環境設定
    PLATFORM_FAIRNESS_RUN_ID。root 是選用的專案根目錄，預設為 ROOT。
    所有檔案來自 results/<run_id>/，格式依 06_fairness_groups 的雙 run 輸出。
    由 M1-NS fairness_gaps.csv 的成對列 m1_run_id 找 M1，任何驗證失敗
    會拋出 IntegrityError 或 ValueError，不回傳部分數字。呼叫頁面應顯示錯誤
    後停止該比較區塊。缺少本機模型工件不影響讀取既有彙總表。

    回傳 dict：
      models['M1' 或 'M1-NS'] = {
        'overall': DataFrame，三種政策的整體表，
        'groups': DataFrame，族群表，
        'gaps_point': DataFrame，差距點估計，
        'gaps_ci': DataFrame，僅該模型自身的差距信賴區間，
        'source': dict，含 level/run_id/model_id/rules_freeze_id/comparison_names/
                  files_sha256（本次讀取檔案的 SHA256，非發送政策）。}
      paired: DataFrame，M1-NS 減 M1 的成對比較，保留 m1_run_id/m1_model_id。
      source: dict，含 level/rules_freeze_id/m1_run_id/m1_ns_run_id/
              comparison_names/comparison_direction。
    各 DataFrame 原樣保留 CSV 欄位及缺失值，不四捨五入、不重算或補造指標。
    兩個模型必須同層級，只接受 L1 或 L2。此介面與 PLATFORM_RUN_ID 獨立，
    不改變 load_platform_run 僅接受 L1-H 的限制，也不讀取 predictions.csv。

    最短範例：bundle = load_fairness_comparison('已完成的M1-NS比較run_id')
    族群表：bundle['models']['M1-NS']['groups']，成對表：bundle['paired']。
    """
    run_id = run_id if run_id is not None else os.getenv('PLATFORM_FAIRNESS_RUN_ID', '')
    if not run_id:
        raise ValueError('請設定 PLATFORM_FAIRNESS_RUN_ID，填入 M1-NS 的 F3 比較 run_id。')
    run_id = identifier(run_id)
    registry = Registry(ROOT if root is None else root)
    ns_meta, ns_raw = _fairness_meta(registry, run_id)
    ns_tables, ns_hashes = _fairness_tables(registry, ns_meta, 'M1-NS')
    paired = ns_tables['gaps'].loc[ns_tables['gaps']['模型'].eq('M1-NS 減 M1')].copy()
    if not {'m1_run_id', 'm1_model_id'} <= set(paired) or paired.empty:
        raise IntegrityError('公平性結果缺少成對比較來源')
    if paired[['m1_run_id', 'm1_model_id']].isna().any().any() or paired['m1_run_id'].nunique() != 1:
        raise IntegrityError('成對比較的 M1 來源不一致')
    m1_id = identifier(paired['m1_run_id'].iloc[0])
    if m1_id == run_id:
        raise IntegrityError('公平性比較必須來自兩個不同的 run')
    m1_meta, m1_raw = _fairness_meta(registry, m1_id)
    if (m1_meta['level'] != ns_meta['level'] or m1_meta['model_id'] == ns_meta['model_id']
            or not paired['m1_model_id'].eq(m1_meta['model_id']).all()):
        raise IntegrityError('公平性成對比較的資料層級或 M1 模型來源不符')
    m1_tables, m1_hashes = _fairness_tables(registry, m1_meta, 'M1')
    _fairness_structure(m1_tables, 'M1')
    _fairness_structure(ns_tables, 'M1-NS')
    group_keys = ['政策比較', '分組方式', '族群']
    if (set(m1_tables['groups'][group_keys].itertuples(index=False, name=None)) !=
            set(ns_tables['groups'][group_keys].itertuples(index=False, name=None))):
        raise IntegrityError('兩模型的公平性族群比較項目不一致')
    expected = {(c, g) for c in FAIRNESS_COMPARISONS for g in FAIRNESS_PARTITIONS}
    _fairness_ci_structure(paired, expected, 'B-A')
    models = {}
    for label, meta, raw, tables, checksums in (
            ('M1', m1_meta, m1_raw, m1_tables, m1_hashes),
            ('M1-NS', ns_meta, ns_raw, ns_tables, ns_hashes)):
        if _fairness_bytes(registry, meta['run_id'], 'run.json') != raw:
            raise IntegrityError('載入期間公平性執行紀錄發生變動')
        if any(hashlib.sha256(_fairness_bytes(registry, meta['run_id'], name)).hexdigest() != digest
               for name, digest in checksums.items()):
            raise IntegrityError('載入期間公平性結果檔發生變動')
        models[label] = {name: tables[name].copy() for name in ('overall', 'groups', 'gaps_point')}
        models[label]['gaps_ci'] = tables['gaps'].loc[tables['gaps']['模型'].eq(label)].copy()
        models[label]['source'] = {key: meta[key] for key in ('level', 'run_id', 'model_id', 'rules_freeze_id')}
        models[label]['source'].update(comparison_names=list(FAIRNESS_COMPARISONS), files_sha256=checksums)
    return {'models': models, 'paired': paired,
            'source': dict(level=ns_meta['level'], rules_freeze_id=FAIRNESS_RULES_ID,
                           m1_run_id=m1_id, m1_ns_run_id=run_id,
                           comparison_names=list(FAIRNESS_COMPARISONS), comparison_direction='M1-NS 減 M1')}
