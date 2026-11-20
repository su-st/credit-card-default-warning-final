"""Monthly staff-only batch, using the platform's verified handoff unchanged.

Records contain aggregate counts and provenance only. Attempts are immutable once
finished. A pre-send checkpoint blocks automatic retries after uncertain delivery.
Only the module CLI loads the project dotenv before importing setting readers.
Ordinary imports never load dotenv. No customer messaging or model changes occur.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
from uuid import uuid4

# Match app/main.py without causing dotenv reads on ordinary imports. This must
# precede app.data and notify imports, which consume process settings.
if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)

from app import data
from notify import line_push
from notify.staff_digest import build_digest
from src.config import ROOT

ABNORMAL = '異常，待人工確認'


def _period(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-(0[1-9]|1[0-2])', value) or value.startswith('0000'):
        raise ValueError('期別必須為 YYYY-MM')
    return value


def _now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='microseconds')


def _save(path, record):
    # The lock serializes this period. Atomic replace keeps UI readers from
    # seeing a partially written pre-send checkpoint or final record.
    temporary = path.with_suffix('.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(record, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def _locked(folder):
    lock = folder / '.lock'
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError('本期批次執行中或上次未正常結束，請人工確認，勿自動重試。') from None
    try:
        handle.close()
        yield
    finally:
        lock.unlink()


def _history(folder):
    records = []
    for path in folder.glob('*.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if (not isinstance(record, dict) or record['period'] != folder.name
                    or not isinstance(record['executed_at'], str)
                    or type(record['notified']) is not bool
                    or type(record['delivery_attempted']) is not bool
                    or not isinstance(record['status'], str)):
                raise ValueError
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError('既有批次紀錄無法驗證，待人工確認。') from None
        records.append(record)
    return sorted(records, key=lambda record: (record['executed_at'], record['batch_id']))


def latest_batch(*, root=None):
    """Read only the latest aggregate record for the page, never credentials/data.

    Returns period, executed_at, list_count, checks, notified, status and source
    identifiers. None means no batches. Invalid records raise a generic error.
    """
    base = Path(ROOT if root is None else root) / 'outputs' / 'batches'
    records = []
    if base.exists():
        for folder in base.iterdir():
            if folder.is_dir() and re.fullmatch(r'[0-9]{4}-(0[1-9]|1[0-2])', folder.name):
                records.extend(_history(folder))
    if not records:
        return None
    latest = max(records, key=lambda record: (record['executed_at'], record['batch_id']))
    keys = ('period', 'executed_at', 'list_count', 'checks', 'notified', 'status',
            'run_id', 'model_id', 'policy_id', 'level')
    return {key: latest[key] for key in keys}


def run_batch(period, *, force=False, root=None):
    """Run one period with fixed PLATFORM_RUN_ID and staff recipient settings.

    force explicitly permits another attempt after a sent/uncertain attempt,
    never bypasses validation. Simulations are not marked notified. CLI returns
    a nonzero exit status for abnormal records. root is for isolated tests.
    """
    period = _period(period)
    base = Path(ROOT if root is None else root)
    folder = base / 'outputs' / 'batches' / period
    folder.mkdir(parents=True, exist_ok=True)
    with _locked(folder):
        batch_id = uuid4().hex
        path = folder / (batch_id + '.json')
        record = dict(batch_id=batch_id, period=period, executed_at=_now(), finished_at=None,
                      level=None, run_id=None, model_id=None, policy_id=None,
                      list_count=None, population_count=None, list_rate=None, max_list_rate=None,
                      checks={}, status=ABNORMAL, notified=False, delivery_attempted=False,
                      force=bool(force), forced_retry=False, previous_batch_ids=[])
        try:
            history = _history(folder)
            blocked = [item for item in history if item['notified'] or item['delivery_attempted']]
            record['previous_batch_ids'] = [item['batch_id'] for item in blocked]
            record['forced_retry'] = bool(force and blocked)
            # Validation is always through app/data.py and follows process
            # PLATFORM_RUN_ID. Never accept a separate model or policy argument.
            bundle = data.load_platform_run(root=base)
            frame, meta = data.page_data(bundle, data.SIMULATION)
            for key in ('level', 'run_id', 'model_id', 'policy_id'):
                record[key] = meta[key]
            record['checks']['platform_validation'] = True
            if frame[['customer_id', 'risk_rank', 'in_list', 'run_id', 'model_id', 'policy_id']].isna().any().any():
                raise ValueError('Incomplete handoff')
            digest = build_digest(frame, platform_url=line_push.PLATFORM_URL)
            count, population = digest.list_count, len(frame)
            policy = bundle['policy'].get('protocol_policy', bundle['policy'].get('policy', {}))
            cap = policy.get('rate') if policy.get('kind') == 'top_n' else policy.get('max_list_rate')
            if 'max_list_rate' not in policy and policy.get('kind') != 'top_n':
                raise ValueError('Missing policy capacity')
            if cap is not None and (isinstance(cap, bool) or not math.isfinite(cap) or not 0 <= cap <= 1):
                raise ValueError('Invalid policy capacity')
            record.update(list_count=count, population_count=population,
                          list_rate=count / population if population else None, max_list_rate=cap)
            record['checks'].update(handoff_complete=True, nonempty=count > 0,
                                    within_policy_cap=cap is None or count <= math.floor(cap * population))
            if not all(record['checks'].values()):
                raise ValueError('Batch checks failed')
            if blocked and not force:
                record['notified'] = any(item['notified'] for item in blocked)
                record['status'] = '已推播，略過' if record['notified'] else ABNORMAL
                record['checks']['duplicate_guard'] = False
            else:
                record['checks']['duplicate_guard'] = True
                # Checkpoint before touching the network. If the process dies
                # after LINE accepts but before saving success, automatic retry
                # stays blocked. --force is an explicit operator decision.
                real_send = line_push.SEND_ENABLED
                record['delivery_attempted'] = bool(real_send)
                record['status'] = ABNORMAL if real_send else '檢查通過，模擬未通知'
                _save(path, record)
                line_push.send_staff_digest(digest)
                record['notified'] = bool(real_send)
                record['status'] = '檢查通過，已通知' if real_send else '檢查通過，模擬未通知'
        except Exception:
            # Never persist exception strings, SDK payloads, accounts or tokens.
            record['status'] = ABNORMAL
            record['checks']['execution_ok'] = False
        record['finished_at'] = _now()
        _save(path, record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description='每月批次預警，僅通知測試行員，不發客戶訊息')
    parser.add_argument('--period', required=True, type=_period)
    parser.add_argument('--force', action='store_true', help='人工確認後強制重試，保留紀錄，不略過資料檢查')
    args = parser.parse_args(argv)
    try:
        record = run_batch(args.period, force=args.force)
    except (OSError, RuntimeError, ValueError):
        print(ABNORMAL)
        return 1
    print(json.dumps({key: record[key] for key in ('period', 'batch_id', 'status', 'notified')}, ensure_ascii=False))
    return int(record['status'] == ABNORMAL)


if __name__ == '__main__':
    raise SystemExit(main())
