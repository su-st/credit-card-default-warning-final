"""C12: synthetic/Mock batches only, no real data, credentials or LINE sends."""
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import batch_run as batch
from app import data
from notify import line_push as line
from notify.staff_digest import DEFAULT_PLATFORM_URL
from src.registry import IntegrityError


@pytest.fixture
def setup_batch(tmp_path, monkeypatch):
    frame = pd.DataFrame({key: ['x'] * 10 for key in data.SIMULATION_COLUMNS})
    frame['customer_id'] = list(range(1, 11))
    frame['risk_rank'] = list(range(1, 11))
    frame['in_list'] = [1, 1] + [0] * 8
    frame['actual_default'] = [1] * 10
    meta = dict(level='L1-H', run_id='fake-run', model_id='fake-model', policy_id='fake-policy',
                predictions_sha256='fake-hash')
    bundle = dict(run=meta, predictions=frame,
                  policy=dict(threshold=.5, protocol_policy=dict(kind='single', max_list_rate=.2)))
    loader = Mock(return_value=bundle)
    sender = Mock(return_value='已送出到測試帳號。')
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    monkeypatch.setattr(data, 'load_platform_run', loader)
    monkeypatch.setattr(line, 'SEND_ENABLED', True)
    monkeypatch.setattr(line, 'PLATFORM_URL', DEFAULT_PLATFORM_URL)
    monkeypatch.setattr(line, 'TEST_USER', 'fake-customer')
    monkeypatch.setattr(line, 'TEST_STAFF', 'fake-staff')
    monkeypatch.setattr(line, 'TOKEN', 'fake-token')
    monkeypatch.setattr(line, 'send_staff_digest', sender)
    for name in ('send_customer_care', 'push_text', 'ApiClient', 'MessagingApi'):
        monkeypatch.setattr(line, name, Mock(side_effect=AssertionError('Forbidden network/customer route')))
    monkeypatch.setattr(batch, 'ROOT', tmp_path)
    return bundle, loader, sender


def records(tmp_path):
    return list((tmp_path / 'outputs' / 'batches' / '2026-11').glob('*.json'))


def test_normal_batch_retains_handoff_and_safe_provenance(setup_batch, tmp_path):
    bundle, loader, sender = setup_batch
    original = bundle['predictions'].copy(deep=True)
    result = batch.run_batch('2026-11')
    assert result['notified'] and result['status'] == '檢查通過，已通知'
    assert result['list_count'] == 2 and result['list_rate'] == .2
    assert result['population_count'] == 10 and result['max_list_rate'] == .2
    assert result['run_id'] == 'fake-run' and result['policy_id'] == 'fake-policy'
    assert result['model_id'] == 'fake-model' and result['level'] == 'L1-H'
    assert result['executed_at'][-6] in '+-' and result['finished_at'] >= result['executed_at']
    assert all(result['checks'].values())
    loader.assert_called_once_with(root=tmp_path)
    digest = sender.call_args.args[0]
    assert digest.list_count == 2
    assert [(entry.customer_id, entry.risk_rank) for entry in digest.entries] == [(1, 1), (2, 2)]
    pd.testing.assert_frame_equal(bundle['predictions'], original)
    saved = records(tmp_path)[0].read_text()
    assert json.loads(saved) == result
    for forbidden in ('fake-token', 'fake-customer', 'fake-staff', 'actual_default', 'customer_id', 'risk_score'):
        assert forbidden not in saved
    line.send_customer_care.assert_not_called()
    line.push_text.assert_not_called()
    line.ApiClient.assert_not_called()


@pytest.mark.parametrize('failure', ['data', 'policy', 'hash', 'missing-column', 'missing-label',
                                    'empty', 'over-cap', 'bad-flag', 'bad-rank', 'no-cap', 'bad-cap', 'public-url'])
def test_abnormal_checks_never_notify(failure, setup_batch, tmp_path, monkeypatch):
    bundle, loader, sender = setup_batch
    if failure in ('data', 'policy', 'hash'):
        loader.side_effect = IntegrityError('sensitive-fake-value')
    elif failure == 'missing-column':
        bundle['predictions'] = bundle['predictions'].drop(columns='risk_rank')
    elif failure == 'missing-label':
        bundle['predictions'] = bundle['predictions'].drop(columns='in_list')
    elif failure == 'empty':
        bundle['predictions']['in_list'] = 0
    elif failure == 'over-cap':
        bundle['predictions'].loc[2, 'in_list'] = 1
    elif failure == 'bad-flag':
        bundle['predictions'].loc[0, 'in_list'] = -1
    elif failure == 'bad-rank':
        bundle['predictions'].loc[0, 'risk_rank'] = None
    elif failure == 'no-cap':
        bundle['policy']['protocol_policy'].pop('max_list_rate')
    elif failure == 'bad-cap':
        bundle['policy']['protocol_policy']['max_list_rate'] = float('nan')
    elif failure == 'public-url':
        monkeypatch.setattr(line, 'PLATFORM_URL', 'https://example.com/')
    result = batch.run_batch('2026-11', force=True)
    assert result['status'] == batch.ABNORMAL and not result['notified']
    assert not result['delivery_attempted']
    sender.assert_not_called()
    assert 'sensitive-fake-value' not in records(tmp_path)[0].read_text()


def test_sent_period_never_resends_even_with_new_run(setup_batch, tmp_path):
    bundle, _, sender = setup_batch
    first = batch.run_batch('2026-11')
    original_bytes = records(tmp_path)[0].read_bytes()
    bundle['run']['run_id'] = 'fake-new-run'
    second = batch.run_batch('2026-11')
    assert second['status'] == '已推播，略過' and second['notified']
    assert second['previous_batch_ids'] == [first['batch_id']]
    sender.assert_called_once()
    assert len(records(tmp_path)) == 2
    assert (tmp_path / 'outputs/batches/2026-11' / (first['batch_id'] + '.json')).read_bytes() == original_bytes
    batch.run_batch('2026-12')
    assert sender.call_count == 2


def test_force_records_previous_attempt_without_overwriting(setup_batch, tmp_path):
    _, _, sender = setup_batch
    first = batch.run_batch('2026-11')
    second = batch.run_batch('2026-11', force=True)
    assert second['force'] and second['forced_retry'] and second['notified']
    assert second['previous_batch_ids'] == [first['batch_id']]
    assert sender.call_count == 2 and len(records(tmp_path)) == 2


def test_simulation_never_uses_sdk_or_marks_as_notified(setup_batch, monkeypatch):
    _, _, sender = setup_batch
    from notify.line_push import _deliver
    monkeypatch.setattr(line, 'SEND_ENABLED', False)
    sender.side_effect = lambda digest: _deliver('fake-staff-preview', 'staff')
    for _ in range(2):
        result = batch.run_batch('2026-11')
        assert not result['notified'] and not result['delivery_attempted']
        assert result['status'] == '檢查通過，模擬未通知'
    assert sender.call_count == 2
    line.ApiClient.assert_not_called()
    line.MessagingApi.assert_not_called()


def test_uncertain_delivery_blocks_retry_but_force_remains_explicit(setup_batch):
    _, _, sender = setup_batch
    sender.side_effect = RuntimeError('fake-token fake-staff')
    first = batch.run_batch('2026-11')
    assert first['status'] == batch.ABNORMAL and first['delivery_attempted']
    assert not first['notified']
    second = batch.run_batch('2026-11')
    assert second['status'] == batch.ABNORMAL and sender.call_count == 1
    batch.run_batch('2026-11', force=True)
    assert sender.call_count == 2


def test_checkpoint_exists_before_send_and_interruption_never_auto_retries(setup_batch, tmp_path):
    _, _, sender = setup_batch
    def interrupt(digest):
        checkpoint = json.loads(records(tmp_path)[0].read_text())
        assert checkpoint['delivery_attempted'] and not checkpoint['notified']
        raise KeyboardInterrupt
    sender.side_effect = interrupt
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch('2026-11')
    sender.side_effect = None
    result = batch.run_batch('2026-11')
    assert result['status'] == batch.ABNORMAL and sender.call_count == 1


def test_same_period_lock_prevents_concurrent_send(setup_batch):
    _, _, sender = setup_batch
    def nested(digest):
        with pytest.raises(RuntimeError, match='人工確認'):
            batch.run_batch('2026-11', force=True)
    sender.side_effect = nested
    assert batch.run_batch('2026-11')['notified']
    sender.assert_called_once()


def test_corrupt_history_fails_closed(setup_batch, tmp_path):
    _, _, sender = setup_batch
    folder = tmp_path / 'outputs/batches/2026-11'
    folder.mkdir(parents=True)
    (folder / 'bad.json').write_text('{')
    assert batch.run_batch('2026-11', force=True)['status'] == batch.ABNORMAL
    sender.assert_not_called()


@pytest.mark.parametrize('period', ['2026-00', '2026-13', '2026-1', '../2026-11', '0000-01', '２０２６-11'])
def test_period_rejects_invalid_values_without_writes(period, setup_batch, tmp_path):
    with pytest.raises(ValueError):
        batch.run_batch(period)
    assert not (tmp_path / 'outputs').exists()
    setup_batch[1].assert_not_called()


def test_policy_capacity_uses_floor_and_top_n_rate(setup_batch):
    bundle, _, sender = setup_batch
    bundle['policy']['protocol_policy'] = dict(kind='top_n', rate=.19)
    result = batch.run_batch('2026-11')
    assert result['status'] == batch.ABNORMAL  # floor(.19 * 10) == 1
    sender.assert_not_called()


def test_explicit_uncapped_policy_is_allowed(setup_batch):
    bundle, _, _ = setup_batch
    bundle['policy']['protocol_policy']['max_list_rate'] = None
    assert batch.run_batch('2026-11')['notified']


def test_latest_batch_ui_displays_only_aggregate_fields(setup_batch):
    assert batch.latest_batch() is None
    result = batch.run_batch('2026-11')
    ui = AppTest.from_string('from app.views.risk_list import _batch_panel\n_batch_panel()').run()
    assert not ui.exception
    text = ' '.join(item.value for item in ui.markdown)
    assert '2026-11' in text and result['executed_at'] in text
    assert '名單人數：2' in text and '是否已通知：是' in text
    assert '檢查結果：檢查通過，已通知' in text
    assert not any(value in text for value in ('fake-token', 'fake-staff', 'fake-customer'))
    assert set(batch.latest_batch()) == {'period', 'executed_at', 'list_count', 'checks', 'notified',
                                       'status', 'run_id', 'model_id', 'policy_id', 'level'}


def test_ui_no_batch_or_corrupt_batch_is_safe(setup_batch, tmp_path):
    script = 'from app.views.risk_list import _batch_panel\n_batch_panel()'
    ui = AppTest.from_string(script).run()
    assert not ui.exception and any('尚無' in c.value for c in ui.caption)
    folder = tmp_path / 'outputs/batches/2026-11'
    folder.mkdir(parents=True)
    (folder / 'bad.json').write_text('{')
    ui = AppTest.from_string(script).run()
    assert not ui.exception and '人工確認' in ui.error[0].value


def test_cli_force_and_exit_status(setup_batch, capsys):
    assert batch.main(['--period', '2026-11', '--force']) == 0
    assert 'fake-token' not in capsys.readouterr().out
    setup_batch[1].side_effect = ValueError('fake-secret')
    assert batch.main(['--period', '2026-12']) == 1
    assert 'fake-secret' not in capsys.readouterr().out


def test_batch_has_no_customer_route_or_dotenv_read():
    import ast
    tree = ast.parse(Path(batch.__file__).read_text())
    forbidden = {'send_customer_care', 'push_text', 'load_dotenv', 'final_test', 'outer_splits'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            assert name not in forbidden


def test_real_platform_loader_integration(synthetic, tmp_path, monkeypatch):
    # Synthetic registry fixtures, not the local PLATFORM_RUN_ID or real data.
    from types import SimpleNamespace
    from test_registry import register, pending
    from src.registry import Registry
    from src.export import export_run

    fixture = SimpleNamespace(**vars(synthetic))
    fixture.train, fixture.hold = synthetic.train.copy(), synthetic.hold.copy()
    fixture.train.index = fixture.train.index - 100000
    fixture.hold.index = fixture.hold.index - 100000
    fixture.oof, fixture.folds = synthetic.oof.copy(), synthetic.folds.copy()
    fixture.oof['customer_id'] -= 100000
    fixture.folds.index = fixture.folds.index - 100000
    registry = Registry(tmp_path)
    record = register(registry, fixture)
    run, _ = pending(registry, record)
    export_run(run, fixture.hold)
    monkeypatch.setenv('PLATFORM_RUN_ID', run.run_id)
    monkeypatch.setattr(line, 'PLATFORM_URL', DEFAULT_PLATFORM_URL)
    monkeypatch.setattr(line, 'SEND_ENABLED', False)
    sender = Mock()
    monkeypatch.setattr(line, 'send_staff_digest', sender)
    result = batch.run_batch('2026-11', root=registry.root)
    assert result['status'] == '檢查通過，模擬未通知'
    assert result['run_id'] == run.run_id and not result['notified']
    sender.assert_called_once()
    # Changing completion blocks the next attempt through the actual loader.
    path = run.path / 'run.json'
    meta = json.loads(path.read_text())
    meta['status'] = 'failed'
    path.write_text(json.dumps(meta))
    assert batch.run_batch('2026-12', root=registry.root)['status'] == batch.ABNORMAL
    sender.assert_called_once()


def test_stale_lock_and_cli_do_not_send_or_print_details(setup_batch, tmp_path, capsys):
    folder = tmp_path / 'outputs/batches/2026-11'
    folder.mkdir(parents=True)
    (folder / '.lock').write_text('')
    assert batch.main(['--period', '2026-11', '--force']) == 1
    assert capsys.readouterr().out.strip() == batch.ABNORMAL
    setup_batch[2].assert_not_called()


def test_simulation_then_real_sends_once(setup_batch, monkeypatch):
    _, _, sender = setup_batch
    monkeypatch.setattr(line, 'SEND_ENABLED', False)
    assert not batch.run_batch('2026-11')['notified']
    monkeypatch.setattr(line, 'SEND_ENABLED', True)
    assert batch.run_batch('2026-11')['notified']
    assert batch.run_batch('2026-11')['status'] == '已推播，略過'
    assert sender.call_count == 2


def test_send_error_is_never_persisted_or_displayed(setup_batch, tmp_path):
    setup_batch[2].side_effect = RuntimeError('fake-token fake-customer fake-staff')
    batch.run_batch('2026-11')
    raw = records(tmp_path)[0].read_text()
    for value in ('fake-token', 'fake-customer', 'fake-staff'):
        assert value not in raw
    ui = AppTest.from_string('from app.views.risk_list import _batch_panel\n_batch_panel()').run()
    assert not ui.exception
    text = ' '.join(element.value for element in ui.markdown)
    assert batch.ABNORMAL in text and '是否已通知：否' in text
    assert not any(value in text for value in ('fake-token', 'fake-customer', 'fake-staff'))


def test_missing_staff_configuration_uses_existing_strict_route(setup_batch, monkeypatch):
    from notify.staff_digest import render_digest
    def strict_staff(digest):
        render_digest(digest)
        line._recipient('staff')
    setup_batch[2].side_effect = strict_staff
    monkeypatch.setattr(line, 'TEST_STAFF', '')
    monkeypatch.setattr(line, 'SEND_ENABLED', False)
    assert batch.run_batch('2026-11')['status'] == batch.ABNORMAL
    line.ApiClient.assert_not_called()


# Reuse synthetic registry fixtures, never local records or a real dataset.
from test_registry import synthetic  # noqa: E402,F401
