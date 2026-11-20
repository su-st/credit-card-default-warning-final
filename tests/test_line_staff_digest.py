"""C7 acceptance (1)-(11). Fake settings and SDK mocks only, no LINE calls."""
from dataclasses import replace
import io
import os
import runpy
import warnings
from contextlib import redirect_stdout
from inspect import signature
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from llm import explain, templates
from notify import line_push as line
from notify.staff_digest import (DEFAULT_PLATFORM_URL, PREVIEW_LIMIT, DigestEntry,
                                 StaffDigest, build_digest, render_digest, validate_platform_url)

SAFE = templates.care_message({})
DIGEST = StaffDigest(2, (DigestEntry(11, 1), DigestEntry(23, 2)))


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED','1')
    monkeypatch.setattr(line,'TOKEN','fake-token')
    monkeypatch.setattr(line,'TEST_USER','fake-customer')
    monkeypatch.setattr(line,'TEST_STAFF','fake-staff')
    monkeypatch.setattr(line,'PLATFORM_URL',DEFAULT_PLATFORM_URL)
    monkeypatch.setattr(line,'SEND_ENABLED',False)
    client=MagicMock();api=Mock()
    monkeypatch.setattr(line,'ApiClient',client)
    monkeypatch.setattr(line,'MessagingApi',api)
    monkeypatch.setattr(explain,'TAIDE_MODEL','')
    return client,api


@pytest.mark.parametrize('kind',['customer','staff'])
def test_c7_1_entries_do_not_accept_recipients(kind,fake_settings):
    function=line.send_customer_care if kind=='customer' else line.send_staff_digest
    assert not {'user_id','recipient','to'} & set(signature(function).parameters)
    for argument in ('user_id','recipient','to'):
        with pytest.raises(TypeError):
            function(SAFE if kind=='customer' else DIGEST,**{argument:'fake-other'})
    fake_settings[0].assert_not_called()


@pytest.mark.parametrize('kind',['customer','staff'])
def test_c7_1_real_mode_routes_only_to_fixed_account(kind,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',True)
    if kind=='customer':line.send_customer_care(SAFE,SAFE)
    else:line.send_staff_digest(DIGEST)
    request=fake_settings[1].return_value.push_message.call_args.args[0]
    assert request.to==('fake-customer' if kind=='customer' else 'fake-staff')
    assert request.messages[0].text==(SAFE if kind=='customer' else render_digest(DIGEST))


@pytest.mark.parametrize('real',[False,True])
@pytest.mark.parametrize('kind',['customer','staff'])
@pytest.mark.parametrize('missing',['',' ',None])
def test_c7_2_missing_account_never_falls_back(kind,real,missing,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',real)
    monkeypatch.setattr(line,'TEST_USER' if kind=='customer' else 'TEST_STAFF',missing)
    with pytest.raises(ValueError,match='請設定'):
        if kind=='customer':line.send_customer_care(SAFE,SAFE)
        else:line.send_staff_digest(DIGEST)
    fake_settings[0].assert_not_called()


def test_c7_3_staff_bypasses_care_check_and_customer_rejects_digest(monkeypatch):
    monkeypatch.setattr(explain,'check_care',Mock(side_effect=AssertionError('staff entered care policy')))
    assert '模擬發送' in line.send_staff_digest(DIGEST)
    with pytest.raises(ValueError):line.send_customer_care(DIGEST,render_digest(DIGEST))


@pytest.mark.parametrize('real',[False,True])
@pytest.mark.parametrize('approval',[None,'',SAFE+' changed'])
def test_c7_4_customer_needs_exact_approval_in_both_modes(real,approval,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',real)
    with pytest.raises(PermissionError):line.send_customer_care(SAFE,approval)
    fake_settings[0].assert_not_called()


@pytest.mark.parametrize('kind',['customer','staff'])
def test_c7_5_simulation_never_constructs_client_with_fake_credentials(kind,fake_settings):
    result=line.send_customer_care(SAFE,SAFE) if kind=='customer' else line.send_staff_digest(DIGEST)
    assert '模擬發送' in result
    fake_settings[0].assert_not_called();fake_settings[1].assert_not_called()


@pytest.mark.parametrize('new,legacy,expected',[
    ('','fake-legacy','fake-legacy'),('fake-new','fake-legacy','fake-new'),('fake-new','','fake-new')])
def test_c7_6_legacy_name_warns_without_printing_settings(new,legacy,expected):
    settings=dict(LINE_TEST_CUSTOMER_ID=new,LINE_TEST_USER_ID=legacy,LINE_TEST_STAFF_ID='fake-staff',
                  LINE_CHANNEL_ACCESS_TOKEN='fake-token',LINE_SEND_ENABLED='false')
    with patch.dict(os.environ,settings),warnings.catch_warnings(),redirect_stdout(io.StringIO()) as out:
        warnings.filterwarnings('ignore',message="'notify.line_push' found in sys.modules",category=RuntimeWarning)
        module=runpy.run_module('notify.line_push',run_name='c7_settings_probe')
    assert module['TEST_USER']==expected
    assert ('LINE_TEST_CUSTOMER_ID' in out.getvalue())==bool(legacy)
    for value in ('fake-legacy','fake-new','fake-staff','fake-token'):assert value not in out.getvalue()


@pytest.mark.parametrize('real',[False,True])
@pytest.mark.parametrize('message',['','請於明天付款','核准免息分期','您好，歡迎與我們聯繫。',render_digest(DIGEST)])
def test_c7_8_customer_forbidden_content_is_rejected(real,message,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',real)
    with pytest.raises(ValueError):line.send_customer_care(message,message)
    fake_settings[0].assert_not_called()


@pytest.mark.parametrize('real',[False,True])
@pytest.mark.parametrize('bad',[SAFE,render_digest(DIGEST),{},
    StaffDigest(1,(DigestEntry('fake-user',1),)),
    StaffDigest(1,(DigestEntry(11,'fake-phone'),)),
    StaffDigest(1,(DigestEntry(11,1),),'https://public.example.com/'),
    StaffDigest(1,(DigestEntry(11,1),),'http://localhost:8501/?email=private'),
    StaffDigest(1,(DigestEntry(11,1),DigestEntry(23,2))),
    StaffDigest(2,(DigestEntry(11,2),DigestEntry(11,1)))])
def test_c7_8_staff_forbidden_content_is_rejected(real,bad,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',real)
    with pytest.raises((ValueError,TypeError)):line.send_staff_digest(bad)
    fake_settings[0].assert_not_called()


@pytest.mark.parametrize('module',['notify.line_push','notify.staff_digest'])
def test_c7_8_cli_never_sends_even_when_enabled(module):
    settings=dict(LINE_TEST_CUSTOMER_ID='fake-customer',LINE_TEST_STAFF_ID='fake-staff',
                  LINE_TEST_USER_ID='',LINE_CHANNEL_ACCESS_TOKEN='fake-token',LINE_SEND_ENABLED='true')
    with patch.dict(os.environ,settings),patch('linebot.v3.messaging.ApiClient') as client, \
            patch('linebot.v3.messaging.MessagingApi') as api,warnings.catch_warnings(),redirect_stdout(io.StringIO()) as out:
        warnings.filterwarnings('ignore',message="'notify.*' found in sys.modules",category=RuntimeWarning)
        runpy.run_module(module,run_name='__main__')
    assert 'CLI 不提供發送功能' in out.getvalue()
    client.assert_not_called();api.assert_not_called()
    for value in ('fake-customer','fake-staff','fake-token'):assert value not in out.getvalue()


@pytest.mark.parametrize('message',[DIGEST,render_digest(DIGEST)])
def test_c7_9_legacy_customer_entry_rejects_staff_message(message,fake_settings):
    with pytest.raises((ValueError,TypeError)):line.push_text(message,approved_text=render_digest(DIGEST))
    fake_settings[0].assert_not_called()


@pytest.mark.parametrize('url',[
    'http://localhost:8501/','http://127.0.0.1:8501','http://[::1]:8501/',
    'http://192.168.1.8:8501/','https://10.1.2.3/','https://172.16.0.5/',
    'https://[fd12::1]/','https://platform.test/'])
def test_c7_11_local_and_test_links_only(url):
    assert validate_platform_url(url).endswith('/')
    assert url.split('?')[0].rstrip('/') in render_digest(replace(DIGEST,platform_url=url))


@pytest.mark.parametrize('url',[
    'https://example.com/','https://platform.test.evil.com/','https://localhost.evil.com/',
    'http://8.8.8.8/','http://0.0.0.0/','http://169.254.169.254/','http://224.0.0.1/',
    'https://user:secret@localhost/','//localhost:8501/','javascript:alert(1)',
    'file:///tmp/result','http://localhost:8501/?next=https://evil.com',
    'http://localhost:8501/#private','http://localhost:8501/customer/private',
    'http://localhost:8501/\nextra','http://localhost\\@evil.com/','http://localhost:99999/',
    'http://localhost:0/','http://%6cocalhost/'])
def test_c7_11_public_or_injected_links_are_rejected(url):
    with pytest.raises(ValueError):validate_platform_url(url)


def test_builder_uses_only_policy_list_and_rank_never_scores_or_labels():
    frame=pd.DataFrame([dict(customer_id=i,risk_rank=21-i,in_list=int(i!=3),
        risk_score=.8,actual_default=1,factor_1='not sent',name='not sent',phone='not sent')
        for i in range(1,21)])
    digest=build_digest(frame)
    assert digest.list_count==19 and len(digest.entries)==PREVIEW_LIMIT
    assert [entry.risk_rank for entry in digest.entries]==sorted(frame.loc[frame.in_list.eq(1),'risk_rank'])[:PREVIEW_LIMIT]
    assert all(entry.customer_id!=3 for entry in digest.entries)
    text=render_digest(digest)
    assert 'not sent' not in text and '0.8' not in text and 'actual_default' not in text
    assert '預警名單人數：19' in text


@pytest.mark.parametrize('kind',['customer','staff'])
def test_sdk_errors_cannot_leak_credentials_or_recipient(kind,monkeypatch,fake_settings,capsys):
    monkeypatch.setattr(line,'SEND_ENABLED',True)
    fake_settings[1].return_value.push_message.side_effect=RuntimeError('fake-token fake-customer fake-staff')
    with pytest.raises(RuntimeError) as error:
        if kind=='customer':line.send_customer_care(SAFE,SAFE)
        else:line.send_staff_digest(DIGEST)
    captured=capsys.readouterr()
    for value in ('fake-token','fake-customer','fake-staff'):
        assert value not in str(error.value)+captured.out+captured.err
    assert error.value.__suppress_context__


RISK_APP = '''
import pandas as pd
from app.views.risk_list import render
render(pd.DataFrame([
 dict(customer_id=i, risk_rank=i, in_list=int(i<=12), risk_score=.75,
      factor_1='not sent', factor_2='', factor_3='', limit_bal=100)
 for i in range(1,15)
]), {'mode':'新資料預警（模擬）'})
'''


@pytest.mark.parametrize('same',[False,True])
@pytest.mark.parametrize('real',[False,True])
def test_c7_10_platform_labels_same_account_and_sends_exact_preview(same,real,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',real)
    if same:monkeypatch.setattr(line,'TEST_USER','fake-staff')
    monkeypatch.setattr(explain,'check_care',Mock(side_effect=AssertionError('staff called care checker')))
    ui=AppTest.from_string(RISK_APP,default_timeout=20).run()
    assert not ui.exception
    assert any('同一測試帳號模擬' in item.value for item in ui.warning)==same
    assert not ui.checkbox
    preview=ui.code[0].value
    assert '預警名單人數：12' in preview and 'not sent' not in preview
    ui.slider(key='browse_count').set_value(2).run()
    assert ui.code[0].value==preview
    ui.button(key='staff_digest_send').click().run()
    assert not ui.exception
    if real:
        request=fake_settings[1].return_value.push_message.call_args.args[0]
        assert request.to=='fake-staff' and request.messages[0].text==preview
        assert ui.success
    else:
        fake_settings[0].assert_not_called()
        assert any('模擬發送' in item.value for item in ui.info)
    shown=' '.join(str(item.value) for items in (ui.caption,ui.warning,ui.info,ui.error,ui.success,ui.code) for item in items)
    for value in ('fake-token','fake-staff','fake-customer'):assert value not in shown


def test_platform_missing_staff_account_is_preview_only(monkeypatch,fake_settings):
    monkeypatch.setattr(line,'TEST_STAFF','')
    ui=AppTest.from_string(RISK_APP,default_timeout=20).run()
    assert not ui.exception and ui.code
    assert ui.button(key='staff_digest_send').disabled
    fake_settings[0].assert_not_called()


def test_platform_staff_failure_is_generic_without_credentials(monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',True)
    fake_settings[1].return_value.push_message.side_effect=RuntimeError('fake-token fake-staff')
    ui=AppTest.from_string(RISK_APP,default_timeout=20).run()
    ui.button(key='staff_digest_send').click().run()
    assert not ui.exception and ui.error
    assert 'fake-token' not in ui.error[0].value and 'fake-staff' not in ui.error[0].value


def test_no_care_approval_is_retained_in_api_between_attempts(monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',True)
    line.send_customer_care(SAFE,SAFE)
    assert fake_settings[1].return_value.push_message.call_count==1
    with pytest.raises(PermissionError):line.send_customer_care(SAFE)
    assert fake_settings[1].return_value.push_message.call_count==1
    line.send_customer_care(SAFE,SAFE)
    assert fake_settings[1].return_value.push_message.call_count==2


@pytest.mark.parametrize('frame',[
    pd.DataFrame([dict(customer_id=1,risk_rank=float('inf'),in_list=1)]),
    pd.DataFrame([dict(customer_id=1,risk_rank=1,in_list=None)]),
    pd.DataFrame([dict(customer_id='private',risk_rank=1,in_list=1)]),
    pd.DataFrame([dict(customer_id=1,risk_rank=1,in_list=1)]*2),
])
def test_malformed_handoff_cannot_become_a_digest(frame):
    with pytest.raises(ValueError):build_digest(frame)


def test_empty_policy_list_is_supported():
    frame=pd.DataFrame([dict(customer_id=1,risk_rank=1,in_list=0)])
    digest=build_digest(frame)
    assert digest.list_count==0 and digest.entries==()
    assert '預警名單人數：0' in render_digest(digest)


CUSTOMER_APP = '''
import pandas as pd
from app.views.customer import render
render(pd.DataFrame([
 dict(customer_id=i, risk_score=.8, risk_rank=i, in_list=1,
      factor_1='9 月延遲繳款 2 個月', factor_2='', factor_3='')
 for i in [1,2]
]), {})
'''


@pytest.mark.parametrize('real',[False,True])
def test_customer_page_only_calls_strict_entry_and_discards_approval(real,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',real)
    legacy=Mock(side_effect=AssertionError('platform called compatibility route'))
    strict=Mock(wraps=line.send_customer_care)
    monkeypatch.setattr(line,'push_text',legacy)
    monkeypatch.setattr(line,'send_customer_care',strict)
    ui=AppTest.from_string(CUSTOMER_APP,default_timeout=20).run()
    ui.button[1].click().run()
    assert ui.button[2].disabled
    ui.checkbox[0].check().run()
    ui.button[2].click().run()
    assert not ui.exception and not ui.error
    strict.assert_called_once_with(SAFE,approved_text=SAFE)
    legacy.assert_not_called()
    assert not ui.checkbox[0].value and ui.button[2].disabled
    ui.run()
    assert strict.call_count==1
    ui.checkbox[0].check().run()
    ui.button[2].click().run()
    assert strict.call_count==2
    assert not ui.checkbox[0].value and ui.button[2].disabled
    legacy.assert_not_called()


def test_legacy_dry_run_is_explicit_and_cannot_send_without_approval(monkeypatch,fake_settings):
    result=line.push_text(SAFE)
    assert '相容模擬發送' in result and '沒有真的送出' in result
    fake_settings[0].assert_not_called()
    monkeypatch.setattr(line,'SEND_ENABLED',True)
    with pytest.raises(PermissionError):line.push_text(SAFE)
    fake_settings[0].assert_not_called()


@pytest.mark.parametrize('real',[False,True])
def test_legacy_approved_calls_delegate_to_strict_customer_entry(real,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'SEND_ENABLED',real)
    strict=Mock(wraps=line.send_customer_care)
    monkeypatch.setattr(line,'send_customer_care',strict)
    line.push_text(SAFE,approved_text=SAFE)
    strict.assert_called_once_with(SAFE,approved_text=SAFE)


def test_legacy_dry_run_still_rejects_missing_account_and_invalid_content(monkeypatch,fake_settings):
    with pytest.raises(ValueError):line.push_text(render_digest(DIGEST))
    with pytest.raises(ValueError):line.push_text(DIGEST)
    monkeypatch.setattr(line,'TEST_USER','')
    with pytest.raises(ValueError):line.push_text(SAFE)
    fake_settings[0].assert_not_called()


@pytest.mark.parametrize('real',[False,True])
@pytest.mark.parametrize('kind',['customer','staff'])
def test_missing_token_never_opens_a_client(kind,real,monkeypatch,fake_settings):
    monkeypatch.setattr(line,'TOKEN','')
    monkeypatch.setattr(line,'SEND_ENABLED',real)
    def call():
        return line.send_customer_care(SAFE,SAFE) if kind=='customer' else line.send_staff_digest(DIGEST)
    if real:
        with pytest.raises(ValueError,match='設定不完整'):call()
    else:assert '模擬發送' in call()
    fake_settings[0].assert_not_called()


def test_overlong_test_hostname_is_rejected():
    with pytest.raises(ValueError):validate_platform_url('http://'+'.'.join(['a'*60]*5)+'.test/')
