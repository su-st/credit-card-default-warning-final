"""C5: fake model responses only; quality is separate from send authorization."""
import ast
from copy import deepcopy
from pathlib import Path
import hashlib
from unittest.mock import Mock

import pytest

from llm import explain, templates
from notify import line_push

DRAFT = '您好，提醒您留意帳單資訊，若有需要歡迎與客服聯繫。'
CUSTOMER = dict(customer_id=1, risk_score=.1, risk_rank=12, in_list=1,
                factor_1='9 月延遲繳款 2 個月', factor_2='', factor_3='')


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    request=Mock(side_effect=AssertionError('Network forbidden'))
    client=Mock(side_effect=AssertionError('LINE forbidden'))
    monkeypatch.setattr(explain.requests,'post',request)
    monkeypatch.setattr(line_push,'ApiClient',client)
    monkeypatch.setattr(explain,'TAIDE_MODEL','')
    yield
    request.assert_not_called()
    client.assert_not_called()


def test_non_template_quality_passes_but_send_policy_and_push_reject(monkeypatch):
    assert explain.quality_check('care',DRAFT,'')==''
    assert explain.check_care(DRAFT)
    monkeypatch.setattr(line_push,'SEND_ENABLED',True)
    with pytest.raises(ValueError,match='關懷訊息未通過檢查'):
        line_push.push_text(DRAFT,approved_text=DRAFT)


@pytest.mark.parametrize('kind,text,facts,reason', [
    ('explain','','','沒有輸出'),('care','  ','','沒有輸出'),
    ('explain','字'*121,'','超過字數'),('care','字'*121,'','超過字數'),
    ('explain','風險分數0.2','風險分數0.1','資料以外的數字'),
    ('care','提醒您於１２日處理。','','具體日期或期限'),
    ('care','目前記錄為１２。','','出現數字'),
    ('care','提醒您的風險分數。','','出現禁用字'),
])
def test_quality_rejection_reasons(kind,text,facts,reason):
    assert reason in explain.quality_check(kind,text,facts)


def test_quality_accepts_only_numbers_in_facts_and_compatibility_checker():
    facts='風險分數0.20，信用額度50,000元。'
    # Existing numeric normalizer supports fullwidth digits and decimal points.
    text='風險分數０．２０，信用額度50,000元。'
    assert explain.quality_check('explain',text,facts)==''
    assert explain.check_explain(text,facts)==''
    with pytest.raises(ValueError):explain.quality_check('unknown',DRAFT,'')


@pytest.mark.parametrize('score',[0,.01,.2,.49,.5,.69,.7,1])
def test_selected_customers_never_get_fixed_risk_grades(score):
    customer={**CUSTOMER,'risk_score':score}
    for text in [templates.risk_explanation(customer),explain._facts(customer),explain.generate('explain',customer)['text']]:
        assert '列入本期預警名單' in text and '第 12 名' in text
        assert '風險因素' in text and '模型判斷依據，不代表違約原因' in text
        assert all(word not in text for word in ['風險低','風險中','風險為低','風險為中','風險等級','主要原因','違約機率'])


def test_outside_list_and_optional_list_size_are_not_inferred_from_score():
    outside={**CUSTOMER,'risk_score':.99,'in_list':0}
    assert '未列入本期預警名單' in explain._facts(outside)
    assert '未列入本期預警名單' in templates.risk_explanation(outside)
    assert '名單共' not in templates.risk_explanation(CUSTOMER)
    assert '名單共 100 人' in templates.risk_explanation({**CUSTOMER,'list_count':100})


@pytest.mark.parametrize('key',['in_list','risk_rank'])
def test_missing_policy_fields_raise_without_grade_fallback(key):
    customer=deepcopy(CUSTOMER);del customer[key]
    for fn in [templates.risk_explanation,explain._facts,lambda c:explain.generate('explain',c)]:
        with pytest.raises(KeyError):fn(customer)


@pytest.mark.parametrize('field,value',[('in_list',2),('risk_rank',0),('risk_rank',1.5),('risk_rank',True)])
def test_invalid_policy_fields_raise(field,value):
    with pytest.raises(ValueError):templates.risk_explanation({**CUSTOMER,field:value})


def function_text(source,name):
    tree=ast.parse(source)
    return ast.get_source_segment(source,next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))


def test_send_whitelist_and_old_experiment_prompts_are_unchanged():
    root=Path(__file__).parents[1]
    # Source-text hashes from main c7a465c; stable across Python versions and shallow CI history.
    expected={'check_care': '62df4b2a62e3289d75d6bdda9941208ce45e88d13284b6f0d7d7dc58e6bc9e02', 'care_message': '89fc1329f25cd6672ac34e6b8b4b7bc99e81411c26cbeacfb20be6e1ca45fa34', 'old_ask': 'bdbda67d13cc7eca5bda1f12ccf8f2f3f3c158fd1cec2f204d31307df4cb742c', 'old_explain': '09d3ed618bc2bb9dc9a7641051a24fb82957f5e6d03f94b5e50efcdbd09e75c0', 'old_care': '7fb0eb7ea3710cce63f46449e68e920cdd03d49fe6cbc7cd798980b57f3e745c'}
    for path,names in [('llm/explain.py',['check_care']),('llm/templates.py',['care_message']),
                       ('experiments/10_taide_eval.py',['old_ask','old_explain','old_care'])]:
        source=(root/path).read_text()
        for name in names:
            assert hashlib.sha256(function_text(source,name).encode()).hexdigest()==expected[name]
    source=(root/'experiments/10_taide_eval.py').read_text()
    def old_rules(text):
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OLD_RULES' for t in n.targets))
        return ast.literal_eval(node.value)
    assert old_rules(source)=='請用繁體中文、口語且客氣的語氣回答，不超過 80 字。只能根據下面提供的資料說明，不可以自行編造數字或原因。'
    tree=ast.parse(source)
    checks=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='quality_check']
    assert len(checks)==4
    assert {ast.unparse(n.args[1]) for n in checks}=={'old_e','old_m',"new_e['raw']","new_m['raw']"}
    assert '"可否直接發送": explain.check_care(new_m["raw"]) == ""' in source
    assert 'out.to_csv(RESULTS / "10_taide_eval.csv"' in source


def test_generate_shows_quality_draft_without_authorizing_send(monkeypatch):
    monkeypatch.setattr(explain,'TAIDE_MODEL','fake')
    chat=Mock(return_value=DRAFT)
    monkeypatch.setattr(explain,'_chat',chat)
    result=explain.generate('care',CUSTOMER)
    assert result=={'text':DRAFT,'how':'TAIDE','raw':DRAFT,'reason':''}
    assert explain.check_care(result['text'])
    for real_mode in [False,True]:
        monkeypatch.setattr(line_push,'SEND_ENABLED',real_mode)
        with pytest.raises(ValueError):line_push.push_text(result['text'],approved_text=result['text'])
    chat.assert_called_once()


@pytest.mark.parametrize('text',[
    '請於明天中午以前繳款，否則會凍結帳戶。','本月底前付款。',
    '請致電客服專線零八零零。','點選 https://example.com 辦理。',
    '請前往 WWW.EXAMPLE.COM 辦理。','保證免息並核准分期。',
    templates.care_message({})+'保證不收費。',
])
def test_approved_quality_extensions_preserve_p0_generation_fallback(text,monkeypatch):
    monkeypatch.setattr(explain,'TAIDE_MODEL','fake')
    monkeypatch.setattr(explain,'_chat',Mock(return_value=text))
    assert explain.quality_check('care',text,'')
    result=explain.generate('care',CUSTOMER)
    assert result['text']==templates.care_message({})
    assert result['raw']==text and result['reason']
    assert '固定句型' in result['how']


@pytest.mark.parametrize('score',[0,.2,.5,.7,1])
@pytest.mark.parametrize('raw',['風險低。','違約風險為中。','風險等級為低。','違約機率偏低。','主要原因是帳單。'])
def test_fake_model_fixed_grades_or_causal_wording_cannot_be_displayed(score,raw,monkeypatch):
    customer={**CUSTOMER,'risk_score':score}
    monkeypatch.setattr(explain,'TAIDE_MODEL','fake')
    monkeypatch.setattr(explain,'_chat',Mock(return_value=raw))
    result=explain.generate('explain',customer)
    assert result['text']==templates.risk_explanation(customer)
    assert result['raw']==raw and result['reason']
    assert '列入本期預警名單' in result['text']


def test_staff_draft_has_noncausal_note_but_keeps_research_raw(monkeypatch):
    raw='列入本期預警名單，風險因素為9 月延遲繳款 2 個月。'
    monkeypatch.setattr(explain,'TAIDE_MODEL','fake')
    monkeypatch.setattr(explain,'_chat',Mock(return_value=raw))
    result=explain.generate('explain',CUSTOMER)
    assert result['how']=='TAIDE' and result['reason']==''
    assert result['raw']==raw
    assert explain.FACTOR_NOTE in result['text']


def test_model_failure_uses_policy_aware_fallback_without_network(monkeypatch):
    monkeypatch.setattr(explain,'TAIDE_MODEL','fake')
    monkeypatch.setattr(explain,'_chat',Mock(side_effect=RuntimeError('synthetic offline')))
    for kind in ['care','explain']:
        result=explain.generate(kind,CUSTOMER)
        assert result['raw']=='' and result['reason']=='synthetic offline'
        assert '固定句型' in result['how']
    with pytest.raises(ValueError):explain.generate('invalid',CUSTOMER)


def test_added_staff_note_does_not_bypass_draft_length_limit(monkeypatch):
    raw='說明'*(explain.MAX_CHARS//2)
    assert explain.quality_check('explain',raw,explain._facts(CUSTOMER))==''
    monkeypatch.setattr(explain,'TAIDE_MODEL','fake')
    monkeypatch.setattr(explain,'_chat',Mock(return_value=raw))
    result=explain.generate('explain',CUSTOMER)
    assert result['text']==templates.risk_explanation(CUSTOMER)
    assert '超過字數' in result['reason']
    assert result['raw']==raw


@pytest.mark.parametrize('category,text', [
    ('具體日期或期限','請於星期六以前完成安排。'),
    ('具體日期或期限','請在下個月初辦理。'),
    ('具體日期或期限','請在三個工作天內處理。'),
    ('具體日期或期限','請於７ 月底前處理。'),
    ('具體日期或期限','請於十二月處理。'),
    ('具體日期或期限','請於X 月處理。'),
    ('聯絡方式','若需協助，請寄信至 care@example.org。'),
    ('聯絡方式','請使用電子郵件：ｃａｒｅ＠ｅｘａｍｐｌｅ．ｏｒｇ。'),
    ('聯絡方式','請瀏覽 help.example.tw/service。'),
    ('聯絡方式','服務電話：零九一二三四五六七八。'),
    ('聯絡方式','請撥打 +886 (2) 2345-6789。'),
    ('承諾或保證','這項服務完全免費。'),
    ('承諾或保證','這項安排不會收取任何手續費。'),
    ('承諾或保證','申請必定通過。'),
    ('威脅或後果','後續將交由催收單位跟進。'),
    ('威脅或後果','我們將採取法律途徑。'),
    ('威脅或後果','後續將停用信用卡。'),
])
def test_category_rules_reject_new_phrasings_and_generation(category,text,monkeypatch):
    # These are category examples absent from the original P0 test corpus.
    p0=Path(__file__).with_name('test_p0_safety.py').read_text()
    assert text not in p0
    assert explain.quality_check('care',text,'')=='出現'+category
    monkeypatch.setattr(explain,'TAIDE_MODEL','fake')
    monkeypatch.setattr(explain,'_chat',Mock(return_value=text))
    result=explain.generate('care',CUSTOMER)
    assert result['text']==templates.care_message({})
    assert result['reason']=='出現'+category and result['raw']==text


def test_category_rules_preserve_safe_generic_contact_and_fixed_template():
    for text in [DRAFT,templates.care_message({}), '若有需要，歡迎透過客服專線與我們聯繫。']:
        assert explain.quality_check('care',text,'')==''
