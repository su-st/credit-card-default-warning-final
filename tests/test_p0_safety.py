import unittest
import os
import runpy
import io
from contextlib import redirect_stdout
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from llm import explain, templates
from notify import line_push

SAFE = templates.care_message({})
APP = '''
import pandas as pd
from app.views.customer import render
render(pd.DataFrame([
 dict(customer_id=i, risk_score=.8, risk_rank=i, in_list=1,
      factor_1='9 月延遲繳款 2 個月', factor_2='', factor_3='')
 for i in [1, 2]]), {})
'''


class SafetyTests(unittest.TestCase):
    def setUp(self):
        # Never use credentials/network, including when local environment has bindings.
        self.patches = [patch.object(explain, 'TAIDE_MODEL', ''),
                        patch.object(line_push, 'SEND_ENABLED', False),
                        patch.object(line_push, 'TOKEN', 'fake-token'),
                        patch.object(line_push, 'TEST_USER', 'test-account'),
                        patch.object(line_push, 'ApiClient')]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def app(self):
        a = AppTest.from_string(APP, default_timeout=20).run()
        self.assertFalse(a.exception)
        return a

    def test_customer_switch_clears_all_content_and_confirmation(self):
        a = self.app()
        a.button[0].click().run()
        a.button[1].click().run()
        a.checkbox[0].check().run()
        self.assertFalse(a.button[2].disabled)
        a.number_input[0].set_value(2).run()
        self.assertFalse(a.exception)
        self.assertFalse(a.info)
        self.assertEqual(a.text_area[0].value, '')
        self.assertFalse(a.checkbox[0].value)
        self.assertTrue(a.button[2].disabled)
        a.number_input[0].set_value(999).run()
        a.number_input[0].set_value(1).run()
        self.assertEqual(a.text_area[0].value, '')
        self.assertFalse(a.info)

    def test_edit_reconfirm_regenerate_and_simulated_send(self):
        a = self.app()
        a.button[1].click().run()
        a.checkbox[0].check().run()
        a.text_area[0].set_value(SAFE + ' ').run()
        self.assertFalse(a.checkbox[0].value)
        self.assertTrue(a.button[2].disabled)
        a.checkbox[0].check().run()
        self.assertFalse(a.button[2].disabled)
        a.button[2].click().run()
        self.assertFalse(a.exception)
        self.assertIn('模擬發送', a.info[0].value)
        self.assertFalse(a.checkbox[0].value)
        self.assertTrue(a.button[2].disabled)
        self.assertFalse(a.success)
        line_push.ApiClient.assert_not_called()
        a.run()
        self.assertTrue(a.button[2].disabled)
        a.button[1].click().run()
        self.assertFalse(a.checkbox[0].value)
        self.assertTrue(a.button[2].disabled)

    def test_invalid_manual_edit_cannot_be_confirmed_for_send(self):
        a = self.app()
        a.text_area[0].set_value('請於明天繳款，否則凍結帳戶。').run()
        a.checkbox[0].check().run()
        self.assertTrue(a.button[2].disabled)
        self.assertTrue(a.warning)
        line_push.ApiClient.assert_not_called()

    def test_conservative_validation_and_generation_fallback(self):
        self.assertEqual(explain.check_care(SAFE), '')
        for text in ['', '請於明天中午以前繳款，否則會凍結帳戶。',
                     '請致電客服專線零八零零。', '點選 https://example.com 辦理。',
                     '保證免息並核准分期。', '本月底前付款。', '請於１２月繳款。',
                     SAFE + '保證不收費。']:
            with self.subTest(text=text):
                self.assertTrue(explain.check_care(text))
                with patch.object(explain, 'TAIDE_MODEL', 'fake'), patch.object(explain, '_chat', return_value=text):
                    r = explain.generate('care', {})
                    self.assertEqual(r['text'], SAFE)
                    self.assertTrue(r['reason'])
                    self.assertIn('固定句型', r['how'])
        with patch.object(explain, 'TAIDE_MODEL', 'fake'), patch.object(explain, '_chat', side_effect=RuntimeError('offline')):
            self.assertEqual(explain.generate('care', {})['text'], SAFE)

    def test_simulation_never_constructs_client_even_with_credentials(self):
        self.assertIn('模擬發送', line_push.push_text(SAFE))
        with self.assertRaises(ValueError):
            line_push.push_text(SAFE, user_id='someone-else')
        with self.assertRaises(ValueError):
            line_push.push_text('請於明天付款')
        line_push.ApiClient.assert_not_called()

    def test_explicit_send_uses_only_configured_test_account_mocked(self):
        with patch.object(line_push, 'SEND_ENABLED', True), patch.object(line_push, 'MessagingApi') as api:
            self.assertFalse(line_push.is_simulation())
            self.assertEqual(line_push.push_text(SAFE, approved_text=SAFE), '已送出到測試帳號。')
            request = api.return_value.push_message.call_args.args[0]
            self.assertEqual(request.to, 'test-account')
            with self.assertRaises(ValueError):
                line_push.push_text(SAFE, 'someone-else')
            self.assertEqual(api.return_value.push_message.call_count, 1)

    def test_real_mode_ui_and_error_are_distinct_mocked(self):
        for failure in [False, True]:
            with self.subTest(failure=failure), patch.object(line_push, 'SEND_ENABLED', True), patch.object(line_push, 'push_text', side_effect=RuntimeError('fail') if failure else None, return_value='已送出到測試帳號。'):
                a = self.app()
                a.button[1].click().run()
                a.checkbox[0].check().run()
                a.button[2].click().run()
                self.assertFalse(a.exception)
                self.assertFalse(a.checkbox[0].value)
                self.assertTrue(a.button[2].disabled)
                self.assertEqual(bool(a.error), failure)
                self.assertEqual(bool(a.success), not failure)

    def test_real_send_requires_matching_approval(self):
        with patch.object(line_push, 'SEND_ENABLED', True):
            for approval in [None, '', SAFE + 'changed']:
                with self.subTest(approval=approval), self.assertRaises(PermissionError):
                    line_push.push_text(SAFE, approved_text=approval)
            line_push.ApiClient.assert_not_called()

    def test_cli_never_sends_even_when_enabled(self):
        with patch.dict(os.environ, {'LINE_SEND_ENABLED': 'true',
                                    'LINE_CHANNEL_ACCESS_TOKEN': 'fake',
                                    'LINE_TEST_USER_ID': 'test-account'}), \
             patch('linebot.v3.messaging.ApiClient') as client, \
             patch('linebot.v3.messaging.MessagingApi') as api, \
             redirect_stdout(io.StringIO()) as output:
            runpy.run_module('notify.line_push', run_name='__main__')
            self.assertIn('CLI 不提供發送功能', output.getvalue())
            client.assert_not_called()
            api.assert_not_called()

    def test_ui_requires_new_approval_after_each_send(self):
        with patch.object(line_push, 'SEND_ENABLED', True), \
             patch.object(line_push, 'push_text', return_value='已送出到測試帳號。') as send:
            a = self.app()
            a.button[1].click().run()
            self.assertTrue(a.button[2].disabled)
            send.assert_not_called()
            a.checkbox[0].check().run()
            a.button[2].click().run()
            send.assert_called_once_with(SAFE, approved_text=SAFE)
            self.assertFalse(a.checkbox[0].value)
            self.assertTrue(a.button[2].disabled)
            a.run()
            self.assertEqual(send.call_count, 1)
            a.checkbox[0].check().run()
            a.button[2].click().run()
            self.assertEqual(send.call_count, 2)
            self.assertFalse(a.checkbox[0].value)


if __name__ == '__main__':
    unittest.main()
