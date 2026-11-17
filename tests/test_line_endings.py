"""C1.2: checkout attributes and strict byte-hash validation."""
from pathlib import Path
import subprocess

import pytest

from src.registry import IntegrityError, Registry, file_hash, write_json


@pytest.mark.parametrize('path', [
    'data/default_of_credit_card_clients.csv',
    'data/another.csv',
    'docs/frozen_rules.md',
    'outputs/frozen/rules-g2a-20261116.json',
    'outputs/frozen/model-example.json',
    'src/registry.py',
])
def test_git_attributes_enforce_lf(path):
    root = Path(__file__).parents[1]
    result = subprocess.check_output(['git', 'check-attr', '-z', 'eol', '--', path], cwd=root)
    assert result.split(b'\0') == [path.encode(), b'eol', b'lf', b'']


def test_crlf_hash_mismatch_is_rejected_with_windows_recovery_hint(tmp_path):
    docs = tmp_path/'docs'
    frozen = tmp_path/'outputs/frozen'
    docs.mkdir()
    frozen.mkdir(parents=True)
    markdown = docs/'frozen_rules.md'
    markdown.write_bytes(b'# Synthetic rules\nApproved fixture\n')
    expected = file_hash(markdown)
    write_json(frozen/'rules-line-endings.json', {
        'rules_freeze_id': 'line-endings', 'markdown_sha256': expected})
    registry = Registry(tmp_path)
    assert registry.load_rules('line-endings')['markdown_sha256'] == expected
    markdown.write_bytes(b'# Synthetic rules\r\nApproved fixture\r\n')
    assert file_hash(markdown) != expected
    with pytest.raises(IntegrityError, match='Frozen Markdown hash mismatch') as error:
        registry.load_rules('line-endings')
    assert 'Windows' in str(error.value)
    assert 'README.md「Windows 使用者」' in str(error.value)
