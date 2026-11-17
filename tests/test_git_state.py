"""C1.1 provenance tests use isolated Git repositories, without research data."""
import subprocess
import sys

import pytest

from src.git_state import git_dirty


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, 'init', '-q')
    (tmp_path/'tracked.py').write_text('initial\n')
    (tmp_path/'results').mkdir()
    (tmp_path/'results/tracked.csv').write_text('initial\n')
    (tmp_path/'.gitignore').write_text('outputs/\n')
    git(tmp_path, 'add', '.')
    git(tmp_path, '-c', 'user.name=C1.1 Test', '-c', 'user.email=test@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Synthetic fixture')
    return tmp_path


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows filenames cannot contain newline characters')
def test_untracked_results_are_clean(repo):
    folder = repo/'results/new run'
    folder.mkdir()
    (folder/'結果\nsummary.csv').write_text('synthetic\n')
    assert git(repo, 'status', '--porcelain')
    assert not git_dirty(repo)


def test_untracked_python_outside_results_is_dirty(repo):
    (repo/'new script.py').write_text('# synthetic\n')
    assert git_dirty(repo)


@pytest.mark.parametrize('path', ['tracked.py', 'results/tracked.csv'])
@pytest.mark.parametrize('staged', [False, True])
def test_tracked_changes_including_results_are_dirty(repo, path, staged):
    (repo/path).write_text('changed\n')
    if staged:
        git(repo, 'add', path)
    assert git_dirty(repo)


def test_nested_results_outside_root_results_are_dirty(repo):
    folder = repo/'other/results'
    folder.mkdir(parents=True)
    (folder/'untracked.py').write_text('# synthetic\n')
    assert git_dirty(repo)


def test_clean_checkout_is_clean(repo):
    assert not git_dirty(repo)


def test_both_metadata_writers_use_shared_provenance(repo, monkeypatch):
    from src import config, registry
    from src._protocol_runs import _metadata
    monkeypatch.setattr(config, 'ROOT', repo)
    monkeypatch.setattr(registry, 'ROOT', repo)
    (repo/'results/untracked.csv').write_text('synthetic\n')
    store = registry.Registry(repo)
    clean = store.start_run('L1', 'synthetic provenance', model_id='synthetic-clean')
    assert clean.meta['git_dirty'] is False
    assert _metadata('L4', 'synthetic metadata only', 'synthetic-clean')['git_dirty'] is False
    (repo/'new.py').write_text('# synthetic\n')
    dirty = store.start_run('L1', 'synthetic provenance', model_id='synthetic-dirty')
    assert dirty.meta['git_dirty'] is True
    assert _metadata('L4', 'synthetic metadata only', 'synthetic-dirty')['git_dirty'] is True
