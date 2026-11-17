"""C1.3 portable metadata paths; pure paths never access real sealed files."""
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from src._sealed import _manifest_key
from src import registry


@pytest.mark.parametrize('relative', [
    'fold-0/inner.csv', 'fold-4/model-policy.json', 'assignments.csv',
])
def test_windows_manifest_keys_use_forward_slashes(relative):
    folder = PureWindowsPath('C:/project/outputs/sealed/synthetic')
    child = folder / relative
    files = {_manifest_key(child, folder): 'synthetic-hash'}
    assert list(files) == [relative]
    assert all('\\' not in key for key in files)


@pytest.mark.parametrize('relative', [
    'fold-0/inner.csv', 'fold-4/model-policy.json', 'assignments.csv',
])
def test_linux_manifest_keys_are_unchanged(relative):
    folder = PurePosixPath('/project/outputs/sealed/synthetic')
    child = folder / relative
    assert _manifest_key(child, folder) == str(child.relative_to(folder)) == relative


def test_windows_registry_relative_uses_forward_slashes(monkeypatch):
    # Exercise the real Registry.relative method with a Windows path flavour.
    store = object.__new__(registry.Registry)
    store.root = PureWindowsPath('C:/project')
    monkeypatch.setattr(registry, 'Path', PureWindowsPath)
    path = PureWindowsPath('C:/project/outputs/models/synthetic/model.json')
    relative = store.relative(path)
    assert relative == 'outputs/models/synthetic/model.json'
    assert '\\' not in relative


def test_linux_registry_relative_is_unchanged(monkeypatch):
    store = object.__new__(registry.Registry)
    store.root = PurePosixPath('/project')
    monkeypatch.setattr(registry, 'Path', PurePosixPath)
    path = store.root/'outputs/models/synthetic/model.json'
    assert store.relative(path) == str(path.relative_to(store.root))
