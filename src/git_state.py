"""Git provenance shared by registry and protocol execution records."""
import subprocess


def git_dirty(root):
    """Detect tracked changes or untracked files outside root-level results/.

    Tracked results remain significant. Git ignores still apply to untracked
    files. NUL-delimited byte paths preserve spaces, Unicode and newlines.
    Git failures propagate rather than silently recording a clean checkout.
    """
    tracked = subprocess.check_output(
        ['git', 'status', '--porcelain=v1', '-z', '--untracked-files=no'], cwd=root)
    if tracked:
        return True
    untracked = subprocess.check_output(
        ['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=root)
    return any(path and not path.startswith(b'results/') for path in untracked.split(b'\0'))
