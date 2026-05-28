import pathlib
import sys


def ensure_repo_root_on_path(caller_file):
    """Ensure the repository's python/ directory is importable.

    If PYTHONPATH already includes the repo root, this is effectively a no-op.
    """
    repo_root = pathlib.Path(caller_file).resolve().parent.parent
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    return repo_root
