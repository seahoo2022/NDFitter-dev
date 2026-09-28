"""Resolve user-supplied paths without embedding a developer's home directory.

Relative paths are anchored to the source checkout, independent of the shell's
working directory. Use an editable install when developing this project.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    """Return a native path string; explicit absolute paths are also accepted."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())
