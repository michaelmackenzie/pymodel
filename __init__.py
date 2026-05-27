"""pymodel package shim.

Expose modules under the local python/ directory while keeping command
entrypoints separate.
"""

__version__ = "dev"

from pathlib import Path
import sys

_pkg_root = Path(__file__).resolve().parent
_python_dir = _pkg_root / "python"

if _python_dir.is_dir():
    __path__.append(str(_python_dir))
    if str(_python_dir) not in sys.path:
        sys.path.insert(0, str(_python_dir))
