"""Compatibility launcher for the renamed Phoenix application.

The tested ``cellbench.core``, ``cellbench.analysis``, and ``cellbench.plots``
modules remain available for existing notebooks and imports. New application
development lives in the modular :mod:`phoenix` package.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phoenix.app import main


if __name__ == "__main__":
    main()
