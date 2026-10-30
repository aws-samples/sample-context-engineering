"""Pytest bootstrap: make ``opencode_gate_a`` importable as a top-level package during test runs.

The offline Gate A layer is a standalone directory under ``validation/`` and is not installed. Adding the
parent ``validation/`` directory to ``sys.path`` lets the tests import it as ``opencode_gate_a`` without a
build step, which keeps the package free of packaging boilerplate it does not need.
"""

from __future__ import annotations

import os
import sys

_VALIDATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VALIDATION_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_DIR)
