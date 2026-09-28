"""``context-core`` — framework-agnostic context-engineering practices.

This package holds ALL decision logic for the three practices, with **no agent-framework import**:
neither ``strands`` nor ``langchain``/``langgraph`` may appear anywhere under ``context_core``. It
operates on a neutral message shape (see :mod:`context_core.message`); each framework binding adapts its
native message type at the boundary and calls into this core.

Subpackages:
    relevance   Practice A — chunk/rerank/select/preview of oversized tool results, plus a store.
    disclosure  Practice B — lexical tool index and lean-catalog / exchange-fold logic.
    graph       Practice D — Card model, scan, scoring, matcher, and per-call projection.

The neutral contract is machine-checked by ``tests/test_no_framework_import.py``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
