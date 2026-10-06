"""Progressive Tool Disclosure for Strands Agents.

The catalog, the exchange fold and the lexical tool index live in ``agent-context-core``
(:mod:`context_core.disclosure`); this package is the Strands binding over them. ``LexicalToolIndex``,
``ToolIndex`` and ``ToolMatch`` are the core's, re-exported.
"""

from context_core.disclosure import LexicalToolIndex, ToolIndex, ToolMatch

from .plugin import ProgressiveToolDisclosure

__all__ = [
    "LexicalToolIndex",
    "ProgressiveToolDisclosure",
    "ToolIndex",
    "ToolMatch",
]
