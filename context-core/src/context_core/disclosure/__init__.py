"""Practice B — progressive tool disclosure: lexical index and catalog/fold logic (core).

Both halves are framework-agnostic and operate on the neutral shapes:

    :mod:`~context_core.disclosure.tool_index`  the ``ToolIndex`` protocol + ``LexicalToolIndex``
    :mod:`~context_core.disclosure.catalog`     the budgeted catalog block and the exchange fold
"""

from .catalog import (
    CATALOG_PROMPT_HEADER,
    DEFAULT_CATALOG_CHARS,
    FIND_TOOLS_NAME,
    GET_TOOL_DETAILS_NAME,
    PLUGIN_TOOL_NAMES,
    SummaryCache,
    build_catalog,
    catalog_prompt_block,
    clamp_summary,
    estimate_tokens,
    fold_closed_exchanges,
    fold_note,
    pairs_intact,
    summary_line,
    truncate_description,
)
from .tool_index import LexicalToolIndex, ToolIndex, ToolMatch, ToolSpec

__all__ = [
    "CATALOG_PROMPT_HEADER",
    "DEFAULT_CATALOG_CHARS",
    "FIND_TOOLS_NAME",
    "GET_TOOL_DETAILS_NAME",
    "PLUGIN_TOOL_NAMES",
    "LexicalToolIndex",
    "SummaryCache",
    "ToolIndex",
    "ToolMatch",
    "ToolSpec",
    "build_catalog",
    "catalog_prompt_block",
    "clamp_summary",
    "estimate_tokens",
    "fold_closed_exchanges",
    "fold_note",
    "pairs_intact",
    "summary_line",
    "truncate_description",
]
