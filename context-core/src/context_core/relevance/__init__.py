"""Practice A — relevance filtering of oversized tool results (framework-agnostic core).

Holds the whole decision pipeline with no agent-framework import: chunking, scoring behind the
``Reranker`` seam, threshold-and-budget selection, verbatim gap-marker assembly (:mod:`.preview`), the
offload backends (:mod:`.store`), and the retrieval-tool search helpers (:mod:`.search`, :mod:`.read`). A framework
binding converts its native tool result to the neutral shape (see :mod:`context_core.message`), pulls
the text out, and calls in here.
"""

from .preview import Chunk, PreviewStats, RelevancePreview
from .read import DEFAULT_CONTEXT_LINES, ReadRequestError, has_read_options, read_content
from .reranker import BedrockReranker, Reranker, RerankerError
from .store import FileStore, InMemoryStore, S3Store, Store

__all__ = [
    "BedrockReranker",
    "Chunk",
    "DEFAULT_CONTEXT_LINES",
    "FileStore",
    "InMemoryStore",
    "PreviewStats",
    "ReadRequestError",
    "RelevancePreview",
    "Reranker",
    "RerankerError",
    "S3Store",
    "Store",
    "has_read_options",
    "read_content",
]
