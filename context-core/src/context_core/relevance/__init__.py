"""Practice A — relevance filtering of oversized tool results (framework-agnostic core).

Holds the whole decision pipeline with no agent-framework import: chunking, scoring behind the
``Reranker`` seam, threshold-and-budget selection, verbatim gap-marker assembly (:mod:`.preview`), the
offload backends (:mod:`.store`), and the retrieval-tool search helpers (:mod:`.search`). A framework
binding converts its native tool result to the neutral shape (see :mod:`context_core.message`), pulls
the text out, and calls in here.
"""

from .preview import Chunk, PreviewStats, RelevancePreview
from .reranker import BedrockReranker, Reranker, RerankerError
from .store import FileStore, InMemoryStore, S3Store, Store

__all__ = [
    "BedrockReranker",
    "Chunk",
    "FileStore",
    "InMemoryStore",
    "PreviewStats",
    "RelevancePreview",
    "Reranker",
    "RerankerError",
    "S3Store",
    "Store",
]
