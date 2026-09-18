"""Relevance filtering of oversized tool results for Strands Agents.

Provides :class:`RelevanceFilter`, a ``Plugin`` that rewrites an oversized textual tool result on the
public ``AfterToolCallEvent`` hook, before the result becomes a conversation message. The raw sub-blocks
go to a ``Store`` first, then the result is replaced by the ``[Relevance: …]`` marker, a verbatim preview
bounded by ``preview_tokens * 4`` characters, and reference tokens the model can read back through the
plugin's own ``retrieve_context`` tool. Selection is verbatim: chosen chunks reach the model
character-for-character and the excess is never summarized, which is what keeps numeric, monetary, and
tabular content exact.

Scoring sits behind the ``Reranker`` protocol, whose default ``BedrockReranker`` is the only remote
dependency and is constructed lazily — supply a ``Reranker`` and the plugin never builds an AWS client.

Example Usage:
    ```python
    from strands import Agent
    from strands_relevance_filter import RelevanceFilter

    agent = Agent(plugins=[RelevanceFilter(max_result_tokens=8000)])
    ```
"""

from .plugin import RelevanceConfig, RelevanceFilter, ShouldFilter
from .reranker import BedrockReranker, Reranker, RerankerError
from .store import FileStore, InMemoryStore, S3Store, Store

__all__ = [
    "BedrockReranker",
    "FileStore",
    "InMemoryStore",
    "RelevanceConfig",
    "RelevanceFilter",
    "Reranker",
    "RerankerError",
    "S3Store",
    "ShouldFilter",
    "Store",
]
