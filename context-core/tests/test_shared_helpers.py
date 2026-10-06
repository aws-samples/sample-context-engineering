"""The helpers every binding used to carry its own copy of, now in the core.

``excerpt_disclaimer`` and ``read_chunks`` (relevance), ``requires_parameters`` (disclosure), and the ``stash``
pass-through of ``expand_artifact`` (graph). The bindings' suites still pin the end-to-end behaviour; these pin the
functions on their own, so a change to the shared copy is caught before any binding runs.
"""

from __future__ import annotations

import asyncio

from context_core.disclosure import requires_parameters
from context_core.graph import GraphState
from context_core.graph.store import InMemoryReferenceStore
from context_core.graph.tools import expand_artifact
from context_core.relevance.preview import Chunk, PreviewStats, _chunk_text, excerpt_disclaimer, read_chunks


def _stats(total_chunks: int = 4, total_lines: int = 40) -> PreviewStats:
    shown = (Chunk(index=1, text="b\n", start_line=11, end_line=20),)
    return PreviewStats(total_chunks=total_chunks, total_lines=total_lines, shown=shown, ranking=(1, 0, 2, 3))


# ---- excerpt_disclaimer ------------------------------------------------------------------------


def test_disclaimer_without_references_says_to_report_the_filtering() -> None:
    text = excerpt_disclaimer(9_000, _stats(), [], "retrieve_all_context")

    assert "this is an EXCERPT" in text
    assert "original: 40 lines, 4 chunks | shown: 1 chunk(s), lines 11-20" in text
    assert text.endswith("Say that the result was filtered instead of computing it from the excerpt.")
    assert "retrieve_all_context" not in text


def test_disclaimer_names_the_tool_and_the_single_reference() -> None:
    text = excerpt_disclaimer(9_000, _stats(), ["ref-1"], "retrieve_all_context")

    assert 'call `retrieve_all_context` with reference "ref-1"' in text
    assert "(4 chunks, ~9,000 tokens)" in text


def test_disclaimer_lists_several_references() -> None:
    text = excerpt_disclaimer(10, _stats(), ["ref-1", "ref-2"], "read_it")

    assert "with references ref-1, ref-2 and" in text


# ---- read_chunks -------------------------------------------------------------------------------


_TEXT = "".join(f"line {n}\n" for n in range(1, 201))


def test_read_chunks_of_empty_text() -> None:
    assert read_chunks("", chunk_tokens=50, ranking=None, max_chunks=2, max_chars=10_000) == "Content is empty (0 lines)."


def test_read_chunks_follows_the_recorded_ranking_in_document_order() -> None:
    chunks = _chunk_text(_TEXT, 50)
    ranking = tuple(reversed(range(len(chunks))))

    text = read_chunks(_TEXT, chunk_tokens=50, ranking=ranking, max_chunks=2, max_chars=100_000)

    assert text.startswith(f"[2 of {len(chunks)} chunks (most relevant first), lines 1-{chunks[-1].end_line} in total")
    assert chunks[-1].text in text and chunks[-2].text in text
    assert chunks[0].text not in text


def test_read_chunks_falls_back_to_document_order_on_a_stale_ranking() -> None:
    chunks = _chunk_text(_TEXT, 50)

    for ranking in (None, (0,)):
        text = read_chunks(_TEXT, chunk_tokens=50, ranking=ranking, max_chunks=1, max_chars=100_000)
        assert text.startswith(f"[1 of {len(chunks)} chunks (document order)")
        assert chunks[0].text in text


def test_read_chunks_states_the_limit_in_tokens() -> None:
    assert "limit ~2,500 tokens]" in read_chunks(_TEXT, chunk_tokens=50, ranking=None, max_chunks=1, max_chars=10_000)


# ---- requires_parameters -----------------------------------------------------------------------


def test_requires_parameters_reads_only_the_required_list() -> None:
    wrapped = {"inputSchema": {"json": {"type": "object", "required": ["city"]}}}
    unwrapped = {"inputSchema": {"type": "object", "required": ["city"]}}
    optional = {"inputSchema": {"json": {"type": "object", "properties": {"city": {}}}}}

    assert requires_parameters(wrapped)
    assert requires_parameters(unwrapped)
    assert not requires_parameters(optional)
    assert not requires_parameters({"inputSchema": {"json": {"required": []}}})


def test_requires_parameters_tolerates_malformed_specs() -> None:
    assert not requires_parameters({})
    assert not requires_parameters({"inputSchema": "nope"})
    assert not requires_parameters({"inputSchema": {"json": "nope"}})
    assert not requires_parameters({"inputSchema": {"json": {"required": "city"}}})


# ---- expand_artifact stash ---------------------------------------------------------------------


class _Stash:
    def __init__(self, blocks: dict[str, object]) -> None:
        self.blocks = blocks
        self.asked: list[str] = []

    async def retrieve(self, reference: str) -> object | None:
        self.asked.append(reference)
        return self.blocks.get(reference)


def _expand(stash: object | None) -> str:
    return asyncio.run(
        expand_artifact(GraphState(), InMemoryReferenceStore(), None, "ref-9", cycle=1, reuse_ttl_cycles=3, stash=stash)
    )


def test_expand_artifact_reads_an_explicit_stash_for_what_the_store_lacks() -> None:
    stash = _Stash({"ref-9": "alpha\nbeta"})

    answer = _expand(stash)

    assert stash.asked == ["ref-9"]
    assert answer.endswith("\n\nalpha\nbeta")


def test_expand_artifact_without_a_stash_is_unchanged() -> None:
    assert _expand(None) == asyncio.run(
        expand_artifact(GraphState(), InMemoryReferenceStore(), None, "ref-9", cycle=1, reuse_ttl_cycles=3)
    )
