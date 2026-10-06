"""Property tests for the tool search index.

Feature: progressive-tool-disclosure-plugin, Property 12: The ``ToolIndex`` contract holds for any implementation.

Validates: Requirements 6.1, 6.2, 6.3, 6.4.

Feature: progressive-tool-disclosure-plugin, Property 13: The lexical index scores by distinct-term frequency and is
deterministic.

Validates: Requirements 6.9, 6.10.

The contract claims are the ones the projection relies on without re-checking: a bounded, descending result whose names
resolve back to the registry, over an input sequence ``build`` left alone. They are asserted against three
implementations — the shipped :class:`LexicalToolIndex`, a deterministic double that ranks by nothing but position, and
an awaitable double — because the protocol admits all three and the caller must not depend on which it got.

The scoring claim is checked against an independent oracle: the test recomposes each specification's indexable text and
counts the need's distinct terms itself, then asserts the whole match list at once, so score, omission of zero-score
specifications and the indexing-order tie break are all covered by one equality. Everything here is standard library
plus Hypothesis: no network, no disk, no model call.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_progressive_tool_disclosure.index import LexicalToolIndex, ToolIndex, ToolMatch

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# A small shared vocabulary, so a generated need and a generated specification actually share terms often enough for the
# scoring branch to be exercised rather than the no-match one.
WORDS = ["list", "accounts", "balance", "wire", "audit", "log", "send", "get", "user", "id", "transfer"]

word_strategy = st.sampled_from(WORDS)

# Underscored names are the point of the tokenizer: ``list_accounts`` has to index as ``list`` plus ``accounts``.
name_strategy = st.lists(word_strategy, min_size=1, max_size=3).map("_".join)

phrase_strategy = st.lists(word_strategy, max_size=6).map(" ".join)

parameter_strategy = st.fixed_dictionaries({"type": st.just("string"), "description": phrase_strategy})

schema_strategy = st.builds(
    lambda description, properties: {"type": "object", "description": description, "properties": properties},
    phrase_strategy,
    st.dictionaries(name_strategy, parameter_strategy, max_size=3),
)

spec_strategy = st.builds(
    lambda name, description, schema: {"name": name, "description": description, "inputSchema": {"json": schema}},
    name_strategy,
    phrase_strategy,
    schema_strategy,
)

specs_strategy = st.lists(spec_strategy, max_size=6)

need_strategy = st.lists(word_strategy, max_size=5).map(" ".join)

# Non-positive values are in range on purpose: they are the "return nothing without scoring" branch of the contract.
top_k_strategy = st.integers(min_value=-2, max_value=6)


class _PositionalToolIndex:
    """Deterministic double that ranks by indexing position alone, highest position first.

    It honors the protocol while sharing none of the lexical implementation's logic, which is what makes it useful
    here: the contract assertions have to hold for an index that ignores the need entirely.
    """

    def __init__(self) -> None:
        """Create an empty index."""
        self._names: list[str] = []

    def build(self, specs: Sequence[dict[str, Any]]) -> None:
        """Record the names to rank, in indexing order."""
        self._names = [spec["name"] for spec in specs if spec.get("name")]

    def search(self, need: str, top_k: int) -> Sequence[ToolMatch]:
        """Return the last-indexed names first, scored by position."""
        if top_k <= 0:
            return []

        ranked = [ToolMatch(name=name, score=float(position)) for position, name in enumerate(self._names)]
        ranked.sort(key=lambda match: -match.score)

        return ranked[:top_k]


class _AwaitableToolIndex:
    """Deterministic double whose operations return awaitables, the other half of the protocol."""

    def __init__(self) -> None:
        """Create an empty index backed by the lexical scorer."""
        self._inner = LexicalToolIndex()

    def build(self, specs: Sequence[dict[str, Any]]) -> Awaitable[None]:
        """Index ``specs`` through an awaitable."""

        async def _build() -> None:
            self._inner.build(specs)

        return _build()

    def search(self, need: str, top_k: int) -> Awaitable[Sequence[ToolMatch]]:
        """Rank against ``need`` through an awaitable."""

        async def _search() -> Sequence[ToolMatch]:
            return self._inner.search(need, top_k)

        return _search()


INDEX_FACTORIES: list[Callable[[], Any]] = [LexicalToolIndex, _PositionalToolIndex, _AwaitableToolIndex]

INDEX_IDS = ["lexical", "positional", "awaitable"]


def _resolve(result: Any) -> Any:
    """Take the value of a protocol operation, awaiting it when the implementation returned an awaitable.

    Args:
        result: What ``build`` or ``search`` returned.

    Returns:
        The resolved value.
    """
    if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
        return asyncio.run(_awaited(result))

    return result


async def _awaited(awaitable: Awaitable[Any]) -> Any:
    """Await ``awaitable`` and return its value."""
    return await awaitable


def _build(index: ToolIndex, specs: list[dict[str, Any]]) -> None:
    """Index ``specs``, resolving an awaitable ``build``."""
    _resolve(index.build(specs))


def _search(index: ToolIndex, need: str, top_k: int) -> Sequence[ToolMatch]:
    """Rank ``need``, resolving an awaitable ``search``."""
    return _resolve(index.search(need, top_k))


def _schema_text(node: object, depth: int) -> list[str]:
    """Collect the indexable text of a schema node, the oracle's own walk.

    Args:
        node: Schema node. Anything that is not a mapping contributes nothing.
        depth: Remaining nesting levels.

    Returns:
        Descriptions and property names at this node and below, in declaration order.
    """
    if depth <= 0 or not isinstance(node, dict):
        return []

    parts: list[str] = []

    description = node.get("description")
    if isinstance(description, str) and description:
        parts.append(description)

    properties = node.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            if name:
                parts.append(name)
            parts.extend(_schema_text(child, depth - 1))

    parts.extend(_schema_text(node.get("items"), depth - 1))

    return parts


def _expected_text(spec: dict[str, Any]) -> str:
    """Recompose the text a specification is expected to be scored against."""
    parts = [spec.get("name") or "", spec.get("description") or ""]
    parts.extend(_schema_text(spec.get("inputSchema", {}).get("json"), 4))

    return " ".join(part for part in parts if part)


def _terms(text: str) -> list[str]:
    """Split ``text`` into the terms the oracle compares, lowercased alphanumeric runs."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _expected_matches(specs: list[dict[str, Any]], need: str, top_k: int) -> list[ToolMatch]:
    """Score ``specs`` against ``need`` independently of the implementation.

    Args:
        specs: Specifications in indexing order.
        need: The need to score against.
        top_k: Maximum number of matches.

    Returns:
        At most ``top_k`` non-zero matches, descending by score, ties broken by indexing order.
    """
    if top_k <= 0:
        return []

    distinct = set(_terms(need))
    if not distinct:
        return []

    scored: list[tuple[float, int, str]] = []
    for position, spec in enumerate(spec for spec in specs if spec.get("name")):
        counts = _terms(_expected_text(spec))
        score = float(sum(counts.count(term) for term in distinct))
        if score > 0:
            scored.append((score, position, spec["name"]))

    scored.sort(key=lambda entry: (-entry[0], entry[1]))

    return [ToolMatch(name=name, score=score) for score, _, name in scored[:top_k]]


@pytest.mark.parametrize("factory", INDEX_FACTORIES, ids=INDEX_IDS)
@given(specs=specs_strategy, need=need_strategy, top_k=top_k_strategy)
@PROPERTY_SETTINGS
def test_search_honors_the_index_contract(
    factory: Callable[[], Any],
    specs: list[dict[str, Any]],
    need: str,
    top_k: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 12.

    The ``ToolIndex`` contract holds for any implementation.

    Validates: Requirements 6.1, 6.2, 6.3, 6.4.
    """
    index = factory()
    snapshot = deepcopy(specs)
    elements = list(specs)

    _build(index, specs)

    # build leaves the sequence alone: same length, same objects in the same slots, same content.
    assert len(specs) == len(elements)
    assert all(received is original for received, original in zip(specs, elements, strict=True))
    assert specs == snapshot

    tru_matches = _search(index, need, top_k)

    assert len(tru_matches) <= max(top_k, 0)
    if top_k <= 0:
        assert list(tru_matches) == []

    scores = [match.score for match in tru_matches]
    assert scores == sorted(scores, reverse=True)

    built_names = {spec["name"] for spec in specs if spec.get("name")}
    assert {match.name for match in tru_matches} <= built_names


@pytest.mark.parametrize("factory", INDEX_FACTORIES, ids=INDEX_IDS)
@given(first=specs_strategy, second=specs_strategy, need=need_strategy, top_k=st.integers(min_value=1, max_value=6))
@PROPERTY_SETTINGS
def test_search_returns_names_from_the_last_build_only(
    factory: Callable[[], Any],
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    need: str,
    top_k: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 12.

    The ``ToolIndex`` contract holds for any implementation.

    Validates: Requirements 6.1, 6.3.
    """
    index = factory()

    _build(index, first)
    _build(index, second)

    tru_names = {match.name for match in _search(index, need, top_k)}
    exp_names = {spec["name"] for spec in second if spec.get("name")}

    assert tru_names <= exp_names


@given(specs=specs_strategy, need=need_strategy, top_k=st.integers(min_value=1, max_value=6))
@PROPERTY_SETTINGS
def test_lexical_scores_by_distinct_term_frequency(
    specs: list[dict[str, Any]],
    need: str,
    top_k: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 13.

    The lexical index scores by distinct-term frequency and is deterministic.

    Validates: Requirements 6.9, 6.10.
    """
    index = LexicalToolIndex()
    index.build(specs)

    tru_matches = list(index.search(need, top_k))
    exp_matches = _expected_matches(specs, need, top_k)

    assert tru_matches == exp_matches

    # Zero-score specifications never appear, which is what lets the search tool report "no match" instead of offering
    # something arbitrary.
    assert all(match.score > 0 for match in tru_matches)


@given(specs=specs_strategy, need=need_strategy, top_k=st.integers(min_value=1, max_value=6))
@PROPERTY_SETTINGS
def test_lexical_search_is_deterministic_across_repeated_runs(
    specs: list[dict[str, Any]],
    need: str,
    top_k: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 13.

    The lexical index scores by distinct-term frequency and is deterministic.

    Validates: Requirements 6.10.
    """
    first = LexicalToolIndex()
    first.build(specs)

    second = LexicalToolIndex()
    second.build(specs)

    tru_runs = [list(first.search(need, top_k)) for _ in range(3)]
    tru_runs.extend(list(second.search(need, top_k)) for _ in range(3))

    assert all(run == tru_runs[0] for run in tru_runs)


@given(
    need=st.lists(word_strategy, min_size=1, max_size=3).map(" ".join),
    description=phrase_strategy,
    size=st.integers(min_value=2, max_value=5),
)
@PROPERTY_SETTINGS
def test_lexical_breaks_score_ties_by_indexing_order(need: str, description: str, size: int) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 13.

    The lexical index scores by distinct-term frequency and is deterministic.

    Validates: Requirements 6.9, 6.10.
    """
    # Identical text under different names, so every score ties and only the indexing order can order the result.
    specs = [
        {
            "name": f"tool_{position}",
            "description": f"{need} {description}",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
        for position in range(size)
    ]

    index = LexicalToolIndex()
    index.build(specs)

    tru_names = [match.name for match in index.search(need, size)]
    exp_names = [f"tool_{position}" for position in range(size)]

    assert tru_names == exp_names
