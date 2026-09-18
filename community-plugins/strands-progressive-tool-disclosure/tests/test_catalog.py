"""Property tests for the Catalog Entry: budgeted description, verbatim name, empty-closed schema.

Feature: progressive-tool-disclosure-plugin, Property 7: A catalog entry is a faithful, budgeted
prefix with a verbatim name.
Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.8, 4.9.

Feature: progressive-tool-disclosure-plugin, Property 8: Every catalog entry has the empty-closed
schema and no extra fields.
Validates: Requirements 4.6, 4.7.

Both properties read the real helpers (``_catalog_entry``, ``_truncate_description``,
``_estimate_tokens``) rather than a double: the claims are about the cut the implementation actually
makes, so a stand-in would assert nothing. Generated descriptions mix sentence terminators, long
unbroken runs and the empty string, so every branch of the cut — sentence boundary, word boundary
with an ellipse, and the character-count fallback — is reachable from the same strategy.

No network: an autouse fixture makes any socket construction raise, so an accidental I/O path in the
catalog helpers fails the test instead of silently reaching out.
"""

import copy
import socket
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_progressive_tool_disclosure.plugin import (
    _catalog_entry,
    _estimate_tokens,
    _truncate_description,
)

CHARS_PER_TOKEN = 4
"""Characters per token, the estimate Requirement 4.1 fixes for the budget."""

ELLIPSIS = "..."
"""The marker the implementation appends when it cuts mid-sentence."""

SENTENCE_ENDINGS = ".!?"
"""Characters that terminate a sentence, for recognizing a sentence-boundary cut."""

EMPTY_CLOSED_SCHEMA = {"json": {"type": "object", "properties": {}, "additionalProperties": False}}
"""The exact ``inputSchema`` Requirement 4.6 demands of every catalog entry."""

CATALOG_ENTRY_FIELDS = {"name", "description", "inputSchema"}
"""The only fields a catalog entry may carry (Requirement 4.7 omits the other two)."""

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Descriptions written from the characters that decide the cut: words, sentence terminators and
# whitespace, plus runs long enough that no word boundary fits a small budget.
description_strategy = st.one_of(
    st.just(""),
    st.text(
        alphabet=st.sampled_from(list("abcdefghij .!? ,-_")),
        min_size=0,
        max_size=120,
    ),
    # A single unbroken run: the case where every boundary-based cut would return nothing.
    st.text(alphabet=st.sampled_from(list("abcdefghij")), min_size=1, max_size=80),
)

# Names are the key the model calls the tool by, so they are generated with the shapes that would
# tempt a normalizer: underscores, mixed case, digits, dots and leading/trailing whitespace.
name_strategy = st.sampled_from(
    [
        "find_tools",
        "list_accounts",
        "HTTPRequest",
        "s3.get_object",
        "tool-with-dashes",
        " padded_name ",
        "a" * 80,
        "名前",
    ]
)

budget_strategy = st.integers(min_value=1, max_value=40)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any socket construction fail, so a hidden I/O path surfaces as a test failure."""

    def deny(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the catalog helpers must perform no I/O")

    monkeypatch.setattr(socket, "socket", deny)


@st.composite
def spec_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Build a full specification as the registry holds one, with the fields the catalog must drop."""
    spec: dict[str, Any] = {
        "name": draw(name_strategy),
        "description": draw(description_strategy),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for."}},
                "required": draw(st.sampled_from([[], ["query"]])),
                "additionalProperties": True,
            }
        },
    }
    if draw(st.booleans()):
        spec["outputSchema"] = {"json": {"type": "object", "properties": {"result": {"type": "string"}}}}
    if draw(st.booleans()):
        spec["annotations"] = {"readOnlyHint": True, "title": "Search"}
    return spec


def is_sentence_cut(text: str, result: str) -> bool:
    """Report whether ``result`` ends at a sentence boundary of ``text`` (Requirement 4.4)."""
    if not result or result[-1] not in SENTENCE_ENDINGS:
        return False
    return len(result) >= len(text) or text[len(result)].isspace()


def is_word_cut(text: str, result: str) -> bool:
    """Report whether ``result`` is a word-boundary cut of ``text`` marked with an ellipse."""
    if not result.endswith(ELLIPSIS):
        return False
    body = result[: -len(ELLIPSIS)]
    # The body is a rstripped prefix, so the character just past it is the whitespace that ended the
    # last whole word kept. An empty body is the degenerate case of a leading run of whitespace.
    return body == "" or (len(body) < len(text) and text[len(body)].isspace())


def is_character_cut(text: str, result: str, max_chars: int) -> bool:
    """Report whether ``result`` is the character-count fallback cut at the budget limit."""
    return len(result) == max_chars and result == text[:max_chars]


@PROPERTY_SETTINGS
@given(spec=spec_strategy(), catalog_tokens=budget_strategy)
def test_catalog_entry_is_a_faithful_budgeted_prefix(spec: dict[str, Any], catalog_tokens: int) -> None:
    """Property 7: the entry's description is a boundary-cut prefix inside the budget, name verbatim.

    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.8, 4.9.
    """
    source = copy.deepcopy(spec)
    description = spec["description"]
    max_chars = catalog_tokens * CHARS_PER_TOKEN

    entry = _catalog_entry(spec, catalog_tokens)
    result = entry["description"]

    # Requirement 4.5: the name is the call key, copied character by character.
    assert entry["name"] == source["name"]

    # Requirement 4.8: the source specification is only read.
    assert spec == source

    if _estimate_tokens(description) <= catalog_tokens:
        # Requirement 4.3, and Requirement 4.9 for the zero-character description.
        assert result == description
        return

    # Requirement 4.1: the estimate never exceeds the budget, ellipse included.
    assert _estimate_tokens(result) <= catalog_tokens

    # Requirement 4.2: a prefix of the source, allowing only an appended ellipse — no rewriting and
    # no reordering, which ``startswith`` on the stripped body verifies directly.
    body = result[: -len(ELLIPSIS)] if result.endswith(ELLIPSIS) else result
    assert description.startswith(body)

    # Requirement 4.4: the cut lands on a sentence boundary, else a word boundary, else the limit.
    assert (
        is_sentence_cut(description, result)
        or is_word_cut(description, result)
        or is_character_cut(description, result, max_chars)
    )

    # Requirement 4.8: the same pair yields the same entry, field by field.
    assert _catalog_entry(source, catalog_tokens) == entry


@PROPERTY_SETTINGS
@given(description=description_strategy, catalog_tokens=budget_strategy)
def test_truncation_is_idempotent_on_a_text_that_fits(description: str, catalog_tokens: int) -> None:
    """Property 7: a cut description already fits, so re-cutting it returns it unchanged.

    Validates: Requirements 4.1, 4.3.
    """
    cut = _truncate_description(description, catalog_tokens)
    assert _estimate_tokens(cut) <= catalog_tokens
    assert _truncate_description(cut, catalog_tokens) == cut


@PROPERTY_SETTINGS
@given(spec=spec_strategy(), catalog_tokens=budget_strategy)
def test_catalog_entry_has_the_empty_closed_schema_and_no_extra_fields(
    spec: dict[str, Any], catalog_tokens: int
) -> None:
    """Property 8: the entry carries the empty, closed schema and nothing beyond the three fields.

    Validates: Requirements 4.6, 4.7.
    """
    entry = _catalog_entry(spec, catalog_tokens)

    # Requirement 4.6: emptied and closed rather than dropped, because providers reject a spec
    # without an input schema.
    assert entry["inputSchema"] == EMPTY_CLOSED_SCHEMA

    # Requirement 4.7: the two fields that never help the model decide whether to search.
    assert set(entry) == CATALOG_ENTRY_FIELDS
    assert "outputSchema" not in entry
    assert "annotations" not in entry

    # A fresh schema per entry: a shared dict would let one consumer's mutation reach every entry.
    other = _catalog_entry(spec, catalog_tokens)
    assert entry["inputSchema"] is not other["inputSchema"]
    entry["inputSchema"]["json"]["properties"]["leaked"] = {"type": "string"}
    assert other["inputSchema"] == EMPTY_CLOSED_SCHEMA


def test_empty_description_yields_an_empty_description_and_a_verbatim_name() -> None:
    """Property 7 edge case: a zero-character description survives as zero characters.

    Validates: Requirements 4.5, 4.9.
    """
    spec: dict[str, Any] = {"name": "find_tools", "description": "", "inputSchema": {"json": {}}}

    entry = _catalog_entry(spec, 1)

    assert entry["description"] == ""
    assert entry["name"] == "find_tools"
