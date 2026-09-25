"""Property tests for the neutral catalog and the exchange fold.

Ported from the Strands ``strands-progressive-tool-disclosure`` suite, which is read-only reference: the
budget/normalization/block properties come from its ``tests/test_catalog.py`` and the fold assertions from
the fold section of its ``tests/test_details_and_summary.py``, with every message written in the neutral
shape (:mod:`context_core.message`) and every tool specification as a plain dict.

Feature: progressive-tool-disclosure-plugin, Property 7: A catalog line is a faithful, budgeted
description of the tool, and the name it is listed under is verbatim.
Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.8, 4.9.

Feature: progressive-tool-disclosure-plugin, Property 8: The catalog and the call's tool list partition the
registry — every registered name is either callable on this call or listed in the prompt, never both and
never neither.

The budget is counted in CHARACTERS (``catalog_chars``), and a line is a summary — truncation at a sentence
or word boundary is the fallback for a summary that could not be produced, and the clamp applied to one that
overran. Generated descriptions mix sentence terminators, long unbroken runs and the empty string, so every
branch of the fallback cut is reachable from the same strategy.

No network: nothing here calls a model, and an autouse fixture makes any outbound socket connection raise,
so an accidental I/O path in the catalog helpers fails the test instead of silently reaching out.
"""

from __future__ import annotations

import copy
import socket
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from context_core.disclosure.catalog import (
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
    pairs_intact,
    summary_line,
    truncate_description,
)

SENTENCE_ENDINGS = ".!?"
"""Characters that terminate a sentence, for recognizing a sentence-boundary cut."""

ELLIPSIS = "..."
"""What marks a description as cut."""

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Descriptions written from the characters that decide the cut: words, sentence terminators and
# whitespace, plus runs long enough that no word boundary fits a small budget.
description_strategy = st.one_of(
    st.just(""),
    st.text(alphabet=st.sampled_from(list("abcdefghij .!? ,-_")), min_size=0, max_size=120),
    # A single unbroken run: the case where every boundary-based cut would return nothing.
    st.text(alphabet=st.sampled_from(list("abcdefghij")), min_size=1, max_size=80),
)

# Names are the key the model calls the tool by, so they are generated with the shapes that would tempt a
# normalizer: underscores, mixed case, digits, dots and leading/trailing whitespace.
name_strategy = st.sampled_from(
    [
        "list_accounts",
        "HTTPRequest",
        "s3.get_object",
        "tool-with-dashes",
        " padded_name ",
        "a" * 80,
        "名前",
    ]
)

chars_strategy = st.integers(min_value=1, max_value=160)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound socket connection fail, so a hidden I/O path surfaces as a test failure."""

    def deny(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the catalog helpers must perform no I/O")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def _spec(name: str, description: str = "") -> dict[str, Any]:
    """A tool specification in the neutral shape, reduced to what the catalog reads."""
    return {"name": name, "description": description, "inputSchema": {"json": {"type": "object"}}}


def _listed_names(block: str) -> list[str]:
    """Names a catalog block lists, in order, read back off its ``- name: summary`` lines."""
    return [line[2:].split(":", 1)[0] for line in block.splitlines() if line.startswith("- ")]


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
    # The body is a rstripped prefix, so the character just past it is the whitespace that ended the last
    # whole word kept. An empty body is the degenerate case of a leading run of whitespace.
    return body == "" or (len(body) < len(text) and text[len(body)].isspace())


def is_character_cut(text: str, result: str, max_chars: int) -> bool:
    """Report whether ``result`` is the character-count fallback cut at the budget limit."""
    return len(result) == max_chars and result == text[:max_chars]


# --------------------------------------------------------------------------------------------------
# The budget is characters, and the fallback cut is a faithful prefix inside it.
# --------------------------------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(description=description_strategy, catalog_chars=chars_strategy)
def test_the_fallback_cut_is_a_faithful_prefix_inside_the_character_budget(
    description: str, catalog_chars: int
) -> None:
    """Property 7: the truncation fallback never exceeds ``catalog_chars`` and never rewrites the text.

    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.9.
    """
    result = truncate_description(description, catalog_chars)

    # Requirement 4.1: the limit is counted in characters, ellipse included.
    assert len(result) <= catalog_chars

    if len(description) <= catalog_chars:
        # Requirement 4.3, and Requirement 4.9 for the zero-character description.
        assert result == description
        return

    # Requirement 4.2: a prefix of the source, allowing only an appended ellipse — no rewriting and no
    # reordering, which ``startswith`` on the stripped body verifies directly.
    body = result[: -len(ELLIPSIS)] if result.endswith(ELLIPSIS) else result
    assert description.startswith(body)

    # Requirement 4.4: the cut lands on a sentence boundary, else a word boundary, else the limit.
    assert (
        is_sentence_cut(description, result)
        or is_word_cut(description, result)
        or is_character_cut(description, result, catalog_chars)
    )


@PROPERTY_SETTINGS
@given(description=description_strategy, catalog_chars=chars_strategy)
def test_truncation_is_idempotent_on_a_text_that_fits(description: str, catalog_chars: int) -> None:
    """Property 7: a cut description already fits, so re-cutting it returns it unchanged.

    Validates: Requirements 4.1, 4.3.
    """
    cut = truncate_description(description, catalog_chars)
    assert len(cut) <= catalog_chars
    assert truncate_description(cut, catalog_chars) == cut


@PROPERTY_SETTINGS
@given(description=description_strategy, catalog_chars=chars_strategy)
def test_cutting_a_description_leaves_the_source_text_alone(description: str, catalog_chars: int) -> None:
    """Property 7: the registered description is only read — the cut returns a new string.

    Validates: Requirement 4.8.
    """
    source = description
    truncate_description(description, catalog_chars)
    assert description == source


def test_the_default_budget_is_counted_in_characters() -> None:
    """The default is a character count, not a token count: eighty characters, about twenty tokens."""
    assert DEFAULT_CATALOG_CHARS == 80
    assert estimate_tokens("a" * DEFAULT_CATALOG_CHARS) == 20


# --------------------------------------------------------------------------------------------------
# A summarizer's answer is normalized to one line inside the budget, whatever it returns.
# --------------------------------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(answer=description_strategy, catalog_chars=chars_strategy)
def test_a_clamped_line_is_always_one_line_inside_the_budget(answer: str, catalog_chars: int) -> None:
    """Property 7: whatever the summarizer answers, the catalog line fits and holds no newline.

    A newline would break the ``- name: summary`` grammar of the block, and a run of whitespace would spend
    the budget on nothing; both are removed before the limit is applied.
    """
    line = clamp_summary(answer, catalog_chars)

    assert len(line) <= catalog_chars
    assert "\n" not in line
    assert "  " not in line


@pytest.mark.parametrize(
    ("answered", "expected"),
    [
        ("  Lists  the\n accounts.  ", "Lists the accounts."),
        ('"Lists the accounts."', "Lists the accounts."),
        ("`Lists the accounts.`", "Lists the accounts."),
        ("", ""),
        ("   ", ""),
        (None, ""),
        (42, ""),
        (["Lists the accounts."], ""),
    ],
)
def test_clamping_normalizes_a_models_answer_and_refuses_a_non_string(answered: object, expected: str) -> None:
    """A model asked for a bare line still answers with quotes or newlines sometimes; both are dropped.

    Anything that is not a string is treated as no answer at all, which is what sends the line to the
    truncation fallback rather than into the prompt.
    """
    assert clamp_summary(answered, 80) == expected


# --------------------------------------------------------------------------------------------------
# The per-tool line, and the cache that computes each description exactly once.
# --------------------------------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(description=description_strategy, catalog_chars=chars_strategy)
def test_a_description_that_fits_is_its_own_summary(description: str, catalog_chars: int) -> None:
    """Property 7: a description inside the budget is used verbatim (whitespace collapsed), no call made.

    Validates: Requirement 4.3.
    """
    line = summary_line(_spec("t", description), catalog_chars)

    assert len(line) <= max(catalog_chars, len(" ".join(description.split())))
    if len(description) <= catalog_chars:
        assert line == " ".join(description.split())
    else:
        assert line == truncate_description(description, catalog_chars)


def test_a_description_is_summarized_once_and_cached() -> None:
    """The cache is keyed by ``(name, description)``, so the same description never costs a second call."""
    calls: list[str] = []

    def summarizer(spec: dict[str, Any], max_chars: int) -> str:
        calls.append(spec["name"])
        return "A model-written line."

    long = "x " * 200
    cache = SummaryCache(summarizer)
    spec = _spec("list_accounts", long)

    assert cache.line(spec, 80) == "A model-written line."
    assert cache.line(spec, 80) == "A model-written line."
    assert cache.line(_spec("list_accounts", long), 80) == "A model-written line."
    assert calls == ["list_accounts"], "the same (name, description) pair was summarized twice"
    assert len(cache) == 1

    # A re-registration with a different description is a different key, so it gets its own line.
    assert cache.line(_spec("list_accounts", long + "changed"), 80) == "A model-written line."
    assert calls == ["list_accounts", "list_accounts"]
    assert len(cache) == 2


def test_a_short_description_never_reaches_the_summarizer() -> None:
    """A line that already fits is the best summary of itself, and costs nothing."""
    calls: list[str] = []

    def summarizer(spec: dict[str, Any], max_chars: int) -> str:
        calls.append(spec["name"])
        return "never used"

    cache = SummaryCache(summarizer)

    assert cache.line(_spec("audit_log", "Reads the audit log."), 80) == "Reads the audit log."
    assert calls == []


def test_a_summarizer_that_fails_falls_back_to_truncation() -> None:
    """No tool is ever left without a line, so a raising summarizer degrades instead of propagating."""

    def summarizer(spec: dict[str, Any], max_chars: int) -> str:
        raise RuntimeError("model unavailable")

    long = "Moves money between two accounts, with a four-eyes approval and a hard daily limit applied."
    cache = SummaryCache(summarizer)

    assert cache.line(_spec("wire_transfer", long), 40) == truncate_description(long, 40)


def test_a_summarizer_answer_over_the_budget_is_clamped() -> None:
    """A summarizer that overruns cannot overrun the catalog: the clamp is applied to what it returned."""
    cache = SummaryCache(lambda spec, max_chars: "y" * 500)

    assert len(cache.line(_spec("t", "z" * 300), 30)) == 30


def test_a_primed_line_is_used_and_clamped() -> None:
    """A framework layer with a model writes the line elsewhere and hands it over; it is still clamped."""
    cache = SummaryCache()
    long = "q" * 300

    assert cache.prime("wire_transfer", long, "  Moves\nmoney.  ") == "Moves money."
    assert cache.get(_spec("wire_transfer", long)) == "Moves money."
    assert cache.line(_spec("wire_transfer", long), 80) == "Moves money."


def test_the_cache_reads_the_specifications_only() -> None:
    """Computing a line must not disturb the specification it was computed from.

    Validates: Requirement 4.8.
    """
    specs = [_spec("list_accounts", "Lists the accounts."), _spec("audit_log", "r " * 200)]
    source = copy.deepcopy(specs)

    SummaryCache().lines_for(specs, 40)

    assert specs == source


# --------------------------------------------------------------------------------------------------
# The block: the rule, then one ``- name: summary`` line per tool that is not callable on this call.
# --------------------------------------------------------------------------------------------------


def test_the_block_lists_every_tool_that_is_not_carrying_a_full_specification() -> None:
    """Property 8: the block is exactly the complement of the active tool list, in arrival order."""
    incoming = [
        _spec(FIND_TOOLS_NAME, "Search for tools."),
        _spec(GET_TOOL_DETAILS_NAME, "Load tools."),
        _spec("list_accounts", "Lists the accounts."),
        _spec("wire_transfer", "Moves money."),
        _spec("audit_log", "Reads the audit log."),
    ]
    summaries = {
        "list_accounts": "Lists the accounts.",
        "wire_transfer": "Moves money.",
        "audit_log": "Reads the audit log.",
    }

    block = catalog_prompt_block(incoming, {FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME, "list_accounts"}, summaries)

    assert _listed_names(block) == ["wire_transfer", "audit_log"]
    assert "- wire_transfer: Moves money." in block
    assert "- audit_log: Reads the audit log." in block


@PROPERTY_SETTINGS
@given(name=name_strategy, summary=description_strategy)
def test_a_listed_name_is_verbatim_and_carries_its_cached_line(name: str, summary: str) -> None:
    """Property 7: the name is the call key, so it is copied character for character into the line.

    Validates: Requirement 4.5.
    """
    block = catalog_prompt_block([_spec(name, "Whatever the registry holds.")], set(), {name: summary})

    expected = f"- {name}: {summary}" if summary else f"- {name}"
    assert block.endswith(expected)


@PROPERTY_SETTINGS
@given(
    listed=st.lists(st.sampled_from(["list_accounts", "wire_transfer", "audit_log", "send_email"]), unique=True),
    extra_full=st.lists(st.sampled_from(["read_document", "parse_pdf"]), unique=True),
)
def test_the_block_and_the_active_tool_list_partition_the_registered_names(
    listed: list[str], extra_full: list[str]
) -> None:
    """Property 8: no name is in both places, and no name is missing from both.

    That is the whole correctness claim of the placement: a name the model reads about is either callable
    now, or listed with the rule for making it callable — never silently absent.
    """
    plugin_tools = [FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME]
    names = plugin_tools + listed + extra_full
    incoming = [_spec(name, f"Does {name}.") for name in names]
    full = set(plugin_tools) | set(extra_full)
    summaries = {name: f"Does {name}." for name in listed}

    block = catalog_prompt_block(incoming, full, summaries)
    in_block = set(_listed_names(block))

    assert not (in_block & full), "a name is both callable and listed as unavailable"
    assert in_block | full == set(names)


def test_a_tool_without_a_summary_is_listed_by_name_alone() -> None:
    """A missing line must not silence the name: the model can still load the tool and look.

    Validates: Requirement 4.9.
    """
    incoming = [_spec(FIND_TOOLS_NAME, "Search."), _spec("audit_log", "Reads the audit log.")]

    block = catalog_prompt_block(incoming, {FIND_TOOLS_NAME}, {})

    assert "- audit_log" in block
    assert "- audit_log:" not in block


def test_an_empty_catalog_adds_no_header() -> None:
    """A header promising a list, with no list under it, is a false statement about the tool set."""
    assert catalog_prompt_block([], set(), {}) == ""

    incoming = [_spec("list_accounts", "Lists the accounts.")]
    assert catalog_prompt_block(incoming, {"list_accounts"}, {"list_accounts": "Lists."}) == ""


def test_the_block_names_both_disclosure_tools_so_the_rule_is_actionable() -> None:
    """A listing the model cannot act on is worse than no listing: the header must name the way out.

    Both names are required, in their respective roles: ``get_tool_details`` is the common path off the
    catalog, and ``find_tools`` the fallback for a need no listed name fits.
    """
    block = catalog_prompt_block([_spec("audit_log", "Reads the audit log.")], set(), {"audit_log": "Reads."})

    assert GET_TOOL_DETAILS_NAME in block
    assert FIND_TOOLS_NAME in block
    assert block.startswith(
        CATALOG_PROMPT_HEADER.format(find_tools=FIND_TOOLS_NAME, get_tool_details=GET_TOOL_DETAILS_NAME)
    )


def test_the_block_only_reads_the_incoming_specifications() -> None:
    """Rendering the catalog must not disturb the specifications the active tool list is built from.

    Validates: Requirement 4.8.
    """
    incoming = [_spec("list_accounts", "Lists the accounts."), _spec("audit_log", "Reads the audit log.")]
    source = copy.deepcopy(incoming)

    catalog_prompt_block(incoming, {"list_accounts"}, {"audit_log": "Reads."})

    assert incoming == source


# --------------------------------------------------------------------------------------------------
# build_catalog: the one entry point a framework layer calls.
# --------------------------------------------------------------------------------------------------


def test_build_catalog_renders_a_budgeted_line_per_inactive_tool() -> None:
    """The name plus a one-line summary of its description, inside the budget, for everything not callable."""
    long = "Moves money between two accounts, with a four-eyes approval and a hard daily limit applied."
    specs = [
        _spec(FIND_TOOLS_NAME, "Search for tools."),
        _spec(GET_TOOL_DETAILS_NAME, "Load tools."),
        _spec("list_accounts", "Lists the accounts."),
        _spec("wire_transfer", long),
    ]

    block = build_catalog(specs, 40, active_tool_names=PLUGIN_TOOL_NAMES)

    assert _listed_names(block) == ["list_accounts", "wire_transfer"]
    assert "- list_accounts: Lists the accounts." in block
    assert f"- wire_transfer: {truncate_description(long, 40)}" in block
    for line in (line for line in block.splitlines() if line.startswith("- ")):
        assert len(line.split(": ", 1)[1]) <= 40 if ": " in line else True


def test_build_catalog_suppressed_by_none_adds_nothing() -> None:
    """``catalog_chars=None`` is a supported configuration: no catalog at all, not an empty header."""
    assert build_catalog([_spec("list_accounts", "Lists the accounts.")], None) == ""


def test_build_catalog_prefers_given_summaries_over_the_cache() -> None:
    """A layer that already has the lines passes them straight through, and the cache is not consulted."""
    cache = SummaryCache(lambda spec, max_chars: pytest.fail("the cache must not be consulted"))
    specs = [_spec("audit_log", "z" * 300)]

    block = build_catalog(specs, 80, summaries={"audit_log": "Reads the audit log."}, cache=cache)

    assert block.endswith("- audit_log: Reads the audit log.")
    assert len(cache) == 0


def test_build_catalog_shares_one_cache_across_calls() -> None:
    """The block is byte-stable across calls, and the description behind it is summarized exactly once."""
    calls: list[str] = []

    def summarizer(spec: dict[str, Any], max_chars: int) -> str:
        calls.append(spec["name"])
        return "Moves money between accounts."

    cache = SummaryCache(summarizer)
    specs = [_spec(FIND_TOOLS_NAME, "Search."), _spec("wire_transfer", "w " * 200)]

    first = build_catalog(specs, 80, active_tool_names={FIND_TOOLS_NAME}, cache=cache)
    second = build_catalog(specs, 80, active_tool_names={FIND_TOOLS_NAME}, cache=cache)

    assert first == second
    assert calls == ["wire_transfer"]


def test_build_catalog_with_every_tool_active_is_empty() -> None:
    """Nothing left to list means nothing added to the prompt."""
    specs = [_spec("list_accounts", "Lists the accounts.")]

    assert build_catalog(specs, 80, active_tool_names={"list_accounts"}) == ""


# --------------------------------------------------------------------------------------------------
# The fold: a closed exchange of a tool this call cannot call becomes one sentence.
# --------------------------------------------------------------------------------------------------


def _exchange(tool_use_id: str, name: str) -> list[dict[str, Any]]:
    """One assistant call to ``name`` and the user message carrying its result."""
    return [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": "ok"}]}}]},
    ]


_ACTIVE = {FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME}
"""What the call's tool list carries once every domain tool has been released."""


def test_a_released_tool_exchange_becomes_one_sentence_and_disclosure_pairs_vanish() -> None:
    """No call shape left to repeat: the result survives as text, the arguments do not."""
    question = {"role": "user", "content": [{"text": "wire 10 to account 1"}]}
    messages = [
        question,
        *_exchange("t1", GET_TOOL_DETAILS_NAME),
        *_exchange("t2", "send_wire"),
        {"role": "assistant", "content": [{"text": "Done."}]},
        {"role": "user", "content": [{"text": "and the balance?"}]},
    ]
    original = copy.deepcopy(messages)

    folded = fold_closed_exchanges(messages, _ACTIVE)

    assert folded == [
        {
            "role": "user",
            "content": [question["content"][0], {"text": "The tool send_wire was called and the result was: ok"}],
        },
        *messages[-2:],
    ]
    assert not any("toolUse" in block for message in folded for block in message["content"])
    assert messages == original


def test_an_active_tool_keeps_its_tool_form() -> None:
    """A tool still in the call's tool list may be called again, so its exchange is left as it is."""
    messages = [
        {"role": "user", "content": [{"text": "q"}]},
        *_exchange("t1", "send_wire"),
        {"role": "assistant", "content": [{"text": "Done."}]},
        {"role": "user", "content": [{"text": "next"}]},
    ]
    assert fold_closed_exchanges(messages, {*_ACTIVE, "send_wire"}) is messages


def test_the_exchange_in_flight_is_kept_and_mixed_calls_keep_their_active_part() -> None:
    """The turn in flight stays in tool form; in a mixed call only the unreachable blocks fold."""
    in_flight = [{"role": "user", "content": [{"text": "q"}]}, *_exchange("t1", "send_wire")]
    assert fold_closed_exchanges(in_flight, _ACTIVE) is in_flight

    mixed = [
        {"role": "user", "content": [{"text": "q"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "t1", "name": "send_wire", "input": {"account": "1"}}},
                {"toolUse": {"toolUseId": "t2", "name": "check_balance", "input": {"account": "1"}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "t1", "status": "error", "content": [{"text": "limit"}]}},
                {"toolResult": {"toolUseId": "t2", "content": [{"json": {"balance": 10}}]}},
            ],
        },
        {"role": "assistant", "content": [{"text": "10"}]},
        {"role": "user", "content": [{"text": "thanks"}]},
    ]
    folded = fold_closed_exchanges(mixed, {*_ACTIVE, "check_balance"})

    assert [m["role"] for m in folded] == ["user", "assistant", "user", "assistant", "user"]
    assert folded[1]["content"] == [mixed[1]["content"][1]]
    # The answer to the active call comes first: a provider rejects text ahead of a toolResult.
    assert folded[2]["content"] == [
        mixed[2]["content"][1],
        {"text": "The tool send_wire was called and failed with: limit"},
    ]


def _reasoning(text: str) -> dict[str, Any]:
    """A signed reasoning block, as a thinking model returns it."""
    return {"reasoningContent": {"reasoningText": {"text": text, "signature": "sig"}}}


def test_the_turn_in_flight_reaches_a_reasoning_model_as_the_same_objects() -> None:
    """A thinking model rejects a modified latest assistant message, so the current turn is never folded."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": "q1"}]},
        {
            "role": "assistant",
            "content": [_reasoning("r1"), {"toolUse": {"toolUseId": "t1", "name": "send_wire", "input": {}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}}]},
        {"role": "assistant", "content": [_reasoning("r2"), {"text": "Sent."}]},
        {"role": "user", "content": [{"text": "q2"}]},
        {
            "role": "assistant",
            "content": [_reasoning("r3"), {"toolUse": {"toolUseId": "t2", "name": "send_wire", "input": {}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t2", "content": [{"text": "ok"}]}}]},
    ]

    folded = fold_closed_exchanges(messages, _ACTIVE)

    # The turn in flight — q2 and its tool loop — is passed through by identity, released tool or not.
    assert list(folded[-3:]) == messages[-3:]
    assert all(a is b for a, b in zip(folded[-3:], messages[-3:], strict=True))
    # The closed turn is folded: the rewritten assistant message lost its reasoning and, emptied, left.
    assert list(folded[:2]) == [
        {"role": "user", "content": [{"text": "q1"}, {"text": "The tool send_wire was called and the result was: ok"}]},
        messages[3],
    ]
    assert folded[1] is messages[3]


def test_non_text_result_parts_survive_the_fold() -> None:
    """An image or document part carries evidence a sentence cannot, so it is kept as it is."""
    image = {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}
    messages = [
        {"role": "user", "content": [{"text": "chart it"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "render_chart", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "rendered"}, image]}}],
        },
        {"role": "assistant", "content": [{"text": "Here."}]},
        {"role": "user", "content": [{"text": "thanks"}]},
    ]

    folded = fold_closed_exchanges(messages, _ACTIVE)

    assert folded[0]["content"] == [
        {"text": "chart it"},
        {"text": "The tool render_chart was called and the result was: rendered"},
        image,
    ]


def test_a_json_result_is_rendered_into_the_sentence() -> None:
    """A structured result still has to reach the model as evidence, so it is serialized into the sentence."""
    messages = [
        {"role": "user", "content": [{"text": "balance?"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "check_balance", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"json": {"balance": 10}}]}}]},
        {"role": "assistant", "content": [{"text": "10"}]},
        {"role": "user", "content": [{"text": "thanks"}]},
    ]

    folded = fold_closed_exchanges(messages, _ACTIVE)

    assert folded[0]["content"][-1] == {
        "text": 'The tool check_balance was called and the result was: {"balance": 10}'
    }


def test_nothing_to_fold_returns_the_same_object() -> None:
    """A history with no closed exchange of a released tool is returned by identity, not copied."""
    messages = [
        {"role": "user", "content": [{"text": "q"}]},
        {"role": "assistant", "content": [{"text": "a"}]},
        {"role": "user", "content": [{"text": "q2"}]},
    ]
    assert fold_closed_exchanges(messages, _ACTIVE) is messages
    assert fold_closed_exchanges([], _ACTIVE) == []


def test_the_disclosure_tools_fold_to_nothing_at_all() -> None:
    """Their own exchanges matter on the next call only: past it they are dead weight, sentence included."""
    messages = [
        {"role": "user", "content": [{"text": "wire 10"}]},
        *_exchange("t1", FIND_TOOLS_NAME),
        *_exchange("t2", GET_TOOL_DETAILS_NAME),
        {"role": "assistant", "content": [{"text": "Loaded."}]},
        {"role": "user", "content": [{"text": "go on"}]},
    ]

    folded = fold_closed_exchanges(messages, _ACTIVE)

    assert [m["role"] for m in folded] == ["user", "assistant", "user"]
    assert folded[0]["content"] == [{"text": "wire 10"}]
    assert "was called and" not in str(folded)


def test_the_drop_set_is_configurable() -> None:
    """A layer naming its tools differently passes its own pair, and nothing is hardcoded against it."""
    messages = [
        {"role": "user", "content": [{"text": "q"}]},
        *_exchange("t1", "load_schemas"),
        {"role": "assistant", "content": [{"text": "Loaded."}]},
        {"role": "user", "content": [{"text": "go on"}]},
    ]

    folded = fold_closed_exchanges(messages, _ACTIVE, drop_names={"load_schemas"})

    assert [m["role"] for m in folded] == ["user", "assistant", "user"]
    assert "was called and" not in str(folded)


def test_the_pair_check_catches_a_use_without_its_result_right_after() -> None:
    """The safety net's predicate: a toolUse must be answered in the very next message."""
    use = {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "send_wire", "input": {}}}]}
    answer = {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}}]}
    question = {"role": "user", "content": [{"text": "q"}]}

    assert pairs_intact([question, use, answer])
    assert not pairs_intact([question, use, question])
    assert not pairs_intact([question, use, {"role": "assistant", "content": [{"text": "x"}]}])


@PROPERTY_SETTINGS
@given(
    released=st.lists(st.sampled_from(["send_wire", "check_balance", "audit_log"]), min_size=1, max_size=4),
    active=st.lists(st.sampled_from(["send_wire", "check_balance", "audit_log"]), unique=True),
)
def test_the_fold_keeps_every_use_answered_and_never_touches_the_input(
    released: list[str], active: list[str]
) -> None:
    """The fold's own invariant holds on its output, and the caller's list comes back unmutated."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": "q"}]}]
    for position, name in enumerate(released):
        messages.extend(_exchange(f"t{position}", name))
    messages.append({"role": "assistant", "content": [{"text": "done"}]})
    messages.append({"role": "user", "content": [{"text": "next"}]})
    source = copy.deepcopy(messages)

    folded = fold_closed_exchanges(messages, {*_ACTIVE, *active})

    assert pairs_intact(list(folded))
    assert messages == source
    # Roles still alternate, which is what a provider requires of a history.
    roles = [m["role"] for m in folded]
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False))
