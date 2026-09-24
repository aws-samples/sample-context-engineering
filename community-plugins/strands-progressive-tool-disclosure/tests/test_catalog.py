"""Property tests for the catalog: the budgeted line, and its placement in the system prompt.

Feature: progressive-tool-disclosure-plugin, Property 7: A catalog line is a faithful, budgeted
description of the tool, and the name it is listed under is verbatim.
Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.8, 4.9.

Feature: progressive-tool-disclosure-plugin, Property 8: The catalog and ``tool_specs`` partition the
registry -- every incoming name is either callable on this call or listed in the prompt, never both and
never neither.

The catalog lives ONLY in the system prompt now: there is no reduced entry in ``tool_specs``, no sigil
prefixing a description, and no flag choosing between the two placements. The budget is counted in
CHARACTERS (``catalog_chars``), and the line is a summary -- truncation at a sentence or word boundary is
the fallback for a summary that could not be produced, and the clamp applied to one that overran.

This file owns the catalog's own surface: the character budget, the normalization of a line, the block
rendered into the prompt, and the configuration that governs both. What the line CONTAINS on each branch
-- verbatim, summarized, truncated -- and what the summary costs are asserted against a real agent in
``tests/test_details_and_summary.py``, so they are not re-derived here.

Every property reads the real helpers (``_catalog_prompt_block``, ``_clamp_summary``,
``_truncate_description``, ``_append_to_system_prompt``) rather than a double: the claims are about the
line the implementation actually writes, so a stand-in would assert nothing. Generated descriptions mix
sentence terminators, long unbroken runs and the empty string, so every branch of the fallback cut --
sentence boundary, word boundary with an ellipse, and the character-count fallback -- is reachable from
the same strategy.

No network: nothing here calls a model, and an autouse fixture makes any outbound socket connection
raise, so an accidental I/O path in the catalog helpers fails the test instead of silently reaching out.
"""

from __future__ import annotations

import copy
import socket
from typing import Any, cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_progressive_tool_disclosure import ProgressiveToolDisclosure
from strands_progressive_tool_disclosure.plugin import (
    _CATALOG_PROMPT_HEADER,
    _DEFAULT_CATALOG_CHARS,
    _ELLIPSIS,
    FIND_TOOLS_NAME,
    GET_TOOL_DETAILS_NAME,
    _append_to_system_prompt,
    _catalog_prompt_block,
    _clamp_summary,
    _estimate_tokens,
    _truncate_description,
)

SENTENCE_ENDINGS = ".!?"
"""Characters that terminate a sentence, for recognizing a sentence-boundary cut."""

BASE_PROMPT = "You are a helpful assistant."
"""An operator's own system prompt, the text the catalog block must never displace."""

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
    """A full specification as the registry holds one, reduced to what the catalog reads."""
    return {"name": name, "description": description, "inputSchema": {"json": {"type": "object"}}}


def _prompt_text(system_prompt: Any) -> str:
    """Flatten a ``SystemPrompt`` of any shape to one searchable string."""
    if system_prompt is None:
        return ""
    if isinstance(system_prompt, str):
        return system_prompt
    return "\n".join(block.get("text", "") for block in system_prompt)


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
    if not result.endswith(_ELLIPSIS):
        return False
    body = result[: -len(_ELLIPSIS)]
    # The body is a rstripped prefix, so the character just past it is the whitespace that ended the
    # last whole word kept. An empty body is the degenerate case of a leading run of whitespace.
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
    result = _truncate_description(description, catalog_chars)

    # Requirement 4.1: the limit is counted in characters now, ellipse included.
    assert len(result) <= catalog_chars

    if len(description) <= catalog_chars:
        # Requirement 4.3, and Requirement 4.9 for the zero-character description.
        assert result == description
        return

    # Requirement 4.2: a prefix of the source, allowing only an appended ellipse -- no rewriting and
    # no reordering, which ``startswith`` on the stripped body verifies directly.
    body = result[: -len(_ELLIPSIS)] if result.endswith(_ELLIPSIS) else result
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
    cut = _truncate_description(description, catalog_chars)
    assert len(cut) <= catalog_chars
    assert _truncate_description(cut, catalog_chars) == cut


@PROPERTY_SETTINGS
@given(description=description_strategy, catalog_chars=chars_strategy)
def test_cutting_a_description_leaves_the_source_text_alone(description: str, catalog_chars: int) -> None:
    """Property 7: the registered description is only read -- the cut returns a new string.

    Validates: Requirement 4.8.
    """
    source = description
    _truncate_description(description, catalog_chars)
    assert description == source


def test_the_default_budget_is_counted_in_characters() -> None:
    """The default is a character count, not a token count: eighty characters, about twenty tokens."""
    assert _DEFAULT_CATALOG_CHARS == 80
    assert _estimate_tokens("a" * _DEFAULT_CATALOG_CHARS) == 20


# --------------------------------------------------------------------------------------------------
# A summarizer's answer is normalized to one line inside the budget, whatever it returns.
# --------------------------------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(answer=description_strategy, catalog_chars=chars_strategy)
def test_a_clamped_line_is_always_one_line_inside_the_budget(answer: str, catalog_chars: int) -> None:
    """Property 7: whatever the summarizer answers, the catalog line fits and holds no newline.

    A newline would break the ``- name: summary`` grammar of the block, and a run of whitespace would
    spend the budget on nothing; both are removed before the limit is applied.
    """
    line = _clamp_summary(answer, catalog_chars)

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
    assert _clamp_summary(answered, 80) == expected


# --------------------------------------------------------------------------------------------------
# The block: the rule, then one ``- name: summary`` line per tool that is not callable on this call.
# --------------------------------------------------------------------------------------------------


def test_the_block_lists_every_tool_that_is_not_carrying_a_full_specification() -> None:
    """Property 8: the block is exactly the complement of the projection, in arrival order."""
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

    block = _catalog_prompt_block(incoming, {FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME, "list_accounts"}, summaries)

    assert _listed_names(block) == ["wire_transfer", "audit_log"]
    assert "- wire_transfer: Moves money." in block
    assert "- audit_log: Reads the audit log." in block


@PROPERTY_SETTINGS
@given(name=name_strategy, summary=description_strategy)
def test_a_listed_name_is_verbatim_and_carries_its_cached_line(name: str, summary: str) -> None:
    """Property 7: the name is the call key, so it is copied character for character into the line.

    Validates: Requirement 4.5.
    """
    block = _catalog_prompt_block([_spec(name, "Whatever the registry holds.")], set(), {name: summary})

    expected = f"- {name}: {summary}" if summary else f"- {name}"
    assert block.endswith(expected)


@PROPERTY_SETTINGS
@given(
    listed=st.lists(st.sampled_from(["list_accounts", "wire_transfer", "audit_log", "send_email"]), unique=True),
    extra_full=st.lists(st.sampled_from(["read_document", "parse_pdf"]), unique=True),
)
def test_the_block_and_the_projection_partition_the_incoming_names(listed: list[str], extra_full: list[str]) -> None:
    """Property 8: no name is in both places, and no name is missing from both.

    That is the whole correctness claim of the placement: a name the model reads about is either
    callable now, or listed with the rule for making it callable -- never silently absent.
    """
    plugin_tools = [FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME]
    names = plugin_tools + listed + extra_full
    incoming = [_spec(name, f"Does {name}.") for name in names]
    full = set(plugin_tools) | set(extra_full)
    summaries = {name: f"Does {name}." for name in listed}

    block = _catalog_prompt_block(incoming, full, summaries)
    in_block = set(_listed_names(block))

    assert not (in_block & full), "a name is both callable and listed as unavailable"
    assert in_block | full == set(names)


def test_a_tool_without_a_summary_is_listed_by_name_alone() -> None:
    """A missing line must not silence the name: the model can still load the tool and look.

    Validates: Requirement 4.9.
    """
    incoming = [_spec(FIND_TOOLS_NAME, "Search."), _spec("audit_log", "Reads the audit log.")]

    block = _catalog_prompt_block(incoming, {FIND_TOOLS_NAME}, {})

    assert "- audit_log" in block
    assert "- audit_log:" not in block


def test_an_empty_catalog_adds_no_header() -> None:
    """A header promising a list, with no list under it, is a false statement about the tool set."""
    assert _catalog_prompt_block([], set(), {}) == ""

    incoming = [_spec("list_accounts", "Lists the accounts.")]
    assert _catalog_prompt_block(incoming, {"list_accounts"}, {"list_accounts": "Lists."}) == ""


def test_the_block_names_both_plugin_tools_so_the_rule_is_actionable() -> None:
    """A listing the model cannot act on is worse than no listing: the header must name the way out.

    Both names are required, in their respective roles: ``get_tool_details`` is the common path off the
    catalog, and ``find_tools`` the fallback for a need no listed name fits.
    """
    block = _catalog_prompt_block([_spec("audit_log", "Reads the audit log.")], set(), {"audit_log": "Reads."})

    assert GET_TOOL_DETAILS_NAME in block
    assert FIND_TOOLS_NAME in block
    assert block.startswith(
        _CATALOG_PROMPT_HEADER.format(find_tools=FIND_TOOLS_NAME, get_tool_details=GET_TOOL_DETAILS_NAME)
    )


def test_the_block_only_reads_the_incoming_specifications() -> None:
    """Rendering the catalog must not disturb the specifications the projection is built from.

    Validates: Requirement 4.8.
    """
    incoming = [_spec("list_accounts", "Lists the accounts."), _spec("audit_log", "Reads the audit log.")]
    source = copy.deepcopy(incoming)

    _catalog_prompt_block(incoming, {"list_accounts"}, {"audit_log": "Reads."})

    assert incoming == source


# --------------------------------------------------------------------------------------------------
# Placement: the block is appended to the operator's prompt, in the shape the prompt arrived in.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected_type"),
    [(None, str), (BASE_PROMPT, str), ([{"text": BASE_PROMPT}], list)],
)
def test_appending_preserves_the_shape_the_prompt_arrived_in(prompt: Any, expected_type: type) -> None:
    """A caller using the list form places cache checkpoints between blocks; flattening would move them."""
    result = _append_to_system_prompt(prompt, "Block.")

    assert isinstance(result, expected_type)
    assert "Block." in _prompt_text(result)
    if prompt is not None:
        assert BASE_PROMPT in _prompt_text(result)


def test_the_operators_prompt_keeps_the_opening_position() -> None:
    """The block is appended, never substituted, so a prefix cache checkpoint does not move."""
    result = _append_to_system_prompt(BASE_PROMPT, "Block.")

    assert _prompt_text(result).startswith(BASE_PROMPT)


def test_appending_to_a_list_prompt_leaves_the_received_list_untouched() -> None:
    """The received prompt is the caller's object; extending it in place would reach back out."""
    prompt = [{"text": BASE_PROMPT}]

    result = _append_to_system_prompt(prompt, "Block.")

    assert prompt == [{"text": BASE_PROMPT}]
    assert result == [{"text": BASE_PROMPT}, {"text": "Block."}]


def test_appending_an_empty_block_returns_the_prompt_unchanged() -> None:
    """By identity, so a call that places nothing cannot invalidate a cached prefix."""
    prompt = [{"text": BASE_PROMPT}]

    assert _append_to_system_prompt(prompt, "") is prompt


# --------------------------------------------------------------------------------------------------
# Configuration: the character limit is validated, and the removed knobs are gone for good.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("catalog_chars", [0, -1, True, 1.0, "80", object()])
def test_a_catalog_limit_that_is_not_a_positive_integer_is_refused(catalog_chars: object) -> None:
    """Construction validates before any handler is registered, so a typo fails loudly and early.

    ``0`` is refused with the rest: a limit of zero characters would emit lines nothing fits in.
    """
    with pytest.raises(ValueError, match="catalog_chars"):
        ProgressiveToolDisclosure(catalog_chars=cast(Any, catalog_chars))


@pytest.mark.parametrize("catalog_chars", [None, 1, 80, 400])
def test_a_valid_catalog_limit_and_a_suppressed_catalog_are_both_accepted(catalog_chars: int | None) -> None:
    """``None`` is a supported configuration: no catalog at all, and the two plugin tools as the hint."""
    assert ProgressiveToolDisclosure(catalog_chars=catalog_chars)._catalog_chars == catalog_chars


def test_a_summarizer_that_is_not_callable_is_refused() -> None:
    """The summarizer is called per tool on the projection path, so a wrong type fails at construction."""
    with pytest.raises(ValueError, match="summarizer"):
        ProgressiveToolDisclosure(summarizer=cast(Any, "use the model"))


@pytest.mark.parametrize("removed", ["catalog_tokens", "catalog_in_system_prompt"])
def test_the_removed_configuration_knobs_are_gone(removed: str) -> None:
    """The catalog is counted in characters, and it is only ever in the system prompt."""
    with pytest.raises(TypeError, match=removed):
        ProgressiveToolDisclosure(**{removed: 20})


@pytest.mark.parametrize("removed", ["_catalog_entry", "_CATALOG_SIGIL", "_DEFAULT_CATALOG_TOKENS"])
def test_the_reduced_catalog_entry_no_longer_exists(removed: str) -> None:
    """Nothing in ``tool_specs`` asserts an empty schema now, so the helpers that built one are gone."""
    from strands_progressive_tool_disclosure import plugin

    assert not hasattr(plugin, removed)
