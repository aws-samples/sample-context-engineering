"""Property tests for the verbatim copy of numeric lines into a Description.

Feature: context-graph-plugin, Property 5: Numeric lines are copied verbatim, never paraphrased.

Validates: Requirements 4.3, 4.4.

The claim is checkable by substring, which is why it is worth a property test: whatever the generated conversation,
every numeric line the Description carries is character-for-character a line of the messages it was drawn from.
Generation goes through the real derivation path — ``derive_card`` over a rendered history — so the assertion covers
the scan, the selection and the composition together rather than ``numeric_lines`` alone.

Three claims per generated history:

- every selected line is an exact line of the source texts, leading whitespace included;
- an unbounded Description is its header followed by exactly those lines, unchanged and in order;
- a bounded Description keeps a leading run of them, still verbatim, with the omitted count accounted for. Under a
  budget too tight for even the header, the Description is a literal prefix of the complete one, so its final line may
  be cut short — a prefix, never a paraphrase.
"""

import re
from dataclasses import replace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_context_graph.cards import derive_card
from strands_context_graph.describe import compose_description
from strands_context_graph.state import Card

UNBOUNDED = 10_000
"""A budget no Description built here can reach, for reading the untruncated form."""

DERIVATION = {"description_tokens": UNBOUNDED, "tags_per_card": 5, "rarity_weight": 0.5}

OMISSION = re.compile(r"^\(\+(\d+) numeric lines omitted\)$")
"""What the Description appends when the selected lines overflow the budget. No generated line can look like it."""

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

word_strategy = st.text(alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz")), min_size=3, max_size=8)

amount_strategy = st.integers(min_value=0, max_value=999_999)

indent_strategy = st.sampled_from(["", " ", "   ", "\t"])

# Lines shaped the way the pattern is meant to select: a bare number, a monetary amount in either separator convention,
# a tabular row, an exponent and a negative adjustment. Indentation is generated too, a stripped row no longer lining up
# with the row above it.
numeric_line_strategy = st.builds(
    lambda indent, body: indent + body,
    indent_strategy,
    st.one_of(
        st.builds("{}: {}".format, word_strategy, amount_strategy),
        st.builds("{} R$ {}.{:03d},00".format, word_strategy, amount_strategy, amount_strategy.map(lambda n: n % 1000)),
        st.builds(
            "{} US$ {},{:03d}.50".format, word_strategy, amount_strategy, amount_strategy.map(lambda n: n % 1000)
        ),
        st.builds("| {} | {} | {} |".format, word_strategy, amount_strategy, amount_strategy),
        st.builds("{}\t{}\t{}".format, word_strategy, amount_strategy, amount_strategy),
        st.builds("{} {}.5e3 requests".format, word_strategy, amount_strategy),
        st.builds("{} -{} adjustments".format, word_strategy, amount_strategy),
    ),
)

# Prose carrying no digit at all, so it is never selected and never confused with a numeric line.
prose_line_strategy = st.lists(word_strategy, min_size=1, max_size=6).map(" ".join)

text_strategy = st.lists(st.one_of(numeric_line_strategy, prose_line_strategy), min_size=1, max_size=8).map("\n".join)

step_strategy = st.tuples(st.sampled_from(["assistant", "tool_result"]), text_strategy)

turn_strategy = st.tuples(prose_line_strategy, st.lists(step_strategy, max_size=3))


def _render(turns: list[tuple[str, list[tuple[str, str]]]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Render structural turns into a history, returning it alongside the texts placed in it.

    Durable identities are assigned sequentially here rather than inside a strategy: they have to be unique across the
    whole conversation, which a per-value strategy cannot promise.

    Args:
        turns: ``(user text, steps)`` pairs, each step a ``(kind, text)`` pair.

    Returns:
        ``(messages, texts)``. ``texts`` holds every text written into a block, in message order, exactly as written.
    """
    messages: list[dict[str, Any]] = []
    texts: list[str] = []
    counter = 0

    def identity() -> str:
        nonlocal counter
        counter += 1
        return f"m{counter}"

    for turn, (ask, steps) in enumerate(turns):
        messages.append({"role": "user", "content": [{"text": ask}], "tracking_id": identity()})
        texts.append(ask)

        for index, (kind, text) in enumerate(steps):
            tool_use_id = f"tu-{turn}-{index}"
            if kind == "assistant":
                messages.append({"role": "assistant", "content": [{"text": text}], "tracking_id": identity()})
                texts.append(text)
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": tool_use_id, "name": "get_balance", "input": {}}}],
                    "tracking_id": identity(),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "status": "success",
                                "content": [{"text": text}],
                            }
                        }
                    ],
                    "tracking_id": identity(),
                }
            )
            texts.append(text)

    return messages, texts


# At least two turns, so the first one is closed and has a Card to derive.
history_strategy = st.lists(turn_strategy, min_size=2, max_size=4).map(_render)


def _first_card(messages: list[dict[str, Any]], texts: list[str]) -> tuple[Card, list[str]]:
    """Derive the Card of the first turn, alongside the source lines that turn's messages carry.

    Args:
        messages: The rendered history. Read only.
        texts: Every text written into the history, in message order.

    Returns:
        ``(card, source_lines)``, where ``source_lines`` holds the lines of the texts the first turn's Card scanned.
    """
    boundaries = [index for index, message in enumerate(messages) if message.get("role") == "user" and _is_ask(message)]
    stop = boundaries[1]
    turn_ids = [message["tracking_id"] for message in messages[:stop] if message.get("tracking_id")]

    # Texts follow message order, and every message of the first turn carries exactly one of them.
    consumed = sum(1 for message in messages[:stop] if message.get("content") and _carries_text(message))
    source_lines = [line for text in texts[:consumed] for line in text.splitlines()]

    return derive_card(messages, turn_ids, 0, **DERIVATION), source_lines  # type: ignore[arg-type]


def _is_ask(message: dict[str, Any]) -> bool:
    """Report whether ``message`` opens a turn: a user message with no ``toolResult`` block."""
    return all("toolResult" not in block for block in message.get("content", []))


def _carries_text(message: dict[str, Any]) -> bool:
    """Report whether ``message`` carries a text block, directly or inside a ``toolResult``."""
    for block in message.get("content", []):
        if "text" in block:
            return True
        result = block.get("toolResult")
        if isinstance(result, dict):
            if any("text" in inner for inner in result.get("content", []) or []):
                return True
    return False


def _header_of(card: Card) -> str:
    """Compose ``card``'s Description header alone, by describing it with no numeric line to append."""
    return compose_description(replace(card, numeric_lines=()), UNBOUNDED)


def _tail_of(description: str, header: str) -> list[str]:
    """Take the lines a Description appends after its header.

    Args:
        description: The composed Description, which must start with ``header``.
        header: The Description's header.

    Returns:
        The appended lines, in order, the omission line included when there is one.
    """
    remainder = description[len(header) :]

    return remainder.split("\n", 1)[1].split("\n") if remainder else []


@given(rendered=history_strategy)
@PROPERTY_SETTINGS
def test_every_selected_numeric_line_is_an_exact_line_of_the_source_text(
    rendered: tuple[list[dict[str, Any]], list[str]],
) -> None:
    """Feature: context-graph-plugin, Property 5: Numeric lines are copied verbatim, never paraphrased.

    Validates: Requirements 4.3, 4.4.
    """
    card, source_lines = _first_card(*rendered)

    for line in card.numeric_lines:
        assert line in source_lines

    # No duplication and no reordering: the selection is the source order, filtered.
    positions = [source_lines.index(line) for line in card.numeric_lines]
    assert positions == sorted(positions)
    assert len(set(card.numeric_lines)) == len(card.numeric_lines)


@given(rendered=history_strategy)
@PROPERTY_SETTINGS
def test_an_unbounded_description_appends_its_numeric_lines_unchanged(
    rendered: tuple[list[dict[str, Any]], list[str]],
) -> None:
    """Feature: context-graph-plugin, Property 5: Numeric lines are copied verbatim, never paraphrased.

    Validates: Requirements 4.3, 4.4.
    """
    card, source_lines = _first_card(*rendered)

    description = compose_description(card, UNBOUNDED)
    header = _header_of(card)

    assert description == "\n".join((header, *card.numeric_lines))
    for line in _tail_of(description, header):
        assert line in source_lines


@given(rendered=history_strategy, description_tokens=st.integers(min_value=1, max_value=80))
@PROPERTY_SETTINGS
def test_a_bounded_description_keeps_a_verbatim_leading_run_of_the_numeric_lines(
    rendered: tuple[list[dict[str, Any]], list[str]],
    description_tokens: int,
) -> None:
    """Feature: context-graph-plugin, Property 5: Numeric lines are copied verbatim, never paraphrased.

    Validates: Requirements 4.3, 4.4.
    """
    card, source_lines = _first_card(*rendered)

    description = compose_description(card, description_tokens)
    complete = compose_description(card, UNBOUNDED)
    header = _header_of(card)

    if not description.startswith(header):
        # The header itself did not fit, so nothing was appended and what is left is a literal prefix of the complete
        # Description: no line was rewritten, one was cut short.
        assert complete.startswith(description)
        return

    tail = _tail_of(description, header)
    omitted = 0
    if tail:
        match = OMISSION.match(tail[-1])
        if match:
            omitted = int(match.group(1))
            tail = tail[:-1]

    assert len(tail) + omitted <= len(card.numeric_lines)
    if omitted:
        assert len(tail) + omitted == len(card.numeric_lines)

    cut_short = complete.startswith(description)
    for index, line in enumerate(tail):
        expected = card.numeric_lines[index]
        if line == expected:
            assert line in source_lines
            continue

        # The one tolerated difference: the final line of a Description that is itself a literal prefix of the complete
        # form may stop mid-line. A prefix, never a paraphrase.
        assert cut_short
        assert index == len(tail) - 1
        assert expected.startswith(line)
