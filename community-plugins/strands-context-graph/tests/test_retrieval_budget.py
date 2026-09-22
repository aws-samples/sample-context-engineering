"""The three changes that answer a turn which cannot find its evidence.

Measured before any of this existed: a turn spent 31 tool calls alternating between the graph's retrieval tools and a
second plugin's, answered nothing, and on two of five models the same shape ran until the event loop hit Python's
recursion limit -- 346 calls and 21 minutes in one turn. Nothing in the package stopped it, for three reasons, each
covered here:

1. The guidance named every retrieval tool unconditionally, including one the caller had de-registered.
2. ``expand_card`` took a single title, so N collapsed turns cost N round trips, each one growing the prompt.
3. No retrieval budget existed, and no retrieval miss is an error, so "not found" reads as "try differently".
"""

from __future__ import annotations

import pytest

from strands_context_graph.compaction import guidance
from strands_context_graph.plugin import ContextGraph
from strands_context_graph.state import _GraphState
from strands_context_graph.tools import expand_card, find_context

from test_tools import card, graph

CYCLE = 3
"""Cycle the calls are made on. Any value: none of these assertions reads the cycle."""

TTL = 5
"""Fed-back Note TTL. Any value, for the same reason."""


def _state(*titles: str) -> _GraphState:
    """Return a state holding one Subject Card per title.

    Built from the existing ``card`` and ``graph`` helpers so these tests assemble state exactly the way the rest of the
    tool tests do -- a second construction path would drift from the real Card shape.

    Args:
        titles: Titles to create Cards for, in order.

    Returns:
        The state, with every Card a Subject Card so ``expand_card`` can raise it.
    """
    return graph(*(card(title, turn=turn) for turn, title in enumerate(titles)))


# --- 1. the guidance names only what is registered ------------------------------------------------


def test_the_guidance_names_every_registered_tool() -> None:
    """With all three registered the guidance is unchanged in substance: three ways back."""
    text = guidance({"expand_card", "expand_artifact", "find_context"})

    assert "expand_card" in text
    assert "expand_artifact" in text
    assert "find_context" in text


def test_the_guidance_omits_a_de_registered_tool() -> None:
    """The bug this file exists for: advertising a tool the agent cannot call.

    Installing the relevance filter alongside means de-registering ``expand_artifact``, and the model was still told to
    call it -- which is how a turn ends up alternating between paths that cannot answer.
    """
    text = guidance({"expand_card", "find_context"})

    assert "expand_artifact" not in text
    assert "expand_card" in text
    assert "find_context" in text


def test_the_guidance_says_so_when_nothing_is_registered() -> None:
    """No retrieval tool at all: the model is told to answer from the summary, not left to guess."""
    text = guidance(set())

    assert "expand_card" not in text
    assert "answer from it" in text


# --- 2. expand_card takes a list ------------------------------------------------------------------


def test_expand_card_raises_every_title_in_one_call() -> None:
    """Three titles, one call, one retrieval cycle -- the point of the batch."""
    state = _state("first", "second", "third")

    answer = expand_card(state, ["first", "second", "third"], cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "'first'" in answer
    assert "'second'" in answer
    assert "'third'" in answer
    assert set(state.choice.by_title) == {"first", "second", "third"}
    assert state.retrieval_cycles == 1


def test_expand_card_still_accepts_a_bare_string() -> None:
    """The schema says array and a model may send the scalar anyway; answer it rather than correct it."""
    state = _state("first")

    answer = expand_card(state, "first", cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "'first'" in answer
    assert "first" in state.choice.by_title


def test_a_partial_batch_reports_both_halves() -> None:
    """A batch with one bad title raises the good ones and says which title matched nothing."""
    state = _state("first")

    answer = expand_card(state, ["first", "nowhere"], cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "'first'" in answer
    assert "'nowhere'" in answer
    assert "first" in state.choice.by_title
    assert "nowhere" not in state.choice.by_title


def test_a_batch_of_only_bad_titles_changes_no_resolution() -> None:
    """Every title unknown: an error naming them, and the choice is left exactly as it was."""
    state = _state("first")
    before = state.choice

    answer = expand_card(state, ["nowhere", "nothing"], cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "no earlier turn" in answer
    assert state.choice is before


def test_an_empty_list_asks_for_titles() -> None:
    """A call with nothing in it is answered with what to pass, not with a traceback."""
    answer = expand_card(_state("first"), [], cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "no title given" in answer


# --- 3. the retrieval ceiling ---------------------------------------------------------------------


def test_the_ceiling_admits_exactly_its_budget() -> None:
    """A ceiling of ``n`` lets ``n`` calls through and refuses the next one."""
    state = _state("first")

    for _ in range(2):
        answer = expand_card(state, ["first"], cycle=CYCLE, reuse_ttl_cycles=TTL, max_retrieval_cycles=2)
        assert "arrives in full" in answer

    refused = expand_card(state, ["first"], cycle=CYCLE, reuse_ttl_cycles=TTL, max_retrieval_cycles=2)

    assert "already spent" in refused
    assert "answer from what" in refused


def test_the_refusal_does_not_spend_more_budget() -> None:
    """A refused call leaves the counter alone, so the message stays truthful however often it is retried."""
    state = _state("first")
    expand_card(state, ["first"], cycle=CYCLE, reuse_ttl_cycles=TTL, max_retrieval_cycles=1)

    for _ in range(3):
        expand_card(state, ["first"], cycle=CYCLE, reuse_ttl_cycles=TTL, max_retrieval_cycles=1)

    assert state.retrieval_cycles == 1


def test_the_ceiling_is_shared_across_the_retrieval_tools() -> None:
    """One budget per turn, not one per tool: alternating between them is the behaviour being stopped."""
    state = _state("first")
    expand_card(state, ["first"], cycle=CYCLE, reuse_ttl_cycles=TTL, max_retrieval_cycles=1)

    refused = find_context(
        state,
        "anything",
        matcher=_NeverCalled(),
        collapse_floor=0.1,
        cycle=CYCLE,
        reuse_ttl_cycles=TTL,
        max_retrieval_cycles=1,
    )

    assert "already spent" in refused


def test_none_restores_unbounded_retrieval() -> None:
    """The opt-out is explicit, because a scenario that genuinely walks many turns must stay possible."""
    state = _state("first")

    for _ in range(20):
        answer = expand_card(state, ["first"], cycle=CYCLE, reuse_ttl_cycles=TTL, max_retrieval_cycles=None)

    assert "arrives in full" in answer
    assert state.retrieval_cycles == 20


class _NeverCalled:
    """A matcher that fails the test if the ceiling let the call through to scoring."""

    def score(self, *args: object, **kwargs: object) -> list[float]:
        """Raise: reaching the matcher means the refusal did not happen.

        Args:
            args: Ignored.
            kwargs: Ignored.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("the ceiling should have refused before any scoring")


# --- the plugin surface ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "8"])
def test_the_constructor_rejects_a_bad_ceiling(bad: object) -> None:
    """Validated like every other count, and before anything is assigned."""
    with pytest.raises(ValueError, match="max_retrieval_cycles"):
        ContextGraph(max_retrieval_cycles=bad)  # type: ignore[arg-type]


def test_the_constructor_accepts_none_and_a_positive_integer() -> None:
    """Both documented values build an instance that registered nothing yet."""
    assert ContextGraph(max_retrieval_cycles=None) is not None
    assert ContextGraph(max_retrieval_cycles=1) is not None
