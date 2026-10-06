"""Unit tests for the three retrieval tools.

Three things are asserted here that no other test file states:

- **Every path answers.** Every miss — an unknown Title, an unknown reference, non-textual content, no storage at all,
  an empty need, an unusable matcher, an unknown tag, nothing clearing the floor — comes back as prose naming what was
  missing, and none of them raises (Requirements 12.3, 12.6, 12.7, 12.12, 16.8).
- **What a miss must not do.** An error changes no Resolution and records no fed-back Note (Requirement 13.8), asserted
  as the Resolution being the same object and ``state.reuse`` being empty rather than as "the message looks like an
  error".
- **No model call, on any path.** The matcher is a table and the agent stub trips on every attribute a model call would
  travel through, so reaching for one fails the test rather than quietly costing a token (Requirement 12.13).

``find_context``'s bound, floor and ordering are asserted with a table matcher whose values are chosen so the resulting
order is decidable by hand, including the two tie-breaks: the turn ordinal, then the Title.
"""

from collections.abc import Sequence
from types import MappingProxyType

import pytest

from strands_context_graph.state import Card, CardChoice, ToolPair, TurnChoice, _GraphState
from strands_context_graph.store import InMemoryReferenceStore, absent_message, non_textual_message, unknown_message
from strands_context_graph.tools import _MAX_CANDIDATES, expand_artifact, expand_card, find_context

CYCLE = 7
"""One cycle counter, so an expiry cycle is checkable by hand."""

TTL = 5
"""One TTL, long enough that a recorded fed-back Note is observable."""

_MODEL_ATTRS = frozenset(
    {"model", "invoke_async", "stream_async", "structured_output", "structured_output_async", "converse"}
)
"""Every attribute a model call on an agent would travel through. Reaching for one during a retrieval is a bug."""


class Agent:
    """An agent stub carrying no plugin registry, so the optional Stash bridge is simply absent."""

    _plugin_registry = None

    def __getattr__(self, name: str) -> object:
        """Fail loudly on any attribute a model call would need."""
        if name in _MODEL_ATTRS:
            raise AssertionError(f"a retrieval tool reached for agent.{name}")
        raise AttributeError(name)


class TableMatcher:
    """A matcher answering from a table, counting its invocations."""

    def __init__(self, table: dict[str, float], *, default: float = 0.0) -> None:
        """Answer ``table[description]`` per Description, falling back on ``default``."""
        self.table = table
        self.default = default
        self.calls = 0

    def score(self, need: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description, in the order received."""
        self.calls += 1
        return [self.table.get(description, self.default) for description in descriptions]


class BrokenMatcher:
    """A matcher that raises, standing in for a transport failure or a timeout."""

    def score(self, need: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Raise, which the tool must read as "no candidate" and never propagate."""
        raise RuntimeError("embedding unavailable")


def card(
    title: str,
    *,
    turn: int = 0,
    kind: str = "subject",
    description: str | None = None,
    tags: tuple[str, ...] = (),
    reference: str | None = None,
) -> Card:
    """Build a Card carrying only the fields the tools read."""
    return Card(
        title=title,
        kind=kind,  # type: ignore[arg-type]
        turn=turn,
        dialogue_ids=("m-1",),
        evidence_ids=(),
        pairs=(ToolPair(tool_use_id="tu-1", tool_name="calculator", tracking_ids=("a",), consumed=True),),
        tool_names=frozenset(),
        references=(),
        numeric_lines=(),
        tags=tags,
        description=description if description is not None else f"description of {title}",
        reference=reference if reference is not None else (title if kind == "artifact" else None),
    )


def graph(*cards: Card, full_pass: bool = False) -> _GraphState:
    """Assemble a state around ``cards``, with a decided (non-full-pass) choice by default."""
    state = _GraphState()
    for entry in cards:
        state.cards[entry.title] = entry
    state.turn = max((entry.turn for entry in cards), default=-1) + 1
    if full_pass:
        state.choice = TurnChoice(by_title=MappingProxyType({}), full_pass=True)
    else:
        state.choice = TurnChoice(
            by_title=MappingProxyType(
                {entry.title: CardChoice(dialogue="title", evidence="description") for entry in cards}
            ),
            full_pass=False,
        )
    return state


# ---- expand_card (Requirements 12.2, 12.3, 12.14, 13.1, 13.5, 13.8) ----


def test_a_named_card_is_raised_on_both_axes_for_the_rest_of_the_turn() -> None:
    state = graph(card("first", turn=0), card("second", turn=1))

    answer = expand_card(state, "second", cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "'second'" in answer
    assert state.choice.by_title["second"] == CardChoice(dialogue="full", evidence="full")
    # Every other Card keeps the Resolution the Turn Choice decided.
    assert state.choice.by_title["first"] == CardChoice(dialogue="title", evidence="description")
    assert state.choice.full_pass is False
    assert state.reuse == {"second": (1.0, CYCLE + TTL)}


def test_the_rewritten_choice_stays_frozen() -> None:
    state = graph(card("first"))

    expand_card(state, "first", cycle=CYCLE, reuse_ttl_cycles=TTL)

    with pytest.raises(TypeError):
        state.choice.by_title["anything"] = None  # type: ignore[index]


def test_a_full_pass_is_left_alone_so_delivery_keeps_its_identity_short_circuit() -> None:
    state = graph(card("first"), full_pass=True)
    before = state.choice

    expand_card(state, "first", cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert state.choice is before
    assert state.choice.full_pass is True
    # The request still crosses into the next turn, which is the whole point of recording it.
    assert state.reuse == {"first": (1.0, CYCLE + TTL)}


def test_an_unknown_title_answers_without_changing_a_resolution_or_recording_a_note() -> None:
    state = graph(card("first"))
    before = state.choice

    answer = expand_card(state, "nowhere", cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "'nowhere'" in answer
    assert state.choice is before
    assert state.reuse == {}


def test_an_artifact_title_is_not_a_subject_card() -> None:
    state = graph(card("artifact-1", kind="artifact"))
    before = state.choice

    answer = expand_card(state, "artifact-1", cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "no earlier turn" in answer
    assert state.choice is before
    assert state.reuse == {}


def test_a_repeat_request_restarts_the_countdown() -> None:
    state = graph(card("first"))

    expand_card(state, "first", cycle=CYCLE, reuse_ttl_cycles=TTL)
    expand_card(state, "first", cycle=CYCLE + 2, reuse_ttl_cycles=TTL)

    assert state.reuse == {"first": (1.0, CYCLE + 2 + TTL)}


def test_a_ttl_of_zero_records_nothing_but_still_raises_the_resolution() -> None:
    state = graph(card("first"))

    expand_card(state, "first", cycle=CYCLE, reuse_ttl_cycles=0)

    assert state.choice.by_title["first"] == CardChoice(dialogue="full", evidence="full")
    assert state.reuse == {}


def test_every_invocation_costs_a_retrieval_cycle_whether_it_found_anything() -> None:
    state = graph(card("first"))

    expand_card(state, "first", cycle=CYCLE, reuse_ttl_cycles=TTL)
    expand_card(state, "nowhere", cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert state.retrieval_cycles == 2


# ---- expand_artifact (Requirements 12.4, 12.5, 12.6, 12.7) ----

REFERENCE = "mem_1_tu-3_0"
"""One reference, shaped like a real placeholder."""

CONTENT = "line one\nR$ 1.200,00 in March\nline three\nline four"
"""One artifact text, with a numeric line so a pattern read has something to find."""


def stored(text: object = CONTENT) -> InMemoryReferenceStore:
    """A store holding ``text`` under ``REFERENCE``."""
    store = InMemoryReferenceStore()
    store.put(REFERENCE, text)
    return store


@pytest.mark.asyncio
async def test_a_whole_read_returns_the_text_verbatim_with_its_cost_stated() -> None:
    state = graph(card("artifact-1", kind="artifact", reference=REFERENCE))

    answer = await expand_artifact(state, stored(), Agent(), REFERENCE, cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert CONTENT in answer
    assert "whole artifact" in answer
    assert "tokens" in answer
    assert state.reuse == {"artifact-1": (1.0, CYCLE + TTL)}


@pytest.mark.asyncio
async def test_a_targeted_read_is_delegated_and_no_resolution_changes() -> None:
    state = graph(card("artifact-1", kind="artifact", reference=REFERENCE))
    before = state.choice

    answer = await expand_artifact(
        state,
        stored(),
        Agent(),
        REFERENCE,
        {"start": 2, "end": 2},
        cycle=CYCLE,
        reuse_ttl_cycles=TTL,
    )

    assert "R$ 1.200,00 in March" in answer
    assert state.choice is before


@pytest.mark.asyncio
async def test_a_pattern_read_keeps_only_the_matching_lines() -> None:
    state = graph(card("artifact-1", kind="artifact", reference=REFERENCE))

    answer = await expand_artifact(
        state, stored(), Agent(), REFERENCE, None, "March", cycle=CYCLE, reuse_ttl_cycles=TTL
    )

    assert "March" in answer


@pytest.mark.asyncio
async def test_a_malformed_line_range_is_named_and_records_nothing() -> None:
    state = graph(card("artifact-1", kind="artifact", reference=REFERENCE))

    answer = await expand_artifact(
        state,
        stored(),
        Agent(),
        REFERENCE,
        {"from": 1},  # type: ignore[arg-type]
        cycle=CYCLE,
        reuse_ttl_cycles=TTL,
    )

    assert "line_range" in answer
    assert state.reuse == {}


@pytest.mark.asyncio
async def test_an_absent_store_answers_with_the_store_module_prose() -> None:
    state = graph(card("artifact-1", kind="artifact", reference=REFERENCE))
    before = state.choice

    answer = await expand_artifact(
        state, InMemoryReferenceStore(), Agent(), REFERENCE, cycle=CYCLE, reuse_ttl_cycles=TTL
    )

    assert answer == absent_message(REFERENCE)
    assert state.choice is before
    assert state.reuse == {}


@pytest.mark.asyncio
async def test_a_stash_that_does_not_hold_the_reference_answers_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    state = graph(card("artifact-1", kind="artifact", reference=REFERENCE))

    class Stash:
        async def retrieve(self, reference: str) -> object | None:
            return None

    from strands_context_graph import store as store_module

    monkeypatch.setattr(store_module, "_stash_of", lambda agent: Stash())

    answer = await expand_artifact(
        state, InMemoryReferenceStore(), Agent(), REFERENCE, cycle=CYCLE, reuse_ttl_cycles=TTL
    )

    assert answer == unknown_message(REFERENCE)
    assert state.reuse == {}


@pytest.mark.asyncio
async def test_a_non_textual_block_is_named_without_a_media_type() -> None:
    state = graph(card("artifact-1", kind="artifact", reference=REFERENCE))

    answer = await expand_artifact(
        state, stored({"image": {"format": "png"}}), Agent(), REFERENCE, cycle=CYCLE, reuse_ttl_cycles=TTL
    )

    assert answer == non_textual_message(REFERENCE)
    assert state.reuse == {}


@pytest.mark.asyncio
async def test_a_reference_the_graph_never_carded_still_reads() -> None:
    state = graph(card("first"))

    answer = await expand_artifact(state, stored(), Agent(), REFERENCE, cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert CONTENT in answer
    # No artifact Card for the Note to land on, which is not a failure.
    assert state.reuse == {}


# ---- find_context (Requirements 12.8, 12.9, 12.10, 12.11, 12.12) ----


def searchable() -> _GraphState:
    """Seven Cards, so the five-candidate cap is observable, with two pairs tied on similarity."""
    return graph(
        card("alpha", turn=0, description="about invoices", tags=("invoice",)),
        card("bravo", turn=1, description="about payments", tags=("payment",)),
        card("charlie", turn=2, description="about refunds", tags=("refund",)),
        card("delta", turn=3, description="about taxes", tags=("1200",)),
        card("echo", turn=4, description="about payroll"),
        card("foxtrot", turn=5, description="about shipping"),
        card("golf", turn=6, description="about nothing at all"),
    )


def test_candidates_are_ordered_by_descending_similarity_and_capped_at_five() -> None:
    state = searchable()
    matcher = TableMatcher(
        {
            "about invoices": 0.9,
            "about payments": 0.8,
            "about refunds": 0.7,
            "about taxes": 0.6,
            "about payroll": 0.55,
            "about shipping": 0.52,
            "about nothing at all": 0.51,
        }
    )

    answer = find_context(state, "money", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL)

    titles = [line.removeprefix("- title: ") for line in answer.splitlines() if line.startswith("- title: ")]
    assert titles == ["alpha", "bravo", "charlie", "delta", "echo"]
    assert len(titles) == _MAX_CANDIDATES
    assert matcher.calls == 1
    # The fed-back Note lands on exactly what was returned.
    assert set(state.reuse) == set(titles)


def test_ties_are_broken_by_turn_then_title() -> None:
    state = graph(
        card("zulu", turn=0, description="same"),
        card("alpha", turn=1, description="same too"),
        card("bravo", turn=1, description="same again"),
    )
    matcher = TableMatcher({}, default=0.8)

    answer = find_context(state, "money", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL)

    titles = [line.removeprefix("- title: ") for line in answer.splitlines() if line.startswith("- title: ")]
    assert titles == ["zulu", "alpha", "bravo"]


def test_a_candidate_below_the_floor_is_not_returned() -> None:
    state = graph(card("alpha", description="about invoices"), card("bravo", turn=1, description="about payments"))
    matcher = TableMatcher({"about invoices": 0.9, "about payments": 0.44})

    answer = find_context(state, "money", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "- title: alpha" in answer
    assert "bravo" not in answer
    assert set(state.reuse) == {"alpha"}


def test_each_candidate_carries_its_title_tags_and_description() -> None:
    state = graph(card("alpha", description="about invoices", tags=("invoice", "1200")))
    matcher = TableMatcher({"about invoices": 0.9})

    answer = find_context(state, "money", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "- title: alpha" in answer
    assert "tags: invoice, 1200" in answer
    assert "about invoices" in answer


def test_a_tag_filter_is_exact_on_the_normalized_value() -> None:
    state = searchable()
    matcher = TableMatcher({}, default=0.9)

    answer = find_context(
        state, "money", "R$ 1.200,00", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL
    )

    titles = [line.removeprefix("- title: ") for line in answer.splitlines() if line.startswith("- title: ")]
    assert titles == ["delta"]


def test_an_unknown_tag_answers_and_records_nothing() -> None:
    state = searchable()
    before = state.choice
    matcher = TableMatcher({}, default=0.9)

    answer = find_context(
        state, "money", "nowhere", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL
    )

    assert "nothing in this conversation matches" in answer
    assert "tagged 'nowhere'" in answer
    assert state.choice is before
    assert state.reuse == {}


def test_an_empty_need_answers_without_reaching_the_matcher() -> None:
    state = searchable()
    matcher = TableMatcher({}, default=0.9)

    answer = find_context(state, "   ", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "nothing in this conversation matches" in answer
    assert matcher.calls == 0
    assert state.reuse == {}


def test_an_unusable_matcher_answers_rather_than_raising() -> None:
    state = searchable()
    before = state.choice

    answer = find_context(
        state, "money", matcher=BrokenMatcher(), collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL
    )

    assert "nothing in this conversation matches" in answer
    assert state.choice is before
    assert state.reuse == {}


def test_nothing_clearing_the_floor_answers_and_records_nothing() -> None:
    state = searchable()
    matcher = TableMatcher({}, default=0.1)

    answer = find_context(state, "money", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "nothing in this conversation matches" in answer
    assert state.reuse == {}


def test_an_empty_graph_answers_rather_than_raising() -> None:
    state = _GraphState()
    matcher = TableMatcher({}, default=0.9)

    answer = find_context(state, "money", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert "nothing in this conversation matches" in answer
    assert matcher.calls == 0


def test_find_context_changes_no_resolution_on_success() -> None:
    state = graph(card("alpha", description="about invoices"))
    before = state.choice
    matcher = TableMatcher({"about invoices": 0.9})

    find_context(state, "money", matcher=matcher, collapse_floor=0.45, cycle=CYCLE, reuse_ttl_cycles=TTL)

    assert state.choice is before


# ---- the plugin registers exactly these three tools (Requirement 12.1) ----


def test_the_plugin_registers_exactly_the_three_retrieval_tools() -> None:
    from strands_context_graph import ContextGraph

    names = {tool.tool_name for tool in ContextGraph().tools}

    assert names == {"expand_card", "expand_artifact", "find_context"}
