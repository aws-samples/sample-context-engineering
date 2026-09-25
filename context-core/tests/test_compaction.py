"""Unit tests of ``render_final_block``: the block derived from the list that actually left.

Every test here builds the pair the function reads — the request, and the removed list — and asserts on the text that
comes out. What the pair encodes is the one thing the module exists for: a part that survived contributes nothing, so
the request alone never decides anything.

The removed list is written by hand rather than produced by the removal. The guards have their own suite, and the
interesting inputs here are the ones a guard produces *rarely* — a single pinned message inside a collapsed Card, half a
tool pair held back by protection — which are cheaper to state directly than to coax a generator into drawing.

``render_final_block`` reads ``messages`` and nothing else, so the fixtures pass the message list directly.
"""

import copy
from types import MappingProxyType

import pytest

from context_core.graph.compaction import _FRAGMENT_INDENT, render_final_block
from context_core.graph.state import Card, CardChoice, ToolPair, TurnChoice, _GraphState

DESCRIPTION_TOKENS = 100
"""The plugin's own default, so the budgeting these tests see is the one production sees.

Wide enough that the small fixtures below never brush the ceiling: a test that silently started hitting the budget would
be asserting on the budget rather than on what it was written to assert.
"""


def frozen_choice(by_title, *, full_pass=False, selected=None):
    """Build a ``TurnChoice`` carrying a mapping no caller can write through, the way production must."""
    return TurnChoice(by_title=MappingProxyType(dict(by_title)), full_pass=full_pass, selected=selected)


def card(
    title,
    turn,
    dialogue_ids,
    evidence_ids,
    *,
    description="",
    numeric_lines=(),
    pairs=(),
    references=(),
    tool_names=frozenset(),
    kind="subject",
):
    """Build a Card carrying only the fields the final block reads."""
    return Card(
        title=title,
        kind=kind,
        turn=turn,
        dialogue_ids=tuple(dialogue_ids),
        evidence_ids=tuple(evidence_ids),
        pairs=tuple(pairs),
        tool_names=tool_names,
        references=tuple(references),
        numeric_lines=tuple(numeric_lines),
        tags=(),
        description=description,
        reference="ref-1" if kind == "artifact" else None,
    )


def state(*cards, choice=None):
    """A graph state holding ``cards`` and, optionally, the frozen choice over them."""
    graph = _GraphState()
    for entry in cards:
        graph.cards[entry.title] = entry
    graph.turn = len(graph.cards)
    if choice is not None:
        graph.choice = choice
    return graph


def context(*tracking_ids):
    """A removed list holding exactly ``tracking_ids``, in order."""
    return [
        {"role": "user", "content": [{"text": f"retained {tracking_id}"}], "tracking_id": tracking_id}
        for tracking_id in tracking_ids
    ]


def collapsed(graph, requested, *retained_ids, description_tokens=DESCRIPTION_TOKENS):
    """Render the block for ``requested`` against a removed list holding ``retained_ids``."""
    return render_final_block(
        context(*retained_ids), graph, frozenset(requested), description_tokens=description_tokens, retrieval_tools=("expand_card", "expand_artifact", "find_context")
    )


# --- a part contributes only when all of it left --------------------------------------------------


def test_a_part_that_fully_left_contributes_its_fragment():
    """Requirement 9.7: the dialogue in description folds the Card's description at the end."""
    graph = state(
        card("t0", 0, ("d0", "d1"), (), description="t0\nbalance: 1.200,00"),
        choice=frozen_choice({"t0": CardChoice("description", "full")}),
    )

    block = collapsed(graph, {"d0", "d1"})

    assert "- t0" in block
    assert "balance: 1.200,00" in block


def test_a_part_preserved_by_a_guard_contributes_nothing():
    """Requirement 9.7: one surviving message makes the whole part full content, by derivation.

    The choice still says description, and the request still holds both identities. What changed is the result — ``d1``
    came back — and the result is what the render reads.
    """
    graph = state(
        card("t0", 0, ("d0", "d1"), (), description="t0\nbalance: 1.200,00"),
        choice=frozen_choice({"t0": CardChoice("description", "full")}),
    )

    assert collapsed(graph, {"d0", "d1"}, "d1") is None


def test_the_surviving_part_is_silent_while_the_other_still_speaks():
    """The two axes are read independently: a pinned dialogue message does not mute the evidence."""
    graph = state(
        card(
            "t0",
            0,
            ("d0",),
            ("e0", "e1"),
            description="t0\nignored",
            numeric_lines=("R$ 47.832,15",),
            pairs=(ToolPair("tu1", "run_query", ("e0", "e1"), consumed=True),),
        ),
        choice=frozen_choice({"t0": CardChoice("description", "description")}),
    )

    block = collapsed(graph, {"d0", "e0", "e1"}, "d0")

    assert "R$ 47.832,15" in block
    assert "ignored" not in block


def test_content_never_appears_at_two_resolutions_in_the_same_call():
    """Requirement 9.7 over the shape a pin creates on one half of a tool pair.

    ``e1`` is retained, so the evidence is whole in the messages and must not be collapsed as well.
    """
    graph = state(
        card(
            "t0",
            0,
            ("d0",),
            ("e0", "e1"),
            description="t0",
            numeric_lines=("R$ 47.832,15",),
            pairs=(ToolPair("tu1", "run_query", ("e0", "e1"), consumed=True),),
        ),
        choice=frozen_choice({"t0": CardChoice("full", "description")}),
    )

    assert collapsed(graph, {"e0", "e1"}, "e1") is None


def test_an_empty_part_contributes_nothing():
    """A turn with no tool call has no evidence to collapse, so it claims none.

    ``t1``'s evidence is empty and its resolution is description, which under a bare subset test would read as "wholly
    absent" and put the Card in the block with numeric lines nothing removed.
    """
    graph = state(
        card("t0", 0, ("d0",), (), description="t0"),
        card("t1", 1, ("d1",), (), description="t1", numeric_lines=("1200",)),
        choice=frozen_choice(
            {
                "t0": CardChoice("description", "full"),
                "t1": CardChoice("full", "description"),
            }
        ),
    )

    block = collapsed(graph, {"d0"})

    assert "- t1" not in block
    assert "1200" not in block


# --- what each axis contributes -------------------------------------------------------------------


def test_a_dialogue_in_title_contributes_the_title_line_and_nothing_else():
    """Requirement 4.2: the entry is the address, and the address alone is the whole entry."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0\nbalance: 1.200,00"),
        choice=frozen_choice({"t0": CardChoice("title", "full")}),
    )

    block = collapsed(graph, {"d0"})

    assert "- t0" in block
    assert "balance: 1.200,00" not in block


def test_the_evidence_contributes_tools_references_and_the_numeric_lines():
    """Requirement 4.2's evidence half, folded literally at the end of the call."""
    graph = state(
        card(
            "t0",
            0,
            (),
            ("e0", "e1"),
            references=("ref-7", "ref-7", "ref-9"),
            numeric_lines=("| ativo | 12,50 |", "total: 3.451,90 BRL"),
            pairs=(
                ToolPair("tu1", "run_query", ("e0",), consumed=True),
                ToolPair("tu2", "run_query", ("e1",), consumed=True),
            ),
        ),
        choice=frozen_choice({"t0": CardChoice("full", "description")}),
    )

    block = collapsed(graph, {"e0", "e1"})

    assert "tools: run_query (2)" in block
    assert "references: ref-7, ref-9" in block
    assert "| ativo | 12,50 |" in block
    assert "total: 3.451,90 BRL" in block


def test_the_evidence_numeric_lines_are_bounded_by_the_description_budget():
    """A Card's entry costs the Description's ceiling, whatever the size of the table behind it.

    ``Card.numeric_lines`` holds every line of the turn that carried a number, so a turn whose tool returned a table
    would otherwise put its whole preview back into every call for the rest of the session.
    """
    lines = tuple(f"row {index} | {index}.000,00 | {index * 7} ms" for index in range(200))
    graph = state(
        card("t0", 0, (), ("e0",), numeric_lines=lines),
        choice=frozen_choice({"t0": CardChoice("full", "description")}),
    )

    block = collapsed(graph, {"e0"}, description_tokens=100)

    fragments = [line for line in block.splitlines() if line.startswith(_FRAGMENT_INDENT)]
    # The budget covers the fragments, which is what grows with the table. The indent, the entry's title line and the
    # fixed markers are per Card and per call, and neither one scales with it.
    assert sum(len(fragment) - len(_FRAGMENT_INDENT) + 1 for fragment in fragments) <= 100 * 4
    assert lines[0] in block
    assert lines[-1] not in block
    # And the point of the ceiling: the whole table would have been an order of magnitude larger.
    assert len(block) < len("\n".join(lines)) // 5


def test_the_omitted_numeric_lines_are_counted_in_the_block():
    """Requirement 4.6 applied to the block: a gap the model can see is a gap it can close.

    Silently keeping the first rows of a table reads as the whole table, and a question asking for the largest value is
    then answered from a subset — wrong, and wrong without a symptom.
    """
    lines = tuple(f"row {index} | {index}.000,00" for index in range(200))
    graph = state(
        card("t0", 0, (), ("e0",), numeric_lines=lines),
        choice=frozen_choice({"t0": CardChoice("full", "description")}),
    )

    block = collapsed(graph, {"e0"}, description_tokens=100)

    assert "numeric lines omitted)" in block


def test_a_budget_too_small_for_a_single_line_keeps_the_addresses():
    """The tools and references lines are what make the gap closable, so they never lose the budget."""
    graph = state(
        card(
            "t0",
            0,
            (),
            ("e0",),
            references=("ref-7",),
            numeric_lines=("total: 3.451,90 BRL",),
            pairs=(ToolPair("tu1", "run_query", ("e0",), consumed=True),),
        ),
        choice=frozen_choice({"t0": CardChoice("full", "description")}),
    )

    block = collapsed(graph, {"e0"}, description_tokens=1)

    assert "tools: run_query (1)" in block
    assert "references: ref-7" in block
    assert "total: 3.451,90 BRL" not in block
    assert "(+1 numeric lines omitted)" in block


def test_the_numeric_lines_are_copied_literally():
    """Requirement 4.4 downstream: the last place the numbers pass through does not rewrite them."""
    line = "BTG Pactual\u00a0..\u00a0R$ 47.832,15"
    graph = state(
        card("t0", 0, (), ("e0",), numeric_lines=(line,)),
        choice=frozen_choice({"t0": CardChoice("full", "description")}),
    )

    assert line in collapsed(graph, {"e0"})


def test_a_line_contributes_once_per_card():
    """Both parts left, and the description already carries what the evidence would repeat."""
    graph = state(
        card(
            "t0",
            0,
            ("d0",),
            ("e0",),
            description="t0\ntools: run_query (1)\n1200",
            numeric_lines=("1200", "3400"),
            pairs=(ToolPair("tu1", "run_query", ("e0",), consumed=True),),
        ),
        choice=frozen_choice({"t0": CardChoice("description", "description")}),
    )

    block = collapsed(graph, {"d0", "e0"})

    assert block.count("tools: run_query (1)") == 1
    assert block.count("1200") == 1
    assert "3400" in block  # what the description had to leave out still gets in


def test_the_title_is_never_repeated_inside_its_own_entry():
    """The description opens on the Card's title, and the entry's first line already is it."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0\nbalance: 1200"),
        choice=frozen_choice({"t0": CardChoice("description", "full")}),
    )

    assert collapsed(graph, {"d0"}).count("t0") == 1


# --- no fragment at all ---------------------------------------------------------------------------


def test_nothing_dropped_returns_none():
    """With no fragment the primitive leaves ``dynamic_trailing_blocks`` alone (Requirement 9.3)."""
    graph = state(
        card("t0", 0, ("d0",), ("e0",), description="t0"),
        choice=frozen_choice({"t0": CardChoice("description", "description")}),
    )

    assert collapsed(graph, set()) is None


def test_an_empty_graph_returns_none():
    """The shape a fresh agent has: no Card, no block."""
    assert (
        render_final_block(
            context(), _GraphState(), frozenset(), description_tokens=DESCRIPTION_TOKENS, retrieval_tools=("expand_card", "expand_artifact", "find_context")
        )
        is None
    )


def test_a_request_wholly_preserved_by_the_guards_returns_none():
    """Every identity came back, so the removal removed nothing and there is nothing to fold."""
    graph = state(
        card("t0", 0, ("d0",), ("e0",), description="t0", numeric_lines=("1200",)),
        choice=frozen_choice({"t0": CardChoice("description", "description")}),
    )

    assert collapsed(graph, {"d0", "e0"}, "d0", "e0") is None


def test_a_card_absent_from_the_choice_contributes_no_fragment():
    """A Card derived after the choice was frozen is read as full content, so it stays silent.

    Its identities cannot be in the request either — this pins the direction of the fail-safe, not a reachable state.
    """
    graph = state(
        card("t0", 0, ("d0",), (), description="t0\nbalance: 1200"),
        choice=frozen_choice({}),
    )

    block = collapsed(graph, {"d0"})

    assert "- t0" in block
    assert "balance: 1200" not in block


# --- ordering and shape ---------------------------------------------------------------------------


def test_the_cards_come_out_in_ascending_turn_order():
    """Requirement 9.9: the block changes only where the resolution changed, so the order is the turn's."""
    graph = state(
        card("late", 7, ("d7",), (), description="late"),
        card("early", 2, ("d2",), (), description="early"),
        card("middle", 4, ("d4",), (), description="middle"),
        choice=frozen_choice({title: CardChoice("description", "full") for title in ("late", "early", "middle")}),
    )

    block = collapsed(graph, {"d2", "d4", "d7"})

    assert block.index("- early") < block.index("- middle") < block.index("- late")


def test_every_card_that_lost_a_part_has_its_title_in_the_block():
    """Requirement 4.2: the Title of every Card is in the call — here, in the block."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0"),
        card("t1", 1, ("d1",), (), description="t1"),
        card("t2", 2, ("d2",), (), description="t2"),
        choice=frozen_choice(
            {
                "t0": CardChoice("title", "full"),
                "t1": CardChoice("description", "full"),
                "t2": CardChoice("full", "full"),
            }
        ),
    )

    block = collapsed(graph, {"d0", "d1"})

    assert "- t0" in block
    assert "- t1" in block
    assert "- t2" not in block  # it lost nothing, so its content is in the retained messages


def test_the_block_names_the_three_retrieval_tools():
    """The gap has to read as closable, or the model answers from the summary instead of asking."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0"),
        choice=frozen_choice({"t0": CardChoice("description", "full")}),
    )

    block = collapsed(graph, {"d0"})

    assert "expand_card" in block
    assert "expand_artifact" in block
    assert "find_context" in block


def test_the_return_is_plain_text():
    """The block has no ``role``, is not a message, and takes no position in the list."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0"),
        choice=frozen_choice({"t0": CardChoice("description", "full")}),
    )

    assert isinstance(collapsed(graph, {"d0"}), str)


@pytest.mark.parametrize("kind", ["subject", "artifact"])
def test_an_artifact_card_renders_like_a_subject(kind):
    """The render has no special case for an artifact: the Description already decided the shape."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0\nreference: ref-1", kind=kind),
        choice=frozen_choice({"t0": CardChoice("description", "full")}),
    )

    assert "reference: ref-1" in collapsed(graph, {"d0"})


# --- a bounded selection --------------------------------------------------------------------------


def test_an_unaddressed_card_contributes_nothing_but_is_counted():
    """A Card the selection left out loses even its Title, and the trailer states the gap instead."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0\nbalance: 1200"),
        card("t1", 1, ("d1",), (), description="t1\nbalance: 3400"),
        choice=frozen_choice(
            {
                "t0": CardChoice("description", "full"),
                "t1": CardChoice("description", "full"),
            },
            selected=frozenset({"t0"}),
        ),
    )

    block = collapsed(graph, {"d0", "d1"})

    assert "- t0" in block
    assert "- t1" not in block
    assert "3400" not in block
    assert "1 earlier turn(s)" in block


def test_a_selection_addressing_every_card_states_no_gap():
    """Nothing was left out, so the trailer is the guidance alone: a count of zero reads as a gap."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0"),
        choice=frozen_choice({"t0": CardChoice("description", "full")}, selected=frozenset({"t0"})),
    )

    block = collapsed(graph, {"d0"})

    assert "earlier turn(s)" not in block
    assert "expand_card" in block


def test_a_gap_alone_is_still_reported():
    """No Card contributed a fragment, but turns were left out, so the model still learns they exist."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0"),
        card("t1", 1, ("d1",), (), description="t1"),
        choice=frozen_choice({"t0": CardChoice("description", "full")}, selected=frozenset({"t0"})),
    )

    block = collapsed(graph, set())

    assert "<collapsed_turns>" not in block
    assert "1 earlier turn(s)" in block


# --- nothing is mutated, and two runs agree -------------------------------------------------------


def test_nothing_is_mutated():
    """Neither the removed list, nor the dicts inside it, nor the graph state comes out changed."""
    graph = state(
        card(
            "t0",
            0,
            ("d0",),
            ("e0",),
            description="t0\n1200",
            numeric_lines=("1200", "3400"),
            pairs=(ToolPair("tu1", "run_query", ("e0",), consumed=True),),
        ),
        choice=frozen_choice({"t0": CardChoice("description", "description")}),
    )
    injection_context = context("kept")
    messages_before = copy.deepcopy(injection_context)
    cards_before = copy.deepcopy(graph.cards)
    requested = frozenset({"d0", "e0"})

    render_final_block(injection_context, graph, requested, description_tokens=DESCRIPTION_TOKENS, retrieval_tools=("expand_card", "expand_artifact", "find_context"))

    assert injection_context == messages_before
    assert graph.cards == cards_before
    assert dict(graph.choice.by_title) == {"t0": CardChoice("description", "description")}
    assert requested == frozenset({"d0", "e0"})


def test_two_runs_over_the_same_inputs_agree_character_for_character():
    """Requirement 9.9: same list, same choice, same block."""
    graph = state(
        card("t0", 0, ("d0",), ("e0",), description="t0\n1200", numeric_lines=("1200", "3400")),
        card("t1", 1, ("d1",), (), description="t1"),
        choice=frozen_choice(
            {
                "t0": CardChoice("description", "description"),
                "t1": CardChoice("title", "full"),
            }
        ),
    )

    assert collapsed(graph, {"d0", "e0", "d1"}) == collapsed(graph, {"d0", "e0", "d1"})


def test_a_retained_message_without_a_durable_identity_is_ignored():
    """An unaddressed message cannot preserve a part, because it names none."""
    graph = state(
        card("t0", 0, ("d0",), (), description="t0\nbalance: 1200"),
        choice=frozen_choice({"t0": CardChoice("description", "full")}),
    )
    injection_context = [{"role": "user", "content": [{"text": "no address at all"}]}]

    assert "balance: 1200" in render_final_block(
        injection_context, graph, frozenset({"d0"}), description_tokens=DESCRIPTION_TOKENS, retrieval_tools=("expand_card", "expand_artifact", "find_context")
    )
