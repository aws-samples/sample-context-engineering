"""Unit tests for turn -> Card derivation by scan.

The assertions here are about the scan and nothing else: where a turn starts and ends, which identities land in which
part, how tool pairs are keyed and consumed, which references a preview names, and which links a registration derives.
Title, Description and Tag wording belongs to ``describe`` and is asserted in its own tests.
"""

from typing import Any

import pytest

from context_core.graph.cards import (
    closed_turn_ranges,
    derive_and_register,
    derive_and_register_artifacts,
    derive_artifact_cards,
    derive_card,
    is_evidence,
    is_turn_boundary,
    partition_turn,
    rebuild,
    register_card,
    tool_pairs_of,
    turn_ranges,
)
from context_core.graph.state import _GraphState

DERIVATION = {"description_tokens": 100, "tags_per_card": 5, "rarity_weight": 0.5}
REGISTRATION = {**DERIVATION, "link_threshold": 0.5}
LINKING = {"tags_per_card": 5, "rarity_weight": 0.5, "link_threshold": 0.5}


def user(text: str, tracking_id: str | None = None) -> dict[str, Any]:
    """Build a plain user ask, the shape that opens a turn."""
    message: dict[str, Any] = {"role": "user", "content": [{"text": text}]}
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def assistant(text: str, tracking_id: str | None = None) -> dict[str, Any]:
    """Build an assistant text message."""
    message: dict[str, Any] = {"role": "assistant", "content": [{"text": text}]}
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def tool_use(tool_use_id: str, name: str, tracking_id: str) -> dict[str, Any]:
    """Build an assistant message carrying one ``toolUse`` block."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": {}}}],
        "tracking_id": tracking_id,
    }


def tool_result(tool_use_id: str, text: str, tracking_id: str) -> dict[str, Any]:
    """Build a user message carrying one ``toolResult`` block, the autonomous half of a turn."""
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_use_id, "status": "success", "content": [{"text": text}]}}],
        "tracking_id": tracking_id,
    }


def conversation() -> list[dict[str, Any]]:
    """Two closed turns plus a turn in progress, the shape every range assertion reads."""
    return [
        user("what is the balance of account 42?", "m1"),
        tool_use("tu-1", "get_balance", "m2"),
        tool_result("tu-1", "balance: R$ 1.200,00", "m3"),
        assistant("the balance is R$ 1.200,00", "m4"),
        user("and the second account?", "m5"),
        assistant("R$ 340,00", "m6"),
        user("compare the two", "m7"),
    ]


def test_a_plain_user_ask_opens_a_turn() -> None:
    assert is_turn_boundary(user("hello")) is True


def test_a_tool_result_does_not_open_a_turn() -> None:
    assert is_turn_boundary(tool_result("tu-1", "done", "m1")) is False


def test_an_assistant_message_does_not_open_a_turn() -> None:
    assert is_turn_boundary(assistant("hello")) is False


def test_turn_ranges_span_boundary_to_next_boundary() -> None:
    assert turn_ranges(conversation()) == ((0, 4), (4, 6), (6, 7))


def test_messages_before_the_first_boundary_belong_to_no_turn() -> None:
    messages = [assistant("a preamble"), user("the question", "m2"), assistant("the answer", "m3")]

    assert turn_ranges(messages) == ((1, 3),)


def test_closed_turn_ranges_exclude_the_turn_in_progress() -> None:
    assert closed_turn_ranges(conversation()) == ((0, 4), (4, 6))


def test_a_single_turn_conversation_has_no_closed_turn() -> None:
    assert closed_turn_ranges([user("only one", "m1")]) == ()


def test_no_boundary_means_no_range() -> None:
    assert turn_ranges([assistant("orphan")]) == ()
    assert closed_turn_ranges([]) == ()


def test_evidence_is_decided_by_content_block() -> None:
    assert is_evidence(tool_use("tu-1", "t", "m1")) is True
    assert is_evidence(tool_result("tu-1", "x", "m2")) is True
    assert is_evidence(user("a question", "m3")) is False
    assert is_evidence(assistant("an answer", "m4")) is False


def test_partition_is_exhaustive_and_disjoint_over_addressed_messages() -> None:
    turn = conversation()[:4]

    dialogue, evidence = partition_turn(turn)

    assert dialogue == ("m1", "m4")
    assert evidence == ("m2", "m3")
    assert set(dialogue) & set(evidence) == set()
    assert set(dialogue) | set(evidence) == {"m1", "m2", "m3", "m4"}


def test_partition_skips_messages_without_a_durable_identity() -> None:
    dialogue, evidence = partition_turn([user("addressed", "m1"), user("unaddressed")])

    assert dialogue == ("m1",)
    assert evidence == ()


def test_tool_pairs_are_keyed_by_tool_use_id() -> None:
    turn = [
        user("do both", "m1"),
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "tu-1", "name": "first", "input": {}}},
                {"toolUse": {"toolUseId": "tu-2", "name": "second", "input": {}}},
            ],
            "tracking_id": "m2",
        },
        tool_result("tu-1", "a", "m3"),
        tool_result("tu-2", "b", "m4"),
    ]

    pairs = tool_pairs_of(turn)

    assert [pair.tool_use_id for pair in pairs] == ["tu-1", "tu-2"]
    assert [pair.tool_name for pair in pairs] == ["first", "second"]
    assert pairs[0].tracking_ids == ("m2", "m3")


def test_a_pair_is_consumed_by_a_later_assistant_text_block() -> None:
    (pair,) = tool_pairs_of(conversation()[:4])

    assert pair.consumed is True


def test_a_turn_ending_in_a_tool_result_leaves_the_pair_unconsumed() -> None:
    (pair,) = tool_pairs_of(conversation()[:3])

    assert pair.consumed is False


def test_an_incomplete_pair_keeps_the_half_it_has_and_is_never_consumed() -> None:
    turn = [user("go", "m1"), tool_use("tu-1", "pending", "m2")]

    (pair,) = tool_pairs_of(turn)

    assert pair.tool_name == "pending"
    assert pair.tracking_ids == ("m2",)
    assert pair.consumed is False


def test_a_tool_result_without_its_tool_use_still_yields_a_pair() -> None:
    (pair,) = tool_pairs_of([tool_result("tu-9", "orphan", "m1")])

    assert (pair.tool_use_id, pair.tool_name, pair.consumed) == ("tu-9", "", False)


def test_derive_card_addresses_messages_and_keeps_no_content() -> None:
    messages = conversation()
    start, stop = closed_turn_ranges(messages)[0]

    card = derive_card(messages, ["m1", "m2", "m3", "m4"], 0, **DERIVATION)

    assert (start, stop) == (0, 4)
    assert card.kind == "subject"
    assert card.turn == 0
    assert card.dialogue_ids == ("m1", "m4")
    assert card.evidence_ids == ("m2", "m3")
    assert card.tool_names == frozenset({"get_balance"})
    assert not hasattr(card, "content")
    # The Title is a literal prefix of the user ask, so the Card carries an address, not a paraphrase.
    assert messages[0]["content"][0]["text"].startswith(card.title)


def test_derive_card_ignores_messages_outside_the_turn() -> None:
    card = derive_card(conversation(), ["m5", "m6"], 1, **DERIVATION)

    assert card.dialogue_ids == ("m5", "m6")
    assert card.tool_names == frozenset()


def test_derive_card_is_deterministic() -> None:
    messages = conversation()

    first = derive_card(messages, ["m1", "m2", "m3", "m4"], 0, **DERIVATION)
    second = derive_card(messages, ["m1", "m2", "m3", "m4"], 0, **DERIVATION)

    assert first == second
    assert first.description == second.description


def test_a_turn_whose_messages_carry_no_identity_yields_an_empty_card() -> None:
    card = derive_card([user("unaddressed")], [], 0, **DERIVATION)

    assert card.dialogue_ids == ()
    assert card.evidence_ids == ()
    assert card.title == ""


@pytest.mark.parametrize(
    ("preview", "expected"),
    [
        ("[image: png, 900 bytes | ref: mem_1_tu-3_0]", ("mem_1_tu-3_0",)),
        ("stashed [ref: tu-3_0]", ("tu-3_0",)),
        ("stashed [refs: tu-3_0, tu-3_1]", ("tu-3_0", "tu-3_1")),
        ("[Stored references:]\n  mem_1_tu-3_0 (text, 4,096 chars)", ("mem_1_tu-3_0",)),
        ("nothing stored here", ()),
    ],
)
def test_references_are_discovered_in_both_placeholder_shapes(preview: str, expected: tuple[str, ...]) -> None:
    messages = [
        user("fetch it", "m1"),
        tool_use("tu-3", "http_request", "m2"),
        tool_result("tu-3", preview, "m3"),
        assistant("done", "m4"),
        user("next", "m5"),
    ]

    card = derive_card(messages, ["m1", "m2", "m3", "m4"], 0, **DERIVATION)

    assert card.references == expected


def test_a_reference_mentioned_in_prose_stops_at_the_field_boundary() -> None:
    card = derive_card(
        [user("see ref: mem_1_tu-3_0 for the details please", "m1"), user("next", "m2")],
        ["m1"],
        0,
        **DERIVATION,
    )

    assert card.references == ("mem_1_tu-3_0",)


def test_register_card_derives_tool_and_artifact_links() -> None:
    state = _GraphState()
    messages = [
        user("fetch it", "m1"),
        tool_use("tu-3", "http_request", "m2"),
        tool_result("tu-3", "[image: png, 900 bytes | ref: mem_1_tu-3_0]", "m3"),
        assistant("done", "m4"),
        user("next", "m5"),
    ]
    card = derive_card(messages, ["m1", "m2", "m3", "m4"], 0, **DERIVATION)

    register_card(state, card, messages, **LINKING)

    edges = {(link.kind, link.target) for link in state.links[card.title]}
    assert ("tool", "http_request") in edges
    assert ("artifact", "mem_1_tu-3_0") in edges
    assert all(link.weight == 1.0 for link in state.links[card.title])


def test_register_card_links_to_the_previous_turns_card() -> None:
    state = _GraphState()
    messages = conversation()

    first = derive_card(messages, ["m1", "m2", "m3", "m4"], 0, **DERIVATION)
    register_card(state, first, messages, **LINKING)
    second = derive_card(messages, ["m5", "m6"], 1, **DERIVATION)
    register_card(state, second, messages, **LINKING)

    assert [(link.kind, link.target) for link in state.links[second.title]] == [("follows", first.title)]
    assert [(link.kind, link.target) for link in state.links[first.title]] == [("tool", "get_balance")]


def test_no_similar_link_without_cached_vectors() -> None:
    state = _GraphState()
    messages = conversation()

    for turn, ids in enumerate((["m1", "m2", "m3", "m4"], ["m5", "m6"])):
        register_card(state, derive_card(messages, ids, turn, **DERIVATION), messages, **LINKING)

    assert all(link.kind != "similar" for edges in state.links.values() for link in edges)


def test_registering_the_same_card_twice_does_not_duplicate_an_edge() -> None:
    state = _GraphState()
    messages = conversation()
    card = derive_card(messages, ["m1", "m2", "m3", "m4"], 0, **DERIVATION)

    register_card(state, card, messages, **LINKING)
    register_card(state, card, messages, **LINKING)

    assert len(state.links[card.title]) == 1


def test_derive_and_register_degrades_to_no_card_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _GraphState()
    messages = conversation()
    register_card(state, derive_card(messages, ["m1", "m2", "m3", "m4"], 0, **DERIVATION), messages, **LINKING)
    cards_before = dict(state.cards)
    links_before = {title: list(edges) for title, edges in state.links.items()}

    monkeypatch.setattr(
        "context_core.graph.cards.compose_description",
        lambda card, tokens: (_ for _ in ()).throw(RuntimeError("derivation exploded")),
    )
    registered = derive_and_register(state, messages, ["m5", "m6"], 1, **REGISTRATION)

    assert registered is None
    assert state.cards == cards_before
    assert state.links == links_before


def test_artifact_cards_keep_only_the_reference() -> None:
    result = {
        "toolUseId": "tu-3",
        "status": "success",
        "content": [{"text": "[image: png, 900 bytes | ref: mem_1_tu-3_0]"}],
    }

    (card,) = derive_artifact_cards(result, "http_request", 2, **DERIVATION)

    assert card.kind == "artifact"
    assert card.title == card.reference == "mem_1_tu-3_0"
    assert card.content_type == "image/png"
    assert card.size_bytes == 900
    assert card.dialogue_ids == ()
    assert card.evidence_ids == ()
    assert card.tool_names == frozenset({"http_request"})
    # Non-textual content carries no lines: bytes that were never text have none to copy.
    assert card.numeric_lines == ()


def test_a_textual_artifact_keeps_the_numeric_lines_of_its_preview() -> None:
    result = {
        "toolUseId": "tu-3",
        "status": "success",
        "content": [{"text": "balance: 1200\n[Stored references:]\n  mem_1_tu-3_0 (text, 4,096 chars)"}],
    }

    (card,) = derive_artifact_cards(result, "get_balance", 1, **DERIVATION)

    assert card.content_type == "text/plain"
    assert card.size_bytes is None
    assert "balance: 1200" in card.numeric_lines


def test_a_result_naming_no_reference_yields_no_artifact_card() -> None:
    result = {"toolUseId": "tu-3", "status": "success", "content": [{"text": "plain output, nothing offloaded"}]}

    assert derive_artifact_cards(result, "http_request", 0, **DERIVATION) == ()


def test_artifact_registration_adds_the_tool_link_only() -> None:
    state = _GraphState()
    messages = conversation()
    result = {
        "toolUseId": "tu-3",
        "status": "success",
        "content": [{"text": "[document: pdf, 2,048 bytes | ref: mem_1_tu-3_0]"}],
    }

    (card,) = derive_and_register_artifacts(state, messages, result, "http_request", 2, **DERIVATION)

    assert state.cards[card.title].reference == card.reference
    assert [(link.kind, link.target) for link in state.links[card.title]] == [("tool", "http_request")]


def test_rebuild_derives_one_card_per_closed_turn() -> None:
    messages = conversation()

    state = rebuild(messages, **REGISTRATION)

    # Two closed turns; the third boundary is the turn in progress, which no Card covers.
    assert state.turn == 2
    assert [card.turn for card in state.cards.values()] == [0, 1]
    assert all(card.kind == "subject" for card in state.cards.values())


def test_rebuild_equals_the_incremental_construction() -> None:
    messages = conversation()
    incremental = _GraphState()
    for turn, ids in enumerate((["m1", "m2", "m3", "m4"], ["m5", "m6"])):
        register_card(incremental, derive_card(messages, ids, turn, **DERIVATION), messages, **LINKING)

    scanned = rebuild(messages, **REGISTRATION)

    assert scanned.cards == incremental.cards
    assert scanned.links == incremental.links


def test_rebuild_is_idempotent_and_survives_an_empty_conversation() -> None:
    messages = conversation()

    once = rebuild(messages, **REGISTRATION)
    twice = rebuild(messages, **REGISTRATION)

    assert once.cards == twice.cards
    assert once.links == twice.links
    assert rebuild([], **REGISTRATION).cards == {}


def test_a_turn_without_identities_consumes_its_ordinal_without_shifting_later_ones() -> None:
    messages = [
        user("unaddressed turn"),
        assistant("no identity either"),
        user("addressed turn", "m3"),
        assistant("answered", "m4"),
        user("in progress", "m5"),
    ]

    state = rebuild(messages, **REGISTRATION)

    assert [card.turn for card in state.cards.values()] == [1]
    assert state.turn == 2


def test_the_live_history_is_never_mutated() -> None:
    messages = conversation()
    before = [dict(message) for message in messages]

    rebuild(messages, **REGISTRATION)

    assert messages == before
