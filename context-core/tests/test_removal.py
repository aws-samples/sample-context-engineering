"""Unit tests for the Removal: the request, and the subsequence that applies it.

The assertions are about form only — subsequence identity, tool-pair integrity, the leading user turn, the pin, and the
finiteness of the reconciliation. The scoring decides which parts are requested; here the choice is handed in directly.
"""

from types import MappingProxyType
from typing import Any

from context_core.graph.removal import apply_removal, removal_ids, remove_messages
from context_core.graph.state import Card, CardChoice, TurnChoice, _GraphState


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


def tool_use(tool_use_id: str, name: str = "search", tracking_id: str | None = None) -> dict[str, Any]:
    """Build the assistant half of a tool pair."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": {}}}],
    }
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def tool_result(tool_use_id: str, text: str = "42", tracking_id: str | None = None) -> dict[str, Any]:
    """Build the user half of a tool pair."""
    message: dict[str, Any] = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_use_id, "status": "success", "content": [{"text": text}]}}],
    }
    if tracking_id:
        message["tracking_id"] = tracking_id
    return message


def pin(message: dict[str, Any]) -> dict[str, Any]:
    """Mark a message pinned the way the main agent does, through ``metadata.custom``."""
    message["metadata"] = {"custom": {"pinned": True}}
    return message


def card(title: str, dialogue_ids: tuple[str, ...], evidence_ids: tuple[str, ...] = ()) -> Card:
    """Build a subject Card holding only the identities this test needs."""
    return Card(
        title=title,
        kind="subject",
        turn=1,
        dialogue_ids=dialogue_ids,
        evidence_ids=evidence_ids,
        pairs=(),
        tool_names=frozenset(),
        references=(),
        numeric_lines=(),
        tags=(),
        description=f"{title} description",
    )


def choice_of(**by_title: CardChoice) -> TurnChoice:
    """Freeze a turn choice from ``title=CardChoice(...)`` pairs."""
    return TurnChoice(MappingProxyType(dict(by_title)), full_pass=False)


def tool_use_ids(messages: list[dict[str, Any]], block: str) -> set[str]:
    """Collect the ``toolUseId`` set carried by ``block`` blocks across ``messages``."""
    return {
        content[block]["toolUseId"]
        for message in messages
        for content in message.get("content", [])
        if block in content
    }


# --- removal_ids: the request, derived from the (Card, part) pair -------------------------------------------------


def test_full_content_everywhere_requests_nothing() -> None:
    """A choice keeping every part whole names no identity, so the delivery is the identity pass."""
    state = _GraphState(cards={"A": card("A", ("d1",), ("e1",))})
    choice = choice_of(A=CardChoice(dialogue="full", evidence="full"))

    assert removal_ids(state, choice, frozenset()) == frozenset()


def test_collapsed_parts_are_requested_independently() -> None:
    """Dialogue and evidence are requested on their own axes."""
    state = _GraphState(cards={"A": card("A", ("d1", "d2"), ("e1",))})

    assert removal_ids(state, choice_of(A=CardChoice("title", "full")), frozenset()) == frozenset({"d1", "d2"})
    assert removal_ids(state, choice_of(A=CardChoice("full", "description")), frozenset()) == frozenset({"e1"})


def test_absent_title_reads_as_full_content() -> None:
    """A Card the choice does not mention is left whole — the fail-safe direction for derivation lag."""
    state = _GraphState(cards={"A": card("A", ("d1",)), "B": card("B", ("d2",))})

    assert removal_ids(state, choice_of(A=CardChoice("title", "title")), frozenset()) == frozenset({"d1"})


def test_turn_in_progress_is_never_requested() -> None:
    """The open turn's identities are subtracted from the request, whatever the choice says."""
    state = _GraphState(cards={"A": card("A", ("d1", "d2"))})
    choice = choice_of(A=CardChoice("title", "title"))

    assert removal_ids(state, choice, frozenset({"d2"})) == frozenset({"d1"})


def test_request_mutates_neither_state_nor_choice() -> None:
    """Deriving the request reads the graph and the choice, and writes to neither."""
    state = _GraphState(cards={"A": card("A", ("d1",), ("e1",))})
    before = (dict(state.cards), state.turn, dict(state.reuse))
    choice = choice_of(A=CardChoice("description", "description"))

    removal_ids(state, choice, frozenset())

    assert (dict(state.cards), state.turn, dict(state.reuse)) == before
    assert set(choice.by_title) == {"A"}


# --- remove_messages: the subsequence and its guards -------------------------------------------------------------


def test_empty_request_returns_the_same_list_object() -> None:
    """Requirement 5.2: nothing requested, nothing copied — the received list comes back by identity."""
    messages = [user("hello", "m1"), assistant("hi", "m2")]

    assert remove_messages(messages, frozenset()) is messages


def test_removal_is_a_true_subsequence_of_the_same_objects() -> None:
    """Requirement 5.1: same objects, same relative order, no duplication, no insertion, no reordering."""
    messages = [user("q1", "m1"), assistant("a1", "m2"), user("q2", "m3"), assistant("a2", "m4")]

    removal = remove_messages(messages, frozenset({"m2"}))

    assert [id(message) for message in removal] == [id(messages[0]), id(messages[2]), id(messages[3])]


def test_removal_mutates_neither_the_list_nor_its_dicts() -> None:
    """Requirement 5.2: the received list and every dict in it come back untouched."""
    messages = [user("q1", "m1"), assistant("a1", "m2"), assistant("a2", "m3")]
    snapshot = [dict(message) for message in messages]

    remove_messages(messages, frozenset({"m2", "m3"}))

    assert messages == snapshot
    assert len(messages) == 3


def test_unnamed_and_unrequested_messages_always_stay() -> None:
    """No ``tracking_id``, or one absent from the request: the fail-safe default sends it."""
    messages = [user("q1", "m1"), assistant("no id"), assistant("a2", "m3")]

    removal = remove_messages(messages, frozenset({"m9"}))

    assert removal == messages
    assert removal is not messages


def test_dropping_a_tool_use_drops_its_result() -> None:
    """Requirement 5.4: the unprotected partner of a dropped half leaves with it."""
    messages = [user("q1", "m1"), tool_use("t1", tracking_id="m2"), tool_result("t1", tracking_id="m3")]

    removal = remove_messages(messages, frozenset({"m2"}))

    assert removal == [messages[0]]
    assert tool_use_ids(removal, "toolUse") == tool_use_ids(removal, "toolResult")


def test_dropping_a_tool_result_drops_its_use() -> None:
    """Requirement 5.4, the other direction: a dropped result takes its ``toolUse`` along."""
    messages = [user("q1", "m1"), tool_use("t1", tracking_id="m2"), tool_result("t1", tracking_id="m3")]

    removal = remove_messages(messages, frozenset({"m3"}))

    assert removal == [messages[0]]


def test_a_protected_partner_keeps_both_halves() -> None:
    """Requirement 5.5: a pinned half promotes its partner back, so the pair stays whole."""
    messages = [
        user("q1", "m1"),
        tool_use("t1", tracking_id="m2"),
        pin(tool_result("t1", tracking_id="m3")),
    ]

    removal = remove_messages(messages, frozenset({"m2", "m3"}))

    assert removal == messages
    assert tool_use_ids(removal, "toolUse") == tool_use_ids(removal, "toolResult") == {"t1"}


def test_tool_use_id_sets_stay_equal_across_interleaved_pairs() -> None:
    """Requirement 5.3: whatever the request, the retained ``toolUse`` and ``toolResult`` ids match."""
    messages = [
        user("q1", "m1"),
        tool_use("t1", tracking_id="m2"),
        tool_result("t1", tracking_id="m3"),
        tool_use("t2", tracking_id="m4"),
        tool_result("t2", tracking_id="m5"),
        assistant("a1", "m6"),
    ]

    for request in (frozenset({"m2"}), frozenset({"m5"}), frozenset({"m3", "m4"}), frozenset({"m2", "m3", "m4", "m5"})):
        removal = remove_messages(messages, request)
        assert tool_use_ids(removal, "toolUse") == tool_use_ids(removal, "toolResult")


def test_a_shared_tool_use_id_reconciles_without_flipping() -> None:
    """Requirement 5.8: one message paired with a protected end and a dropped end terminates, keeping the protected."""
    shared: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": "t1", "name": "search", "input": {}}},
            {"toolUse": {"toolUseId": "t2", "name": "read", "input": {}}},
        ],
        "tracking_id": "m2",
    }
    messages = [
        user("q1", "m1"),
        shared,
        pin(tool_result("t1", tracking_id="m3")),
        tool_result("t2", tracking_id="m4"),
    ]

    removal = remove_messages(messages, frozenset({"m2", "m3", "m4"}))

    assert removal == messages
    assert tool_use_ids(removal, "toolUse") == tool_use_ids(removal, "toolResult") == {"t1", "t2"}


def test_the_first_user_message_opens_the_removal() -> None:
    """Requirement 5.6: the leading user turn is kept, so the Removal opens with a ``user`` role."""
    messages = [user("q1", "m1"), assistant("a1", "m2"), user("q2", "m3")]

    removal = remove_messages(messages, frozenset({"m1", "m2"}))

    assert removal[0] is messages[0]
    assert removal[0]["role"] == "user"


def test_a_pinned_message_survives_any_request() -> None:
    """Requirement 5.7: an explicit pin wins over the request, whatever Card the message belongs to."""
    messages = [user("q1", "m1"), pin(assistant("a1", "m2")), assistant("a2", "m3")]

    removal = remove_messages(messages, frozenset({"m2", "m3"}))

    assert removal == [messages[0], messages[1]]


def test_a_non_empty_history_never_projects_empty() -> None:
    """With no user turn and no pin, the first message is promoted rather than sending an empty list."""
    messages = [assistant("a1", "m1"), assistant("a2", "m2")]

    removal = remove_messages(messages, frozenset({"m1", "m2"}))

    assert removal == [messages[0]]


def test_an_empty_history_stays_empty() -> None:
    """Nothing to keep, and no index to promote."""
    assert remove_messages([], frozenset({"m1"})) == []


def test_validation_is_on_form_only() -> None:
    """Requirement 5.9: content merit plays no part — two identical shapes with different text project alike."""
    rich = [user("q1", "m1"), assistant("a very substantial answer", "m2")]
    poor = [user("q1", "m1"), assistant("ok", "m2")]

    assert len(remove_messages(rich, frozenset({"m2"}))) == len(remove_messages(poor, frozenset({"m2"})))


# --- apply_removal: the wiring over the pair ---------------------------------------------------------------------


def test_apply_removal_returns_the_removal_and_the_request() -> None:
    """The compaction needs both halves to compute what actually left."""
    messages = [
        user("q1", "m1"),
        assistant("a1", "d1"),
        tool_use("t1", tracking_id="e1"),
        tool_result("t1", "42", "e2"),
    ]
    state = _GraphState(cards={"A": card("A", ("m1", "d1"), ("e1", "e2"))})
    choice = choice_of(A=CardChoice("title", "description"))

    removal, requested = apply_removal(messages, state, choice, frozenset())

    assert requested == frozenset({"m1", "d1", "e1", "e2"})
    # ``m1`` is the leading user turn and stays, so the pair leaves and the request overstates what left.
    assert removal == [messages[0]]


def test_apply_removal_is_the_identity_pass_on_a_full_choice() -> None:
    """An empty request comes back as the received list object, no copy."""
    messages = [user("q1", "m1"), assistant("a1", "d1")]
    state = _GraphState(cards={"A": card("A", ("m1", "d1"))})

    removal, requested = apply_removal(messages, state, choice_of(A=CardChoice("full", "full")), frozenset())

    assert removal is messages
    assert requested == frozenset()
