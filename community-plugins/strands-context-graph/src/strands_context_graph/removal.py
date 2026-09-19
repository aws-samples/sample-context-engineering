"""The Removal: which durable identities to drop, and the subsequence that drops them.

Two halves, one module. :func:`removal_ids` asks — it derives the identities from the (Card, part) pair and nothing
else. :func:`remove_messages` decides — it owns the four guards that may keep an identity the request named, and the two
monotone closures that make the fixed point structural. :func:`apply_removal` is wiring over the pair, and is what
``projection`` calls from the ``InvokeModelStage.Input`` middleware: the Removal is computed here, the delivery of a new
invocation context (and the fold of the surviving Descriptions) is decided there.

It asks, it does not decide, because ``remove_messages`` may preserve any requested identity by pin, by the first user
message, or by tool-pair reconciliation; the compaction computes what actually left by comparing the request against the
returned list. Hence ``ids`` and not ``dropped``.

Absence carries the lag between the two halves: the graph is derived on ``MessageAddedEvent`` and read on
``InvokeModelStage.Input``, so a call can land before the previous turn's Card exists or while the current turn is still
open. Neither needs a check, since an underived Card has no key in ``state.cards`` and the turn in progress is
subtracted via ``current_turn_ids``. A failed derivation costs full content for that turn with no conditional to get
wrong. A title missing from the choice reads as full, the same fail-safe direction. An artifact Card gets no special
case; keeping raw tool returns out of full content is upheld upstream in the scoring.

``remove_messages`` runs on the call's ``context.messages``, the defensive copy built when the model call is assembled,
never on ``agent.messages``. It is a pure function of ``(messages, drop_ids)`` and derives nothing about the graph: the
request arrives as an already-computed frozen set, so a derivation finishing mid-turn cannot change the selection under
the Removal's feet. It reads form only — tool pair, first user message, pin — and never the merit of any content
(Requirement 5.9).

**Private-API dependency** (Requirement 17.4). ``_get_tool_use_ids`` from
``strands.agent.conversation_manager.compression.pin_message`` is imported statically, alongside the public
``is_pinned``, and is a *hard* dependency: a rename fails at import rather than degrading. That is deliberate. A local
copy of the rule would drift from the SDK's, and the failure mode of a drifted tool-pair index is a ``toolUse`` sent
without its ``toolResult`` — a protocol error the provider rejects, discovered at runtime and attributed to the model.
Loud at import beats silent at inference. A public tool-pair accessor published next to ``is_pinned`` would remove the
coupling entirely; see the README's private-API table for the whole set.
"""

from __future__ import annotations

from collections import deque

from strands.agent.conversation_manager.compression.pin_message import _get_tool_use_ids, is_pinned
from strands.types.content import Messages

from .state import Resolution, TurnChoice, _GraphState

__all__ = [
    "apply_removal",
    "remove_messages",
    "removal_ids",
]


def removal_ids(
    state: _GraphState,
    choice: TurnChoice,
    current_turn_ids: frozenset[str],
) -> frozenset[str]:
    """Durable identities the Removal should attempt to drop.

    Derives from the (Card, part) pair and nothing else, mutating neither the graph state nor ``agent.messages``.

    Args:
        state: The graph state. Read only.
        choice: The turn choice, frozen at ``BeforeInvocationEvent``. A title absent from ``choice.by_title`` reads as
            full content.
        current_turn_ids: Durable identities of the turn in progress. Never in the return.

    Returns:
        The requested identities. Empty when the choice keeps every part in full content, making the assembled context
        identical field by field to the one produced without the feature.
    """
    requested: set[str] = set()

    for title, card in state.cards.items():
        card_choice = choice.by_title.get(title)
        if card_choice is None:
            continue

        for part_ids, resolution in (
            (card.dialogue_ids, card_choice.dialogue),
            (card.evidence_ids, card_choice.evidence),
        ):
            if _is_full(resolution):
                continue
            requested.update(part_ids)

    return frozenset(requested - current_turn_ids)


def remove_messages(messages: Messages, drop_ids: frozenset[str]) -> Messages:
    """Return the Removal: the subsequence of messages to send to the provider.

    A subsequence in the strict sense — the same message objects, in the same relative order, with no duplication, no
    insertion and no reordering (Requirement 5.1). Never mutates ``messages`` nor the message dicts inside it, and
    returns the same list object when the request is empty (Requirement 5.2).

    A message is dropped only when its durable identity is in the request. A message without a ``tracking_id``, or with
    one absent from the request, always stays: the structural fail-safe for derivation lag, where a late derivation
    sends more context, never less.

    Three guards override the drop decision. An explicit ``is_pinned`` from the main agent wins over any request
    (Requirement 5.7). The first ``role == "user"`` message is never dropped, since providers reject a conversation that
    does not open with a user turn (Requirement 5.6). And tool pairs are reconciled: a ``toolUse`` and its
    ``toolResult`` always travel together, so dropping one drops the other (Requirement 5.4), unless one end is
    protected, in which case both ends stay (Requirement 5.5) — leaving the ``toolUseId`` sets of the retained
    ``toolUse`` and ``toolResult`` blocks equal (Requirement 5.3).

    Args:
        messages: The turn's message list. Read only.
        drop_ids: Durable identities requested for removal, frozen for the whole turn.

    Returns:
        A subsequence of ``messages`` holding the same message objects in the same relative order; the same list object
        when ``drop_ids`` is empty; never empty for a non-empty input.
    """
    if not drop_ids:
        return messages
    first_user_index = _first_user_index(messages)
    protected = _protected_indices(messages, first_user_index)
    keep = _provisional_keep(messages, drop_ids, protected)
    per_index, groups = _tool_pair_groups(messages)
    _reconcile_tool_pairs(keep, per_index, groups, protected)
    if messages and not any(keep):
        # A non-empty history always projects at least one message: providers reject an empty one. Reachable only with
        # nothing protected (no pin, no user turn) and pair reconciliation dropping whatever pass 1 had kept.
        protected = protected | {0}
        keep[0] = True
        _reconcile_tool_pairs(keep, per_index, groups, protected)
    return [message for index, message in enumerate(messages) if keep[index]]


def apply_removal(
    messages: Messages,
    state: _GraphState,
    choice: TurnChoice,
    current_turn_ids: frozenset[str],
) -> tuple[Messages, frozenset[str]]:
    """Derive the request and apply it, returning the Removal and the request that produced it.

    The compaction needs both: what actually left is ``requested`` minus the identities still present in the returned
    list, and that subtraction keeps a pinned message from paying for its content twice. Mutates nothing — not
    ``messages``, not a dict inside it, not ``agent.messages``, not the graph state — and returns the same list object
    when the request is empty.

    Args:
        messages: The call's message list. Read only.
        state: The graph state. Read only.
        choice: The turn choice, frozen at ``BeforeInvocationEvent``.
        current_turn_ids: Durable identities of the turn in progress. Never removed.

    Returns:
        The Removal, a subsequence of ``messages`` holding the same message objects in the same relative order with no
        duplication and no insertion, and the request it was derived from.
    """
    requested = removal_ids(state, choice, current_turn_ids)
    return remove_messages(messages, requested), requested


def _is_full(resolution: Resolution) -> bool:
    """Whether a part stays whole, in which case none of its identities is requested."""
    return resolution == "full"


def _first_user_index(messages: Messages) -> int:
    """Index of the first message with ``role == "user"``, or ``-1`` when there is none."""
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            return index
    return -1


def _protected_indices(messages: Messages, first_user_index: int) -> set[int]:
    """Positions a request can never drop: an explicit pin, and the leading user turn.

    Args:
        messages: The turn's message list. Read only.
        first_user_index: Position of the leading user turn, or ``-1`` when absent.

    Returns:
        The set of protected positions.
    """
    protected = {index for index in range(len(messages)) if is_pinned(messages, index)}
    if first_user_index >= 0:
        protected.add(first_user_index)
    return protected


def _provisional_keep(messages: Messages, drop_ids: frozenset[str], protected: set[int]) -> list[bool]:
    """Decide keep or drop per index, before tool pairs are reconciled.

    Provisional: the tool-pair guard may still drop a kept message or promote a dropped one back.

    Args:
        messages: The turn's message list. Read only.
        drop_ids: Durable identities requested for removal.
        protected: Positions no request can drop.

    Returns:
        One flag per index of ``messages``: ``True`` to keep, ``False`` to drop.
    """
    keep = [True] * len(messages)
    for index, message in enumerate(messages):
        tracking_id = message.get("tracking_id")
        if not tracking_id or tracking_id not in drop_ids:
            continue  # Unrequested or unnamed: the fail-safe default is to send it.
        if index in protected:
            continue  # A pin from the main agent, or the user turn the provider requires.
        keep[index] = False
    return keep


def _tool_pair_groups(messages: Messages) -> tuple[list[set[str]], dict[str, list[int]]]:
    """Index the tool pairs of the history in one pass.

    Args:
        messages: The turn's message list. Read only.

    Returns:
        The ``toolUseId`` set carried by each index, and the map from ``toolUseId`` to every index carrying it, i.e. the
        messages forming that pair.
    """
    per_index: list[set[str]] = []
    groups: dict[str, list[int]] = {}
    for index, message in enumerate(messages):
        tool_use_ids = _get_tool_use_ids(message)
        per_index.append(tool_use_ids)
        for tool_use_id in tool_use_ids:
            groups.setdefault(tool_use_id, []).append(index)
    return per_index, groups


def _reconcile_tool_pairs(
    keep: list[bool],
    per_index: list[set[str]],
    groups: dict[str, list[int]],
    protected: set[int],
) -> None:
    """Rewrite ``keep`` in place so no tool pair is ever split.

    Sending a ``toolUse`` without its ``toolResult`` is a protocol error, so a pair is reconciled toward DROP: dropping
    one end drops the other. When one end is protected the reconciliation goes the other way and both ends stay; a
    request never causes a protected message to leave.

    Growth in one direction only makes termination structural, so the reconciliation is finite and leaves no further
    rule applicable (Requirement 5.8): KEEP promotion starts from a fixed set of protected positions and only adds, DROP
    propagation only removes and never touches a promoted position. A message carrying two ``toolUseId``s, one paired
    with a protected end and one with a dropped end, would otherwise flip forever between the two rules.

    Args:
        keep: Per-index flags from pass 1. Mutated in place.
        per_index: ``toolUseId`` set carried by each index.
        groups: Map from ``toolUseId`` to the indices forming that pair.
        protected: Positions no request can drop.
    """
    # Closure 1 — KEEP: protection reaches every partner, transitively.
    must_keep = set(protected)
    pending = deque(protected)
    while pending:
        for tool_use_id in per_index[pending.popleft()]:
            for partner in groups[tool_use_id]:
                if partner not in must_keep:
                    must_keep.add(partner)
                    pending.append(partner)
    for index in must_keep:
        keep[index] = True
    # Closure 2 — DROP: a dropped end takes its partners with it, except the promoted ones.
    dropped = {index for index, kept in enumerate(keep) if not kept}
    pending = deque(dropped)
    while pending:
        for tool_use_id in per_index[pending.popleft()]:
            for partner in groups[tool_use_id]:
                if partner in dropped or partner in must_keep:
                    continue
                dropped.add(partner)
                keep[partner] = False
                pending.append(partner)
