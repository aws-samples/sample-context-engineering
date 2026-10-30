"""Practice D (context graph) for the OpenCode host: the neutral core behind the bridge's ``graph_*`` requests.

OpenCode keeps the transcript. The TypeScript plugin hands this module the messages OpenCode is about to send to the model
(the ``experimental.chat.messages.transform`` hook), in a small serialized shape, and receives back **which OpenCode
messages to leave out of that one request** and **one text block to append**. Nothing here writes to the host: the stored
session is the host's, the projection is request-only, and a failure anywhere answers "change nothing".

The graph itself is :func:`context_core.graph.project`, unchanged. This module is only the binding, and the binding has
four jobs:

1. **Convert** the OpenCode transcript to the neutral shape (:func:`to_neutral`). An OpenCode assistant message carries the
   tool call and its result together, so it becomes *two* neutral messages -- the assistant's text and ``toolUse`` blocks,
   then a ``user`` message of ``toolResult`` blocks -- that share one ``tracking_id``: the OpenCode message id. Without a
   ``tracking_id`` a message has no Card and the graph silently does nothing, so every neutral message carries one.
2. **Project** it, and read the result back in OpenCode terms: a message is removed when *none* of the neutral messages it
   produced survived the projection (conservative: a partly surviving message stays whole), and the block is the text the
   core folded into the last ``user`` message.
3. **Keep the per-conversation state**, keyed by OpenCode session id, so a subagent session has a graph of its own.
4. **Offer the two retrieval tools** (``expand_card`` and ``find_context``) as host-neutral functions over that state.
   ``expand_artifact`` is deliberately absent: it resolves references some offloader minted, and OpenCode has none.

How the host's calling pattern maps onto the core, which is what keeps this faithful to the Strands plugin:

- OpenCode calls the hook once per *model call* (every step of a tool loop), not once per user turn. The Strands plugin
  scores once per turn -- at ``BeforeInvocationEvent`` -- freezes the choice, and delivers it on every model call of the
  turn, so the context cannot shift mid-reasoning and a retrieval tool's raise lasts exactly the rest of the turn. Here
  the first call of a user turn runs :func:`~context_core.graph.project` (close the last turn into a Card, score, freeze,
  deliver) and every later call of the same turn runs :func:`~context_core.graph.deliver` over the frozen choice. Computing
  the choice on every call would silently undo ``expand_card``: the evidence axis is decided by message order, so a fresh
  choice puts the Card's tool results straight back into Description.
- The retrieval budget (``max_retrieval_cycles``) is per user turn, and :func:`~context_core.graph.project` resets it, so
  it resets exactly when a new turn opens.
- The fed-back Note ages by a cycle counter. Strands counts event-loop cycles (model calls); here the projection ordinal
  advances once on the turn's first call and once more on every later call, which is the same unit.
- Scoring (the only network use) needs a live admission, exactly as Practice A does; without one the graph still derives
  Cards but never scores, which the core reads as a full pass.

Everything is fail-open. A projection that raises, a transcript that is not a list and a state that no longer matches the
transcript (OpenCode compacted or reverted it) all answer "change nothing", and the last one also rebuilds the graph from
the messages that are left.

Two modes, because the reference algorithm and a prompt cache do not get along
-----------------------------------------------------------------------------

``mode="reference"`` (the default) is the reference practice D: every user turn the graph re-decides, from the question, which
Cards are shown whole, as a description or as a title, and folds what it removed into one block at the end of the request.
Measured live on OpenCode (campaign v7) it cuts the tokens a model reads by about half and raises the bill by about 3x, because
a provider's prompt cache is a prefix cache: a Card that flips back to whole content changes the request early, everything after
it is written to the cache again, and a cache write costs 25 times a read at the price list in use.

``mode="inplace"`` keeps the Cards and the two retrieval tools and replaces the policy. When the next user turn begins, the
bulk of the turn that just closed -- the assistant messages that call tools -- becomes ONE short message *at the position the
turn had* (its Card: the title, the tools it used and the numeric lines it produced, copied literally), while the user's
question and the assistant's own text answers stay as written. By default (``keep_dialogue=0``) nothing stays: the whole assistant
side of the turn is that one message, directly after the user's question, from the moment the turn closes. ``keep_dialogue=N``
keeps the question and the text answers of the newest N closed turns, and ``keep_recent`` turns can be left untouched before
those. The decision is a pure function of the transcript, so it needs no state, no embedding model and no network, gives the
same answer after a restart, and is never reversed by the projection. A recall (``expand_card``) returns the turn's content as
the tool result, at the end of the request, instead of reinserting it where it was.

Why a turn is final the moment it closes (measured, campaigns v8 and v9 on OpenCode): the request is read from the provider's
cache only up to a checkpoint some earlier request wrote, and OpenCode puts one on the system prompt and on the LAST TWO messages
of each request. The first call of a turn ends at the user's question, so the question is a checkpoint, and replacing what
follows it by a summary when the turn closes keeps the cache warm right up to the question: only the summary and the next
question are written. A second change to the same turn one turn later (what ``keep_dialogue=1`` does when it drops the answer)
lands where no request ended, and the whole compact history after the system prompt is written again, every turn: 57 of 57 first
calls in each of three repetitions, 3x the baseline's cache writes, +16.6 % dollars. Final at close: -31.1 % dollars, 1.0-1.4x
the baseline's writes. What this gives up against the reference is the relevance decision, so it is a variant of D, closer to
the "context curator" idea the design folded into D, and it is gated and reported as such.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .graph import Card, GraphState, Thresholds, deliver, is_turn_boundary, project
from .graph.describe import title_for
from .graph.matcher import SimilarityMatcher
from .graph.retrieval import expand_card as _expand_card
from .graph.retrieval import find_context as _find_context
from .graph.retrieval import recall_card as _recall_card
from .message import NeutralMessage

__all__ = [
    "COLLAPSE_NOTE",
    "DEFAULT_EMBED_MODEL_ID",
    "MODES",
    "RETRIEVAL_TOOLS",
    "Collapse",
    "GraphConfig",
    "GraphConfigError",
    "OpenCodeGraph",
    "card_id",
    "plan_inplace",
    "to_neutral",
    "wire_turn_ranges",
]

_LOGGER = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL_ID = "cohere.embed-multilingual-v3"
"""The embedding model the graph's thresholds are calibrated against. Another model needs its own calibration."""

RETRIEVAL_TOOLS = ("expand_card", "find_context")
"""The retrieval tools the OpenCode plugin registers when the graph is on, named by the final block's guidance."""

_CLEARED_OUTPUT = "[Old tool result content cleared]"
"""What OpenCode sends the model for a tool result it has marked compacted; the serializer writes it into ``output``."""

MODES = ("reference", "inplace")
"""The two ways the binding decides what to take out of a request. See the module docstring."""

COLLAPSE_NOTE = (
    "Earlier turns of this conversation are kept as collapsed summaries like the ones below: the title, the tools used and "
    "the numeric lines are copied literally. To get one back in full, call expand_card with its title exactly as shown, or "
    "find_context with a description of what you need."
)
"""Said once, in the summary of the earliest collapsed turn, which never changes, so it costs nothing after the first request."""

_MAX_MESSAGES = 20_000
"""A transcript longer than this is not projected: a bound on work, far above any real session."""

_COUNTERS = (
    "projections",
    "choices",
    "deliveries",
    "full_passes",
    "engaged",
    "removed_messages",
    "block_chars",
    "expand_calls",
    "find_calls",
    "resets",
    "scoring_unavailable",
    "errors",
)
"""The payload-free counters a session keeps, in the order the status reports them."""

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


class GraphConfigError(ValueError):
    """A ``CONTEXT_GRAPH*`` environment variable is malformed. The message names the variable, never a value that matters."""


def _flag(env: Mapping[str, str], key: str) -> bool:
    raw = (env.get(key) or "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise GraphConfigError(f"{key} must be on or off, got {raw!r}")


def _number(env: Mapping[str, str], key: str, default: float, *, low: float, high: float) -> float:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise GraphConfigError(f"{key} must be a number, got {raw!r}") from error
    if not low <= value <= high:
        raise GraphConfigError(f"{key} must be within [{low}, {high}], got {value}")
    return value


def _count(env: Mapping[str, str], key: str, default: int, *, floor: int) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise GraphConfigError(f"{key} must be an integer, got {raw!r}") from error
    if value < floor:
        raise GraphConfigError(f"{key} must be at least {floor}, got {value}")
    return value


def _optional_count(env: Mapping[str, str], key: str, default: int | None) -> int | None:
    raw = (env.get(key) or "").strip().lower()
    if not raw:
        return default
    if raw in {"none", "off", "unbounded"}:
        return None
    return _count(env, key, default or 1, floor=1)


def _mode(env: Mapping[str, str]) -> str:
    raw = (env.get("CONTEXT_GRAPH_MODE") or "").strip().lower() or "reference"
    if raw not in MODES:
        raise GraphConfigError(f"CONTEXT_GRAPH_MODE must be one of {', '.join(MODES)}, got {raw!r}")
    return raw


@dataclass(frozen=True)
class GraphConfig:
    """Practice D's configuration, resolved from the environment. Off unless ``CONTEXT_GRAPH`` says on.

    The defaults are the benchmark's ``GRAPH_TUNING`` (the one set every Strands, LangGraph and Hermes graph arm ran
    under), so a run here is comparable with the published figures: a changed knob is a different experiment.

    Attributes:
        enabled: Whether the graph runs at all. Off, every ``graph_*`` request answers ``skipped`` and the host is
            untouched.
        expand_threshold: Note at or above which a Card is Full Content, budget permitting.
        collapse_floor: Note below which a Card keeps only its Title.
        link_threshold: Similarity at or above which two Cards link.
        description_tokens: Token ceiling of a Card's Description.
        body_budget: Token ceiling across the Cards in Full Content, or ``None`` for no ceiling.
        max_retrieval_cycles: Retrieval calls one user turn may spend across the two tools, or ``None`` for unbounded.
        reuse_ttl_cycles: Projections a fed-back Note survives.
        tags_per_card: Tags derived per Card, which ``find_context`` matches on.
        rarity_weight: How much rarity counts when Tags are selected.
        min_cards: Below this many Cards the choice is skipped.
        neighbors_per_candidate: ``similar`` neighbours ``find_context`` lists under each candidate.
        embed_model_id: The Bedrock embedding model the matcher uses.
        mode: ``reference`` (the practice as the other hosts run it) or ``inplace`` (collapse each closed turn where it is,
            once, for good; see the module docstring). ``inplace`` always names Cards by turn, as ``unique_titles`` does.
        keep_recent: In ``inplace`` mode, how many of the most recent closed turns stay entirely as they were. Each one kept
            whole is rewritten into the cache, tool results included, when it collapses one turn later, so this is a cost knob as
            much as a quality one. ``0`` keeps none.
        keep_dialogue: In ``inplace`` mode, how many closed turns come next in age: their tool-calling assistant messages are
            already one summary, and their question and text answers are still there. Past ``keep_recent + keep_dialogue``
            turns a turn is one summary. ``0`` (the default, and the configuration measured to pay: campaign v9) collapses each
            turn completely as soon as it is out of the ``keep_recent`` window. Anything above 0 changes a turn twice, the
            second time where no earlier request left a cache checkpoint; campaign v8 measured that at +16.6 % dollars.
        recall_max_chars: In ``inplace`` mode, the most characters one ``expand_card`` call returns, shared by the turns asked for.
        unique_titles: Give every user turn its own Card even when two turns open with the same words. The graph names a
            Card by a literal prefix of the user's message, so a later turn with the same prefix silently takes over the
            earlier turn's Card: the earlier turn is then in no Card, is never collapsed, and the Card's id points at
            different messages over time. With this on, the second occurrence of a title is read by the graph as
            ``(#2) <the user's words>``, the third as ``(#3) ...``. Off by default, which is the reference behaviour.
    """

    enabled: bool = False
    expand_threshold: float = 0.62
    collapse_floor: float = 0.45
    link_threshold: float = 0.50
    description_tokens: int = 100
    body_budget: int | None = 40_000
    max_retrieval_cycles: int | None = 8
    reuse_ttl_cycles: int = 5
    tags_per_card: int = 5
    rarity_weight: float = 0.70
    min_cards: int = 3
    neighbors_per_candidate: int = 0
    embed_model_id: str = DEFAULT_EMBED_MODEL_ID
    unique_titles: bool = False
    mode: str = "reference"
    keep_recent: int = 0
    keep_dialogue: int = 0
    recall_max_chars: int = 40_000

    @property
    def names_cards_by_turn(self) -> bool:
        """Whether every user turn gets a Card of its own (``unique_titles``, and always in ``inplace`` mode)."""
        return self.unique_titles or self.mode == "inplace"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> GraphConfig:
        """Build the configuration from ``env``.

        Args:
            env: The environment mapping (usually ``os.environ``).

        Returns:
            The configuration. With ``CONTEXT_GRAPH`` unset or off, the other variables are not read.

        Raises:
            GraphConfigError: When the graph is on and a variable is malformed, or the thresholds are inconsistent.
        """
        if not _flag(env, "CONTEXT_GRAPH"):
            return cls()

        defaults = cls()
        expand = _number(env, "CONTEXT_GRAPH_EXPAND_THRESHOLD", defaults.expand_threshold, low=0.0, high=1.0)
        collapse = _number(env, "CONTEXT_GRAPH_COLLAPSE_FLOOR", defaults.collapse_floor, low=0.0, high=1.0)
        if collapse > expand:
            raise GraphConfigError("CONTEXT_GRAPH_COLLAPSE_FLOOR must not exceed CONTEXT_GRAPH_EXPAND_THRESHOLD")
        return cls(
            enabled=True,
            expand_threshold=expand,
            collapse_floor=collapse,
            link_threshold=_number(env, "CONTEXT_GRAPH_LINK_THRESHOLD", defaults.link_threshold, low=0.0, high=1.0),
            description_tokens=_count(env, "CONTEXT_GRAPH_DESCRIPTION_TOKENS", defaults.description_tokens, floor=1),
            body_budget=_optional_count(env, "CONTEXT_GRAPH_BODY_BUDGET", defaults.body_budget),
            max_retrieval_cycles=_optional_count(
                env, "CONTEXT_GRAPH_MAX_RETRIEVAL_CYCLES", defaults.max_retrieval_cycles
            ),
            reuse_ttl_cycles=_count(env, "CONTEXT_GRAPH_REUSE_TTL", defaults.reuse_ttl_cycles, floor=0),
            tags_per_card=_count(env, "CONTEXT_GRAPH_TAGS_PER_CARD", defaults.tags_per_card, floor=1),
            rarity_weight=_number(env, "CONTEXT_GRAPH_RARITY_WEIGHT", defaults.rarity_weight, low=0.0, high=1.0),
            min_cards=_count(env, "CONTEXT_GRAPH_MIN_CARDS", defaults.min_cards, floor=1),
            neighbors_per_candidate=_count(env, "CONTEXT_GRAPH_NEIGHBORS", defaults.neighbors_per_candidate, floor=0),
            embed_model_id=(env.get("CONTEXT_GRAPH_EMBED_MODEL_ID") or "").strip() or DEFAULT_EMBED_MODEL_ID,
            unique_titles=_flag(env, "CONTEXT_GRAPH_UNIQUE_TITLES"),
            mode=_mode(env),
            keep_recent=_count(env, "CONTEXT_GRAPH_KEEP_RECENT", defaults.keep_recent, floor=0),
            keep_dialogue=_count(env, "CONTEXT_GRAPH_KEEP_DIALOGUE", defaults.keep_dialogue, floor=0),
            recall_max_chars=_count(env, "CONTEXT_GRAPH_RECALL_MAX_CHARS", defaults.recall_max_chars, floor=1_000),
        )

    def thresholds(self) -> Thresholds:
        """The core's frozen record of what the projection decides under, naming this host's two retrieval tools."""
        return Thresholds(
            expand_threshold=self.expand_threshold,
            collapse_floor=self.collapse_floor,
            description_tokens=self.description_tokens,
            tags_per_card=self.tags_per_card,
            rarity_weight=self.rarity_weight,
            min_cards=self.min_cards,
            link_threshold=self.link_threshold,
            reuse_ttl_cycles=self.reuse_ttl_cycles,
            retrieval_tools=RETRIEVAL_TOOLS,
        )

    def summary(self) -> dict[str, Any]:
        """The knobs as plain numbers, for the startup response and the run record. No secret is ever in here."""
        return {
            "enabled": self.enabled,
            "expand_threshold": self.expand_threshold,
            "collapse_floor": self.collapse_floor,
            "link_threshold": self.link_threshold,
            "description_tokens": self.description_tokens,
            "body_budget": self.body_budget,
            "max_retrieval_cycles": self.max_retrieval_cycles,
            "reuse_ttl_cycles": self.reuse_ttl_cycles,
            "tags_per_card": self.tags_per_card,
            "rarity_weight": self.rarity_weight,
            "min_cards": self.min_cards,
            "neighbors_per_candidate": self.neighbors_per_candidate,
            "embed_model_id": self.embed_model_id,
            "unique_titles": self.unique_titles,
            "mode": self.mode,
            "keep_recent": self.keep_recent,
            "keep_dialogue": self.keep_dialogue,
            "recall_max_chars": self.recall_max_chars,
        }


# -- the OpenCode transcript, as the neutral shape ---------------------------------------------------


def _text_of(part: Mapping[str, Any]) -> str | None:
    """The text of a ``text`` part, or ``None`` for any other kind of part or an empty text."""
    if part.get("type") != "text":
        return None
    text = part.get("text")
    return text if isinstance(text, str) and text.strip() else None


def _tool_halves(part: Mapping[str, Any]) -> tuple[NeutralMessage, NeutralMessage] | None:
    """The ``toolUse`` and ``toolResult`` blocks of one tool part, or ``None`` while the call has no result yet.

    A call that is still pending or running has no result to pair with, and a ``toolUse`` sent without its
    ``toolResult`` is a protocol error, so it is left out of the neutral view altogether.
    """
    call_id = part.get("callID")
    status = part.get("status")
    if not isinstance(call_id, str) or not call_id or status not in {"completed", "error"}:
        return None
    raw_input = part.get("input")
    name = part.get("tool") if isinstance(part.get("tool"), str) else ""
    if status == "completed":
        output = part.get("output")
        result_text = output if isinstance(output, str) else ""
    else:
        error = part.get("error")
        result_text = error if isinstance(error, str) else ""
    use: NeutralMessage = {
        "toolUse": {"toolUseId": call_id, "name": name, "input": raw_input if isinstance(raw_input, dict) else {}}
    }
    result: NeutralMessage = {
        "toolResult": {
            "toolUseId": call_id,
            "status": "success" if status == "completed" else "error",
            "content": [{"text": result_text}],
        }
    }
    return use, result


def _with_unique_title(texts: list[str], seen: set[str]) -> list[str]:
    """``texts`` with a ``(#n) `` marker in front when their title was already taken by an earlier turn.

    The marker depends only on the turns before this one, so a turn keeps the same title (and so the same Card id) as the
    conversation grows.
    """
    words = " ".join(texts)
    marker = ""
    occurrence = 1
    while title_for(marker + words) in seen:
        occurrence += 1
        marker = f"(#{occurrence}) "
    seen.add(title_for(marker + words))
    return [marker + texts[0], *texts[1:]] if marker else texts


def to_neutral(
    transcript: Sequence[Mapping[str, Any]], *, unique_titles: bool = False, skip_tools: Collection[str] = ()
) -> list[NeutralMessage]:
    """Convert a serialized OpenCode transcript to neutral messages, each carrying its OpenCode id as ``tracking_id``.

    The wire shape (one entry per OpenCode message, oldest first)::

        {"id": str, "role": "user" | "assistant",
         "parts": [{"type": "text", "text": str, "synthetic"?: bool, "ignored"?: bool}
                   | {"type": "tool", "tool": str, "callID": str, "status": str,
                      "input": object, "output"?: str, "error"?: str}]}

    A user message becomes one neutral ``user`` message of its real text (synthetic and ignored text is the host's own
    plumbing and does not count as the user's words); one with no text opens no turn and is skipped. An assistant message
    becomes an ``assistant`` message (its text, then its ``toolUse`` blocks) and, when it called tools, a ``user`` message
    of the matching ``toolResult`` blocks, both under the same ``tracking_id``. Anything else (reasoning, step markers,
    files) has no neutral form and is left to the host: a message that yields nothing is never a removal candidate.

    Args:
        transcript: The serialized messages, oldest first. Malformed entries are skipped.
        unique_titles: Read a user turn whose opening words repeat an earlier turn's as ``(#n) <words>``, so that the graph
            gives it a Card of its own. See :attr:`GraphConfig.unique_titles`.
        skip_tools: Tool names whose calls have no neutral form. The graph's own retrieval tools go here in ``inplace`` mode:
            what a recall returned is the old turn's content, and describing the NEW turn with it would put the old turn's
            numbers into the new turn's Card.

    Returns:
        The neutral messages, in order. Every one carries a ``tracking_id`` and a list ``content``.
    """
    neutral: list[NeutralMessage] = []
    seen_titles: set[str] = set()
    for entry in transcript:
        if not isinstance(entry, Mapping):
            continue
        message_id = entry.get("id")
        role = entry.get("role")
        raw_parts = entry.get("parts")
        if not isinstance(message_id, str) or not message_id or not isinstance(raw_parts, Sequence):
            continue
        parts = [part for part in raw_parts if isinstance(part, Mapping)]

        if role == "user":
            texts = [
                text
                for part in parts
                if not part.get("synthetic") and not part.get("ignored") and (text := _text_of(part)) is not None
            ]
            if texts:
                if unique_titles:
                    texts = _with_unique_title(texts, seen_titles)
                neutral.append(
                    {"role": "user", "content": [{"text": text} for text in texts], "tracking_id": message_id}
                )
        elif role == "assistant":
            content: list[NeutralMessage] = [
                {"text": text} for part in parts if not part.get("ignored") and (text := _text_of(part)) is not None
            ]
            results: list[NeutralMessage] = []
            for part in parts:
                if part.get("type") != "tool" or part.get("tool") in skip_tools:
                    continue
                halves = _tool_halves(part)
                if halves is not None:
                    content.append(halves[0])
                    results.append(halves[1])
            if content:
                neutral.append({"role": "assistant", "content": content, "tracking_id": message_id})
            if results:
                neutral.append({"role": "user", "content": results, "tracking_id": message_id})
    return neutral


def card_id(session_id: str, title: str) -> str:
    """The opaque identity of a Card in the evidence: stable for the same turn, and never shared between sessions.

    The graph names a Card by its Title, which is a literal prefix of the user's words. Evidence is payload-free, so it
    carries a digest of the title scoped to the session instead: the same Card keeps the same id in every request of a
    session (and after a restart), and two sessions cannot collide even when their users asked the same thing.
    """
    return "c_" + hashlib.sha256(f"{session_id}\0{title}".encode()).hexdigest()[:16]


def _last_user(messages: Sequence[NeutralMessage]) -> NeutralMessage | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message
    return None


def _folded_block(original: Sequence[NeutralMessage], projected: Sequence[NeutralMessage]) -> str | None:
    """Read the text the core folded into the last ``user`` message, by comparing it with the original.

    The core appends exactly one text block to the last ``user`` message and touches nothing else of it, so the block is
    that message's content past the original length. The last ``user`` message belongs to the turn in progress, which the
    removal never touches, so it is the same message in both lists.
    """
    before = _last_user(original)
    after = _last_user(projected)
    if before is None or after is None:
        return None
    old = before.get("content") or []
    new = after.get("content") or []
    if len(new) != len(old) + 1 or list(new[:-1]) != list(old):
        return None
    tail = new[-1]
    text = tail.get("text") if isinstance(tail, dict) else None
    if not isinstance(text, str) or not text.strip():
        return None
    return text.lstrip("\n")


def _open_turn(neutral: Sequence[NeutralMessage]) -> str | None:
    """The OpenCode id of the message that opened the turn in progress, or ``None`` when no message opens one."""
    for message in reversed(neutral):
        if is_turn_boundary(message):
            identity = message.get("tracking_id")
            return identity if isinstance(identity, str) else None
    return None


# -- the in-place plan: which closed turns collapse, and into what ------------------------------------


@dataclass(frozen=True)
class Collapse:
    """One closed turn whose assistant side is replaced, in the request, by a single short message.

    Attributes:
        at_id: The OpenCode id of the first message the summary replaces. The summary takes that message's position.
        removed_ids: The ids of the assistant messages the summary replaces, ``at_id`` first: the whole assistant side of the
            turn, or only its tool-calling messages while the turn still keeps its dialogue. All of them leave the request.
        text: The summary: the turn's Card Description in a ``<collapsed_turn>`` wrapper, and the explanation of the
            wrapper in front of it when this is the earliest collapse of the conversation.
        title: The Title of the turn's Card, which ``expand_card`` takes.
    """

    at_id: str
    removed_ids: tuple[str, ...]
    text: str
    title: str


def _real_text_in(entry: Mapping[str, Any]) -> bool:
    """Whether a serialized user message carries text the user wrote (not synthetic, not ignored): what opens a turn."""
    parts = entry.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return False
    return any(
        isinstance(part, Mapping)
        and not part.get("synthetic")
        and not part.get("ignored")
        and _text_of(part) is not None
        for part in parts
    )


def wire_turn_ranges(transcript: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int], ...]:
    """Split a serialized transcript into ``(start, stop)`` index ranges, one per user turn, in order.

    The rule is the one :func:`to_neutral` applies, so the ranges here and the turns the graph derives Cards for are the
    same turns: a turn opens at a user message with real text (synthetic and ignored text does not count) and runs up to
    the next one. Messages before the first turn belong to none. The last range is the turn in progress.

    Args:
        transcript: The serialized messages, oldest first. Read only.

    Returns:
        The ranges, in order. Empty when no message opens a turn.
    """
    starts = [
        index
        for index, entry in enumerate(transcript)
        if isinstance(entry, Mapping) and entry.get("role") == "user" and _real_text_in(entry)
    ]
    return tuple(
        (start, starts[position + 1] if position + 1 < len(starts) else len(transcript))
        for position, start in enumerate(starts)
    )


def _calls_a_tool(entry: Mapping[str, Any]) -> bool:
    """Whether a serialized assistant message has a tool call: it is evidence, the bulk of a turn. Text alone is dialogue."""
    parts = entry.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return False
    return any(isinstance(part, Mapping) and part.get("type") == "tool" for part in parts)


def _collapse_of(
    transcript: Sequence[Mapping[str, Any]],
    start: int,
    stop: int,
    owner: Mapping[str, Card],
    *,
    first: bool,
    dialogue_kept: bool,
) -> Collapse | None:
    """The collapse of the turn at ``transcript[start:stop]``, or ``None`` when the turn must stay as it is.

    A turn collapses only when nothing about it is in doubt: it has a Card; everything after its question is an assistant
    message the host produced for it (a user-role message inside the range is something this code does not understand, and
    a compaction summary is the only copy of what it summarizes); every id is a distinct non-empty string; and the Card
    points only at messages of this range, which is what says the Card was derived from this very turn.

    With ``dialogue_kept`` the summary replaces only the messages that call a tool (a turn with none has nothing to replace
    and stays as it is); the text-only assistant messages stay. Either way the summary is the same text, so a turn that moves
    from the first state to the second keeps the bytes of its summary and loses only the dialogue after it.
    """
    opener = transcript[start].get("id")
    if not isinstance(opener, str):
        return None
    card = owner.get(opener)
    if card is None:
        return None
    assistant_ids: list[str] = []
    evidence_ids: list[str] = []
    for entry in transcript[start + 1 : stop]:
        if not isinstance(entry, Mapping) or entry.get("role") != "assistant" or entry.get("summary"):
            return None
        identity = entry.get("id")
        if not isinstance(identity, str) or not identity or identity in assistant_ids:
            return None
        assistant_ids.append(identity)
        if _calls_a_tool(entry):
            evidence_ids.append(identity)
    if not assistant_ids or not {*card.dialogue_ids, *card.evidence_ids} <= {opener, *assistant_ids}:
        return None
    removed = evidence_ids if dialogue_kept else assistant_ids
    if not removed:
        return None
    summary = f"<collapsed_turn>\n{card.description}\n</collapsed_turn>"
    return Collapse(
        at_id=removed[0],
        removed_ids=tuple(removed),
        text=f"{COLLAPSE_NOTE}\n\n{summary}" if first else summary,
        title=card.title,
    )


def plan_inplace(
    transcript: Sequence[Mapping[str, Any]], cards: Mapping[str, Card], keep_recent: int, keep_dialogue: int = 0
) -> list[Collapse]:
    """Which closed turns collapse in this request, and into what.

    Counting closed turns from the newest (age 1), a turn is in one of three states: as it was (age up to ``keep_recent``);
    its tool-calling messages one summary, its question and text answers kept (the next ``keep_dialogue`` ages); and one
    summary for the whole assistant side of the turn (older). The newest turn is the one that just closed, because the turn in
    progress is never touched.

    A pure function of the transcript and the Cards derived from it. It reads no score, no earlier answer and no clock, so
    the same transcript gives the same plan on the next request, after a restart, and in a replay. A turn moves from one state
    to the next only when a newer turn closes, and the bytes of its summary do not change when it does, so the start of the
    request is the same from one request to the next except at the one place the newest change landed.

    Args:
        transcript: The serialized messages, oldest first. Read only.
        cards: The session's Cards by Title. Read only.
        keep_recent: How many of the most recent closed turns stay as they were.
        keep_dialogue: How many closed turns after those keep their question and text answers.

    Returns:
        The collapses in transcript order. A closed turn that cannot be collapsed safely is missing from the plan, and so
        stays as it is. The earliest collapse carries the note that explains the summaries.
    """
    ranges = wire_turn_ranges(transcript)
    closed = ranges[:-1]
    collapsing = max(0, len(closed) - keep_recent)
    owner = {
        identity: card
        for card in cards.values()
        if card.kind == "subject"
        for identity in (*card.dialogue_ids, *card.evidence_ids)
    }
    plan: list[Collapse] = []
    for index, (start, stop) in enumerate(closed[:collapsing]):
        collapse = _collapse_of(
            transcript, start, stop, owner, first=not plan, dialogue_kept=index >= collapsing - keep_dialogue
        )
        if collapse is not None:
            plan.append(collapse)
    return plan


def _render_removed(messages: Sequence[Mapping[str, Any]]) -> str:
    """The content of a collapsed turn's assistant messages, as text a model can read: its words, and each tool call.

    A tool call is written as its name and its input on one line, then its result. The text a tool returned is copied
    as it is, because the point of a recall is the exact figures.
    """
    lines: list[str] = []
    for message in messages:
        parts = message.get("parts")
        if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
            continue
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "text" and not part.get("ignored"):
                text = _text_of(part)
                if text is not None:
                    lines.append(text)
            elif part.get("type") == "tool":
                arguments = json.dumps(
                    part.get("input") or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                lines.append(f"[tool {part.get('tool') or '?'} {arguments}]")
                result = part.get("output") if part.get("status") == "completed" else part.get("error")
                lines.append(result if isinstance(result, str) and result else f"(no result: {part.get('status')})")
    return "\n".join(lines)


# -- the per-session state and the bridge-facing API -------------------------------------------------


@dataclass
class _GraphSession:
    """One OpenCode session's graph, with payload-free counters. A subagent session is its own session."""

    state: GraphState | None = None
    open_turn: str | None = None
    projections: int = 0
    choices: int = 0
    deliveries: int = 0
    full_passes: int = 0
    engaged: int = 0
    removed_messages: int = 0
    block_chars: int = 0
    expand_calls: int = 0
    find_calls: int = 0
    resets: int = 0
    scoring_unavailable: int = 0
    errors: int = 0
    last: dict[str, Any] = field(default_factory=dict)
    # In-place mode only. The transcript the last projection was given, which is where a recall reads the content of a
    # collapsed turn from, and the turns that projection collapsed (Title to the ids of the messages its summary replaced).
    wire: Sequence[Mapping[str, Any]] | None = None
    collapsed: dict[str, tuple[str, ...]] = field(default_factory=dict)


class OpenCodeGraph:
    """Practice D over OpenCode sessions: ``project`` for the hook, ``expand_card`` and ``find_context`` for the tools.

    One instance per bridge. It holds no credential and opens no connection: the matcher comes from ``matcher_factory``,
    called at most once and only on a projection that is allowed to score.

    Args:
        config: The resolved graph configuration.
        matcher_factory: Builds the similarity matcher on first need. ``None`` (or a factory that fails) means the graph
            never scores and so never collapses anything.
    """

    def __init__(
        self,
        config: GraphConfig,
        *,
        matcher_factory: Callable[[], SimilarityMatcher | None] | None = None,
    ) -> None:
        self._config = config
        self._thresholds = config.thresholds()
        self._matcher_factory = matcher_factory
        self._matcher: SimilarityMatcher | None = None
        self._matcher_failed = False
        self._sessions: dict[str, _GraphSession] = {}

    @property
    def enabled(self) -> bool:
        """Whether the graph is on."""
        return self._config.enabled

    @property
    def config(self) -> GraphConfig:
        """The configuration this graph runs under."""
        return self._config

    def _scoring_matcher(self, scoring_allowed: bool) -> SimilarityMatcher | None:
        """The matcher, built on first need, or ``None`` when scoring is not allowed or cannot be set up."""
        if not scoring_allowed or self._matcher_failed or self._matcher_factory is None:
            return None
        if self._matcher is None:
            try:
                self._matcher = self._matcher_factory()
            except Exception:
                _LOGGER.warning("graph matcher unavailable; the graph will not score", exc_info=True)
                self._matcher_failed = True
                return None
        return self._matcher

    def project(
        self,
        session_id: str,
        transcript: Sequence[Mapping[str, Any]],
        *,
        scoring_allowed: bool,
    ) -> dict[str, Any]:
        """Project one model call's transcript, and answer which messages to leave out and what to append.

        Args:
            session_id: The OpenCode session the transcript belongs to.
            transcript: The serialized messages, oldest first. See :func:`to_neutral`.
            scoring_allowed: Whether this request may use the network (a live admission backs it). Without it the graph
                still derives Cards and never scores, which is a full pass.

        Returns:
            ``{"operation": "projected", "removed_ids": [...], "final_block": str | None, "stats": {...}}`` when the call
            changes, ``{"operation": "unchanged", "stats": {...}}`` when it does not, ``{"operation": "skipped",
            "reason": ...}`` when the graph is off or the request is unusable, ``{"operation": "error", ...}`` when the
            projection itself failed. Only ``projected`` asks the host to change anything.
        """
        if not self._config.enabled:
            return {"operation": "skipped", "reason": "graph_disabled"}
        if not isinstance(transcript, Sequence) or isinstance(transcript, (str, bytes)):
            return {"operation": "skipped", "reason": "no_messages"}
        if len(transcript) > _MAX_MESSAGES:
            return {"operation": "skipped", "reason": "transcript_too_long"}

        session = self._sessions.setdefault(session_id, _GraphSession())
        try:
            return self._project(session_id, session, transcript, scoring_allowed=scoring_allowed)
        except Exception:
            session.errors += 1
            _LOGGER.warning("graph projection failed; the call goes unchanged", exc_info=True)
            return {"operation": "error", "code": "GRAPH_ERROR"}

    def _project(
        self,
        session_id: str,
        session: _GraphSession,
        transcript: Sequence[Mapping[str, Any]],
        *,
        scoring_allowed: bool,
    ) -> dict[str, Any]:
        inplace = self._config.mode == "inplace"
        neutral = to_neutral(
            transcript,
            unique_titles=self._config.names_cards_by_turn,
            skip_tools=RETRIEVAL_TOOLS if inplace else (),
        )
        order = tuple(dict.fromkeys(str(message["tracking_id"]) for message in neutral))
        known = set(order)

        prior = session.state
        if prior is not None and prior.cards:
            referenced = {
                identity for card in prior.cards.values() for identity in (*card.dialogue_ids, *card.evidence_ids)
            }
            if not referenced <= known:
                # OpenCode compacted or reverted the session: the Cards describe messages that are gone. Start over from
                # what is left; the scan in the core rebuilds the graph from the messages alone.
                session.resets += 1
                prior = None

        open_turn = _open_turn(neutral)
        if inplace:
            return self._project_inplace(session_id, session, transcript, neutral, prior, open_turn)

        matcher: SimilarityMatcher | None = None
        if prior is not None and open_turn is not None and session.open_turn == open_turn:
            # A later model call of the turn the choice was frozen for: deliver it, do not take it again. The projection
            # ordinal is the cycle counter the fed-back Note ages by, and every model call is one cycle.
            new_state = prior
            new_state.turn += 1
            projected: Sequence[NeutralMessage] = deliver(neutral, new_state, self._thresholds)
            session.deliveries += 1
            mode = "delivered"
        else:
            matcher = self._scoring_matcher(scoring_allowed)
            projected, new_state = project(
                neutral,
                state=prior,
                matcher=matcher,
                body_budget=self._config.body_budget,
                thresholds=self._thresholds,
            )
            session.state = new_state
            session.open_turn = open_turn
            session.choices += 1
            mode = "chosen"
            if matcher is not None and len(new_state.cards) >= self._config.min_cards and new_state.choice.full_pass:
                # The core reads a matcher that could not score as "send everything". That is correct, and also silent, so
                # it is counted: a graph that never engages because the embedding model is unreachable must be visible.
                session.scoring_unavailable += 1
        session.projections += 1

        cards = len(new_state.cards)
        collapsed = sum(
            1 for choice in new_state.choice.by_title.values() if choice.dialogue != "full" or choice.evidence != "full"
        )
        stats: dict[str, Any] = {
            "mode": mode,
            "cards": cards,
            "collapsed_cards": collapsed,
            "choice_full_pass": new_state.choice.full_pass,
            "turn": new_state.turn,
            "retrieval_cycles": new_state.retrieval_cycles,
            "scored": matcher is not None,
            # How many times OpenCode compacted or reverted the session under this graph: each one is a defined break in
            # Card identity (the Cards are rebuilt from what is left), which the evidence calls an epoch.
            "resets": session.resets,
        }
        if mode == "chosen":
            # The Cards and the messages each one stands for, as opaque ids. Once per turn: the choice is frozen after.
            stats["card_sources"] = self._card_sources(session_id, new_state)

        if projected is neutral:
            session.full_passes += 1
            session.last = {"operation": "unchanged", **stats}
            return {"operation": "unchanged", "stats": stats}

        surviving = {str(message["tracking_id"]) for message in projected if message.get("tracking_id")}
        removed_ids = [identity for identity in order if identity not in surviving]
        final_block = _folded_block(neutral, projected)
        if not removed_ids and final_block is None:
            session.full_passes += 1
            session.last = {"operation": "unchanged", **stats}
            return {"operation": "unchanged", "stats": stats}

        session.engaged += 1
        session.removed_messages += len(removed_ids)
        session.block_chars += len(final_block or "")
        stats["removed"] = len(removed_ids)
        stats["block_chars"] = len(final_block or "")
        session.last = {"operation": "projected", **stats}
        return {"operation": "projected", "removed_ids": removed_ids, "final_block": final_block, "stats": stats}

    def _project_inplace(
        self,
        session_id: str,
        session: _GraphSession,
        transcript: Sequence[Mapping[str, Any]],
        neutral: Sequence[NeutralMessage],
        prior: GraphState | None,
        open_turn: str | None,
    ) -> dict[str, Any]:
        """The ``inplace`` policy: derive the Cards, then replace the assistant side of each old closed turn by its summary.

        The Cards come from the same core call the reference policy uses, given no matcher, so nothing is scored and the
        call touches no network. It runs once per user turn, on the turn's first model call, because a closed turn's Card
        cannot change while the turn after it is open; the later calls of the turn reuse the state. The plan is computed
        on every call, because OpenCode rebuilds the request from storage on every call and the edit has to be made again.
        """
        if prior is not None and open_turn is not None and session.open_turn == open_turn:
            state = prior
            state.turn += 1
            session.deliveries += 1
            mode = "delivered"
        else:
            _, state = project(neutral, state=prior, matcher=None, body_budget=None, thresholds=self._thresholds)
            session.state = state
            session.open_turn = open_turn
            session.choices += 1
            mode = "chosen"
        session.projections += 1

        session.wire = transcript
        plan = plan_inplace(transcript, state.cards, self._config.keep_recent, self._config.keep_dialogue)
        session.collapsed = {collapse.title: collapse.removed_ids for collapse in plan}

        stats: dict[str, Any] = {
            "mode": mode,
            "policy": "inplace",
            "cards": len(state.cards),
            "collapsed_cards": len(plan),
            "choice_full_pass": not plan,
            "turn": state.turn,
            "retrieval_cycles": state.retrieval_cycles,
            "scored": False,
            "resets": session.resets,
        }
        if mode == "chosen":
            stats["card_sources"] = self._card_sources(session_id, state)

        if not plan:
            session.full_passes += 1
            session.last = {"operation": "unchanged", **stats}
            return {"operation": "unchanged", "stats": stats}

        removed_ids = [identity for collapse in plan for identity in collapse.removed_ids]
        summary_chars = sum(len(collapse.text) for collapse in plan)
        session.engaged += 1
        session.removed_messages += len(removed_ids)
        session.block_chars += summary_chars
        stats["removed"] = len(removed_ids)
        stats["block_chars"] = summary_chars
        session.last = {"operation": "projected", **stats}
        return {
            "operation": "projected",
            "removed_ids": removed_ids,
            "final_block": None,
            "collapses": [
                {"at_id": collapse.at_id, "text": collapse.text, "title": collapse.title} for collapse in plan
            ],
            "stats": stats,
        }

    @staticmethod
    def _card_sources(session_id: str, state: GraphState) -> list[dict[str, Any]]:
        """Each subject Card as ``{"card_id", "sources"}``: the host message ids it stands for, in turn order."""
        cards = sorted(
            (card for card in state.cards.values() if card.kind == "subject"), key=lambda c: (c.turn, c.title)
        )
        return [
            {"card_id": card_id(session_id, card.title), "sources": [*card.dialogue_ids, *card.evidence_ids]}
            for card in cards
        ]

    def card_sources(self, session_id: str) -> list[dict[str, Any]]:
        """The session's Cards and their sources now. Empty when the session has no graph yet."""
        session = self._sessions.get(session_id)
        return [] if session is None or session.state is None else self._card_sources(session_id, session.state)

    # -- the two retrieval tools --------------------------------------------------------------------

    def expand_card(self, session_id: str, titles: Sequence[str] | str) -> str:
        """Raise the named Cards to Full Content for the rest of the turn, and feed the request into the next Note.

        Args:
            session_id: The session the tool was called in.
            titles: The Titles, as the final block showed them.

        Returns:
            The confirmation, the error naming what matched nothing, or the refusal once the turn's budget is spent.
        """
        session = self._sessions.get(session_id)
        if not self._config.enabled or session is None or session.state is None:
            return "expand_card | no earlier turns of this conversation are collapsed yet, so there is nothing to bring back"
        session.expand_calls += 1
        state = session.state
        if self._config.mode == "inplace":
            # Nothing is raised and nothing is put back in the request: the turn's content comes back as this result.
            by_id = {entry.get("id"): entry for entry in session.wire or () if isinstance(entry, Mapping)}

            def content_of(title: str) -> str | None:
                ids = session.collapsed.get(title)
                messages = [by_id[identity] for identity in ids or () if identity in by_id]
                return _render_removed(messages) if ids and len(messages) == len(ids) else None

            return _recall_card(
                state,
                titles,
                collapsed=session.collapsed,
                content_of=content_of,
                max_chars=self._config.recall_max_chars,
                max_retrieval_cycles=self._config.max_retrieval_cycles,
            )
        return _expand_card(
            state,
            titles,
            cycle=state.turn,
            reuse_ttl_cycles=self._config.reuse_ttl_cycles,
            max_retrieval_cycles=self._config.max_retrieval_cycles,
        )

    def find_context(self, session_id: str, need: str, tag: str | None = None, *, scoring_allowed: bool) -> str:
        """Find earlier turns that match ``need`` by their Descriptions.

        Args:
            session_id: The session the tool was called in.
            need: What the model is looking for, in its own words.
            tag: Restrict the candidates to Cards carrying this Tag.
            scoring_allowed: Whether this request may use the network. Without it nothing can be scored.

        Returns:
            At most five candidates, or the empty result naming the need.
        """
        session = self._sessions.get(session_id)
        if not self._config.enabled or session is None or session.state is None:
            return (
                "find_context | no earlier turns of this conversation are collapsed yet, so there is nothing to search"
            )
        session.find_calls += 1
        state = session.state
        return _find_context(
            state,
            need,
            tag,
            matcher=self._scoring_matcher(scoring_allowed),
            collapse_floor=self._config.collapse_floor,
            cycle=state.turn,
            reuse_ttl_cycles=self._config.reuse_ttl_cycles,
            max_retrieval_cycles=self._config.max_retrieval_cycles,
            neighbors_per_candidate=self._config.neighbors_per_candidate,
        )

    # -- lifecycle and evidence ---------------------------------------------------------------------

    def discard(self, session_id: str) -> bool:
        """Forget a session's graph. Returns whether there was one."""
        return self._sessions.pop(session_id, None) is not None

    def status(self) -> dict[str, Any]:
        """Payload-free counters per session and in total: what the graph did, never what it saw.

        ``durable_writes`` is structurally zero. The graph holds its state in this process, writes nothing to the host
        and keeps no file, so there is no code path that could make it anything else; it is reported so the evidence says
        so in the same place as the rest.
        """
        sessions = {
            session_id: {
                **{counter: getattr(session, counter) for counter in _COUNTERS},
                "cards": len(session.state.cards) if session.state is not None else 0,
                "last": session.last,
            }
            for session_id, session in self._sessions.items()
        }
        totals = {
            counter: sum(getattr(session, counter) for session in self._sessions.values()) for counter in _COUNTERS
        }
        return {
            "enabled": self._config.enabled,
            "config": self._config.summary(),
            "matcher_ready": self._matcher is not None,
            "durable_writes": 0,
            "totals": totals,
            "sessions": sessions,
        }
