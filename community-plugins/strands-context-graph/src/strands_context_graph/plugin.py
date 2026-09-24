"""``ContextGraph``: the construct, its configuration, and the validation it enforces.

Validation happens here and only here, so a misconfiguration fails at construction. Construction is pure bookkeeping —
no network call, no model client, no AWS client, no async task, no agent yet — so a construction that raises has
registered nothing (Requirement 2.17). Validation is of shape, never of merit: a bool where a ratio was expected, a
float where a count was expected, a range violation. The one relational check, ``collapse_floor <= expand_threshold``,
is shape too, since a floor above the ceiling leaves the middle resolution unreachable.

There is no ``model`` parameter: the Card is derived by scan, and the similarity matcher is the graph's only remote call
(Requirements 2.18, 2.19).

**Private-API dependencies** (Requirement 17.4). The wiring reads three private surfaces, all of them *soft* and all of
them read through ``getattr`` with a shape check, so what they cost when a rename lands is a notice not emitted, never
an exception:

- ``agent._plugin_registry._plugins`` — the map :func:`_wired_plugins` reads to find a memory manager that arrived in
  the ``plugins=[...]`` list. Unreadable, it reads as "nothing else is wired", so no ordering notice goes out. A public
  read-only view of an agent's plugins would remove it.
- ``_provide_memory_context`` / ``_injection_config`` on a candidate — how :func:`_folds_memory` recognises a memory
  fold by member rather than by type, which is also what lets a test double and a future implementation qualify.
  Renamed, the manager is not recognised and the ordering notice it would have triggered is not emitted. A public
  capability marker for "this plugin folds into the model input" would remove it.
- ``agent._middleware_registry._handlers`` — read by :func:`_delivery_precedes_memory_fold` to see whether anything
  folds *behind* the delivery. Unreadable, the answer is "cannot be shown", which warns rather than reassures: the
  notice is the conservative direction, so degradation here costs a false alarm and never a missed one.

The ordering itself remains the caveat of Requirement 17.2 — it has no standalone escape hatch and fails as a worse
answer rather than an exception — and the delivery path's own couplings are documented in :mod:`.projection`. The
README's private-API table lists the whole set with what would remove each one.
"""

from __future__ import annotations

import logging
import math
import warnings
import weakref
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from strands._middleware.stages import InvokeModelStage
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks.events import AfterToolCallEvent, BeforeInvocationEvent, MessageAddedEvent
from strands.plugins import Plugin
from strands.tools.decorator import tool
from strands.types.tools import ToolContext

from . import tools
from .cards import (
    closed_turn_ranges,
    derive_and_register,
    derive_and_register_artifacts,
    is_turn_boundary,
    link_newly_measurable,
    rebuild_into,
)
from .matcher import SimilarityMatcher
from .projection import Projection
from .scoring import (
    compute_notes,
    distribute,
    expire_reuse,
    full_pass_choice,
    titles_in_turn_order,
    warm_up_choice,
)
from .state import TurnChoice, _GraphState, _GraphStates
from .store import InMemoryReferenceStore, record_references

if TYPE_CHECKING:
    from strands.agent.agent import Agent
    from strands.types.content import Messages
    from strands.types.tools import ToolResult

_ReferenceStores: TypeAlias = "weakref.WeakKeyDictionary[Agent, InMemoryReferenceStore]"
"""Per-agent reference store map, keyed weakly for the same reason ``_GraphStates`` is: a reference is discovered from
one agent's tool results and means nothing on another, and the store is dropped along with the agent it belongs to
(Requirements 14.1, 14.4)."""

__all__ = ["ContextGraph"]

logger = logging.getLogger(__name__)

_DEFAULT_NAME = "strands:context-graph"
"""Default plugin name; override to tell multiple instances apart in logs."""

_DEFAULT_EXPAND_THRESHOLD = 0.55
"""Note at or above which a Card is Full Content, budget permitting."""

_DEFAULT_COLLAPSE_FLOOR = 0.45
"""Note below which a Card keeps only its Title. Cosine similarity of two texts of the same language does not approach
zero, so a floor near zero sits outside the range the matcher answers in. This value and ``link_threshold`` answer to
the default matcher's distribution and do not carry over to another one."""

_DEFAULT_DESCRIPTION_TOKENS = 100
"""Token ceiling of a Description."""

_DEFAULT_TAGS_PER_CARD = 5
"""How many identifiers define a Card."""

_DEFAULT_NEIGHBORS_PER_CANDIDATE = 3
"""How many ``similar`` neighbours ``find_context`` lists under each candidate.

Three rather than zero because the edge already exists and nothing read it: the measurement was paid for
on the write path and the relation it holds -- two turns discussing related things -- is precisely what
the candidate ranking cannot see, since that ranking compares each Description to the question and never
to another Description.

Three rather than more because a neighbour is a hint, not evidence. Each one costs a title and a score,
and a list long enough to need reading would be a second ranking the model has to arbitrate against the
first. The five candidates stay the answer; the neighbours say where else to look.
"""

_DEFAULT_BODY_BUDGET: int | None = None
"""Token ceiling across Cards in Full Content. ``None`` means no ceiling, as explicit configuration."""

_DEFAULT_MIN_CARDS = 3
"""Below this many Cards the whole choice is skipped: the only possible decision is "send it all"."""

_DEFAULT_LINK_THRESHOLD = 0.50
"""Similarity at or above which two Cards link to each other."""

_DEFAULT_REUSE_TTL_CYCLES = 5
"""Model cycles a Fed-Back Note survives."""

_DEFAULT_MAX_RETRIEVAL_CYCLES = 8
"""Retrieval calls one turn may spend before the tools start refusing.

The counter already existed as instrumentation (Requirement 17.8); this is the ceiling it answers to. Without one, a
turn whose evidence is genuinely unreachable has no reason to end: every retrieval tool returns text rather than an
error, so a model that keeps asking keeps being answered. Measured across models, that is not hypothetical -- one turn
spent 346 retrieval calls and 21 minutes before the event loop hit Python's recursion limit, and the same turn on a
model that gives up early still burned 31 calls and answered nothing.

Eight is deliberately above any healthy turn observed (a turn that needs recovery uses one to three calls) and far below
the point where the prompt growth from the calls themselves becomes the problem. Raise it for a scenario that genuinely
walks many turns; set it to ``None`` to restore the old unbounded behaviour.
"""

_ORDERING_WARNING = (
    "context_graph=<ordering> | a memory manager is registered on this agent and this plugin cannot guarantee that it "
    "removes messages before that manager folds its block in the same InvokeModelStage.Input | the graph must project "
    "first, or the folded memory block may be projected out of the call | register the ContextGraph so its delivery "
    "handler is the first input handler of that stage"
)
"""The one wiring-time notice of Requirement 9.6: the ordering requirement, and the risk of violating it.

Developer-facing, so it goes out through ``warnings.warn`` and not the logger: it is about how the plugin is being
wired, not about something that went wrong while it ran, and the standard library already dedupes it once per call site,
which is what "exactly one" means here. Naming a risk is the whole of it — the plugin folds no message back in and
reorders no other plugin's middleware (Requirement 16.7)."""

_ORDERING_WARNING_STACKLEVEL = 4
"""Frames to skip so the notice points at the wiring, not at this module.

``init_agent`` is reached as plugin construction site -> ``Agent.__init__`` -> the plugin registry -> here, the same
depth the SDK's own configuration warnings use. Best effort by nature: the exact depth is the registry's call chain, so
a change there moves the reported line without changing what the notice says."""

_MANAGER_WARNING = (
    "context_graph=<conversation_manager> | conversation_manager=<{manager}> is not a NullConversationManager | a "
    "manager that trims the live history removes messages from agent.messages, so it may drop content this plugin only "
    "meant to fold, and raising that Card's Resolution back up then recovers nothing | pass "
    "conversation_manager=NullConversationManager()"
)
"""The precondition notice of Requirement 1.6, formatted with the manager's type name.

Developer-facing for the same reason the ordering notice is: it is about how the agent is configured, not about
something that went wrong while the plugin ran. It degrades and never blocks — the warned agent is wired exactly like an
agent under ``NullConversationManager`` (Requirement 1.7) — and the manager itself is left alone: not removed, not
replaced, not reconfigured."""

_MANAGER_WARNING_STACKLEVEL = _ORDERING_WARNING_STACKLEVEL
"""Frames to skip, the same count and for the same reason as the ordering notice: both are emitted from ``init_agent``
and both should point at the wiring site rather than at this module."""

_RARITY_WEIGHT = 0.70
"""Weight of rarity against repetition when ranking textual Tag candidates. Fixed rather than configurable: it balances
two terms of one internal ranking, and the construct exposes no parameter for it."""


def _identities_in(messages: Messages) -> tuple[str, ...]:
    """Return the Durable Identities of ``messages``, in order and without duplicates.

    The same shape the rebuild scan hands :func:`~.cards.derive_and_register`, which is what makes the incremental
    construction and the scan produce the same Card for the same turn (Requirements 14.3, 14.5). It is also the one
    place a vanished message drops out of the graph: an identity is collected only from a message that is present in
    ``agent.messages``, so a Card can never reference one that is not (Requirement 14.7).

    Args:
        messages: The messages of one turn, in order. Read only.

    Returns:
        The identities. A message carrying none contributes nothing: it is a message without a Card.
    """
    return tuple(dict.fromkeys(identity for message in messages if (identity := message.get("tracking_id"))))


def _question_of(messages: Messages) -> str:
    """Return the text the turn's Cards are scored against: the texts of the last user message.

    Read off the invocation's input messages rather than ``agent.messages``: ``BeforeInvocationEvent`` fires before the
    turn's message is appended, so the question is not in the history yet.

    Args:
        messages: The messages to read the question from, in order. Read only.

    Returns:
        The texts of the last user message, joined. Empty when no user message carries text, which scores every Card
        against nothing and therefore leaves the whole graph at its floor.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        texts = [
            block["text"]
            for block in message.get("content") or ()
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if texts:
            return " ".join(texts)

    return ""


def _cycle_of(agent: object) -> int:
    """Return the model cycle counter of ``agent``, which is the clock the fed-back Note is aged by.

    ``agent.event_loop_metrics.cycle_count`` and nothing else (Requirement 13.3), so a slow provider call or a burst of
    messages inside one cycle never ages a Note. Read defensively: the retrieval tools reach here from a
    ``ToolContext``, whose ``agent`` is typed loosely for backwards compatibility.

    Args:
        agent: The agent of the call. Read only.

    Returns:
        The counter, or ``0`` when the agent exposes none, granting a Note the full TTL rather than none, the direction
        every other fail-safe here takes.
    """
    count = getattr(getattr(agent, "event_loop_metrics", None), "cycle_count", 0)
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


def _wired_plugins(agent: object) -> tuple[object, ...]:
    """Return everything already wired to ``agent`` that could be a memory manager.

    Two places to look, because the SDK wires a memory manager through either: the plugin registry, when it arrives in
    the ``plugins=[...]`` list, and ``agent.memory_manager``, when it arrives through its own parameter. Both are read
    defensively — the registry's map is private, the attribute is optional — and an agent exposing neither reads as an
    agent with no memory manager, which is the quiet direction.

    Args:
        agent: The agent being wired. Read only.

    Returns:
        The candidates, in no particular order and possibly containing the same object twice.
    """
    plugins = getattr(getattr(agent, "_plugin_registry", None), "_plugins", None)
    candidates = tuple(plugins.values()) if isinstance(plugins, Mapping) else ()
    manager = getattr(agent, "memory_manager", None)
    return candidates if manager is None else (*candidates, manager)


def _folds_memory(plugin: object) -> bool:
    """Return whether ``plugin`` is a memory manager that folds a retrieved block into the model input.

    Checked by member, the way the matcher is: a memory manager is whatever renders memory context for a call, so a test
    double and a future implementation both qualify without inheriting from anything. Injection turned off is not a
    memory fold — nothing is ever folded, so there is no block an ordering violation could project out — and it is read
    here rather than warned about, which is what keeps the notice about a real risk.

    Args:
        plugin: A candidate from :func:`_wired_plugins`. Read only.

    Returns:
        ``True`` when the object folds memory into the call.
    """
    if not callable(getattr(plugin, "_provide_memory_context", None)):
        return False
    return getattr(plugin, "_injection_config", None) is not False


def _delivery_precedes_memory_fold(agent: object, *, delivery_is_first: bool) -> bool:
    """Return whether the graph's delivery is provably the first input handler of the call-assembly stage.

    The one thing this plugin does control: :meth:`~.projection.Projection.register` inserts the delivery at index zero
    of ``InvokeModelStage``'s handler list, and the registry runs input handlers in list order, so a memory fold sitting
    behind it in that same list folds into a message list from which nothing will later be discarded (Requirement 9.5).

    What it cannot control, and therefore cannot claim: a fold registered anywhere other than that list. An unreadable
    registry, an empty list, a list holding nothing but this plugin's own handler, and a build on which the move to
    index zero could not be performed at all read the same way — the fold is not behind us where we can see it — so the
    answer is ``False`` and the caller warns.

    The position is taken from ``delivery_is_first`` rather than re-derived here, because the registry wraps an input
    handler in a closure of its own before storing it: what sits in the list is not the object that was registered, so
    nothing in the list can be compared back to the delivery. The registration reports whether its own move succeeded,
    and that report is the only sound evidence available.

    Args:
        agent: The agent being wired, after the delivery has been registered. Read only.
        delivery_is_first: What :meth:`~.projection.Projection.register` reported: whether it could put the delivery at
            the front of the stage's handler list.

    Returns:
        ``True`` when an input handler follows the delivery in the same stage.
    """
    if not delivery_is_first:
        return False

    handlers = getattr(getattr(agent, "_middleware_registry", None), "_handlers", None)
    tagged = handlers.get(InvokeModelStage) if isinstance(handlers, Mapping) else None
    if not tagged:
        return False
    # Index zero is this plugin's delivery, ``delivery_is_first`` having established it; anything behind it that
    # transforms the call's input is a fold this removal precedes.
    return any(getattr(entry, "phase", None) == "input" for entry in tuple(tagged)[1:])


def _validate_ratio(value: object, parameter: str) -> None:
    """Reject anything that is not a finite real number in the closed range ``[0.0, 1.0]``.

    ``bool`` is rejected explicitly: it passes as a number in Python, and ``True`` silently meaning ``1.0`` is
    configuration that looks like it works. ``nan`` falls out of the range comparison on its own, so it gets no separate
    check. The parameter is typed ``object`` so the checks run on what the caller passed.

    Args:
        value: Value received by the constructor.
        parameter: Name of the parameter, for the message.

    Raises:
        ValueError: When ``value`` is a bool, not a real number, not finite, or out of range.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{parameter}=<{value!r}> | must be a finite real number in the closed range 0.0 to 1.0")


def _validate_count(value: object, parameter: str) -> None:
    """Reject anything that is not an integer greater than or equal to ``1``.

    ``bool`` and ``float`` are both rejected: ``True`` would configure a ceiling of one, and ``2.5`` Tags per Card is
    not a quantity that exists.

    Args:
        value: Value received by the constructor.
        parameter: Name of the parameter, for the message.

    Raises:
        ValueError: When ``value`` is a bool, not an ``int``, or less than ``1``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{parameter}=<{value!r}> | must be an integer greater than or equal to 1")


def _validate_body_budget(body_budget: object) -> None:
    """Reject anything that is neither ``None`` nor an integer greater than or equal to ``1``.

    ``None`` is the absence of a Full Content ceiling and is supported; ``0`` is not, since a budget of zero tokens
    would deny Full Content to every Card while the thresholds claim otherwise.

    Args:
        body_budget: Value received by the constructor.

    Raises:
        ValueError: When ``body_budget`` is neither ``None`` nor an ``int`` of at least ``1``.
    """
    if body_budget is None:
        return
    if isinstance(body_budget, bool) or not isinstance(body_budget, int) or body_budget < 1:
        raise ValueError(f"body_budget=<{body_budget!r}> | must be None or an integer greater than or equal to 1")


def _validate_reuse_ttl_cycles(value: object) -> None:
    """Reject anything that is not an integer greater than or equal to ``0``.

    ``0`` is meaningful here — a Fed-Back Note discarded at the end of the turn that created it — hence the floor of
    zero rather than ``_validate_count``'s floor of one.

    Args:
        value: Value received by the constructor.

    Raises:
        ValueError: When ``value`` is a bool, not an ``int``, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"reuse_ttl_cycles=<{value!r}> | must be an integer greater than or equal to 0")


def _validate_max_retrieval_cycles(value: object) -> None:
    """Reject anything that is neither ``None`` nor an integer greater than or equal to ``1``.

    ``None`` is the explicit opt-out -- unbounded retrieval, the behaviour before the ceiling existed -- so it has to be
    distinguishable from a caller who passed nothing. A ceiling of ``0`` is rejected rather than treated as "never
    retrieve": a Card the model cannot open at all is better expressed by not registering the tools.

    Args:
        value: Value received by the constructor.

    Raises:
        ValueError: When ``value`` is a bool, not ``None`` or an ``int``, or below ``1``.
    """
    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"max_retrieval_cycles=<{value!r}> | must be None or an integer greater than or equal to 1")


def _validate_matcher(matcher: object) -> None:
    """Reject anything that is neither ``None`` nor an object exposing a callable ``score``.

    Checked by member, not by ``isinstance``: the matcher contract is structural, so any object carrying the operation
    is a valid implementation.

    Args:
        matcher: Value received by the constructor.

    Raises:
        ValueError: When ``matcher`` is not ``None`` and lacks a callable ``score``.
    """
    if matcher is None:
        return
    if not callable(getattr(matcher, "score", None)):
        raise ValueError(f"matcher=<{matcher!r}> | must expose a callable 'score' member")


def _validate_name(name: object) -> None:
    """Reject anything that is neither ``None`` nor a string of length greater than zero.

    Args:
        name: Value received by the constructor.

    Raises:
        ValueError: When ``name`` is neither ``None`` nor a non-empty string.
    """
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError(f"name=<{name!r}> | must be None or a non-empty string")


class ContextGraph(Plugin):
    """Projects an agent's short-term memory as a graph of Cards, never mutating ``agent.messages``.

    The graph derives a Card per closed turn by scan and decides a Resolution per Card: Title, Description or Full
    Content. Nothing about the derivation reaches a language model — the only remote call in any configuration is the
    matcher's embedding.

    Wired at four engagement points: three hooks, plus the delivery handler on ``InvokeModelStage.Input``, which is the
    one place the set of messages sent to the provider changes. Three retrieval tools let the model reach back into a
    Card the choice collapsed.

    Pair it with ``NullConversationManager``: this is a precondition, not a suggestion. Another conversation manager
    edits the live message list before the call is assembled, so it can physically drop what the graph only meant to
    fold, and raising a Card's Resolution back up then recovers nothing.

    Args:
        expand_threshold: Note at or above which a Card is Full Content, budget permitting. Defaults to ``0.55``.
        collapse_floor: Note below which a Card keeps only its Title. Defaults to ``0.45``. Must not exceed
            ``expand_threshold``.
        description_tokens: Token ceiling of a Description. Defaults to ``100``.
        tags_per_card: How many identifiers define a Card. Defaults to ``5``.
        neighbors_per_candidate: How many ``similar`` neighbours ``find_context`` lists under each
            candidate it returns. Defaults to ``3``.

            This is the only reader of the ``similar`` edge. The edge is measured on the write path and
            stored with its similarity as the weight, but it propagates no Note by design
            (``_STRUCTURAL_WEIGHTS`` omits it) and no retrieval path traversed it, so it was paid for and
            read by nothing at all.

            It answers a question the ranking cannot: ``find_context`` scores each Description against
            the QUESTION and never against another Description, so two turns that discuss the same thing
            in different words are invisible to each other there. The edge already holds exactly that
            relation. A neighbour costs one title and is what ``expand_card`` takes as its argument, so
            the model can follow one without spending another search. ``0`` lists none.
        body_budget: Token ceiling across Cards in Full Content, or ``None`` for no ceiling. Defaults to ``None``.
        min_cards: Below this many Cards the choice is skipped entirely. Defaults to ``3``.
        link_threshold: Similarity at or above which two Cards link. Defaults to ``0.50``.
        reuse_ttl_cycles: Model cycles a Fed-Back Note survives. Defaults to ``5``. ``0`` discards it at the end of the
            turn that created it.
        max_retrieval_cycles: Retrieval calls one turn may spend before ``expand_card``, ``expand_artifact`` and
            ``find_context`` refuse and tell the model to answer from what it has. Defaults to ``8``. ``None`` restores
            unbounded retrieval. The ceiling exists because no retrieval miss is an error -- every one of these tools
            answers with text -- so a turn whose evidence is unreachable otherwise has nothing to stop it.
        include_artifact_tool: Register ``expand_artifact``. Defaults to ``True``. Pass ``False`` when a plugin that
            offloads tool results is installed beside this one -- typically ``RelevanceFilter`` -- because each then
            ships a retrieval tool over a store the other cannot read, and the model has two plausible tools for one
            job. Measured on this repository's benchmark, excluding it took the three-plugin stack from 84.5% to 94.4%
            weighted accuracy. ``expand_card`` and ``find_context`` are unaffected and have no switch: they reach back
            into the conversation's own turns, which is a job no offloader does.
        matcher: Similarity matcher, or ``None`` for the default asymmetric multilingual embedding. Checked by member,
            so an implementation inherits from nothing. Resolved on first need, so construction opens no client.
        name: Plugin name, for logging and duplicate detection. Defaults to ``"strands:context-graph"``.

    Raises:
        ValueError: On any invalid argument, naming the parameter and what it accepts.

    Example:
        ```python
        from strands import Agent
        from strands.agent.conversation_manager import NullConversationManager
        from strands_context_graph import ContextGraph

        agent = Agent(
            conversation_manager=NullConversationManager(),
            plugins=[ContextGraph()],
        )
        ```
    """

    name = _DEFAULT_NAME

    def __init__(
        self,
        *,
        expand_threshold: float = _DEFAULT_EXPAND_THRESHOLD,
        collapse_floor: float = _DEFAULT_COLLAPSE_FLOOR,
        description_tokens: int = _DEFAULT_DESCRIPTION_TOKENS,
        tags_per_card: int = _DEFAULT_TAGS_PER_CARD,
        neighbors_per_candidate: int = _DEFAULT_NEIGHBORS_PER_CANDIDATE,
        body_budget: int | None = _DEFAULT_BODY_BUDGET,
        min_cards: int = _DEFAULT_MIN_CARDS,
        link_threshold: float = _DEFAULT_LINK_THRESHOLD,
        reuse_ttl_cycles: int = _DEFAULT_REUSE_TTL_CYCLES,
        max_retrieval_cycles: int | None = _DEFAULT_MAX_RETRIEVAL_CYCLES,
        include_artifact_tool: bool = True,
        matcher: SimilarityMatcher | None = None,
        name: str | None = None,
    ) -> None:
        """Validate the configuration and fix it for the lifetime of the instance.

        Every check runs before the first attribute is assigned, so a ``ValueError`` leaves an instance that was never
        handed to an agent and therefore registered nothing (Requirement 2.17). Nothing is built here beyond plain
        attributes: no network call, no model client, no AWS client, no async task (Requirement 2.16).
        """
        _validate_name(name)

        _validate_ratio(expand_threshold, "expand_threshold")
        _validate_ratio(collapse_floor, "collapse_floor")
        _validate_ratio(link_threshold, "link_threshold")
        # Checked after both are known to be ratios: a floor above the ceiling leaves the middle resolution unreachable,
        # so the ladder would have two steps while the configuration says three.
        if float(collapse_floor) > float(expand_threshold):
            raise ValueError(
                f"collapse_floor=<{collapse_floor!r}> | must be less than or equal to "
                f"expand_threshold=<{expand_threshold!r}>"
            )
        _validate_count(description_tokens, "description_tokens")
        _validate_count(tags_per_card, "tags_per_card")
        if (
            isinstance(neighbors_per_candidate, bool)
            or not isinstance(neighbors_per_candidate, int)
            or neighbors_per_candidate < 0
        ):
            raise ValueError(
                f"neighbors_per_candidate=<{neighbors_per_candidate!r}> | must be an integer greater than or equal to 0"
            )
        _validate_count(min_cards, "min_cards")
        _validate_body_budget(body_budget)
        _validate_reuse_ttl_cycles(reuse_ttl_cycles)
        _validate_max_retrieval_cycles(max_retrieval_cycles)
        if not isinstance(include_artifact_tool, bool):
            raise ValueError(f"include_artifact_tool=<{include_artifact_tool!r}> | must be True or False")
        _validate_matcher(matcher)

        self.name = name or _DEFAULT_NAME
        # Fixed here and never revisited: every turn of this instance uses these values unchanged (Requirement 2.15).
        # ``float`` normalizes the ratios, since ``_validate_ratio`` admits any ``Real`` and the rest of the package
        # expects a float. The counts arrive already pinned to their type by their validators.
        self._expand_threshold = float(expand_threshold)
        self._collapse_floor = float(collapse_floor)
        self._description_tokens = description_tokens
        self._tags_per_card = tags_per_card
        self._neighbors_per_candidate = neighbors_per_candidate
        self._body_budget = body_budget
        self._min_cards = min_cards
        self._link_threshold = float(link_threshold)
        self._reuse_ttl_cycles = reuse_ttl_cycles
        self._max_retrieval_cycles = max_retrieval_cycles
        self._include_artifact_tool = include_artifact_tool
        self._matcher = matcher

        # The default matcher, once something has needed it. Kept apart from ``_matcher`` so ``matcher=None`` stays
        # observable as the configuration it was, and resolved on first need: construction opens no client and reaches
        # no network (Requirements 2.14, 2.16).
        self._resolved_matcher: SimilarityMatcher | None = None
        # Per agent, weakly keyed: the graph is dropped along with the agent it belongs to, and two agents wired to this
        # same instance never share a Card (Requirements 14.1, 14.4, 1.5).
        self._states: _GraphStates = weakref.WeakKeyDictionary()
        # The plugin's own reference store, one per agent and always present: standalone is the ordinary configuration,
        # and a ``ContextManager`` Stash is only the optional bridge behind it (Requirement 15.1). Created on first need
        # like the graph state, so construction still builds nothing.
        self._stores: _ReferenceStores = weakref.WeakKeyDictionary()

        # The delivery handler, built once: it holds no per-agent state, reading the graph out of the map above, so one
        # instance serves every agent ``init_agent`` registers it on (see ``Projection.register``).
        self._projection = Projection(
            self._states,
            description_tokens=self._description_tokens,
            # Read late, on every render: ``_tools`` is a plain list and a caller may de-register a retrieval tool
            # after construction -- which the relevance filter's two-store collision requires -- so a snapshot taken
            # here would advertise a tool the agent no longer has.
            retrieval_tools=lambda: {tool.tool_name for tool in self._tools},
        )

        # Always empty: the plugin registers what it needs in ``init_agent`` via ``agent.add_hook``, so the per-agent
        # handler count is verifiable by inspection rather than through discovery. Set before ``super().__init__()``,
        # whose guard then leaves it alone; the three retrieval tools are still auto-discovered.
        self._hooks = []
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        """Register the four engagement points on ``agent`` and settle the two wiring-time questions.

        Exactly one handler of each type, per agent: one ``InvokeModelStage.Input`` middleware handler and one hook each
        for ``MessageAddedEvent``, ``AfterToolCallEvent`` and ``BeforeInvocationEvent`` (Requirement 1.1). The three
        retrieval tools arrive through the plugin registry's own discovery, which finds the three ``@tool`` members and
        nothing else, so the tool registry gains exactly those (Requirement 1.2). Registering this same instance on a
        second agent adds one of each there and leaves the first agent's graph untouched, every piece of state being
        keyed by the agent (Requirement 1.5).

        Nothing about the agent is reconfigured: ``system_prompt``, ``messages`` and the tool registry come out as they
        went in, the retrieval tools aside (Requirement 1.3), and nothing is written to message metadata here or
        anywhere else (Requirement 1.10). The state is created eagerly, so a wired agent that has added no message
        presents a graph with no Card and a turn ordinal of ``0`` rather than no graph — and a fresh state is a full
        pass, so that agent is delivered to exactly as one without the plugin.

        The delivery goes in at index zero of ``InvokeModelStage``, which is the half of the ordering rule this plugin
        controls: the removal runs before any fold behind it in that list, so the memory manager folds into a message
        list from which nothing will later be discarded (Requirement 9.5). Ordering detection follows registration on
        purpose — the ordering it reports on is the one that now exists.

        Both notices degrade and never block: an agent that gets either is wired exactly like an agent that gets neither
        (Requirements 1.7, 16.7). Interference from a fold this plugin could not get ahead of stays this wiring-time
        caveat and is not something the runtime tries to recover from (Requirement 16.7 again, and 17.2).

        Args:
            agent: The agent to wire up.
        """
        self._warn_on_destructive_manager(agent)
        self._drop_artifact_tool_if_excluded()
        self._state_for(agent)
        agent.add_hook(self._on_before_invocation, BeforeInvocationEvent)
        agent.add_hook(self._on_message_added, MessageAddedEvent)
        agent.add_hook(self._on_after_tool_call, AfterToolCallEvent)
        delivery_is_first = self._projection.register(agent)
        self._warn_on_memory_fold_ordering(agent, delivery_is_first=delivery_is_first)

    def _drop_artifact_tool_if_excluded(self) -> None:
        """De-register ``expand_artifact`` when the caller excluded it, matched by name.

        The same shape :meth:`RelevanceFilter.init_agent` uses for its own retrieval tool, and here for the same reason:
        installed beside that filter there are two plausible tools for one job, each resolving a store the other cannot
        read, so the model reaches for whichever looks right and gets a miss it cannot act on. Measured on the benchmark
        this repository carries, dropping this one took the full stack from 84.5% to 94.4% weighted accuracy.

        Matched by ``tool_name`` rather than by a literal attribute so a rename upstream costs the de-registration
        rather than the run. Idempotent: a second agent finds the name already gone.

        The guidance text is unaffected by design -- it reads the registered set on every render, so a de-registered tool
        stops being advertised to the model without a second switch. Advertising a tool the registry does not hold is
        what produced ``tool not found in registry`` five times in one measured run.
        """
        if self._include_artifact_tool:
            return

        excluded = self.expand_artifact.tool_name
        self._tools = [tool for tool in self._tools if tool.tool_name != excluded]

    @property
    def retrieval_tool_names(self) -> tuple[str, ...]:
        """Names of the retrieval tools this instance registers, in registration order.

        Published because a caller cannot otherwise know them without hard-coding strings, and one caller in particular
        must: :class:`ProgressiveToolDisclosure` projects the call's tool list down to a catalog, and a tool reduced to a
        catalog entry carries an EMPTY ``inputSchema``. Every tool here needs arguments -- a Title, a reference, a search
        need -- so a hidden one is called with nothing, cancelled by the disclosure plugin's premature-call guard, and
        only then exposed. The model pays a round trip to learn what this plugin's own folded-context guidance already
        told it to do.

        So the retrieval tools belong in that plugin's ``always_available``, and deriving the list from here keeps it
        correct when ``include_artifact_tool`` is false or a tool is renamed::

            graph = ContextGraph(include_artifact_tool=False)
            disclosure = ProgressiveToolDisclosure(always_available=[*graph.retrieval_tool_names, "retrieve_context"])

        Read at call time rather than fixed at construction, so it reflects a de-registration that has already happened.
        Reading it BEFORE the plugin is wired to an agent reports the full set, since the exclusion is applied in
        :meth:`init_agent`; call it after wiring, or read ``include_artifact_tool``'s value, when the distinction
        matters.

        Returns:
            The tool names, which is three by default and two when the artifact tool was excluded.
        """
        return tuple(tool.tool_name for tool in self._tools)

    @staticmethod
    def _warn_on_destructive_manager(agent: Agent) -> None:
        """Emit the one precondition notice when the agent's conversation manager can remove messages.

        ``NullConversationManager`` is the precondition, not a suggestion: a sliding window drops the oldest prefix and
        a summarizer replaces spans with a summary, both on the live list and before the call is assembled, so either
        can physically remove a message this plugin only meant to fold — and raising that Card's Resolution back up then
        recovers nothing (Requirement 1.6). Under the null manager nothing is said (Requirement 1.8).

        One notice per wiring site, which the standard library's default filter gives for free, and nothing recorded on
        the instance, so a second agent wired from a second site is told too. An agent exposing no conversation manager
        at all reads as an agent with nothing that trims, the same quiet direction :func:`_wired_plugins` takes.

        Args:
            agent: The agent being wired.
        """
        manager = getattr(agent, "conversation_manager", None)
        if manager is None or isinstance(manager, NullConversationManager):
            return

        warnings.warn(_MANAGER_WARNING.format(manager=type(manager).__name__), stacklevel=_MANAGER_WARNING_STACKLEVEL)

    @staticmethod
    def _warn_on_memory_fold_ordering(agent: Agent, *, delivery_is_first: bool) -> None:
        """Emit the one wiring-time notice when a memory fold may end up ahead of the removal.

        Two conditions, both required: a memory manager that actually folds is wired to the agent, and the delivery
        cannot be shown to precede that fold. With no memory manager there is nothing to order against, and with the
        delivery provably first the ordering rule already holds, so both cases stay silent (Requirement 9.6).

        One notice per wiring site, which the standard library's default filter gives for free — no warn-once
        bookkeeping of our own, and nothing recorded on the instance, so a second agent wired from a second site is told
        too.

        Args:
            agent: The agent being wired, after the delivery has been registered.
            delivery_is_first: What the registration reported about the delivery's position in the stage.
        """
        if not any(_folds_memory(candidate) for candidate in _wired_plugins(agent)):
            return
        if _delivery_precedes_memory_fold(agent, delivery_is_first=delivery_is_first):
            return

        warnings.warn(_ORDERING_WARNING, stacklevel=_ORDERING_WARNING_STACKLEVEL)

    def _state_for(self, agent: Agent) -> _GraphState:
        """Return this agent's graph state, creating it on first use.

        Args:
            agent: The agent whose state to return.

        Returns:
            The state, weakly keyed by the agent so it is dropped along with it and never shared with another agent
            (Requirements 14.1, 14.4).
        """
        state = self._states.get(agent)
        if state is None:
            state = _GraphState()
            self._states[agent] = state
        return state

    def _store_for(self, agent: Agent) -> InMemoryReferenceStore:
        """Return this agent's reference store, creating it on first use.

        Args:
            agent: The agent whose store to return.

        Returns:
            The store, weakly keyed by the agent so it is dropped along with it. Always present, even on an agent where
            nothing ever offloads, in which case it simply stays empty (Requirements 15.1, 15.9).
        """
        store = self._stores.get(agent)
        if store is None:
            store = InMemoryReferenceStore()
            self._stores[agent] = store
        return store

    def _card_config(self) -> dict[str, Any]:
        """Return the values a Card is derived under, shared by the incremental step and by the rebuild scan.

        One source for both, which is what makes them comparable: the scan is only equal to the incremental graph if it
        ran under the same ceilings and the same link threshold (Requirement 14.5). ``rarity_weight`` belongs here even
        though it is not a construction parameter, since it decides which Tags a Card keeps.

        Returns:
            The keyword arguments both derivation entry points take.
        """
        return {
            "description_tokens": self._description_tokens,
            "tags_per_card": self._tags_per_card,
            "rarity_weight": _RARITY_WEIGHT,
            "link_threshold": self._link_threshold,
        }

    # ---- the write half: the turn boundary ----------------------------------------------------

    def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Close the Card of the turn that just ended, or derive the whole graph by scan.

        A turn is closed by what comes after it, so the boundary message this hook sees is what closes the turn before
        it: the Card of that turn is derived here, Title, Description, Tags and Links together, off the messages
        themselves. No language model, no disk, no network — the derivation reads text and tool names and nothing else
        (Requirement 3.10). Mid-turn messages close nothing and derive nothing.

        This is also the only place the rebuild scan may run (Requirement 14.6). A state holding no Card in front of a
        conversation that has at least one closed boundary is a process that inherited the history — a restore
        populates ``agent.messages`` directly and fires no event the incremental step could have seen — and one scan
        derives the whole graph from the messages, the same Cards, Links and Tags the incremental path would have built,
        with no serialized format to read and nothing remote to reach (Requirements 14.5, 14.8). Cards reference only
        messages present in ``agent.messages``, so a message that vanished is referenced by nothing and therefore takes
        part in no Resolution decision, without a raise (Requirement 14.7). The vector cache is per process and is left
        alone by the scan: a miss costs one embedding on the next turn the read half runs, never a Card.

        A late write half is a turn with no Card yet, which projects Full Content — the behavior without the plugin —
        and per-Card failures are absorbed by :func:`~.cards.derive_and_register`, which logs one ``warning`` carrying
        ``exc_info`` and registers nothing (Requirements 16.4, 16.5).

        Args:
            event: The message event, for the agent and the message just added.
        """
        agent = event.agent
        state = self._state_for(agent)
        messages = agent.messages

        try:
            closed = closed_turn_ranges(messages)
            if not closed:
                # Nothing has closed yet: no turn to card, and nothing to rebuild from.
                return

            if not state.cards:
                # The restore case, and equally the first closed boundary of a fresh agent, where the scan and the
                # incremental step derive the same single Card and the two paths' turn ordinals stay aligned from here.
                self._rebuild(agent, state)
                return

            if not is_turn_boundary(event.message):
                # Mid-turn: nothing closed, so nothing to derive.
                return

            start, stop = closed[-1]
            turn_ids = _identities_in(messages[start:stop])
            if not turn_ids:
                # A turn whose messages all lack a Durable Identity: messages without a Card, projected whole.
                return

            # The ordinal is the position of the boundary among the closed turns, not ``state.turn``, so a turn's Card
            # is the same whether it was derived turn by turn or by one scan (Requirement 14.3).
            derive_and_register(state, messages, turn_ids, len(closed) - 1, **self._card_config())
        except Exception:
            logger.warning("closing the turn's card failed | the turn's messages go whole", exc_info=True)

    def _rebuild(self, agent: Agent, state: _GraphState) -> None:
        """Derive the whole graph of ``agent``'s conversation onto ``state`` in one scan.

        Idempotent by way of ``rebuild_into``, which clears the Cards and Links before the scan, and which sets the turn
        ordinal from the count of closed boundaries so the scan and the incremental path agree on it. Failures do not
        propagate: a graph that did not come back holds no Card, and a message with no Card travels at Full Content
        (Requirements 16.4, 16.5).

        Args:
            agent: The agent whose conversation the graph is derived from. Read only.
            state: The graph state to fill. Left as it was when the scan failed.
        """
        try:
            rebuild_into(state, agent.messages, **self._card_config())
        except Exception:
            logger.warning("graph rebuild by scan failed | the whole conversation goes at full content", exc_info=True)

    # ---- the write half: the artifact side of one tool call -----------------------------------

    def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Register the artifact Cards of a tool result an offloader replaced, and record their references.

        Two writes, both of addresses only: an artifact Card keeping exclusively the reference, and the plugin's own
        store learning that the reference exists (Requirements 3.7, 15.1). The raw content is not kept anywhere here —
        the Card holds the address so nothing in the graph can rot when the underlying content changes — and no language
        model, disk or network is reached: the references are read off the placeholder text the result already carries.

        The fast path, not the only one: the rebuild scan reads the same references off the same preview text, which
        makes this hook's registration order against the offloader's irrelevant, costing at most a turn of latency.

        A result naming no reference registers nothing and logs nothing. That is the ordinary shape of a result no
        offloader replaced, and it is the whole of the nothing-offloaded path: no artifact Card, an empty store, the
        subject Cards untouched, and no branch asking whether an offloader is installed (Requirement 15.9). Failures do
        not propagate and the graph keeps no artifact Card (Requirements 16.4, 16.8).

        Args:
            event: The tool call event, for the agent, the result and the tool's name.
        """
        agent = event.agent
        state = self._state_for(agent)

        try:
            # Widened to ``object`` deliberately: the event annotates the field as a ``ToolResult``, so the guard below
            # reads as unreachable against the annotation while being what a failed tool call needs at runtime.
            raw: object = event.result
            if not isinstance(raw, dict):
                # A failed tool call carries an exception where a result would be: nothing to address.
                return

            cards = derive_and_register_artifacts(
                state,
                agent.messages,
                cast("ToolResult", raw),
                str(event.tool_use.get("name") or ""),
                state.turn,
                description_tokens=self._description_tokens,
                tags_per_card=self._tags_per_card,
                rarity_weight=_RARITY_WEIGHT,
            )
            # Names with nothing behind them, which is the most this site can state: the offloader has already replaced
            # the content by the time the hook sees the result, so there is no decoded block to pair and none is
            # invented. ``expand_artifact`` falls through to the optional Stash bridge for the content itself.
            record_references(
                self._store_for(agent),
                (card.reference for card in cards if card.reference),
            )
        except Exception:
            logger.warning("artifact card derivation failed | the graph keeps none", exc_info=True)

    # ---- the read half: one hook, on the critical path ----------------------------------------

    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Compute the Turn Choice, freeze it, and advance the turn ordinal.

        No language model call, no disk access, and ``agent.messages`` is not touched. The one remote call is the
        matcher's single embedding round, and the warm-up short circuit skips even that whenever the size of the graph
        already settles the answer (Requirements 10.4, 11.3).

        The choice is stored once here and read by every model call of the turn, the autonomous tool loop's included, so
        the context cannot shift mid-reasoning (Requirements 8.5, 8.6). It is frozen by type: ``TurnChoice.by_title`` is
        a ``MappingProxyType``, never a live dict.

        Args:
            event: The invocation event, for the agent and the turn's input messages.
        """
        state = self._state_for(event.agent)
        # Aged immediately before the choice reads it, and nowhere else: the choice is the only place the fed-back Note
        # is ever summed (Requirements 13.2, 13.4). Recomputing the choice here is also what ends a tool's Resolution
        # raise with the turn that asked for it (Requirement 12.14).
        expire_reuse(state, _cycle_of(event.agent))
        state.choice = self._compute_choice(state, event)
        # After the increment the ordinal names the turn now opening, whose boundary the write half is about to see. The
        # Card of the turn just before it does not exist yet — a turn is closed by what comes after it — so that turn
        # has no Card and its messages travel at Full Content (Requirement 16.5).
        state.turn += 1
        # Counted per turn, so it starts each turn at zero.
        state.retrieval_cycles = 0

    def _compute_choice(self, state: _GraphState, event: BeforeInvocationEvent) -> TurnChoice:
        """Score the graph against the turn's question, fill the vector index, and hand out the body budget.

        Any failure at any step degrades to the full pass — the behavior without the plugin — with exactly one warning
        carrying ``exc_info`` and no failure state kept, so the next turn computes a choice again (Requirement 16.1).

        Args:
            state: The graph state. Read, except for ``vectors``, which this fills from the embedding the scoring round
                already paid for.
            event: The invocation event, for the question and the agent.

        Returns:
            The frozen choice of this turn.
        """
        try:
            skipped = warm_up_choice(state, expand_threshold=self._expand_threshold, min_cards=self._min_cards)
            if skipped is not None:
                # Below ``min_cards`` the only possible decision is "send it all", so no embedding is paid for a
                # decision the size of the graph already made (Requirements 8.12, 11.3).
                return skipped

            question = _question_of(event.messages if event.messages is not None else event.agent.messages)
            matcher = self._matcher_for()
            # The one embedding round of the turn: the question as the query, the Descriptions as the documents
            # (Requirement 10.4).
            notes = compute_notes(state, question, matcher)
            if not notes:
                # The matcher failed or answered malformed, which reads as "score nothing, send everything" and not as
                # "nothing is relevant".
                return full_pass_choice()

            # Only on this path, and only after the scoring round: the vectors are already in the matcher's own cache
            # under the document purpose, so filling the index sends nothing. Above the ``min_cards`` short circuit it
            # would have sent a request for a decision that was never taken, which Requirement 11.3 forbids.
            self._cache_description_vectors(state, matcher)

            return distribute(
                notes,
                state,
                expand_threshold=self._expand_threshold,
                collapse_floor=self._collapse_floor,
                body_budget=self._body_budget,
            )
        except Exception:
            logger.warning("turn choice failed | the whole conversation goes at full content", exc_info=True)
            return full_pass_choice()

    def _cache_description_vectors(self, state: _GraphState, matcher: SimilarityMatcher) -> None:
        """Fill ``state.vectors`` from the Descriptions the scoring round just embedded.

        Without this the index stays empty, and an empty index is not a slow path but a missing feature: the similarity
        Link is measured from this index alone, by a hook Requirement 3.2 keeps free of network calls, so an unfilled
        index makes every pair unmeasurable and the ``similar`` Link never forms at all. The three structural Link kinds
        still form, which is why a graph with no ``similar`` edge looks like a working graph in the counters.

        Two properties keep it free. It runs only where :meth:`_compute_choice` has already scored, so the document
        vectors sit in the matcher's cache; and ``vectors`` is optional on the protocol, so a matcher that does not
        publish one leaves the index as it was rather than failing the turn.

        Stale Titles are dropped rather than left to accumulate: a Description that changed makes its entry unusable
        anyway — :func:`_cached_similarity` compares the cached text before trusting the vector — and a Title no longer
        in the graph will not be asked about again.

        Args:
            state: The graph state. ``vectors`` is replaced; nothing else is touched.
            matcher: The matcher that just scored, and therefore already holds these vectors.
        """
        published = getattr(matcher, "vectors", None)
        if published is None:
            return

        titles = titles_in_turn_order(state)
        descriptions = tuple(state.cards[title].description for title in titles)
        try:
            vectors = published(descriptions)
        except Exception:
            # Same posture as the scoring round: an unusable index costs Links, never the turn.
            logger.debug("graph description vectors unavailable for %d card(s)", len(titles), exc_info=True)
            return

        if len(vectors) != len(titles):
            # Includes the empty answer the matcher returns when the embedding was unavailable. A partial index would
            # pair vectors with the wrong Titles, which is worse than no index.
            return

        filled = {
            title: (description, tuple(vector))
            for title, description, vector in zip(titles, descriptions, vectors, strict=True)
        }
        # A Title whose entry is unchanged has already been measured against everything; one that is new, or whose
        # Description was rewritten, is what the second pass is for.
        newly_measurable = [title for title, entry in filled.items() if state.vectors.get(title) != entry]
        state.vectors = filled

        if newly_measurable:
            link_newly_measurable(state, newly_measurable, link_threshold=self._link_threshold)

    def _matcher_for(self) -> SimilarityMatcher:
        """Return the similarity matcher, building the default one on first need.

        Resolved here and never at construction, which opens no client and reaches no network (Requirements 2.16, 11.1).
        The resolved object is kept apart from the supplied one, so ``matcher=None`` stays observable as the
        configuration it was (Requirement 11.2).

        Returns:
            The supplied matcher, or the default asymmetric multilingual embedding matcher.
        """
        if self._matcher is not None:
            return self._matcher

        if self._resolved_matcher is None:
            from .matcher import EmbeddingSimilarityMatcher

            # Its Bedrock client is built lazily too, so this construction is still free of I/O (Requirement 11.4).
            self._resolved_matcher = EmbeddingSimilarityMatcher()

        return self._resolved_matcher

    # ---- the three retrieval tools: bodies in ``tools.py``, configuration and state here ------
    # Auto-discovered by the plugin registry, which registers exactly these three and no other
    # (Requirement 12.1): they are the only ``@tool`` members of the class.

    @tool(context=True)
    async def expand_card(self, titles: list[str], tool_context: ToolContext) -> str:
        """Bring back the full content of one or more earlier turns, by their titles.

        Earlier turns may reach you as a title and a short description instead of their messages. When
        a description is not enough to answer, call this with the titles exactly as they were shown and
        those turns arrive in full for the rest of this turn.

        Ask for every turn you need in ONE call: a list of titles costs one retrieval where the same
        titles one at a time cost one each, and each extra call grows the conversation you are about to
        reason over.

        Args:
            titles: Titles of the turns you want back, copied as they were shown to you.
            tool_context: Injected by the framework. Not user-facing.

        Returns:
            Confirmation that the turn will arrive in full, or an error naming the title asked for.
        """
        agent = tool_context.agent
        return tools.expand_card(
            self._state_for(agent),
            titles,
            cycle=_cycle_of(agent),
            reuse_ttl_cycles=self._reuse_ttl_cycles,
            max_retrieval_cycles=self._max_retrieval_cycles,
        )

    @tool(context=True)
    async def expand_artifact(
        self,
        reference: str,
        tool_context: ToolContext,
        line_range: dict[str, int] | None = None,
        pattern: str | None = None,
    ) -> str:
        """Read a stored artifact that an earlier turn of THIS conversation referred to by address.

        Use this for a reference that appeared in the conversation as a placeholder standing in for content
        too large to keep -- an image, a document, an export. The reference is the address that placeholder
        carried.

        If another tool told you it had replaced a tool result with a preview and handed you a reference,
        that reference belongs to that tool, not to this one: use the tool that minted it. This one resolves
        only addresses this plugin recorded, and answers by naming the miss when handed any other.

        Prefer a line range or a pattern: without either, the whole artifact comes back and costs its
        full token count again.

        Args:
            reference: The artifact reference, copied as it was shown to you.
            tool_context: Injected by the framework. Not user-facing.
            line_range: ``{"start": int, "end": int}`` to read only those lines.
            pattern: Return only the lines matching this pattern.

        Returns:
            The requested part of the artifact, or an error naming what was missing.
        """
        agent = tool_context.agent
        return await tools.expand_artifact(
            self._state_for(agent),
            self._store_for(agent),
            agent,
            reference,
            line_range,
            pattern,
            cycle=_cycle_of(agent),
            reuse_ttl_cycles=self._reuse_ttl_cycles,
            max_retrieval_cycles=self._max_retrieval_cycles,
        )

    @tool(context=True)
    async def find_context(self, need: str, tool_context: ToolContext, tag: str | None = None) -> str:
        """Find earlier turns of this conversation that match what you need, described in your words.

        Use this when you suspect the conversation already covered something but you cannot see it in
        what reached you. Describe the need, not a title.

        Args:
            need: What you are looking for, in your own words.
            tool_context: Injected by the framework. Not user-facing.
            tag: Restrict the search to turns carrying this tag.

        Returns:
            Up to five candidate turns with their title, tags and description, or an empty result
            naming the need received.
        """
        agent = tool_context.agent
        return tools.find_context(
            self._state_for(agent),
            need,
            tag,
            matcher=self._matcher_for(),
            collapse_floor=self._collapse_floor,
            cycle=_cycle_of(agent),
            reuse_ttl_cycles=self._reuse_ttl_cycles,
            max_retrieval_cycles=self._max_retrieval_cycles,
            neighbors_per_candidate=self._neighbors_per_candidate,
        )
