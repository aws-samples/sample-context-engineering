"""The merge gate: the graph removes before a memory manager folds, observed on a running agent.

This is the package's single most fragile coupling, and the only one that fails as a *worse answer* rather than as an
exception. Both the graph's delivery and a memory manager's injection are ``InvokeModelStage.Input`` handlers, they run
in list order, and the graph must be first (Requirement 9.5): a memory block folded into a message list the removal
then filters is a block that can leave the call without anything saying so. Where that order cannot be shown to hold,
the plugin owes a wiring-time notice instead (Requirement 9.6). The two together are one obligation with two
discharges, and this file asserts the obligation rather than either branch — see
:func:`assert_ordering_obligation_discharged`.

``tests/test_plugin_ordering.py`` covers the detection: given a stub agent, which notice fires. This file covers the
outcome, and nothing here is stubbed on the path under test. The agent is a real ``Agent``, the memory manager is the
SDK's real ``MemoryManager`` wired through the real ``memory_manager=`` parameter, and the fold is therefore the SDK's
own injection middleware built by its own primitive. Only the two ends of the wire are local: the memory store is an
in-memory one returning a fixed entry, and the model is a scripted stub that records the payload it was handed. No
credentials, no network, no sleep.

What makes the assertion binding rather than incidental. Three observations, of which the first is the ordering itself
and the other two guard the failure shapes it does not cover:

1. **What the memory fold was handed**, which is the discriminating one. The manager's ``query`` callback receives the
   message list as it stood when the fold ran (:data:`FOLD_OBSERVATIONS`). If the graph went first, that list is already
   projected: the collapsed turn's answer is gone from it and the compaction's block is in it. If the fold went first,
   it sees the unprojected conversation. The two views differ in *both* directions, so neither can be satisfied by
   accident, and reversing the order was confirmed to flip both halves.
2. **What reached the provider.** On the same call the memory sentinel survived whole, riding the last user message next
   to the graph's own block, with ``dynamic_trailing_blocks == 2`` accounting for both. Worth being precise about what
   this does and does not catch: it is *not* on its own evidence of the order. Both folds target the last user message
   and the removal always keeps the turn in progress, so in this scenario the memory block reaches the provider under
   either order. What it catches is the adjacent failure — a removal that discarded the fold, or a delivery that rebuilt
   the message list without it — which is why observation 1 carries the ordering claim and this one is kept beside it.
3. **Where the handlers sit.** Two input handlers on the stage, the graph's delivery at index zero. Shape rather than
   proof, for the reason :func:`test_the_delivery_holds_index_zero_with_the_memory_fold_behind_it` gives, and it is what
   catches a delivery that stops claiming the front of the list.

The turn that carries all three has to be a turn where the graph *actually removed something*: a run in which nothing
collapses would satisfy every payload assertion trivially. :data:`COLLAPSED_ANSWER` is the sentinel that settles it, and
:func:`test_the_gate_run_actually_collapsed_a_turn` is the guard that refuses to let this file pass on a run where the
graph did nothing.
"""

import asyncio
import copy
import warnings
from collections.abc import AsyncGenerator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from strands import Agent
from strands._middleware.stages import InvokeModelStage
from strands.agent.conversation_manager import NullConversationManager
from strands.memory import MemoryManager
from strands.memory.types import MemoryEntry
from strands.models.model import Model
from strands.plugins.plugin import Plugin

from strands_context_graph import ContextGraph

# ---- the sentinels ------------------------------------------------------------------------------

MEMORY_FACT = "the ledger migration was signed off by finance in february"
"""What the memory store returns, and the sentinel for "the folded memory block survived".

Free of digits on purpose: a numeric line is copied into a Card's Description verbatim, so a numeric sentinel could
reach the provider inside the graph's own block and stop telling the two apart.
"""

WARM_UP_ASK = "open the ledger project with me"
"""Turn one. The removal always keeps the first user message, so this Card is never the collapsed one."""

MIGRATION_ASK = "the migration cost breakdown"
"""Turn two's ask, and therefore the Title of the Card this file collapses: a short message is its own Title."""

COLLAPSED_ANSWER = "we moved every account over to the new cluster overnight"
"""Turn two's answer: present in a payload means that turn travelled whole, absent means it was collapsed."""

FILLER_ASK = "anything else worth noting"
"""Turn three, which exists only so turn two's Card is closed by the time turn four computes its choice."""

GATE_ASK = "so what did that migration actually cost us"
"""Turn four: the turn whose choice collapses turn two, and the call the gate is asserted on."""

COLLAPSED_HEADER = "<collapsed_turns>"
"""The compaction's marker, which is how a payload is asked whether the graph folded a block into it."""

MEMORY_HEADER = "<memory>"
"""The SDK's default memory injection format, which is how a payload is asked whether the manager folded one."""

# ---- the tuning, borrowed from the end-to-end scenario for the same reasons -----------------------

EXPAND_THRESHOLD = 0.90
"""Far from the default, so one table score sits clearly above it and one clearly below."""

COLLAPSE_FLOOR = 0.20
"""The other side of the same wide interval."""

MIN_CARDS = 2
"""``BeforeInvocationEvent`` fires before the turn's message is appended and a turn is closed by the message that
follows it, so turn four's choice sees the Cards of turns one and two. Two is what reaches the scoring path at all."""

HIGH_SIMILARITY = 0.95
"""Above the ceiling: Full Content."""

LOW_SIMILARITY = 0.30
"""Inside the interval: Description, which is the collapse this file needs."""


# ---- the offline ends of the wire ----------------------------------------------------------------


class FixedMemoryStore:
    """A memory store of one entry, so the real ``MemoryManager`` can run with no backend.

    The attributes are the ``MemoryStore`` protocol's declarative half; ``search`` is the only method it requires.
    Read-only and without extraction, so the manager wires injection and nothing else.
    """

    name = "gate-store"
    description = "one fixed entry, for the ordering gate"
    max_search_results = 1
    writable = False
    extraction = False

    async def search(self, query: str, options: Any = None) -> list[MemoryEntry]:
        """Return the one entry, whatever was asked."""
        return [MemoryEntry(content=MEMORY_FACT)]


@dataclass(frozen=True)
class RecordedCall:
    """One provider call, as the model received it.

    Attributes:
        messages: The messages handed to the provider, deep-copied.
        system_prompt: The system prompt.
        dynamic_trailing_blocks: The SDK's count of per-call trailing blocks on the last message. ``None`` when the
            argument was not passed at all, which is how a call with no fold at all reads.
    """

    messages: list[dict[str, Any]]
    system_prompt: str | None
    dynamic_trailing_blocks: int | None

    @property
    def text(self) -> str:
        """Every piece of text in the payload as one string, for asking what reached the provider."""
        return "\n".join(
            block["text"]
            for message in self.messages
            for block in message.get("content", ())
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )

    @property
    def last_user_text(self) -> str:
        """The text of the payload's final message, which is where both folds land."""
        last = self.messages[-1]
        assert last["role"] == "user", "the payload's last message is the turn in progress"
        return "\n".join(
            block["text"] for block in last["content"] if isinstance(block, dict) and isinstance(block.get("text"), str)
        )


class ScriptedModel(Model):
    """A model that replays a script and records every payload, so no turn reaches a network.

    Args:
        script: One assistant text answer per provider call, in order.
    """

    def __init__(self, script: Sequence[str]) -> None:
        """Keep the script and start the recording."""
        self.script = list(script)
        self.calls: list[RecordedCall] = []

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Not part of this scenario."""
        raise AssertionError("structured output was requested")
        yield

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Record the payload, then replay the next scripted answer as a stream of events."""
        self.calls.append(
            RecordedCall(
                messages=copy.deepcopy(list(messages)),
                system_prompt=system_prompt,
                dynamic_trailing_blocks=kwargs.get("dynamic_trailing_blocks"),
            )
        )
        if len(self.calls) > len(self.script):
            raise AssertionError(f"the model was called {len(self.calls)} times for a script of {len(self.script)}")
        for event in _events_for(self.script[len(self.calls) - 1]):
            yield event


def _events_for(text: str) -> Iterator[dict[str, Any]]:
    """Render one scripted plain-text answer as the stream events the SDK parses."""
    yield {"messageStart": {"role": "assistant"}}
    yield {"contentBlockStart": {"start": {}}}
    yield {"contentBlockDelta": {"delta": {"text": text}}}
    yield {"contentBlockStop": {}}
    yield {"messageStop": {"stopReason": "end_turn"}}


@dataclass
class TableMatcher:
    """Fixed similarities: low for the Card this file collapses, high for every other.

    Keyed on the Description's text rather than on position, so the answer does not move when the graph gains a Card.

    Attributes:
        questions: The question of every scoring round, in order.
    """

    questions: list[str] = field(default_factory=list)

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one score per Description, in the order received."""
        self.questions.append(question)
        return [LOW_SIMILARITY if MIGRATION_ASK in text else HIGH_SIMILARITY for text in descriptions]


# ---- what the memory fold saw ---------------------------------------------------------------------

FOLD_OBSERVATIONS: list[list[dict[str, Any]]] = []
"""The message list handed to the memory fold, once per model call, in order.

This is observation (1) of the gate, and the only place the ordering is visible from the *other* side of it: the query
callback runs inside the SDK's injection middleware, on ``context.messages`` as that middleware received them. Filled by
:func:`record_and_query`, which the manager is configured with.
"""


def record_and_query(context: Any) -> str:
    """Record the messages the memory fold was handed, then return a constant query.

    A constant query keeps the store's answer a function of nothing, so the injected block is identical on every call
    and the payload assertions are about placement rather than about content.

    Args:
        context: The SDK's ``InjectionQueryContext``. Read only.

    Returns:
        The query the store is searched with.
    """
    FOLD_OBSERVATIONS.append(copy.deepcopy(list(context.messages)))
    return "migration"


def texts_of(messages: Sequence[dict[str, Any]]) -> str:
    """Every piece of text in ``messages``, as one string."""
    return "\n".join(
        block["text"]
        for message in messages
        for block in message.get("content", ())
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


# ---- the run ------------------------------------------------------------------------------------

ASKS = (WARM_UP_ASK, MIGRATION_ASK, FILLER_ASK, GATE_ASK)
"""The four user messages, in order. One provider call each: no tool is registered on the agent."""

SCRIPT = ("ready when you are", COLLAPSED_ANSWER, "nothing else for now", "it is in the turn you folded")
"""One scripted answer per call."""


def build_memory_manager() -> MemoryManager:
    """The SDK's real memory manager, folding on every call of the turn.

    ``trigger="everyTurn"`` matches the graph's own delivery trigger, so both handlers run on every call rather than
    only on the turn's first one — which is what puts them in the same stage on the same call, where ordering matters.
    """
    return MemoryManager(
        stores=[FixedMemoryStore()],
        search_tool_config=False,
        injection={"trigger": "everyTurn", "query": record_and_query},
    )


@dataclass(frozen=True)
class GateRun:
    """Everything the gate is asserted against.

    Attributes:
        graph: The plugin instance that was wired.
        agent: The agent the conversation ran on.
        calls: The provider payloads, in order.
        observations: The message lists the memory fold was handed, in order.
        warnings: The warnings the wiring produced.
    """

    graph: ContextGraph
    agent: Agent
    calls: list[RecordedCall]
    observations: list[list[dict[str, Any]]]
    warnings: list[warnings.WarningMessage]


def run_gate_conversation() -> GateRun:
    """Drive the four-turn conversation on a real agent wired to both the graph and a real memory manager.

    Driven through ``asyncio.run`` so every test below is an ordinary synchronous one reading recorded data: the
    conversation is over before anything is asserted, and nothing a test touches is tied to a loop.
    """
    FOLD_OBSERVATIONS.clear()
    model = ScriptedModel(SCRIPT)
    graph = ContextGraph(
        expand_threshold=EXPAND_THRESHOLD,
        collapse_floor=COLLAPSE_FLOOR,
        min_cards=MIN_CARDS,
        matcher=TableMatcher(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = Agent(
            model=model,
            conversation_manager=NullConversationManager(),
            plugins=[graph],
            memory_manager=build_memory_manager(),
        )

    async def conversation() -> None:
        for ask in ASKS:
            await agent.invoke_async(ask)

    asyncio.run(conversation())
    return GateRun(
        graph=graph,
        agent=agent,
        calls=model.calls,
        observations=list(FOLD_OBSERVATIONS),
        warnings=list(caught),
    )


@pytest.fixture(scope="module")
def gate() -> GateRun:
    """The gate run, shared by every assertion in this file: one conversation, read many ways."""
    return run_gate_conversation()


def input_handlers(agent: Agent) -> tuple[Any, ...]:
    """The call-assembly stage's handler list, which is where the claimed ordering is either true or not."""
    return tuple(agent._middleware_registry._handlers[InvokeModelStage])


# ---- the guard: this file is worthless on a run that collapsed nothing ----------------------------


def test_the_gate_run_actually_collapsed_a_turn(gate):
    """The precondition of every assertion below: the graph removed a real turn from the gate call.

    Without this, a delivery that quietly stopped projecting anything would satisfy the payload assertions by doing
    nothing at all, and the gate would guard nothing.
    """
    gate_call = gate.calls[-1]

    assert COLLAPSED_ANSWER not in gate_call.text, "nothing was removed, so this file proves nothing"
    assert COLLAPSED_HEADER in gate_call.text
    assert MIGRATION_ASK in gate_call.text, "the collapsed turn is named in the folded block"


def test_the_conversation_ran_as_scripted(gate):
    """One provider call per turn, and a fold observation for each: the two recordings line up call for call."""
    assert len(gate.calls) == len(ASKS)
    assert len(gate.observations) == len(ASKS)


# ---- observation 1: what the memory fold was handed ----------------------------------------------


def test_the_memory_fold_was_handed_the_already_projected_messages(gate):
    """Requirement 9.5, read from inside the memory manager: the fold ran on the graph's output.

    The strongest form of the ordering assertion available, because it is symmetric — had the fold run first it would
    have seen the collapsed turn's answer and no compaction block, which is the exact inverse of what is asserted here.
    """
    seen = texts_of(gate.observations[-1])

    assert COLLAPSED_ANSWER not in seen, "the memory fold saw the unprojected history: it ran before the removal"
    assert COLLAPSED_HEADER in seen, "the memory fold saw no compaction block: it ran before the graph's delivery"


def test_the_memory_fold_still_saw_the_turn_it_was_asked_about(gate):
    """The removal is per Card, not a truncation: the question being answered reached the fold intact."""
    seen = texts_of(gate.observations[-1])

    assert GATE_ASK in seen
    assert WARM_UP_ASK in seen


def test_the_early_turns_were_handed_the_unprojected_history(gate):
    """The same observation before the graph has a choice to make: below ``min_cards`` nothing is removed.

    Included so the assertion above is known to be discriminating rather than vacuously true of every call.
    """
    assert COLLAPSED_HEADER not in texts_of(gate.observations[0])


# ---- observation 2: what reached the provider ----------------------------------------------------


def test_the_folded_memory_block_survived_into_the_payload(gate):
    """The memory block reached the provider whole, alongside the graph's own projection of the same call.

    The outcome Requirement 9.6 warns about not getting. It is the adjacent guard rather than the ordering assertion
    itself — see the module docstring on why a surviving block is not by itself evidence of the order.
    """
    gate_call = gate.calls[-1]

    assert MEMORY_FACT in gate_call.text
    assert MEMORY_HEADER in gate_call.text


def test_both_blocks_ride_on_the_last_user_message(gate):
    """Both folds target the last user message, and the removal always keeps the turn in progress: both are there."""
    last_user_text = gate.calls[-1].last_user_text

    assert COLLAPSED_HEADER in last_user_text
    assert MEMORY_FACT in last_user_text


def test_the_trailing_block_count_accounts_for_both_folds(gate):
    """The SDK's own count, and the arithmetic a lost block would break: one block each, both counted."""
    assert gate.calls[-1].dynamic_trailing_blocks == 2


def test_the_memory_block_reached_the_provider_on_every_call(gate):
    """The graph never costs the memory manager a fold, whether or not the turn had anything to collapse."""
    assert all(MEMORY_FACT in call.text for call in gate.calls)


def test_nothing_the_graph_projected_entered_the_durable_history(gate):
    """The premise of the whole projection: neither fold was written back, and every turn is still readable."""
    history = str(gate.agent.messages)

    assert COLLAPSED_HEADER not in history
    assert MEMORY_FACT not in history
    assert COLLAPSED_ANSWER in history
    assert all(ask in history for ask in ASKS)


# ---- observation 3: where the handlers sit --------------------------------------------------------


def test_the_delivery_holds_index_zero_with_the_memory_fold_behind_it(gate):
    """The wiring the run depended on, asserted directly: two input handlers, the graph's first.

    Position, not identity: the registry wraps an input handler in a closure of its own before storing it, so nothing in
    the list can be compared back to the delivery. What makes index zero the graph's is the run above — the fold saw the
    graph's output — and what this asserts is that the shape which produced it is still the shape in the registry.
    """
    handlers = input_handlers(gate.agent)

    assert len(handlers) == 2
    assert all(getattr(handler, "phase", None) == "input" for handler in handlers)


# ---- the obligation, and its two discharges ------------------------------------------------------


def assert_ordering_obligation_discharged(agent: Agent, caught: Sequence[warnings.WarningMessage]) -> str:
    """Assert that exactly one of Requirement 9.5 and Requirement 9.6 was discharged, and return which.

    The obligation is a disjunction, so neither branch alone is the assertion: a wiring that can show the order is
    silent, a wiring that cannot must warn, and a wiring that does neither is the regression this gate exists to catch.
    Asserting the disjunction rather than a branch is what lets the same check run over both wirings below.

    Args:
        agent: The wired agent. Read only.
        caught: The warnings the wiring produced.

    Returns:
        ``"ordered"`` when the order is provable from the stage, ``"warned"`` when the notice fired instead.
    """
    handlers = input_handlers(agent)
    ordered = len(handlers) > 1 and all(getattr(handler, "phase", None) == "input" for handler in handlers)
    warned = any("InvokeModelStage" in str(warning.message) for warning in caught)

    assert ordered != warned, (
        "the ordering obligation was not discharged: "
        f"the stage {'does' if ordered else 'does not'} show a fold behind the delivery and the notice "
        f"{'fired' if warned else 'did not fire'}"
    )
    return "ordered" if ordered else "warned"


def test_the_real_wiring_discharges_the_obligation_by_ordering(gate):
    """Requirement 9.5 is the branch a real ``memory_manager=`` wiring takes, so no notice is owed."""
    assert assert_ordering_obligation_discharged(gate.agent, gate.warnings) == "ordered"


class UnobservableMemoryManager(Plugin):
    """A memory manager whose fold is not in the stage the graph can see, which is the case 9.6 exists for.

    Faithful where it counts: it carries the two members the detection reads — a memory manager is recognised by member,
    not by type — and it registers no ``InvokeModelStage.Input`` handler, which is precisely the shape the graph cannot
    order itself against. A real manager reaches this shape by folding from somewhere else entirely; the observable
    consequence is the same, and it is the notice rather than the payload.
    """

    name = "gate:unobservable-memory"

    def __init__(self) -> None:
        """Declare injection as enabled, so the detection reads a fold that really would happen."""
        super().__init__()
        self._injection_config: dict[str, Any] = {}

    async def _provide_memory_context(self, messages: Any, config: Any) -> str:
        """The render callback that makes this object a memory manager, by member."""
        return MEMORY_FACT


def test_an_unorderable_wiring_discharges_the_obligation_by_warning():
    """Requirement 9.6: where the order cannot be shown, the notice fires — on a real agent, at wiring time."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = Agent(
            model=ScriptedModel(()),
            conversation_manager=NullConversationManager(),
            plugins=[UnobservableMemoryManager(), ContextGraph()],
        )

    assert assert_ordering_obligation_discharged(agent, caught) == "warned"


def test_the_notice_names_the_requirement_and_the_risk():
    """What a developer is actually told: which order is required, and what a violation costs them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Agent(
            model=ScriptedModel(()),
            conversation_manager=NullConversationManager(),
            plugins=[UnobservableMemoryManager(), ContextGraph()],
        )

    message = next(str(warning.message) for warning in caught if "InvokeModelStage" in str(warning.message))
    assert "before" in message
    assert "projected out" in message
