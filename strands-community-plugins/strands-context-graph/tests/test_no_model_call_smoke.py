"""Smoke test of the no-model-call contract.

Validates: Requirements 2.16, 2.19.

The claim is a negative one, and the only way to assert a negative is to make the forbidden call impossible to miss. Two
traps are laid for the duration of every test here:

- the agent's model records any ``stream`` / ``structured_output`` entry and then raises, so a language model call shows
  up both as a recorded entry and as a failure wherever it is swallowed;
- ``boto3`` and ``botocore.config`` are replaced in ``sys.modules`` by modules that record the attribute asked of them
  and then raise, so *building* an embedding client is caught rather than only calling one. Recording matters more than
  raising here: ``EmbeddingSimilarityMatcher`` turns any client-construction failure into an empty score sequence and a
  debug log, so an exception alone would leave the turn looking healthy. The recorded list is what the assertions read.

Against those traps the whole graph lifecycle is driven over a range of histories — turns with and without tool pairs,
offloaded results in both placeholder shapes, messages arriving without a ``tracking_id`` — through every engagement
point the plugin owns: the two write halves, the read half that scores the graph, the ``InvokeModelStage.Input``
delivery, all three retrieval tools, and the delivery a second time as the autonomous tool loop's next call would.

Requirement 2.16 is the construction half: building the plugin, and wiring it to an agent, reach no client, no network
and no async task, with the default matcher left unresolved. Requirement 2.19 is the operation half: none of the above
runs a language model call. The last test pins the exception the contract names — with the default matcher in place the
one thing that does reach for a client is the matcher's embedding, and even then no model call is made — so the other
tests' empty lists read as "nothing tried", not as "nothing was wired at all".
"""

import asyncio
import sys
import types
from collections.abc import Sequence
from typing import Any

import pytest
from strands import Agent
from strands._middleware.stages import InvokeModelContext
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks.events import AfterToolCallEvent, BeforeInvocationEvent, MessageAddedEvent
from strands.models.model import Model
from strands.types.tools import ToolContext

from strands_context_graph import ContextGraph

BODY_BUDGET = 40
"""Small enough that the Full Content resolutions run out partway down the graph, so the delivery both removes and
folds instead of passing the call straight through."""

TOOL_NAMES = ["run_query", "search", "fetch_report"]

TRAPPED_MODULES = ("boto3", "botocore.config")
"""The two imports ``EmbeddingSimilarityMatcher._ensure_client`` performs, and the only route to an AWS client in the
package: ``boto3`` is imported there and nowhere else at module scope."""


class _RecordingModel(Model):
    """A model that records the call and then refuses it.

    Both halves are needed. The raise fails any test that reaches a model on a path that lets the exception through; the
    record catches the paths that would swallow it.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def update_config(self, **kwargs):
        """Accept anything; nothing reads it."""

    def get_config(self):
        """No configuration to report."""
        return {}

    async def stream(self, *args, **kwargs):
        """Record and refuse: no hook, tool or middleware stage may reach this."""
        self.calls.append("stream")
        raise AssertionError("the language model was called")
        yield

    async def structured_output(self, *args, **kwargs):
        """Record and refuse: no hook, tool or middleware stage may reach this."""
        self.calls.append("structured_output")
        raise AssertionError("the language model was called")
        yield


class _TrapModule(types.ModuleType):
    """A stand-in module that records the attribute asked of it, then raises.

    ``import boto3`` finds this in ``sys.modules`` and succeeds; the first ``boto3.Session`` — or the
    ``from botocore.config import Config`` just before it — is what gets recorded.
    """

    def __init__(self, name: str, accesses: list[str]) -> None:
        super().__init__(name)
        self.accesses = accesses

    def __getattr__(self, attribute: str) -> Any:
        # ``accesses`` lives in the instance dict, so this is only ever reached for something the code under test asked
        # for. Dunder lookups from the import machinery arrive here too, and are as much of a reach for AWS as the rest.
        self.accesses.append(f"{self.__name__}.{attribute}")
        raise AssertionError(f"an AWS client was built: {self.__name__}.{attribute}")


class _CountingMatcher:
    """A deterministic matcher that records its calls: every other Card is relevant, the rest are not.

    Which Card it picks is arbitrary — the contract under test is about calls not made, not about ranking — but a split
    verdict is what makes the delivery remove and fold, and alternating by position guarantees the split on every graph
    of more than one Card. The call count is what keeps the assertions non-vacuous: the scoring path really ran.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description, in the order received."""
        self.calls.append(len(descriptions))
        return [0.9 if index % 2 == 0 else 0.1 for index, _ in enumerate(descriptions)]


@pytest.fixture
def aws_trap(monkeypatch):
    """Replace the AWS imports by recording traps for the duration of one test.

    Returns the list the traps append to: empty means nothing anywhere under test reached for a client.
    """
    accesses: list[str] = []
    for name in TRAPPED_MODULES:
        monkeypatch.setitem(sys.modules, name, _TrapModule(name, accesses))
    return accesses


def user(text: str, tracking_id: str | None) -> dict[str, Any]:
    """A plain user ask: a turn boundary, since it carries no tool result."""
    message: dict[str, Any] = {"role": "user", "content": [{"text": text}]}
    if tracking_id is not None:
        message["tracking_id"] = tracking_id
    return message


def assistant(text: str, tracking_id: str) -> dict[str, Any]:
    """An assistant answer, part of its turn's dialogue."""
    return {"role": "assistant", "content": [{"text": text}], "tracking_id": tracking_id}


def tool_use(tool_use_id: str, name: str, tracking_id: str) -> dict[str, Any]:
    """The assistant half of a tool pair."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": {}}}],
        "tracking_id": tracking_id,
    }


def tool_result(tool_use_id: str, text: str, tracking_id: str) -> dict[str, Any]:
    """The user half of a tool pair. Not a turn boundary: it carries a tool result."""
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": text}], "status": "success"}}],
        "tracking_id": tracking_id,
    }


def plain_turns() -> list[dict[str, Any]]:
    """Dialogue only: three closed turns carrying numbers, and the open turn's question."""
    return [
        user("what did the migration cost", "m0"),
        assistant("about 412 USD across 3 accounts", "m1"),
        user("and the quarter before", "m2"),
        assistant("318 USD", "m3"),
        user("which account dominated", "m4"),
        assistant("the archive account, at 1,204.50 USD", "m5"),
        user("summarize the trend", "m6"),
    ]


def offloaded_turns() -> list[dict[str, Any]]:
    """Tool pairs whose results an offloader replaced, in both placeholder shapes.

    The inline ``| ref: ...`` is what the offloader leaves behind; the standalone ``[refs: a, b]`` is the Stash's. Both
    are what give the ``AfterToolCallEvent`` half and ``expand_artifact`` something to work on, and neither may cost a
    call of any kind.
    """
    return [
        user("pull the cost report", "m0"),
        tool_use("tu-1", "fetch_report", "m1"),
        tool_result("tu-1", "[table: 4 rows, 900 bytes | ref: mem_1_tu-1_0]", "m2"),
        assistant("the archive account leads at 1,204.50 USD", "m3"),
        user("cross-check against the ledger", "m4"),
        tool_use("tu-2", "run_query", "m5"),
        tool_result("tu-2", "ledger extract\n[refs: mem_1_tu-2_0, mem_1_tu-2_1]", "m6"),
        assistant("the ledger agrees to within 2 USD", "m7"),
        user("anything else worth flagging", "m8"),
    ]


def untracked_turns() -> list[dict[str, Any]]:
    """Turns whose messages arrive without a ``tracking_id``: the derivation-lag shape, addressable by nothing."""
    return [
        user("what changed in the pipeline", None),
        assistant("two stages were merged", "m1"),
        user("show me the timings", None),
        tool_use("tu-9", "search", "m3"),
        tool_result("tu-9", "stage a 12.5 s, stage b 4.0 s", "m4"),
        assistant("stage a dominates at 12.5 s", None),
        user("is that acceptable", "m6"),
    ]


def one_open_turn() -> list[dict[str, Any]]:
    """The shortest history there is: a single question, no closed boundary, so no Card exists yet."""
    return [user("where do I start", "m0")]


HISTORIES = {
    "plain-turns": plain_turns,
    "offloaded-turns": offloaded_turns,
    "untracked-turns": untracked_turns,
    "one-open-turn": one_open_turn,
}
"""The range of histories every lifecycle claim here is driven over."""


def build(graph: ContextGraph | None = None) -> tuple[Agent, _RecordingModel]:
    """Build a real agent under the ``NullConversationManager`` precondition, with the recording model in place."""
    model = _RecordingModel()
    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant.",
        conversation_manager=NullConversationManager(),
        plugins=[graph] if graph is not None else [],
    )
    return agent, model


def context_over(agent: Agent) -> InvokeModelContext:
    """An ``InvokeModelContext`` whose ``messages`` is the live history itself, as the real stage hands it over."""
    return InvokeModelContext(
        agent=agent,
        messages=agent.messages,
        system_prompt="You are a helpful assistant.",
        tool_specs=[{"name": name, "description": "runs", "inputSchema": {}} for name in TOOL_NAMES],
        tool_choice=None,
        invocation_state={},
        model=agent.model,
        projected_input_tokens=1234,
        dynamic_trailing_blocks=0,
    )


def tool_context(agent: Agent) -> ToolContext:
    """The context a retrieval tool is called with, carrying the agent whose graph it reads."""
    return ToolContext(
        tool_use={"toolUseId": "tu-retrieval", "name": "expand_card", "input": {}},
        agent=agent,
        invocation_state={},
    )


def offloaded_results(messages: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Every tool result carrying a reference, as ``(tool_use_id, tool name, preview text)``."""
    names: dict[str, str] = {}
    for message in messages:
        for block in message.get("content") or ():
            use = block.get("toolUse") if isinstance(block, dict) else None
            if isinstance(use, dict):
                names[str(use.get("toolUseId"))] = str(use.get("name") or "")

    found: list[tuple[str, str, str]] = []
    for message in messages:
        for block in message.get("content") or ():
            result = block.get("toolResult") if isinstance(block, dict) else None
            if not isinstance(result, dict):
                continue
            text = "".join(part.get("text", "") for part in result.get("content") or ())
            if "ref" in text:
                tool_use_id = str(result.get("toolUseId"))
                found.append((tool_use_id, names.get(tool_use_id, "run_query"), text))
    return found


def first_question(messages: list[dict[str, Any]]) -> str:
    """The text of the first user ask, which is as good a ``need`` as any: the ranking is not under test here."""
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in message.get("content") or ():
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                return block["text"]
    return "anything"


def drive_lifecycle(graph: ContextGraph, agent: Agent, incoming: list[dict[str, Any]]) -> None:
    """Run every engagement point the plugin owns, in the order a real turn runs them.

    Card derivation first, message by message; then the artifact half for each offloaded result; then the read half that
    scores the graph and freezes the turn's choice; then the delivery, the three retrieval tools, and the delivery again
    as the autonomous tool loop's next model call would.
    """
    for message in incoming:
        agent.messages.append(message)
        graph._on_message_added(MessageAddedEvent(agent=agent, message=message))

    for tool_use_id, name, text in offloaded_results(agent.messages):
        graph._on_after_tool_call(
            AfterToolCallEvent(
                agent=agent,
                selected_tool=None,
                tool_use={"toolUseId": tool_use_id, "name": name, "input": {}},
                invocation_state={},
                result={"toolUseId": tool_use_id, "status": "success", "content": [{"text": text}]},
            )
        )

    graph._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=agent.messages))
    asyncio.run(graph._projection.deliver(context_over(agent)))

    state = graph._states[agent]
    context = tool_context(agent)
    titles = [title for title, card in state.cards.items() if card.kind == "subject"]
    asyncio.run(graph.expand_card(titles=[titles[0] if titles else "no such turn"], tool_context=context))
    asyncio.run(graph.find_context(need=first_question(agent.messages), tool_context=context))
    references = [card.reference for card in state.cards.values() if card.kind == "artifact" and card.reference]
    asyncio.run(
        graph.expand_artifact(reference=references[0] if references else "mem_1_absent_0", tool_context=context)
    )

    asyncio.run(graph._projection.deliver(context_over(agent)))


@pytest.mark.parametrize("history", list(HISTORIES), ids=list(HISTORIES))
def test_the_whole_graph_operation_runs_without_a_language_model_call(aws_trap, history):
    """Requirement 2.19: four handlers, two deliveries and three tools later, no model and no client were reached."""
    matcher = _CountingMatcher()
    graph = ContextGraph(matcher=matcher, min_cards=1, body_budget=BODY_BUDGET)
    agent, model = build(graph)

    drive_lifecycle(graph, agent, HISTORIES[history]())

    assert model.calls == []
    assert aws_trap == []
    # Non-vacuity: the supplied matcher is the graph's only scorer, and it was consulted, so the scoring path really ran
    # rather than short-circuiting before anything could have called out. One open turn alone has nothing to score.
    assert bool(matcher.calls) == bool(graph._states[agent].cards)


@pytest.mark.parametrize("history", list(HISTORIES), ids=list(HISTORIES))
def test_card_derivation_alone_runs_without_a_language_model_call(aws_trap, history):
    """Requirement 2.19 at the derivation seam, on both routes into it: incremental, and rebuild-by-scan.

    Card derivation is where a model call would be the natural design, so it is asserted on its own as well as inside
    the full lifecycle: Titles, Descriptions, Tags and Links all come out of a scan.
    """
    incoming = HISTORIES[history]()

    incremental = ContextGraph(matcher=_CountingMatcher(), min_cards=1)
    agent, model = build(incremental)
    for message in incoming:
        agent.messages.append(message)
        incremental._on_message_added(MessageAddedEvent(agent=agent, message=message))

    # The other route: a process that inherited the conversation and derives the whole graph by one scan.
    restored = ContextGraph(matcher=_CountingMatcher(), min_cards=1)
    inheritor, inheritor_model = build(restored)
    inheritor.messages.extend(incoming)
    restored._on_message_added(MessageAddedEvent(agent=inheritor, message=incoming[-1]))

    assert model.calls == []
    assert inheritor_model.calls == []
    assert aws_trap == []
    # Both routes derived the same Cards, and a history with a closed boundary derived some: the scan did the work.
    assert set(restored._states[inheritor].cards) == set(incremental._states[agent].cards)
    assert bool(incremental._states[agent].cards) == (len(incoming) > 1)


def test_construction_and_wiring_reach_no_client_no_network_and_no_async_task(aws_trap):
    """Requirement 2.16: with the default matcher, constructing and wiring build nothing and resolve nothing."""

    async def construct_and_wire() -> tuple[ContextGraph, Agent, _RecordingModel, set[asyncio.Task[Any]]]:
        # Inside a running loop, so a construction that scheduled background work would leave a task behind to find.
        before = asyncio.all_tasks()
        graph = ContextGraph()
        agent, model = build(graph)
        return graph, agent, model, asyncio.all_tasks() - before

    graph, agent, model, created = asyncio.run(construct_and_wire())

    assert created == set()
    assert aws_trap == []
    assert model.calls == []
    # ``matcher=None`` stays observable as the configuration it was, and the default is still unresolved: the client the
    # matcher would build is what construction is forbidden to reach for.
    assert graph._matcher is None
    assert graph._resolved_matcher is None
    # The wiring did happen, so the empty lists above are about what was not called and not about what was not set up.
    assert agent in graph._states
    assert {"expand_card", "expand_artifact", "find_context"} <= set(agent.tool_registry.registry)


def test_the_default_matchers_embedding_is_the_only_call_the_contract_allows(aws_trap):
    """Requirement 2.19's exception, pinned: the embedding reaches for a client, and no model call is made anywhere.

    Also the proof that the trap is armed. The other tests read an empty list as "nothing was tried"; this one shows the
    list fills as soon as something does try, and that the something is the matcher and nothing else.
    """
    graph = ContextGraph(min_cards=1)
    agent, model = build(graph)

    drive_lifecycle(graph, agent, plain_turns())

    assert [access for access in aws_trap if access.startswith(TRAPPED_MODULES)]
    assert model.calls == []
    # The client could not be built, so the matcher reported unavailability and the turn degraded to Full Content
    # everywhere instead of failing — an embedding is the contract's only remote call, never a required one.
    assert graph._resolved_matcher is not None
