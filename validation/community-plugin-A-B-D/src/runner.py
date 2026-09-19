"""Builds each configuration and replays the scenario against real Bedrock calls.

The sibling harness in ``validation/01-designA-B-D`` measures the same three strategies as vended
plugins of a forked SDK. This one measures the three **community packages** against an unmodified
``strands-agents``, and three differences follow from that -- none of them cosmetic:

**The baseline installs nothing.** The vended harness put its ``ContextOffloader`` in every
configuration, the baseline included, because without it the 60k-250k character payloads overflow
the window and the baseline fails rather than merely costing more. Here the baseline is the
unmodified agent, which is the honest control: it measures the cost of doing nothing, and a turn
that exceeds the model's context limit is recorded as an error and reported. That is the
measurement, not a defect.

**Each strategy brings its own remote client.** The vended stack shared one ``ContextManager``, so
one metered reranker covered everything. Here ``RelevanceFilter`` owns a ``Reranker`` and
``ContextGraph`` owns a ``SimilarityMatcher``, so metering is done by subclassing each package's
*public* class -- ``BedrockReranker`` and ``EmbeddingSimilarityMatcher`` -- rather than the SDK's
private ones.

**The referenced-source bridge is ours to build.** ``ProgressiveToolDisclosure`` accepts a
``referenced_source`` publicly, but the community ``ContextGraph`` publishes no accessor for the
tools a stepped-down Card still mentions (``_GraphState.referenced`` is declared and never
written). Without the bridge, a Card dropping to its Description takes its tools' ``inputSchema``
out of the call while still describing them, and the model calls a tool it no longer knows the
shape of -- recovered by the disclosure plugin's premature-call guard at the cost of one cycle.
:func:`_graph_referenced_source` closes that gap from the Cards themselves, and
``premature_cancellations`` is reported so the cost of the gap stays visible.

Every configuration runs with ``NullConversationManager``. For the graph that is a documented
precondition -- any other manager edits the live message list before the call is assembled, so it
can physically drop what the graph only meant to fold. For the comparison it also keeps history
handling from being a confounding variable between configurations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models import BedrockModel

# The three community packages. Imported at module scope, not lazily: a missing package is a broken
# installation and should fail at import with the package's own name, rather than midway through a
# paid run.
from strands_context_graph import ContextGraph, EmbeddingSimilarityMatcher
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure
from strands_relevance_filter import BedrockReranker, FileStore, RelevanceFilter

from . import accuracy, scenario, tools
from .config import (
    AGENT_MODEL_ID,
    ARTIFACTS_DIR,
    AWS_PROFILE,
    EMBED_MODEL_ID,
    GRAPH_ALONE,
    GRAPH_WITH_RELEVANCE,
    REGION,
    RERANK_MODEL_ID,
    RUN_CONFIGS,
    THRESHOLDS,
    RunConfig,
    estimate_tokens,
)
from .metrics import RunCollector, install_log_tagging

logger = logging.getLogger(__name__)


def boto_session() -> boto3.Session:
    """Return the session every client in a run is built from.

    Uses the configured profile when there is one, and the default credential chain otherwise. No
    credential is read from or written to disk by this harness either way: a profile backed by
    ``credential_process`` mints short-lived ones on demand, and the default chain resolves
    whatever the environment already provides.
    """
    if AWS_PROFILE:
        return boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    return boto3.Session(region_name=REGION)


def _client_config() -> BotocoreConfig:
    """Timeouts sized for parallel mode.

    The default read timeout expires on the heaviest turn when several configurations run
    concurrently: a turn that pulls a 60k-character document and then makes several retrieval
    calls is slow on its own, and all of them share one account's throughput. A lost turn scores as
    an accuracy failure and inflates the very variance the repeats exist to measure -- a harness
    artefact reported as a strategy result.

    Retries are adaptive so throttling is absorbed rather than surfacing as a lost turn.
    """
    return BotocoreConfig(
        read_timeout=180,
        connect_timeout=20,
        retries={"max_attempts": 5, "mode": "adaptive"},
    )


def _agent_model(session: boto3.Session) -> BedrockModel:
    """Build the agent's model, tagged so its invocations are separable in the log group."""
    # No temperature: Opus 4.8 rejects the parameter outright with a ValidationException
    # ("`temperature` is deprecated for this model"). Determinism across configurations therefore
    # comes from a fixed scenario and fixed payloads, not from a sampling knob.
    model = BedrockModel(
        boto_session=session,
        boto_client_config=_client_config(),
        model_id=AGENT_MODEL_ID,
        max_tokens=4_096,
    )
    install_log_tagging(model.client, role="agent")
    return model


class _MeteredReranker(BedrockReranker):
    """Counts the relevance filter's rerank traffic in the unit Bedrock bills.

    Subclasses the package's **public** ``BedrockReranker``, which is what makes this harness
    measurable without reaching into the package's internals: ``score`` is part of the ``Reranker``
    protocol, so the count is taken at the contract boundary.

    A search unit is one query against up to 100 documents, which is exactly the batch the reranker
    pages into, so the count is ``ceil(chunks / max_sources_per_query)`` per call. Counted before
    the call delegates: a failure halfway through still consumed what it sent.

    Note that ``RelevancePreview`` keeps its own ``search_units`` tally for the same traffic. Both
    are reported -- they should agree, and a disagreement means one of the two is counting
    something the other is not.

    Attributes:
        rerank_calls: Calls to ``score``, one per filtered tool result.
        search_units: Billable batches submitted.
        rerank_documents: Documents submitted across those batches.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the reranker and zero the counters."""
        super().__init__(*args, **kwargs)
        self.rerank_calls = 0
        self.search_units = 0
        self.rerank_documents = 0

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        """Count the batch, then delegate to the package's own scoring."""
        if chunks:
            self.rerank_calls += 1
            self.rerank_documents += len(chunks)
            # ``max(1, ...)`` guards a subclass that zeroed the attribute; the division is the
            # package's own batching rule and must not raise inside a measurement.
            self.search_units += -(-len(chunks) // max(1, self.max_sources_per_query))
        return await super().score(query, chunks)


class _MeteredMatcher(EmbeddingSimilarityMatcher):
    """Counts what the graph's similarity matcher actually sends to Bedrock.

    Overrides ``_invoke`` rather than ``score``, because that is what makes the count *billable*
    instead of merely indicative: ``score`` is called with every Description on every turn, while
    ``_invoke`` receives only the cache misses, deduplicated. Metering the outer method would bill
    a session for vectors it never paid for -- precisely the claim the cache exists to make.

    Tokens are estimated from characters, not read from the response: Cohere's embed API returns no
    usage block. The estimate uses the same divisor as the rest of the harness, so the embedding
    column is on the same footing as every other token figure here.

    Attributes:
        embed_calls: Bedrock round trips made, one per batch of at most 96 texts.
        embed_texts: Texts that reached Bedrock, after cache and deduplication.
        embed_tokens: Estimated input tokens of those texts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the matcher and zero the counters."""
        super().__init__(*args, **kwargs)
        self.embed_calls = 0
        self.embed_texts = 0
        self.embed_tokens = 0

    def _invoke(self, texts: list[str], purpose: str) -> list[list[float]]:
        """Count the batch that missed the cache, then delegate."""
        # Counted before the call: a failure halfway through still consumed what it sent.
        self.embed_calls += 1
        self.embed_texts += len(texts)
        self.embed_tokens += sum(estimate_tokens(text) for text in texts)
        return super()._invoke(texts, purpose)  # type: ignore[arg-type]


_FULL = "full"
"""The resolution name meaning a Card's part entered the call whole."""

_GRAPH_ARTIFACT_TOOL = "expand_artifact"
"""The graph's artifact-retrieval tool, dropped when the relevance filter is also installed.

Measured, not assumed. On the first 60-turn run the ``all`` configuration was the only one that
could not answer A5 -- the turn whose answer is one row of a 90-character-per-line statement -- and
the model said why in its own answer: "every export's artifact reference has come back unreachable
... I can't read the stored artifacts". It had called ``expand_artifact`` with a reference the
relevance filter had minted.

The two packages each ship their own retrieval tool over their own store, and nothing bridges them:
``RelevanceFilter`` stores the raw sub-blocks it replaced and hands out references its
``retrieve_context`` resolves, while ``ContextGraph`` records addresses it saw in placeholder text
and resolves them through a store of its own. The graph's README is explicit that its bridge to
another plugin's stash is built entirely on private symbols and degrades to "answers as prose naming
the miss" -- which is exactly what happened, and the vended stack never hit it because relevance
lived *inside* the ContextManager whose stash the graph bridged to.

So when both are installed there are two plausible tools for one job and only one of them can
resolve the reference. Dropping the graph's leaves exactly one artifact path. Its two other tools --
``expand_card`` and ``find_context`` -- are untouched: they reach back into the conversation's own
turns, which is a different job and one the relevance filter does not do.
"""


def _drop_graph_artifact_tool(graph: ContextGraph) -> bool:
    """Remove the graph's artifact-retrieval tool from the set it registers.

    The same de-registration ``RelevanceFilter.init_agent`` performs on its own retrieval tool when
    ``include_retrieval_tool`` is false, applied from the outside because ``ContextGraph`` exposes no
    equivalent switch. Matched by ``tool_name`` rather than by a literal attribute so a rename
    upstream costs the de-registration, not the run.

    Args:
        graph: The plugin instance, before it is handed to an agent.

    Returns:
        Whether the tool was found and removed.
    """
    try:
        before = len(graph._tools)
        graph._tools = [t for t in graph._tools if t.tool_name != _GRAPH_ARTIFACT_TOOL]
        return len(graph._tools) < before
    except Exception:  # noqa: BLE001 - losing the de-registration must not lose the run
        logger.warning("could not drop the graph's %s tool | both retrieval paths stay", _GRAPH_ARTIFACT_TOOL)
        return False


def _graph_referenced_source(graph: ContextGraph) -> Any:
    """Return a ``referenced_source`` callable naming the tools of every stepped-down Card.

    The gap this closes is a real one and belongs to the community packaging, not to the design:
    ``ProgressiveToolDisclosure`` takes a ``referenced_source`` publicly, and ``ContextGraph``
    publishes nothing to feed it -- ``_GraphState.referenced`` is declared in the dataclass and
    never written by any code path in the package.

    Without it, a Card that drops to its Description keeps *describing* its tools while their
    ``inputSchema`` leaves the call, because the disclosure plugin's own referenced block is
    computed from the ``toolUse`` blocks of the projected messages, and the graph has already
    removed those. The model then calls a tool whose shape it cannot see. That is recoverable --
    the premature-call guard cancels the call, exposes the schema and asks for a retry -- but it
    costs a cycle every time, and the cycle is the model's, not the harness's.

    Reads the plugin's per-agent state, which is private. That is the same coupling
    :func:`_collect_plugin_counters` already accepts, and it degrades the same way: any failure
    returns no names, which is exactly the behaviour without the bridge.

    Args:
        graph: The graph plugin instance wired to the agents this source will serve.

    Returns:
        A callable taking the agent of the call and returning tool names.
    """

    def source(agent: Agent) -> tuple[str, ...]:
        try:
            state = graph._states.get(agent)
            if state is None:
                return ()

            choice = state.choice
            if choice.full_pass:
                # Every Card is whole, so every toolUse is still in the call and the disclosure
                # plugin's own referenced block already covers them.
                return ()

            selected = choice.selected
            names: set[str] = set()
            for title, card in state.cards.items():
                addressed = selected is None or title in selected
                card_choice = choice.by_title.get(title)
                stepped_down = not addressed or (
                    card_choice is not None
                    and (card_choice.dialogue != _FULL or card_choice.evidence != _FULL)
                )
                if stepped_down:
                    names.update(card.tool_names)
            return tuple(sorted(names))
        except Exception:  # noqa: BLE001 - a bridge failure must cost names, never the run
            logger.debug("graph referenced-source bridge failed | composing from history alone", exc_info=True)
            return ()

    return source


def build_plugins(config: RunConfig, session: boto3.Session) -> list[Any]:
    """Assemble the plugin list for one configuration.

    Returns an empty list for the baseline, which is the point of the baseline: no plugin, no
    projection, no filtering, and the payloads enter the history whole.

    Args:
        config: The configuration being built. Metered clients are stashed on ``config.extra`` so
            the end-of-run counters can read them.
        session: The run's boto3 session, so every remote call a strategy makes is billed to the
            same account as the agent's own and never falls back on ambient credentials.

    Returns:
        The plugins to hand to the agent, in construction order. Handler order is not decided here:
        the graph moves its own delivery to index zero of the stage.
    """
    plugins: list[Any] = []

    if config.relevance:
        storage_root = ARTIFACTS_DIR / config.name
        storage_root.mkdir(parents=True, exist_ok=True)

        reranker = _MeteredReranker(
            model_id=RERANK_MODEL_ID,
            boto_session=session,
            boto_client_config=_client_config(),
        )
        config.extra["_reranker"] = reranker

        relevance = RelevanceFilter(
            # File-backed rather than in-memory: the payloads are large, the run is long, and a
            # reference the model reads back hours into a run must still resolve.
            store=FileStore(str(storage_root)),
            max_result_tokens=THRESHOLDS.max_result_tokens,
            config={
                "reranker": reranker,
                "relevance_threshold": THRESHOLDS.relevance_threshold,
                "chunk_tokens": THRESHOLDS.chunk_tokens,
                "preview_tokens": THRESHOLDS.preview_tokens,
            },
        )
        config.extra["_relevance"] = relevance
        plugins.append(relevance)

    # Built before the disclosure plugin, because the disclosure plugin needs the bridge as its
    # supplemental referenced source.
    graph: ContextGraph | None = None
    if config.graph:
        matcher = _MeteredMatcher(EMBED_MODEL_ID, boto_session=session)
        config.extra["_matcher"] = matcher

        # Which tuning applies is a measured result, not a preference: with the relevance filter
        # installed the payload is already a preview by the time the Card is derived, so folding
        # harder costs recall without buying tokens. See the two docstrings in config.
        tuning = GRAPH_WITH_RELEVANCE if config.relevance else GRAPH_ALONE
        config.extra["_graph_tuning"] = (
            "with-relevance" if config.relevance else "alone"
        )

        graph = ContextGraph(
            expand_threshold=tuning.expand_threshold,
            collapse_floor=tuning.collapse_floor,
            link_threshold=tuning.link_threshold,
            description_tokens=tuning.description_tokens,
            body_budget=tuning.body_budget,
            min_cards=THRESHOLDS.min_cards,
            matcher=matcher,
        )
        if config.relevance:
            # Two retrieval tools for one job, over two stores that do not know each other, is what
            # cost the first run its A5 answer. See _GRAPH_ARTIFACT_TOOL for the measurement.
            config.extra["_graph_artifact_tool_dropped"] = _drop_graph_artifact_tool(graph)
        config.extra["_graph"] = graph
        plugins.append(graph)

    if config.disclosure:
        plugins.append(
            ProgressiveToolDisclosure(
                catalog_tokens=THRESHOLDS.catalog_tokens,
                ttl_cycles=THRESHOLDS.ttl_cycles,
                top_k=THRESHOLDS.top_k,
                # A retrieval tool must never need discovery: the model is told to use it in the
                # guidance text that replaces the payload, and a cycle spent finding it would be an
                # artefact of the harness rather than of the strategy. All four are listed because
                # which ones exist depends on the configuration, and a name absent from the call is
                # simply ignored by the plugin.
                always_available=[
                    "retrieve_context",
                    "expand_card",
                    "expand_artifact",
                    "find_context",
                    "list_accounts",
                ],
                referenced_source=_graph_referenced_source(graph) if graph is not None else None,
            )
        )

    return plugins


def build_agent(config: RunConfig, session: boto3.Session, collector: RunCollector) -> Agent:
    """Construct the agent under test and attach the measurement middleware.

    Args:
        config: The configuration under test.
        session: The run's boto3 session.
        collector: Receiver of the per-call measurements.

    Returns:
        The agent, with the collector's middleware registered last in the invoke stage.
    """
    from strands._middleware.stages import InvokeModelStage

    agent = Agent(
        model=_agent_model(session),
        tools=tools.all_tools(),
        system_prompt=scenario.SYSTEM_PROMPT,
        # The graph's documented precondition, and the thing that keeps history handling from being
        # a confounding variable between configurations.
        conversation_manager=NullConversationManager(),
        plugins=build_plugins(config, session),
        callback_handler=None,
    )

    # Registered after construction, so it runs last in the stage and observes the projection the
    # plugins produced rather than the pre-plugin baseline.
    agent._middleware_registry.add_middleware(InvokeModelStage, collector.middleware())

    return agent


def _installed_plugins(agent: Agent) -> list[Any]:
    """Return the plugins registered on ``agent``.

    The registry keeps them in a name-keyed dict on a private attribute; reading it defensively
    means a rename upstream costs us the counters, not the run.
    """
    try:
        return list(agent._plugin_registry._plugins.values())
    except Exception:  # noqa: BLE001
        return []


def _collect_plugin_counters(agent: Agent, config: RunConfig) -> dict[str, Any]:
    """Read the strategy-specific evidence off the plugins after a run.

    These counters are the difference between "tokens went down" and "the strategy did what it
    claims". Every read is defensive: they live on private attributes and a missing one should
    blank a field, not fail a run.

    The key names are deliberately the ones the sibling harness emits -- ``disclosure``,
    ``offloader``, ``graph``, ``embedding_cost``, ``rerank_cost`` -- so the whole reporting path
    (``metrics``, ``compare``, ``report``, ``chart``) is reused unchanged and a run from either
    harness renders through the same cost model. ``offloader`` in particular keeps its name even
    though no offloader exists here: it is the key the cost model sums rerank spend from, and a new
    name would price relevance filtering at zero.

    Args:
        agent: The agent whose plugins are read.
        config: The configuration, carrying the metered clients on ``extra``.

    Returns:
        The counters, one section per strategy that was installed.
    """
    counters: dict[str, Any] = {}

    for plugin in _installed_plugins(agent):
        name = type(plugin).__name__

        if name == "ProgressiveToolDisclosure":
            try:
                state = plugin._states[agent]
                counters["disclosure"] = {
                    "searches": state.searches,
                    # The cost of the missing referenced-source accessor, when the bridge is off or
                    # could not answer: each cancellation is one model cycle spent re-learning a
                    # schema the call had already dropped.
                    "premature_cancellations": state.premature_cancellations,
                    "exposed_at_end": sorted(state.exposed),
                    "exposed_count_at_end": len(state.exposed),
                }
            except Exception as error:  # noqa: BLE001
                counters["disclosure"] = {"error": f"state unavailable: {error}"}

        elif name == "RelevanceFilter":
            # Named ``offloader`` on purpose: see the docstring. The preview builder is created on
            # the first filtered result, so ``None`` here means the filter never fired -- which is
            # itself a finding, not a missing measurement.
            try:
                preview = getattr(plugin, "_preview", None)
                counters["offloader"] = {
                    "preview_strategy": "relevance",
                    "search_units": getattr(preview, "search_units", 0) or 0,
                    "fired": preview is not None,
                }
            except Exception as error:  # noqa: BLE001
                counters["offloader"] = {"error": f"counters unavailable: {error}"}

        elif name == "ContextGraph":
            # The graph's evidence that it decided rather than merely ran. Without these, "tokens
            # went down" and "the ladder collapsed to one rung" read identically in the report.
            try:
                state = plugin._states[agent]
                choice = state.choice
                dialogue = {"full": 0, "description": 0, "title": 0}
                # ``title`` on the evidence axis is not a rung of the ladder: it means the
                # selection did not address the Card, so nothing of it reached the call.
                evidence = {"full": 0, "description": 0, "title": 0}
                for title in state.cards:
                    card_choice = None if choice.full_pass else choice.by_title.get(title)
                    dialogue["full" if card_choice is None else card_choice.dialogue] += 1
                    evidence["full" if card_choice is None else card_choice.evidence] += 1
                selected = choice.selected
                counters["graph"] = {
                    "addressed": len(state.cards) if selected is None else len(selected),
                    "unaddressed": 0 if selected is None else len(state.cards) - len(selected),
                    "cards": len(state.cards),
                    "subject_cards": sum(1 for card in state.cards.values() if card.kind == "subject"),
                    "artifact_cards": sum(1 for card in state.cards.values() if card.kind == "artifact"),
                    "links": sum(len(edges) for edges in state.links.values()),
                    "turn": state.turn,
                    "full_pass_at_end": choice.full_pass,
                    "dialogue_at_end": dialogue,
                    "evidence_at_end": evidence,
                    "reuse_at_end": len(state.reuse),
                    "vectors_cached": len(state.vectors),
                    # The community plugin emits no per-turn telemetry, so the ladder curves the
                    # sibling harness charts do not exist here. Stated rather than omitted: an
                    # absent field reads as a harness that forgot to measure.
                    "per_turn_telemetry": "unavailable (the community package logs failures only)",
                }
            except Exception as error:  # noqa: BLE001
                counters["graph"] = {"error": f"state unavailable: {error}"}

    matcher = config.extra.get("_matcher")
    if matcher is not None:
        counters["embedding_cost"] = {
            "calls": matcher.embed_calls,
            "texts": matcher.embed_texts,
            "input_tokens": matcher.embed_tokens,
            "model_id": matcher.model_id,
        }

    reranker = config.extra.get("_reranker")
    if reranker is not None:
        # Reported alongside the filter's own tally above. They should agree; a disagreement means
        # one of the two is counting traffic the other is not, which is worth knowing before the
        # cost column is trusted.
        counters["rerank_observed"] = {
            "calls": reranker.rerank_calls,
            "search_units": reranker.search_units,
            "documents": reranker.rerank_documents,
            "model_id": RERANK_MODEL_ID,
        }

    counters["live_messages_at_end"] = len(agent.messages)
    counters["registered_tools"] = len(agent.tool_names)
    if "_graph_tuning" in config.extra:
        # Which of the two measured tunings this arm ran under. Without it, two graph runs are
        # indistinguishable in the record even though they folded differently.
        counters["graph_tuning"] = config.extra["_graph_tuning"]
    if "_graph_artifact_tool_dropped" in config.extra:
        # Recorded because it changes what the model could reach, and a reader comparing two runs
        # has to be able to see which one had one artifact path and which had two.
        counters["graph_artifact_tool_dropped"] = config.extra["_graph_artifact_tool_dropped"]
    return counters


async def run_configuration(
    config: RunConfig,
    *,
    turn_limit: int | None = None,
    phases: tuple[str, ...] | None = None,
    total_turns: int | None = None,
) -> RunCollector:
    """Replay the scenario under one configuration and return its measurements.

    Args:
        config: The configuration to run.
        turn_limit: Run only the first N turns.
        phases: Restrict to these scenario phases.
        total_turns: Pad the script with unscored filler up to this many turns.

    Returns:
        The collector holding every measurement of the run.
    """
    session = boto_session()
    collector = RunCollector(config.name)
    agent = build_agent(config, session, collector)

    script = scenario.turns(limit=turn_limit, phases=phases, total=total_turns)
    logger.info("=== %s: %d turns, %d tools ===", config.name, len(script), len(agent.tool_names))

    run_started = time.perf_counter()

    for index, turn in enumerate(script):
        record = collector.begin_turn(index, turn.label, turn.prompt)
        turn_started = time.perf_counter()
        try:
            result = await agent.invoke_async(turn.prompt)
            record.response_text = str(result)
            record.response_chars = len(record.response_text)
        except Exception as error:  # noqa: BLE001 - one bad turn must not lose the run
            # The baseline is expected to arrive here once the history outgrows the model's context
            # window. That is the measurement: a configuration whose turns stop completing has
            # answered the question the comparison asks.
            record.error = f"{type(error).__name__}: {error}"
            collector.errors.append(f"{turn.label}: {record.error}")
            logger.warning("turn %s failed: %s", turn.label, record.error)
        finally:
            record.turn_seconds = time.perf_counter() - turn_started
            record.live_message_count = len(agent.messages)
            record.tool_calls = _tool_calls_in_last_turn(agent)

        logger.info(
            "%-10s %-18s %6.1fs  msgs=%-4d calls=%d",
            config.name,
            turn.label,
            record.turn_seconds,
            record.live_message_count,
            len(collector.calls),
        )

    collector.wall_seconds = time.perf_counter() - run_started
    collector.plugin_counters = _collect_plugin_counters(agent, config)

    # Scored here rather than in the report so the accuracy figure travels with the raw results and
    # can be re-examined without a re-run.
    collector.accuracy = accuracy.score_run([turn.to_dict() for turn in collector.turns])
    logger.info(
        "%-10s accuracy: %.1f%% weighted, %d/%d turns materially correct, %d errors",
        config.name,
        collector.accuracy["weighted_accuracy"] * 100,
        collector.accuracy["turns_materially_correct"],
        collector.accuracy["turns_scored"],
        len(collector.errors),
    )

    return collector


def _tool_calls_in_last_turn(agent: Agent) -> list[str]:
    """Names of the tools invoked in the most recent turn."""
    names: list[str] = []
    for message in reversed(agent.messages):
        if message.get("role") == "user" and not any(
            "toolResult" in block for block in message.get("content", []) if isinstance(block, dict)
        ):
            break
        for block in message.get("content", []):
            if isinstance(block, dict) and "toolUse" in block:
                names.append(block["toolUse"].get("name", "?"))
    return list(reversed(names))


async def run_all(
    names: list[str],
    *,
    turn_limit: int | None = None,
    phases: tuple[str, ...] | None = None,
    repeats: int = 1,
    parallel: bool = True,
    total_turns: int | None = None,
) -> dict[str, list[RunCollector]]:
    """Run the named configurations ``repeats`` times each.

    Parallel by default. The configurations are independent agents with independent histories, so
    running them concurrently changes nothing about the tokens they assemble or the answers they
    give -- and it takes the wall time of a five-configuration comparison down to the length of its
    slowest member.

    The cost is that latency figures become throughput-under-contention rather than isolated
    latency. Token and accuracy numbers are unaffected, which is why parallel is the sensible
    default and ``--sequential`` exists for when the timing columns are the point.

    Repeats are interleaved rather than grouped, so drift in service latency over a long session
    spreads across configurations instead of landing on whichever ran last.

    Args:
        names: Configuration names to run.
        turn_limit: Run only the first N turns.
        phases: Restrict to these scenario phases.
        repeats: Replay each configuration this many times.
        parallel: Run the configurations concurrently.
        total_turns: Pad the script with unscored filler up to this many turns.

    Returns:
        One list of collectors per configuration name, in replay order.
    """
    results: dict[str, list[RunCollector]] = {name: [] for name in names}

    for repeat in range(repeats):
        if repeats > 1:
            logger.info("--- repeat %d/%d ---", repeat + 1, repeats)

        if parallel:
            collectors = await asyncio.gather(
                *(
                    run_configuration(
                        RUN_CONFIGS[name],
                        turn_limit=turn_limit,
                        phases=phases,
                        total_turns=total_turns,
                    )
                    for name in names
                ),
                return_exceptions=True,
            )
            for name, collector in zip(names, collectors):
                if isinstance(collector, BaseException):
                    logger.error("configuration %s raised", name, exc_info=collector)
                    failed = RunCollector(name)
                    failed.errors.append(f"{type(collector).__name__}: {collector}")
                    results[name].append(failed)
                else:
                    results[name].append(collector)
        else:
            for name in names:
                collector = await run_configuration(
                    RUN_CONFIGS[name],
                    turn_limit=turn_limit,
                    phases=phases,
                    total_turns=total_turns,
                )
                results[name].append(collector)
                await asyncio.sleep(2)  # let throttling budgets recover between runs

    return results
