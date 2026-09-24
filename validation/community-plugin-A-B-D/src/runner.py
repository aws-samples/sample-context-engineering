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
import os
import time
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models import BedrockModel, CacheConfig

# The three community packages. Imported at module scope, not lazily: a missing package is a broken
# installation and should fail at import with the package's own name, rather than midway through a
# paid run.
from strands_context_graph import ContextGraph, EmbeddingSimilarityMatcher
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure
from strands_relevance_filter import BedrockReranker, FileStore, RelevanceFilter

from . import accuracy, config, metrics, scenario, tools
from .density_rerank import DensityReranker
from .config import (
    AGENT_MODEL_ID,
    ARTIFACTS_DIR,
    AWS_PROFILE,
    EMBED_MODEL_ID,
    GRAPH_TUNING,
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
    #
    # One TTL for all three checkpoints: Bedrock requires them non-increasing across toolConfig,
    # system and messages, so a single value is the only setting that cannot be rejected. tools_ttl
    # must be passed explicitly -- it defaults to None, which caches the system prompt and leaves
    # the tool schema uncached, and the tool schema is 38% of this harness's input.
    # "default" means cache ON at Bedrock's own TTL, emitted as {"type": "default"} with no ttl
    # field. That is the only form botocore 1.40 accepts: its Converse model declares cachePoint
    # with "type" alone, so an explicit "5m"/"1h" is rejected before the request leaves the host
    # with ParamValidationError. An explicit TTL needs a newer botocore.
    ttl = None if config.CACHE_TTL == "default" else config.CACHE_TTL
    section = True if ttl is None else ttl
    cache_config = (
        CacheConfig(strategy="auto", ttl=ttl, tools_ttl=section, system_prompt_ttl=section)
        if config.CACHE_TTL
        else None
    )
    model = BedrockModel(
        boto_session=session,
        boto_client_config=_client_config(),
        model_id=AGENT_MODEL_ID,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        **({"cache_config": cache_config} if cache_config else {}),
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


class _MeteredDensityReranker(_MeteredReranker, DensityReranker):
    """Metering and the citable-density prior, composed rather than merged.

    Both classes delegate through ``super().score``, so the method resolution order does the work:
    ``_MeteredReranker`` counts the batch, ``DensityReranker`` lifts the dense chunks, and
    ``BedrockReranker`` makes the one remote call. Neither class had to learn about the other.

    Selected by ``VALIDATION_DENSITY_RERANK=1``, so one experiment arm differs from the control by this
    class and nothing else.
    """


DENSITY_RERANK = os.environ.get("VALIDATION_DENSITY_RERANK") == "1"
"""Whether the citable-density prior is in play, read once at import.

An experiment switch and not a setting: it names which hypothesis a run is testing, and every run
records it, so a result can never be read without knowing which scorer produced it.
"""


RELEVANCE_RETRIEVAL_TOOL = os.environ.get("VALIDATION_RELEVANCE_RETRIEVAL_TOOL") == "1"
"""Whether the relevance filter registers its own ``retrieve_context`` tool, read once at import.

Off by default, matching the plugin's own default: the filter's contract ends at the tool result, so
the arm measures preview quality alone. Set ``VALIDATION_RELEVANCE_RETRIEVAL_TOOL=1`` to reproduce
the published figures, which were all measured with the tool present.
"""


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
"""The graph's artifact-retrieval tool. Registered in every arm since the filter lost its own.

Kept as the record of why it was ever dropped, because the reason was measured, not assumed. On the
first 60-turn run the ``all`` configuration was the only one that
could not answer A5 -- the turn whose answer is one row of a 90-character-per-line statement -- and
the model said why in its own answer: "every export's artifact reference has come back unreachable
... I can't read the stored artifacts". It had called ``expand_artifact`` with a reference the
relevance filter had minted.

At that time the two packages each shipped their own retrieval tool over their own store, and nothing
bridged them: ``RelevanceFilter`` stored the raw sub-blocks it replaced and handed out references its
``retrieve_context`` resolved, while ``ContextGraph`` recorded addresses it saw in placeholder text
and resolved them through a store of its own. The graph's README is explicit that its bridge to
another plugin's stash is built entirely on private symbols and degrades to "answers as prose naming
the miss" -- which is exactly what happened, and the vended stack never hit it because relevance
lived *inside* the ContextManager whose stash the graph bridged to.

So with both installed there were two plausible tools for one job and only one could resolve the
reference, and dropping the graph's left exactly one artifact path. That collision is gone:
``RelevanceFilter`` now defaults to ``include_retrieval_tool=False``, mints no reference and writes
no store, so ``expand_artifact`` is the only artifact path there is and dropping it leaves none.
The graph's two other tools -- ``expand_card`` and ``find_context`` -- were never part of this:
they reach back into the conversation's own turns, a different job the relevance filter does not do.
"""


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
        # Namespaced by run tag as well as configuration: the tag is what keeps two runs that
        # execute at the same time -- a benchmark sweeping one model per process, say -- from
        # writing into each other's stored sub-blocks and serving the wrong content back through
        # retrieve_context. Configuration alone was enough only while one run existed at a time.
        storage_root = ARTIFACTS_DIR / (metrics.RUN_TAG or "untagged") / config.name
        storage_root.mkdir(parents=True, exist_ok=True)

        reranker_class = _MeteredDensityReranker if DENSITY_RERANK else _MeteredReranker
        reranker = reranker_class(
            model_id=RERANK_MODEL_ID,
            boto_session=session,
            boto_client_config=_client_config(),
        )
        config.extra["_reranker"] = reranker

        relevance = RelevanceFilter(
            # File-backed rather than in-memory: kept for the run's own inspection of what was cut.
            # With the retrieval tool off the plugin writes nothing here, so the directory stays
            # empty unless VALIDATION_RELEVANCE_RETRIEVAL_TOOL turns the tool back on.
            store=FileStore(str(storage_root)),
            # The filter's job ends at the tool result: llm -> tool -> filtered result -> llm. A
            # retrieval tool would put every recovered chunk into the history as a message that is
            # re-sent on every later call, which is what made this arm cost MORE than no plugin on
            # Haiku 4.5 (+21.6%) while saving on Opus.
            include_retrieval_tool=RELEVANCE_RETRIEVAL_TOOL,
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

        # One tuning for every arm the graph appears in. The split by "is the relevance filter also
        # installed" is gone: the filter acts on a tool result before it enters the history, the graph
        # acts on a history that already exists, so the filter only makes the graph's input smaller —
        # it does not change what folding a history should cost. See GraphTuning's docstring.
        tuning = GRAPH_TUNING
        config.extra["_graph_tuning"] = "unified"

        graph = ContextGraph(
            expand_threshold=tuning.expand_threshold,
            collapse_floor=tuning.collapse_floor,
            link_threshold=tuning.link_threshold,
            description_tokens=tuning.description_tokens,
            body_budget=tuning.body_budget,
            min_cards=THRESHOLDS.min_cards,
            max_retrieval_cycles=tuning.max_retrieval_cycles,
            reuse_ttl_cycles=tuning.reuse_ttl_cycles,
            tags_per_card=tuning.tags_per_card,
            neighbors_per_candidate=tuning.neighbors_per_candidate,
            # Always on. This used to be ``not config.relevance``, to leave exactly one artifact
            # path when the filter shipped a competing ``retrieve_context`` over a store of its own.
            # The filter no longer registers a retrieval tool at all, so there is no second path to
            # disambiguate from and the condition only took the capability away.
            # See _GRAPH_ARTIFACT_TOOL for the measurement that motivated the old drop.
            include_artifact_tool=True,
            matcher=matcher,
        )
        config.extra["_graph_artifact_tool_dropped"] = False
        config.extra["_graph"] = graph
        plugins.append(graph)

    if config.disclosure:
        plugins.append(
            ProgressiveToolDisclosure(
                catalog_chars=THRESHOLDS.catalog_chars,
                ttl_cycles=THRESHOLDS.ttl_cycles,
                top_k=THRESHOLDS.top_k,
                # A retrieval tool must never need discovery: the model is told to use it in the
                # guidance text that replaces the payload, and a cycle spent loading it would be an
                # artefact of the harness rather than of the strategy.
                #
                # Derived from the plugins rather than hard-coded, so excluding the graph's artifact
                # tool or renaming one cannot leave a stale name here. `list_accounts` is the one
                # literal: it is a domain tool of the scenario, not a plugin's.
                always_available=[
                    *(graph.retrieval_tool_names if graph is not None else ()),
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
                    "loads": state.loads,
                    # Calls to a catalog tool that skipped get_tool_details, recovered by the guard.
                    "premature_cancellations": state.premature_cancellations,
                    "exposed_at_end": sorted(state.exposed),
                    "exposed_count_at_end": len(state.exposed),
                    # Tokens the catalog summaries cost; billed in compare.py at the agent's rates.
                    "summary_usage": dict(state.summary_usage),
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


def _partial_answer(agent: Agent) -> str:
    """Return the assistant text the agent left in history, for a turn cut short at ``max_tokens``.

    Strands adds the partial message to the conversation before raising, so the text exists; the
    harness simply was not reading it. Only a trailing ``assistant`` message counts: a turn that
    failed on a tool error ends on a tool result, and inventing an answer out of an earlier turn's
    message would score one turn with another turn's work.

    Args:
        agent: The agent whose history to read. Not modified.

    Returns:
        The concatenated text blocks of the trailing assistant message, or ``""`` when the history
        does not end in one -- which is every failure that is not an output truncation.
    """
    try:
        messages = agent.messages
        if not messages or messages[-1].get("role") != "assistant":
            return ""

        blocks = messages[-1].get("content") or []
        return "\n".join(block["text"] for block in blocks if isinstance(block, dict) and "text" in block)
    except Exception:  # noqa: BLE001 - a measurement must not raise inside an exception handler
        logger.debug("could not recover a partial answer from history", exc_info=True)
        return ""


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
            # An output cut short at ``max_tokens`` is not the same failure, and scoring it as an
            # empty answer made it look like one. Strands states that the partial message was added
            # to the history -- so the model DID say something, and the harness was throwing it away
            # and then scoring the silence as materially wrong. Measured on GLM 4.7 Flash, one to two
            # scored turns per run, on exactly the arms that fold context: folding makes the model
            # restate figures verbatim, which is what runs an answer past the cap.
            #
            # Recovering the text is not leniency. It scores what the model actually produced, which
            # is the only thing the comparison is entitled to judge -- a truncated answer that never
            # reaches its figure still fails its check, and now it fails for the right reason.
            recovered = _partial_answer(agent)
            if recovered:
                record.response_text = recovered
                record.response_chars = len(recovered)
                record.answer_truncated = True
                logger.info(
                    "turn %s truncated at max_tokens | scoring the %d chars the model did produce",
                    turn.label,
                    len(recovered),
                )
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
