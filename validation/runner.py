"""Builds each configuration and replays the scenario against real Bedrock calls.

One decision worth stating: every configuration, including ``baseline``, runs with the
``ContextOffloader`` installed. Without it, the 60k-250k character tool payloads would
overflow the window outright and the baseline would fail rather than merely cost more —
which would make the comparison a story about crashing, not about tokens. So the
baseline is the offloader with its default positional prefix preview, and the
``relevance`` configuration changes only the preview strategy. That isolates what
relevance filtering actually contributes.

Every configuration runs with ``NullConversationManager`` so that history handling is not a
confounding variable between them: a sliding window or summarizer would mutate
``agent.messages`` before the call is assembled, which the graph strategy rides on.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import statistics
import time
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models import BedrockModel
from strands.models.model import Model
from strands._context_manager.methods.reranker import BedrockReranker
from strands.vended_plugins._embedding import BedrockEmbedder

from . import accuracy, scenario, tools
from .config import (
    AGENT_MODEL_ID,
    ARTIFACTS_DIR,
    AWS_PROFILE,
    REGION,
    RERANK_MODEL_ID,
    RUN_CONFIGS,
    SESSIONS_DIR,
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
    ``credential_process`` mints short-lived ones on demand, and the default chain resolves whatever
    the environment already provides.
    """
    if AWS_PROFILE:
        return boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    return boto3.Session(region_name=REGION)


def _client_config() -> BotocoreConfig:
    """Timeouts sized for parallel mode.

    The default read timeout expires on the heaviest turn when five configurations run
    concurrently: a turn that pulls a 60k-character document and then makes several
    retrieval calls is slow on its own, and five of them share one account's throughput. An
    earlier run lost that turn to a read timeout in two configurations, which scored as an
    accuracy failure and inflated the very variance the repeats exist to measure — a harness
    artefact reported as a strategy result.

    Retries are adaptive so throttling is absorbed rather than surfacing as a lost turn.
    """
    return BotocoreConfig(
        read_timeout=180,
        connect_timeout=20,
        retries={"max_attempts": 5, "mode": "adaptive"},
    )


def _agent_model(session: boto3.Session) -> BedrockModel:
    # No temperature: Opus 4.8 rejects the parameter outright with a ValidationException
    # ("`temperature` is deprecated for this model"). Determinism across configurations
    # therefore comes from a fixed scenario and fixed payloads, not from a sampling knob.
    model = BedrockModel(
        boto_session=session,
        boto_client_config=_client_config(),
        model_id=AGENT_MODEL_ID,
        max_tokens=4_096,
    )
    install_log_tagging(model.client, role="agent")
    return model


class _MeteredModel(Model):
    """Wraps a model and accumulates the usage of every call made through it.

    Subclasses ``Model`` rather than duck-typing it: a plugin that validates its ``model``
    argument with ``isinstance`` rejects a bare delegate at construction, so metering has to
    subclass rather than wrap.

    Delegates by attribute so it stays a drop-in for any model implementation, and counts by
    reading the usage off the stream rather than estimating from the request.
    """

    def __init__(self, inner: Model) -> None:
        self._inner = inner
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class does not define, so the abstract members
        # implemented below take precedence and everything else falls through to the wrapped
        # model unchanged.
        return getattr(self._inner, name)

    def get_config(self) -> Any:
        return self._inner.get_config()

    def update_config(self, **model_config: Any) -> None:
        self._inner.update_config(**model_config)

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        async for event in self._inner.structured_output(*args, **kwargs):
            yield event

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Pass the call through, reading usage off the provider's metadata chunk.

        Usage is taken from the raw ``metadata`` chunk rather than from the ``stop`` tuple the
        agent's middleware sees: that tuple is assembled by ``stream_messages``, one layer
        above ``model.stream``, so at this level it never appears. Reading the wrong layer
        counted the calls correctly and the tokens as zero.
        """
        self.calls += 1
        async for event in self._inner.stream(*args, **kwargs):
            try:
                payload = event if isinstance(event, dict) else getattr(event, "_data", None)
                if isinstance(payload, dict):
                    usage = (payload.get("metadata") or {}).get("usage")
                    if isinstance(usage, dict):
                        self.input_tokens += usage.get("inputTokens") or 0
                        self.output_tokens += usage.get("outputTokens") or 0
            except Exception:  # noqa: BLE001 - metering must not break a curation pass
                pass
            yield event


class _MeteredEmbedder(BedrockEmbedder):
    """Counts what the graph's similarity matcher actually sends to Bedrock.

    Subclasses rather than wraps, and overrides ``_invoke`` rather than ``embed``, because
    those are the two things that make the count *billable* instead of merely indicative:
    ``embed`` is called with every description on every scored turn, while ``_invoke`` receives
    only the cache misses, deduplicated. Metering the outer method would bill a session for
    vectors it never paid for — which is precisely the claim the cache exists to make.

    Tokens are estimated from characters, not read from the response: Cohere's embed API returns
    no usage block. The estimate uses the same divisor as the rest of the harness, so the
    embedding column is on the same footing as every other token figure here.

    Attributes:
        embed_calls: Bedrock round trips made, one per batch of at most 96 texts.
        embed_texts: Texts that reached Bedrock, after cache and deduplication.
        embed_tokens: Estimated input tokens of those texts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.embed_calls = 0
        self.embed_texts = 0
        self.embed_tokens = 0

    def _invoke(self, texts: list[str], purpose: str) -> list[list[float]]:
        # Counted before the call: a failure halfway through still consumed what it sent.
        self.embed_calls += 1
        self.embed_texts += len(texts)
        self.embed_tokens += sum(estimate_tokens(text) for text in texts)
        return super()._invoke(texts, purpose)


class _MeteredReranker(BedrockReranker):
    """Counts the graph's rerank traffic in the unit Bedrock bills.

    A search unit is one query against up to 100 documents, which is exactly the batch the
    reranker pages into — so the count is ``ceil(chunks / max_sources_per_query)`` per call,
    mirroring what ``RelevancePreview`` already does for the offloader. Kept separate from the
    offloader's own tally: under ``graph-all`` both are live, and adding them together would
    hide which strategy is paying.

    Attributes:
        rerank_calls: Calls to ``score``, one per selection refinement.
        search_units: Billable batches submitted.
        rerank_documents: Documents submitted across those batches.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rerank_calls = 0
        self.search_units = 0
        self.rerank_documents = 0

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        if chunks:
            self.rerank_calls += 1
            self.rerank_documents += len(chunks)
            self.search_units += math.ceil(len(chunks) / max(1, self.max_sources_per_query))
        return await super().score(query, chunks)


def build_plugins(config: RunConfig, session: boto3.Session) -> list[Any]:
    """Assemble the plugin list for one configuration."""
    from strands._context_manager.context_manager import ContextManager
    from strands._context_manager.strategies.offload import Offload
    from strands.storage import LocalFileStorage
    from strands.vended_plugins.context_offloader import ContextOffloader, FileStorage
    from strands.vended_plugins.progressive_tool_disclosure import ProgressiveToolDisclosure

    storage_root = ARTIFACTS_DIR / config.name
    storage_root.mkdir(parents=True, exist_ok=True)

    offloader_kwargs: dict[str, Any] = {
        "storage": FileStorage(str(storage_root)),
        "max_result_tokens": THRESHOLDS.max_result_tokens,
        "preview_tokens": THRESHOLDS.preview_tokens,
        "evict_after_cycles": THRESHOLDS.evict_after_cycles,
    }

    if config.relevance:
        # Relevance is a ContextManager offload strategy now, not an offloader preview mode, so this
        # configuration swaps the offloader out rather than reconfiguring it. Both cannot preview the
        # same result: the offloader rewrites it to a positional preview on AfterToolCallEvent, which
        # would leave relevance scoring the offloader's leftovers instead of the raw payload.
        #
        # The manager carries its own stash, so the payload is still persisted and still retrievable
        # (through retrieve_context rather than retrieve_offloaded_content). A threshold, not a
        # utilization, is what registers the base eager hook — the offloader's own proactive timing:
        # each oversized tool result is rewritten as it enters the conversation.
        # The manager travels on ``config.extra`` rather than in ``plugins``: the SDK rejects a
        # ContextManager passed through ``plugins`` because session persistence detects it only on
        # the ``context_manager`` parameter. Every other configuration leaves the key absent, so the
        # agent builder passes ``None`` and nothing about them changes.
        config.extra["_context_manager"] = ContextManager(
            strategies=[
                Offload.relevance(
                    "tool_results",
                    {
                        "reranker": _MeteredReranker(
                            model_id=RERANK_MODEL_ID,
                            boto_session=session,
                        ),
                        "relevance_threshold": THRESHOLDS.relevance_threshold,
                        "chunk_tokens": THRESHOLDS.chunk_tokens,
                        "preview_tokens": THRESHOLDS.preview_tokens,
                    },
                ).when(threshold=THRESHOLDS.max_result_tokens),
            ],
            stash={"storage": LocalFileStorage(str(storage_root))},
        )
        plugins: list[Any] = []
    else:
        plugins = [ContextOffloader(**offloader_kwargs)]

    # Built before the disclosure plugin, because the disclosure plugin needs the bound method
    # as its supplemental referenced source. Order in this list does not decide handler order:
    # the graph moves its own handler to index zero of the stage.
    graph = _graph_strategy(config, session) if config.graph else None
    if graph is not None:
        config.extra["_graph"] = graph
        plugins.append(graph)

    if config.disclosure:
        plugins.append(
            ProgressiveToolDisclosure(
                catalog_tokens=THRESHOLDS.catalog_tokens,
                ttl_cycles=THRESHOLDS.ttl_cycles,
                top_k=THRESHOLDS.top_k,
                # The retrieval tool must never need discovery: the model is told to use it
                # in the guidance text that replaces the payload, and a cycle spent finding it
                # would be an artefact of the harness rather than of the strategy. Both names
                # appear because the offloader and the ContextManager register different tools,
                # and only one of them is installed in any given configuration.
                always_available=["retrieve_offloaded_content", "retrieve_context", "list_accounts"],
                # Without this, a Card stepping down to a Description drops its tool from the
                # full-specification block to the pre-specification one, and the model reads
                # about a tool it no longer knows how to call. That is a defect the graph
                # introduces, so the graph is what has to close it.
                referenced_source=graph.referenced_tool_names if graph is not None else None,
            )
        )

    return plugins


_GRAPH_KNOBS = (
    "expand_threshold",
    "collapse_floor",
    "description_tokens",
    "tags_per_card",
    "rarity_weight",
    "body_budget",
    "min_cards",
    "link_threshold",
    "reuse_ttl_cycles",
    "recent_cards",
    "select_top_k",
    "persist",
)
"""Graph parameters a variant may override through ``extra``.

Anything absent is left to the plugin's own default rather than restated here: a harness that
copies a default is a harness that silently measures the stale value after the default moves.
"""


def _graph_strategy(config: RunConfig, session: boto3.Session) -> Any:
    """Build the graph strategy a configuration asks for.

    The matcher is constructed with the run's session so its embedding calls are billed to the
    same account as everything else, and so the harness never falls back on ambient credentials.
    """
    from strands.vended_plugins.context_graph import ContextStrategy, EmbeddingSimilarityMatcher

    overrides = {knob: config.extra[knob] for knob in _GRAPH_KNOBS if knob in config.extra}

    if config.extra.get("rerank"):
        # The second stage of the selection. It only refines which Cards the note's pick contains,
        # and a failure is a skipped step, so this cannot make the run fail — only slower.
        reranker = _MeteredReranker(model_id=RERANK_MODEL_ID, boto_session=session)
        config.extra["_graph_reranker"] = reranker
        overrides["reranker"] = reranker

    # Kept on the config so the run can bill the graph for the embedding calls its ranking
    # makes: a model call of its own, invisible to the agent's middleware, and reporting the
    # saving without it is not a comparison.
    embedder = _MeteredEmbedder(purpose="document", boto_session=session)
    config.extra["_graph_embedder"] = embedder

    return ContextStrategy(
        strategy="graph",
        matcher=EmbeddingSimilarityMatcher(embedder=embedder),
        **overrides,
    )


def build_agent(
    config: RunConfig,
    session: boto3.Session,
    collector: RunCollector,
    *,
    session_id: str | None = None,
) -> Agent:
    """Construct the agent under test and attach the measurement middleware.

    ``session_id`` installs a ``FileSessionManager``, which is what makes a resume possible: the
    history and ``agent.state`` are restored from disk into a *new* agent, exactly as an ephemeral
    runtime would. File and not S3 because the backend is not what is being measured — the graph
    rides ``agent.state``, so every session manager carries it identically, and a local directory
    keeps the measurement free of network variance it would otherwise attribute to the strategy.
    """
    from strands._middleware.stages import InvokeModelStage

    plugins = build_plugins(config, session)

    session_manager = None
    if session_id is not None:
        from strands.session import FileSessionManager

        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        session_manager = FileSessionManager(session_id=session_id, storage_dir=str(SESSIONS_DIR))

    # Set only by the relevance configurations. Absent means ``None``, which is what every
    # offloader-based configuration wants.
    context_manager = config.extra.get("_context_manager")

    agent = Agent(
        model=_agent_model(session),
        tools=tools.all_tools(),
        system_prompt=scenario.SYSTEM_PROMPT,
        # Same manager for every configuration so that history handling is not a
        # confounding variable between them.
        conversation_manager=NullConversationManager(),
        session_manager=session_manager,
        plugins=plugins,
        context_manager=context_manager,
        callback_handler=None,
    )

    # Registered after construction, so it runs last in the stage and observes the
    # projection the plugins produced rather than the pre-plugin baseline.
    agent._middleware_registry.add_middleware(InvokeModelStage, collector.middleware())

    return agent


def _installed_plugins(agent: Agent) -> list[Any]:
    """Return the plugins registered on ``agent``.

    The registry keeps them in a name-keyed dict on a private attribute; reading it
    defensively means a rename upstream costs us the counters, not the run.
    """
    try:
        return list(agent._plugin_registry._plugins.values())
    except Exception:  # noqa: BLE001
        return []


def _collect_plugin_counters(agent: Agent, config: RunConfig) -> dict[str, Any]:
    """Read the strategy-specific evidence off the plugins after a run.

    These counters are the difference between "tokens went down" and "the strategy did
    what it claims". Every read is defensive: the counters live on private attributes and
    a missing one should blank a field, not fail a run.
    """
    counters: dict[str, Any] = {}

    for plugin in _installed_plugins(agent):
        name = type(plugin).__name__

        if name == "ProgressiveToolDisclosure":
            try:
                state = plugin._states[agent]
                counters["disclosure"] = {
                    "searches": state.searches,
                    "premature_cancellations": state.premature_cancellations,
                    "exposed_at_end": sorted(state.exposed),
                    "exposed_count_at_end": len(state.exposed),
                }
            except Exception:  # noqa: BLE001
                counters["disclosure"] = {"error": "state unavailable"}

        elif name == "ContextOffloader":
            counters["offloader"] = {"preview_strategy": "prefix"}
        elif name == "ContextManager":
            # Relevance lives here now, so its billable scoring is read off the strategy rather
            # than off the offloader. It stays under the "offloader" key because that is what the
            # cost model sums into rerank_cost; a new key would price relevance at zero.
            try:
                strategies = {getattr(s, "name", "?"): s for s in plugin._strategies}
                relevance = strategies.get("offload:relevance")
                preview = getattr(relevance, "_preview", None) if relevance is not None else None
                counters["offloader"] = {
                    "preview_strategy": "relevance" if relevance is not None else "none",
                    "strategies": sorted(strategies),
                    "search_units": getattr(preview, "search_units", 0),
                }
            except Exception:  # noqa: BLE001
                counters["offloader"] = {"error": "counters unavailable"}

        elif name == "ContextStrategy":
            # The graph's evidence that it did what it claims. Without these, "tokens went
            # down" and "the ladder collapsed to one rung" read identically in the report.
            try:
                state = plugin._states[agent]
                choice = state.choice
                dialogue = {"full": 0, "description": 0, "title": 0}
                # ``title`` on the evidence axis is not a rung of the ladder: it means the selection
                # did not address the Card, so nothing of it reached the call.
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
                    "referenced_at_end": len(state.referenced),
                    "vectors_cached": len(state.vectors),
                }
            except Exception as error:  # noqa: BLE001
                counters["graph"] = {"error": f"state unavailable: {error}"}

    embedder = config.extra.get("_graph_embedder")
    if embedder is not None:
        counters["embedding_cost"] = {
            "calls": embedder.embed_calls,
            "texts": embedder.embed_texts,
            "input_tokens": embedder.embed_tokens,
            "model_id": embedder.model_id,
        }

    graph_reranker = config.extra.get("_graph_reranker")
    if graph_reranker is not None:
        counters["rerank_cost"] = {
            "calls": graph_reranker.rerank_calls,
            "search_units": graph_reranker.search_units,
            "documents": graph_reranker.rerank_documents,
        }

    counters["live_messages_at_end"] = len(agent.messages)
    counters["registered_tools"] = len(agent.tool_names)
    return counters


_GRAPH_LOGGER = "strands.vended_plugins.context_graph"
"""The package logger, so every module's records are read, not only the plugin's.

Scoped to ``.plugin`` at first, which silently lost the one record that says a fresh process *loaded*
the graph instead of deriving it — that one is emitted by ``persistence``. A pattern with nothing to
match against reads as "it never happened", which is the shape of wrong answer this harness is built
to avoid."""

_GRAPH_PATTERNS = {
    "choices": re.compile(
        r"turn choice computed \| turn=<(?P<turn>\d+)>"
        r" \| dialogue_full=<(?P<dialogue_full>\d+)>"
        r" \| dialogue_description=<(?P<dialogue_description>\d+)>"
        r" \| dialogue_title=<(?P<dialogue_title>\d+)>"
        r" \| evidence_full=<(?P<evidence_full>\d+)>"
        r" \| evidence_description=<(?P<evidence_description>\d+)>"
        r"(?: \| evidence_title=<(?P<evidence_title>\d+)>)?"
        r"(?: \| unaddressed=<(?P<unaddressed>\d+)>)?"
        r" \| choice_micros=<(?P<choice_micros>\d+)>"
    ),
    "cycles": re.compile(r"retrieval cycles counted \| turn=<(?P<turn>\d+)> \| retrieval_cycles=<(?P<cycles>\d+)>"),
    "deliveries": re.compile(
        r"delivery produced \| received=<(?P<received>\d+)> \| projected=<(?P<projected>\d+)>"
        r" \| final_blocks=<(?P<final_blocks>\d+)> \| input_tokens=<(?P<input_tokens>\d+)>"
    ),
    "ratios": re.compile(
        r"card compaction ratio \| title=<(?P<title>.*?)> \| full_tokens=<(?P<full_tokens>\d+)>"
        r" \| description_tokens=<(?P<description_tokens>\d+)> \| ratio=<(?P<ratio>[\d.]+)>"
    ),
    "referenced": re.compile(r"supplemental referenced source published \| names=<(?P<names>\d+)>"),
    # The two records that say which route a fresh process took to get its graph back. Without them
    # "it recovered" cannot be told from "it loaded", which is the whole difference persistence makes.
    "restored": re.compile(r"graph state restored \| cards=<(?P<cards>\d+)> \| turn=<(?P<turn>\d+)>"),
    "notes": re.compile(
        r"note spread computed \| cards=<(?P<cards>\d+)> \| min=<(?P<min>[\d.]+)>"
        r" \| median=<(?P<median>[\d.]+)> \| max=<(?P<max>[\d.]+)>"
        r" \| below_floor=<(?P<below_floor>\d+)> \| above_threshold=<(?P<above_threshold>\d+)>"
    ),
}
"""One pattern per record the graph emits. Reading the record rather than re-deriving the
number is what makes the harness verify the instrumentation at the same time: a record that
stops being emitted, or loses a field, becomes an empty row instead of a wrong one."""


class _GraphRecorder(logging.Handler):
    """Collect the graph's own per-turn records for the duration of one run.

    The end-of-run counters only see the last turn, and the numbers that decide whether the
    graph works are curves: the three resolutions per turn, and the retrieval cycles per turn.
    Those exist only as log records, so this reads them there.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.rows: dict[str, list[dict[str, Any]]] = {key: [] for key in _GRAPH_PATTERNS}
        self.previous_level = logging.NOTSET
        self.previous_propagate = True

    def emit(self, record: logging.LogRecord) -> None:
        """Match one record against the known patterns and keep the fields it carries."""
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a diagnostic must not fail a run
            return
        for key, pattern in _GRAPH_PATTERNS.items():
            match = pattern.search(message)
            if match:
                self.rows[key].append(match.groupdict())
                return

    def summary(self) -> dict[str, Any]:
        """Reduce the collected rows to the curves and the aggregates worth reporting.

        The per-turn lists are kept whole: a mean over a curve hides exactly the shape the
        curve exists to show, and the retrieval-cycle curve going flat rather than descending
        is the reading that says the graph traded tokens for latency.
        """
        # ``or 0`` and not ``int(value)``: the optional groups of the pattern are ``None`` when the
        # plugin emitted the older record shape, and a missing field reads as zero rather than as a
        # crash that would lose every other row with it.
        choices = [{key: int(value or 0) for key, value in row.items()} for row in self.rows["choices"]]
        cycles = [(int(row["turn"]), int(row["cycles"])) for row in self.rows["cycles"]]
        deliveries = [{key: int(value) for key, value in row.items()} for row in self.rows["deliveries"]]
        ratios = [float(row["ratio"]) for row in self.rows["ratios"]]
        subject_ratios = [ratio for ratio in ratios if ratio > 0.0]
        published = [int(row["names"]) for row in self.rows["referenced"]]

        summary: dict[str, Any] = {
            "turns_recorded": len(choices),
            "dialogue_per_turn": [
                [row["dialogue_full"], row["dialogue_description"], row["dialogue_title"]] for row in choices
            ],
            "evidence_per_turn": [[row["evidence_full"], row["evidence_description"]] for row in choices],
            # How many Cards the selection did not address, per turn. The whole reading of the
            # selection: a zero here on every turn means it excluded nothing, so whatever else the
            # run shows is not the selection's effect.
            "unaddressed_per_turn": [int(row.get("unaddressed") or 0) for row in choices],
            "unaddressed_max": max((int(row.get("unaddressed") or 0) for row in choices), default=0),
            "choice_micros_per_turn": [row["choice_micros"] for row in choices],
            "retrieval_cycles_per_turn": [count for _turn, count in cycles],
            "retrieval_cycles_total": sum(count for _turn, count in cycles),
            "deliveries": len(deliveries),
            "final_blocks_total": sum(row["final_blocks"] for row in deliveries),
            "full_pass_deliveries": sum(1 for row in deliveries if row["final_blocks"] == 0),
            "compaction_ratio_samples": len(subject_ratios),
            "referenced_published_max": max(published, default=0),
        }
        if choices:
            summary["choice_micros_mean"] = round(
                sum(row["choice_micros"] for row in choices) / len(choices), 1
            )
            # Three populated rungs on the dialogue axis is the evidence that the ladder is a
            # ladder. If this is zero, the graph is making a two-way decision, not a three-way one.
            summary["turns_with_three_dialogue_rungs"] = sum(
                1
                for row in choices
                if row["dialogue_full"] and row["dialogue_description"] and row["dialogue_title"]
            )
        if subject_ratios:
            summary["compaction_ratio_median"] = round(statistics.median(subject_ratios), 2)
            summary["compaction_ratio_mean"] = round(statistics.mean(subject_ratios), 2)
        restored = self.rows["restored"]
        summary["graph_loads_from_the_session"] = len(restored)
        summary["cards_loaded"] = max((int(row["cards"]) for row in restored), default=0)

        notes = self.rows["notes"]
        if notes:
            # The spread is what says whether a threshold sits inside the range the matcher answers
            # in. A rung nobody reached has two causes and the rung counts cannot separate them.
            summary["note_min_of_run"] = min(float(row["min"]) for row in notes)
            summary["note_max_of_run"] = max(float(row["max"]) for row in notes)
            summary["note_median_mean"] = round(
                statistics.mean(float(row["median"]) for row in notes), 4
            )
            summary["notes_below_floor_total"] = sum(int(row["below_floor"]) for row in notes)
            summary["notes_above_threshold_total"] = sum(int(row["above_threshold"]) for row in notes)
            summary["notes_scored_total"] = sum(int(row["cards"]) for row in notes)

        if cycles:
            half = len(cycles) // 2
            first = sum(count for _turn, count in cycles[:half])
            second = sum(count for _turn, count in cycles[half:])
            summary["retrieval_cycles_first_half"] = first
            summary["retrieval_cycles_second_half"] = second
            summary["retrieval_cycles_reading"] = (
                "descending" if second < first else ("flat" if second == first else "ascending")
            )
        return summary


_graph_recording = False
"""Whether a graph recorder is already attached somewhere in this process.

The plugin's logger is process-wide and its records carry no agent identifier, so two graph
configurations recording at the same time each capture *both* sessions' records. That is not a
noisy measurement, it is a wrong one: an 18-turn run reports 36 turns, and the two configurations
report identical curves. So the second recorder declines rather than producing a number that
looks fine and is not. Use ``--sequential`` to record more than one graph configuration in a run.
"""


def _start_graph_recording(config: RunConfig) -> _GraphRecorder | None:
    """Attach the graph recorder, raising the plugin logger to DEBUG for the run.

    ``delivery produced`` and ``card compaction ratio`` are guarded by a DEBUG level check in
    the plugin, so the level has to be raised for them to exist at all. Propagation is muted
    while it is raised, unless the caller already asked for DEBUG with ``--verbose`` — in which
    case they want to see the records and muting them would take that away.
    """
    global _graph_recording

    if not config.graph:
        return None

    if _graph_recording:
        logger.warning(
            "%s: a graph recorder is already attached, so the per-turn curves are not recorded "
            "for this configuration. Re-run with --sequential to record both.",
            config.name,
        )
        return None

    plugin_logger = logging.getLogger(_GRAPH_LOGGER)
    recorder = _GraphRecorder()
    _graph_recording = True
    recorder.previous_level = plugin_logger.level
    recorder.previous_propagate = plugin_logger.propagate
    plugin_logger.propagate = plugin_logger.isEnabledFor(logging.DEBUG)
    plugin_logger.addHandler(recorder)
    plugin_logger.setLevel(logging.DEBUG)
    return recorder


def _stop_graph_recording(recorder: _GraphRecorder | None) -> dict[str, Any]:
    """Detach the recorder, restore the logger, and return the rows it collected."""
    global _graph_recording

    if recorder is None:
        return {}

    _graph_recording = False
    plugin_logger = logging.getLogger(_GRAPH_LOGGER)
    plugin_logger.removeHandler(recorder)
    plugin_logger.setLevel(recorder.previous_level)
    plugin_logger.propagate = recorder.previous_propagate
    return recorder.summary()


async def run_configuration(
    config: RunConfig,
    *,
    turn_limit: int | None = None,
    phases: tuple[str, ...] | None = None,
    total_turns: int | None = None,
    resume_at: int | None = None,
) -> RunCollector:
    """Replay the scenario under one configuration and return its measurements.

    ``resume_at`` splits the replay in two: the turns before it run on one agent, that agent is
    dropped, and a **new** agent is built over the same session for the rest. That is the shape an
    ephemeral runtime has, and it is the only way this harness can exercise what persistence exists
    for — a long-lived process never restores anything, so the load path never runs.

    What the split proves depends on the configuration, and the contrast is the measurement: with
    ``persist=True`` the second agent loads the graph and decides with it on its first turn; without
    it, the second agent derives the graph by scan, which costs ~2.9s over a long conversation.
    """
    session = boto_session()
    collector = RunCollector(config.name)
    session_id = f"{config.name}-{int(time.time())}" if resume_at is not None else None
    agent = build_agent(config, session, collector, session_id=session_id)
    recorder = _start_graph_recording(config)

    script = scenario.turns(limit=turn_limit, phases=phases, total=total_turns)
    logger.info("=== %s: %d turns, %d tools ===", config.name, len(script), len(agent.tool_names))

    run_started = time.perf_counter()

    for index, turn in enumerate(script):
        if resume_at is not None and index == resume_at:
            # The restart. The agent is dropped, taking its graph with it — the state is weakly keyed
            # by the agent, so this is the same loss a process exit causes. The new agent restores the
            # history from the session, which fires no MessageAddedEvent, so whether it can decide on
            # this turn is exactly the question persistence answers.
            agent = build_agent(config, session, collector, session_id=session_id)
            collector.resumed_at_turn = index
            collector.resumed_message_count = len(agent.messages)
            logger.info(
                "%s: resumed at turn %d with %d restored messages",
                config.name,
                index,
                len(agent.messages),
            )

        record = collector.begin_turn(index, turn.label, turn.prompt)
        turn_started = time.perf_counter()
        try:
            result = await agent.invoke_async(turn.prompt)
            record.response_text = str(result)
            record.response_chars = len(record.response_text)
        except Exception as error:  # noqa: BLE001 - one bad turn must not lose the run
            record.error = f"{type(error).__name__}: {error}"
            collector.errors.append(f"{turn.label}: {record.error}")
            logger.warning("turn %s failed", turn.label, exc_info=True)
        finally:
            record.turn_seconds = time.perf_counter() - turn_started
            record.live_message_count = len(agent.messages)
            record.tool_calls = _tool_calls_in_last_turn(agent, record.live_message_count)

        logger.info(
            "%-7s %-18s %6.1fs  msgs=%-4d calls=%d",
            config.name,
            turn.label,
            record.turn_seconds,
            record.live_message_count,
            len(collector.calls),
        )

    collector.wall_seconds = time.perf_counter() - run_started
    collector.plugin_counters = _collect_plugin_counters(agent, config)

    recorded = _stop_graph_recording(recorder)
    if recorded:
        collector.plugin_counters.setdefault("graph", {})["per_turn"] = recorded

    # Scored here rather than in the report so the accuracy figure travels with the raw
    # results and can be re-examined without a re-run.
    collector.accuracy = accuracy.score_run([turn.to_dict() for turn in collector.turns])
    logger.info(
        "%-11s accuracy: %.1f%% weighted, %d/%d turns materially correct",
        config.name,
        collector.accuracy["weighted_accuracy"] * 100,
        collector.accuracy["turns_materially_correct"],
        collector.accuracy["turns_scored"],
    )

    return collector


def _tool_calls_in_last_turn(agent: Agent, _count: int) -> list[str]:
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
    resume_at: int | None = None,
) -> dict[str, list[RunCollector]]:
    """Run the named configurations ``repeats`` times each.

    Parallel by default. The configurations are independent agents with independent
    histories, so running them concurrently changes nothing about the tokens they assemble
    or the answers they give — and it takes the wall time of a five-configuration
    comparison from roughly fifteen minutes to the length of its slowest member.

    The cost is that latency figures become throughput-under-contention rather than
    isolated latency: five agents sharing a Bedrock account queue behind each other. Token
    and accuracy numbers are unaffected, which is why parallel is the sensible default and
    ``--sequential`` exists for when the timing columns are the point.

    Repeats are interleaved rather than grouped, so drift in service latency over a long
    session spreads across configurations instead of landing on whichever ran last.
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
                        resume_at=resume_at,
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
                    resume_at=resume_at,
                )
                results[name].append(collector)
                await asyncio.sleep(2)  # let throttling budgets recover between runs

    return results
