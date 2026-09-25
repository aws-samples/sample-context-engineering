"""Build and drive one LangChain v1 ``create_agent`` per configuration, and measure it.

This is the only module of the harness that is a rewrite rather than a copy. Everything the
comparison is *about* -- the script, the mocked tools, the ground truth, the scoring, the rates --
comes over from ``validation/community-plugin-A-B-D`` unchanged; what changes here is the four
mechanical facts of driving a LangGraph agent instead of a Strands one:

1. **The agent is a compiled graph, not an object with a message list.** History lives in the
   checkpointer, so a 60-turn conversation is 60 ``invoke`` calls against one ``thread_id`` rather
   than 60 calls on one mutable ``Agent``. An ``InMemorySaver`` is not optional here: the relevance
   middleware clears its own tool exchanges with ``RemoveMessage(REMOVE_ALL_MESSAGES)`` on
   ``after_agent``, which is a state write and needs somewhere to land.
2. **The strategies are ``AgentMiddleware``, and order is a list.** The Strands stack let the graph
   move its own handler to index zero; here the list position *is* the nesting. First entry is
   outermost for ``wrap_model_call``, so the ``all`` arm reads graph, disclosure, relevance -- the
   same order the Strands arm produced by construction.
3. **Measurement is a middleware too.** The Strands collector was registered into
   ``InvokeModelStage`` after construction so it ran innermost and observed the projection. The
   equivalent is :class:`MetricsMiddleware` placed LAST in the list, which makes it the innermost
   ``wrap_model_call`` and therefore the one that sees what actually goes on the wire.
4. **Usage comes off the response message, not off a stream.** ``AIMessage.usage_metadata`` carries
   input, output and the cache breakdown, so ``metrics._absorb_usage`` -- written against Strands'
   streamed events -- is bypassed by :func:`_absorb_usage` here. The ``ModelCallRecord`` it fills is
   the copied dataclass, unchanged, which is what keeps ``report.py`` and the cost model reusable.

Every counter key is deliberately the one the Strands harness emits -- ``disclosure``, ``offloader``,
``graph``, ``embedding_cost``, ``rerank_observed``, ``registered_tools`` -- because the cost model
sums rerank spend from ``offloader`` and a renamed key would silently price relevance filtering at
zero.

**Nothing in this module invokes Bedrock at import or at construction time.** Building an arm builds
clients and middleware and nothing else, which is what lets the whole wiring be tested against a
fake chat model for no money. See ``tmp/`` smoke tests and the README.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from context_core.graph import EmbeddingSimilarityMatcher
from context_core.relevance import BedrockReranker, FileStore
from langgraph_context_graph import ContextGraphMiddleware
from langgraph_progressive_tool_disclosure import ProgressiveToolDisclosureMiddleware
from langgraph_relevance_filter import RelevanceFilterMiddleware

from . import accuracy, config, metrics, scenario, tools
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
from .metrics import ModelCallRecord, RunCollector, install_log_tagging

logger = logging.getLogger(__name__)


# --- AWS plumbing -------------------------------------------------------------------


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
    """Timeouts sized for parallel mode, identical to the Strands harness's.

    The default read timeout expires on the heaviest turn when several configurations run
    concurrently: a turn that pulls a 60k-character document and then makes several retrieval calls
    is slow on its own, and all of them share one account's throughput. A lost turn scores as an
    accuracy failure and inflates the very variance the repeats exist to measure -- a harness
    artefact reported as a strategy result.
    """
    return BotocoreConfig(
        read_timeout=180,
        connect_timeout=20,
        retries={"max_attempts": 5, "mode": "adaptive"},
    )


def _agent_model(session: boto3.Session) -> Any:
    """Build the chat model the arm drives, tagged so its invocations are separable in the log group.

    ``ChatBedrockConverse`` rather than ``init_chat_model("bedrock:…")``: the harness needs to pass
    its OWN boto3 client so that the log-tagging hooks, the timeouts and the credential resolution
    are the same ones the reranker and the embedder use. The string form builds a client internally
    and gives no handle to tag.

    **Constructing this does not call Bedrock.** A client is a local object; the first network call
    happens when the graph invokes the model, which only ``run_configuration`` does.

    Args:
        session: The run's boto3 session.

    Returns:
        The configured chat model, not yet invoked.
    """
    from langchain_aws import ChatBedrockConverse

    client = session.client("bedrock-runtime", config=_client_config())
    install_log_tagging(client, role="agent")
    # No temperature, matching the Strands harness: Opus 4.8 rejects the parameter outright
    # ("`temperature` is deprecated for this model"). Determinism across configurations comes from a
    # fixed scenario and fixed payloads, not from a sampling knob.
    return ChatBedrockConverse(
        client=client,
        model=AGENT_MODEL_ID,
        max_tokens=config.MAX_OUTPUT_TOKENS,
    )


# --- Metered auxiliaries ------------------------------------------------------------


class _MeteredReranker(BedrockReranker):
    """Counts the relevance filter's rerank traffic in the unit Bedrock bills.

    Subclasses ``context_core.relevance.BedrockReranker`` -- the public class the middleware accepts
    through its ``config["reranker"]`` -- so the count is taken at the contract boundary rather than
    by reaching into the package.

    A search unit is one query against up to 100 documents, which is exactly the batch the reranker
    pages into, so the count is ``ceil(chunks / max_sources_per_query)`` per call. Counted before
    the call delegates: a failure halfway through still consumed what it sent.

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
    ``_invoke`` receives only the cache misses, deduplicated. Metering the outer method would bill a
    session for vectors it never paid for -- precisely the claim the cache exists to make.

    Tokens are estimated from characters, not read from the response: Cohere's embed API returns no
    usage block. The estimate uses the same divisor as the rest of the harness.

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


RELEVANCE_RETRIEVAL_TOOL = os.environ.get("VALIDATION_RELEVANCE_RETRIEVAL_TOOL", "1") != "0"
"""Whether the relevance filter stores raw content and registers ``retrieve_all_context``.

On by default, matching the middleware's own default, in EVERY arm -- the combined one included, as
in the Strands harness (see :func:`build_middleware`, where the graph also gets the filter's stash).
Set ``VALIDATION_RELEVANCE_RETRIEVAL_TOOL=0`` to measure the excerpt alone, in every arm that
carries the filter.
"""


_GRAPH_STATE_MODULES = ("context_core.graph.state",)
"""Module whose dataclasses the checkpointer is told to accept (the classes, via :func:`_graph_state_types`).

**A future-breakage fix, not cosmetics.** The graph middleware persists its state as
``context_core.graph.state`` dataclasses -- ``_GraphState``, ``Card``, ``Link``, ``ToolPair``,
``TurnChoice`` -- and LangGraph's msgpack serde warns on deserializing a type it was not told
about, stating plainly that it "will be blocked in a future version". A blocked deserialize would
not warn, it would lose the graph on every restore, and the graph arm would silently measure a
conversation that starts from scratch every turn.

Named here rather than inside the middleware package because this is the CHECKPOINTER's allowlist:
it belongs to whoever constructs the saver, which is this harness.
"""


def _graph_state_types() -> tuple[type, ...]:
    """Every class ``context_core.graph.state`` defines -- what a checkpoint of the graph holds.

    The serde's allowlist is keyed by ``(module, class name)``: a bare module name matches nothing, so
    passing one makes the allowlist strict AND empty, and every graph type is then blocked on restore
    (the graph arm would silently restart from scratch each turn). Passing the classes themselves lets
    the serde normalize them to the right keys.
    """
    import inspect as _inspect

    from context_core.graph import state as graph_state

    return tuple(
        each
        for each in vars(graph_state).values()
        if _inspect.isclass(each) and each.__module__ == graph_state.__name__
    )


def _checkpointer() -> InMemorySaver:
    """Return the run's checkpointer, with the graph's state types on its allowlist.

    The saver is what makes 60 invokes one conversation: history lives here rather than on an agent
    object. It is also where the relevance middleware's ``RemoveMessage(REMOVE_ALL_MESSAGES)`` write
    lands when it clears its own tool exchanges at the end of a turn.

    In-memory rather than file-backed on purpose. A run is one process and the JSON is the artefact;
    a durable checkpoint would add a resume path this harness has nothing to measure on, and a stale
    one would let a replay inherit the previous replay's history and be scored on a context it did
    not build.
    """
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_graph_state_types()))
    except Exception:  # noqa: BLE001 - an allowlist is an optimisation, not a requirement
        logger.debug("could not register the graph state modules with the serde", exc_info=True)
        return InMemorySaver()


# NOTE: an earlier version of this harness carried an ``AsyncContextGraphMiddleware`` subclass that
# supplied ``awrap_model_call`` by running the sync projection on a worker thread, because
# ``ContextGraphMiddleware`` used to implement only the sync ``wrap_model_call`` and LangChain does not
# bridge a sync hook into an ``ainvoke`` run (it raises ``NotImplementedError``). That gap is now closed
# in the package itself: ``ContextGraphMiddleware`` ships a native ``awrap_model_call``, so the harness
# uses the real middleware directly and the workaround was removed. The stack still runs under
# ``ainvoke`` because the relevance filter's reranker protocol is genuinely async.


# --- Measurement middleware ---------------------------------------------------------


def _absorb_usage(record: ModelCallRecord, message: AIMessage) -> None:
    """Copy a response's usage onto ``record``.

    Replaces ``metrics._absorb_usage``, which reads Strands' streamed usage events. LangChain
    normalises the provider's numbers onto ``AIMessage.usage_metadata`` instead, with the cache
    breakdown under ``input_token_details`` -- so this is a different read of the same quantities and
    the record it fills is the copied dataclass, untouched.

    Every field is read defensively. A provider that reports no usage leaves the ``usage_*`` columns
    ``None``, which the report renders as a gap rather than as a zero -- and a zero would read as
    "this call was free".

    Args:
        record: The record to fill. Mutated in place.
        message: The response message. Left unmodified.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    record.usage_input_tokens = usage.get("input_tokens")
    record.usage_output_tokens = usage.get("output_tokens")
    record.usage_total_tokens = usage.get("total_tokens")

    details = usage.get("input_token_details") or {}
    # Only set when the provider reported it: see the docstring on why a default of 0 is wrong.
    if "cache_read" in details:
        record.cache_read_tokens = details["cache_read"]
    if "cache_creation" in details:
        record.cache_write_tokens = details["cache_creation"]

    metadata = getattr(message, "response_metadata", None) or {}
    record.stop_reason = metadata.get("stopReason") or metadata.get("finish_reason")


class MetricsMiddleware(AgentMiddleware):
    """Records one :class:`~.metrics.ModelCallRecord` per model call, as the call goes out.

    Placed LAST in the middleware list so it is the innermost ``wrap_model_call``: it therefore sees
    the request the strategies produced -- the folded history, the trimmed tool list -- and not the
    pre-projection one. Getting this backwards would measure the baseline in every arm.

    ``wrap_model_call`` is the only hook that sees both the assembled request and the response, which
    is what lets one handler report inputs and usage for the same call.

    Attributes:
        collector: The run's collector, appended to on every call.
    """

    def __init__(self, collector: RunCollector) -> None:
        """Attach to ``collector`` without touching it."""
        super().__init__()
        self.collector = collector

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Measure one model call, then hand it on."""
        record = self._open(request)
        started = time.perf_counter()
        try:
            response = handler(request)
        except Exception as error:  # noqa: BLE001 - record and re-raise
            self._fail(record, error, started)
            raise
        return self._close(record, response, started)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """The async half. Required, not optional: the runner drives the agent with ``ainvoke``.

        LangChain raises ``NotImplementedError`` rather than bridging a middleware that defines only
        the sync hook, so a sync-only collector would record ZERO calls on every arm and the whole
        token column would come back empty. Measured: that is exactly what happened before this
        method existed.
        """
        record = self._open(request)
        started = time.perf_counter()
        try:
            response = await handler(request)
        except Exception as error:  # noqa: BLE001 - record and re-raise
            self._fail(record, error, started)
            raise
        return self._close(record, response, started)

    # --- shared body, so the two hooks cannot drift -------------------------------

    def _open(self, request: Any) -> ModelCallRecord:
        """Record what this call is about to send, and register it on the collector."""
        collector = self.collector
        messages = list(getattr(request, "messages", None) or [])
        bound = list(getattr(request, "tools", None) or [])

        record = ModelCallRecord(
            turn_index=collector._turn_index,
            call_index=collector._call_index,
            message_count=len(messages),
            message_chars=metrics._json_chars([_message_payload(each) for each in messages]),
            tool_spec_count=len(bound),
            tool_spec_chars=metrics._json_chars([_tool_payload(each) for each in bound]),
            system_prompt_chars=metrics._json_chars(_system_prompt_of(request)),
            tool_names_sent=[_tool_name_of(each) for each in bound],
        )
        collector._call_index += 1
        collector.calls.append(record)
        return record

    def _fail(self, record: ModelCallRecord, error: BaseException, started: float) -> None:
        """Record a call that raised before it answered."""
        record.error = f"{type(error).__name__}: {error}"
        self.collector.errors.append(record.error)
        record.model_seconds = time.perf_counter() - started

    def _close(self, record: ModelCallRecord, response: Any, started: float) -> Any:
        """Record the response's usage and return it untouched."""
        record.model_seconds = time.perf_counter() - started
        for message in _result_messages(response):
            if isinstance(message, AIMessage):
                _absorb_usage(record, message)
                break
        return response


def _system_prompt_of(request: Any) -> str:
    """Return the request's system prompt as text, however this LangChain version carries it."""
    prompt = getattr(request, "system_prompt", None)
    if isinstance(prompt, str):
        return prompt
    message = getattr(request, "system_message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _message_payload(message: Any) -> Any:
    """Return a message's wire-relevant content, for the character count.

    Role and content only. The point of the count is the bytes that ride on the call, and LangChain
    hangs a good deal of local bookkeeping -- ids, response metadata, per-provider extras -- off a
    message that Bedrock never sees. Counting those would inflate every arm equally and make the
    figures incomparable with the Strands harness's.
    """
    return {
        "role": getattr(message, "type", "?"),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", None) or [],
    }


def _tool_payload(bound: Any) -> Any:
    """Return a bound tool in the Converse shape, for the schema character count."""
    if isinstance(bound, dict):
        return bound
    try:
        return tools._converse_spec(bound)
    except Exception:  # noqa: BLE001 - a measurement must never break a run
        return {"name": _tool_name_of(bound), "description": getattr(bound, "description", "")}


def _tool_name_of(bound: Any) -> str:
    """Return a bound tool's name, whether it is a ``BaseTool`` or a raw declaration."""
    if isinstance(bound, dict):
        body = bound.get("function") if isinstance(bound.get("function"), dict) else bound
        return str(body.get("name", "?"))
    return str(getattr(bound, "name", "?"))


def _result_messages(response: Any) -> list[Any]:
    """Return the messages a ``wrap_model_call`` response carries, in order."""
    result = getattr(response, "result", None)
    if isinstance(result, list):
        return result
    if isinstance(response, list):
        return response
    return [response] if isinstance(response, AIMessage) else []


# --- Arm construction ---------------------------------------------------------------


def build_middleware(config: RunConfig, session: boto3.Session) -> list[AgentMiddleware]:
    """Assemble one configuration's middleware list, outermost first.

    Returns an empty list for the baseline, which is the point of the baseline: no middleware, no
    projection, no filtering, and the payloads enter the history whole.

    **Order is the composition.** LangChain nests ``wrap_model_call`` in list order, so the first
    entry is outermost. The ``all`` arm therefore reads graph, disclosure, relevance, which is the
    same nesting the Strands stack reached by having the graph move its own handler to index zero:
    the graph folds the history first, disclosure then folds the exchanges of every tool the call
    does not carry, and relevance acts on a tool result before either of them sees it.

    Args:
        config: The configuration being built. Metered clients are stashed on ``config.extra`` so
            the end-of-run counters can read them.
        session: The run's boto3 session, so every remote call a strategy makes is billed to the
            same account as the agent's own and never falls back on ambient credentials.

    Returns:
        The middleware to hand to ``create_agent``, outermost first. The metrics middleware is NOT
        included; :func:`build_agent` appends it so it lands innermost.
    """
    graph: ContextGraphMiddleware | None = None
    disclosure: ProgressiveToolDisclosureMiddleware | None = None
    relevance: RelevanceFilterMiddleware | None = None

    if config.relevance:
        # Namespaced by run tag as well as configuration: the tag is what keeps two runs executing
        # at the same time from writing into each other's stored sub-blocks and serving the wrong
        # content back through retrieve_all_context.
        storage_root = ARTIFACTS_DIR / (metrics.RUN_TAG or "untagged") / config.name
        storage_root.mkdir(parents=True, exist_ok=True)

        reranker = _MeteredReranker(
            model_id=RERANK_MODEL_ID,
            boto_session=session,
            boto_client_config=_client_config(),
        )
        config.extra["_reranker"] = reranker

        # On in every arm, as in the Strands harness (RELEVANCE_RETRIEVAL_TOOL, default on, is the only
        # switch there). With disclosure installed the tool stays in the catalog and is loaded only for
        # the rare question that needs a whole result, exactly as the Strands harness leaves it.
        include_retrieval_tool = RELEVANCE_RETRIEVAL_TOOL
        config.extra["_relevance_retrieval_tool"] = include_retrieval_tool

        relevance = RelevanceFilterMiddleware(
            # File-backed rather than in-memory, so the run can inspect what was cut.
            store=FileStore(str(storage_root)),
            include_retrieval_tool=include_retrieval_tool,
            max_result_tokens=THRESHOLDS.max_result_tokens,
            config={
                "reranker": reranker,
                "relevance_threshold": THRESHOLDS.relevance_threshold,
                "chunk_tokens": THRESHOLDS.chunk_tokens,
                "preview_tokens": THRESHOLDS.preview_tokens,
            },
        )
        config.extra["_relevance"] = relevance

    if config.graph:
        matcher = _MeteredMatcher(EMBED_MODEL_ID, boto_session=session)
        config.extra["_matcher"] = matcher

        # One tuning for every arm the graph appears in, exactly as in the Strands harness: the
        # filter acts on a tool result before it enters the history and the graph acts on a history
        # that already exists, so the filter only makes the graph's input smaller -- it does not
        # change what folding a history should cost.
        tuning = GRAPH_TUNING
        config.extra["_graph_tuning"] = "unified"

        graph = ContextGraphMiddleware(
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
            matcher=matcher,
            # Always on, as in the Strands harness (runner.py there passes True explicitly).
            include_artifact_tool=True,
            # The filter's store as the second resolution layer, so a [ref: mem_N_...] the filter
            # minted resolves through expand_artifact too. Strands gets this from the ContextManager
            # Stash when one is installed; LangGraph has none, so it is wired explicitly.
            stash=relevance.stash if relevance is not None else None,
        )
        config.extra["_graph"] = graph
        config.extra["_graph_artifact_tool_dropped"] = False

        config.extra["_graph_stash"] = relevance is not None

    if config.disclosure:
        disclosure = ProgressiveToolDisclosureMiddleware(
            catalog_chars=THRESHOLDS.catalog_chars,
            ttl_cycles=THRESHOLDS.ttl_cycles,
            top_k=THRESHOLDS.top_k,
            # A retrieval tool must never need discovery: the model is told to use it in the
            # guidance text that replaces the payload, and a cycle spent loading it would be an
            # artefact of the harness rather than of the strategy. Derived from the graph's own
            # ``tools`` rather than hard-coded, so renaming one upstream cannot leave a stale name
            # here. ``list_accounts`` is the one literal: it is a domain tool of the scenario.
            always_available=[
                *(each.name for each in (graph.tools if graph is not None else ())),
                "list_accounts",
            ],
        )
        config.extra["_disclosure"] = disclosure

    # Outermost first. A None is simply absent, so a single-strategy arm gets a one-item list.
    return [each for each in (graph, disclosure, relevance) if each is not None]


def build_agent(config: RunConfig, session: boto3.Session, collector: RunCollector) -> Any:
    """Compile the agent under test, with the measurement middleware innermost.

    **This makes no network call.** It builds clients, middleware and a graph; the first Bedrock
    request happens on the first ``invoke``, which only :func:`run_configuration` issues.

    Args:
        config: The configuration under test.
        session: The run's boto3 session.
        collector: Receiver of the per-call measurements.

    Returns:
        The compiled agent, with an ``InMemorySaver`` so the conversation accumulates across turns.
    """
    middleware = build_middleware(config, session)
    suite = tools.all_tools()

    agent = create_agent(
        model=_agent_model(session),
        tools=suite,
        system_prompt=scenario.SYSTEM_PROMPT,
        # Appended last, so it is the innermost wrap_model_call and observes the projection the
        # strategies produced rather than the pre-projection request.
        middleware=[*middleware, MetricsMiddleware(collector)],
        # Required, not a convenience: the relevance middleware drops its own tool exchanges on
        # after_agent via RemoveMessage, which is a state write with nowhere to land without one.
        # It is also what makes 60 invokes one conversation instead of 60 fresh ones.
        checkpointer=_checkpointer(),
    )

    config.extra["_registered_tools"] = len(suite) + sum(
        len(getattr(each, "tools", ()) or ()) for each in middleware
    )
    return agent


# --- Counters -----------------------------------------------------------------------


def _graph_state_of(state: Any) -> Any:
    """Return the graph's own state out of the agent's, or ``None`` when the arm has no graph."""
    values = getattr(state, "values", None)
    if isinstance(values, dict):
        return values.get("context_graph")
    return state.get("context_graph") if isinstance(state, dict) else None


def _collect_plugin_counters(
    config: RunConfig,
    collector: RunCollector,
    state: Any,
) -> dict[str, Any]:
    """Read the strategy-specific evidence after a run.

    These counters are the difference between "tokens went down" and "the strategy did what it
    claims". Every read is defensive: a missing one should blank a field, not fail a run.

    The key names are the ones the Strands harness emits, so the whole reporting path renders a run
    from either harness through the same cost model. ``offloader`` in particular keeps its name even
    though no offloader exists here: it is the key the cost model sums rerank spend from, and a new
    name would price relevance filtering at zero.

    **One counter is derived rather than read.** The LangGraph disclosure middleware keeps
    ``loaded_tools`` in graph state and its premature-cancellation tally and summary usage on the
    instance, like the Strands plugin's per-agent object, but keeps no search tally. Searches are
    therefore counted from the recorded ``find_tools`` invocations, the same event seen from the
    harness side.

    Args:
        config: The configuration, carrying the middleware and metered clients on ``extra``.
        collector: The run's collector, for the tool-invocation counts.
        state: The agent's final state snapshot, or ``None`` when the run never got one.

    Returns:
        The counters, one section per strategy that was installed.
    """
    counters: dict[str, Any] = {}
    tool_uses: dict[str, int] = {}
    for turn in collector.turns:
        for name in turn.tool_calls:
            tool_uses[name] = tool_uses.get(name, 0) + 1

    if config.extra.get("_disclosure") is not None:
        disclosure_mw = config.extra["_disclosure"]
        try:
            values = getattr(state, "values", None) if state is not None else None
            loaded = dict((values or {}).get("loaded_tools") or {}) if isinstance(values, dict) else {}
            counters["disclosure"] = {
                "searches": tool_uses.get("find_tools", 0),
                "loads": tool_uses.get("get_tool_details", 0),
                "loaded_tools_at_end": sorted(loaded),
                "loaded_count_at_end": len(loaded),
                "premature_cancellations": disclosure_mw.premature_cancellations,
                # The default summarizer is the agent's own model, as in the Strands plugin;
                # compare.py bills these tokens at the agent's rates.
                "summary_usage": dict(disclosure_mw.summary_usage),
            }
        except Exception as error:  # noqa: BLE001
            counters["disclosure"] = {"error": f"state unavailable: {error}"}

    relevance = config.extra.get("_relevance")
    if relevance is not None:
        # Named ``offloader`` on purpose: see the docstring. ``fired`` is read from the metered
        # reranker rather than from a private attribute of the middleware -- if the filter never
        # engaged, nothing was reranked, and that is itself a finding.
        reranker = config.extra.get("_reranker")
        counters["offloader"] = {
            "preview_strategy": "relevance",
            "search_units": getattr(reranker, "search_units", 0) or 0,
            "fired": bool(getattr(reranker, "rerank_calls", 0)),
            "retrieval_tool_registered": config.extra.get("_relevance_retrieval_tool"),
        }

    graph_state = _graph_state_of(state) if state is not None else None
    if config.extra.get("_graph") is not None:
        # The graph's evidence that it decided rather than merely ran. Without these, "tokens went
        # down" and "the ladder collapsed to one rung" read identically in the report.
        try:
            if graph_state is None:
                counters["graph"] = {"error": "no context_graph in final state"}
            else:
                choice = graph_state.choice
                cards = graph_state.cards
                dialogue = {"full": 0, "description": 0, "title": 0}
                # ``title`` on the evidence axis is not a rung of the ladder: it means the selection
                # did not address the Card, so nothing of it reached the call.
                evidence = {"full": 0, "description": 0, "title": 0}
                for title in cards:
                    card_choice = None if choice.full_pass else choice.by_title.get(title)
                    dialogue["full" if card_choice is None else card_choice.dialogue] += 1
                    evidence["full" if card_choice is None else card_choice.evidence] += 1
                selected = choice.selected
                counters["graph"] = {
                    "addressed": len(cards) if selected is None else len(selected),
                    "unaddressed": 0 if selected is None else len(cards) - len(selected),
                    "cards": len(cards),
                    "subject_cards": sum(1 for card in cards.values() if card.kind == "subject"),
                    "artifact_cards": sum(1 for card in cards.values() if card.kind == "artifact"),
                    "links": sum(len(edges) for edges in graph_state.links.values()),
                    "turn": graph_state.turn,
                    "full_pass_at_end": choice.full_pass,
                    "dialogue_at_end": dialogue,
                    "evidence_at_end": evidence,
                    "reuse_at_end": len(graph_state.reuse),
                    "vectors_cached": len(graph_state.vectors),
                    "per_turn_telemetry": "unavailable (the middleware logs failures only)",
                }
        except Exception as error:  # noqa: BLE001
            counters["graph"] = {"error": f"state unavailable: {error}"}

    matcher = config.extra.get("_matcher")
    if matcher is not None:
        counters["embedding_cost"] = {
            "calls": matcher.embed_calls,
            "texts": matcher.embed_texts,
            "input_tokens": matcher.embed_tokens,
            "model_id": EMBED_MODEL_ID,
        }

    reranker = config.extra.get("_reranker")
    if reranker is not None:
        counters["rerank_observed"] = {
            "calls": reranker.rerank_calls,
            "search_units": reranker.search_units,
            "documents": reranker.rerank_documents,
            "model_id": RERANK_MODEL_ID,
        }

    counters["registered_tools"] = config.extra.get("_registered_tools", 0)
    if "_graph_tuning" in config.extra:
        counters["graph_tuning"] = config.extra["_graph_tuning"]
    if "_graph_artifact_tool_dropped" in config.extra:
        counters["graph_artifact_tool_dropped"] = config.extra["_graph_artifact_tool_dropped"]

    messages = []
    values = getattr(state, "values", None) if state is not None else None
    if isinstance(values, dict):
        messages = values.get("messages") or []
    counters["live_messages_at_end"] = len(messages)
    counters["framework"] = "langgraph"
    return counters


# --- Driving a run ------------------------------------------------------------------


def _tool_calls_of(messages: list[Any], since: int) -> list[str]:
    """Names of the tools invoked in the messages appended since index ``since``.

    Read off ``ToolMessage.name`` rather than off the assistant's ``tool_calls``, so a call the
    strategies cancelled or folded away is not counted as an invocation that happened.
    """
    return [
        message.name or "?"
        for message in messages[since:]
        if isinstance(message, ToolMessage)
    ]


def _answer_of(messages: list[Any], since: int) -> str:
    """Return the text of the last assistant message appended since index ``since``.

    Only a trailing ``AIMessage`` counts. A turn that failed on a tool error ends on a tool result,
    and inventing an answer out of an earlier turn's message would score one turn with another
    turn's work.
    """
    for message in reversed(messages[since:]):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") in (None, "text")
                )
    return ""


def _messages_of(agent: Any, thread: dict[str, Any]) -> list[Any]:
    """Return the conversation the checkpointer currently holds for ``thread``."""
    try:
        return list(agent.get_state(thread).values.get("messages") or [])
    except Exception:  # noqa: BLE001 - a measurement must never break a run
        logger.debug("could not read state for thread", exc_info=True)
        return []


async def run_configuration(
    config: RunConfig,
    *,
    turn_limit: int | None = None,
    phases: tuple[str, ...] | None = None,
    total_turns: int | None = None,
) -> RunCollector:
    """Replay the scenario under one configuration and return its measurements.

    **This is the only function in the harness that spends money.** It issues one ``ainvoke`` per
    turn against one ``thread_id``, so the conversation accumulates in the checkpointer exactly as
    the Strands harness accumulated it on one ``Agent``.

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

    thread = {"configurable": {"thread_id": f"{config.name}-{uuid.uuid4().hex[:8]}"}}
    script = scenario.turns(limit=turn_limit, phases=phases, total=total_turns)
    logger.info(
        "=== %s: %d turns, %d tools ===", config.name, len(script), config.extra.get("_registered_tools", 0)
    )

    run_started = time.perf_counter()

    for index, turn in enumerate(script):
        record = collector.begin_turn(index, turn.label, turn.prompt)
        before = len(_messages_of(agent, thread))
        turn_started = time.perf_counter()
        try:
            await agent.ainvoke({"messages": [HumanMessage(turn.prompt)]}, thread)
        except Exception as error:  # noqa: BLE001 - one bad turn must not lose the run
            # The baseline is expected to arrive here once the history outgrows the model's context
            # window. That is the measurement: a configuration whose turns stop completing has
            # answered the question the comparison asks.
            record.error = f"{type(error).__name__}: {error}"
            collector.errors.append(f"{turn.label}: {record.error}")
            logger.warning("turn %s failed: %s", turn.label, record.error)
        finally:
            record.turn_seconds = time.perf_counter() - turn_started
            messages = _messages_of(agent, thread)
            record.live_message_count = len(messages)
            record.tool_calls = _tool_calls_of(messages, before)
            # Read from state rather than from the invoke's return value, so a turn that raised
            # after the model had already answered is still scored on what the model produced. An
            # output cut short at max_tokens is that case: it is not the same failure as an overflow,
            # and scoring its silence as materially wrong penalised exactly the arms that fold
            # context -- folding is what makes a model restate figures verbatim, and restating is
            # what runs an answer past the cap.
            answer = _answer_of(messages, before)
            if answer:
                record.response_text = answer
                record.response_chars = len(answer)
                if record.error is not None:
                    record.answer_truncated = True
                    logger.info(
                        "turn %s truncated | scoring the %d chars the model did produce",
                        turn.label,
                        len(answer),
                    )

        logger.info(
            "%-10s %-18s %6.1fs  msgs=%-4d calls=%d",
            config.name,
            turn.label,
            record.turn_seconds,
            record.live_message_count,
            len(collector.calls),
        )
        if len(record.tool_calls) > 20:
            # A turn this long is a loop until shown otherwise: name what it called, so the log alone
            # says which tool the model kept reaching for.
            counts = Counter(record.tool_calls).most_common(5)
            logger.warning("%-10s %-18s long turn | tool calls=%d | top=%s", config.name, turn.label,
                           len(record.tool_calls), counts)

    state = None
    try:
        state = agent.get_state(thread)
    except Exception:  # noqa: BLE001
        logger.debug("final state unavailable", exc_info=True)
    collector.plugin_counters = _collect_plugin_counters(config, collector, state)

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

    Parallel by default. The configurations are independent graphs with independent checkpointers, so
    running them concurrently changes nothing about the tokens they assemble or the answers they
    give. The cost is that latency figures become throughput-under-contention rather than isolated
    latency; token and accuracy numbers are unaffected.

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
