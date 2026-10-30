"""A persistent JSONL stdin/stdout bridge that drives the real OpenCode host through U1.

The TypeScript plugin under ``.opencode/plugins/`` is a thin interface: it reacts to the public
``tool.execute.after`` hook and the session lifecycle event hook, and speaks newline-delimited JSON
(JSONL) to *this* process over a child pipe. Every decision about admission, capability, baseline and
refusal lives here and in :mod:`context_core.opencode` / :mod:`context_core.runtime`; the plugin never
reimplements any of it.

This module is the only place in ``context_core`` that resolves AWS configuration, and it does so
exclusively from the environment — never from a hardcoded profile, region, model id or endpoint:

- ``AWS_PROFILE`` — **required**; the bridge refuses to start without it and never defaults a profile.
- ``AWS_REGION`` or ``AWS_DEFAULT_REGION`` — optional region, the first that is set.
- ``CONTEXT_RERANK_MODEL_ID`` — reranking model id (default ``amazon.rerank-v1:0``).
- ``CONTEXT_EXECUTION_MODEL_ID`` — the OpenCode agent model used for ``live`` admission; it has no
  default and is required on that path.
- ``CONTEXT_EXECUTION_ALLOWED_MODELS`` — comma-separated allowlist for the agent model used by ``live``
  admission; it is empty by default.
- ``CONTEXT_RELEVANCE_THRESHOLD`` / ``CONTEXT_CHUNK_TOKENS`` / ``CONTEXT_PREVIEW_TOKENS`` /
  ``CONTEXT_MAX_RESULT_TOKENS`` — the preview tuning and the gate threshold.
- ``CONTEXT_OPENCODE_TOOLS`` — comma-separated allowlist of tool classes eligible for Practice A.

A boto3 :class:`~boto3.session.Session` is created with the explicit profile and handed to
:class:`~context_core.relevance.BedrockReranker`; the session, the reranker and the preview builder are
constructed lazily on the first result that actually needs scoring, so a bridge that only ever sees
small results (or whose tools are not in the allowlist) resolves no credentials and makes no inference
call. The reranker is the real one — there is deliberately **no fake scorer** here.

The wire protocol is a request/response exchange, one JSON object per line:

- ``startup`` — publish Practice A over the configured coverage set (B/D ``unsupported``).
- ``admit`` — run the closed U1 execution admission (``offline``/``replay``/``live``) over the existing
  :mod:`context_core.contracts.execution` contracts and return a sanitized admission receipt
  (``admission_id``, ``policy_digest``, preflight, approval/budget/nonce provenance) or a closed
  refusal. No credential is resolved and no payload is transmitted here; ``live`` requires an explicit
  approval by a permitted actor, a configured budget, an allow-listed remote backend and model, a
  requested preflight and a single-use nonce. Admission never defaults to ``live``.
- ``transform`` — relevance-filter one oversized tool result; returns ``previewed`` (filtered preview),
  ``baseline`` (integral original preserved on a recoverable dependency failure), ``skipped`` (tool not
  covered or result under threshold) or ``refused`` (a closed seven-field envelope, no payload). A
  covered, oversized result would resolve Bedrock credentials and bill an inference call, so it is
  gated on an admitted ``live`` admission receipt: without one, the bridge makes no call and refuses
  closed. On a successful preview the integral original is preserved verbatim in the session namespace
  and the response carries its opaque ``reference_id`` so the owner can later retrieve it in full.
- ``retrieve`` — return a previously preserved baseline verbatim to its owning session, under
  authorization, namespace and lease; returns ``retrieved`` (authorized content) or a closed refusal
  (missing session/reference, non-owner, expired lease). Every attempt is tallied per session without
  recording any payload. Retrieval is read-only: it resolves no credential and makes no inference call.
- ``shutdown`` — discard a session namespace synchronously through the runtime.
- ``health`` — a payload-free liveness probe.
- ``graph_project`` / ``graph_expand_card`` / ``graph_find_context`` / ``graph_status`` — Practice D (context graph),
  opt-in through ``CONTEXT_GRAPH=on`` and answered ``skipped`` otherwise. ``graph_project`` takes the transcript OpenCode
  is about to send and answers which messages to leave out of that one request and what text to append; the other three
  are the two retrieval tools and the payload-free counters. Scoring uses the embedding model
  ``CONTEXT_GRAPH_EMBED_MODEL_ID`` (default ``cohere.embed-multilingual-v3``) and, like Practice A, only under an admitted
  ``live`` admission. See :mod:`context_core.opencode_graph`.

Every request may carry an opaque ``id`` (correlation) and an ``idempotency_key``; a retried key replays
the committed response rather than producing a second effect. Responses are always JSON-serialisable and
a refusal carries only the closed envelope — no payload, reference, size, hash or free-text message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import IO, Any

import boto3
from botocore.config import Config as BotocoreConfig

from .contracts.canonical import digest as _canonical_digest
from .contracts.errors import make_refusal
from .contracts.execution import (
    AdmissionPolicy,
    Approval,
    ExecutionAdmission,
    ExecutionBudget,
    ExecutionPreflight,
    ExecutionRequest,
    admit,
)
from .contracts.identity import ContractIdentity
from .graph.matcher import EmbeddingSimilarityMatcher, SimilarityMatcher
from .observability import MetricsRecorder
from .opencode import (
    OPENCODE_HOST,
    REQUIRED_PUBLIC_HOOKS,
    OpenCodeAdapter,
    OpenCodeAdapterOutcome,
    OpenCodeCoverageEvidence,
    OpenCodeCoverageSet,
)
from .relevance import (
    BedrockReranker,
    PreviewStats,
    ReadRequestError,
    RelevancePreview,
    RerankerError,
    has_read_options,
    read_content,
)
from .opencode_graph import GraphConfig, GraphConfigError, OpenCodeGraph
from .runtime import DependencyUnavailable

__all__ = [
    "DEFAULT_RERANK_MODEL_ID",
    "BridgeConfig",
    "BridgeConfigError",
    "MetricsRecorder",
    "OpenCodeBridge",
    "decode_request",
    "encode_response",
    "main",
    "normalize_tool_name",
    "run_diagnostic_fixture",
    "serve",
]

_LOGGER = logging.getLogger(__name__)

# -- Fixed identity of this bridge build ------------------------------------------------------------

DEFAULT_RERANK_MODEL_ID = "amazon.rerank-v1:0"
"""Default Bedrock reranking model id; overridden only by ``CONTEXT_RERANK_MODEL_ID``."""

_HOST_VERSION = "1.18.34"
"""The OpenCode host version this bridge is wired for."""

_ADAPTER_VERSION = "0.1.0"
"""The bridge/adapter build version published at startup."""

_CONTRACT_SCHEMA = {"schema": "u1"}
_CONTRACT_VERSION = "1.0"

# -- Preview tuning defaults (mirrors the langgraph binding, read from the environment) -------------

_DEFAULT_RELEVANCE_THRESHOLD = 0.5
_DEFAULT_CHUNK_TOKENS = 2_500
_DEFAULT_PREVIEW_TOKENS = 1_000
_DEFAULT_MAX_RESULT_TOKENS = 8_000
_CHARS_PER_TOKEN = 4

# Built-in OpenCode tool names are already pure classes. MCP names are normalized only for the two
# observed CE-60 server forms; arbitrary prefixes are never stripped.
_MCP_PREFIXES = ("mcp_ce60-fixture_", "ce60-fixture_")

# Only built-in OpenCode tool classes are covered by default. MCP classes must be named explicitly in
# CONTEXT_OPENCODE_TOOLS; normalizing an incoming MCP name never expands this allowlist implicitly.
_DEFAULT_TOOL_CLASSES = (
    "read",
    "grep",
    "glob",
    "list",
    "webfetch",
    "bash",
)

# -- Approved NFR retention values for a memory-only session namespace ------------------------------
# These are the operational lease/quota values the shared-context-contracts NFR profile approves for the
# memory-only path (TECH4: in-memory store per namespace/session, quotas/lease/discard mandatory). They
# are not host configuration, so they are fixed here rather than read from the environment. A namespace
# lives for the life of this process and is discarded synchronously on a session ``shutdown`` request.
_LEASE_TTL_SECONDS = 3_600.0
_HEARTBEAT_INTERVAL_SECONDS = 300.0
_ABANDONMENT_GRACE_SECONDS = 300.0
_MEMORY_QUOTA_BYTES = 64 * 1024 * 1024

_WORKSPACE_ID = "opencode-workspace"
"""Single logical workspace for this bridge process; sessions partition the namespace within it."""

_ADMISSION_DEADLINE_MS = 30_000
"""Deadline, in milliseconds, stamped on an execution-admission request. Admission does no I/O."""

_ADMISSION_POLICY_ID = "opencode-bridge-admission"
"""Stable identity folded into the admission policy digest recorded on every admission receipt."""


def normalize_tool_name(tool_name: str | None) -> str | None:
    """Normalize one incoming OpenCode/MCP tool name without consulting the allowlist.

    The function recognizes only ``ce60-fixture_<tool>`` and
    ``mcp_ce60-fixture_<tool>``. A name without either prefix is treated as a pure built-in or an
    explicitly configured class. The caller must still check the returned class against the explicit
    coverage set; normalization never broadens that set, and unknown or uncovered classes are skipped.
    """
    if not isinstance(tool_name, str):
        return None
    value = tool_name.strip()
    if not value:
        return None
    for prefix in _MCP_PREFIXES:
        if value.startswith(prefix):
            normalized = value[len(prefix) :]
            return normalized or None
    return value


class BridgeConfigError(ValueError):
    """A required environment variable is missing or malformed.

    Raised by :meth:`BridgeConfig.from_env` before any request is served. The message names the
    offending variable but never a credential value.
    """


@dataclass(frozen=True)
class BridgeConfig:
    """Immutable bridge configuration resolved exclusively from the environment.

    Attributes:
        aws_profile: The AWS profile name used to build the boto3 session. Required; never defaulted.
        aws_region: The AWS region, or ``None`` when neither ``AWS_REGION`` nor ``AWS_DEFAULT_REGION``
            is set (boto3 then resolves it from the profile/config).
        rerank_model_id: The Bedrock reranking model id or ARN.
        execution_model_id: The OpenCode agent model id used for ``live`` admission, or ``None`` when
            live execution is not configured.
        relevance_threshold: Minimum score, in ``[0.0, 1.0]``, for a chunk to enter the preview.
        chunk_tokens: Scoring granularity — the approximate token budget of one scored chunk.
        preview_tokens: Approximate token budget of the whole preview.
        max_result_tokens: Gate threshold; a result at or below this estimated token count is left
            untouched (``skipped``) rather than filtered.
        tool_classes: The explicitly configured tool classes eligible for Practice A relevance filtering. Incoming
            OpenCode/MCP names are normalized to one of these classes only at the coverage decision.
        execution_allowed_models: The Bedrock model ids explicitly allow-listed for a ``live`` execution
            admission. Empty by default, so ``live`` is refused fail-closed until a model is named;
            never defaulted to the rerank model id, so billing is only ever opened deliberately.
        metrics_file: Optional per-arm JSONL path for payload-free bridge metrics.
        graph: Practice D (context graph) configuration. Off unless ``CONTEXT_GRAPH`` is on, in which case every
            ``graph_*`` request answers ``skipped`` and nothing of Practice A changes.
    """

    aws_profile: str
    aws_region: str | None
    rerank_model_id: str
    execution_model_id: str | None
    relevance_threshold: float
    chunk_tokens: int
    preview_tokens: int
    max_result_tokens: int
    tool_classes: tuple[str, ...]
    execution_allowed_models: tuple[str, ...]
    metrics_file: str | None = None
    graph: GraphConfig = GraphConfig()

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BridgeConfig:
        """Build a configuration from ``env``, validating the required profile and numeric fields.

        Args:
            env: The environment mapping (usually ``os.environ``).

        Returns:
            The resolved configuration.

        Raises:
            BridgeConfigError: If ``AWS_PROFILE`` is missing/blank, a numeric variable is not a valid
                number, the relevance threshold is outside ``[0.0, 1.0]``, a token budget is not a
                positive integer, or the tool allowlist is empty.
        """
        profile = (env.get("AWS_PROFILE") or "").strip()
        if not profile:
            raise BridgeConfigError(
                "AWS_PROFILE is required: the bridge never defaults a profile for Bedrock access"
            )

        region = (env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "").strip() or None
        rerank_model_id = (env.get("CONTEXT_RERANK_MODEL_ID") or "").strip() or DEFAULT_RERANK_MODEL_ID
        execution_model_id = (env.get("CONTEXT_EXECUTION_MODEL_ID") or "").strip() or None

        threshold = _env_float(env, "CONTEXT_RELEVANCE_THRESHOLD", _DEFAULT_RELEVANCE_THRESHOLD)
        if not 0.0 <= threshold <= 1.0:
            raise BridgeConfigError("CONTEXT_RELEVANCE_THRESHOLD must be within [0.0, 1.0]")

        chunk_tokens = _env_positive_int(env, "CONTEXT_CHUNK_TOKENS", _DEFAULT_CHUNK_TOKENS)
        preview_tokens = _env_positive_int(env, "CONTEXT_PREVIEW_TOKENS", _DEFAULT_PREVIEW_TOKENS)
        max_result_tokens = _env_positive_int(env, "CONTEXT_MAX_RESULT_TOKENS", _DEFAULT_MAX_RESULT_TOKENS)

        raw_tools = env.get("CONTEXT_OPENCODE_TOOLS")
        tool_classes: tuple[str, ...]
        raw_tool_values = _DEFAULT_TOOL_CLASSES if raw_tools is None else raw_tools.split(",")
        explicit_tool_classes: list[str] = []
        for part in raw_tool_values:
            tool_class = part.strip()
            if tool_class and tool_class not in explicit_tool_classes:
                explicit_tool_classes.append(tool_class)
        if not explicit_tool_classes:
            if raw_tools is None:  # pragma: no cover - the built-in default is non-empty
                raise BridgeConfigError("the default OpenCode tool coverage is empty")
            raise BridgeConfigError("CONTEXT_OPENCODE_TOOLS was set but names no tool classes")
        tool_classes = tuple(explicit_tool_classes)

        raw_models = env.get("CONTEXT_EXECUTION_ALLOWED_MODELS")
        # Empty by default: a ``live`` admission is fail-closed until a model id is explicitly named,
        # so the bridge never opens billing on a model nobody allow-listed.
        execution_allowed_models = tuple(
            dict.fromkeys(part.strip() for part in (raw_models or "").split(",") if part.strip())
        )

        try:
            graph = GraphConfig.from_env(env)
        except GraphConfigError as error:
            raise BridgeConfigError(str(error)) from error

        return cls(
            aws_profile=profile,
            aws_region=region,
            rerank_model_id=rerank_model_id,
            execution_model_id=execution_model_id,
            relevance_threshold=threshold,
            chunk_tokens=chunk_tokens,
            preview_tokens=preview_tokens,
            max_result_tokens=max_result_tokens,
            tool_classes=tool_classes,
            execution_allowed_models=execution_allowed_models,
            metrics_file=(env.get("CONTEXT_METRICS_FILE") or "").strip() or None,
            graph=graph,
        )


def _model_provenance(config: BridgeConfig) -> dict[str, str | None]:
    """Return model identities without carrying credentials or tool payloads."""
    return {
        "execution_model_id": config.execution_model_id,
        "rerank_model_id": config.rerank_model_id,
    }


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    """Parse ``env[key]`` as a float, or return ``default`` when it is unset/blank."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise BridgeConfigError(f"{key} must be a number, got {raw!r}") from error


def _env_positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    """Parse ``env[key]`` as a positive int, or return ``default`` when it is unset/blank."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise BridgeConfigError(f"{key} must be an integer, got {raw!r}") from error
    if value < 1:
        raise BridgeConfigError(f"{key} must be a positive integer, got {value}")
    return value


# -- JSONL codec --------------------------------------------------------------------------------------


def decode_request(line: str) -> dict[str, Any]:
    """Decode one JSONL request line into a mapping.

    Args:
        line: A single line of input, with or without the trailing newline.

    Returns:
        The decoded request object.

    Raises:
        BridgeConfigError: Never. This raises :class:`ValueError` for malformed input.
        ValueError: If the line is blank, is not valid JSON, or does not decode to a JSON object.
    """
    stripped = line.strip()
    if not stripped:
        raise ValueError("empty request line")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ValueError(f"request is not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"request must be a JSON object, got {type(payload).__name__}")  # noqa: TRY004
    return payload


def encode_response(response: Mapping[str, Any]) -> str:
    """Encode one response mapping as a single JSONL line (newline-terminated).

    Args:
        response: A JSON-serialisable response object.

    Returns:
        The compact JSON encoding followed by a single ``"\\n"``.

    Raises:
        TypeError: If ``response`` holds a value the JSON encoder cannot represent.
    """
    return json.dumps(response, separators=(",", ":"), ensure_ascii=False) + "\n"


# -- The bridge -------------------------------------------------------------------------------------


@dataclass
class _Session:
    """Per-session state kept for the life of the bridge process.

    Attributes:
        namespace_id: The memory-only namespace opened for this session.
        baseline_reference_ids: The opaque reference ids of the integral baselines preserved for this
            session after a Practice A preview, so the full result can be retrieved verbatim later.
        rankings: For each preserved reference, the chunk indices by descending relevance recorded when
            its preview was built, so a ``max_chunks`` read hands back more chunks in relevance order
            without scoring the text a second time. Indices only, never content.
    """

    namespace_id: str
    baseline_reference_ids: set[str] = field(default_factory=set)
    rankings: dict[str, tuple[int, ...]] = field(default_factory=dict)


class OpenCodeBridge:
    """Drive OpenCode tool results and session lifecycle through the U1 OpenCode adapter.

    The bridge owns one :class:`~context_core.opencode.OpenCodeAdapter`, a lazily built relevance
    preview, a per-session namespace map and a small idempotency cache. It resolves AWS configuration
    only when a result actually needs scoring, and never reaches outside the configured profile.

    Args:
        config: The resolved bridge configuration.
    """

    def __init__(self, config: BridgeConfig, *, clock: Callable[[], float] | None = None) -> None:
        """Build the adapter and wire the lazy transform callback (see class docstring).

        ``clock`` is the monotonic clock the retention leases run on; it exists so a test can move time.
        """
        self._config = config
        self._contract_identity = ContractIdentity.of_schema(_CONTRACT_VERSION, _CONTRACT_SCHEMA)
        self._coverage = _build_coverage_set(config.tool_classes)
        self._tool_classes = frozenset(config.tool_classes)
        self._adapter = OpenCodeAdapter(
            self._contract_identity,
            self._coverage,
            transform_fn=self._transform,
            host_version=_HOST_VERSION,
            adapter_version=_ADAPTER_VERSION,
            clock=clock,
        )
        self._sessions: dict[str, _Session] = {}
        self._idempotency: dict[str, dict[str, Any]] = {}
        self._preview: RelevancePreview | None = None
        # Per-session retrieval tally (attempts/successes), kept for the life of the process. It counts
        # authorized retrievals without ever recording the retrieved payload, and survives even a
        # retrieve against a session that never opened a namespace (counted as a failed attempt).
        self._retrieval_counts: dict[str, dict[str, int]] = {}
        self._metrics = MetricsRecorder(config.metrics_file)
        self._last_rerank_attempt = False
        self._last_rerank_result = "not_attempted"
        self._last_preview_stats: PreviewStats | None = None

        # Execution admission (U1 ``context_core.contracts.execution``). The policy binds the live
        # contract/manifest/fixture digests this bridge holds, the single permitted remote backend, and
        # the explicitly allow-listed models. No credential or network is ever touched here — admission
        # is pure local policy. Admitted receipts are kept so a ``transform`` that would call Bedrock can
        # require one before resolving any credential.
        self._fixture_catalog_digest = _canonical_digest(
            {"fixture_catalog": "opencode-bridge", "tool_classes": sorted(config.tool_classes)}
        )
        manifest_digest = self._adapter.manifest.manifest_digest
        self._admission_policy = AdmissionPolicy(
            policy_digest=_canonical_digest(
                {
                    "policy": _ADMISSION_POLICY_ID,
                    "allowlisted_backends": ["remote-explicit"],
                    "allowlisted_models": sorted(config.execution_allowed_models),
                    "capability_manifest_digest": manifest_digest,
                    "fixture_catalog_digest": self._fixture_catalog_digest,
                    "contract_version": self._contract_identity.contract_version,
                    "contract_digest": self._contract_identity.contract_digest,
                }
            ),
            expected_contract_identity=self._contract_identity,
            expected_capability_manifest_digest=manifest_digest,
            expected_fixture_catalog_digest=self._fixture_catalog_digest,
            allowlisted_backends=frozenset({"remote-explicit"}),
            allowlisted_models=frozenset(config.execution_allowed_models),
        )
        self._admissions: dict[str, ExecutionAdmission] = {}

        # Practice D, off unless CONTEXT_GRAPH is on. The matcher is built on the first projection allowed to score, so
        # a bridge with the graph off (or without a live admission) resolves no credential and calls no embedding model.
        self._graph = OpenCodeGraph(config.graph, matcher_factory=self._build_graph_matcher)

    # -- lazy graph matcher ------------------------------------------------------------------------

    def _build_graph_matcher(self) -> SimilarityMatcher:
        """Build the graph's embedding matcher with the explicit, environment-provided profile.

        The client is built here, not by the matcher, so its retry budget is bounded: the matcher never raises (a failed
        embedding reads as "send everything"), but a long retry loop would hold the model call it sits in front of. Two
        attempts and short timeouts keep the worst case well inside the plugin's request timeout.

        Returns:
            The matcher over ``CONTEXT_GRAPH_EMBED_MODEL_ID``. Construction opens no connection.
        """
        session = boto3.Session(profile_name=self._config.aws_profile, region_name=self._config.aws_region)
        client = session.client(
            service_name="bedrock-runtime",
            config=BotocoreConfig(
                connect_timeout=5,
                read_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
                user_agent_extra="context-core",
            ),
        )
        return EmbeddingSimilarityMatcher(model_id=self._config.graph.embed_model_id, client=client)

    # -- lazy relevance preview --------------------------------------------------------------------

    def _resolve_preview(self) -> RelevancePreview:
        """Return the preview builder, constructing the boto3 session and reranker on first use.

        Deferring construction keeps a bridge that never filters a large result free of any AWS
        credential resolution or inference call. The reranker is the real
        :class:`~context_core.relevance.BedrockReranker`; the session is built with the explicit,
        environment-provided profile.

        Returns:
            The memoized preview builder.
        """
        if self._preview is None:
            session = boto3.Session(
                profile_name=self._config.aws_profile, region_name=self._config.aws_region
            )
            reranker = BedrockReranker(self._config.rerank_model_id, boto_session=session)
            self._preview = RelevancePreview(
                reranker,
                relevance_threshold=self._config.relevance_threshold,
                chunk_tokens=self._config.chunk_tokens,
                preview_tokens=self._config.preview_tokens,
            )
        return self._preview

    def _transform(self, neutral_input: Any) -> str:
        """The neutral Practice A relevance callback handed to the runtime.

        Receives ``{"text": str, "query": str}`` and returns the verbatim, budget-bounded preview. A
        scoring failure is converted to :class:`~context_core.runtime.DependencyUnavailable` so the
        runtime preserves the integral baseline instead of emitting a partial result.

        Args:
            neutral_input: The neutral payload — a mapping with ``text`` and ``query`` keys.

        Returns:
            The preview text.

        Raises:
            DependencyUnavailable: When the reranker is unavailable or returns an unusable response.
        """
        if not isinstance(neutral_input, Mapping):  # pragma: no cover - guarded by the caller
            raise DependencyUnavailable("local_process_unavailable")
        text = neutral_input.get("text")
        query = neutral_input.get("query") or ""
        if not isinstance(text, str) or not isinstance(query, str):  # pragma: no cover - defensive
            raise DependencyUnavailable("local_process_unavailable")
        self._last_rerank_attempt = True
        try:
            preview = self._resolve_preview()
            transformed, self._last_preview_stats = _run_sync(preview.build_with_stats(text, query))
        except RerankerError as error:
            self._last_rerank_result = "failure"
            raise DependencyUnavailable("reranker_unavailable") from error
        except Exception:
            self._last_rerank_result = "failure"
            raise
        self._last_rerank_result = "success"
        return transformed

    def _record_transform_metrics(
        self,
        *,
        session_id: str | None,
        tool_class: str,
        operation: str,
        input_bytes: int,
        output_bytes: int,
        filtered: bool = False,
        source_already_truncated: bool = False,
        fallback: bool = False,
    ) -> None:
        """Persist only the sanitized counters and flags for one transform."""
        counts = self._retrieval_counts.get(session_id or "", {"attempts": 0, "successes": 0})
        self._metrics.record_transform(
            tool_class=tool_class,
            operation=operation,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            filtered=filtered,
            source_already_truncated=source_already_truncated,
            fallback=fallback,
            rerank_attempt=self._last_rerank_attempt,
            rerank_result=self._last_rerank_result,
            retrieval_attempts=counts["attempts"],
            retrieval_successes=counts["successes"],
        )

    # -- request dispatch --------------------------------------------------------------------------

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Drive one decoded request to exactly one JSON-serialisable response.

        Dispatches on the request ``type``. A deterministic refusal carries a closed envelope; any
        unexpected failure becomes a redacted ``error`` response rather than crashing the server. A
        repeated ``idempotency_key`` replays the committed response without a second effect.

        Args:
            request: The decoded request object.

        Returns:
            The response object, always carrying the request ``id`` for correlation.
        """
        request_id = _opt_str(request.get("id"))
        idempotency_key = _opt_str(request.get("idempotency_key"))
        if idempotency_key is not None and idempotency_key in self._idempotency:
            return self._idempotency[idempotency_key]

        request_type = _opt_str(request.get("type"))
        try:
            if request_type == "startup":
                response = self._handle_startup(request)
            elif request_type == "admit":
                response = self._handle_admit(request)
            elif request_type == "transform":
                response = self._handle_transform(request)
            elif request_type == "heartbeat":
                response = self._handle_heartbeat(request)
            elif request_type == "retrieve":
                response = self._handle_retrieve(request)
            elif request_type == "shutdown":
                response = self._handle_shutdown(request)
            elif request_type == "graph_project":
                response = self._handle_graph_project(request)
            elif request_type == "graph_expand_card":
                response = self._handle_graph_expand_card(request)
            elif request_type == "graph_find_context":
                response = self._handle_graph_find_context(request)
            elif request_type == "graph_status":
                response = {"operation": "graph_status", **self._graph.status()}
            elif request_type == "health":
                response = {"operation": "healthy", "adapter_version": _ADAPTER_VERSION}
            else:
                response = {"operation": "error", "code": "UNKNOWN_REQUEST_TYPE"}
        except Exception:  # the server must never crash on a single bad request
            _LOGGER.exception("opencode_bridge.handle_failed", extra={"request_id": request_id or ""})
            response = {"operation": "error", "code": "BRIDGE_ERROR"}

        response["id"] = request_id
        response["type"] = "response"
        if idempotency_key is not None:
            response["idempotency_key"] = idempotency_key
            self._idempotency[idempotency_key] = response
        return response

    def _handle_startup(self, _request: Mapping[str, Any]) -> dict[str, Any]:
        """Publish Practice A over the configured coverage set, with B/D unsupported."""
        outcome = self._adapter.startup(
            public_hook_set=REQUIRED_PUBLIC_HOOKS,
            adapter_version=_ADAPTER_VERSION,
            capability_manifest_digest=self._adapter.manifest.manifest_digest,
        )
        if outcome.status == "refused":
            return _refusal_response(outcome.refusal)
        metadata = (outcome.result.metadata if outcome.result is not None else None) or {}
        response: dict[str, Any] = {
            "operation": "started",
            "host": metadata.get("host"),
            "host_version": metadata.get("host_version"),
            "adapter_version": _ADAPTER_VERSION,
            "capabilities": metadata.get("capabilities", []),
            "manifest_digest": self._adapter.manifest.manifest_digest,
            "model_provenance": _model_provenance(self._config),
            "tool_classes": sorted(self._tool_classes),
        }
        if self._graph.enabled:
            # Only when on, so a graph-off startup response is byte-for-byte what Practice A alone published.
            response["graph"] = self._graph.config.summary()
        return response

    def _handle_admit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run the closed U1 execution admission and return a sanitized receipt or a closed refusal.

        The bridge binds the authoritative digests it holds (contract identity, capability manifest and
        fixture catalog) onto the :class:`~context_core.contracts.execution.ExecutionRequest`; everything
        else — the execution mode, approval, budget, preflight intent and single-use nonce — comes from
        the request, which the plugin only ever populates for ``live`` when the operator set the explicit
        environment. The live request model is read from ``CONTEXT_EXECUTION_MODEL_ID`` and is kept
        separate from the reranker model used by Practice A. Admission is pure local policy: no
        credential is resolved and no payload is transmitted on any path. An admitted receipt is retained
        so a later ``transform`` can require it; the response carries ``admission_id``/``policy_digest``/
        preflight, model and approval/budget/nonce provenance, but never a credential or payload. A
        refusal carries only the closed envelope.

        Args:
            request: The decoded ``admit`` request.

        Returns:
            The sanitized admission response, or a closed refusal response.
        """
        corr = _opt_str(request.get("id")) or secrets.token_hex(8)
        mode = _opt_str(request.get("execution_mode")) or "offline"

        approval_id = _opt_str(request.get("approval_id"))
        approval = Approval(
            status="explicit" if approval_id else "absent",
            approval_id=approval_id,
            actor_class=_opt_str(request.get("actor_class")),  # type: ignore[arg-type]
        )

        ceiling = _opt_float(request.get("budget_ceiling"))
        unit = _opt_str(request.get("budget_unit"))
        budget = ExecutionBudget(
            status="configured" if ceiling is not None and unit else "absent",
            ceiling_units=ceiling,
            currency_or_unit=unit,
        )

        nonce = _opt_str(request.get("nonce")) or secrets.token_hex(16)
        preflight = ExecutionPreflight(
            nonce=nonce,
            requested=_opt_bool(request.get("preflight_requested"), default=mode == "live"),
        )

        # Offline/replay stay on a sanitized local backend. Live admission names the OpenCode agent
        # model, while the reranker remains a separate Bedrock model used only by ``_resolve_preview``.
        execution_model_id = self._config.execution_model_id
        model_id = self._config.rerank_model_id
        if mode == "live":
            if execution_model_id is None:
                return _refusal_response(
                    make_refusal("ADMISSION_POLICY_DENIED", "preflight", corr, capability="A")
                )
            model_id = execution_model_id
        backend_id = "remote-explicit" if mode == "live" else "local-scorer"

        execution_request = ExecutionRequest(
            request_id=corr,
            idempotency_key=_opt_str(request.get("idempotency_key")) or corr,
            idempotency_scope="opencode-admit",
            request_digest=_canonical_digest(
                {
                    "execution_mode": mode,
                    "backend_id": backend_id,
                    "model_id": model_id,
                    "approval_status": approval.status,
                    "budget_status": budget.status,
                    "nonce": nonce,
                }
            ),
            contract_identity=self._contract_identity,
            execution_mode=mode,  # type: ignore[arg-type]
            host=OPENCODE_HOST,
            host_version=_HOST_VERSION,
            model_id=model_id,
            backend_id=backend_id,  # type: ignore[arg-type]
            capability_manifest_digest=self._adapter.manifest.manifest_digest,
            fixture_catalog_digest=self._fixture_catalog_digest,
            approval=approval,
            budget=budget,
            preflight=preflight,
            deadline_ms=_ADMISSION_DEADLINE_MS,
        )

        admission = admit(execution_request, self._admission_policy, issued_at=time.time())
        if admission.status == "refused":
            return _refusal_response(admission.refusal)

        self._admissions[admission.admission_id] = admission
        return {
            "operation": "admitted",
            "admission_id": admission.admission_id,
            "execution_mode": mode,
            "policy_digest": admission.policy_digest,
            "model_provenance": _model_provenance(self._config),
            "capability_manifest_digest": admission.capability_manifest_digest,
            "fixture_catalog_digest": admission.fixture_catalog_digest,
            "contract_version": admission.contract_identity.contract_version,
            "contract_digest": admission.contract_identity.contract_digest,
            "preflight": {
                "credential_resolution": admission.preflight_result.credential_resolution,
                "payload_transmission": admission.preflight_result.payload_transmission,
            },
            "approval": {
                "status": admission.approval.status,
                "approval_id": admission.approval.approval_id,
                "actor_class": admission.approval.actor_class,
            },
            "budget": {
                "status": admission.budget.status,
                "ceiling_units": admission.budget.ceiling_units,
                "currency_or_unit": admission.budget.currency_or_unit,
            },
            "nonce": admission.nonce,
        }

    def _has_live_admission(self, admission_id: str | None) -> bool:
        """Report whether ``admission_id`` names a retained, admitted ``live`` admission receipt.

        A receipt only clears this gate when it was admitted and its preflight opened credential
        resolution — which admission grants exclusively for ``live``. Offline/replay receipts, a missing
        id, or any refused admission never open the Bedrock path.

        Args:
            admission_id: The admission id carried on the transform request, or ``None``.

        Returns:
            ``True`` when a credential-resolving live admission backs the id.
        """
        if not admission_id:
            return False
        admission = self._admissions.get(admission_id)
        return (
            admission is not None
            and admission.status == "admitted"
            and admission.preflight_result.credential_resolution == "permitted_after_admission"
        )

    def _handle_transform(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize/classify one tool result and record a payload-free terminal observation."""
        tool = _opt_str(request.get("tool"))
        session_id = _opt_str(request.get("session_id"))
        text = request.get("text")
        normalized = normalize_tool_name(tool)
        tool_class = normalized or "unknown"
        input_bytes = len(text.encode("utf-8")) if isinstance(text, str) else 0
        self._last_rerank_attempt = False
        self._last_rerank_result = "not_attempted"
        self._last_preview_stats = None

        def skipped(reason: str) -> dict[str, Any]:
            output_bytes = input_bytes
            self._record_transform_metrics(
                session_id=session_id,
                tool_class=tool_class,
                operation="skipped",
                input_bytes=input_bytes,
                output_bytes=output_bytes,
            )
            return {"operation": "skipped", "reason": reason, "tool_class": tool_class}

        if not isinstance(text, str) or not text:
            return skipped("no_text")
        if normalized is None or normalized not in self._tool_classes:
            # Unknown and uncovered names are explicit no-ops: they are never relevance-filtered.
            return skipped("tool_not_covered")
        if not session_id:
            return skipped("no_session")
        self._touch(session_id)
        if _approximate_tokens(text) <= self._config.max_result_tokens:
            return skipped("under_threshold")

        # Past the gate a covered, oversized result would build the boto3 session and bill a Bedrock
        # ``rerank`` call. That is only permitted under an admitted ``live`` execution admission whose
        # preflight opened credential resolution. Without that receipt the bridge resolves no credential,
        # makes no call, and refuses closed with no payload (NFR8.2). The default never resolves live.
        admission_id = _opt_str(request.get("admission_id"))
        if not self._has_live_admission(admission_id):
            self._record_transform_metrics(
                session_id=session_id,
                tool_class=tool_class,
                operation="refused",
                input_bytes=input_bytes,
                output_bytes=input_bytes,
            )
            corr = _opt_str(request.get("id")) or ""
            return _refusal_response(
                make_refusal("ADMISSION_POLICY_DENIED", "transform", corr, capability="A")
            )

        query = request.get("query")
        query_text = query if isinstance(query, str) else ""
        namespace_id = self._namespace_for(session_id)

        decision = self._adapter.issue_decision("A")
        binding = self._adapter.bind(
            decision,
            operation="transform",
            workspace_id=_WORKSPACE_ID,
            session_id=session_id,
            namespace_id=namespace_id,
            idempotency_key=_opt_str(request.get("idempotency_key")),
            request_id=_opt_str(request.get("id")),
        )
        try:
            outcome = self._adapter.process(
                binding, neutral_input={"text": text, "query": query_text}, baseline=text
            )
        except Exception:
            self._record_transform_metrics(
                session_id=session_id,
                tool_class=tool_class,
                operation="refused",
                input_bytes=input_bytes,
                output_bytes=input_bytes,
            )
            raise

        source_already_truncated = outcome.source_shape_id is not None
        operation = outcome.operation if outcome.operation in {"previewed", "baseline", "refused"} else "refused"
        output_bytes = input_bytes
        if outcome.content is not None:
            output_bytes = len(_as_text(outcome.content).encode("utf-8"))
        self._record_transform_metrics(
            session_id=session_id,
            tool_class=tool_class,
            operation=operation,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            filtered=operation == "previewed",
            source_already_truncated=source_already_truncated,
            fallback=operation == "baseline" and not source_already_truncated,
        )

        # On a successful preview the integral original is replaced in the transcript, so preserve it
        # verbatim in the session namespace and return its reference so it can be retrieved later. The
        # reference is returned only to the owning session; a fallback baseline is already integral and
        # needs no separate preservation.
        reference_id = None
        excerpt = None
        if outcome.operation == "previewed":
            reference_id = self._preserve_baseline(session_id, namespace_id, text)
            stats = self._last_preview_stats
            if stats is not None:
                excerpt = _excerpt_summary(stats)
                session = self._sessions.get(session_id)
                if reference_id is not None and session is not None:
                    session.rankings[reference_id] = stats.ranking
        return _transform_response(outcome, reference_id=reference_id, excerpt=excerpt)

    def _preserve_baseline(self, session_id: str, namespace_id: str, text: str) -> str | None:
        """Store the integral baseline for ``session_id`` and return its opaque reference, or ``None``.

        Best-effort: a refusal or any non-``stored`` terminal leaves the preview in place without a
        reference rather than failing the transform. The content is stored in the session's own
        memory-only namespace, so a later retrieve is an owner read.

        Args:
            session_id: The owning session.
            namespace_id: The session's namespace.
            text: The integral original result to preserve verbatim.

        Returns:
            The opaque reference id, or ``None`` when the baseline could not be stored.
        """
        store_binding = self._adapter.bind(
            self._adapter.issue_decision("A"),
            operation="store",
            workspace_id=_WORKSPACE_ID,
            session_id=session_id,
            namespace_id=namespace_id,
        )
        outcome = self._adapter.store_baseline(store_binding, content=text.encode("utf-8"))
        if outcome.operation == "stored" and outcome.reference_id is not None:
            session = self._sessions.get(session_id)
            if session is not None:
                session.baseline_reference_ids.add(outcome.reference_id)
            return outcome.reference_id
        return None

    def _touch(self, session_id: str | None) -> bool:
        """Renew the retention lease of ``session_id`` if it has a namespace; a missing one is a no-op.

        Every request that names a live session is proof of life. Without this a session's references stop
        resolving a few minutes after they were stored, however actively the session is used, because the
        memory-only lease expects the host to keep proving the session is alive.
        """
        session = self._sessions.get(session_id or "")
        return False if session is None else self._adapter.heartbeat(session.namespace_id)

    def _handle_heartbeat(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Renew a session's lease on the host's schedule (the plugin sends one while a session is open)."""
        session_id = _opt_str(request.get("session_id"))
        if not session_id:
            return {"operation": "skipped", "reason": "no_session"}
        return {"operation": "heartbeat", "renewed": self._touch(session_id)}

    def _handle_retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Retrieve an authorized baseline for a session, or refuse closed; count the attempt either way.

        Retrieval is read-only: it resolves no credential and makes no inference call. The attempt is
        tallied per session (success or failure) without ever recording the retrieved payload. A missing
        session or reference, or any store refusal (non-owner, expired lease, closed reference), yields a
        closed seven-field envelope that never confirms whether the reference exists.

        Args:
            request: The decoded ``retrieve`` request with ``session_id`` and ``reference_id``.

        Returns:
            A ``retrieved`` response carrying the authorized content, a ``skipped`` response when no
            session is named, or a closed refusal response.
        """
        corr = _opt_str(request.get("id")) or ""
        session_id = _opt_str(request.get("session_id"))
        reference_id = _opt_str(request.get("reference_id"))

        if not session_id:
            return {"operation": "skipped", "reason": "no_session"}
        if not reference_id:
            return _refusal_response(
                make_refusal("REQUEST_SCHEMA_INVALID", "retrieve", corr, capability="A")
            )

        session = self._sessions.get(session_id)
        if session is None:
            # No namespace was ever opened for this session, so there is nothing to retrieve. Count the
            # attempt and refuse closed without confirming whether any reference exists.
            self._count_retrieval(session_id, success=False)
            return _refusal_response(
                make_refusal("REFERENCE_EXPIRED_OR_DELETED", "retrieve", corr, capability="A")
            )

        self._touch(session_id)
        binding = self._adapter.bind(
            self._adapter.issue_decision("A"),
            operation="retrieve",
            workspace_id=_WORKSPACE_ID,
            session_id=session_id,
            namespace_id=session.namespace_id,
            request_id=_opt_str(request.get("id")),
        )
        outcome = self._adapter.retrieve(binding, reference_id=reference_id)
        if outcome.operation == "retrieved":
            content = _as_text(outcome.content)
            options = {
                key: request.get(key)
                for key in ("pattern", "line_range", "context_lines", "max_chunks", "max_tokens")
                if request.get(key) is not None
            }
            read_mode = "full"
            if has_read_options(**options):
                try:
                    content = read_content(
                        content,
                        default_max_tokens=self._config.max_result_tokens,
                        chunk_tokens=self._config.chunk_tokens,
                        ranking=session.rankings.get(reference_id),
                        **options,
                    )
                except ReadRequestError as error:
                    # The caller is the authorized owner and can fix the request: say how, quoting no
                    # content. The attempt delivered nothing, so it counts as a failed retrieval.
                    self._count_retrieval(session_id, success=False)
                    return {"operation": "invalid_request", "message": str(error)}
                read_mode = _read_mode(options)
            self._count_retrieval(session_id, success=True)
            return {
                "operation": "retrieved",
                "content": content,
                "content_digest": outcome.content_digest,
                "read_mode": read_mode,
            }
        self._count_retrieval(session_id, success=False)
        return _refusal_response(outcome.refusal)

    def _count_retrieval(self, session_id: str, *, success: bool) -> None:
        """Tally one retrieval attempt (and success) for ``session_id`` without recording any payload.

        Args:
            session_id: The session the retrieval was scoped to.
            success: Whether the retrieval returned authorized content.
        """
        counts = self._retrieval_counts.setdefault(session_id, {"attempts": 0, "successes": 0})
        counts["attempts"] += 1
        if success:
            counts["successes"] += 1
        self._metrics.record_retrieval(success=success)

    def _handle_shutdown(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Discard a session namespace synchronously, if one was opened for it."""
        session_id = _opt_str(request.get("session_id"))
        if not session_id:
            return {"operation": "skipped", "reason": "no_session"}
        # The graph's per-session state is request-scoped bookkeeping, not a retained baseline: forget it with the session.
        self._graph.discard(session_id)
        session = self._sessions.pop(session_id, None)
        if session is None:
            # Nothing was ever stored for this session; discard is a no-op with no namespace to touch.
            return {"operation": "discarded", "existed": False}

        binding = self._adapter.bind(
            self._adapter.issue_decision("A"),
            operation="shutdown",
            workspace_id=_WORKSPACE_ID,
            session_id=session_id,
            namespace_id=session.namespace_id,
            practice="none",
            idempotency_key=_opt_str(request.get("idempotency_key")),
            request_id=_opt_str(request.get("id")),
        )
        outcome = self._adapter.shutdown(binding)
        if outcome.operation == "refused":
            return _refusal_response(outcome.refusal)
        return {"operation": "discarded", "existed": True}

    # -- Practice D (context graph) ----------------------------------------------------------------

    def _graph_request(self, request: Mapping[str, Any]) -> tuple[str | None, bool, dict[str, Any] | None]:
        """Resolve what every ``graph_*`` request needs: the session, whether it may score, or a ready refusal.

        Returns:
            ``(session_id, scoring_allowed, early_response)``. A non-``None`` ``early_response`` is the whole answer
            (the graph is off, or no session is named).
        """
        if not self._graph.enabled:
            return None, False, {"operation": "skipped", "reason": "graph_disabled"}
        session_id = _opt_str(request.get("session_id"))
        if not session_id:
            return None, False, {"operation": "skipped", "reason": "no_session"}
        self._touch(session_id)
        # Scoring is the graph's only network use. It follows the same rule as Practice A's rerank: an admitted ``live``
        # receipt, or no call at all.
        return session_id, self._has_live_admission(_opt_str(request.get("admission_id"))), None

    def _handle_graph_project(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Project one model call's transcript; answer the message ids to leave out and the text to append."""
        session_id, scoring_allowed, early = self._graph_request(request)
        if early is not None or session_id is None:
            return early or {"operation": "skipped", "reason": "no_session"}
        transcript = request.get("messages")
        if not isinstance(transcript, list):
            return {"operation": "skipped", "reason": "no_messages"}
        return self._graph.project(session_id, transcript, scoring_allowed=scoring_allowed)

    def _handle_graph_expand_card(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run the ``expand_card`` tool over the session's graph."""
        session_id, _scoring, early = self._graph_request(request)
        if early is not None or session_id is None:
            return early or {"operation": "skipped", "reason": "no_session"}
        titles = request.get("titles")
        if isinstance(titles, str):
            wanted: list[str] = [titles]
        elif isinstance(titles, list):
            wanted = [title for title in titles if isinstance(title, str)]
        else:
            wanted = []
        return {"operation": "graph_result", "text": self._graph.expand_card(session_id, wanted)}

    def _handle_graph_find_context(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run the ``find_context`` tool over the session's graph."""
        session_id, scoring_allowed, early = self._graph_request(request)
        if early is not None or session_id is None:
            return early or {"operation": "skipped", "reason": "no_session"}
        need = request.get("need")
        tag = request.get("tag")
        text = self._graph.find_context(
            session_id,
            need if isinstance(need, str) else "",
            tag if isinstance(tag, str) and tag.strip() else None,
            scoring_allowed=scoring_allowed,
        )
        return {"operation": "graph_result", "text": text}

    def _namespace_for(self, session_id: str) -> str:
        """Return the namespace id for ``session_id``, opening one with the approved lease on first use."""
        session = self._sessions.get(session_id)
        if session is None:
            namespace = self._adapter.open_namespace(
                _WORKSPACE_ID,
                session_id,
                lease_ttl=_LEASE_TTL_SECONDS,
                heartbeat_interval=_HEARTBEAT_INTERVAL_SECONDS,
                abandonment_grace=_ABANDONMENT_GRACE_SECONDS,
                memory_quota=_MEMORY_QUOTA_BYTES,
            )
            session = _Session(namespace_id=namespace.namespace_id)
            self._sessions[session_id] = session
        return session.namespace_id


# -- response mapping -------------------------------------------------------------------------------


def _excerpt_summary(stats: PreviewStats) -> dict[str, Any]:
    """Summarize what a preview selection did, as plain numbers the host turns into a disclaimer.

    Line counts, chunk counts and the 1-indexed line spans the preview shows. No content is included.
    """
    return {
        "total_lines": stats.total_lines,
        "total_chunks": stats.total_chunks,
        "shown_chunks": len(stats.shown),
        "shown_spans": [[start, end] for start, end in stats.shown_spans],
    }


def _transform_response(
    outcome: OpenCodeAdapterOutcome, *, reference_id: str | None = None, excerpt: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Map an adapter outcome to a transform response.

    Only ``previewed`` and ``baseline`` carry content the plugin substitutes; a refusal carries the
    closed envelope and no payload, and the plugin leaves the result untouched. When a preview preserved
    the integral baseline, ``reference_id`` names it so the owner can retrieve the full result verbatim.
    """
    if outcome.operation in ("previewed", "baseline"):
        response: dict[str, Any] = {
            "operation": outcome.operation,
            "content": _as_text(outcome.content),
            "content_digest": outcome.content_digest,
        }
        if reference_id is not None:
            response["reference_id"] = reference_id
        if excerpt is not None:
            response["excerpt"] = excerpt
        return response
    if outcome.operation == "refused":
        return _refusal_response(outcome.refusal)
    # No other terminal is reachable for a transform; surface it as a redacted error, never a payload.
    return {"operation": "error", "code": "UNEXPECTED_OUTCOME"}  # pragma: no cover


def _read_mode(options: Mapping[str, Any]) -> str:
    """Name the view a retrieve request asked for, by the reference tool's precedence."""
    for key, mode in (
        ("line_range", "line_range"),
        ("pattern", "pattern"),
        ("max_chunks", "max_chunks"),
        ("context_lines", "head"),
        ("max_tokens", "bounded_full"),
    ):
        if key in options:
            return mode
    return "full"


def _refusal_response(refusal: Any) -> dict[str, Any]:
    """Wrap a closed refusal envelope in a response carrying only its seven fields."""
    if refusal is None:  # pragma: no cover - refused outcomes always carry an envelope
        return {"operation": "error", "code": "BRIDGE_ERROR"}
    return {"operation": "refused", "refusal": refusal.to_dict()}


def _as_text(content: Any) -> str:
    """Return ``content`` as text, decoding bytes as UTF-8 (replacing undecodable bytes)."""
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content if isinstance(content, str) else str(content)


def _approximate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` with the shared ``ceil(chars / 4)`` heuristic."""
    return -(-len(text) // _CHARS_PER_TOKEN)


def _opt_str(value: Any) -> str | None:
    """Return ``value`` when it is a string, else ``None``."""
    return value if isinstance(value, str) else None


def _opt_float(value: Any) -> float | None:
    """Return ``value`` coerced to ``float`` when it is a real number or numeric string, else ``None``.

    A ``bool`` is rejected (``True``/``False`` are not a budget ceiling), so an accidental boolean never
    reads as ``1.0``/``0.0``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _opt_bool(value: Any, *, default: bool) -> bool:
    """Return ``value`` as a bool, falling back to ``default`` when it is absent/unrecognised."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _build_coverage_set(tool_classes: tuple[str, ...]) -> OpenCodeCoverageSet:
    """Build a non-empty Practice A coverage set over ``tool_classes``.

    The six mandatory evidence digests are derived canonically from the covered tool classes, so the
    coverage identity is stable and well-formed. They assert the coverage composition the bridge
    publishes; the relevance scoring itself is the real reranker, never a fake.

    Args:
        tool_classes: The tool classes eligible for Practice A; must be non-empty.

    Returns:
        The coverage set backing the adapter's Practice A manifest.
    """
    facets = (
        "payload_integrity",
        "substitution",
        "retrieval",
        "metadata_preservation",
        "isolation",
        "fallback",
    )
    digests = {
        facet: _canonical_digest({"facet": facet, "tool_classes": sorted(tool_classes)}) for facet in facets
    }
    evidence = OpenCodeCoverageEvidence(
        payload_integrity_digest=digests["payload_integrity"],
        substitution_digest=digests["substitution"],
        retrieval_digest=digests["retrieval"],
        metadata_preservation_digest=digests["metadata_preservation"],
        isolation_digest=digests["isolation"],
        fallback_digest=digests["fallback"],
    )
    return OpenCodeCoverageSet(
        coverage_set_id="opencode-bridge-A",
        included_tool_classes=tuple(tool_classes),
        evidence=evidence,
        fixture_ids=tuple(f"fx-{name}" for name in tool_classes),
    )


def run_diagnostic_fixture(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Exercise MCP name matching and metrics locally without admission, AWS or a fake provider."""
    diagnostic_env = dict(env or os.environ)
    # The profile value is required by the normal live configuration contract, but the diagnostic keeps
    # every result below the gate and never constructs a boto3 session. This is a local label, not a
    # credential lookup or provider substitute.
    diagnostic_env.setdefault("AWS_PROFILE", "offline-fixture")
    # The fixture opts into the MCP classes it probes; production defaults remain built-ins only.
    diagnostic_env.setdefault(
        "CONTEXT_OPENCODE_TOOLS",
        "list_accounts,list_investment_transactions,read_aws_documentation",
    )
    diagnostic_env["CONTEXT_MAX_RESULT_TOKENS"] = "100000"
    config = BridgeConfig.from_env(diagnostic_env)
    bridge = OpenCodeBridge(config)
    requests = (
        "ce60-fixture_list_accounts",
        "mcp_ce60-fixture_list_investment_transactions",
        "read_aws_documentation",
        "mcp_ce60-fixture_not_covered",
    )
    observations: list[dict[str, str]] = []
    for index, tool in enumerate(requests):
        response = bridge.handle(
            {
                "type": "transform",
                "id": f"diagnostic-{index}",
                "tool": tool,
                "session_id": "diagnostic-session",
                "text": "offline fixture probe",
            }
        )
        observations.append(
            {
                "tool_class": str(response.get("tool_class", normalize_tool_name(tool) or "unknown")),
                "operation": str(response.get("operation", "error")),
            }
        )
    return {
        "mode": "diagnostic-fixture",
        "tool_classes": [item["tool_class"] for item in observations],
        "operations": [item["operation"] for item in observations],
        "metrics": bridge._metrics.summary(),
    }


# -- async-in-sync helper ---------------------------------------------------------------------------


def _run_sync(coro: Any) -> Any:
    """Run ``coro`` to completion from synchronous code.

    The server loop is synchronous while the core's preview is async. There is no running loop in the
    server thread, so :func:`asyncio.run` drives the coroutine on a fresh loop.

    Args:
        coro: The awaitable to drive.

    Returns:
        Its result; its exception propagates unchanged.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A running loop in this thread would be unexpected for the stdin server; fall back to a private
    # loop so the call still completes deterministically.
    import concurrent.futures  # local import: only reached off the normal synchronous path

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:  # pragma: no cover - rare path
        return pool.submit(asyncio.run, coro).result()


# -- server loop ------------------------------------------------------------------------------------


def serve(bridge: OpenCodeBridge, stdin: IO[str], stdout: IO[str]) -> None:
    """Serve JSONL requests from ``stdin``, writing one response line per request to ``stdout``.

    Reads until EOF. A line that fails to decode is answered with a redacted ``error`` response rather
    than terminating the loop, so one malformed line never takes the bridge down.

    Args:
        bridge: The bridge handling each request.
        stdin: The line-delimited request stream.
        stdout: The response stream; flushed after every response.
    """
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = decode_request(line)
        except ValueError:
            stdout.write(encode_response({"type": "response", "id": None, "operation": "error", "code": "BAD_REQUEST"}))
            stdout.flush()
            continue
        response = bridge.handle(request)
        stdout.write(encode_response(response))
        stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Resolve configuration and serve JSONL, or run the local diagnostic fixture mode."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = list(sys.argv[1:] if argv is None else argv)
    if "--diagnostic-fixture" in args:
        try:
            print(json.dumps(run_diagnostic_fixture(), separators=(",", ":"), ensure_ascii=False))
        except (BridgeConfigError, OSError) as error:
            _LOGGER.error("opencode_bridge.diagnostic_failed: %s", error)
            return 2
        return 0
    try:
        config = BridgeConfig.from_env(os.environ)
    except BridgeConfigError as error:
        _LOGGER.error("opencode_bridge.config_invalid: %s", error)
        return 2
    bridge = OpenCodeBridge(config)
    serve(bridge, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
