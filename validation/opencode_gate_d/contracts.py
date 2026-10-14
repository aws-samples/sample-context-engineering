"""The evidence contract of Gate D: what one model request of practice D must be able to show.

Gate D asks a narrow question of the OpenCode binding of the context graph: *is it request-only?* It must change what one model
request carries and nothing the host keeps. The contract below is ``OpenCodeGateDEvidence`` of the functional design, one
record per request, plus the closed vocabulary around it. Everything here is a pure value check: no host, no network, no clock.

A record never holds a word of the conversation. It holds digests, opaque ids and booleans.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from opencode_gate_a.canonical import digest as canonical_digest

__all__ = [
    "ABSENCE_OF_DURABLE_EFFECT",
    "CARD_SOURCES_OPTIONAL_SCENARIOS",
    "GateDDecision",
    "GateDRequestEvidence",
    "REQUIRED_SCENARIOS",
    "Scenario",
    "absence_digest",
    "digest_is_well_formed",
]

Scenario = Literal["tool_loop", "subagent", "compact", "resume", "recovery"]

REQUIRED_SCENARIOS: Final[tuple[Scenario, ...]] = ("tool_loop", "subagent", "compact", "resume", "recovery")
"""The five scenarios FR5.2-FR5.4 name. A Gate D evidence set that misses one is ``no_go``."""

CARD_SOURCES_OPTIONAL_SCENARIOS: Final[frozenset[str]] = frozenset({"pre_card"})
"""A request made before the conversation has a closed turn has no Card to name. It says so; it is not silently exempt."""

ABSENCE_OF_DURABLE_EFFECT: Final[dict[str, Any]] = {
    "graph_messages_in_store": 0,
    "durable_writes": 0,
    "stored_transcript_changed": False,
}
"""What 'no extra durable effect' means, as a value. ``absence_digest`` is the digest of the observation of these three facts."""

_DIGEST = re.compile(r"^(sha256:)?[0-9a-f]{64}$")


def digest_is_well_formed(value: object) -> bool:
    """Whether ``value`` is a SHA-256 hex digest, with or without the ``sha256:`` framing."""
    return isinstance(value, str) and _DIGEST.match(value) is not None


def _bare(value: str) -> str:
    return value.removeprefix("sha256:")


def absence_digest(observation: dict[str, Any] | None = None) -> str:
    """The digest of an observation of durable effects (default: none at all)."""
    return canonical_digest(ABSENCE_OF_DURABLE_EFFECT if observation is None else observation)


@dataclass(frozen=True)
class GateDRequestEvidence:
    """Evidence for one model request of practice D (entity ``OpenCodeGateDEvidence``).

    Attributes:
        evidence_id: Opaque id of this record.
        gate_a_evidence_id: The Gate A evidence this request is recorded under. The functional design requires a Gate A
            ``go``; under an owner waiver it names the Gate A record that was NOT ``go`` and the evaluator says so.
        request_id: Opaque correlation id of the model request.
        stream_id: The session (or scenario run) the request belongs to; Card stability is judged within one stream and
            Card identity must not collide across streams.
        transcript_before_digest: Digest of the durable transcript as the host handed it to the request.
        transcript_after_digest: Digest of the durable transcript read back after the request. Equal to the one before.
        card_ids_and_sources: ``(card_id, sources)`` pairs: each Card and the host message ids it stands for.
        scenarios: The scenarios this request exercises.
        full_pass_digest: Digest of the model input when the projection is a full pass (``expand_threshold=0``). Present on
            the requests that were run as a full pass, ``None`` on the others (a live request has no full-pass twin).
        baseline_model_input_digest: Digest of the model input without the feature. A full pass must reproduce it.
        canonical_equality: Whether the durable transcript before and after are equal under the canonical protocol.
        absence_digest: Digest of the observation of durable effects; must equal :func:`absence_digest` of none.
        projected: Whether this request actually changed what the model was sent (a collapse happened).
        epoch: How many times the host compacted or reverted the session before this request. Card identity is judged within
            one epoch of one stream: a compaction is a defined transition, the one way an id may legitimately point at
            different messages afterwards. Across streams it is still never shared.
        outcome: ``pending`` until the evaluator decides.
    """

    evidence_id: str
    gate_a_evidence_id: str
    request_id: str
    stream_id: str
    transcript_before_digest: str
    transcript_after_digest: str
    card_ids_and_sources: tuple[tuple[str, tuple[str, ...]], ...]
    scenarios: tuple[str, ...]
    full_pass_digest: str | None
    baseline_model_input_digest: str
    canonical_equality: bool
    absence_digest: str
    projected: bool = False
    epoch: int = 0
    outcome: Literal["pending", "go", "no_go"] = "pending"
    gate: Literal["Gate_D"] = "Gate_D"

    def defects(self) -> list[str]:
        """Everything wrong with this record as a record: missing or malformed fields. Empty means well formed."""
        problems: list[str] = []
        for name in ("evidence_id", "gate_a_evidence_id", "request_id", "stream_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                problems.append(f"{name} is missing")
        required = ["transcript_before_digest", "transcript_after_digest", "baseline_model_input_digest", "absence_digest"]
        if "full_pass" in self.scenarios:
            # Only a request that was actually run as a full pass has a full-pass model input to show.
            required.append("full_pass_digest")
        for name in required:
            if not digest_is_well_formed(getattr(self, name)):
                problems.append(f"{name} is missing or not a sha256 digest")
        if not self.scenarios:
            problems.append("scenarios is empty")
        unknown = [name for name in self.scenarios if name not in REQUIRED_SCENARIOS and name not in _EXTRA_SCENARIOS]
        if unknown:
            problems.append(f"unknown scenario(s): {', '.join(sorted(unknown))}")
        for card_id, sources in self.card_ids_and_sources:
            if not card_id or not sources:
                problems.append("a Card has no id or no source")
                break
        return problems

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe projection, digests and ids only."""
        return {
            "evidence_id": self.evidence_id,
            "gate": self.gate,
            "gate_a_evidence_id": self.gate_a_evidence_id,
            "request_id": self.request_id,
            "stream_id": self.stream_id,
            "transcript_before_digest": self.transcript_before_digest,
            "transcript_after_digest": self.transcript_after_digest,
            "card_ids_and_sources": [[card_id, list(sources)] for card_id, sources in self.card_ids_and_sources],
            "scenarios": list(self.scenarios),
            "full_pass_digest": self.full_pass_digest,
            "baseline_model_input_digest": self.baseline_model_input_digest,
            "canonical_equality": self.canonical_equality,
            "absence_digest": self.absence_digest,
            "projected": self.projected,
            "epoch": self.epoch,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateDRequestEvidence:
        """Rebuild a record from :meth:`to_dict`."""
        return cls(
            evidence_id=data["evidence_id"],
            gate_a_evidence_id=data["gate_a_evidence_id"],
            request_id=data["request_id"],
            stream_id=data["stream_id"],
            transcript_before_digest=data["transcript_before_digest"],
            transcript_after_digest=data["transcript_after_digest"],
            card_ids_and_sources=tuple((card_id, tuple(sources)) for card_id, sources in data["card_ids_and_sources"]),
            scenarios=tuple(data["scenarios"]),
            full_pass_digest=data.get("full_pass_digest"),
            baseline_model_input_digest=data["baseline_model_input_digest"],
            canonical_equality=bool(data["canonical_equality"]),
            absence_digest=data["absence_digest"],
            projected=bool(data.get("projected", False)),
            epoch=int(data.get("epoch", 0)),
            outcome=data.get("outcome", "pending"),
        )

    def transcripts_equal(self) -> bool:
        """Whether the durable transcript before and after the request is the same, digest for digest."""
        return digest_is_well_formed(self.transcript_before_digest) and _bare(self.transcript_before_digest) == _bare(
            self.transcript_after_digest
        )

    def full_pass_matches_baseline(self) -> bool:
        """Whether the full-pass model input reproduces the input without the feature."""
        return (
            digest_is_well_formed(self.full_pass_digest)
            and digest_is_well_formed(self.baseline_model_input_digest)
            and _bare(str(self.full_pass_digest)) == _bare(self.baseline_model_input_digest)
        )

    def absence_is_clean(self) -> bool:
        """Whether the durable-effects observation is the empty one."""
        return digest_is_well_formed(self.absence_digest) and _bare(self.absence_digest) == _bare(absence_digest())


_EXTRA_SCENARIOS: Final[frozenset[str]] = frozenset({"full_pass", "pre_card", "live"})
"""Scenario names that annotate a request without being one of the five required ones."""


@dataclass(frozen=True)
class GateDDecision:
    """The decision the evaluator reports.

    Attributes:
        outcome: ``go``, ``no_go`` or ``unsupported`` (the precondition was not met and was not waived).
        capability_state: ``unsupported`` unless the contract passed; under an owner waiver the best it can be is
            ``experimental``, never ``supported``.
        precondition: ``gate_a_go``, ``waived_by_owner`` or ``unmet``.
        reasons: Why, one stable code per failed check. Empty on ``go``.
        checks: Each check and whether it held, for the report.
        requests_evaluated: How many request records were judged.
        scenarios_covered: The union of the scenarios over the records.
        gate_a_outcome: The Gate A outcome this decision was made against, recorded and never changed by it.
    """

    outcome: Literal["go", "no_go", "unsupported"]
    capability_state: Literal["unsupported", "experimental", "supported"]
    precondition: Literal["gate_a_go", "waived_by_owner", "unmet"]
    reasons: tuple[str, ...]
    checks: dict[str, bool]
    requests_evaluated: int
    scenarios_covered: tuple[str, ...]
    gate_a_outcome: str
    waiver: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "Gate_D",
            "outcome": self.outcome,
            "capability_state": self.capability_state,
            "precondition": self.precondition,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
            "requests_evaluated": self.requests_evaluated,
            "scenarios_covered": list(self.scenarios_covered),
            "gate_a_outcome": self.gate_a_outcome,
            "waiver": dict(self.waiver),
            "gate_a_reclassified": False,
        }


def all_scenarios(requests: Iterable[GateDRequestEvidence]) -> tuple[str, ...]:
    """The scenarios covered by ``requests``, in the canonical order, extras last."""
    seen = {name for request in requests for name in request.scenarios}
    ordered = [name for name in REQUIRED_SCENARIOS if name in seen]
    return tuple(ordered + sorted(seen - set(ordered)))


def streams(requests: Sequence[GateDRequestEvidence]) -> dict[tuple[str, int], list[GateDRequestEvidence]]:
    """The records grouped by ``(stream, epoch)``, each in the order given."""
    grouped: dict[tuple[str, int], list[GateDRequestEvidence]] = {}
    for request in requests:
        grouped.setdefault((request.stream_id, request.epoch), []).append(request)
    return grouped
