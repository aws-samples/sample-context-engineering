"""Answer-correctness scoring for the validation harness.

Cost without correctness is not a result. A strategy that drops tokens by discarding the
passage the question needed would top the cost table, and nothing in the timing or token
figures would notice. So each turn carries expectations, and the report reads cost against
accuracy rather than on its own.

Two deliberate choices.

**Deterministic, not a judge.** The tools are mocked, so every factual question has one
computable answer. Checking for that answer's exact string is objective, free, and adds no
latency — where an LLM judge would add a call per turn per configuration, cost money, and
introduce its own variance into the thing being measured. Ground truth here is derived from
the tool implementations, not written by hand, so it cannot drift from what the tools return.

**Exact numeric strings.** Expectations match values as the tools formatted them
("R$ 24.143,12"), which also tests the offloader's protected-content guard: a preview that
paraphrased or reformatted a number would fail the check even if the figure were arithmetically
right. That is intended. A financial assistant that restates values in its own format has
introduced an error class, whether or not the arithmetic survived.

Turns whose answer depends on real AWS documentation are scored on the terms the answer must
contain rather than on a single string, because the wording is the documentation's, not ours.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


def _normalize(text: str) -> str:
    """Casefold and strip accents so a check does not fail on diacritics alone."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", stripped).casefold()


@dataclass(frozen=True)
class Check:
    """One scoreable assertion about a turn's answer.

    Attributes:
        name: Short identifier for the report.
        all_of: Every string must appear. Use for facts that must be stated together.
        any_of: At least one must appear. Use where the model has a legitimate choice of
            wording, or where several equivalent renderings of a figure are acceptable.
        none_of: None may appear. Use for the specific wrong answers a degraded context
            produces — a check that only rewards right answers cannot distinguish a correct
            answer from a confidently wrong one.
        weight: Relative importance within the turn.
        critical: When true, failing this check marks the turn as materially wrong, not just
            incomplete. Reserved for the facts the question was actually asking for.
    """

    name: str
    all_of: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()
    none_of: tuple[str, ...] = ()
    weight: float = 1.0
    critical: bool = False

    def evaluate(self, answer: str) -> tuple[bool, str]:
        """Return whether the check passes, and why it failed when it does not."""
        haystack = _normalize(answer)

        missing = [needle for needle in self.all_of if _normalize(needle) not in haystack]
        if missing:
            return False, f"missing required: {missing}"

        if self.any_of and not any(_normalize(needle) in haystack for needle in self.any_of):
            return False, f"none of the accepted forms present: {list(self.any_of)}"

        present = [needle for needle in self.none_of if _normalize(needle) in haystack]
        if present:
            return False, f"contains forbidden: {present}"

        return True, ""


# --- Ground truth ------------------------------------------------------------------
#
# Derived from the tool implementations in tools.py, and verified by computation rather
# than transcription. See ground_truth.py for the derivation:
#
#   T8  largest CDB redemption over 90 days = R$ 907,35 on 2026-08-25 (unique maximum)
#   T9  largest FinBank holding = Tesouro IPCA+ 2029, R$ 24.143,12 = 50.47% of R$ 47.832,15
#   T7  mean Lambda duration over 24 points = 163.05 ms

CHECKS: dict[str, tuple[Check, ...]] = {
    # --- Line A -------------------------------------------------------------------
    "A1-accounts": (
        Check(name="lists-btg-account", any_of=("0001/12345-6",), critical=True),
        Check(
            name="lists-every-institution",
            all_of=("FinBank", "TestBank", "NeoBank", "MidBank", "SampleBank"),
            weight=2.0,
            critical=True,
        ),
        Check(name="reports-balances", any_of=("47.832,15", "12.409,88", "88.204,73")),
    ),
    "A2-positions": (
        Check(
            name="reports-three-positions",
            all_of=("14.454,00", "24.143,12", "9.235,03"),
            weight=3.0,
            critical=True,
        ),
        Check(
            name="names-instruments",
            all_of=("CDB FinBank 2028", "Tesouro IPCA+ 2029", "Fundo FinBank Absoluto"),
            weight=2.0,
            critical=True,
        ),
    ),
    "A3-allocation": (
        Check(
            name="allocation-shares",
            all_of=("42,1", "35,8", "22,1"),
            weight=2.0,
            critical=True,
        ),
    ),
    "A4-projection": (
        Check(name="base-scenario", any_of=("51.204,77", "7,06"), critical=True),
        Check(name="mentions-both-scenarios", any_of=("hawkish",), weight=1.0),
    ),
    "A5-statement": (
        Check(
            name="largest-redemption-value",
            any_of=("907,35", "907.35"),
            weight=3.0,
            critical=True,
        ),
        Check(
            name="largest-redemption-date",
            any_of=("2026-08-25", "25/08/2026", "august 25"),
            weight=2.0,
            critical=True,
        ),
    ),
    # --- Line B -------------------------------------------------------------------
    "B1-status": (
        Check(name="connector-degraded", any_of=("DEGRADED", "degraded"), critical=True),
        Check(
            name="root-cause",
            all_of=("MFA_CHALLENGE_TIMEOUT",),
            weight=2.0,
            critical=True,
        ),
        Check(name="failure-count", any_of=("7 failures", "7 consecutive", "consecutive_failures", "7 attempts")),
    ),
    "B2-logs": (
        Check(
            name="identifies-error-classes",
            any_of=("MFA_CHALLENGE_TIMEOUT", "TLS_HANDSHAKE_RESET"),
            weight=2.0,
            critical=True,
        ),
        Check(name="mentions-retry-pattern", any_of=("30s", "30 s", "backoff", "retry", "attempt")),
    ),
    "B3-force-sync": (
        Check(name="sync-job-id", any_of=("sync-7f3a9c21",), weight=2.0, critical=True),
        Check(name="accepted", any_of=("accept", "enqueued", "queued", "true")),
    ),
    "B4-rotate": (
        Check(name="case-id", any_of=("CASE-20260826-0042",), weight=2.0, critical=True),
        Check(name="rotated", any_of=("rotated", "rotation", "rotating")),
    ),
    "B5-verify": (
        Check(
            name="reports-status-again",
            any_of=("DEGRADED", "degraded", "MFA_CHALLENGE_TIMEOUT"),
            critical=True,
        ),
        Check(name="mentions-warn-logs", any_of=("WARN", "warn", "warning")),
    ),
    # --- Line C -------------------------------------------------------------------
    "C1-lambda-docs": (
        # The answer is the AWS documentation's, so the check is on the concepts the
        # documentation actually states, not on a sentence we authored.
        Check(
            name="async-retry-semantics",
            any_of=("twice", "2 times", "two more times", "two attempts", "retries the function twice"),
            weight=2.0,
            critical=True,
        ),
        Check(
            name="mentions-queue-or-dlq",
            any_of=("queue", "dead-letter", "DLQ", "on-failure", "failure destination"),
        ),
        Check(
            name="execution-environment-reuse",
            any_of=("reuse", "reusable", "execution environment", "frozen", "freeze"),
        ),
        # A degraded preview on this turn produces a confident refusal rather than a wrong
        # number, so the refusal itself is the failure signal worth catching.
        Check(
            name="did-not-refuse",
            none_of=(
                "i could not find it in the documentation",
                "i was unable to access",
                "the documentation does not specify",
                "i could not obtain",
            ),
            critical=True,
        ),
    ),
    "C2-s3-naming": (
        # Dots are permitted in general-purpose bucket names but discouraged, and they break
        # virtual-hosted-style TLS and Transfer Acceleration. Either the verdict or the
        # reasoning is acceptable; both is better.
        Check(
            name="addresses-dots",
            any_of=("dot", "period", "dots", "."),
            weight=1.0,
        ),
        Check(
            name="reasoned-verdict",
            any_of=(
                "not recommended", "discouraged", "avoid", "should not",
                "valid", "permitted", "allowed", "invalid",
            ),
            weight=2.0,
            critical=True,
        ),
        Check(
            name="cites-a-rule",
            any_of=(
                "3 and 63", "3 to 63", "63 characters", "lowercase",
                "transfer acceleration", "ssl", "tls", "https",
                "virtual-hosted",
            ),
        ),
    ),
    "C3-metrics": (
        Check(
            name="average-duration",
            any_of=("163,05", "163.05", "163,0", "163.0", "163 ms", "163ms"),
            weight=2.0,
            critical=True,
        ),
        Check(name="cost-estimate", any_of=("cost", "USD", "$")),
    ),
    "C4-iam": (
        Check(
            name="describes-role",
            any_of=("ReadOnlyAccess", "connector-secrets-read", "connector-worker"),
            weight=2.0,
            critical=True,
        ),
        Check(name="least-privilege-reasoning", any_of=("least privilege", "privil", "permiss", "scope")),
    ),
    "C5-dynamo": (
        Check(
            name="compares-capacity-modes",
            all_of=("on-demand",),
            any_of=("provisioned",),
            weight=2.0,
            critical=True,
        ),
        Check(
            name="did-not-refuse",
            none_of=(
                "i could not find it in the documentation",
                "the documentation does not specify",
                "i was unable to access",
            ),
            critical=True,
        ),
    ),
    # --- Return -------------------------------------------------------------------
    "R1-largest-asset": (
        # The turn the graph is judged on: it refers back to A2, ten turns earlier, and
        # drops two subjects at once. If line A was compacted and not restored, it shows here.
        Check(
            name="largest-asset-named",
            any_of=("Tesouro IPCA+ 2029", "Tesouro IPCA"),
            weight=2.0,
            critical=True,
        ),
        Check(
            name="largest-asset-value",
            any_of=("24.143,12",),
            weight=2.0,
            critical=True,
        ),
        Check(
            name="share-of-total",
            any_of=("50,47", "50.47", "50,5", "50.5", "50%", "50,4", "approximately 50"),
            weight=2.0,
            critical=True,
        ),
        # Phrased narrowly on purpose. Forbidding a bare "I did not pull" would mark a correct
        # answer wrong: the model can give the FinBank figures and then note that it had not
        # fetched the *SampleBank* positions, offering to. That is accurate
        # scoping, not lost context. A negative check has to name the failure it is looking
        # for — here, an inability to recall the positions from the opening turn — or it
        # penalises precision.
        Check(
            name="did-not-lose-context",
            none_of=(
                "i do not have the btg positions",
                "i no longer have the positions",
                "i did not pull the btg positions",
                "i need to look up the positions again",
                "i do not have access to the start of the conversation",
                "could you repeat the positions",
                "i do not remember the positions",
            ),
            weight=2.0,
            critical=True,
        ),
    ),
    "R2-cross-reference": (
        # The hardest recall in the script: one fact from A5 and one from A2, with ten turns
        # of two unrelated subjects in between. Requires both to be simultaneously present.
        Check(
            name="recalls-redemption-value",
            any_of=("907,35", "907.35"),
            weight=3.0,
            critical=True,
        ),
        Check(
            name="recalls-fund-position",
            any_of=("9.235,03",),
            weight=3.0,
            critical=True,
        ),
        Check(
            name="answers-the-comparison",
            any_of=("smaller", "larger", "less", "more", "greater"),
            weight=2.0,
            critical=True,
        ),
        Check(
            name="did-not-lose-context",
            none_of=(
                "i do not have the statement",
                "i do not have the btg positions",
                "i need to look up again",
                "i do not have access to the start of the conversation",
                "could you repeat",
            ),
            weight=2.0,
            critical=True,
        ),
    ),
    "R3-consolidate": (
        Check(
            name="covers-portfolio",
            any_of=("47.832,15", "net worth", "portfolio", "Tesouro", "CDB"),
            critical=True,
        ),
        Check(
            name="covers-connector",
            any_of=("connector", "sync"),
            critical=True,
        ),
    ),
}


FILLER_CHECKS: dict[str, tuple[Check, ...]] = {
    # Expectations for the scored filler, keyed by the *kind* its label carries rather than by the
    # label itself -- there is one turn per generated label and a dictionary entry per label would not
    # survive a change of script length. Every check below asserts a literal the mocked tool returns
    # identically for every account, which is the criterion scenario.SCORED_FILLER_SPECS selects on.
    #
    # Each kind carries exactly one critical check, on the most distinctive thing the tool returns.
    # A grounding check, deliberately: what it asks is "did the answer come from the tool", which is
    # the question a long conversation puts at risk. Computation is the spine's job -- C3 is where the
    # agent has to average the datapoint series rather than quote it.
    "allocation": (
        Check(
            name="allocation-shares",
            all_of=("42,1", "35,8", "22,1"),
            weight=2.0,
            critical=True,
        ),
    ),
    "projection": (
        Check(
            name="projection-values",
            any_of=("51.204,77", "7,06"),
            weight=2.0,
            critical=True,
        ),
    ),
    "connector": (
        Check(name="connector-state", any_of=("DEGRADED", "degraded"), critical=True),
        Check(
            name="connector-root-cause",
            all_of=("MFA_CHALLENGE_TIMEOUT",),
            weight=2.0,
            critical=True,
        ),
    ),
    "logs": (
        Check(
            name="log-error-classes",
            any_of=("MFA_CHALLENGE_TIMEOUT", "TLS_HANDSHAKE_RESET"),
            weight=2.0,
            critical=True,
        ),
    ),
    "metrics": (
        Check(
            name="names-the-series",
            any_of=("AWS/Lambda", "Duration", "duration"),
            weight=2.0,
            critical=True,
        ),
    ),
    "iam": (
        Check(
            name="role-policies",
            any_of=("ReadOnlyAccess", "connector-secrets-read"),
            weight=2.0,
            critical=True,
        ),
    ),
    "objects": (
        Check(
            name="object-keys",
            any_of=("statements/", ".csv"),
            weight=2.0,
            critical=True,
        ),
    ),
}
"""Expectations for the generated scored turns, looked up by the kind in the label.

Keyed by kind and not by label because the labels are generated: the number of them follows the
script's length, so a per-label entry would go stale the first time ``--total-turns`` changed.
"""

UNIVERSAL_CHECKS: tuple[Check, ...] = (
    # Applied to every scored turn, on top of its own expectations. A negative check, so it costs a
    # turn nothing unless the answer volunteers the claim: it only fires when the model states an
    # account count, and every count it could state is wrong, because the fixture holds five.
    #
    # Measured before this existed: the full stack claimed six accounts in 36 of 42 filler turns,
    # duplicating one identifier and mislabelling two institutions, while the four single-strategy
    # configurations never did. That is folding drift -- the list came back from a Card's Description
    # instead of the tool result -- and it is exactly the failure a context strategy has to be judged
    # on. It scored as nothing, because filler carried no expectation at all.
    Check(
        name="no-fabricated-account-count",
        none_of=(
            "6 accounts",
            "six accounts",
            "7 accounts",
            "seven accounts",
            "4 accounts",
            "four accounts",
        ),
        weight=1.0,
        critical=True,
    ),
)
"""Checks every scored turn carries, whatever else it asserts.

The place for a claim that is wrong no matter which question produced it. Kept deliberately small: a
universal check that fires on a legitimate answer would corrupt every figure at once.
"""


def _checks_for(label: str) -> tuple[Check, ...] | None:
    """Return the expectations for ``label``, or ``None`` when the turn is unscored.

    Two lookups, in order. The hand-written spine is keyed by its exact label. A generated scored turn
    is keyed by the *kind* after the first dash, which is how one static expectation serves however
    many turns the script's length produces.

    Args:
        label: The turn's label.

    Returns:
        The checks, or ``None`` for a label neither lookup knows -- an unscored filler turn, whose
        label carries the kind ``filler`` on purpose so it matches nothing here.
    """
    checks = CHECKS.get(label)
    if checks is not None:
        return checks
    _, _, kind = label.partition("-")
    return FILLER_CHECKS.get(kind)


@dataclass
class TurnScore:
    """Accuracy outcome for one turn."""

    label: str
    passed: float = 0.0
    total: float = 0.0
    failures: list[str] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)
    scored: bool = True

    @property
    def ratio(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def materially_correct(self) -> bool:
        """No critical check failed: the turn answered what was asked."""
        return not self.critical_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.ratio, 3),
            "weight_passed": self.passed,
            "weight_total": self.total,
            "materially_correct": self.materially_correct,
            "failures": self.failures,
            "critical_failures": self.critical_failures,
            "scored": self.scored,
        }


def score_turn(label: str, answer: str) -> TurnScore:
    """Score one answer against the expectations for its turn.

    The universal checks are appended to whatever the turn asserts on its own, so a claim that is
    wrong regardless of the question is caught on every scored turn rather than on the one that
    happened to ask about it.
    """
    checks = _checks_for(label)
    if not checks:
        return TurnScore(label=label, scored=False)

    score = TurnScore(label=label)
    for check in (*checks, *UNIVERSAL_CHECKS):
        score.total += check.weight
        ok, reason = check.evaluate(answer or "")
        if ok:
            score.passed += check.weight
        else:
            score.failures.append(f"{check.name}: {reason}")
            if check.critical:
                score.critical_failures.append(check.name)
    return score


def score_run(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Score every turn of a run and summarize.

    ``weighted_accuracy`` is the fraction of expectation weight met across the run.
    ``turns_materially_correct`` counts turns with no critical failure — the stricter and
    more decision-relevant figure, since a turn can score 0.7 while missing the one number
    the user asked for.
    """
    scores = [score_turn(turn["label"], turn.get("response_text", "")) for turn in turns]
    scored = [score for score in scores if score.scored]

    total_weight = sum(score.total for score in scored)
    passed_weight = sum(score.passed for score in scored)
    correct = [score for score in scored if score.materially_correct]

    return {
        "weighted_accuracy": round(passed_weight / total_weight, 4) if total_weight else 0.0,
        "turns_scored": len(scored),
        "turns_materially_correct": len(correct),
        "material_correctness": round(len(correct) / len(scored), 4) if scored else 0.0,
        "critical_failures_total": sum(len(score.critical_failures) for score in scored),
        "per_turn": [score.to_dict() for score in scores],
    }
