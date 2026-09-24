"""A reranker that scores a chunk on relevance AND on whether it carries citable literals.

**The hypothesis.** Every check the all-three stack loses on a small model asks for a literal that was in
a tool payload: an enum value (``DEGRADED``), an error class (``MFA_CHALLENGE_TIMEOUT``), a metric
namespace (``AWS/Lambda``), a policy name (``ReadOnlyAccess``), a figure (``907,35``). The relevance
filter is asked to keep the chunks worth keeping, and it scores them semantically against a query that
is the turn's question plus the tool call's own arguments -- a query that, by construction, never
contains the literal being looked for. So a log chunk holding the answer competes on how much it *reads
like* the phrase "root cause", against boilerplate that often reads more like it.

What distinguishes the chunk that can answer is not topical similarity, it is **citable density**: it
carries figures, identifiers, enum-shaped tokens, key/value pairs. The sibling context-graph plugin
already leans on exactly this heuristic -- ``describe.numeric_lines`` keeps a line because it carries a
number, not because it matches a question.

So this scorer keeps the semantic ranking and adds a bounded density prior on top. It is not a
replacement for relevance; a chunk of unrelated numbers must not outrank the relevant one, which is what
the cap on the bonus buys.

**Why this is a configuration and not a fork.** ``Reranker`` is a public, documented protocol whose
docstring invites exactly this: "Implement this protocol to plug a custom scorer into the plugin." The
preview stays verbatim, the threshold still decides admission, the budget still decides how much fits.
Only the ranking changes.
"""

from __future__ import annotations

import re

from strands_relevance_filter.reranker import BedrockReranker

_FIGURE = re.compile(r"\d[\d.,]*")
"""A number in any of the shapes the scenario's payloads use, thousands separators included."""

_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b")
"""An enum or error-class shaped token: ``MFA_CHALLENGE_TIMEOUT``, ``TLS_HANDSHAKE_RESET``."""

_NAMESPACED = re.compile(r"\b[A-Za-z][A-Za-z0-9]*/[A-Za-z][A-Za-z0-9]*\b")
"""A namespaced name: ``AWS/Lambda``, ``statements/2026``."""

_KEY_VALUE = re.compile(r"\b[a-z_]{3,}\s*[:=]\s*\S")
"""A key/value pair, which is how these payloads state a fact rather than describe one."""

_WEIGHTS = ((_IDENTIFIER, 0.40), (_NAMESPACED, 0.20), (_FIGURE, 0.25), (_KEY_VALUE, 0.15))
"""Contribution per pattern, ordered by how specifically each one marks a citable answer.

An enum-shaped identifier weighs most because nothing else in these payloads looks like one and a check
asking for it accepts no paraphrase. A bare figure weighs less: prose carries figures too.
"""

_MAX_BONUS = 0.25
"""Ceiling on the density bonus, in the reranker's own ``[0.0, 1.0]`` scale.

Deliberately small. The measured rerank distribution on this account puts a strong semantic match near
0.29 and an unrelated chunk near 0.03, so 0.25 is enough to lift a dense chunk past sparse competition
and NOT enough to lift an irrelevant one past a genuinely strong match. A density prior that could
outvote relevance would have replaced the plugin's job rather than assisted it.
"""

_SATURATION = 3
"""Occurrences of a pattern at which its contribution is full.

Presence is the signal, not volume: a chunk with one error class and a chunk with nine are both "the
chunk that names the error", and without saturation a wall of numbers would outrank the line that
matters.
"""


class DensityReranker(BedrockReranker):
    """Bedrock relevance scoring, plus a bounded prior for chunks that carry citable literals.

    Honours the ``Reranker`` contract exactly as the base class does: one score per chunk, in the order
    received, each inside ``[0.0, 1.0]``, all-or-nothing on failure. The bonus is applied after the
    remote scores come back and after they were validated, so a violation cannot be introduced here.

    Attributes:
        density_applied: Chunks whose score the prior actually moved, for the run's counters.
        density_total_bonus: Sum of the bonuses applied, so a report can state how hard this leaned.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Build the underlying Bedrock reranker and zero the counters.

        Args:
            args: Passed to :class:`BedrockReranker`.
            kwargs: Passed to :class:`BedrockReranker`.
        """
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.density_applied = 0
        self.density_total_bonus = 0.0

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        """Score each chunk semantically, then lift the ones carrying citable literals.

        Args:
            query: Scoring query, as the plugin built it.
            chunks: Chunk texts in chunk-index order. Not mutated.

        Returns:
            One score per chunk, aligned by index, each clamped into ``[0.0, 1.0]``.

        Raises:
            RerankerError: Propagated from the base implementation; a failure stays all-or-nothing.
        """
        scores = await super().score(query, chunks)

        lifted: list[float] = []
        for chunk, base in zip(chunks, scores, strict=True):
            bonus = _density_bonus(chunk)
            if bonus:
                self.density_applied += 1
                self.density_total_bonus += bonus
            lifted.append(min(1.0, base + bonus))

        return lifted


def _density_bonus(chunk: str) -> float:
    """Return the citable-density bonus for one chunk, capped at :data:`_MAX_BONUS`.

    Args:
        chunk: The chunk text, read only.

    Returns:
        A bonus in ``[0.0, _MAX_BONUS]``. Zero for a chunk carrying none of the patterns, which is what
        leaves prose ranked purely on relevance.
    """
    bonus = 0.0
    for pattern, weight in _WEIGHTS:
        hits = len(pattern.findall(chunk))
        if hits:
            bonus += weight * min(1.0, hits / _SATURATION)

    return min(_MAX_BONUS, bonus)
