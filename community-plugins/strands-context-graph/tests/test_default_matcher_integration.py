"""Credential-gated end-to-end test of the default ``EmbeddingSimilarityMatcher`` against Bedrock.

Every other matcher test injects a fake client, which proves the shape of the request but not that the real
``cohere.embed-multilingual-v3`` round answers with a usable score distribution. This module closes that gap: it calls
the live model with the asymmetric purposes the graph relies on (``search_query`` for the question,
``search_document`` for each Description) and checks the answer the thresholds are calibrated against.

Nothing here runs unless it is asked for. The gate is an explicit opt-in env var *plus* resolvable credentials, a
region and an importable ``boto3``; any one missing skips the module before a socket is opened, so an offline CI run
skips rather than fails. Even when enabled, a provisioned account without access to the model still skips: the matcher
never raises, it reports unavailability as an empty sequence, and an empty sequence here means "no model access", not
"wrong answer".

Enable with:

```bash
STRANDS_CONTEXT_GRAPH_BEDROCK_TESTS=1 hatch test tests/test_default_matcher_integration.py
```

Requirements: 10.3 (asymmetric multilingual purposes in the default matcher), 11.4 (client resolved through the SDK's
Bedrock path, on first need).
"""

import io
import json
import os
from collections.abc import Sequence
from typing import Any

import pytest

from strands_context_graph.matcher import EmbeddingSimilarityMatcher

ENABLE_VAR = "STRANDS_CONTEXT_GRAPH_BEDROCK_TESTS"

MODEL_ID = "cohere.embed-multilingual-v3"

QUESTION = "How much did the Q3 cloud infrastructure bill come to?"

RELATED_DESCRIPTION = (
    "Turn 3: reviewed the Q3 cloud infrastructure invoice with get_invoice (1 call). "
    "Total infrastructure spend for the quarter was 48,200.00 USD across compute and storage."
)

UNRELATED_DESCRIPTION = (
    "Turn 7: debugged a flaky unit test in the date parser with run_tests (2 calls). "
    "The failure was a timezone off-by-one in the weekday rollover."
)


def _resolve_session() -> Any:
    """Return a boto3 session that can reach Bedrock, or skip the module.

    Raises:
        Skipped: When the opt-in is absent, ``boto3`` is missing, or no credentials/region resolve. Each of these is a
            reason not to attempt a call at all, so the skip happens before any client is built.
    """
    if os.environ.get(ENABLE_VAR) != "1":
        pytest.skip(f"{ENABLE_VAR}=1 not set: the live Bedrock matcher test is opt-in", allow_module_level=True)

    try:
        import boto3
    except ImportError:  # pragma: no cover - boto3 is a test-time extra, not a package dependency.
        pytest.skip("boto3 is not installed: no Bedrock path to exercise", allow_module_level=True)

    session = boto3.Session()
    if session.get_credentials() is None:
        pytest.skip("no AWS credentials resolve: nothing provisioned to call", allow_module_level=True)
    if not session.region_name:
        pytest.skip("no AWS region resolves: a Bedrock call has no endpoint", allow_module_level=True)

    return session


SESSION = _resolve_session()


class RecordingClient:
    """Wraps the real ``bedrock-runtime`` client, keeping each request body so the purposes can be asserted."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.bodies: list[dict[str, Any]] = []
        self.model_ids: list[str] = []

    def invoke_model(self, *, modelId: str, body: str) -> dict[str, Any]:
        """Forward one embed request and record it. Parameter names are botocore's.

        Returns:
            The provider response, with the body re-wrapped so the matcher can still read it once.
        """
        self.bodies.append(json.loads(body))
        self.model_ids.append(modelId)
        response = self.inner.invoke_model(modelId=modelId, body=body)
        # The stream is consumed here to record nothing and re-offered, because a botocore body reads only once.
        return {**response, "body": io.BytesIO(response["body"].read())}

    @property
    def input_types(self) -> list[str]:
        """Return the ``input_type`` of every recorded Cohere request, in call order."""
        return [body["input_type"] for body in self.bodies if "input_type" in body]


def scored(matcher: EmbeddingSimilarityMatcher, question: str, descriptions: Sequence[str]) -> Sequence[float]:
    """Score through the live model, skipping when the account cannot reach it.

    Args:
        matcher: Matcher under test.
        question: The turn's question.
        descriptions: The Card Descriptions.

    Returns:
        One similarity per Description.

    Raises:
        Skipped: When the matcher reports unavailability, which for a gated run means the model is not provisioned for
            this account rather than that the scores are wrong.
    """
    scores = matcher.score(question, descriptions)
    if not scores:
        pytest.skip(f"{MODEL_ID} is not reachable from these credentials: no score to assert")

    return scores


def test_live_round_returns_one_score_per_description_in_range() -> None:
    matcher = EmbeddingSimilarityMatcher(MODEL_ID, boto_session=SESSION)
    descriptions = [RELATED_DESCRIPTION, UNRELATED_DESCRIPTION]

    scores = scored(matcher, QUESTION, descriptions)

    assert len(scores) == len(descriptions)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_live_round_scores_the_related_description_above_the_unrelated_one() -> None:
    matcher = EmbeddingSimilarityMatcher(MODEL_ID, boto_session=SESSION)

    related, unrelated = scored(matcher, QUESTION, [RELATED_DESCRIPTION, UNRELATED_DESCRIPTION])

    assert related > unrelated, f"related={related!r} did not outscore unrelated={unrelated!r}"


def test_live_round_scores_across_languages() -> None:
    # Multilingual is the reason for the model choice: the same subject in another language must not read as unrelated.
    matcher = EmbeddingSimilarityMatcher(MODEL_ID, boto_session=SESSION)
    question = "Quanto custou a infraestrutura de nuvem no terceiro trimestre?"

    related, unrelated = scored(matcher, question, [RELATED_DESCRIPTION, UNRELATED_DESCRIPTION])

    assert related > unrelated, f"cross-language related={related!r} did not outscore unrelated={unrelated!r}"


def test_live_round_sends_asymmetric_purposes() -> None:
    client = RecordingClient(SESSION.client("bedrock-runtime"))
    matcher = EmbeddingSimilarityMatcher(MODEL_ID, client=client)

    scored(matcher, QUESTION, [RELATED_DESCRIPTION, UNRELATED_DESCRIPTION])

    assert set(client.model_ids) == {MODEL_ID}
    # One request per purpose, because one Cohere request carries one input_type; order is thread-dependent.
    assert sorted(client.input_types) == ["search_document", "search_query"]


def test_client_is_resolved_lazily_and_only_once() -> None:
    matcher = EmbeddingSimilarityMatcher(MODEL_ID, boto_session=SESSION)

    assert matcher._client is None, "construction opened a client"

    scored(matcher, QUESTION, [RELATED_DESCRIPTION])
    resolved = matcher._client

    assert resolved is not None, "scoring did not resolve a client"

    scored(matcher, "a different question entirely", [RELATED_DESCRIPTION])

    assert matcher._client is resolved, "a second round rebuilt the client"


def test_unchanged_descriptions_are_served_from_the_cache() -> None:
    client = RecordingClient(SESSION.client("bedrock-runtime"))
    matcher = EmbeddingSimilarityMatcher(MODEL_ID, client=client)
    descriptions = [RELATED_DESCRIPTION, UNRELATED_DESCRIPTION]

    scored(matcher, QUESTION, descriptions)
    first_round_calls = len(client.bodies)
    scored(matcher, "and what about storage specifically?", descriptions)

    # Only the new question reaches the provider: the Descriptions are unchanged, so their document vectors are cached.
    assert len(client.bodies) == first_round_calls + 1
    assert client.input_types[first_round_calls:] == ["search_query"]
