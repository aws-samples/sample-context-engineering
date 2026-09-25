"""Unit tests for the SimilarityMatcher protocol and the default embedding matcher.

No AWS: every test injects a fake ``bedrock-runtime`` client that records the request bodies it received. The
assertions are about the contract the graph depends on — one score per Description in [0.0, 1.0], the received sequence
left alone, the asymmetric purposes, the cache, the lazy client, and an empty sequence instead of a raise.
"""

import io
import json
import logging
from collections.abc import Sequence
from typing import Any

import pytest

from context_core.graph.matcher import EmbeddingSimilarityMatcher, SimilarityMatcher


class FakeBedrockClient:
    """Records every ``invoke_model`` body and answers with unit vectors derived from the text."""

    def __init__(self, vectors: dict[str, list[float]] | None = None, error: Exception | None = None) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.model_ids: list[str] = []
        self._vectors = vectors or {}
        self._error = error

    def invoke_model(self, *, modelId: str, body: str) -> dict[str, Any]:
        """Answer one embed request, or raise the configured failure. Parameter names are botocore's."""
        if self._error is not None:
            raise self._error

        request = json.loads(body)
        self.bodies.append(request)
        self.model_ids.append(modelId)

        if "texts" in request:
            payload = {"embeddings": [self._vector_for(text) for text in request["texts"]]}
        else:
            payload = {"embedding": self._vector_for(request["inputText"])}

        return {"body": io.BytesIO(json.dumps(payload).encode())}

    def _vector_for(self, text: str) -> list[float]:
        """Return the configured vector for a text, or a deterministic default."""
        return self._vectors.get(text, [1.0, 0.0, 0.0])


def matcher_with(client: FakeBedrockClient, **kwargs: Any) -> EmbeddingSimilarityMatcher:
    """Build a matcher over a fake client."""
    return EmbeddingSimilarityMatcher(client=client, **kwargs)


def test_protocol_is_satisfied_by_member_without_inheriting() -> None:
    class Custom:
        def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
            return [0.5] * len(descriptions)

    custom = Custom()

    assert isinstance(custom, SimilarityMatcher)
    assert callable(getattr(custom, "score", None))
    assert isinstance(EmbeddingSimilarityMatcher(), SimilarityMatcher)


def test_construction_opens_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the client was resolved at construction")

    monkeypatch.setattr(boto3, "Session", explode)

    matcher = EmbeddingSimilarityMatcher()

    assert matcher.model_id == "cohere.embed-multilingual-v3"
    # And an empty batch still needs nothing: the guard returns before any resolution.
    assert matcher.score("anything", []) == []


def test_score_returns_one_value_per_description_in_range() -> None:
    client = FakeBedrockClient(
        vectors={
            "which asset is my largest?": [1.0, 0.0],
            "positions": [1.0, 0.0],
            "weather": [0.0, 1.0],
            "opposite": [-1.0, 0.0],
        }
    )
    matcher = matcher_with(client)

    scores = matcher.score("which asset is my largest?", ["positions", "weather", "opposite"])

    assert len(scores) == 3
    assert all(0.0 <= value <= 1.0 for value in scores)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    # Negative similarity is clamped, not preserved: "points the other way" reads as unrelated.
    assert scores[2] == pytest.approx(0.0)


def test_score_uses_asymmetric_purposes() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client)

    matcher.score("question", ["first", "second"])

    by_input_type = {body["input_type"]: body["texts"] for body in client.bodies}
    assert by_input_type == {"search_query": ["question"], "search_document": ["first", "second"]}
    assert set(client.model_ids) == {"cohere.embed-multilingual-v3"}


def test_score_leaves_the_received_sequence_untouched() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client)
    descriptions = ["alpha", "beta", "gamma"]
    snapshot = list(descriptions)

    scores = matcher.score("question", descriptions)

    assert descriptions == snapshot
    assert len(scores) == len(snapshot)


def test_score_is_aligned_by_index_with_repeated_descriptions() -> None:
    client = FakeBedrockClient(vectors={"question": [1.0, 0.0], "same": [1.0, 0.0], "other": [0.0, 1.0]})
    matcher = matcher_with(client)

    scores = matcher.score("question", ["same", "other", "same"])

    assert scores[0] == pytest.approx(scores[2])
    assert scores[1] == pytest.approx(0.0)
    # A repeated text is embedded once, so the document request carries two texts and not three.
    document_body = next(body for body in client.bodies if body["input_type"] == "search_document")
    assert document_body["texts"] == ["same", "other"]


def test_unchanged_descriptions_are_served_from_the_cache() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client)

    matcher.score("first question", ["alpha", "beta"])
    matcher.score("second question", ["alpha", "beta"])

    bodies = [body for body in client.bodies if body["input_type"] == "search_document"]
    assert [body["texts"] for body in bodies] == [["alpha", "beta"]]
    queries = [body["texts"] for body in client.bodies if body["input_type"] == "search_query"]
    assert queries == [["first question"], ["second question"]]


def test_only_the_changed_description_is_re_embedded() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client)

    matcher.score("question", ["alpha", "beta"])
    matcher.score("question", ["alpha", "beta changed"])

    bodies = [body["texts"] for body in client.bodies if body["input_type"] == "search_document"]
    assert bodies == [["alpha", "beta"], ["beta changed"]]


def test_the_same_text_is_cached_per_purpose() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client)

    matcher.score("alpha", ["alpha"])

    # One text, two purposes, two requests: a query vector must never be served where a document one was asked for.
    assert [body["input_type"] for body in client.bodies].count("search_query") == 1
    assert [body["input_type"] for body in client.bodies].count("search_document") == 1


def test_empty_descriptions_short_circuits_without_a_call() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client)

    assert matcher.score("question", []) == []
    assert matcher.vectors([]) == []
    assert client.bodies == []


def test_failure_returns_an_empty_sequence_and_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    client = FakeBedrockClient(error=RuntimeError("bedrock is down"))
    matcher = matcher_with(client)

    with caplog.at_level(logging.DEBUG, logger="context_core.graph.matcher"):
        scores = matcher.score("question", ["alpha", "beta"])

    assert scores == []
    records = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert len(records) == 1
    assert records[0].exc_info is not None


def test_malformed_response_returns_an_empty_sequence() -> None:
    class ShortClient(FakeBedrockClient):
        def invoke_model(self, *, modelId: str, body: str) -> dict[str, Any]:
            return {"body": io.BytesIO(json.dumps({"embeddings": []}).encode())}

    matcher = matcher_with(ShortClient())

    assert matcher.score("question", ["alpha"]) == []


def test_vectors_reuses_the_document_cache() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client)
    descriptions = ["alpha", "beta"]

    matcher.score("question", descriptions)
    calls_after_score = len(client.bodies)
    vectors = matcher.vectors(descriptions)

    assert len(client.bodies) == calls_after_score
    assert [list(vector) for vector in vectors] == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_vectors_returns_empty_when_embedding_is_unavailable() -> None:
    matcher = matcher_with(FakeBedrockClient(error=RuntimeError("bedrock is down")))

    assert matcher.vectors(["alpha"]) == []


def test_a_non_cohere_model_is_called_one_text_per_request() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client, model_id="amazon.titan-embed-text-v2:0")

    scores = matcher.score("question", ["alpha", "beta"])

    assert len(scores) == 2
    assert [body["inputText"] for body in client.bodies] == ["question", "alpha", "beta"]


def test_cache_is_bounded_by_cache_size() -> None:
    client = FakeBedrockClient()
    matcher = matcher_with(client, cache_size=1)

    matcher.vectors(["alpha"])
    matcher.vectors(["alpha"])
    assert len(client.bodies) == 1

    # One pair of room: "beta" evicts "alpha", so "alpha" is paid for again.
    matcher.vectors(["beta"])
    matcher.vectors(["alpha"])

    assert [body["texts"] for body in client.bodies] == [["alpha"], ["beta"], ["alpha"]]


@pytest.mark.parametrize("cache_size", [0, -1, True, 1.5, "8"])
def test_invalid_cache_size_is_rejected_at_construction(cache_size: Any) -> None:
    with pytest.raises(ValueError, match="cache_size"):
        EmbeddingSimilarityMatcher(cache_size=cache_size)
