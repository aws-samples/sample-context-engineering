"""Tests for the BedrockReranker scoring backend, with no network calls.

The client is a synchronous boto3 client with a single ``rerank`` method plus a
``meta.region_name`` attribute. BedrockReranker exposes no constructor seam for
injecting a client, so each test constructs the reranker (which builds a real
boto client object but issues no call) and then overwrites the private
``_client`` attribute with a stub. That is the only injection point the class
offers; no network I/O happens because ``rerank`` is never reached on the real
client.
"""

import math
from typing import Any

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from context_core.relevance.reranker import BedrockReranker, RerankerError


class _StubClient:
    """Stub Bedrock client: records rerank calls and replays canned responses."""

    def __init__(self, responses: list[Any] | None = None, region_name: str = "us-east-1") -> None:
        self._responses = list(responses) if responses is not None else []
        self.calls: list[dict[str, Any]] = []
        self.meta = type("Meta", (), {"region_name": region_name})()

    def rerank(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("rerank called more times than responses provided")
        return self._responses.pop(0)


def _make_reranker(client: _StubClient, model_id: str = "amazon.rerank-v1:0") -> BedrockReranker:
    """Build a reranker whose network client is the given stub."""
    reranker = BedrockReranker(model_id=model_id, region_name="us-east-1")
    reranker._client = client  # only injection point the class exposes
    # _model_arn was computed from the real client's region at construction; for a
    # bare id that is the us-east-1 foundation-model ARN, which is what we assert on.
    return reranker


def _result(index: int, score: float) -> dict[str, Any]:
    return {"index": index, "relevanceScore": score}


# --------------------------------------------------------------------------- #
# Guard clauses                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_empty_query_raises_before_client_called():
    client = _StubClient(responses=[])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("", ["chunk"])
    assert client.calls == []


@pytest.mark.asyncio
async def test_whitespace_query_raises_before_client_called():
    client = _StubClient(responses=[])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("   \t\n ", ["chunk"])
    assert client.calls == []


@pytest.mark.asyncio
async def test_empty_chunks_returns_empty_without_client():
    client = _StubClient(responses=[])
    reranker = _make_reranker(client)
    assert await reranker.score("query", []) == []
    assert client.calls == []


# --------------------------------------------------------------------------- #
# Normal scoring and index remapping                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_normal_call_one_score_per_chunk_aligned_by_index():
    client = _StubClient(responses=[{"results": [_result(0, 0.9), _result(1, 0.5), _result(2, 0.1)]}])
    reranker = _make_reranker(client)
    scores = await reranker.score("q", ["a", "b", "c"])
    assert scores == [0.9, 0.5, 0.1]
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_results_out_of_order_are_remapped_by_index():
    client = _StubClient(responses=[{"results": [_result(2, 0.1), _result(0, 0.9), _result(1, 0.5)]}])
    reranker = _make_reranker(client)
    scores = await reranker.score("q", ["a", "b", "c"])
    assert scores == [0.9, 0.5, 0.1]


@pytest.mark.asyncio
async def test_index_with_no_result_keeps_zero():
    client = _StubClient(responses=[{"results": [_result(0, 0.9), _result(2, 0.3)]}])
    reranker = _make_reranker(client)
    scores = await reranker.score("q", ["a", "b", "c"])
    assert scores == [0.9, 0.0, 0.3]


@pytest.mark.asyncio
async def test_boundary_scores_zero_and_one_accepted():
    client = _StubClient(responses=[{"results": [_result(0, 0.0), _result(1, 1.0)]}])
    reranker = _make_reranker(client)
    assert await reranker.score("q", ["a", "b"]) == [0.0, 1.0]


# --------------------------------------------------------------------------- #
# Pagination                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pagination_batches_and_remaps(monkeypatch):
    """With max_sources_per_query smaller than the chunk count, chunks are split
    into batches, every chunk lands in exactly one batch, and each batch's local
    index is remapped to the global chunk index.
    """
    # Two batches of 2: chunks [c0,c1] then [c2,c3,c4] -> batch size 2 forces 3 batches.
    responses = [
        {"results": [_result(0, 0.10), _result(1, 0.11)]},  # global 0,1
        {"results": [_result(1, 0.23), _result(0, 0.22)]},  # global 2,3 (out of order)
        {"results": [_result(0, 0.34)]},  # global 4
    ]
    client = _StubClient(responses=responses)
    reranker = _make_reranker(client)
    monkeypatch.setattr(reranker, "max_sources_per_query", 2)

    scores = await reranker.score("q", ["c0", "c1", "c2", "c3", "c4"])
    assert scores == [0.10, 0.11, 0.22, 0.23, 0.34]
    assert len(client.calls) == 3

    # Verify each chunk landed in exactly one batch, in order.
    def batch_texts(call):
        return [
            s["inlineDocumentSource"]["textDocument"]["text"] for s in call["sources"]
        ]

    assert batch_texts(client.calls[0]) == ["c0", "c1"]
    assert batch_texts(client.calls[1]) == ["c2", "c3"]
    assert batch_texts(client.calls[2]) == ["c4"]


# --------------------------------------------------------------------------- #
# Malformed response shapes                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_response_missing_results_key_raises():
    client = _StubClient(responses=[{"noresults": []}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_response_not_a_dict_raises():
    client = _StubClient(responses=[["not", "a", "dict"]])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_entry_not_a_mapping_raises():
    client = _StubClient(responses=[{"results": ["not-a-mapping"]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_missing_index_raises():
    client = _StubClient(responses=[{"results": [{"relevanceScore": 0.5}]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_non_integer_index_raises():
    client = _StubClient(responses=[{"results": [{"index": "0", "relevanceScore": 0.5}]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_bool_index_raises():
    """A bool is an int subclass; the validator rejects it explicitly."""
    client = _StubClient(responses=[{"results": [{"index": True, "relevanceScore": 0.5}]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a", "b"])


@pytest.mark.asyncio
async def test_result_index_out_of_range_raises():
    client = _StubClient(responses=[{"results": [_result(5, 0.5)]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_non_finite_score_raises():
    client = _StubClient(responses=[{"results": [{"index": 0, "relevanceScore": math.inf}]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_nan_score_raises():
    client = _StubClient(responses=[{"results": [{"index": 0, "relevanceScore": math.nan}]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_score_above_one_raises():
    client = _StubClient(responses=[{"results": [_result(0, 1.5)]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_score_below_zero_raises():
    client = _StubClient(responses=[{"results": [_result(0, -0.1)]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_result_missing_score_raises():
    client = _StubClient(responses=[{"results": [{"index": 0}]}])
    reranker = _make_reranker(client)
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


# --------------------------------------------------------------------------- #
# AWS error translation                                                       #
# --------------------------------------------------------------------------- #


class _RaisingClient(_StubClient):
    def __init__(self, exc: Exception, region_name: str = "us-east-1") -> None:
        super().__init__(responses=[], region_name=region_name)
        self._exc = exc

    def rerank(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise self._exc


@pytest.mark.asyncio
async def test_client_error_becomes_reranker_error():
    exc = ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "Rerank")
    reranker = _make_reranker(_RaisingClient(exc))
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


@pytest.mark.asyncio
async def test_botocore_error_becomes_reranker_error():
    class _SomeBotoError(BotoCoreError):
        fmt = "boom"

    reranker = _make_reranker(_RaisingClient(_SomeBotoError()))
    with pytest.raises(RerankerError):
        await reranker.score("q", ["a"])


# --------------------------------------------------------------------------- #
# Model ARN construction                                                      #
# --------------------------------------------------------------------------- #


def test_bare_model_id_resolved_to_foundation_model_arn():
    client = _StubClient(responses=[])
    reranker = _make_reranker(client, model_id="amazon.rerank-v1:0")
    assert reranker._model_arn == "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"


def test_full_arn_passed_through_unchanged():
    arn = "arn:aws:bedrock:eu-west-1::foundation-model/cohere.rerank-v3-5:0"
    reranker = BedrockReranker(model_id=arn, region_name="us-east-1")
    assert reranker._model_arn == arn


@pytest.mark.asyncio
async def test_model_arn_is_sent_in_rerank_request():
    client = _StubClient(responses=[{"results": [_result(0, 0.5)]}])
    reranker = _make_reranker(client, model_id="amazon.rerank-v1:0")
    await reranker.score("q", ["a"])
    sent_arn = client.calls[0]["rerankingConfiguration"]["bedrockRerankingConfiguration"][
        "modelConfiguration"
    ]["modelArn"]
    assert sent_arn == "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"
