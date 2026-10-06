"""The scoring protocol: one question against N Card Descriptions.

Asymmetric by construction: the short side is a question and the long side is a Description of up to
``description_tokens``, which asks for a query purpose against a document purpose. Passing the wrong purpose does not
fail, it silently returns a worse vector, so the purpose is part of the call and the vector cache is keyed by the
``(purpose, text)`` pair rather than by text alone.

Unavailability is reported by returning an empty sequence, never by raising: the caller degrades to Full Content
everywhere rather than failing the turn.

Embedding, not rerank. The Turn Choice sits on the critical path at least once per turn, where a rerank call costs
seconds; an embedding round costs a fraction of that because only the question is new — every unchanged Description is
served from the cache. Multilingual by default for the same reason the default thresholds exist: a question and a
Description written in different languages must not score as unrelated.

The embedding client is a ``bedrock-runtime`` client built on first need, following the SDK's Bedrock path (a
``boto3.Session``, botocore timeouts, ``user_agent_extra="strands-agents"``). ``boto3`` is imported inside that step and
nowhere else in the module, so importing this package and constructing a matcher pull no AWS client into scope.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # Typing only: a runtime import here would open the AWS dependency at construction.
    import boto3

__all__ = ["EmbeddingSimilarityMatcher", "SimilarityMatcher"]

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "cohere.embed-multilingual-v3"
"""Multilingual by default: an English-only model scores two renderings of one subject as unrelated."""

_EmbeddingPurpose = Literal["query", "document"]
"""Which side of the asymmetric comparison a text is: the question, or the Description being searched."""

_COHERE_INPUT_TYPES: dict[str, str] = {"query": "search_query", "document": "search_document"}
"""Maps a purpose onto the ``input_type`` Cohere's embedding models expect."""

_COHERE_MAX_BATCH = 96
"""Texts per Cohere embed call. A longer batch is paged."""

_DEFAULT_CACHE_SIZE = 512
"""Distinct ``(purpose, text)`` pairs kept per matcher.

Bounded rather than unbounded: a long session keeps producing Descriptions, and the graph only ever scores the Cards it
still holds. Descriptions repeat across turns, so even a small cache absorbs nearly all of the traffic.
"""

_DEFAULT_TIMEOUT_SECONDS = 10
"""Connect and read timeout. A call that outlives this fails fast, so the turn degrades instead of stalling."""


class _EmbeddingError(Exception):
    """Embedding is unavailable or its response is unusable.

    Private because it never reaches a caller: :meth:`EmbeddingSimilarityMatcher.score` turns it into an empty
    sequence, which the graph reads as "score nothing, send everything".
    """


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, clamped to ``[0.0, 1.0]``.

    Negative similarity is clamped rather than preserved: the thresholds are expressed on a zero-to-one scale, and for
    that decision "points the other way" and "unrelated" are the same answer.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        Similarity in ``[0.0, 1.0]``. Zero when the lengths differ or either vector has zero magnitude, so a degenerate
        input reads as "unrelated" instead of raising.
    """
    if len(left) != len(right):
        return 0.0

    dot: float = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm: float = sum(a * a for a in left) ** 0.5
    right_norm: float = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0

    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _as_vector(item: Any) -> list[float]:
    """Coerce one response entry into a vector of floats.

    Args:
        item: Entry to coerce.

    Returns:
        The entry as a list of floats.

    Raises:
        _EmbeddingError: When the entry is not a non-empty list of real numbers.
    """
    if not isinstance(item, list) or not item:
        raise _EmbeddingError(f"malformed embedding response: expected a non-empty list, got {type(item).__name__}")

    for value in item:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _EmbeddingError(f"malformed embedding response: vector holds a non-number, {value!r}")

    return [float(value) for value in item]


@runtime_checkable
class SimilarityMatcher(Protocol):
    """Scores how strongly the turn's question relates to each Card's Description."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description received, in the same order.

        Args:
            question: The turn's question. Embedded under the query purpose.
            descriptions: The Cards' Descriptions. Embedded under the document purpose. Must not be mutated, reordered
                or resized.

        Returns:
            Exactly ``len(descriptions)`` values, each in ``[0.0, 1.0]``. An empty sequence signals unavailability, and
            the caller degrades to Full Content everywhere. Implementations must not raise.
        """
        ...


class EmbeddingSimilarityMatcher:
    """Default matcher: an asymmetric multilingual embedding round with a vector cache.

    One embedding round per turn — the question under the query purpose, the Descriptions under the document purpose —
    and within it only the uncached texts reach the provider. The cache is keyed by the ``(purpose, text)`` pair, so an
    unchanged Description costs nothing next turn and the same string is never served as a query vector where a document
    vector was asked for.

    The client is resolved on first need, so construction performs no I/O and a matcher that is never scored (the graph
    below ``min_cards``, for instance) opens no client at all.

    Never raises: a transport failure, a timeout or an unusable response becomes an empty sequence plus one debug log
    carrying ``exc_info``. A caller whose correctness depends on the score should embed on its own and let the error
    through.

    The default ``expand_threshold``, ``collapse_floor`` and ``link_threshold`` are calibrated against *this* matcher's
    score distribution. They are not portable: another embedding model needs its own calibration, and reusing these
    numbers with it is a silent mis-calibration rather than an error.

    Args:
        model_id: Embedding model. Defaults to a multilingual Cohere model; a non-Cohere model is called one text per
            request and has no notion of purpose.
        boto_session: Optional boto3 session. The client is still built on first use, so passing one performs no I/O.
        region_name: Region, used only when no session is supplied.
        client: Pre-built ``bedrock-runtime`` client, taking precedence over the arguments above. Lets an application
            share one client across matchers.
        cache_size: Distinct ``(purpose, text)`` pairs to keep, or ``None`` for no bound.

    Example:
        ```python
        from strands_context_graph import EmbeddingSimilarityMatcher

        matcher = EmbeddingSimilarityMatcher()
        scores = matcher.score("which asset is my largest?", [card.description for card in cards])
        ```
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        *,
        boto_session: boto3.Session | None = None,
        region_name: str | None = None,
        client: Any = None,
        cache_size: int | None = _DEFAULT_CACHE_SIZE,
    ) -> None:
        """Record the embedding configuration without opening a client.

        Args:
            model_id: Embedding model. Defaults to a multilingual model.
            boto_session: Optional boto3 session.
            region_name: Region, used only when no session is supplied.
            client: Pre-built ``bedrock-runtime`` client, taking precedence over the arguments above.
            cache_size: Distinct ``(purpose, text)`` pairs to keep, or ``None`` for no bound.

        Raises:
            ValueError: When ``cache_size`` is neither ``None`` nor an integer greater than or equal to 1.
        """
        if cache_size is not None and (
            isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1
        ):
            raise ValueError(f"cache_size=<{cache_size!r}> | must be None or an integer greater than or equal to 1")

        self.model_id = model_id
        self._session = boto_session
        self._region_name = region_name
        self._client = client
        self._cache_size = cache_size
        # Insertion-ordered so the evicted entry is the least recently used one, and a hit moves to the end.
        self._cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return the cosine similarity of ``question`` against each Description.

        Args:
            question: The turn's question, embedded under the query purpose.
            descriptions: The Cards' Descriptions, embedded under the document purpose. Read only: elements, order and
                size are all left untouched.

        Returns:
            Exactly ``len(descriptions)`` values in ``[0.0, 1.0]``, aligned by index. Empty when the embedding is
            unavailable, which the caller reads as "score nothing, send everything".
        """
        if not descriptions:
            return []

        # Copied before crossing the boundary: the caller's sequence is not ours to hand out.
        texts = list(descriptions)

        try:
            question_vector, description_vectors = self._embed_both(question, texts)
        except _EmbeddingError:
            logger.debug("graph similarity embedding failed for %d description(s)", len(texts), exc_info=True)
            return []

        return [_cosine_similarity(question_vector, vector) for vector in description_vectors]

    def vectors(self, descriptions: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return the document vector of each Description, for the caller's own index.

        Free after :meth:`score`: those vectors are already cached under the document purpose, so no call goes out. The
        similarity Link between two Cards is measured from this index and never from a fresh call, so a caller that
        never fills it can assert no such Link.

        Args:
            descriptions: The Cards' Descriptions. Read only.

        Returns:
            One vector per Description, in order. Empty when the embedding is unavailable, which leaves the index as it
            was and the Link unmeasurable rather than asserted absent.
        """
        if not descriptions:
            return []

        try:
            return self._embed(list(descriptions), "document")
        except _EmbeddingError:
            logger.debug("graph description vectors unavailable for %d text(s)", len(descriptions), exc_info=True)
            return []

    def _embed_both(self, question: str, texts: list[str]) -> tuple[list[float], list[list[float]]]:
        """Embed the question and the Descriptions, overlapping the two round trips.

        Two requests and not one because the asymmetry forces it: the question goes under the query purpose, the
        Descriptions under the document purpose, and one Cohere request carries one ``input_type``.

        Threads rather than the event loop, because :meth:`score` is called from a synchronous hook on the critical path
        and cannot await. The client is materialized first, on this thread: a botocore client is safe to call from
        several threads but is built lazily, and two threads racing to build it is the one hazard here.

        Args:
            question: The turn's question.
            texts: The Descriptions, already copied.

        Returns:
            The question's vector, and one vector per Description in order.

        Raises:
            _EmbeddingError: Propagated from either request, for :meth:`score` to turn into an empty sequence.
        """
        self._ensure_client()

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="context-graph-embed") as pool:
            pending_question = pool.submit(self._embed, [question], "query")
            pending_descriptions = pool.submit(self._embed, texts, "document")
            # The question first: the smaller request, so its failure is reported without waiting on the larger one.
            (question_vector,) = pending_question.result()
            return question_vector, pending_descriptions.result()

    def _ensure_client(self) -> Any:
        """Return the ``bedrock-runtime`` client, building it on first need.

        ``boto3`` and ``botocore`` are imported here and nowhere else, which is what keeps construction free of an AWS
        client.

        Returns:
            The ``bedrock-runtime`` client.

        Raises:
            _EmbeddingError: When the client cannot be built — a missing dependency, an absent region or unresolvable
                credentials all read as "embedding unavailable".
        """
        if self._client is not None:
            return self._client

        try:
            import boto3
            from botocore.config import Config as BotocoreConfig

            session = self._session or boto3.Session(region_name=self._region_name)
            config = BotocoreConfig(
                connect_timeout=_DEFAULT_TIMEOUT_SECONDS,
                read_timeout=_DEFAULT_TIMEOUT_SECONDS,
                user_agent_extra="strands-agents",
            )
            self._client = session.client(service_name="bedrock-runtime", config=config)
        except Exception as error:
            raise _EmbeddingError(f"embedding client unavailable: {error}") from error

        return self._client

    def _embed(self, texts: list[str], purpose: _EmbeddingPurpose) -> list[list[float]]:
        """Embed each text under one purpose, serving the cached pairs without a call.

        Args:
            texts: Texts to embed, in the order the vectors are wanted.
            purpose: Purpose to embed them under.

        Returns:
            One vector per entry of ``texts``, aligned by index.

        Raises:
            _EmbeddingError: When a request fails or a response does not cover every text submitted.
        """
        if not texts:
            return []

        # Deduplicated so a batch repeating a text pays for it once.
        pending: list[str] = []
        for text in texts:
            if (purpose, text) not in self._cache and text not in pending:
                pending.append(text)

        if pending:
            for text, vector in zip(pending, self._invoke(pending, purpose), strict=True):
                self._remember(purpose, text, vector)

        return [self._recall(purpose, text) for text in texts]

    def _remember(self, purpose: str, text: str, vector: list[float]) -> None:
        """Store a vector, evicting the least recently used entry when the cache is full.

        Args:
            purpose: Purpose the text was embedded under.
            text: Text that was embedded.
            vector: Vector to store.
        """
        self._cache[(purpose, text)] = vector
        if self._cache_size is not None:
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _recall(self, purpose: str, text: str) -> list[float]:
        """Return a cached vector, marking it as recently used.

        Args:
            purpose: Purpose the text was embedded under.
            text: Text that was embedded.

        Returns:
            The cached vector.

        Raises:
            _EmbeddingError: When the entry is absent, which means the response did not cover every text submitted.
        """
        key = (purpose, text)
        if key not in self._cache:
            raise _EmbeddingError(f"no embedding returned for text of length {len(text)} under purpose {purpose!r}")

        self._cache.move_to_end(key)
        return self._cache[key]

    def _invoke(self, texts: list[str], purpose: _EmbeddingPurpose) -> list[list[float]]:
        """Call the model for the texts that are not cached.

        Args:
            texts: Texts to embed, none of them cached.
            purpose: Purpose to embed them under.

        Returns:
            One vector per entry of ``texts``, aligned by index.

        Raises:
            _EmbeddingError: When a request fails or a response is unusable.
        """
        if self.model_id.startswith("cohere."):
            vectors: list[list[float]] = []
            for offset in range(0, len(texts), _COHERE_MAX_BATCH):
                batch = texts[offset : offset + _COHERE_MAX_BATCH]
                payload = self._call({"texts": batch, "input_type": _COHERE_INPUT_TYPES[purpose]})
                embeddings = payload.get("embeddings")
                if not isinstance(embeddings, list):
                    raise _EmbeddingError("malformed embedding response: missing 'embeddings'")
                if len(embeddings) != len(batch):
                    raise _EmbeddingError(f"embedding response covered {len(embeddings)} of {len(batch)} texts")
                vectors.extend(_as_vector(item) for item in embeddings)
            return vectors

        # Anything single-text, Titan included: one request per text, and purpose has no equivalent to carry it in.
        single: list[list[float]] = []
        for text in texts:
            payload = self._call({"inputText": text})
            if "embedding" not in payload:
                raise _EmbeddingError("malformed embedding response: missing 'embedding'")
            single.append(_as_vector(payload["embedding"]))
        return single

    def _call(self, body: dict[str, Any]) -> dict[str, Any]:
        """Invoke the model once and parse its JSON response.

        Args:
            body: Request body to serialize.

        Returns:
            The parsed response payload.

        Raises:
            _EmbeddingError: When the request fails, times out, or the response is not a JSON object. Botocore's
                timeouts and client errors land here alongside a malformed body, because the caller's answer to all of
                them is the same one.
        """
        try:
            client = self._ensure_client()
            response = client.invoke_model(modelId=self.model_id, body=json.dumps(body))
            payload: Any = json.loads(response["body"].read())
        except _EmbeddingError:
            raise
        except Exception as error:
            raise _EmbeddingError(f"embedding call failed: {error}") from error

        if not isinstance(payload, dict):
            raise _EmbeddingError(f"malformed embedding response: expected an object, got {type(payload).__name__}")

        return payload
