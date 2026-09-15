"""AWS documentation corpus used as the payload of the mocked tools.

Real documentation matters here for two reasons. First, the prefix preview only looks
bad on text whose opening is boilerplate — navigation chrome, breadcrumbs, "was this
page helpful" — which is exactly what AWS doc pages start with. Synthetic lorem ipsum
would flatter the baseline. Second, the reranker has to score prose that actually
varies in relevance, not shuffled tokens.

Pages are downloaded once and cached under ``.cache/``, so every configuration in a
comparison run sees byte-identical payloads and the numbers are comparable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

from .config import CACHE_DIR

logger = logging.getLogger(__name__)

_DOC_SOURCES: dict[str, str] = {
    "lambda_invocation": "https://docs.aws.amazon.com/lambda/latest/dg/lambda-invocation.html",
    "lambda_foundation": "https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtime-environment.html",
    "s3_naming": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html",
    "s3_security": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html",
    "dynamodb_capacity": "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ReadWriteCapacityMode.html",
    "dynamodb_partition": "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.CoreComponents.html",
    "vpc_subnets": "https://docs.aws.amazon.com/vpc/latest/userguide/configure-subnets.html",
    "iam_best_practices": "https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html",
    "bedrock_inference": "https://docs.aws.amazon.com/bedrock/latest/userguide/inference-parameters.html",
    "rds_backups": "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_WorkingWithAutomatedBackups.html",
    "eks_networking": "https://docs.aws.amazon.com/eks/latest/userguide/eks-networking.html",
    "cloudwatch_alarms": "https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/AlarmThatSendsEmail.html",
}
"""Twelve pages across seven services. Enough distinct subject matter that a reranker
scoring one question against all of them has a real discrimination job to do."""

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


def _html_to_text(html: str) -> str:
    """Strip markup while keeping line structure.

    Line structure is load-bearing: the offloader's chunker splits on line boundaries
    and reports 1-indexed line numbers back to the model, so collapsing newlines here
    would erase the very coordinates the retrieval tool accepts.
    """
    without_code = _SCRIPT_RE.sub(" ", html)
    text = _TAG_RE.sub("", without_code)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    text = _WS_RE.sub(" ", text)
    return _BLANK_RE.sub("\n\n", text).strip()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.txt"


def _fetch(key: str, url: str) -> str:
    """Download and cache one page. Falls back to a stub when the network is unavailable."""
    path = _cache_path(key)
    if path.exists():
        return path.read_text(encoding="utf-8")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        response = httpx.get(
            url,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "strands-context-validation/1.0"},
        )
        response.raise_for_status()
        text = _html_to_text(response.text)
    except Exception:
        logger.warning("could not download %s; using offline stub", url, exc_info=True)
        text = f"[offline stub for {key}] {url}\n" + ("AWS documentation content unavailable.\n" * 200)

    path.write_text(text, encoding="utf-8")
    return text


def load(key: str) -> str:
    """Return the cached text of one documentation page."""
    if key not in _DOC_SOURCES:
        raise KeyError(f"unknown corpus key: {key!r}")
    return _fetch(key, _DOC_SOURCES[key])


def load_all() -> dict[str, str]:
    """Return every page, downloading and caching on first call."""
    return {key: _fetch(key, url) for key, url in _DOC_SOURCES.items()}


def warm() -> dict[str, int]:
    """Populate the cache and report the character count of each page."""
    return {key: len(text) for key, text in load_all().items()}


def concatenated(*keys: str, min_chars: int = 0) -> str:
    """Join several pages, repeating the sequence until ``min_chars`` is reached.

    Repetition is how a payload is pushed past ``max_result_tokens`` without inventing
    filler: the text stays real AWS prose, and the reranker still has to locate the
    relevant passage inside a much larger haystack.
    """
    parts = [f"===== {key} =====\n{load(key)}" for key in keys]
    body = "\n\n".join(parts)
    if min_chars <= 0 or len(body) >= min_chars:
        return body

    chunks = [body]
    total = len(body)
    round_no = 2
    while total < min_chars:
        appendix = f"\n\n===== appendix (pass {round_no}) =====\n{body}"
        chunks.append(appendix)
        total += len(appendix)
        round_no += 1
    return "".join(chunks)
