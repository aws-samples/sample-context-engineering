"""Playwright-backed web fetch tool.

This is the harness's most honest test of the preview strategies. A rendered page is
not prose: it opens with cookie banners, nav trees and "was this page helpful", and the
passage that answers the question sits somewhere in the middle. A positional prefix
preview spends its entire budget on the chrome. A relevance preview has to find the
middle.

A real browser is used rather than an HTTP fetch because the target pages render their
body client-side; an HTTP GET returns a shell, which would understate the payload and
remove the very structure that makes the comparison meaningful.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from strands import tool

from .config import CACHE_DIR

logger = logging.getLogger(__name__)

_BROWSER_TIMEOUT_MS = 45_000
_WS_RE = re.compile(r"[ \t]{2,}")
_BLANK_RE = re.compile(r"\n{3,}")

_ALLOWED_PREFIXES = (
    "https://docs.aws.amazon.com/",
    "https://aws.amazon.com/",
    "https://www.amazon.com/",
    "https://boto3.amazonaws.com/",
)
"""Public documentation and retail pages only.

The allowlist is not decoration: the tool is driven by model output, and a browser that
follows an arbitrary model-supplied URL is a request-forgery primitive. Restricting the
scheme to HTTPS and the host to known public documentation keeps the tool inert as an
attack surface while still exercising real network content.
"""


def _cache_path(url: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", url)[:120].strip("-")
    return CACHE_DIR / f"web-{slug}.txt"


def _tidy(text: str) -> str:
    """Normalize whitespace while preserving line structure for the chunker."""
    return _BLANK_RE.sub("\n\n", _WS_RE.sub(" ", text)).strip()


async def _render(url: str) -> str:
    """Load ``url`` in headless Chromium and return the rendered text."""
    from playwright.async_api import async_playwright

    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
                ),
                viewport={"width": 1440, "height": 900},
            )
            await page.goto(url, timeout=_BROWSER_TIMEOUT_MS, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=8_000)
            except Exception:
                pass  # networkidle is a nicety; domcontentloaded already gave us a body
            body = await page.inner_text("body")
            title = await page.title()
        finally:
            await browser.close()

    return f"TITLE: {title}\nURL: {url}\n\n{_tidy(body)}"


@tool
async def fetch_web_page(url: str, refresh: bool = False, max_chars: int = 250_000) -> str:
    """Download a public web page with a real browser and return its rendered text.

    The page is rendered in headless Chromium, so client-side content is included. The
    result is the visible text of the whole document — navigation, body and footer — in
    document order, which is typically tens of thousands of characters.

    Only public AWS documentation and amazon.com pages can be fetched. Any other host is
    rejected without a network request.

    Args:
        url: Absolute HTTPS URL to fetch. Must be under docs.aws.amazon.com,
            aws.amazon.com, boto3.amazonaws.com or www.amazon.com.
        refresh: When true, bypasses the local cache and re-renders the page. Defaults
            to false, which serves a previously rendered copy when one exists.
        max_chars: Truncate the returned text at this many characters, between 1000 and
            500000. Defaults to 250000.
    """
    if not any(url.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
        return (
            f"refused: {url!r} is not an allowed target. "
            f"Allowed prefixes: {', '.join(_ALLOWED_PREFIXES)}"
        )

    limit = max(1_000, min(int(max_chars), 500_000))
    path = _cache_path(url)

    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")[:limit]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        text = await asyncio.wait_for(_render(url), timeout=90)
    except Exception as error:  # noqa: BLE001 - a fetch failure must not break the run
        logger.warning("playwright fetch failed for %s", url, exc_info=True)
        if path.exists():
            return path.read_text(encoding="utf-8")[:limit]
        return f"fetch failed for {url}: {type(error).__name__}: {error}"

    path.write_text(text, encoding="utf-8")
    return text[:limit]


async def prewarm(urls: list[str]) -> dict[str, int]:
    """Render each URL once so a comparison run serves identical bytes to every config."""
    sizes: dict[str, int] = {}
    for url in urls:
        text = await fetch_web_page(url)  # type: ignore[misc]
        sizes[url] = len(text)
    return sizes
