"""The plugin's own reference store: the coupling inverted.

The integrated build reads an offloader's Stash and is a guest in it. Standalone, the plugin owns a store that is
**always present** (Requirement 15.1), and ``ContextManager`` interop becomes an optional fallback bridge. What this
module owns is the contract, the default, and the resolution order ``tools.expand_artifact`` reads through: own store
first, the Stash second when a manager happens to be registered, prose naming the miss last (Requirements 15.2, 15.3,
15.4).

Three properties are load-bearing:

- **A Card keeps the reference, the store keeps the block.** Nothing in the graph can rot when the underlying content
  changes, because the graph never holds a copy of it (Requirements 15.1, 3.7).
- **The block is decoded, never the raw return.** What goes in is what a retrieval hands back — already-decoded text, or
  a block that yields no text at all. A block that yields none is reported without a media type, since a decoded block
  carries none to name (Requirement 15.7); that reporting is the caller's, and the store stays neutral about it.
- **Zero offloading is the ordinary path, not a degradation.** With nothing replacing a tool return, no reference is
  discovered, the store stays empty, there is no artifact Card, and the subject Cards are untouched — no exception, and
  no branch asking whether an offloader is installed (Requirement 15.9).

A reference reaches the store from two sites, both of which call :func:`record_references`: the ``AfterToolCallEvent``
fast path, which may also carry the decoded blocks, and the preview-scan, which carries only the names it read off the
placeholder. The second case records a **known reference with no block**: ``retrieve`` returns ``None`` for it, so
resolution falls through to the bridge exactly as an unknown reference does. Knowing the name buys nothing at resolution
time and is kept only because a store that forgets a reference it has seen cannot be told apart from one that never saw
it — a distinction worth having in a log, never in a decision.

The default store lives for as long as its plugin instance and is never persisted: like the graph state, it is discarded
with the agent. It applies no eviction, so a process that offloads without bound holds the decoded blocks without bound.
That is the same exposure the host offloader already has, and a cap here would make a reference silently stop resolving
against its own store — worse than memory the caller can see.

**Private-API dependencies** (Requirements 17.3, 17.4). Every private symbol this module reads is *soft*: none is
imported statically, all four go through :func:`_optional_symbol`, and a symbol that does not resolve — or that resolves
under its old name and then raises, its shape having changed — is treated as absent. What each one costs when it goes:

- ``ContextManager`` / ``agent._plugin_registry._plugins`` / ``_stash`` — **no Stash interop.** The bridge is built
  entirely on these three, so any of them is enough to switch it off, which leaves the plugin standalone: a supported
  configuration, and the only one on an agent with no offloader. Every reference the plugin recorded itself still
  resolves; a Stash-only reference answers as prose naming the miss. A public reference-store protocol on
  ``ContextManager`` would remove all three at once.
- ``_extract_text`` — **bare strings only.** A decoded ``str`` is still read as the text it plainly is, so own-store
  reads keep working; anything richer reports as non-textual rather than being decoded here, because decoding a block is
  the manager's job and guessing at it is how a reference resolves to the wrong content. A public text-recovery helper,
  or the same protocol, would remove it.
- ``_search_content`` — **whole reads only.** A ``line_range``/``pattern`` request degrades to prose saying targeted
  reads are unavailable; reads of the whole reference are unaffected, since they never needed the helper. A public
  artifact-search helper would remove it.

No branch here raises on a missing or misbehaving symbol (Requirement 16.8), and the README's private-API table lists
these alongside the couplings of the delivery path, which are of the other shape.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "ArtifactOutcome",
    "InMemoryReferenceStore",
    "ReferenceStore",
    "ResolvedArtifact",
    "absent_message",
    "estimate_tokens",
    "non_textual_message",
    "read_artifact",
    "record_references",
    "resolve_artifact",
    "unknown_message",
]

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
"""Characters per token, the same coarse estimate ``describe.py`` and the Stash's retrieval tool use. Only ever used to
*report* a cost back to the model, never to decide what is returned."""

_CONTEXT_LINES = 5
"""Lines kept around each pattern match, the default the manager's own retrieval tool applies."""

_MAX_RESULT_TOKENS = 10_000
"""Output ceiling of a targeted read (Requirement 15.8), the same budget the ``ContextManager``'s retrieval tool
applies. Sharing the ceiling keeps a retrieval from re-offloading itself: an answer built to the same bound is left
alone. A whole read is deliberately *not* bounded here — it is bounded by having been asked for, and the caller states
its cost back to the model."""

_EXTRACT_TEXT_SYMBOL = ("strands._context_manager.retrieval_tool", "_extract_text")
"""Where the manager's text recovery lives. Looked up by name, never imported statically, so a rename degrades."""

_SEARCH_CONTENT_SYMBOL = ("strands.vended_plugins.context_offloader.search", "_search_content")
"""Where the offloader's line/pattern matching lives. Same treatment: reused, never reimplemented, never assumed."""

_CONTEXT_MANAGER_SYMBOL = ("strands._context_manager.context_manager", "ContextManager")
"""The optional bridge's anchor type. Absent it, the plugin is simply standalone."""


@runtime_checkable
class ReferenceStore(Protocol):
    """What the plugin needs of a reference store: read one block, write one block.

    ``retrieve`` is asynchronous and ``put`` is not, and the asymmetry is the point: a read may cross a process boundary
    (a bucket, a cache, a Stash), while a write happens on the ``AfterToolCallEvent`` fast path, which the agent loop
    awaits. An implementation whose write is remote should queue it, not block the hook.

    Both halves are total: an unknown reference is ``None``, never an exception, because the caller turns a miss into
    prose for the model and an exception would report the *tool* broken rather than the *request*.
    """

    async def retrieve(self, reference: str) -> object | None:
        """Return the decoded block behind ``reference``, or ``None`` when this store does not hold it."""
        ...

    def put(self, reference: str, block: object) -> None:
        """Record ``block`` under ``reference``, replacing any block already recorded under it."""
        ...


class InMemoryReferenceStore:
    """The default store: a dict of decoded blocks, present in every configuration.

    Absent a ``block``, a reference is still recorded, as a name with nothing behind it (see the module docstring). The
    two states are told apart by ``in`` versus the result of :meth:`retrieve`: a known reference with no block is
    ``reference in store`` and ``await store.retrieve(reference) is None``.
    """

    def __init__(self) -> None:
        """Start empty, which is also the steady state when nothing on the agent offloads."""
        self._blocks: dict[str, object | None] = {}

    async def retrieve(self, reference: str) -> object | None:
        """Return the decoded block behind ``reference``.

        Args:
            reference: The key a placeholder named.

        Returns:
            The decoded block, or ``None`` when the reference is unknown or known with no block. The caller treats both
            the same way: fall through to the optional bridge, then to prose naming the miss.
        """
        return self._blocks.get(reference)

    def put(self, reference: str, block: object) -> None:
        """Record ``block`` under ``reference``.

        Args:
            reference: The key a placeholder named.
            block: The decoded block. Re-recording a reference replaces what was there, since the reference is the
                identity and the store is not a history.
        """
        self._blocks[reference] = block

    def note(self, reference: str) -> None:
        """Record ``reference`` as known with no block, the most the preview-scan can state.

        Args:
            reference: The key read off the placeholder. A reference that already has a block keeps it: a preview naming
                what the fast path already stored must not erase it.
        """
        self._blocks.setdefault(reference, None)

    def references(self) -> frozenset[str]:
        """Return every reference the store knows, with or without a block."""
        return frozenset(self._blocks)

    def __contains__(self, reference: object) -> bool:
        """Whether ``reference`` is known, regardless of whether a block sits behind it."""
        return reference in self._blocks

    def __len__(self) -> int:
        """How many references are known, with or without a block."""
        return len(self._blocks)


def record_references(
    store: ReferenceStore,
    references: Iterable[str],
    blocks: Sequence[object] = (),
) -> tuple[str, ...]:
    """Populate ``store`` from one offloaded tool result, or from one preview-scan.

    The two population sites differ only in what they can offer. The ``AfterToolCallEvent`` fast path may hand the
    decoded blocks alongside the references; the preview-scan has read the names off placeholder text and has no block
    to give. Blocks are paired with references **positionally, and only when the two counts agree**, which is the one
    pairing a placeholder actually states — the offloader writes one reference per offloaded block, in order. On any
    other count the references are recorded without blocks rather than guessed at: a wrong pairing would resolve a
    reference to someone else's content, which is worse in every direction than a miss.

    Never raises (Requirement 16.8). A store whose write fails logs one warning and leaves the remaining references to
    the next attempt; nothing here can stop the hook from completing.

    Args:
        store: The store to populate. Mutated.
        references: References named by the placeholder, in order of appearance. Empty on the nothing-offloaded path,
            which records nothing and is not a failure (Requirement 15.9).
        blocks: Decoded blocks, in the same order, when the caller has them. Empty from the preview-scan.

    Returns:
        The references recorded, in order, without duplicates. Empty when there was nothing to record, or when every
        write failed.
    """
    ordered: list[str] = []
    for reference in references:
        if reference and reference not in ordered:
            ordered.append(reference)

    if not ordered:
        return ()

    paired = blocks if len(blocks) == len(ordered) else ()
    if blocks and not paired:
        logger.debug(
            "reference count %d does not match block count %d | recording references without blocks",
            len(ordered),
            len(blocks),
        )

    recorded: list[str] = []
    for index, reference in enumerate(ordered):
        try:
            if paired:
                store.put(reference, paired[index])
            elif isinstance(store, InMemoryReferenceStore):
                store.note(reference)
            else:
                # A foreign store has no notion of a name with nothing behind it, and inventing one by writing a
                # sentinel block would make a miss resolve. The reference stays unrecorded there.
                continue
        except Exception:
            logger.warning("reference=<%s> | reference store write failed", reference, exc_info=True)
            continue
        recorded.append(reference)

    return tuple(recorded)


# ---- resolution: own store first, the optional Stash bridge second -----------------------------

ArtifactOutcome = Literal["text", "non_textual", "unknown", "absent"]
"""What a resolution attempt ended as.

- ``"text"``: a block was found and it yields text — the only outcome with content to read.
- ``"non_textual"``: a block was found and it yields no text. Reported *without* a media type, whichever source it came
  from, because a decoded block carries none to name (Requirement 15.7).
- ``"unknown"``: a bridge was available and did not hold the reference.
- ``"absent"``: nothing beyond the own store was available to ask — no ``ContextManager``, a manager with
  ``stash=False``, or a registry that could not be read. One outcome for all three (Requirement 15.4), because to the
  model they are one situation: the reference did not resolve.
"""


@dataclass(frozen=True)
class ResolvedArtifact:
    """The result of resolving one reference, before anything is read out of it.

    Attributes:
        outcome: What the attempt ended as.
        text: The recovered text, present only for ``outcome == "text"``.
        source: Which store answered — ``"own"`` or ``"stash"`` — or ``None`` when neither did. Kept for the log and for
            tests of the routing; the answer the model receives does not depend on it.
    """

    outcome: ArtifactOutcome
    text: str | None = None
    source: Literal["own", "stash"] | None = None


async def resolve_artifact(store: ReferenceStore, agent: object, reference: str) -> ResolvedArtifact:
    """Resolve ``reference`` in the plugin's own store, then across the optional ``ContextManager`` bridge.

    The order is the inverted coupling (Requirements 15.2, 15.3): the store the plugin owns is always present and always
    asked first, and the Stash is consulted only for what the store does not hold. Never raises — every failure of every
    layer collapses onto an outcome, because the caller turns an outcome into prose and an exception would report the
    *tool* broken rather than the *request*.

    Args:
        store: The plugin's own store. Read only.
        agent: The agent of the call, for the optional bridge. Read only, and typed loosely on purpose: the bridge is
            found by walking a private registry that may not exist at all.
        reference: The reference as it was shown to the model.

    Returns:
        The resolution. ``"absent"`` and ``"unknown"`` both mean "no content", and the caller words them differently
        only because the first says the storage was never there.
    """
    block = await _retrieve(store, reference, source_name="own")
    if block is not None:
        return _resolved(block, "own")

    stash = _stash_of(agent)
    if stash is None:
        return ResolvedArtifact(outcome="absent")

    block = await _retrieve(stash, reference, source_name="stash")
    if block is None:
        return ResolvedArtifact(outcome="unknown")

    return _resolved(block, "stash")


def read_artifact(text: str, *, line_range: tuple[int, int] | None = None, pattern: str | None = None) -> str:
    """Read the resolved ``text`` whole, or the part ``line_range``/``pattern`` name.

    A whole read is the text itself, character for character (Requirement 15.5): no model call, no truncation, no
    reformatting. A targeted read is delegated to the offloader's ``_search_content`` and bounded by
    ``_MAX_RESULT_TOKENS`` (Requirements 15.6, 15.8), so the plugin reimplements neither the matching nor the bound.

    Args:
        text: The resolved text.
        line_range: ``(start, end)``, 1-indexed and inclusive, or ``None``. The caller parses the model's mapping into
            this pair, since only the caller knows whether a malformed range was supplied or none was.
        pattern: Regex or keyword keeping only matching lines, or ``None``.

    Returns:
        The text verbatim for a whole read, or the formatted matching lines for a targeted one.

    Raises:
        ValueError: When the range falls outside the content, when the search helper is unavailable in the installed
            SDK, or when the helper that resolved no longer accepts the keywords passed here. All three are conditions
            the caller already words as prose, so they travel as one exception type.
    """
    if line_range is None and pattern is None:
        return text

    search = _search_helper()
    if search is None:
        raise ValueError(
            "targeted reads are unavailable against this SDK build | read the reference whole instead, or pass no "
            "line_range and no pattern"
        )

    try:
        return search(
            text,
            pattern=pattern,
            line_range=line_range,
            context_lines=_CONTEXT_LINES,
            max_chars=_MAX_RESULT_TOKENS * _CHARS_PER_TOKEN,
        )
    except ValueError:
        # The helper's own refusal — a range outside the content — already carries the words the caller needs.
        raise
    except Exception as error:
        # A helper that resolved under its old name but no longer takes these keywords. Worded like an absent helper
        # rather than propagated: from the model's side a targeted read is simply unavailable, and a private symbol
        # changing shape must not surface as a broken tool (Requirement 16.8).
        logger.debug("targeted read failed inside the SDK search helper | degrading to prose", exc_info=True)
        raise ValueError(
            "targeted reads are unavailable against this SDK build | read the reference whole instead, or pass no "
            "line_range and no pattern"
        ) from error


def estimate_tokens(text: str) -> int:
    """Coarse token count of ``text``, for stating the cost of a whole read back to the model.

    Args:
        text: The text about to be returned.

    Returns:
        At least ``1``, so an answer never reports costing nothing.
    """
    return max(1, len(text) // _CHARS_PER_TOKEN)


def absent_message(reference: str) -> str:
    """The one message covering every way there was nothing beyond the own store to ask (Requirement 15.4).

    No ``ContextManager``, a manager configured with ``stash=False``, and a registry that cannot be read are one
    message, because they differ only in a detail the model can act on in no way.

    Args:
        reference: The reference asked for, named back so the model can tell which of its requests missed.

    Returns:
        Prose naming the reference and the absence.
    """
    return (
        f"expand_artifact | no artifact storage holds reference '{reference}' on this agent | nothing was ever "
        "offloaded under that reference, which means the full results are already in the conversation"
    )


def unknown_message(reference: str) -> str:
    """The message for a reference that storage was asked for and did not hold.

    Args:
        reference: The reference asked for.

    Returns:
        Prose naming the reference and how to name one correctly.
    """
    return (
        f"expand_artifact | unknown reference '{reference}' | copy a reference exactly as it was shown to you in a "
        "turn's title or preview"
    )


def non_textual_message(reference: str) -> str:
    """The message for a block that yields no text — **without naming a media type** (Requirement 15.7).

    A decoded block carries no media type to name, and inventing one would be a guess the model would then repeat. The
    rule is the same whether the block came from the plugin's own store or from the Stash.

    Args:
        reference: The reference asked for.

    Returns:
        Prose naming the reference and stating that the content is not text.
    """
    return (
        f"expand_artifact | reference '{reference}' holds non-textual content | line_range and pattern do not apply to "
        "it, and it cannot be returned as text"
    )


def _resolved(block: object, source: Literal["own", "stash"]) -> ResolvedArtifact:
    """Turn a found block into a resolution, recovering its text through the manager's helper.

    Args:
        block: The decoded block the store or the Stash returned.
        source: Which of the two answered.

    Returns:
        A ``"text"`` resolution, or a ``"non_textual"`` one when the block yields none.
    """
    text = _extract_text(block)
    if text is None:
        return ResolvedArtifact(outcome="non_textual", source=source)
    return ResolvedArtifact(outcome="text", text=text, source=source)


async def _retrieve(source: object, reference: str, *, source_name: str) -> object | None:
    """Read ``reference`` through ``source``, or answer ``None``.

    Args:
        source: The own store or the Stash — anything with an awaitable ``retrieve``.
        reference: The reference to read.
        source_name: Which one it is, for the debug log only.

    Returns:
        The block, or ``None``. Every failure collapses here — an unknown reference, uninitialized storage, an
        unreachable backend, a ``retrieve`` that is not even awaitable — because from the model's side they are one
        situation. The distinction stays in the debug log.
    """
    try:
        retrieve = getattr(source, "retrieve", None)
        if not callable(retrieve):
            return None
        block: object | None = await retrieve(reference)
    except Exception:
        logger.debug(
            "reference=<%s> source=<%s> | did not resolve | continuing the resolution order",
            reference,
            source_name,
            exc_info=True,
        )
        return None
    return block


def _stash_of(agent: object) -> object | None:
    """The ``ContextManager``'s Stash on ``agent``, or ``None`` when there is none to read.

    Found by type over the agent's plugin registry rather than held as a constructor argument, so the plugin bridges to
    a manager it was not told about. ``None`` covers a manager configured with ``stash=False``, a registry that cannot
    be read, and an SDK build where ``ContextManager`` no longer resolves — each of which leaves the plugin standalone,
    a supported configuration and not a failure.

    Args:
        agent: The agent of the call. Read only.

    Returns:
        The Stash, or ``None``.
    """
    manager_type = _optional_symbol(*_CONTEXT_MANAGER_SYMBOL)
    if not isinstance(manager_type, type):
        return None

    registry = getattr(agent, "_plugin_registry", None)
    plugins = getattr(registry, "_plugins", None)
    if not isinstance(plugins, dict):
        return None

    for plugin in plugins.values():
        if isinstance(plugin, manager_type):
            return getattr(plugin, "_stash", None)
    return None


def _extract_text(block: object) -> str | None:
    """Recover text from a decoded block through the manager's ``_extract_text`` (Requirement 15.6).

    Args:
        block: The decoded block.

    Returns:
        The text, or ``None`` when the block yields none.

    Note:
        When the helper does not resolve, a decoded ``str`` is still read as the text it plainly is, so an own-store
        read keeps working on an SDK build that renamed the helper. Everything richer than a bare string is reported as
        non-textual rather than decoded here, since decoding a block is the manager's job and guessing at it is how a
        reference starts resolving to the wrong content. A helper that resolved but *raises* — the shape of the block it
        accepts having changed under the same name — is treated the same way, so the bridge degrades in one direction
        only (Requirement 17.3).
    """
    helper = _optional_symbol(*_EXTRACT_TEXT_SYMBOL)
    if callable(helper):
        try:
            text = helper(block)
        except Exception:
            logger.debug("the SDK text-recovery helper raised | falling back to the bare-string reading", exc_info=True)
        else:
            return text if isinstance(text, str) else None
    return block if isinstance(block, str) else None


def _search_helper() -> Callable[..., str] | None:
    """The offloader's ``_search_content``, or ``None`` when it does not resolve.

    Returns:
        The helper, or ``None`` — in which case targeted reads degrade to prose and whole reads keep working.
    """
    helper = _optional_symbol(*_SEARCH_CONTENT_SYMBOL)
    return helper if callable(helper) else None


def _optional_symbol(module: str, name: str) -> Any | None:
    """Look up one optional private SDK symbol by name, answering ``None`` rather than raising.

    Resolved by name at call time, never by a static import, which is the whole of the hardening: a rename, a moved
    module, or an SDK build that never had the symbol all degrade to "no Stash interop" instead of an ``ImportError`` at
    import time (Requirement 16.8). The import cost is also not paid by an agent that never expands an artifact.

    Args:
        module: Dotted module path.
        name: Attribute to read off it.

    Returns:
        The symbol, or ``None``.
    """
    try:
        return getattr(import_module(module), name, None)
    except Exception:
        logger.debug(
            "module=<%s> symbol=<%s> | optional SDK symbol did not resolve | continuing without it",
            module,
            name,
            exc_info=True,
        )
        return None
