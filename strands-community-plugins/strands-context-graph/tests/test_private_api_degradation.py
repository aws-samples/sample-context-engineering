"""What happens when a private SDK symbol is renamed, moved, or changes shape under its own name.

``test_store.py`` already covers the *absent* symbol on the Stash bridge — a lookup that answers ``None``. What is
asserted here is the other half of the same promise (Requirements 16.8, 17.3): a symbol that still resolves and then
*misbehaves*, which is what a signature change looks like from this side, degrades exactly as an absent one does. A
rename is the easy case; a symbol that resolves and raises is the one that reaches production.

Four surfaces, one rule each:

- ``_extract_text`` raises: bare strings still read as text, richer blocks report as non-textual, nothing propagates.
- ``_search_content`` raises: targeted reads degrade to the same prose an absent helper produces, and its own
  ``ValueError`` — a range outside the content — keeps its words instead of being flattened into that prose.
- ``MiddlewareRegistry._handlers`` is unreadable: the delivery stays registered and reports that it is not first, so the
  ordering notice fires. Registration never raises.
- The Stash bridge's ``_stash``: a manager that does not carry it leaves the plugin standalone, not broken.
"""

import warnings
import weakref

import pytest
from strands._middleware.registry import MiddlewareRegistry
from strands._middleware.stages import InvokeModelStage
from strands.agent.conversation_manager import NullConversationManager

from strands_context_graph import store as store_module
from strands_context_graph.plugin import ContextGraph
from strands_context_graph.projection import Projection

REFERENCE = "mem_1_tu-3_0"
"""One reference, shaped like a real placeholder."""


def _symbol_table(overrides):
    """A ``_optional_symbol`` replacement answering from ``overrides`` and ``None`` for everything else."""

    def _lookup(module, name):
        return overrides.get((module, name))

    return _lookup


def _exploding(*_args, **_kwargs):
    """A private helper that resolved under its old name and no longer accepts what it is handed."""
    raise TypeError("unexpected keyword argument")


class _Stash:
    """A stand-in for a ContextManager's Stash: one async read, nothing else."""

    def __init__(self, blocks=None):
        self.blocks = blocks or {}

    async def retrieve(self, reference):
        """Return the block behind ``reference``, or ``None``."""
        return self.blocks.get(reference)


class _ManagerWithoutStash:
    """A ContextManager on a build where the Stash moved out from under ``_stash``."""


class _Registry:
    def __init__(self, plugins):
        self._plugins = plugins


class _Agent:
    def __init__(self, plugins=None):
        if plugins is not None:
            self._plugin_registry = _Registry(plugins)


# ---- the text-recovery helper raises instead of answering ---------------------------------------


@pytest.mark.asyncio
async def test_a_raising_text_helper_still_reads_a_bare_string(monkeypatch):
    """The fallback reading applies to a helper that misbehaves, not only to one that is absent."""
    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._EXTRACT_TEXT_SYMBOL: _exploding}),
    )
    store = store_module.InMemoryReferenceStore()
    store.put(REFERENCE, "decoded text")

    resolved = await store_module.resolve_artifact(store, _Agent(), REFERENCE)

    assert resolved.outcome == "text"
    assert resolved.text == "decoded text"
    assert resolved.source == "own"


@pytest.mark.asyncio
async def test_a_raising_text_helper_reports_a_richer_block_as_non_textual(monkeypatch):
    """Decoding a block is the manager's job: with its helper broken, the block is reported, never guessed at."""
    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._EXTRACT_TEXT_SYMBOL: _exploding}),
    )
    store = store_module.InMemoryReferenceStore()
    store.put(REFERENCE, {"image": {"format": "png"}})

    resolved = await store_module.resolve_artifact(store, _Agent(), REFERENCE)

    assert resolved.outcome == "non_textual"
    assert resolved.text is None


# ---- the search helper raises instead of answering ----------------------------------------------


def test_a_raising_search_helper_degrades_to_the_same_prose_as_an_absent_one(monkeypatch):
    """A changed signature and a rename are one situation to the model: targeted reads are unavailable."""
    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._SEARCH_CONTENT_SYMBOL: _exploding}),
    )

    with pytest.raises(ValueError, match="targeted reads are unavailable"):
        store_module.read_artifact("a\nb\nc", pattern="b")


def test_a_raising_search_helper_leaves_a_whole_read_alone(monkeypatch):
    """A whole read never needed the helper, so nothing about it degrades."""
    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._SEARCH_CONTENT_SYMBOL: _exploding}),
    )

    assert store_module.read_artifact("a\nb\nc") == "a\nb\nc"


def test_the_search_helpers_own_refusal_keeps_its_words(monkeypatch):
    """A range outside the content is the helper answering correctly, not a private symbol having moved."""

    def _refusing(*_args, **_kwargs):
        raise ValueError("line range 9-10 is outside the content")

    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._SEARCH_CONTENT_SYMBOL: _refusing}),
    )

    with pytest.raises(ValueError, match="outside the content"):
        store_module.read_artifact("a\nb\nc", line_range=(9, 10))


# ---- the Stash bridge's own anchor moved -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_manager_that_no_longer_carries_the_stash_is_absent(monkeypatch):
    """``_stash`` is a third of the bridge: without it the plugin is standalone, which is not a failure."""
    monkeypatch.setattr(
        store_module,
        "_optional_symbol",
        _symbol_table({store_module._CONTEXT_MANAGER_SYMBOL: _ManagerWithoutStash}),
    )
    agent = _Agent({"cm": _ManagerWithoutStash()})

    resolved = await store_module.resolve_artifact(store_module.InMemoryReferenceStore(), agent, REFERENCE)

    assert resolved.outcome == "absent"
    assert resolved.source is None


@pytest.mark.asyncio
async def test_the_own_store_answers_with_the_bridge_switched_off(monkeypatch):
    """The point of the inverted coupling: no bridge at all costs no reference the plugin recorded itself."""
    monkeypatch.setattr(store_module, "_optional_symbol", _symbol_table({}))
    store = store_module.InMemoryReferenceStore()
    store.put(REFERENCE, "decoded text")

    resolved = await store_module.resolve_artifact(store, _Agent({"cm": _ManagerWithoutStash()}), REFERENCE)

    assert resolved.outcome == "text"
    assert resolved.text == "decoded text"


# ---- the middleware registry's private handler map moved ---------------------------------------


class _MemoryManager:
    """The two members the ordering detection reads off a memory manager."""

    def __init__(self):
        self._injection_config = {}

    async def _provide_memory_context(self, messages, config):
        """The render callback that makes this object a memory manager, by member."""
        return "remembered"


class _RenamedMapRegistry:
    """A registry of an SDK release that renamed its private handler map.

    The public ``add_middleware`` still works — that is the point of it being public — so registration succeeds and only
    the move to the front of the stage, which reads the private name, has nothing to read.
    """

    def __init__(self):
        self._inner = MiddlewareRegistry()

    def add_middleware(self, stage_or_phase, handler):
        """Register through the public surface, exactly as the real registry does."""
        return self._inner.add_middleware(stage_or_phase, handler)

    @property
    def _middleware_handlers(self):
        """Where this build keeps what used to live under ``_handlers``."""
        return self._inner._handlers


class _WiringAgent:
    """The members ``init_agent`` reads, over whichever registry the case under test needs."""

    def __init__(self, plugins=None, registry=None):
        self.messages = []
        self.state = None
        self._middleware_registry = registry if registry is not None else MiddlewareRegistry()
        self._plugin_registry = _Registry(dict(plugins or {}))
        self.memory_manager = None
        self.conversation_manager = NullConversationManager()
        self.hooks = []

    def add_hook(self, callback, event_type=None):
        """Record the hook registration, which is all this needs of it."""
        self.hooks.append((event_type, callback))


def test_registration_survives_an_unreadable_handler_map():
    """The public ``add_middleware`` did the registering; only the move to index zero reads the private map."""
    agent = _WiringAgent(registry=_RenamedMapRegistry())

    assert Projection(weakref.WeakKeyDictionary(), description_tokens=100, retrieval_tools=lambda: ("expand_card", "expand_artifact", "find_context")).register(agent) is False
    assert agent._middleware_registry._middleware_handlers[InvokeModelStage]


def test_an_unreadable_handler_map_warns_about_the_ordering():
    """Position unprovable is position not claimed: the notice fires rather than the wiring crashing."""
    agent = _WiringAgent(plugins={"strands:memory": _MemoryManager()}, registry=_RenamedMapRegistry())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ContextGraph().init_agent(agent)

    assert len(caught) == 1
    assert "InvokeModelStage" in str(caught[0].message)


def test_an_unreadable_handler_map_stays_silent_with_nothing_to_order_against():
    """Degradation costs a notice about a real risk, never a notice about none."""
    agent = _WiringAgent(registry=_RenamedMapRegistry())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ContextGraph().init_agent(agent)

    assert caught == []
    assert len(agent.hooks) == 3


def test_a_memory_fold_behind_an_unmovable_delivery_is_not_claimed_as_ordered():
    """The move failed, so the fold sitting behind the delivery in the list proves nothing — and the notice fires."""
    registry = _RenamedMapRegistry()
    registry.add_middleware(InvokeModelStage.Input, _other_fold)
    agent = _WiringAgent(plugins={"strands:memory": _MemoryManager()}, registry=registry)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ContextGraph().init_agent(agent)

    assert len(caught) == 1


async def _other_fold(context):
    """A second input handler, standing in for a memory manager's fold in the same stage."""
    return context
