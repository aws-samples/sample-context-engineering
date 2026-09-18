"""Compatibility smoke test of the one private seam this package couples to.

Validates: Requirements 12.4, 12.5.

The plugin rewrites ``tool_specs`` through SDK internals: ``InvokeModelStage`` and
``InvokeModelContext`` from ``strands._middleware.stages``, registered with
``agent._middleware_registry.add_middleware``. Underscore-prefixed names carry no stability
guarantee, so a minor SDK release can move them. This file is where that shows up: three assertions
over the shape the plugin relies on, each one reached through the package's own ``_compat`` module so
the test fails at the same import the plugin would fail at.

Every check runs offline against a stub model and a hand-built context. There is no model call, no
index build and no network, so the run is deterministic and the failure, when the seam moves, is
unambiguous: CI breaks rather than a user's runtime silently losing the projection.
"""

import dataclasses
from typing import Any

from strands import Agent
from strands._middleware.registry import MiddlewareRegistry
from strands._middleware.types import MiddlewareInputPhase
from strands.models.model import Model

from strands_progressive_tool_disclosure._compat import InvokeModelContext, InvokeModelStage

TOOL_SPECS = [{"name": "run_query", "description": "runs a query", "inputSchema": {}}]


class _StubModel(Model):
    """A model that exists to be constructed. Nothing here calls it, and a call would be a bug."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: this test asserts a shape, never a call."""
        raise AssertionError("the language model was called")
        yield

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: this test asserts a shape, never a call."""
        raise AssertionError("the language model was called")
        yield


def _context_arguments(agent: Agent) -> dict[str, Any]:
    """Constructor arguments for an ``InvokeModelContext``, restricted to the fields this SDK declares.

    The context gained fields across the supported SDK range, and the fields around ``tool_specs`` are
    not what this file is about: filtering by the declared field names keeps a new neighbouring field
    from failing the test for the wrong reason, while a *missing* ``tool_specs`` still fails loudly.
    """
    candidates: dict[str, Any] = {
        "agent": agent,
        "messages": [{"role": "user", "content": [{"text": "what did the migration cost"}]}],
        "system_prompt": "You are a helpful assistant.",
        "tool_specs": list(TOOL_SPECS),
        "tool_choice": None,
        "invocation_state": {},
        "model": agent.model,
    }
    declared = {field.name for field in dataclasses.fields(InvokeModelContext)}
    return {name: value for name, value in candidates.items() if name in declared}


def _input_handler_count(registry: MiddlewareRegistry) -> int:
    """How many input-phase handlers the registry holds for the invoke-model stage."""
    stage = InvokeModelStage.Input._stage
    return sum(1 for tagged in registry._handlers.get(stage, []) if tagged.phase == "input")


def test_invoke_model_stage_input_is_importable_and_is_an_input_phase_token():
    """Requirement 12.4: the stage token the plugin registers on exists and is still an input phase.

    Importability alone is not the contract — ``add_middleware`` dispatches on the token's type, so a
    token that survived the rename but stopped being an ``MiddlewareInputPhase`` would register the
    handler in the wrong phase and quietly never rewrite anything.
    """
    assert InvokeModelStage.Input is not None
    assert isinstance(InvokeModelStage.Input, MiddlewareInputPhase)


def test_invoke_model_context_has_a_replaceable_tool_specs_field():
    """Requirement 12.4: ``tool_specs`` is a field of the context, and ``replace`` carries a new one.

    The plugin never mutates the received context; it returns ``dataclasses.replace(context,
    tool_specs=...)``. So the field has to exist *and* be settable through the constructor, which is
    what ``replace`` goes through.
    """
    field_names = {field.name for field in dataclasses.fields(InvokeModelContext)}
    assert "tool_specs" in field_names

    agent = Agent(model=_StubModel(), system_prompt="You are a helpful assistant.")
    context = InvokeModelContext(**_context_arguments(agent))
    assert context.tool_specs == TOOL_SPECS

    projected = dataclasses.replace(context, tool_specs=[])
    assert projected.tool_specs == []
    # The received context is left as it was: the plugin's degradation path returns it unchanged.
    assert context.tool_specs == TOOL_SPECS


def test_add_middleware_exists_on_the_agents_middleware_registry_and_accepts_the_input_token():
    """Requirement 12.4: the agent carries a middleware registry, and it takes the plugin's handler.

    Asserted on a real ``Agent`` rather than on the registry class, because the attribute name
    ``_middleware_registry`` is as much part of the seam as the method is: the plugin's
    ``init_agent`` reaches for both.
    """
    agent = Agent(model=_StubModel(), system_prompt="You are a helpful assistant.")

    registry = agent._middleware_registry
    assert isinstance(registry, MiddlewareRegistry)
    assert callable(registry.add_middleware)

    async def handler(context: InvokeModelContext) -> InvokeModelContext:
        """Stand-in for the plugin's projection handler. Registered, never invoked."""
        return context

    # Registration is the assertion: a moved or re-typed token raises or lands the handler elsewhere.
    # The registry wraps the handler before storing it, so the count of input-phase handlers on this
    # stage is what can be checked, not the identity of the callable.
    before = _input_handler_count(registry)
    registry.add_middleware(InvokeModelStage.Input, handler)
    assert _input_handler_count(registry) == before + 1
