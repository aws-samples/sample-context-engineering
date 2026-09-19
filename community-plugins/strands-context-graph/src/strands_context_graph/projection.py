"""The delivery: the one ``InvokeModelStage.Input`` handler where the messages sent to the provider change.

Two operations, one handler, one ``try``. The removal drops the collapsed identities from the call's own message list
(:mod:`.removal`), and the compaction folds the Descriptions of what actually left into the call's last user message
(:mod:`.compaction`). They are one handler because Requirement 16.2 does not admit the state in between: a removal
applied with no block folded is a call that lost content and says nothing about the loss. Any failure therefore returns
the **received** context by object identity, with one warning, and remembers nothing, so the next model call attempts
the delivery again (Requirement 16.3).

Neither ``agent.messages`` nor the received context is touched. The new context comes out of ``dataclasses.replace``,
which replaces ``messages`` and ``dynamic_trailing_blocks`` and carries every other field over unchanged (Requirements
1.4, 9.1, 9.2, 9.3). ``agent.messages`` is never read from here and never written to, so every message of the history
stays readable whatever Resolution the choice assigned it (Requirement 9.10).

The fold is not reimplemented: :func:`~strands.injection._message_injection._create_injection_middleware` is the SDK's
own injection primitive, which appends the text to the last ``user`` message with no ``role`` of its own and increments
``dynamic_trailing_blocks`` by the blocks it appended. ``trigger="everyTurn"`` is a requirement and not a preference
(Requirement 9.4): the default ``"userTurn"`` fires only on the turn's first call, so the autonomous tool loop's calls
would go out with the removal applied and no final block — exactly the split state the single ``try`` exists to prevent.

A full pass returns the received context itself: no new context, no new list, no final block, so the call is identical
field by field to the one produced without the plugin (Requirements 9.8, 1.11). A request that comes out empty after the
guards is the same case, read one step later.

**Private-API dependencies** (Requirement 17.4). Three, of two different shapes.

``InvokeModelStage`` with its ``.Input`` phase, ``_create_injection_middleware``, and ``dynamic_trailing_blocks`` (which
the primitive maintains) are *hard*: they are imported statically and a rename fails at import. There is no fallback to
write, because they are not a convenience over something public — the stage is the only seam where the messages sent to
the provider can change without mutating ``agent.messages``, and the primitive is the only thing that appends a trailing
block without inventing a message. A public per-call input-transform stage and a public "append a trailing block" API
would remove both couplings; the upper bound on the ``strands-agents`` range is what covers them until then.

``MiddlewareRegistry._handlers`` is *soft*, and :meth:`Projection.register` is where the difference shows. The
registration itself uses the public ``add_middleware``; only the move to index zero reads the private map, so that move
is the one part allowed to fail. On a build that renamed or reshaped it the delivery stays registered in wiring order
and :meth:`Projection.register` reports ``False``, which turns into the wiring-time ordering notice of Requirement 9.6 —
the same failure mode as a memory fold registered outside this stage, which no amount of hardening reaches. Declared
middleware ordering in the SDK would remove this one. See the README's private-API table for the whole set.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from strands._middleware.stages import InvokeModelStage
from strands.injection._message_injection import _create_injection_middleware  # verbatim reuse

from .cards import turn_ranges
from .compaction import render_final_block
from .removal import apply_removal

if TYPE_CHECKING:
    from strands._middleware.stages import InvokeModelContext
    from strands.agent.agent import Agent
    from strands.injection.types import InjectionContext
    from strands.types.content import Messages

    from .state import _GraphStates

__all__ = ["Projection", "current_turn_ids"]

logger = logging.getLogger(__name__)


def current_turn_ids(messages: Messages) -> frozenset[str]:
    """Durable identities of the turn in progress: the trailing range of ``messages``.

    :func:`~.cards.turn_ranges`' last range is the open turn by definition, so this is the slice
    :func:`~.cards.closed_turn_ranges` excludes, read from the other side. The removal subtracts it, which is what keeps
    the question the model is answering out of every Resolution decision.

    Args:
        messages: The call's message list. Read only.

    Returns:
        The identities of the open turn. Empty when no message opens a turn.
    """
    ranges = turn_ranges(messages)
    if not ranges:
        return frozenset()
    start, stop = ranges[-1]
    return frozenset(identity for message in messages[start:stop] if (identity := message.get("tracking_id")))


@dataclass
class _InFlight:
    """What the fold's render callback needs from the delivery that called it.

    The fold handler is built once, at construction, so its render callback cannot receive the call's request as an
    argument. This record carries it across through a ``ContextVar``: async-safe and per task, so two agents delivering
    concurrently never read each other's request.

    ``error`` is the other half of atomic degradation. The injection primitive fails open, returning the context it was
    handed, which by then already carries the removal — the intermediate state Requirement 16.2 forbids. So the render
    catches its own failure, returns ``None``, and the delivery re-raises it into the single ``try`` that returns the
    *received* context by identity, with one warning.

    Attributes:
        requested: The identities the removal asked to drop on this call.
        error: The exception the compaction raised, or ``None``.
    """

    requested: frozenset[str]
    error: BaseException | None = None


_DELIVERY: contextvars.ContextVar[_InFlight | None] = contextvars.ContextVar(
    "context_graph_delivery",
    default=None,
)
"""The delivery in flight, read by the render and set by :meth:`Projection.deliver`. Set and reset around one ``await``,
so no failure state survives the call (Requirement 16.3)."""


class Projection:
    """The delivery handler, and its registration on an agent's ``InvokeModelStage``.

    Built once per plugin instance and shared by every agent the instance is wired to: it holds no per-agent state of
    its own, reading the graph out of the weakly keyed map it is handed, so two agents never see each other's Cards.

    Args:
        states: The plugin's per-agent state map. Read only, never written to from here.
        description_tokens: Token ceiling of one Card's entry in the final block.
    """

    def __init__(self, states: _GraphStates, *, description_tokens: int) -> None:
        """Build the fold once, so the primitive's trigger is fixed for the lifetime of the instance."""
        self._states = states
        self._description_tokens = description_tokens
        # Called from inside ``deliver`` rather than registered on the stage: the removal must have happened before the
        # render runs, and one handler is what makes the pair atomic.
        self._fold = _create_injection_middleware(self._render, trigger="everyTurn")

    def register(self, agent: Agent) -> bool:
        """Add :meth:`deliver` as the *first* input handler of ``InvokeModelStage``.

        Input handlers run in registration order, which is the order of the ``plugins=[...]`` list. Index zero resolves
        the required ordering whatever that list says: the graph removes before any transient memory injection of the
        same stage, whose text must not be folded into a message the removal then drops (Requirement 9.5). What index
        zero cannot resolve is a fold registered outside this stage, which is the wiring-time caveat of Requirement 9.6.

        The move to index zero reaches into ``registry._handlers``, the one private member of the registry this plugin
        touches, and it is the *only* part of the registration allowed to fail: a build that renamed the map, or stores
        the handlers as something other than a list, leaves the delivery registered at the back of the stage instead of
        crashing the wiring. Ordering then becomes exactly the risk the caller already warns about (Requirement 9.6),
        which :func:`~.plugin._delivery_precedes_memory_fold` reads off the same map and reports independently.

        Args:
            agent: The agent whose middleware registry to register on.

        Returns:
            Whether the delivery ended up at the front of the stage. The caller turns a ``False`` into the ordering
            notice; the registration itself succeeded either way.
        """
        registry = agent._middleware_registry
        registry.add_middleware(InvokeModelStage.Input, self.deliver)
        try:
            handlers = registry._handlers[InvokeModelStage]
            handlers.insert(0, handlers.pop())
        except Exception:
            logger.warning(
                "could not move the delivery to the front of InvokeModelStage | it stays registered in wiring order, "
                "so the ordering against a memory manager is no longer guaranteed",
                exc_info=True,
            )
            return False
        return True

    async def deliver(self, context: InvokeModelContext) -> InvokeModelContext:
        """Apply the removal and the compaction, as one step that either happens or does not.

        Args:
            context: The per-call invocation context. Read only, never modified in place.

        Returns:
            A new context carrying the removal and the final block, or the received context itself — by object
            identity on a full pass, on an empty request, and on any failure.
        """
        try:
            state = self._states.get(context.agent)
            # A state that does not exist yet is a fresh state, and a fresh state is a full pass.
            if state is None or state.choice.full_pass:
                return context

            removed, requested = apply_removal(
                context.messages,
                state,
                state.choice,
                current_turn_ids(context.messages),
            )
            if not requested:
                # Nothing to drop, so nothing to describe: the same list object, the same context object.
                return context

            in_flight = _InFlight(requested)
            token = _DELIVERY.set(in_flight)
            try:
                # The primitive assembles its ``InjectionContext`` over the messages of the context it receives, so
                # handing it the removed list lets the compaction subtract what actually left from what was asked for.
                # No new plumbing.
                folded = await self._fold(replace(context, messages=removed))
            finally:
                _DELIVERY.reset(token)

            if in_flight.error is not None:
                # Raised inside the compaction, swallowed by the primitive's fail-open, re-raised here so the
                # degradation covers the removal too.
                raise in_flight.error

            return folded
        except Exception:
            logger.warning("delivery failed | passing the received context through unchanged", exc_info=True)
            return context

    def _render(self, injection_context: InjectionContext) -> str | None:
        """Render the final block for the delivery in flight, or ``None`` when there is nothing to fold.

        ``injection_context.messages`` is the removed list, the primitive building its context after the substitution.
        Failures are recorded rather than raised: on a raise the primitive would fail open and hand the removed context
        down the chain, the split state Requirement 16.2 forbids.

        Args:
            injection_context: The context the fold built, over the already removed list. Read only.

        Returns:
            The text to fold, or ``None``.
        """
        in_flight = _DELIVERY.get()
        if in_flight is None:
            return None

        state = self._states.get(injection_context.agent)
        if state is None:
            return None

        try:
            return render_final_block(
                injection_context,
                state,
                in_flight.requested,
                description_tokens=self._description_tokens,
            )
        except Exception as error:
            # Handed back to ``deliver``, which degrades atomically. Raising here would leave the removal in place.
            in_flight.error = error
            return None
