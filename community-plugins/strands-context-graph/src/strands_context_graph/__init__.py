"""Short-term memory as a graph of Cards for Strands Agents.

Provides :class:`ContextGraph`. Short-term memory becomes a graph of Cards rather than a linear message list: one Card
per closed turn, derived by deterministic scan, no model call. A Card enters a call at one of three Resolutions — Title
(always present), Description (rule-derived, capped by ``description_tokens``) or Full Content. Each turn a Note is
computed per Card from the similarity between the turn's question and the Card's Description, then propagated one jump
along the Links; the Note picks the Resolution. Resolution descends only for budget, never as a verdict, so a Card that
will not fit whole keeps its Description and Title.

The plugin reads the per-call copy the event loop hands to ``InvokeModelStage`` and never mutates ``agent.messages``.
Pair it with ``NullConversationManager``: this is a precondition, not a suggestion. Any other conversation manager edits
the live message list before the call is assembled, so it can physically drop what the graph only meant to fold, and the
folded Description block then describes messages that no longer exist. The plugin emits one warning at wiring time when
it sees a manager that is not ``NullConversationManager``, and still registers everything; that warning is the whole
protection you get.

The matcher is the :class:`SimilarityMatcher` protocol. The default :class:`EmbeddingSimilarityMatcher` is the graph's
only remote call: one embedding round per turn, cached by ``(purpose, text)``. A custom implementation needs only a
callable ``score`` member returning exactly ``len(descriptions)`` values in ``[0.0, 1.0]``, which is why the protocol is
exported alongside the construct. ``expand_threshold``, ``collapse_floor`` and ``link_threshold`` are calibrated against
the default matcher's score distribution and are not portable; another matcher needs its own calibration.

``expand_threshold=0.0, collapse_floor=0.0`` projects every Card at Full Content, producing a call identical field for
field to the one produced without the plugin, which tells a graph problem apart from a pre-existing one.

Example Usage:
    ```python
    from strands import Agent
    from strands.agent.conversation_manager import NullConversationManager
    from strands_context_graph import ContextGraph

    agent = Agent(
        conversation_manager=NullConversationManager(),
        plugins=[ContextGraph()],
    )
    ```
"""

from .matcher import EmbeddingSimilarityMatcher, SimilarityMatcher
from .plugin import ContextGraph

__all__ = [
    "ContextGraph",
    "EmbeddingSimilarityMatcher",
    "SimilarityMatcher",
]
