"""Practice D — context graph: Card model, scan, scoring, matcher, projection (core).

The one entry point is :func:`~context_core.graph.projection.project`: it takes a neutral message list plus the graph
state carried over from the last call and returns the projected message list with the updated state. No framework
import anywhere in this subpackage -- see :mod:`context_core.message` for the shape the projection reads.
"""

from .cards import closed_turn_ranges, derive_and_register, is_turn_boundary, rebuild_into, turn_ranges
from .compaction import guidance, render_final_block
from .matcher import EmbeddingSimilarityMatcher, SimilarityMatcher
from .projection import Thresholds, current_turn_ids, project
from .removal import apply_removal, removal_ids, remove_messages
from .scoring import compute_notes, distribute, full_pass_choice, warm_up_choice
from .state import Card, CardChoice, CardKind, GraphState, Link, LinkKind, Resolution, ToolPair, TurnChoice
from .store import InMemoryReferenceStore, ReferenceStore, record_references

__all__ = [
    "Card",
    "CardChoice",
    "CardKind",
    "EmbeddingSimilarityMatcher",
    "GraphState",
    "InMemoryReferenceStore",
    "Link",
    "LinkKind",
    "ReferenceStore",
    "Resolution",
    "SimilarityMatcher",
    "Thresholds",
    "ToolPair",
    "TurnChoice",
    "apply_removal",
    "closed_turn_ranges",
    "compute_notes",
    "current_turn_ids",
    "derive_and_register",
    "distribute",
    "full_pass_choice",
    "guidance",
    "is_turn_boundary",
    "project",
    "rebuild_into",
    "record_references",
    "removal_ids",
    "remove_messages",
    "render_final_block",
    "turn_ranges",
    "warm_up_choice",
]
