"""Minimal vision-language planning (VLP) layer for the smart walker.

This package adds a rule-based planning and explanation layer on top of
existing depth-based safe-lane evaluation and VLM semantics.

It does NOT change or depend on the internal details of the existing
`smart_walker` package; instead it expects simple numeric/boolean
summaries of the scene and user intent.
"""

from .state import WorldState, UserIntent, Suggestion, SafetyLevel
from .planner import decide_suggestion
