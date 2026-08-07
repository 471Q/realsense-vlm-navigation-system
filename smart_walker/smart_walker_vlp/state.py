from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Direction(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"
    STOP = "stop"


class SafetyLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"


@dataclass
class WorldState:
    """Minimal summary of what VLP needs to know about the scene.

    All distances are in meters, as perceived from current pose.
    If a distance is None, it is considered "unknown".
    """

    front_clearance: Optional[float] = None
    left_clearance: Optional[float] = None
    right_clearance: Optional[float] = None
    back_clearance: Optional[float] = None

    # Whether the camera view is good enough to assess the scene.
    visibility_ok: bool = True

    # Optional semantic flags you can fill from Qwen / ontology.
    person_in_front: bool = False
    crowded_front: bool = False


@dataclass
class UserIntent:
    """User's immediate navigation intent.

    Typically derived from keyboard (e.g. W/A/S/D) or a higher-level
    command mapped to a Direction.
    """

    direction: Direction


@dataclass
class Suggestion:
    """Planner output: what to do and how safe it is, in words.

    The `primary_action` is what you would map back to key events or
    velocity commands. `message` is directly showable to the user.
    """

    primary_action: Direction
    safety: SafetyLevel
    message: str
