from __future__ import annotations

from typing import Tuple

from .state import Direction, SafetyLevel, Suggestion, UserIntent, WorldState


# Default thresholds in meters; you can tune these or make them configurable.
HARD_STOP_DISTANCE = 0.4
CAUTION_DISTANCE = 0.8


def _distance_for_direction(state: WorldState, direction: Direction) -> float | None:
    if direction == Direction.FORWARD:
        return state.front_clearance
    if direction == Direction.BACKWARD:
        return state.back_clearance
    if direction == Direction.LEFT:
        return state.left_clearance
    if direction == Direction.RIGHT:
        return state.right_clearance
    # For pure rotation we do not use a single distance measure.
    return None


def _classify_safety(distance: float | None) -> SafetyLevel:
    if distance is None:
        return SafetyLevel.DANGER
    if distance < HARD_STOP_DISTANCE:
        return SafetyLevel.DANGER
    if distance < CAUTION_DISTANCE:
        return SafetyLevel.CAUTION
    return SafetyLevel.SAFE


def _best_lateral_direction(state: WorldState) -> Tuple[Direction | None, SafetyLevel]:
    """Pick left vs right based on clearance and safety.

    Returns (direction, safety). If both sides are bad/unknown, direction is None.
    """

    left_d = state.left_clearance
    right_d = state.right_clearance

    left_s = _classify_safety(
        left_d) if left_d is not None else SafetyLevel.DANGER
    right_s = _classify_safety(
        right_d) if right_d is not None else SafetyLevel.DANGER

    # Prefer any side that is SAFE over CAUTION/DANGER.
    if left_s == SafetyLevel.SAFE and right_s != SafetyLevel.SAFE:
        return Direction.LEFT, left_s
    if right_s == SafetyLevel.SAFE and left_s != SafetyLevel.SAFE:
        return Direction.RIGHT, right_s

    # If both same safety level, pick the one with more clearance.
    if (left_d is not None) and (right_d is not None) and left_s == right_s:
        if left_d >= right_d:
            return Direction.LEFT, left_s
        return Direction.RIGHT, right_s

    # If one side is caution and the other danger, pick caution.
    if left_s == SafetyLevel.CAUTION and right_s == SafetyLevel.DANGER:
        return Direction.LEFT, left_s
    if right_s == SafetyLevel.CAUTION and left_s == SafetyLevel.DANGER:
        return Direction.RIGHT, right_s

    return None, SafetyLevel.DANGER


def decide_suggestion(state: WorldState, intent: UserIntent) -> Suggestion:
    """Core rule-based VLP decision.

    Maps (world state, user intent) to a Suggestion: approve, warn, or
    override the requested move, and provide a natural-language message.
    """

    # If we cannot see, ask for camera motion first.
    if not state.visibility_ok:
        return Suggestion(
            primary_action=Direction.ROTATE_LEFT,
            safety=SafetyLevel.DANGER,
            message=(
                "I cannot clearly see the environment. Please slowly rotate "
                "the camera left and right so I can reassess the scene."
            ),
        )

    # Check safety in direction of intent.
    intended_distance = _distance_for_direction(state, intent.direction)
    intended_safety = _classify_safety(intended_distance)

    # If intent is safe or caution, we try to respect it.
    if intended_safety == SafetyLevel.SAFE:
        if intent.direction == Direction.FORWARD and intended_distance is not None:
            msg = f"Path ahead is clear for about {intended_distance:.1f} meters. You can proceed forward."
        elif intent.direction == Direction.BACKWARD and intended_distance is not None:
            msg = f"Backwards is clear for about {intended_distance:.1f} meters. You can go back."
        else:
            msg = "The path in your intended direction looks clear. You can proceed."

        # Add simple semantic warning if a person is ahead but far.
        if state.person_in_front and intent.direction == Direction.FORWARD:
            msg += " There is a person ahead; keep a comfortable distance."

        return Suggestion(primary_action=intent.direction, safety=SafetyLevel.SAFE, message=msg)

    if intended_safety == SafetyLevel.CAUTION:
        if intended_distance is not None:
            msg = (
                f"You can move {intent.direction.value}, but there is an obstacle about "
                f"{intended_distance:.1f} meters away. Proceed slowly and be ready to stop."
            )
        else:
            msg = (
                f"You can move {intent.direction.value}, but the exact distance to obstacles is uncertain. "
                "Proceed with caution and be ready to stop."
            )

        if state.crowded_front and intent.direction == Direction.FORWARD:
            msg += " The area ahead looks crowded; consider slowing down or waiting."

        return Suggestion(primary_action=intent.direction, safety=SafetyLevel.CAUTION, message=msg)

    # If the intended direction is dangerous, search for alternatives.
    # First try lateral moves.
    lateral_dir, lateral_safety = _best_lateral_direction(state)

    if lateral_dir is not None and lateral_safety != SafetyLevel.DANGER:
        if lateral_safety == SafetyLevel.SAFE:
            lateral_msg = (
                f"The way {intent.direction.value} is blocked. The safest option is to move "
                f"slightly {lateral_dir.value} to go around the obstacle."
            )
        else:
            lateral_msg = (
                f"The way {intent.direction.value} is blocked. You may move slightly {lateral_dir.value}, "
                "but do so carefully; space is limited."
            )

        return Suggestion(primary_action=lateral_dir, safety=lateral_safety, message=lateral_msg)

    # If both sides are bad, check if going back is safer.
    back_distance = _distance_for_direction(state, Direction.BACKWARD)
    back_safety = _classify_safety(back_distance)

    if back_safety != SafetyLevel.DANGER:
        if back_distance is not None:
            msg = (
                "Your intended path is blocked. The safest option is to go back a little; "
                f"you have about {back_distance:.1f} meters of free space behind you."
            )
        else:
            msg = (
                "Your intended path is blocked. The safest option is to go back a little, "
                "but be cautious as the exact clearance behind is uncertain."
            )

        return Suggestion(primary_action=Direction.BACKWARD, safety=back_safety, message=msg)

    # Last resort: ask the user to stop and adjust the view.
    return Suggestion(
        primary_action=Direction.STOP,
        safety=SafetyLevel.DANGER,
        message=(
            "I cannot find a safe direction to move right now. Please stop and slowly "
            "adjust the camera or your position so I can get a better view of the environment."
        ),
    )
