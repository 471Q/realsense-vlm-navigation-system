"""Tiny offline demo for the VLP decision logic.

Run this with:

    python -m smart_walker_vlp.demo

It does NOT connect to RealSense; it just shows how different
clearances and intents lead to different suggestions.
"""

from .state import Direction, UserIntent, WorldState
from .planner import decide_suggestion


def _print_case(description: str, state: WorldState, intent: UserIntent) -> None:
    print("\n===", description, "===")
    suggestion = decide_suggestion(state, intent)
    print("Intent:", intent.direction.value)
    print("Suggestion action:", suggestion.primary_action.value)
    print("Safety:", suggestion.safety.value)
    print("Message:", suggestion.message)


def main() -> None:
    # Case 1: clear forward path
    state1 = WorldState(front_clearance=2.0,
                        left_clearance=0.5, right_clearance=0.5)
    _print_case("Clear forward path", state1, UserIntent(Direction.FORWARD))

    # Case 2: blocked forward, clear right
    state2 = WorldState(front_clearance=0.2,
                        left_clearance=0.6, right_clearance=1.5)
    _print_case("Blocked ahead, clear right", state2,
                UserIntent(Direction.FORWARD))

    # Case 3: blocked everywhere except back
    state3 = WorldState(front_clearance=0.2, left_clearance=0.3,
                        right_clearance=0.3, back_clearance=1.2)
    _print_case("Blocked ahead and sides, free back",
                state3, UserIntent(Direction.FORWARD))

    # Case 4: bad visibility
    state4 = WorldState(front_clearance=None, left_clearance=None,
                        right_clearance=None, visibility_ok=False)
    _print_case("Unknown scene, bad visibility",
                state4, UserIntent(Direction.FORWARD))


if __name__ == "__main__":
    main()
