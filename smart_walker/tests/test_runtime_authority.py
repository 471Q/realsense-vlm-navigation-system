"""The deterministic action policy and its thresholds.

This is the layer that decides what the walker tells someone to do, so it is the layer where an
untested boundary matters most. On 22 August 2026 seven of the eight threshold comparisons here
could be loosened by one operator with the whole suite still passing, including the distance at
which an object stops the walker and the tolerance that chooses between two similar sectors.

Every threshold is pinned from both sides: the value just inside the boundary and the value on it.
A test that only exercises the middle of a range proves the range exists and says nothing about
where it ends, which is precisely what the previous suite did.

The decision is read through `motion_decision` rather than through any intermediate, because that
is the field the release object carries and the user acts on.
"""

import unittest

import numpy as np

from tests import support
from tests.support import hdsg


ALL_CLEAR = ("CLEAR", "CLEAR", "CLEAR")


class ObjectThresholdTests(unittest.TestCase):
    """`determine_authority` defaults: stop below 0.70 m, caution below 1.50 m, hazard at 2.00 m."""

    def decision(self, distance, **kwargs):
        authority = hdsg.determine_authority(
            "FORWARD", "SAFE", support.sectors(3.13, 3.13, 3.13, ALL_CLEAR),
            objects=support.detected_object(bearing="CENTRE", distance_m=distance, **kwargs),
            sector_choice_tolerance_m=0.10,
        )
        return authority["motion_decision"]

    def test_an_object_below_the_stop_distance_stops_the_walker(self):
        self.assertEqual(self.decision(0.69), "STOP")

    def test_an_object_exactly_at_the_stop_distance_does_not_stop_it(self):
        """0.70 is the boundary and the comparison is strict. Loosening it to `<=` moves the stop
        distance for every run, and nothing in the previous suite would have noticed."""
        self.assertEqual(self.decision(0.70), "SLOW")

    def test_an_object_below_the_caution_distance_slows_the_walker(self):
        self.assertEqual(self.decision(1.49), "SLOW")

    def test_an_object_exactly_at_the_caution_distance_does_not_slow_it(self):
        self.assertEqual(self.decision(1.50), "PROCEED")

    def test_a_hazard_stops_at_a_distance_an_ordinary_object_would_not(self):
        """The hazard rule is inclusive at 2.00 m, which is why it is written differently from the
        other two, and the difference has to be pinned or the two rules drift together."""
        self.assertEqual(self.decision(2.00, ontology_class="hazard"), "STOP")
        self.assertEqual(self.decision(2.00), "PROCEED")

    def test_a_hazard_beyond_the_hazard_distance_does_not_stop(self):
        self.assertEqual(self.decision(2.01, ontology_class="hazard"), "PROCEED")

    def test_the_nearest_object_in_a_sector_binds_the_action(self):
        """Two objects in one sector: the instruction must rest on the closer of them."""
        near = support.detected_object(track_id=1, bearing="CENTRE", distance_m=0.60)
        far = support.detected_object(track_id=2, bearing="CENTRE", distance_m=1.40)
        for objects in (near + far, far + near):
            with self.subTest(order=[item["fact_id"] for item in objects]):
                authority = hdsg.determine_authority(
                    "FORWARD", "SAFE", support.sectors(3.13, 3.13, 3.13, ALL_CLEAR),
                    objects=objects, sector_choice_tolerance_m=0.10,
                )
                # The sectors join the binding when both contribute, so the nearest object is
                # identified by which fact is primary rather than by the list being a singleton.
                self.assertEqual(authority["action_binding"]["primary_fact_id"], "object:1")

    def test_an_object_with_no_distance_is_ignored(self):
        """An unmeasured object cannot bind a decision, because there is nothing to bind it to."""
        objects = support.detected_object(bearing="CENTRE", distance_m=None)
        authority = hdsg.determine_authority(
            "FORWARD", "SAFE", support.sectors(3.13, 3.13, 3.13, ALL_CLEAR),
            objects=objects, sector_choice_tolerance_m=0.10,
        )
        self.assertEqual(authority["motion_decision"], "PROCEED")


class SectorChoiceTests(unittest.TestCase):
    """With the centre blocked and both sides open, which side is chosen and when it asks instead.

    The tolerance is passed explicitly as 0.50 rather than left at its 0.10 default. A difference of
    exactly 0.10 cannot be constructed from two floats, so at the default the boundary is
    unreachable and `>` against `>=` is untestable. This is worth knowing separately: at exactly the
    tolerance the outcome depends on which two clearances produced the difference.
    """

    def authority(self, left, right, previous=None, tolerance=0.50):
        return hdsg.determine_authority(
            "FORWARD", "SAFE",
            support.sectors(left, 0.40, right, ("CLEAR", "BLOCKED", "CLEAR")),
            previous_selected_sector=previous, sector_choice_tolerance_m=tolerance,
        )

    def test_a_difference_beyond_the_tolerance_chooses_the_wider_side(self):
        self.assertEqual(self.authority(2.51, 2.00)["selected_sector"], "LEFT")

    def test_a_difference_exactly_at_the_tolerance_asks_instead(self):
        """The comparison is strict, so equal-to-tolerance counts as similar and the walker asks.
        Loosening it to `>=` would let a difference the policy calls insignificant pick a
        direction."""
        outcome = self.authority(2.50, 2.00)
        self.assertEqual(outcome["selected_sector"], "NONE")
        self.assertEqual(outcome["interaction_state"], "AWAITING_SECTOR_CHOICE")

    def test_a_previous_choice_is_kept_when_the_two_are_similar(self):
        """Re-deciding on a difference the policy calls insignificant would swing the instruction
        between left and right while the user is acting on it."""
        self.assertEqual(self.authority(2.50, 2.00, previous="RIGHT")["selected_sector"], "RIGHT")

    def test_a_previous_choice_is_overridden_when_the_difference_is_real(self):
        self.assertEqual(self.authority(3.00, 2.00, previous="RIGHT")["selected_sector"], "LEFT")

    def test_both_sides_are_offered_when_it_asks(self):
        self.assertEqual(set(self.authority(2.50, 2.00)["selection_options"]), {"LEFT", "RIGHT"})


class AdvisoryTests(unittest.TestCase):
    def test_the_scene_advisory_is_the_more_severe_of_the_two(self):
        for object_advisory, expected in (("SAFE", "CAUTION"), ("CAUTION", "CAUTION"),
                                          ("STOP", "STOP")):
            with self.subTest(object_advisory=object_advisory):
                authority = hdsg.determine_authority(
                    "FORWARD", object_advisory,
                    support.sectors(3.13, 1.74, 1.16, ("CLEAR", "CONSTRAINED", "CONSTRAINED")),
                    sector_choice_tolerance_m=0.10,
                )
                self.assertEqual(authority["scene_advisory"], expected)

    def test_an_unrecognised_advisory_is_coerced_to_stop(self):
        """An unrecognised value must fail towards the most severe reading rather than fall through
        as though it were safe."""
        authority = hdsg.determine_authority(
            "FORWARD", "NONSENSE", support.sectors(3.13, 3.13, 3.13, ALL_CLEAR),
            sector_choice_tolerance_m=0.10,
        )
        self.assertEqual(authority["object_advisory"], "STOP")
        self.assertEqual(authority["scene_advisory"], "STOP")

    def test_three_clear_sectors_are_safe_and_fewer_are_not(self):
        for statuses, expected in (
            (ALL_CLEAR, "SAFE"),
            (("CLEAR", "CONSTRAINED", "CLEAR"), "CAUTION"),
            (("BLOCKED", "BLOCKED", "BLOCKED"), "STOP"),
        ):
            with self.subTest(statuses=statuses):
                authority = hdsg.determine_authority(
                    "FORWARD", "SAFE", support.sectors(2.0, 2.0, 2.0, statuses),
                    sector_choice_tolerance_m=0.10,
                )
                self.assertEqual(authority["sector_advisory"], expected)

    def test_an_object_binds_the_scene_only_when_it_is_the_more_severe(self):
        """Equal severity leaves the sector binding in place, so the explanation names the
        measurement that decided rather than whichever was considered last."""
        authority = hdsg.determine_authority(
            "FORWARD", "SAFE",
            support.sectors(3.13, 0.40, 1.16, ("CLEAR", "BLOCKED", "CONSTRAINED")),
            objects=support.detected_object(bearing="LEFT", distance_m=1.40),
            sector_choice_tolerance_m=0.10,
        )
        self.assertEqual(authority["scene_binding"]["source"], "SECTOR")


class IntentTests(unittest.TestCase):
    def test_backward_intent_reports_the_rear_as_unobserved(self):
        """The camera faces forward, so the walker cannot speak to what is behind it."""
        authority = hdsg.determine_authority(
            "BACKWARD", "SAFE", support.sectors(3.13, 3.13, 3.13, ALL_CLEAR),
            sector_choice_tolerance_m=0.10,
        )
        self.assertEqual(authority["interaction_state"], "REORIENTATION_REQUIRED")
        self.assertIn("condition:rear_unobserved",
                      authority["action_binding"]["accepted_fact_ids"])

    def test_no_clear_sector_stops_and_says_why(self):
        authority = hdsg.determine_authority(
            "FORWARD", "SAFE", support.sectors(0.40, 0.40, 0.40, ("BLOCKED",) * 3),
            sector_choice_tolerance_m=0.10,
        )
        self.assertEqual(authority["motion_decision"], "STOP")
        self.assertIn("condition:no_clear_sector",
                      authority["action_binding"]["accepted_fact_ids"])

    def test_every_intent_produces_an_admissible_decision(self):
        for intent in ("FORWARD", "LEFT", "RIGHT", "BACKWARD", "NONE"):
            with self.subTest(intent=intent):
                authority = hdsg.determine_authority(
                    intent, "SAFE", support.sectors(3.13, 3.13, 3.13, ALL_CLEAR),
                    sector_choice_tolerance_m=0.10,
                )
                self.assertIn(authority["motion_decision"],
                              {"PROCEED", "SLOW", "REDIRECT", "STOP"})
                self.assertIn((authority["motion_decision"], authority["selected_sector"]),
                              hdsg.ACTION_TEMPLATES,
                              "every decision must have a sentence to render it")


class MotionTrackerTests(unittest.TestCase):
    """Movement is confirmed only after repeated observations past a threshold.

    Confirmation exists so that sensor jitter does not report a moving obstacle. Without a test
    either side of it, the threshold and the confirmation count are both free to drift.
    """

    def frames(self, distances, movement=0.20, stationary=0.05, confirmations=3,
               ontology_class="agent"):
        """Drives the tracker over a static camera watching one object change distance.

        The frame carries texture and does not change between observations. Both matter. Optical
        flow needs features, so a blank frame leaves camera motion uncompensated and every motion
        score stays at zero, which silently makes any assertion about movement unreachable. A frame
        that does not change means the global flow is nil, so a change in the object's depth is the
        object moving rather than the walker approaching it.
        """
        tracker = hdsg.MotionTracker(
            movement_threshold_m=movement, stationary_threshold_m=stationary,
            confirmation_observations=confirmations,
        )
        frame = np.random.default_rng(7).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        state = []
        for index, distance in enumerate(distances):
            objects = support.detected_object(track_id=7, label="person", bearing="CENTRE",
                                              distance_m=distance,
                                              ontology_class=ontology_class)
            state = tracker.update(frame, list(objects), float(index) * 100.0)
        return state[0]

    def test_the_camera_is_compensated_before_any_motion_is_scored(self):
        """Without compensation every score is zero, so a test asserting movement would be
        asserting nothing. This pins the precondition the other tests rest on."""
        self.assertIn("MOTION_CAMERA_COMPENSATED",
                      self.frames([3.00, 2.60, 2.20])["motion_reason_codes"])

    def test_an_approaching_object_is_confirmed_as_moving(self):
        self.assertEqual(self.frames([3.00, 2.60, 2.20, 1.80, 1.40])["motion_state"], "MOVING")

    def test_an_object_holding_still_is_confirmed_as_stationary(self):
        self.assertEqual(self.frames([2.00, 2.01, 2.00, 2.01, 2.00])["motion_state"], "STATIONARY")

    def test_one_step_past_the_threshold_is_not_yet_confirmed(self):
        """A single jump is noise, and reporting it as movement is what confirmation prevents."""
        self.assertNotEqual(self.frames([3.00, 2.00])["motion_state"], "MOVING")

    def test_an_ineligible_class_is_never_reported_as_moving(self):
        """Only an agent or a rolling obstacle may be classified as moving. Furniture that appears
        to approach is the walker approaching it, and saying otherwise would put a moving hazard on
        the display that is not there."""
        state = self.frames([3.00, 2.60, 2.20, 1.80, 1.40], ontology_class="furniture")
        self.assertNotEqual(state["motion_state"], "MOVING")
        self.assertIn("MOTION_CLASS_NOT_ELIGIBLE", state["motion_reason_codes"])


if __name__ == "__main__":
    unittest.main()
