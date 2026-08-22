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

    def test_the_class_does_not_decide_whether_something_moved(self):
        """Furniture that approaches under a compensated camera is reported as moving.

        Motion used to require an ontology class of agent or rolling obstacle. An object outside
        those two was not left unclassified: after enough observations it was recorded as STATIONARY
        with a confidence of 1.0, so a chair being pushed towards the user was asserted with
        complete certainty not to be moving. The prior about which things move is wrong in the
        direction that costs most, because the objects that matter are the ones behaving
        unexpectedly.
        """
        state = self.frames([3.00, 2.60, 2.20, 1.80, 1.40], ontology_class="furniture")
        self.assertEqual("MOVING", state["motion_state"])

    def test_furniture_holding_still_is_still_stationary(self):
        """Removing the class gate must not make everything look like it is moving. The evidence,
        not the class, is what separates the two."""
        state = self.frames([2.00, 2.01, 2.00, 2.01, 2.00], ontology_class="furniture")
        self.assertEqual("STATIONARY", state["motion_state"])

    def test_a_stationary_verdict_reports_measured_confidence(self):
        """The forced verdict carried a confidence of 1.0 that nothing had measured. A verdict now
        reports what the evidence supports."""
        state = self.frames([2.00, 2.01, 2.00, 2.01, 2.00], ontology_class="furniture")
        self.assertEqual("STATIONARY", state["motion_state"])
        self.assertIn("MOTION_WITHIN_STATIONARY_TOLERANCE", state["motion_reason_codes"])


class DepthCoverageTests(unittest.TestCase):
    """How much of the reasoning band carried a reading, and the caution rule built on it.

    This replaced the low-light rule on 23 August 2026. That rule tested a flag written as False on
    every frame, beside a `low_light_gray_mean_lt: 40` nothing computed, and measurement showed the
    threshold pointed the wrong way as well: the backlit staircase captures average 101 grey and are
    brighter than the ordinary room at 73 to 94. Depth coverage separated the same frames cleanly,
    17.6 to 36.8 per cent against 71.7 to 72.0.
    """

    def setUp(self):
        import scripts.realsense_shared_control as sw

        self.sw = sw
        import yaml

        self.cfg = yaml.safe_load(
            (support.CONFIG / "pipeline.yaml").read_text(encoding="utf-8")
        )

    def depth(self, valid_fraction):
        """A depth frame whose reasoning band is `valid_fraction` readable.

        Only rows 55 to 95 per cent count, so the frame is built to that band rather than overall.
        """
        frame = np.zeros((100, 100), dtype=np.float32)
        band = frame[55:95, :]
        readable = int(round(valid_fraction * band.size))
        band.reshape(-1)[:readable] = 1.5
        frame[55:95, :] = band
        return frame

    def test_it_measures_the_band_the_decision_is_made_from(self):
        """A frame readable only outside rows 55 to 95 per cent has no usable coverage.

        Measuring the whole image would report depth the sector medians never saw.
        """
        frame = np.zeros((100, 100), dtype=np.float32)
        frame[0:50, :] = 2.0
        self.assertEqual(0.0, self.sw.valid_depth_fraction(frame))

    def test_a_zero_is_no_reading_rather_than_a_surface_at_zero_metres(self):
        self.assertEqual(0.0, self.sw.valid_depth_fraction(np.zeros((100, 100), dtype=np.float32)))

    def test_a_missing_frame_reports_nothing_rather_than_zero(self):
        """None and 0.0 mean different things, and only one of them should raise a caution."""
        self.assertIsNone(self.sw.valid_depth_fraction(None))
        self.assertIsNone(self.sw.valid_depth_fraction(np.zeros((0, 0), dtype=np.float32)))

    def facts(self, valid_fraction):
        return {"objects": [], "free_space": {"corridor_min_width_m": None}, "hazards": [],
                "uncertainty": {"valid_depth_fraction": valid_fraction}, "explain": {}}

    def test_coverage_below_the_threshold_raises_caution(self):
        result = self.sw.compute_baseline_risk(self.facts(0.35), self.cfg)
        self.assertEqual("caution", result["risk"])
        self.assertIn("caution:depth_coverage", result["rules_fired"])

    def test_coverage_at_the_threshold_does_not(self):
        threshold = self.cfg["risk_rules_baseline"]["caution"]["min_valid_depth_fraction"]
        result = self.sw.compute_baseline_risk(self.facts(threshold), self.cfg)
        self.assertNotIn("caution:depth_coverage", result["rules_fired"])

    def test_an_unmeasured_frame_does_not_raise_caution(self):
        """None is the absence of a measurement, and the absence of a measurement is not a reading
        of zero. A packet built before the first depth frame must not be called cautious for it."""
        result = self.sw.compute_baseline_risk(self.facts(None), self.cfg)
        self.assertNotIn("caution:depth_coverage", result["rules_fired"])

    def test_a_configuration_without_the_rule_skips_it(self):
        """Replaying an archived run must not apply a rule that run was never subject to."""
        cfg = {"risk_rules_baseline": {
            "stop": {"nearest_obstacle_m_lt": 0.7, "corridor_min_width_m_lt": 0.6,
                     "hazard_within_m_lte": 2.0},
            "caution": {"nearest_obstacle_m_lt": 1.5, "multi_near_objects_count_gte": 2},
        }}
        result = self.sw.compute_baseline_risk(self.facts(0.05), cfg)
        self.assertEqual("safe", result["risk"])

    def test_the_fraction_reaches_the_fact_packet(self):
        """Recorded on every observation whether or not it crosses the threshold, because the
        threshold is provisional and only the accumulated distribution can settle it."""
        packet = support.fact_packet()
        self.assertIn("depth_valid_fraction", packet["observation"])
        self.assertIsNone(packet["observation"]["depth_valid_fraction"])


class ThresholdConsistencyTests(unittest.TestCase):
    """Every threshold has one value, wherever it is read from.

    Each of these numbers lived in two or three places until 23 August 2026. `determine_authority`
    carried 1.50 and 2.00 as parameter defaults while `pipeline.yaml` carried the same two under
    different names; `build_fact_packet` wrote both as literals into the record that states how a
    run was configured, so that record would have kept reporting 1.50 after the configuration was
    changed; and the sector clear distance was 1.8 on the command line while every test packet
    recorded 2.0.

    The last of those is the one that matters. The thresholds record exists so that a run in the
    archive can be read back and understood. A number in it that the run did not use is worse than
    no number at all, because it will be believed.
    """

    def setUp(self):
        import yaml

        self.cfg = yaml.safe_load(
            (support.CONFIG / "pipeline.yaml").read_text(encoding="utf-8")
        )

    def test_the_configuration_agrees_with_the_runtime_constants(self):
        for path, constant in (
            (("risk_rules_baseline", "stop", "nearest_obstacle_m_lt"), hdsg.OBJECT_STOP_BELOW_M),
            (("risk_rules_baseline", "caution", "nearest_obstacle_m_lt"),
             hdsg.OBJECT_CAUTION_BELOW_M),
            (("risk_rules_baseline", "stop", "hazard_within_m_lte"),
             hdsg.HAZARD_STOP_AT_OR_BELOW_M),
            (("sector", "clear_at_or_above_m"), hdsg.SECTOR_CLEAR_AT_OR_ABOVE_M),
        ):
            with self.subTest(key=".".join(path)):
                value = self.cfg
                for key in path:
                    value = value[key]
                self.assertEqual(float(value), float(constant))

    def test_the_recorded_thresholds_are_the_thresholds_in_force(self):
        recorded = support.fact_packet()["configuration"]["thresholds_m"]
        self.assertEqual(recorded["object_stop_below"], hdsg.OBJECT_STOP_BELOW_M)
        self.assertEqual(recorded["object_caution_below"], hdsg.OBJECT_CAUTION_BELOW_M)
        self.assertEqual(recorded["hazard_stop_at_or_below"], hdsg.HAZARD_STOP_AT_OR_BELOW_M)
        self.assertEqual(recorded["sector_clear_at_or_above"], hdsg.SECTOR_CLEAR_AT_OR_ABOVE_M)

    def test_a_caller_that_changes_a_threshold_changes_what_is_recorded(self):
        """The record must follow the value in force, not a literal beside it."""
        packet = hdsg.build_fact_packet(
            run_id="run_t", event_id="evt_1", observation_id="obs_1", ticket_id=None,
            timestamp_ms=1.0, intent="FORWARD", trigger_type="USER_REQUESTED",
            request_id="MORE_DETAIL", response_mode="MORE_DETAIL", previous_signature=None,
            objects=[], sectors=support.sectors(),
            authority=hdsg.determine_authority("FORWARD", "SAFE", support.sectors()),
            mirror_view=False, detector_model="yolov8n.pt", detector_confidence=0.35,
            pipeline_config_path=support.CONFIG / "pipeline.yaml",
            ontology_path=support.CONFIG / "ontology.yaml",
            clear_threshold_m=1.2, blocked_threshold_m=0.5, sector_choice_tolerance_m=0.10,
            motion_tracker=hdsg.MotionTracker(),
            object_caution_below_m=1.1, hazard_stop_at_or_below_m=3.3,
        )
        recorded = packet["configuration"]["thresholds_m"]
        self.assertEqual(recorded["object_caution_below"], 1.1)
        self.assertEqual(recorded["hazard_stop_at_or_below"], 3.3)
        self.assertEqual(recorded["sector_clear_at_or_above"], 1.2)


if __name__ == "__main__":
    unittest.main()
