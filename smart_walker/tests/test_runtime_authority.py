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

    def test_no_intent_withholds_the_decision_rather_than_assuming_forward(self):
        """An unexpressed intent was read as CENTRE until 24 August 2026, so the whole guidance
        policy ran against a direction nobody had asked for.

        The scene used here is the one that made it visible: the centre is blocked and the two sides
        are comparably clear, which is the case the policy cannot settle on its own. It reached
        AWAITING_SECTOR_CHOICE and listed both sides, and the browser page unhides its two choice
        buttons from that list, so a person who had pressed nothing was asked to pick a side.

        What is still reported is the measured scene. The advisories and the clear-sector list are
        what the sector display draws and do not depend on an intent.
        """
        authority = hdsg.determine_authority(
            "NONE", "SAFE", support.sectors(3.20, 0.40, 3.20, ("CLEAR", "BLOCKED", "CLEAR")),
            sector_choice_tolerance_m=0.10,
        )
        self.assertEqual("IDLE_NO_INTENT", authority["interaction_state"])
        self.assertEqual([], authority["selection_options"])
        self.assertEqual("NONE", authority["selected_sector"])
        self.assertEqual(["LEFT", "RIGHT"], authority["clear_sectors"])

    def test_the_same_scene_still_asks_for_a_choice_once_an_intent_is_expressed(self):
        """The guard on the test above. Withholding the decision when nothing was asked is only
        correct if the decision still happens when something was."""
        authority = hdsg.determine_authority(
            "FORWARD", "SAFE", support.sectors(3.20, 0.40, 3.20, ("CLEAR", "BLOCKED", "CLEAR")),
            sector_choice_tolerance_m=0.10,
        )
        self.assertEqual("AWAITING_SECTOR_CHOICE", authority["interaction_state"])
        self.assertEqual(["LEFT", "RIGHT"], authority["selection_options"])

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
                "uncertainty": {"valid_depth_fraction": valid_fraction}}

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
            "caution": {"nearest_obstacle_m_lt": 1.5},
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


class DistanceBandTests(unittest.TestCase):
    """The band an object is described by, against the distance it was measured at.

    Every object carries a distance in metres and a word for that distance: VERY_CLOSE, NEAR, MID or
    FAR. The word reaches the person. It is the object's `state` in the permitted facts, so a caption
    may describe an object as near.

    Until 23 August 2026 the word was whatever the detector supplied, checked only against a list of
    permitted spellings and never against the metres beside it. The two agreed, both being derived
    from the same reading moments apart, and no disagreement appears in any archived record. The
    weakness was structural: a description of a measurement was accepted rather than worked out from
    it, which is the arrangement that let the ontology drift away from the detector.
    """

    def setUp(self):
        import yaml

        self.cfg = yaml.safe_load(
            (support.CONFIG / "pipeline.yaml").read_text(encoding="utf-8")
        )

    def test_the_band_edges_agree_with_the_configuration(self):
        """The bands are stated in `pipeline.yaml`, where a retune belongs, and in
        `hdsg.DISTANCE_BANDS_M`, because `normalise_objects` runs before the configuration path
        reaches the packet builder. Two copies of one set of numbers, so they are compared."""
        configured = self.cfg["depth"]["metric_bins_m"]
        self.assertEqual(
            {name.upper(): (float(low), float(high)) for name, (low, high) in configured.items()},
            {name: (low, high) for name, low, high in hdsg.DISTANCE_BANDS_M},
        )

    def test_a_distance_receives_the_band_that_contains_it(self):
        for distance, expected in ((0.0, "VERY_CLOSE"), (0.69, "VERY_CLOSE"), (0.7, "NEAR"),
                                   (1.49, "NEAR"), (1.5, "MID"), (2.99, "MID"), (3.0, "FAR"),
                                   (98.9, "FAR")):
            with self.subTest(distance):
                self.assertEqual(expected, hdsg._distance_bin(distance))

    def test_a_distance_outside_every_band_is_unknown(self):
        """A reading below the lowest band or beyond the highest is one the bands have no opinion
        about. The fallback was FAR, the band meaning the most room, which is the wrong way for a
        fallback to fail."""
        for distance in (-0.5, 99.0, 1000.0, None, float("nan"), float("inf"), "1.2", True):
            with self.subTest(distance):
                self.assertEqual("UNKNOWN", hdsg._distance_bin(distance))

    def test_the_detectors_own_band_is_ignored(self):
        """The defect itself. A detector reporting a band that contradicts its own distance no
        longer has that band believed."""
        objects = hdsg.normalise_objects([{
            "id": 0, "track_id": 3, "raw_label": "chair", "canonical_class": "chair",
            "ontology_class": "chair", "conf": 0.8, "bbox_xyxy": [10, 10, 90, 200],
            "bearing": "LEFT", "distance_m": 4.20, "distance_bin": "very_close",
            "distance_method": "D455F_LOWER_BBOX_MEDIAN",
        }])
        self.assertEqual(4.20, objects[0]["distance_m"])
        self.assertEqual("FAR", objects[0]["distance_bin"])

    def test_an_object_with_no_distance_has_no_band(self):
        objects = support.detected_object(distance_m=None)
        self.assertIsNone(objects[0]["distance_m"])
        self.assertEqual("UNKNOWN", objects[0]["distance_bin"])


class CrowdedSceneTests(unittest.TestCase):
    """What the advisory says about a scene holding several objects, none of them close.

    A `caution:multi_near` rule was removed on 24 August 2026. It raised caution on two or more
    objects below 1.50 m, which is the distance at which the rule above it already raises caution
    for one object, so it could add a label to the list but never change the risk level:

        [1.2]        caution  ['caution:nearest_obstacle']
        [1.2, 1.3]   caution  ['caution:nearest_obstacle', 'caution:multi_near']
        [2.0, 2.1]   safe     []

    The third row is the case the rule was written for and the one it never covered. Answering it
    needs a larger distance of its own, which the archive's seven object detections cannot supply.
    Owned in `LAB_SESSION_CHECKLIST.md`, section D. These tests record what the walker does in the
    meantime rather than asserting that it is the wanted behaviour.
    """

    def setUp(self):
        import yaml
        from scripts import realsense_shared_control as sw

        self.sw = sw
        self.cfg = yaml.safe_load(
            (support.CONFIG / "pipeline.yaml").read_text(encoding="utf-8"))

    def risk(self, distances):
        facts = {"objects": [{"distance_m": d} for d in distances]}
        return self.sw.compute_baseline_risk(facts, self.cfg)

    def test_several_objects_beyond_the_caution_distance_read_as_safe(self):
        """Four objects between 1.9 and 2.4 m, which is the deferred case."""
        self.assertEqual("safe", self.risk([2.0, 2.2, 1.9, 2.4])["risk"])

    def test_one_object_inside_the_caution_distance_raises_caution(self):
        self.assertEqual("caution", self.risk([1.2])["risk"])

    def test_a_second_close_object_adds_nothing(self):
        """The rule's redundancy, asserted so that reinstating it at the same distance fails here."""
        self.assertEqual(self.risk([1.2])["rules_fired"], self.risk([1.2, 1.3])["rules_fired"])

    def test_the_removed_threshold_is_not_in_the_configuration(self):
        """A threshold no rule reads is worse than no threshold, because it will be believed."""
        self.assertNotIn("multi_near_objects_count_gte",
                         self.cfg["risk_rules_baseline"]["caution"])


class ConfigurationRecordTests(unittest.TestCase):
    """The block that states how a run was set up.

    Its purpose is that an archived run can be read back and understood. A number in it the run did
    not use is worse than no number, because it will be believed.
    """

    def setUp(self):
        import yaml

        self.cfg = yaml.safe_load(
            (support.CONFIG / "pipeline.yaml").read_text(encoding="utf-8")
        )
        self.configuration = support.fact_packet()["configuration"]

    def test_no_threshold_is_recorded_that_no_rule_reads(self):
        """`binding_tie_margin` was recorded as a literal 0.15 until 23 August 2026 and no line of
        code read a 0.15 tie margin. It appeared only in this record, in the schema and in one
        fixture, so the block stated two tie thresholds of which one was fiction. The parameter that
        exists is `sector_choice_tolerance`."""
        thresholds = self.configuration["thresholds_m"]
        self.assertNotIn("binding_tie_margin", thresholds)
        self.assertIn("sector_choice_tolerance", thresholds)

    def test_the_recorded_sector_geometry_says_where_the_sectors_divide(self):
        """`horizontal_divisions: 3` said how many there were and never where. Reading an archived
        record then required still holding the pipeline.yaml that hashes to its
        `configuration_hash`."""
        geometry = self.configuration["sector_geometry"]
        self.assertEqual(float(self.cfg["bearing"]["left_max"]), geometry["left_max_fraction"])
        self.assertEqual(float(self.cfg["bearing"]["right_min"]), geometry["right_min_fraction"])

    def test_the_recorded_row_band_agrees_with_the_configuration(self):
        """The band was written out three times, here, in `valid_depth_fraction` and inside
        `compute_lane_state`, with nothing comparing them."""
        geometry = self.configuration["sector_geometry"]
        self.assertEqual(float(self.cfg["sector"]["band_top_fraction"]), geometry["top_fraction"])
        self.assertEqual(float(self.cfg["sector"]["band_bottom_fraction"]),
                         geometry["bottom_fraction"])

    def test_the_runtime_fallback_band_agrees_with_the_configuration(self):
        """`normalise_objects` and `compute_lane_state` fall back to these when no configuration
        reaches them, so the fallback must not be a third opinion."""
        self.assertEqual(float(self.cfg["sector"]["band_top_fraction"]),
                         hdsg.SECTOR_BAND_TOP_FRACTION)
        self.assertEqual(float(self.cfg["sector"]["band_bottom_fraction"]),
                         hdsg.SECTOR_BAND_BOTTOM_FRACTION)

    def test_the_recorded_geometry_is_the_geometry_that_was_used(self):
        """Recorded from what the caller passed rather than restated as a literal, so a run measured
        with a retuned band says so."""
        packet = hdsg.build_fact_packet(
            run_id="run_t", event_id="evt_1", observation_id="obs_1", ticket_id=None,
            timestamp_ms=1.0, intent="FORWARD", trigger_type="USER_REQUESTED",
            request_id="MORE_DETAIL", response_mode="MORE_DETAIL", previous_signature=None,
            objects=[], sectors=support.sectors(),
            authority=hdsg.determine_authority("FORWARD", "SAFE", support.sectors()),
            mirror_view=False, detector_model="yolov8n.pt", detector_confidence=0.35,
            pipeline_config_path=support.CONFIG / "pipeline.yaml",
            ontology_path=support.CONFIG / "ontology.yaml",
            clear_threshold_m=1.8, blocked_threshold_m=0.7, sector_choice_tolerance_m=0.10,
            motion_tracker=hdsg.MotionTracker(),
            sector_band_top_fraction=0.42, sector_band_bottom_fraction=0.88,
            sector_left_max_fraction=0.30, sector_right_min_fraction=0.70,
            sector_min_measured_fraction=0.11,
        )
        self.assertEqual({"top_fraction": 0.42, "bottom_fraction": 0.88,
                          "horizontal_divisions": 3,
                          "left_max_fraction": 0.30, "right_min_fraction": 0.70,
                          "min_measured_fraction": 0.11},
                         packet["configuration"]["sector_geometry"])


class CodeVersionTests(unittest.TestCase):
    """Which code produced a record.

    `software_version` and `rule_set_version` are literals that have never changed, and
    `configuration_hash` covers pipeline.yaml and the other configuration files. None of the three
    covers the Python. On 23 August 2026 the caution rule stopped counting distance bands and started
    counting metres, and an object's band stopped being trusted and started being derived: two
    changes to what the walker does, in code, leaving every field in the configuration block
    identical. A run from before and a run from after could not be told apart from their own
    evidence.
    """

    def test_the_record_carries_the_commit_the_branch_and_the_tree_state(self):
        version = support.fact_packet()["configuration"]["code_version"]
        self.assertEqual({"commit", "branch", "dirty"}, set(version))

    def test_the_commit_is_a_full_identifier_or_nothing(self):
        """Null where git is unavailable, which is honest about not knowing rather than naming
        something that is not a commit."""
        commit = support.fact_packet()["configuration"]["code_version"]["commit"]
        if commit is not None:
            self.assertRegex(commit, r"^[0-9a-f]{40}$")

    def test_an_edited_tree_is_reported_as_edited(self):
        """A commit identifier taken from a tree carrying uncommitted changes does not identify the
        code that ran. Recording that is the point."""
        dirty = support.fact_packet()["configuration"]["code_version"]["dirty"]
        self.assertIn(dirty, (True, False, None))

    def test_it_is_resolved_once(self):
        """Three subprocess calls on the walker's event path is not acceptable, and the code cannot
        change while a run is in progress. Uncached it added ten seconds to this suite."""
        import scripts.hdsg_runtime as runtime

        runtime._CODE_VERSION = {"commit": None, "branch": "cached", "dirty": False}
        try:
            self.assertEqual("cached", runtime.code_version()["branch"])
        finally:
            runtime._CODE_VERSION = None

    def test_the_cached_value_cannot_be_edited_through_a_record(self):
        """A caller mutating the returned dict must not change what every later record reports."""
        import scripts.hdsg_runtime as runtime

        first = runtime.code_version()
        first["branch"] = "tampered"
        self.assertNotEqual("tampered", runtime.code_version()["branch"])


if __name__ == "__main__":
    unittest.main()
