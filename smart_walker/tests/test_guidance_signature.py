"""The fingerprint that decides whether the walker speaks again.

An unchanged signature leaves the existing caption on screen and calls no model, so anything the
signature omits is something the walker can fail to react to. Two policies govern it and they do not
agree on how much distance it must carry, which is what these tests pin down.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401
from scripts import hdsg_runtime as hdsg  # noqa: E402

BLOCKED_AHEAD = {
    "left": {"status": "CLEAR", "clearance_m": 3.0},
    "centre": {"status": "BLOCKED", "clearance_m": 0.5},
    "right": {"status": "BLOCKED", "clearance_m": 0.5},
}
ALL_CLEAR = {name: {"status": "CLEAR", "clearance_m": 3.0}
             for name in ("left", "centre", "right")}


def obj(distance_m, label="person", bearing="LEFT", track_id=3, moving=True):
    return {"id": track_id, "track_id": track_id, "raw_label": label, "canonical_class": label,
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "distance_m": distance_m, "conf": 0.9,
            "bearing": bearing, "motion_state": "MOVING" if moving else "STATIC"}


def signature(objects, sectors, intent="FORWARD"):
    normalised = hdsg.normalise_objects(objects)
    authority = hdsg.determine_authority(intent, "STOP", sectors, normalised)
    authority["moving_object_fact_ids"] = [item["fact_id"] for item in normalised
                                           if item.get("motion_state") == "MOVING"]
    return authority, hdsg.guidance_signature(authority, "VALID", normalised)


class TheThresholdsAreTheOnesTheDecisionUses(unittest.TestCase):
    """A band computed against a different number from the status it reports would let the trigger
    and the decision disagree about the same object."""

    def test_the_bands_split_at_the_deterministic_thresholds(self):
        self.assertEqual(0.7, hdsg.OBJECT_STOP_BELOW_M)
        self.assertEqual(1.5, hdsg.OBJECT_CAUTION_BELOW_M)
        for distance, expected in ((2.0, "SAFE"), (1.5, "SAFE"), (1.49, "CAUTION"),
                                   (0.7, "CAUTION"), (0.69, "STOP"), (0.01, "STOP")):
            with self.subTest(distance=distance):
                self.assertEqual(expected, hdsg.object_distance_band(distance))

    def test_an_unmeasured_distance_has_its_own_band(self):
        """Not folded into SAFE. An object whose depth failed is not known to be far away, and
        collapsing the two would hide the moment a measurement returns.

        Zero belongs here rather than in STOP. `_finite` treats a non-positive distance as no
        reading throughout, and `determine_authority` skips such an object entirely, so banding it
        as STOP would have the signature report a hazard the decision does not see.
        """
        for value in (None, 0.0, -1.0, float("nan"), float("inf"), "close"):
            with self.subTest(value=value):
                self.assertEqual("UNMEASURED", hdsg.object_distance_band(value))

    def test_an_overridden_threshold_moves_the_band(self):
        self.assertEqual("SAFE", hdsg.object_distance_band(0.9, 0.7, 0.8))
        self.assertEqual("CAUTION", hdsg.object_distance_band(0.9, 0.7, 1.5))

    def test_the_packet_bands_against_the_threshold_it_was_decided_with(self):
        """`build_fact_packet` takes the run's thresholds. Left to the module defaults the signature
        would band an object at one distance while the decision used another."""
        source = (support.ROOT / "scripts" / "hdsg_runtime.py").read_text(encoding="utf-8")
        self.assertIn("object_stop_below_m=blocked_threshold_m,", source)
        self.assertIn("object_caution_below_m=object_caution_below_m)", source)

    def test_the_client_passes_the_same_two(self):
        source = (support.ROOT / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(
            encoding="utf-8")
        self.assertIn("object_stop_below_m=blocked_threshold_m,", source)
        self.assertIn("object_caution_below_m=object_caution_below_m,", source)


class AMovingObjectCrossingAThresholdIsAChange(unittest.TestCase):
    """HDSG_MOVING_OBJECT_DISPLAY_POLICY.md section 4 requires a new caption when a moving object
    enters a different deterministic distance state.

    The crossing usually changes the motion decision too, and the signature already carried that.
    It does not always. With the centre blocked by a wall, a moving person on the left closing from
    0.90 m to 0.40 m crosses the 0.70 m stop threshold while the decision stays STOP, and until
    25 August 2026 the two frames produced identical signatures: the person walked into stop range
    and the walker kept whatever it had already said.
    """

    def test_the_crossing_changes_the_signature_with_the_decision_unchanged(self):
        before_authority, before = signature([obj(0.90)], BLOCKED_AHEAD)
        after_authority, after = signature([obj(0.40)], BLOCKED_AHEAD)
        self.assertEqual("STOP", before_authority["motion_decision"])
        self.assertEqual(before_authority["motion_decision"],
                         after_authority["motion_decision"])
        self.assertEqual(before_authority["selected_sector"],
                         after_authority["selected_sector"])
        self.assertEqual(before_authority["action_binding"], after_authority["action_binding"])
        self.assertNotEqual(before, after)

    def test_each_band_is_reached(self):
        bands = [signature([obj(distance)], BLOCKED_AHEAD)[1].split("|")[-1]
                 for distance in (2.0, 0.90, 0.40)]
        self.assertEqual(["object:person:LEFT:MOVING:SAFE",
                          "object:person:LEFT:MOVING:CAUTION",
                          "object:person:LEFT:MOVING:STOP"], bands)

    def test_movement_within_one_band_is_still_suppressed(self):
        """The point of the signature. A distance trigger inside one state is excluded by
        HDSG_HYBRID_CAPTION_POLICY.md section 5, and reinstating it recreates the periodic caption
        the whole mechanism exists to remove."""
        self.assertEqual(signature([obj(0.65)], BLOCKED_AHEAD)[1],
                         signature([obj(0.10)], BLOCKED_AHEAD)[1])


class AStationaryObjectCarriesNoBand(unittest.TestCase):
    """HDSG_HYBRID_CAPTION_POLICY.md section 5 keeps a material-distance-change trigger out of the
    approved baseline. Only the moving-object policy authorises one, so only a moving object gets a
    band. A stationary object crossing a threshold still changes its status, and the decision
    carries that where the decision changes.
    """

    def test_a_stationary_object_gets_no_band(self):
        _, text = signature([obj(0.40, moving=False)], BLOCKED_AHEAD)
        self.assertIn("object:person:LEFT:PRESENT", text)
        self.assertNotIn("PRESENT:STOP", text)

    def test_a_stationary_object_closing_within_one_decision_is_suppressed(self):
        self.assertEqual(signature([obj(0.90, moving=False)], BLOCKED_AHEAD)[1],
                         signature([obj(0.40, moving=False)], BLOCKED_AHEAD)[1])

    def test_a_stationary_object_that_changes_the_decision_still_shows(self):
        """The general case is not silent, it is carried by the decision rather than by a band."""
        self.assertNotEqual(signature([obj(2.0, moving=False)], ALL_CLEAR)[1],
                            signature([obj(0.40, moving=False)], ALL_CLEAR)[1])

    def test_starting_to_move_is_a_change_on_its_own(self):
        self.assertNotEqual(signature([obj(0.90, moving=False)], BLOCKED_AHEAD)[1],
                            signature([obj(0.90, moving=True)], BLOCKED_AHEAD)[1])


class TheTrackerIdentifierIsDeliberatelyAbsent(unittest.TestCase):
    """Objects are described by label, bearing and movement state rather than by their number, so a
    tracker that renumbers the same chair does not re-announce it.

    The cost is that a person leaving on the left and a different person arriving at the same
    bearing and band produce the same signature. HDSG_MOVING_OBJECT_DISPLAY_POLICY.md section 4 says
    "the same tracked object", which reads as though identity is carried. Held as a test so the
    behaviour is stated rather than discovered, and recorded as a divergence rather than resolved.
    """

    def test_renumbering_the_same_object_is_not_a_change(self):
        self.assertEqual(signature([obj(0.90, track_id=3)], BLOCKED_AHEAD)[1],
                         signature([obj(0.90, track_id=7)], BLOCKED_AHEAD)[1])

    def test_a_different_object_at_the_same_bearing_and_band_is_not_a_change_either(self):
        self.assertEqual(signature([obj(0.90, track_id=3)], BLOCKED_AHEAD)[1],
                         signature([obj(0.80, track_id=7)], BLOCKED_AHEAD)[1])

    def test_a_different_class_is_a_change(self):
        self.assertNotEqual(signature([obj(0.90, label="person")], BLOCKED_AHEAD)[1],
                            signature([obj(0.90, label="chair")], BLOCKED_AHEAD)[1])

    def test_a_different_bearing_is_a_change(self):
        self.assertNotEqual(signature([obj(0.90, bearing="LEFT")], BLOCKED_AHEAD)[1],
                            signature([obj(0.90, bearing="RIGHT")], BLOCKED_AHEAD)[1])


if __name__ == "__main__":
    unittest.main()
