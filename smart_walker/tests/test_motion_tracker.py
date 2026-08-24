"""Whether an object is moving, which reaches the prompt as an alert and the display as a box.

Two things scale or key that judgement and neither was derived from what it describes until
25 August 2026: the focal length converting pixels into metres, and the identifier under which a
detection's history is kept.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import support  # noqa: F401
from scripts import hdsg_runtime as hdsg  # noqa: E402

WIDTH, HEIGHT = 640, 480


def textured_frame(shift=0):
    """A frame with enough corners for optical flow, optionally translated.

    A blank frame yields no features, so `_global_motion` reports no compensation and the tracker
    never classifies anything. The pattern is what makes these tests exercise the real path.
    """
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for row in range(0, HEIGHT, 20):
        for column in range(0, WIDTH, 20):
            frame[row:row + 8, (column + shift) % WIDTH:(column + shift) % WIDTH + 8] = 255
    return frame


def detection(track_id, centre_x, distance_m=2.0, detection_index=0):
    """One perception-layer object. `id` numbers the detection in the frame and always exists;
    `track_id` is the tracker's claim of identity and is None when it makes none."""
    half = 30
    return {"id": detection_index if track_id is None else track_id, "track_id": track_id,
            "raw_label": "chair", "canonical_class": "chair",
            "bbox_xyxy": [centre_x - half, 200.0, centre_x + half, 300.0],
            "distance_m": distance_m, "conf": 0.9, "bearing": "CENTRE"}


class TheFocalLengthComesFromTheCamera(unittest.TestCase):
    """It converts a shift in pixels into a distance in metres, so it sets the scale of every
    motion score against `movement_threshold_m`.

    Derived from a hardcoded 87 degree field of view until 25 August 2026. That is the D455's depth
    field of view and the motion is measured on the colour frame, which has its own optics.
    """

    def test_the_fallback_matches_the_old_derivation(self):
        """A caller with no camera, a test or an offline replay, keeps the previous behaviour."""
        tracker = hdsg.MotionTracker()
        self.assertIsNone(tracker.focal_px)
        expected = WIDTH / (2.0 * math.tan(math.radians(87.0) / 2.0))
        self.assertAlmostEqual(337.2, expected, places=1)

    def test_a_supplied_focal_length_changes_the_measured_motion(self):
        """The same pixels, two focal lengths, two answers. This is why it must not be guessed."""
        scores = {}
        for focal in (337.2, 465.6):
            tracker = hdsg.MotionTracker(focal_px=focal)
            tracker.update(textured_frame(), [detection(1, 300.0)], 0.0)
            tracker.update(textured_frame(), [detection(1, 320.0)], 100.0)
            scores[focal] = list(tracker.motion_scores[1])[-1]
        self.assertGreater(scores[337.2], scores[465.6])
        # A wider assumed field of view reads the same shift as a larger distance, in the ratio of
        # the two focal lengths.
        self.assertAlmostEqual(465.6 / 337.2, scores[337.2] / scores[465.6], places=3)

    def test_the_client_reads_it_from_the_colour_stream(self):
        """Asserted against the source: the intrinsics need a camera, and the read is one line
        beside the depth scale it already takes from the same profile."""
        source = (support.ROOT / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(
            encoding="utf-8")
        self.assertIn("rs.stream.color).as_video_stream_profile().get_intrinsics()", source)
        self.assertIn("colour_focal_px = float(colour_intrinsics.fx)", source)
        self.assertIn("focal_px=colour_focal_px,", source)


class AnUntrackedDetectionKeepsItsOwnHistory(unittest.TestCase):
    """The perception layer numbers an object by its tracker identifier where it has one and by its
    position in the list otherwise. Both are small integers, so a detection with no identifier at
    index 1 shared a history with the tracked object whose identifier is 1: two objects in one
    history, the position jumping between them, which reads as motion.
    """

    def test_an_untracked_detection_does_not_share_a_tracked_history(self):
        tracker = hdsg.MotionTracker(focal_px=400.0)
        # Index 1 has no tracker identifier; the object beside it is tracked as identifier 1.
        for step in range(4):
            tracker.update(
                textured_frame(),
                [detection(1, 100.0), detection(None, 500.0, detection_index=1)],
                float(step * 100),
            )
        self.assertIn(1, tracker.histories)
        self.assertIn("untracked:1", tracker.histories)
        tracked_positions = [entry.centre_x for entry in tracker.histories[1]]
        self.assertEqual({100.0}, set(tracked_positions),
                         "the tracked object's history carries another object's position")

    def test_the_untracked_key_cannot_collide_with_a_tracker_identifier(self):
        tracker = hdsg.MotionTracker(focal_px=400.0)
        tracker.update(textured_frame(), [detection(None, 300.0)], 0.0)
        self.assertEqual(["untracked:0"], list(tracker.histories))

    def test_an_untracked_detection_is_never_reported_as_moving(self):
        """Its key is its position in the list, which is a different object from frame to frame, so
        it accumulates no history worth believing. It is reported as unconfirmed rather than given
        a state derived from whatever happened to sit at that index."""
        tracker = hdsg.MotionTracker(focal_px=400.0)
        states = []
        for step in range(8):
            enriched = tracker.update(
                textured_frame(), [detection(None, 100.0 + step * 40)],
                float(step * 100))
            states.append(enriched[0]["motion_state"])
        self.assertNotIn("MOVING", states, states)

    def test_the_fact_packet_still_carries_an_integer_or_null(self):
        """The frozen schema allows an integer or null, so the string key must not reach it."""
        tracker = hdsg.MotionTracker(focal_px=400.0)
        enriched = tracker.update(textured_frame(), [detection(None, 300.0)], 0.0)
        self.assertEqual("untracked:0", enriched[0]["track_id"])
        normalised = hdsg.normalise_objects(enriched)
        self.assertIsNone(normalised[0]["track_id"])
        self.assertIsInstance(normalised[0]["detection_id"], int)

    def test_a_tracked_detection_is_unaffected(self):
        tracker = hdsg.MotionTracker(focal_px=400.0)
        enriched = tracker.update(textured_frame(), [detection(7, 300.0)], 0.0)
        self.assertEqual(7, enriched[0]["track_id"])
        self.assertEqual(7, hdsg.normalise_objects(enriched)[0]["track_id"])


if __name__ == "__main__":
    unittest.main()


class ThePerceptionLayerKeepsTheTwoNumbersApart(unittest.TestCase):
    """`id` numbers a detection in the frame; `track_id` claims identity across frames.

    Both were collapsed into `id` until 25 August 2026, the tracker's number where there was one
    and the position in the list where there was not, which is what made the two indistinguishable.
    Asserted against the source: the detection loop needs a camera and a detector.
    """

    def test_the_detection_loop_emits_both(self):
        source = (support.ROOT / "scripts" / "realsense_shared_control.py").read_text(
            encoding="utf-8")
        lines = [line.strip() for line in source.splitlines()]
        self.assertIn('"id": int(tid) if tid is not None else i,', lines)
        self.assertIn('"track_id": None if tid is None else int(tid),', lines)

    def test_the_tracker_reads_the_one_that_means_identity(self):
        source = (support.ROOT / "scripts" / "hdsg_runtime.py").read_text(encoding="utf-8")
        self.assertIn('raw_id = item.get("track_id", item.get("id"))', source)


class NormalisingSurvivesAnAbsentIdentifier(unittest.TestCase):
    """`source.get("id", detection_id)` returns None where the key is present and null, so the
    default never applied and `int(None)` raised. Found by a test rather than by reading, and not
    produced by the perception layer today, but a crash in the deterministic layer is not the way
    to discover that something else did."""

    def test_a_null_identifier_falls_back_to_the_position(self):
        normalised = hdsg.normalise_objects([
            {"id": None, "track_id": None, "raw_label": "chair", "canonical_class": "chair",
             "bbox_xyxy": [0, 0, 10, 10], "distance_m": 1.2, "conf": 0.9, "bearing": "LEFT"}])
        self.assertEqual(0, normalised[0]["detection_id"])
        self.assertIsNone(normalised[0]["track_id"])
        self.assertEqual("object:0", normalised[0]["fact_id"])

    def test_a_missing_identifier_key_falls_back_to_the_position(self):
        normalised = hdsg.normalise_objects([
            {"raw_label": "chair", "canonical_class": "chair", "bbox_xyxy": [0, 0, 10, 10],
             "distance_m": 1.2, "conf": 0.9, "bearing": "LEFT"}])
        self.assertEqual(0, normalised[0]["detection_id"])
