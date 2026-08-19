"""Tests for RGB-D recording and replay.

Replay fidelity is the point of this module: a condition replayed from a recording must receive
the same input the live run saw, or the matched-condition comparison it exists to support is not
matched. The round-trip tests below are therefore the substantive ones.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import hdsg_recording as rec
from scripts import hdsg_runtime as hdsg


class RecordingRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp.name) / "run"
        self.colour = np.zeros((48, 64, 3), np.uint8)
        self.colour[:24, :32] = (17, 200, 43)
        self.colour[24:, 32:] = (250, 5, 128)

    def tearDown(self):
        self._temp.cleanup()

    def _write_one(self, depth_m, observation_id="obs_000001"):
        recorder = rec.ObservationRecorder(self.directory).start()
        recorder.record(observation_id, self.colour, depth_m, 1234.5)
        stats = recorder.stop()
        self.assertTrue(stats["complete"], stats)
        self.assertEqual(stats["written"], 1)
        return rec.ObservationReader(self.directory)

    def test_colour_survives_the_round_trip_exactly(self):
        """PNG is used rather than JPEG so replay feeds the detector identical pixels."""
        reader = self._write_one(np.full((48, 64), 1.5, np.float32))
        entry = reader.entries()[0]
        colour, _ = reader.load(entry)
        np.testing.assert_array_equal(colour, self.colour)

    def test_depth_survives_the_round_trip_at_millimetre_units(self):
        depth = np.zeros((48, 64), np.float32)
        depth[:, :16] = 0.700
        depth[:, 16:32] = 1.523
        depth[:, 32:48] = 2.416
        depth[:, 48:] = 12.000
        reader = self._write_one(depth)
        _, restored = reader.load(reader.entries()[0])
        # The sensor reports integer millimetres, so a value that came from it round-trips
        # exactly. The tolerance covers only float32 representation of the division.
        np.testing.assert_allclose(restored, depth, atol=1e-6)

    def test_invalid_depth_becomes_zero_rather_than_a_distance(self):
        """NaN and infinity must not reappear as a plausible clearance on replay."""
        depth = np.full((48, 64), 2.0, np.float32)
        depth[0, 0] = np.nan
        depth[0, 1] = np.inf
        depth[0, 2] = -np.inf
        depth[0, 3] = -5.0
        reader = self._write_one(depth)
        _, restored = reader.load(reader.entries()[0])
        self.assertEqual(restored[0, 0], 0.0)
        self.assertEqual(restored[0, 1], 0.0)
        self.assertEqual(restored[0, 2], 0.0)
        self.assertEqual(restored[0, 3], 0.0)
        self.assertEqual(restored[1, 1], 2.0)

    def test_depth_beyond_the_sixteen_bit_ceiling_is_clipped_not_wrapped(self):
        """A wrapped value would reappear as a short, false clearance."""
        depth = np.full((4, 4), 300.0, np.float32)  # 300 m is 300000 mm, far beyond uint16
        reader = self._write_one(depth)
        _, restored = reader.load(reader.entries()[0])
        self.assertAlmostEqual(float(restored.max()), rec.UINT16_MAX / 1000.0, places=3)
        self.assertGreater(float(restored.min()), 60.0)

    def test_manifest_records_the_depth_scale_and_geometry(self):
        reader = self._write_one(np.full((48, 64), 1.0, np.float32))
        entry = reader.entries()[0]
        self.assertEqual(entry["observation_id"], "obs_000001")
        self.assertEqual(entry["sensor_timestamp_ms"], 1234.5)
        self.assertEqual(entry["depth_units"], "millimetre")
        self.assertTrue(entry["depth_scale_is_exact"])
        self.assertEqual((entry["width"], entry["height"]), (64, 48))

    def test_a_non_standard_depth_scale_is_flagged_as_inexact(self):
        recorder = rec.ObservationRecorder(self.directory, depth_scale=0.00025).start()
        recorder.record("obs_000001", self.colour, np.full((48, 64), 1.0, np.float32), 0.0)
        recorder.stop()
        entry = rec.ObservationReader(self.directory).entries()[0]
        self.assertFalse(entry["depth_scale_is_exact"])

    def test_observations_replay_in_recorded_order(self):
        recorder = rec.ObservationRecorder(self.directory).start()
        for index in range(5):
            depth = np.full((48, 64), 1.0 + index, np.float32)
            recorder.record(hdsg.next_identifier("obs", index + 1), self.colour, depth, index * 10.0)
        stats = recorder.stop()
        self.assertEqual(stats["written"], 5)

        reader = rec.ObservationReader(self.directory)
        seen = [(entry["observation_id"], float(depth.mean()))
                for entry, _, depth in reader.observations()]
        self.assertEqual([item[0] for item in seen], [f"obs_{n:06d}" for n in range(1, 6)])
        self.assertEqual([item[1] for item in seen], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_a_dropped_observation_marks_the_run_incomplete(self):
        """A recording that silently lost frames would invalidate a replay without warning."""
        recorder = rec.ObservationRecorder(self.directory, queue_size=1)
        # No worker started, so nothing drains the queue and the second record overflows it.
        recorder.directory.mkdir(parents=True, exist_ok=True)
        recorder._thread = object()  # type: ignore[assignment]
        self.assertTrue(recorder.record("obs_000001", self.colour, np.zeros((4, 4), np.float32), 0.0))
        self.assertFalse(recorder.record("obs_000002", self.colour, np.zeros((4, 4), np.float32), 0.0))
        stats = recorder.stats()
        self.assertEqual(stats["dropped"], 1)
        self.assertFalse(stats["complete"])

    def test_reader_rejects_a_directory_with_no_manifest(self):
        empty = Path(self._temp.name) / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            rec.ObservationReader(empty)

    def test_reader_rejects_a_depth_file_that_is_not_sixteen_bit(self):
        import cv2
        reader = self._write_one(np.full((48, 64), 1.0, np.float32))
        entry = reader.entries()[0]
        cv2.imwrite(str(self.directory / entry["depth_path"]), self.colour)
        with self.assertRaises(ValueError):
            reader.load(entry)

    def test_reader_accepts_single_channel_depth_in_either_shape(self):
        """OpenCV returns a single-channel PNG as (H, W) or (H, W, 1) depending on what else is
        loaded in the process. Importing the entry script's dependencies is enough to change it,
        so a reader that insisted on two dimensions worked in isolation and failed on the rig."""
        import unittest.mock
        reader = self._write_one(np.full((48, 64), 1.234, np.float32))
        entry = reader.entries()[0]
        expected, _ = reader.load(entry), None

        three_dimensional = np.full((48, 64, 1), 1234, np.uint16)
        with unittest.mock.patch.object(rec.cv2, "imread") as imread:
            imread.side_effect = [self.colour, three_dimensional]
            _, depth = reader.load(entry)
        self.assertEqual(depth.shape, (48, 64))
        np.testing.assert_allclose(depth, 1.234, atol=1e-6)


class FactPacketReferenceTests(unittest.TestCase):
    """The observation references must name files the recorder actually writes.

    HDSG_EVALUATION_CONTRACT.md section 12 requires an input file or recording identifier. A
    reference that does not match the recorder's filenames would satisfy the letter of that
    requirement while being unusable for replay, so the two are bound together here rather than
    left to agree by convention.
    """

    def _packet(self, **extra):
        sectors = {
            name: {"fact_id": f"sector:{name}", "clearance_m": 2.0, "valid": True,
                   "invalid_reason_codes": [], "status": "CLEAR"}
            for name in ("left", "centre", "right")
        }
        return hdsg.build_fact_packet(
            run_id="run_x", event_id="evt_1", observation_id="obs_000007", ticket_id="t1",
            timestamp_ms=1.0, intent="FORWARD", trigger_type="MOTION_INTENT_STARTED",
            request_id="AUTO_GUIDANCE", response_mode="AUTOMATIC", previous_signature=None,
            objects=[], sectors=sectors,
            authority=hdsg.determine_authority("FORWARD", "SAFE", sectors),
            mirror_view=False, detector_model="yolov8n.pt", detector_confidence=0.25,
            pipeline_config_path=Path("x"), ontology_path=Path("y"),
            clear_threshold_m=1.8, blocked_threshold_m=0.7, sector_choice_tolerance_m=0.1,
            motion_tracker=hdsg.MotionTracker(), **extra,
        )

    def test_a_live_packet_references_memory(self):
        packet = self._packet()
        self.assertTrue(packet["observation"]["rgb_ref"].startswith("memory://"))
        self.assertEqual(packet["identity"]["source_mode"], "LIVE")

    def test_a_recorded_packet_references_the_recorder_filenames(self):
        packet = self._packet(recording_dir="run_20260820_101500", source_mode="REPLAY")
        observation_id = packet["identity"]["observation_id"]
        self.assertEqual(
            packet["observation"]["rgb_ref"],
            f"run_20260820_101500/{rec.colour_filename(observation_id)}",
        )
        self.assertEqual(
            packet["observation"]["depth_ref"],
            f"run_20260820_101500/{rec.depth_filename(observation_id)}",
        )
        self.assertEqual(packet["identity"]["source_mode"], "REPLAY")

    def test_the_referenced_files_are_the_ones_a_reader_resolves(self):
        """Closes the loop: write a real recording, then resolve the packet's reference to it."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = "run_20260820_101500"
            recorder = rec.ObservationRecorder(root / run).start()
            recorder.record(
                "obs_000007", np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 1.5, np.float32), 0.0
            )
            self.assertTrue(recorder.stop()["complete"])

            packet = self._packet(recording_dir=run, source_mode="REPLAY")
            resolved = root / packet["observation"]["depth_ref"]
            self.assertTrue(resolved.is_file(), f"reference does not resolve: {resolved}")

            reader = rec.ObservationReader(root / run)
            _, depth = reader.load(reader.entries()[0])
            np.testing.assert_allclose(depth, 1.5, atol=1e-6)


class ReplayDeterminismTests(unittest.TestCase):
    """A replayed observation must produce the same deterministic decision as the live one."""

    def test_sector_facts_are_identical_across_a_record_and_replay_cycle(self):
        from scripts.realsense_vlm_on_change_qwen import compute_lane_state

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name) / "run"
            colour = np.zeros((480, 640, 3), np.uint8)
            depth = np.zeros((480, 640), np.float32)
            depth[:, :213] = 1.78
            depth[:, 213:426] = 1.52
            depth[:, 426:] = 2.42

            live_state = compute_lane_state(depth, mirror_view=False, clear_t=1.8, near_hi=0.7)
            live_sectors = hdsg.sectors_from_lane_state(live_state)
            live_authority = hdsg.determine_authority("FORWARD", "SAFE", live_sectors)

            recorder = rec.ObservationRecorder(directory).start()
            recorder.record("obs_000001", colour, depth, 0.0)
            self.assertTrue(recorder.stop()["complete"])

            _, replay_depth = rec.ObservationReader(directory).load(
                rec.ObservationReader(directory).entries()[0]
            )
            replay_state = compute_lane_state(
                replay_depth, mirror_view=False, clear_t=1.8, near_hi=0.7
            )
            replay_sectors = hdsg.sectors_from_lane_state(replay_state)
            replay_authority = hdsg.determine_authority("FORWARD", "SAFE", replay_sectors)

            self.assertEqual(live_sectors, replay_sectors)
            self.assertEqual(
                live_authority["motion_decision"], replay_authority["motion_decision"]
            )
            self.assertEqual(
                live_authority["selected_sector"], replay_authority["selected_sector"]
            )
            # The scene this reconstructs is the one from the live run that motivated the
            # constrained-sector redirect, so the replayed decision should be that redirect.
            self.assertEqual(replay_authority["motion_decision"], "REDIRECT")
            self.assertEqual(replay_authority["selected_sector"], "RIGHT")


if __name__ == "__main__":
    unittest.main()
