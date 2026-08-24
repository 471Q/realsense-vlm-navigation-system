"""The RGB-D recording, which is the evidence Chapter 5 replays.

A defect here damages the record rather than the walker. The module promises two things: that the
file is a faithful copy of what the sensor gave, and that a run which failed to record every
observation says so. Both were untrue in one respect each until 25 August 2026.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

import support  # noqa: F401
from scripts import hdsg_recording as recording  # noqa: E402


def frames(value=1.5, size=8):
    return (np.zeros((size, size, 3), dtype=np.uint8),
            np.full((size, size), value, dtype=np.float32))


class TheFileSaysWhatTheSensorSaid(unittest.TestCase):
    """A pixel with no reading is stored as zero, whatever produced it.

    An out-of-range value was clipped to 65535 until 25 August 2026. The reader turns that back into
    zero, so the round trip was right and nothing downstream was wrong; the file was not. Anyone
    opening the PNG with another tool saw 65.535 metres where the walker saw nothing, which is the
    sentinel confusion removed from `capture_thread` on 24 August, preserved on disk.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        recorder = recording.ObservationRecorder(self.directory).start()
        colour, depth = frames(size=4)
        depth[0] = [1.234, 80.0, float("nan"), float("inf")]
        depth[1] = [-1.0, 65.535, 0.0, 2.0]
        recorder.record("obs_1", colour, depth, 1.0)
        self.stats = recorder.stop()
        self.raw = cv2.imread(str(self.directory / "obs_1.depth_mm.png"), cv2.IMREAD_UNCHANGED)

    def test_an_ordinary_distance_is_stored_in_millimetres(self):
        self.assertEqual(1234, int(self.raw[0, 0]))

    def test_a_distance_beyond_the_sixteen_bit_ceiling_is_stored_as_no_reading(self):
        self.assertEqual(0, int(self.raw[0, 1]))

    def test_the_ceiling_itself_is_stored_as_no_reading(self):
        """65535 millimetres is the sensor's own sentinel, so a real reading of exactly that
        distance cannot be told apart from an absent one and is recorded as absent."""
        self.assertEqual(0, int(self.raw[1, 1]))

    def test_a_non_finite_reading_is_stored_as_no_reading(self):
        self.assertEqual(0, int(self.raw[0, 2]))
        self.assertEqual(0, int(self.raw[0, 3]))

    def test_a_negative_reading_is_stored_as_no_reading(self):
        self.assertEqual(0, int(self.raw[1, 0]))

    def test_nothing_on_disk_carries_the_sentinel(self):
        self.assertEqual(0, int(np.count_nonzero(self.raw == recording.UINT16_MAX)))

    def test_the_reader_returns_metres(self):
        reader = recording.ObservationReader(self.directory)
        _, depth = reader.load(reader.entries()[0])
        self.assertAlmostEqual(1.234, float(depth[0, 0]), places=4)
        self.assertEqual(0.0, float(depth[0, 1]))

    def test_the_reader_still_voids_the_sentinel_for_older_recordings(self):
        """The archived run of 22 August was written before this change and holds 65535 pixels."""
        directory = Path(tempfile.mkdtemp())
        (directory / "manifest.jsonl").write_text(
            '{"observation_id":"obs_1","sensor_timestamp_ms":1.0,'
            '"colour_path":"obs_1.color.png","depth_path":"obs_1.depth_mm.png"}\n',
            encoding="utf-8")
        cv2.imwrite(str(directory / "obs_1.color.png"), np.zeros((4, 4, 3), dtype=np.uint8))
        legacy = np.full((4, 4), recording.UINT16_MAX, dtype=np.uint16)
        legacy[0, 0] = 1200
        cv2.imwrite(str(directory / "obs_1.depth_mm.png"), legacy)
        reader = recording.ObservationReader(directory)
        _, depth = reader.load(reader.entries()[0])
        self.assertEqual(1.2, round(float(depth[0, 0]), 3))
        self.assertEqual(0.0, float(depth[1, 1]))


class AnIncompleteRecordingSaysSo(unittest.TestCase):
    """The module docstring promises exactly this and it did not hold.

    Ten observations queued behind a slow writer produced `written: 1, dropped: 0, complete: True`
    with one frame on disk. `stop` joined with a timeout, abandoned the thread when the timeout
    passed, and returned the counters without looking at what was still queued.
    """

    def slow_recorder(self, directory, seconds=0.4):
        recorder = recording.ObservationRecorder(directory).start()
        original = recorder._write

        def slow(item):
            time.sleep(seconds)
            original(item)

        recorder._write = slow
        return recorder

    def test_what_is_still_queued_when_the_join_gives_up_is_counted(self):
        recorder = self.slow_recorder(Path(tempfile.mkdtemp()))
        colour, depth = frames()
        for index in range(10):
            recorder.record(f"obs_{index}", colour, depth, float(index))
        stats = recorder.stop(timeout_s=0.5)
        self.assertFalse(stats["complete"])
        self.assertGreater(stats["dropped"], 0)
        self.assertEqual(10, stats["offered"])
        # At most one observation is unaccounted, the one the writer was part way through when the
        # join gave up. It may then finish and be counted as written, so the identity is asserted
        # to within one rather than exactly. `complete` is false either way, which is the guarantee
        # that matters: the count errs towards reporting the recording incomplete.
        accounted = stats["written"] + stats["dropped"] + stats["failed"]
        self.assertIn(accounted, (9, 10), stats)

    def test_an_observation_offered_after_stop_is_counted(self):
        """It returned False and incremented nothing, so it left no trace at all."""
        recorder = recording.ObservationRecorder(Path(tempfile.mkdtemp())).start()
        recorder.stop()
        colour, depth = frames()
        self.assertFalse(recorder.record("obs_late", colour, depth, 1.0))
        self.assertEqual(1, recorder.stats()["dropped"])
        self.assertFalse(recorder.stats()["complete"])

    def test_an_observation_offered_before_start_is_counted(self):
        recorder = recording.ObservationRecorder(Path(tempfile.mkdtemp()))
        colour, depth = frames()
        self.assertFalse(recorder.record("obs_early", colour, depth, 1.0))
        self.assertFalse(recorder.stats()["complete"])

    def test_a_recording_that_finished_reports_itself_complete(self):
        """The counters must not simply always say incomplete."""
        directory = Path(tempfile.mkdtemp())
        recorder = recording.ObservationRecorder(directory).start()
        colour, depth = frames()
        for index in range(5):
            recorder.record(f"obs_{index}", colour, depth, float(index))
        stats = recorder.stop()
        self.assertEqual({"offered": 5, "written": 5, "dropped": 0, "failed": 0,
                          "complete": True}, stats)
        self.assertEqual(5, len(recording.ObservationReader(directory).entries()))

    def test_an_observation_that_vanishes_without_being_counted_is_still_a_gap(self):
        """The counters have to agree with each other, not merely avoid recording a failure.

        A write that neither succeeds nor raises leaves `written`, `dropped` and `failed` all at
        zero while an observation has gone. Nothing else here catches that: the drain count covers
        what was still queued, and the failure count covers what threw. Comparing the three against
        `offered` is what turns a silently missing frame into an incomplete recording.
        """
        recorder = recording.ObservationRecorder(Path(tempfile.mkdtemp())).start()
        recorder._write = lambda item: None
        colour, depth = frames()
        recorder.record("obs_1", colour, depth, 1.0)
        stats = recorder.stop()
        self.assertEqual(
            {"offered": 1, "written": 0, "dropped": 0, "failed": 0, "complete": False}, stats)

    def test_a_write_that_raises_is_counted_as_failed(self):
        recorder = recording.ObservationRecorder(Path(tempfile.mkdtemp())).start()

        def refuse(item):
            raise RuntimeError("the disk is full")

        recorder._write = refuse
        colour, depth = frames()
        recorder.record("obs_1", colour, depth, 1.0)
        stats = recorder.stop()
        self.assertEqual(1, stats["failed"])
        self.assertFalse(stats["complete"])


if __name__ == "__main__":
    unittest.main()
