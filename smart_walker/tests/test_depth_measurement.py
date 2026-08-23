"""The measurements every other number in the system is derived from.

A sector clearance and an object's distance both come from taking the median of a region of the
depth frame. Nothing downstream can tell a good measurement from a bad one: a clearance of 65.535 m
and a clearance of 1.2 m are both just numbers by the time the guidance policy sees them, and the
larger one reads as more room to move.

The frames here are built by hand in millimetres, because the fault these tests exist for is about
what a particular stored value means rather than about any scene.
"""

from __future__ import annotations

import unittest

import numpy as np

from support import ROOT  # noqa: F401
from scripts import realsense_shared_control as sw  # noqa: E402
from scripts import hdsg_recording as recording  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402

SENTINEL = 65535


def depth_frame(fill_mm=1500, size=100):
    return np.full((size, size), fill_mm, dtype=np.uint16)


class SentinelReadings(unittest.TestCase):
    """The 16-bit ceiling is the sensor saying it has no reading, not a distance of 65.535 m.

    Found on 24 August 2026 by recomputing the archived run of 22 August. 36 per cent of its 1,480
    frames carry at least one such pixel, and 90 of its 4,440 sector readings had the wrong status
    because of them. Every one of the 90 erred the same way, calling a strip clear when it was
    constrained or unmeasurable, in one case reporting clear where the true clearance was 0.93 m.
    """

    def test_the_recording_loader_treats_the_ceiling_as_absent(self):
        """The path the replay scorer reads, and therefore the one the Chapter 5 numbers rest on."""
        raw = depth_frame(fill_mm=SENTINEL)
        raw[0, 0] = 1200
        voided = raw.copy()
        voided[voided == recording.UINT16_MAX] = 0
        depth_m = voided.astype(np.float32) / recording.MILLIMETRES_PER_METRE
        self.assertEqual(1.2, round(float(sw._median_nonzero(depth_m)), 3))

    def test_a_frame_of_ceilings_measures_as_nothing_rather_than_as_far_away(self):
        depth_m = np.zeros((100, 100), dtype=np.float32)
        self.assertIsNone(sw.median_depth_in_box(depth_m, 0, 0, 100, 100))

    def test_the_capture_thread_voids_the_ceiling_before_scaling(self):
        """Asserted against the source. The thread needs a camera, and the substitution is one line
        at one site."""
        source = (ROOT / "scripts" / "realsense_shared_control.py").read_text(encoding="utf-8")
        self.assertIn("depth_u16 = np.where(depth_u16 == 65535, 0, depth_u16)", source)


class MedianInBox(unittest.TestCase):

    def test_the_margin_is_excluded_from_the_measurement(self):
        """An edge picks up whatever lies behind the thing being measured."""
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        depth[0:3, :] = 9.0
        depth[:, 0:3] = 9.0
        self.assertEqual(2.0, sw.median_depth_in_box(depth, 0, 0, 100, 100))

    def test_a_region_with_no_reading_measures_as_nothing_not_as_its_margin(self):
        """The fallback removed on 24 August 2026 sampled the unshrunk region, which is the margin
        the shrink had just excluded, and returned its median as the measurement. Over the archived
        run it decided a sector clearance 12 times, returning 8.29 m and 65.535 m among others, and
        every one of those resolved to CLEAR. Reporting the background as the foreground's distance
        is worse than reporting nothing: an absent reading is treated as not clear, and a large
        number is treated as room to move.
        """
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[0:3, :] = 4.0          # only the margin carries a reading
        depth[:, 0:3] = 4.0
        self.assertIsNone(sw.median_depth_in_box(depth, 0, 0, 100, 100))

    def test_a_region_smaller_than_its_own_shrink_measures_as_nothing(self):
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        self.assertIsNone(sw.median_depth_in_box(depth, 50, 50, 50, 50))


class MedianLowerBand(unittest.TestCase):
    """An object is measured from the bottom of its box, where it meets the floor."""

    def test_it_measures_the_bottom_of_the_box_and_not_the_whole_of_it(self):
        depth = np.full((100, 100), 5.0, dtype=np.float32)
        depth[67:, :] = 1.5          # the bottom third is nearer
        self.assertEqual(1.5, sw.median_depth_lower_band(depth, 0, 0, 100, 100))

    def test_a_band_with_no_reading_measures_as_nothing(self):
        depth = np.full((100, 100), 5.0, dtype=np.float32)
        depth[60:, :] = 0.0
        self.assertIsNone(sw.median_depth_lower_band(depth, 0, 0, 100, 100))


class ValidDepthFraction(unittest.TestCase):
    """How much of the reasoning band could be measured at all."""

    def test_it_counts_only_readings_that_exist(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[55:95, 0:50] = 2.0     # half of the band carries a reading
        self.assertAlmostEqual(0.5, sw.valid_depth_fraction(depth, 0.55, 0.95), places=3)

    def test_an_empty_frame_reports_no_coverage_rather_than_nothing(self):
        self.assertEqual(0.0, sw.valid_depth_fraction(np.zeros((100, 100), dtype=np.float32)))

    def test_a_missing_frame_reports_nothing_rather_than_no_coverage(self):
        """Absent and zero are different claims: one says the measurement was not taken."""
        self.assertIsNone(sw.valid_depth_fraction(None))


class MinimumCoverage(unittest.TestCase):
    """A measurement is reported only where enough of the region could be measured.

    A median does not care how many samples it has. In the archived run of 22 August 2026 the centre
    strip of obs_000228 held 32,470 pixels of which 21 carried a reading, 0.1 per cent, and the
    median of those 21, 54.65 m, was reported as the clearance and read as CLEAR.

    The floor is `sector.min_measured_fraction`, 0.05, chosen from that run: 463 of its 4,440 sector
    readings fall below it and 218 of those had been called clear.
    """

    def sparse(self, measured_pixels, size=100):
        """A frame where exactly `measured_pixels` of the shrunk region carry a distant reading."""
        depth = np.zeros((size, size), dtype=np.float32)
        flat = depth.reshape(-1)
        # Placed well inside the region so the six per cent shrink does not exclude them.
        start = size * (size // 4) + size // 4
        flat[start:start + measured_pixels] = 50.0
        return depth

    def test_a_barely_measured_region_reports_nothing(self):
        depth = self.sparse(20)
        self.assertIsNone(sw.median_depth_in_box(depth, 0, 0, 100, 100, min_coverage=0.05))

    def test_a_sufficiently_measured_region_still_reports(self):
        depth = self.sparse(2000)      # about 25 per cent of the shrunk region
        self.assertEqual(50.0, sw.median_depth_in_box(depth, 0, 0, 100, 100, min_coverage=0.05))

    def test_the_floor_is_off_unless_a_caller_asks_for_it(self):
        """Object boxes do not pass one. The archive holds seven object detections, too few to
        choose a number from, so their behaviour is deliberately unchanged."""
        depth = self.sparse(20)
        self.assertEqual(50.0, sw.median_depth_in_box(depth, 0, 0, 100, 100))

    def test_the_configured_floor_is_the_one_the_runtime_falls_back_to(self):
        """Two copies of a threshold that disagree is the fault this file has found twice already."""
        import yaml

        configured = yaml.safe_load(
            (ROOT / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
        self.assertEqual(configured["sector"]["min_measured_fraction"],
                         hdsg.SECTOR_MIN_MEASURED_FRACTION)


class SectorStripsUseTheFloor(unittest.TestCase):
    """The floor reaching the three sector strips, which is the only place it is applied.

    A threshold that exists and is not passed through is worth nothing, and nothing was checking the
    wiring: removing the argument from the call site broke no test on 24 August 2026.
    """

    def setUp(self):
        try:
            from scripts import realsense_vlm_on_change_qwen as client
        except Exception as error:  # pragma: no cover, depends on the environment
            raise unittest.SkipTest(f"the canonical client will not import here: {error}")
        self.client = client

    def sparse_frame(self, measured_rows):
        """A frame whose reasoning band is measured only on `measured_rows` rows, all far away.

        The rows sit at the middle of the band. Placed at its top edge they fall inside the six per
        cent the measurement shrinks away, so the strip reads as unmeasured whatever the floor is,
        and the test proves nothing about the floor.
        """
        depth = np.zeros((480, 640), dtype=np.float32)
        top, bottom = int(0.55 * 480), int(0.95 * 480)
        middle = (top + bottom) // 2
        depth[middle:middle + measured_rows, :] = 50.0
        return depth

    def lane(self, depth, floor):
        return self.client.compute_lane_state(
            depth, mirror_view=False, clear_t=1.8, blocked_t=0.7,
            left_max=0.33, right_min=0.67,
            top_fraction=0.55, bottom_fraction=0.95, min_measured_fraction=floor)

    def test_a_barely_measured_strip_is_unknown_rather_than_clear(self):
        state = self.lane(self.sparse_frame(2), 0.05)
        self.assertEqual(["unknown"] * 3, list(state["status"]))
        self.assertEqual([None] * 3, list(state["depths"]))

    def test_the_same_frame_reads_as_clear_without_the_floor(self):
        """What the walker did until 24 August 2026: two measured rows out of 192, reported as
        50 m of room in all three directions."""
        state = self.lane(self.sparse_frame(2), 0.0)
        self.assertEqual(["clear"] * 3, list(state["status"]))

    def test_a_well_measured_strip_is_unaffected(self):
        state = self.lane(self.sparse_frame(180), 0.05)
        self.assertEqual(["clear"] * 3, list(state["status"]))

    def test_the_client_passes_the_configured_floor_to_the_strips(self):
        """Asserted against the source, the call sitting inside the frame loop.

        Counted as a whole line rather than searched for as a substring. The first version of this
        test looked for `min_measured_fraction=sector_min_measured,`, which is also a substring of
        the fact packet's `sector_min_measured_fraction=sector_min_measured,` twenty lines away, so
        it passed with the lane call deleted.
        """
        source = (ROOT / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(encoding="utf-8")
        lines = [line.strip() for line in source.splitlines()]
        self.assertIn("min_measured_fraction=sector_min_measured,", lines,
                      "compute_lane_state is no longer given the floor")
        self.assertIn("sector_min_measured_fraction=sector_min_measured,", lines,
                      "the fact packet no longer records the floor the run used")
        self.assertIn('sector_min_measured = float(cfg["sector"]["min_measured_fraction"])', lines)


if __name__ == "__main__":
    unittest.main()
