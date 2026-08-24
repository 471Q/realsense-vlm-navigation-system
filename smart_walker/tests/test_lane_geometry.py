"""Where each sector clearance was measured, and what is allowed to read it.

`compute_lane_state` returns the three clearances and the region each was taken from. Both displays
draw that region. Until 25 August 2026 they recomputed it instead, hardcoding the row band as 0.55
and 0.95 and dividing the width into exact thirds, so the band drawn was neither the configured one
nor the one the bearing fractions imply, and under `--mirror_view` each label sat over the third the
reading did not come from.

The function's docstring had claimed for some time that the displays read this result. They did not.
"""

from __future__ import annotations

import re
import unittest

import numpy as np
import yaml

import support
from support import ROOT  # noqa: F401
from scripts import realsense_vlm_on_change_qwen as client  # noqa: E402


WIDTH, HEIGHT = 640, 480
LEFT_MAX, RIGHT_MIN = 0.33, 0.66
TOP, BOTTOM = 0.55, 0.95


def three_lane_frame(near_left=3.0, centre=1.5, near_right=0.5):
    """A frame whose three thirds carry three different distances, measured across the whole band."""
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    y1, y2 = int(TOP * HEIGHT), int(BOTTOM * HEIGHT)
    depth[y1:y2, 0:212] = near_left
    depth[y1:y2, 212:423] = centre
    depth[y1:y2, 423:WIDTH] = near_right
    return depth


def lane(depth, mirror=False, floor=0.05):
    return client.compute_lane_state(
        depth, mirror_view=mirror, clear_t=1.8, blocked_t=0.7,
        left_max=LEFT_MAX, right_min=RIGHT_MIN,
        top_fraction=TOP, bottom_fraction=BOTTOM, min_measured_fraction=floor)


class TheGeometryIsReturnedWithTheReadings(unittest.TestCase):

    def test_each_column_range_is_the_one_its_clearance_was_measured_from(self):
        state = lane(three_lane_frame())
        self.assertEqual([3.0, 1.5, 0.5], list(state["depths"]))
        self.assertEqual([(0, 212), (212, 423), (423, 640)], list(state["cols"]))

    def test_the_columns_are_derived_from_the_bearing_fractions_not_from_thirds(self):
        """The fault the docstring describes, drawn rather than measured. Exact thirds put the
        boundaries at 213 and 426; the predicate `bearing_from_bbox` applies puts them at 212 and
        423, so two columns of 640 belonged to a different sector than the objects standing in
        them."""
        state = lane(three_lane_frame())
        thirds = [(int(i * WIDTH / 3), int((i + 1) * WIDTH / 3)) for i in range(3)]
        self.assertNotEqual(thirds, list(state["cols"]))

    def test_the_rows_are_the_configured_band(self):
        state = lane(three_lane_frame())
        self.assertEqual((int(TOP * HEIGHT), int(BOTTOM * HEIGHT)), state["rows"])

    def test_the_configured_band_is_the_one_the_client_passes(self):
        """A retune of `pipeline.yaml` that the displays do not follow is the whole point of the
        change, so the fractions used here are checked against the file."""
        cfg = yaml.safe_load((support.CONFIG / "pipeline.yaml").read_text(encoding="utf-8"))
        self.assertEqual(TOP, cfg["sector"]["band_top_fraction"])
        self.assertEqual(BOTTOM, cfg["sector"]["band_bottom_fraction"])


class MirroringMovesTheGeometryWithTheReading(unittest.TestCase):
    """`--mirror_view` does not flip the picture. It relabels left and right so everything
    downstream speaks in the person's frame, which is why the region has to travel with the value.
    """

    def test_a_mirrored_left_carries_the_columns_it_was_measured_from(self):
        state = lane(three_lane_frame(), mirror=True)
        self.assertEqual([0.5, 1.5, 3.0], list(state["depths"]))
        self.assertEqual([(423, 640), (212, 423), (0, 212)], list(state["cols"]))

    def test_every_reading_still_matches_its_own_region(self):
        """The property the displays depend on, asserted directly: the value shown over a region is
        the value measured from it, mirrored or not."""
        depth = three_lane_frame()
        for mirror in (False, True):
            state = lane(depth, mirror=mirror)
            for value, (x1, x2) in zip(state["depths"], state["cols"]):
                with self.subTest(mirror=mirror, columns=(x1, x2)):
                    measured = float(np.median(depth[int(TOP * HEIGHT) + 5, x1 + 5:x2 - 5]))
                    self.assertEqual(measured, value)


class NothingUnreadIsReturned(unittest.TestCase):
    """Seven fields were returned and read by nothing.

    `advisory` was a second scene-level SAFE, CAUTION or STOP standing beside
    `compute_baseline_risk`, and `auto_suggest` a seven-way movement suggestion, one branch of which
    proposed reversing, decided by six branches and consulted by nobody. Two decision policies in
    one file is the fault removed from the perception layer on 24 August 2026.
    """

    def test_only_what_is_read_is_returned(self):
        self.assertEqual({"rows", "cols", "depths", "status"}, set(lane(three_lane_frame())))


class TheDisplaysReadTheGeometry(unittest.TestCase):
    """Asserted against the source of both sinks, neither of which can be driven without a camera.

    Both are checked for the absence of the two hardcoded numbers as well as for the presence of the
    read, because a display that reads the geometry and then draws its own would pass the second
    test alone.
    """

    def setUp(self):
        self.client_source = (ROOT / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(
            encoding="utf-8")
        self.browser_source = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

    def code_lines(self, source, comment_markers):
        """The source with its comment lines removed, so a comment recording the old numbers is not
        read as the old numbers."""
        lines = []
        for line in source.splitlines():
            stripped = line.strip()
            if any(stripped.startswith(marker) for marker in comment_markers):
                continue
            lines.append(line)
        return "\n".join(lines)

    def test_the_opencv_overlay_reads_the_measured_band(self):
        code = self.code_lines(self.client_source, ("#",))
        self.assertIn("(y1, y2), band_cols = latest_lane_geometry", code)
        self.assertIn("x1, x2 = band_cols[index]", code)
        self.assertNotIn("int(0.55 * height)", code)
        self.assertNotIn("int(index * width / 3)", code)

    def test_the_browser_reads_the_measured_band(self):
        code = self.code_lines(self.browser_source, ("//", "*", "/*"))
        self.assertIn("const [y1, y2] = geometry.rows;", code)
        self.assertIn("const [x1, x2] = geometry.cols[i];", code)
        self.assertNotIn("0.55 * h", code)
        self.assertNotIn("i * w / 3", code)

    def test_the_client_publishes_the_band_to_the_browser(self):
        code = self.code_lines(self.client_source, ("#",))
        self.assertIn('"sector_bands"', code)
        self.assertIn("latest_lane_geometry = (", code)

    def test_the_published_name_does_not_collide_with_the_fact_packet_field(self):
        """`sector_geometry` is a Fact Packet field of a different shape, carrying the band
        fractions and the coverage floor. Two records using one name for two things is how a reader
        comes to believe a display is showing what the packet recorded."""
        self.assertNotIn('"sector_geometry"', self.client_source.split("MEASUREMENTS")[0])


class AFaultIsNotReportedAsABlindCamera(unittest.TestCase):
    """A returned None means the camera gave no depth, which becomes three unknown sectors and a
    stop. A bare `except Exception` made every defect in this function indistinguishable from that,
    and an unmeasurable scene is a real state: 114 of the 1,480 frames of the archived run.
    """

    def test_a_malformed_frame_is_still_handled(self):
        self.assertIsNone(lane(np.zeros((2,), dtype=np.float32)))

    def test_a_missing_frame_is_still_handled(self):
        self.assertIsNone(lane(None))

    def test_the_handler_names_the_failures_it_expects(self):
        source = (ROOT / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(encoding="utf-8")
        body = source.split("def compute_lane_state", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("except (AttributeError, IndexError, TypeError, ValueError)", body)
        self.assertNotIn("except Exception:", body)


class TheCoverageFloorCannotBeOmitted(unittest.TestCase):
    """It defaulted to 0.0, which is no floor at all.

    This file states the opposite principle for `detector_classes`: a property that holds by
    construction must not be able to switch itself off because a caller left an argument out.
    """

    def test_the_floor_is_a_required_keyword(self):
        import inspect

        parameter = inspect.signature(client.compute_lane_state).parameters["min_measured_fraction"]
        self.assertIs(inspect.Parameter.empty, parameter.default)
        self.assertIs(inspect.Parameter.KEYWORD_ONLY, parameter.kind)

    def test_calling_without_it_fails_rather_than_measuring_without_it(self):
        with self.assertRaises(TypeError):
            client.compute_lane_state(three_lane_frame(), mirror_view=False, clear_t=1.8,
                                      blocked_t=0.7, left_max=LEFT_MAX, right_min=RIGHT_MIN)


if __name__ == "__main__":
    unittest.main()
