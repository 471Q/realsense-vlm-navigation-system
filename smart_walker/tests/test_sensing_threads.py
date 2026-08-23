"""The two threads that turn camera frames into measured objects.

Neither needs a camera to be tested. `capture_thread` is given a stand-in for the RealSense pipeline
and `inference_thread` a stand-in for the detector, which is enough to exercise what the threads
themselves decide: which frame is published, and what happens when the sensor fails.
"""

from __future__ import annotations

import threading
import time
import unittest
from queue import Queue, Empty

import numpy as np

import support
from support import ROOT  # noqa: F401
from scripts import realsense_shared_control as sw  # noqa: E402


class FakeFrame:
    def __init__(self, array):
        self.array = array

    def get_data(self):
        return self.array


class FakeFrames:
    def __init__(self, depth_u16, colour):
        self.depth = FakeFrame(depth_u16)
        self.colour = FakeFrame(colour)

    def get_depth_frame(self):
        return self.depth

    def get_color_frame(self):
        return self.colour


class FakeAlign:
    def process(self, frames):
        return frames


class FakePipe:
    """Hands out prepared framesets, then does whatever `after` says.

    `after` is either "raise", standing for the sensor failing mid-run, or "block", standing for a
    camera that has simply gone quiet.
    """

    def __init__(self, framesets, after="block"):
        self.framesets = list(framesets)
        self.after = after
        self.calls = 0

    def wait_for_frames(self):
        self.calls += 1
        if self.framesets:
            return self.framesets.pop(0)
        if self.after == "raise":
            raise RuntimeError("the camera stopped responding")
        time.sleep(0.005)
        return FakeFrames(np.zeros((4, 4), dtype=np.uint16), np.zeros((4, 4, 3), dtype=np.uint8))


def frameset(depth_mm_value, size=4):
    return FakeFrames(np.full((size, size), depth_mm_value, dtype=np.uint16),
                      np.zeros((size, size, 3), dtype=np.uint8))


def run_briefly(target, args, seconds=0.3):
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    return thread


class CaptureThreadFailure(unittest.TestCase):
    """A dead sensing thread must end the run rather than leave it apparently running.

    Until 24 August 2026 the thread printed a traceback and returned without setting the stop event.
    The main loop then went on finding its queue empty and looping: the window stayed up showing the
    last frame and the keys still answered, while nothing was recorded. With the web sink the
    traceback was not on screen at all.
    """

    def test_a_camera_failure_sets_the_stop_event(self):
        stop_evt = threading.Event()
        run_briefly(sw.capture_thread,
                    (FakePipe([frameset(1500)], after="raise"), FakeAlign(), 0.001,
                     Queue(maxsize=1), stop_evt))
        self.assertTrue(stop_evt.is_set(),
                        "the run continues with no frames and no sign of the failure")

    def test_the_thread_stops_when_the_run_does(self):
        stop_evt = threading.Event()
        stop_evt.set()
        pipe = FakePipe([frameset(1500)])
        thread = run_briefly(sw.capture_thread,
                             (pipe, FakeAlign(), 0.001, Queue(maxsize=1), stop_evt))
        self.assertFalse(thread.is_alive())
        self.assertEqual(0, pipe.calls)


class CaptureThreadOutput(unittest.TestCase):

    def publish(self, depth_mm_value):
        out_q = Queue(maxsize=1)
        stop_evt = threading.Event()
        pipe = FakePipe([frameset(depth_mm_value)], after="raise")
        run_briefly(sw.capture_thread, (pipe, FakeAlign(), 0.001, out_q, stop_evt))
        return out_q.get_nowait()

    def test_the_published_depth_is_in_metres(self):
        """The scale is the camera's, 0.001 here, so 1500 millimetres is 1.5 m."""
        self.assertAlmostEqual(1.5, float(self.publish(1500).depth_m[0, 0]), places=4)

    def test_the_sixteen_bit_ceiling_is_published_as_no_reading(self):
        """65535 is the sensor saying it has no reading. Read as 65.535 m until 24 August 2026."""
        self.assertEqual(0.0, float(self.publish(65535).depth_m[0, 0]))

    def test_only_the_newest_frame_is_kept(self):
        """A slow detector makes the picture older, not the lag longer."""
        out_q = Queue(maxsize=1)
        stop_evt = threading.Event()
        pipe = FakePipe([frameset(1000), frameset(2000), frameset(3000)], after="raise")
        run_briefly(sw.capture_thread, (pipe, FakeAlign(), 0.001, out_q, stop_evt))
        self.assertAlmostEqual(3.0, float(out_q.get_nowait().depth_m[0, 0]), places=4)
        with self.assertRaises(Empty):
            out_q.get_nowait()


class FailingModel:
    def track(self, **kwargs):
        raise RuntimeError("the detector failed")

    def predict(self, **kwargs):
        raise RuntimeError("the detector failed")


class InferenceThreadFailure(unittest.TestCase):

    def test_a_detector_failure_sets_the_stop_event(self):
        in_q: Queue = Queue(maxsize=1)
        in_q.put(sw.FramePacket(color=np.zeros((4, 4, 3), dtype=np.uint8),
                                depth_m=np.zeros((4, 4), dtype=np.float32), ts_ms=0))
        stop_evt = threading.Event()
        run_briefly(sw.inference_thread,
                    ({"bearing": {"left_max": 0.33, "right_min": 0.66}}, None, FailingModel(),
                     in_q, Queue(maxsize=1), stop_evt, 640, 0.25, False, None, True,
                     "botsort.yaml"))
        self.assertTrue(stop_evt.is_set())


class FakeTensor:
    """Stands in for the torch tensors the detector returns, which the thread moves off the GPU."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def int(self):
        return FakeTensor(self.array.astype(int))

    def tolist(self):
        return self.array.tolist()


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)
        self.id = None if ids is None else FakeTensor(ids)

    def __len__(self):
        return len(self.xyxy.array)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class DetectingModel:
    def __init__(self, boxes, names):
        self.result = FakeResult(boxes, names)
        self.names = names

    def track(self, **kwargs):
        return [self.result]

    predict = track


class DetectionsBecomeObjects(unittest.TestCase):
    """What the detector's boxes turn into, which is the input every later stage reasons over."""

    def detect(self, labels, ids, depth_value=1.5):
        names = {index: label for index, label in enumerate(labels)}
        boxes = FakeBoxes(xyxy=[[100.0, 100.0, 300.0, 400.0]] * len(labels),
                          conf=[0.9] * len(labels),
                          cls=list(range(len(labels))),
                          ids=ids)
        in_q: Queue = Queue(maxsize=1)
        in_q.put(sw.FramePacket(color=np.zeros((480, 640, 3), dtype=np.uint8),
                                depth_m=np.full((480, 640), depth_value, dtype=np.float32),
                                ts_ms=0))
        out_q: Queue = Queue(maxsize=1)
        mapper = sw.OntologyMapper(support.CONFIG / "ontology.yaml")
        cfg = {"bearing": {"left_max": 0.33, "right_min": 0.66},
               "depth": {"metric_bins_m": {"very_close": [0.0, 0.7], "near": [0.7, 1.5],
                                           "mid": [1.5, 3.0], "far": [3.0, 99.0]}}}
        stop_evt = threading.Event()
        thread = threading.Thread(
            target=sw.inference_thread,
            args=(cfg, mapper, DetectingModel(boxes, names), in_q, out_q, stop_evt,
                  640, 0.25, False, None, True, "botsort.yaml"),
            daemon=True)
        thread.start()
        try:
            return out_q.get(timeout=1.0).objects
        finally:
            stop_evt.set()
            thread.join(timeout=0.5)

    def test_a_tracked_object_carries_its_tracker_number(self):
        self.assertEqual("chair #7", self.detect(["chair"], [7])[0]["display_label"])

    def test_an_untracked_object_carries_its_label_alone(self):
        self.assertEqual("chair", self.detect(["chair"], None)[0]["display_label"])

    def test_a_person_is_labelled_no_differently_from_anything_else(self):
        """A branch treating "person" separately stood here until 24 August 2026 and produced this
        same string for every label and every id, so the file stated a rule about people that did
        nothing. Removed. This records that the behaviour did not change with it."""
        self.assertEqual("person #7", self.detect(["person"], [7])[0]["display_label"])

    def test_the_distance_is_measured_and_banded(self):
        obj = self.detect(["chair"], [7], depth_value=1.5)[0]
        self.assertEqual(1.5, obj["distance_m"])
        self.assertEqual("mid", obj["distance_bin"])
        self.assertEqual("D455F_LOWER_BBOX_MEDIAN", obj["distance_method"])

    def test_an_unmeasurable_object_reports_no_distance_rather_than_a_far_one(self):
        obj = self.detect(["chair"], [7], depth_value=0.0)[0]
        self.assertIsNone(obj["distance_m"])
        self.assertEqual("unknown", obj["distance_bin"])
        self.assertIsNone(obj["distance_method"])

    def test_something_that_cannot_be_an_obstacle_never_becomes_one(self):
        """`not_obstacles` holds one COCO class, tie, which the detector reports on a person's
        chest. Without this it would map to unknown_obstacle, which stops the walker at 0.70 m."""
        self.assertEqual([], self.detect(["tie"], [7]))

    def test_the_dropped_list_is_the_one_the_ontology_declares(self):
        """Asserted so that a change of weights that empties the list is visible here rather than
        only as a walker that stops for a necktie."""
        mapper = sw.OntologyMapper(support.CONFIG / "ontology.yaml")
        self.assertEqual({"tie"}, mapper.not_obstacles)


class TheMotionSensorPathIsGone(unittest.TestCase):
    """Removed on 24 August 2026, having never run.

    The only caller passed `imu_enabled=False` with no state object, and the client enables only the
    colour and depth streams, so the motion stream was never started either. Switching the flag on
    would have started discarding camera frames: after publishing a frame the thread polled for
    further framesets looking for motion readings and dropped every frameset that was not one.
    Whether the walker's own movement is wanted as a fact is section D of the laboratory checklist.
    """

    def test_no_motion_sensor_state_remains(self):
        for name in ("IMUState", "IMU_MAX_DRAIN_PER_LOOP"):
            self.assertFalse(hasattr(sw, name), f"{name} is back")

    def test_the_capture_thread_takes_no_motion_arguments(self):
        import inspect

        self.assertEqual(["pipe", "align", "depth_scale", "out_q", "stop_evt"],
                         list(inspect.signature(sw.capture_thread).parameters))

    def test_the_client_enables_only_the_colour_and_depth_streams(self):
        """Asserted against the source. Starting a motion stream nothing reads would restore the
        backlog the removed drain existed to manage."""
        source = (ROOT / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(encoding="utf-8")
        self.assertEqual(2, source.count("rs_cfg.enable_stream("))
        for absent in ("rs.stream.gyro", "rs.stream.accel"):
            self.assertNotIn(absent, source)


if __name__ == "__main__":
    unittest.main()
