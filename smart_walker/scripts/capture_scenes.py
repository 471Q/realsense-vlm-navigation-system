"""Captures labelled RGB-D scenes for offline perception work.

Written to answer one question that cannot be answered from a desk: whether a stair, a drop-off or
a doorway can be found in this walker's depth and colour data at all. Section 10.3 of
`HDSG_VERIFIED_GENERATION_POLICY.md` leaves the hazard detection method open between a geometric
floor analysis and an open-vocabulary detector, and neither can be assessed without recordings of
the things they are supposed to find. The repository contained none.

    python scripts/capture_scenes.py --out captures/stairs_survey

    SPACE   capture the current frame under the current label
    N / P   next or previous label
    Q       finish

Deliberately separate from `realsense_vlm_on_change_qwen.py`. That script records only under
`--evaluate`, which starts the whole guidance loop and needs the language model server running.
Capturing a staircase requires neither, and coupling the two would mean a failed model server
prevented data collection.

The output format is `ObservationRecorder`'s, so anything already able to read an evaluation
recording can read these: an 8-bit lossless PNG for colour, a 16-bit single channel PNG in
millimetres for depth, and one manifest line per observation. A second file, `labels.jsonl`,
records what each capture was aimed at, because a depth frame of a staircase and a depth frame of a
flat corridor are not distinguishable without someone saying which is which.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    from scripts.hdsg_recording import ObservationRecorder
except ImportError:  # invoked as a plain script rather than as part of the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hdsg_recording import ObservationRecorder  # type: ignore


# What the hazard bucket names, plus the two controls. A capture of flat floor and of an ordinary
# obstacle is what tells a false positive from a true one, so both are in the list rather than left
# to be remembered later.
LABELS = (
    "stairs_down",
    "stairs_up",
    "drop_off",
    "ramp",
    "threshold",
    "doorway",
    "handrail",
    "clear_floor",
    "obstacle_only",
)

# Ranges worth having for each hazard label. A detector that finds a staircase at one metre and not
# at three is not useful to someone walking towards it, and that distinction only exists in the data
# if the captures were taken at more than one distance.
DISTANCES_M = (1.0, 2.0, 3.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture labelled RGB-D scenes for perception work")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory to write frames, the manifest and the labels into")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def open_camera(args: argparse.Namespace):
    """Starts the camera with depth aligned to colour, and returns the scale depth is reported in."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    return pipeline, rs.align(rs.stream.color), depth_scale


def overlay(frame, label: str, index: int, captured: dict) -> None:
    """Draws the current label and the running counts, so gaps are visible while still on site."""
    lines = [f"[{index + 1}/{len(LABELS)}]  {label}    captured: {captured.get(label, 0)}",
             "SPACE capture     N/P label     Q finish"]
    for row, text in enumerate(lines):
        position = (12, 30 + row * 28)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    summary = "  ".join(f"{name}:{captured.get(name, 0)}" for name in LABELS)
    cv2.putText(frame, summary, (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, summary, (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline, align, depth_scale = open_camera(args)
    recorder = ObservationRecorder(args.out, depth_scale=depth_scale).start()
    labels_path = args.out / "labels.jsonl"
    captured: dict[str, int] = {}
    index = 0
    sequence = 0

    print(f"[capture] depth scale {depth_scale} m per unit")
    print(f"[capture] writing to {args.out}")
    print(f"[capture] suggested distances per hazard label: "
          f"{', '.join(f'{d:.0f} m' for d in DISTANCES_M)}")
    try:
        with labels_path.open("a", encoding="utf-8") as labels_file:
            while True:
                frames = align.process(pipeline.wait_for_frames())
                depth_frame = frames.get_depth_frame()
                colour_frame = frames.get_color_frame()
                if not depth_frame or not colour_frame:
                    continue
                colour = np.asanyarray(colour_frame.get_data())
                depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

                preview = colour.copy()
                overlay(preview, LABELS[index], index, captured)
                cv2.imshow("capture", preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord("n"):
                    index = (index + 1) % len(LABELS)
                elif key == ord("p"):
                    index = (index - 1) % len(LABELS)
                elif key == ord(" "):
                    label = LABELS[index]
                    sequence += 1
                    observation_id = f"{label}_{sequence:04d}"
                    # The frames are copied because the recorder writes on another thread and the
                    # next iteration overwrites the buffers the camera returned.
                    queued = recorder.record(observation_id, colour.copy(), depth_m.copy(),
                                             float(depth_frame.get_timestamp()))
                    captured[label] = captured.get(label, 0) + 1
                    labels_file.write(json.dumps({
                        "observation_id": observation_id,
                        "label": label,
                        "captured_at_utc": datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds").replace("+00:00", "Z"),
                        "queued": queued,
                    }) + "\n")
                    labels_file.flush()
                    print(f"[capture] {observation_id}{'' if queued else '  DROPPED, queue full'}")
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        stats = recorder.stop()
        print(f"[capture] {json.dumps(stats)}")
        for label in LABELS:
            print(f"  {label:16} {captured.get(label, 0)}")
        if not stats.get("complete", False):
            print("[capture] INCOMPLETE: some frames were dropped or failed to write.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
