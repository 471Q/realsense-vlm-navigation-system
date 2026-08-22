# scripts/realsense_per_frame_facts.py
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from ultralytics import YOLO

# ----- paths (script-relative) -----
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_DIR = Path(os.environ.get("SMART_WALKER_CONFIG_DIR", ROOT / "config"))
PIPELINE_CFG = CONFIG_DIR / "pipeline.yaml"


def load_cfg():
    with PIPELINE_CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bearing_from_bbox(box, img_w, bearing_cfg):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    frac = cx / float(img_w)
    if frac < bearing_cfg["left_max"]:
        return "left"
    if frac > bearing_cfg["right_min"]:
        return "right"
    return "centre"


def median_depth_in_box(depth_m: np.ndarray, x1, y1, x2, y2):
    H, W = depth_m.shape[:2]
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(W, int(np.ceil(x2)))
    y2 = min(H, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_m[y1:y2, x1:x2]
    vals = patch[np.nonzero(patch)]  # ignore zeros (no measurement)
    if vals.size == 0:
        return None
    return float(np.median(vals))


def distance_bin_from_m(d_m, bins_cfg):
    if d_m is None or np.isnan(d_m):
        return "unknown"
    for name, (lo, hi) in bins_cfg.items():
        if lo <= d_m < hi:
            return name
    return "far"


def draw_overlay(color, objects):
    vis = color.copy()
    for o in objects:
        x1, y1, x2, y2 = map(int, o["bbox_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        dist_txt = "?" if o["distance_m"] is None else f"{o['distance_m']:.2f}m"
        label = f"{o['raw_label']} {o['conf']:.2f} {o['bearing']} {dist_txt} ({o['distance_bin']})"
        cv2.putText(vis, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-oiv7.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--print_json_every", type=int, default=15)
    args = ap.parse_args()

    cfg = load_cfg()
    print(f"[smart_walker] Using config: {PIPELINE_CFG}")

    # ---- RealSense: config + pipeline ----
    pipe = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.color, args.width,
                       args.height, rs.format.bgr8, args.fps)
    conf.enable_stream(rs.stream.depth, args.width,
                       args.height, rs.format.z16, args.fps)
    profile = pipe.start(conf)

    # Depth units → metres (device-specific)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    # RealSense docs: mapping depth units→meters
    print(f"[smart_walker] depth_scale = {depth_scale:.6f} m per unit")

    # Align depth to color (SDK-recommended)
    align = rs.align(rs.stream.color)  # aligns depth to the color stream

    model = YOLO(args.model)
    print("Press 'q' to quit. Press 'p' to print current facts_json.")
    frame_idx, last_json_print = 0, -1

    try:
        while True:
            frames = pipe.wait_for_frames()  # blocks until a new frameset arrives
            aligned = align.process(frames)  # depth aligned to color
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue

            # to NumPy
            color_np = np.asanyarray(color.get_data())   # BGR8
            depth_u16 = np.asanyarray(depth.get_data())  # z16 units
            depth_m = depth_u16.astype(np.float32) * depth_scale

            # run detector on color frame (NumPy is supported)
            res = model.predict(source=color_np, imgsz=args.imgsz,
                                conf=args.conf, verbose=False)[0]
            names = res.names
            objs = []
            H, W = depth_m.shape[:2]
            for i, b in enumerate(res.boxes):
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                raw_label = names[cls]
                bearing = bearing_from_bbox(
                    (x1, y1, x2, y2), W, cfg["bearing"])
                d_m = median_depth_in_box(depth_m, x1, y1, x2, y2)
                d_bin = distance_bin_from_m(d_m, cfg["depth"]["metric_bins_m"])
                objs.append({
                    "id": i,
                    "raw_label": raw_label,
                    "canonical_class": None,   # will be filled by mapper later
                    "ontology_class": None,    # will be filled by mapper later
                    "conf": round(conf, 3),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "distance_m": None if d_m is None else round(d_m, 2),
                    "distance_bin": d_bin,
                    "bearing": bearing
                })

            facts = {
                "frame_id": frame_idx,
                "timestamp_ms": int(time.time() * 1000),
                "caption": "",
                "risk": "",
                "objects": objs,
                "free_space": {"corridor_min_width_m": None, "nearest_obstacle_m": None},
                "hazards": [],
                "uncertainty": {"depth_std": None, "low_light": False},
                "source_depth": "rgbd",
                "explain": {"rules_fired": [], "min_distance_m": None, "class_counts": {}}
            }

            vis = draw_overlay(color_np, objs)
            cv2.imshow("smart_walker — RealSense per-frame facts", vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('p') or (args.print_json_every > 0 and frame_idx % args.print_json_every == 0 and frame_idx != last_json_print):
                print(json.dumps(facts, indent=2))
                last_json_print = frame_idx

            frame_idx += 1

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
