# scripts/rgbd_facts_from_pair.py
from __future__ import annotations
import argparse
import json
import time
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_DIR = Path(os.environ.get("SMART_WALKER_CONFIG_DIR", ROOT / "config"))
PIPELINE_CFG = CONFIG_DIR / "pipeline.yaml"

PRESETS = {
    "meters": 1.0,          # depth PNG already in metres
    "realsense_mm": 1/1000,  # 16-bit millimetres -> metres (our bag export)
    "tum_5000": 1/5000,     # TUM depth encoding
    "units": None           # raw z16 * depth_scale (pass --depth_scale)
}


def load_cfg():
    if not PIPELINE_CFG.exists():
        raise FileNotFoundError(f"Missing config: {PIPELINE_CFG}")
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


def distance_bin_from_m(d_m, bins_cfg: dict[str, list[float]]):
    if d_m is None or np.isnan(d_m):
        return "unknown"
    for name, (lo, hi) in bins_cfg.items():
        if lo <= d_m < hi:
            return name
    return "far"


def median_depth_in_box(depth_m: np.ndarray, x1, y1, x2, y2):
    H, W = depth_m.shape[:2]
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(W, int(np.ceil(x2)))
    y2 = min(H, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_m[y1:y2, x1:x2]
    # RealSense / RGB-D datasets: 0 means "no valid measurement" -> ignore zeros
    vals = patch[np.nonzero(patch)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def draw_overlay(rgb, objects):
    out = rgb.copy()
    for o in objects:
        x1, y1, x2, y2 = map(int, o["bbox_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        dist_txt = "?" if o["distance_m"] is None else f"{o['distance_m']:.2f}m"
        label = f"{o['raw_label']} {o['conf']:.2f} {o['bearing']} {dist_txt} ({o['distance_bin']})"
        cv2.putText(out, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True,
                    help="Path to color image (aligned with depth)")
    ap.add_argument("--depth", required=True, help="Path to 16-bit depth PNG")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="realsense_mm",
                    help="How to convert depth PNG to metres")
    ap.add_argument("--depth_scale", type=float, default=None,
                    help="Required if --preset=units (metres per unit)")
    ap.add_argument("--model", default="yolov8n-oiv7.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    cfg = load_cfg()
    scale = PRESETS[args.preset]
    if args.preset == "units":
        if args.depth_scale is None:
            raise ValueError(
                "With --preset=units you must pass --depth_scale (metres per unit).")
        scale = args.depth_scale

    # ---- Load RGB ----
    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Failed to read RGB: {args.rgb}")

    # ---- Load depth robustly (preserve bit depth; handle multi-channel) ----
    # keep original depth
    depth_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"Failed to read depth: {args.depth}")
    # If depth has channels (e.g., mistakenly saved as BGR), convert to 1-channel
    if depth_raw.ndim == 3 and depth_raw.shape[2] in (3, 4):
        # Convert to grayscale; OpenCV supports 16-bit here too
        code = cv2.COLOR_BGR2GRAY if depth_raw.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        # still uint16 if input was uint16
        depth_raw = cv2.cvtColor(depth_raw, code)
    if depth_raw.dtype != np.uint16:
        # Try to re-load with explicit ANYDEPTH to preserve 16-bit if supported
        tmp = cv2.imread(args.depth, cv2.IMREAD_ANYDEPTH)
        if tmp is not None and tmp.dtype == np.uint16 and tmp.ndim == 2:
            depth_raw = tmp
        else:
            raise RuntimeError(
                f"Depth PNG must be 16-bit single channel (uint16). Got dtype={depth_raw.dtype}, "
                f"shape={depth_raw.shape}. Did you point to color.png by mistake?"
            )

    H, W = depth_raw.shape[:2]
    if rgb.shape[0] != H or rgb.shape[1] != W:
        raise RuntimeError(f"RGB ({rgb.shape[1]}x{rgb.shape[0]}) and depth ({W}x{H}) sizes differ; "
                           "they must be aligned 1:1.")

    # Convert to metres
    depth_m = depth_raw.astype(np.float32) * float(scale)

    # ---- Run detector ----
    model = YOLO(args.model)
    res = model.predict(source=rgb, imgsz=args.imgsz,
                        conf=args.conf, verbose=False)[0]
    names = res.names

    objs = []
    for i, b in enumerate(res.boxes):
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        raw_label = names[cls]
        bearing = bearing_from_bbox((x1, y1, x2, y2), W, cfg["bearing"])
        d_m = median_depth_in_box(depth_m, x1, y1, x2, y2)
        d_bin = distance_bin_from_m(d_m, cfg["depth"]["metric_bins_m"])
        objs.append({
            "id": i,
            "raw_label": raw_label,
            "canonical_class": None,
            "ontology_class": None,
            "conf": round(conf, 3),
            "bbox_xyxy": [x1, y1, x2, y2],
            "distance_m": None if d_m is None else round(d_m, 2),
            "distance_bin": d_bin,
            "bearing": bearing
        })

    facts = {
        "frame_id": 0,
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

    vis = draw_overlay(rgb, objs)
    cv2.imshow("smart_walker — RGBD facts (per pair)", vis)
    print(json.dumps(facts, indent=2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
