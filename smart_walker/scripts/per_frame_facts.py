# scripts/per_frame_facts.py
from __future__ import annotations
import argparse
import time
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ---------- Resolve project paths (robust, script-relative) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent                       # .../smart_walker
DEFAULT_CONFIG_DIR = ROOT / "config"

# Allow override via env var if you want to point elsewhere
CONFIG_DIR = Path(os.environ.get(
    "SMART_WALKER_CONFIG_DIR", DEFAULT_CONFIG_DIR))
PIPELINE_CFG_PATH = CONFIG_DIR / "pipeline.yaml"


def load_cfg():
    if not PIPELINE_CFG_PATH.exists():
        raise FileNotFoundError(
            f"Could not find pipeline config at:\n  {PIPELINE_CFG_PATH}\n"
            f"Expected folder structure:\n  {ROOT}\n    └─ config/pipeline.yaml\n\n"
            "Tip: ensure you run the script from anywhere; paths are script-relative."
        )
    with PIPELINE_CFG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bearing_from_bbox(box, img_w, cfg):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    frac = cx / float(img_w)
    if frac < cfg["bearing"]["left_max"]:
        return "left"
    if frac > cfg["bearing"]["right_min"]:
        return "right"
    return "centre"


def to_objects(yolo_result, img_w, cfg):
    objs = []
    names = yolo_result.names
    for i, b in enumerate(yolo_result.boxes):
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        raw_label = names[cls]
        bearing = bearing_from_bbox((x1, y1, x2, y2), img_w, cfg)
        objs.append({
            "id": i,
            "raw_label": raw_label,
            "canonical_class": None,      # filled later by ontology mapper
            "ontology_class": None,       # filled later by ontology mapper
            "conf": round(conf, 3),
            "bbox_xyxy": [x1, y1, x2, y2],
            "distance_m": None,           # depth later
            "distance_bin": "unknown",    # depth later
            "bearing": bearing
        })
    return objs


def draw_overlay(frame, objs, caption=""):
    for o in objs:
        x1, y1, x2, y2 = map(int, o["bbox_xyxy"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{o["raw_label"]} {o["conf"]:.2f} {o["bearing"]}'
        cv2.putText(frame, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if caption:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(frame, caption, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="0 for webcam, or path to a video/image")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Ultralytics model weight")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--print_json_every", type=int, default=15)
    args = ap.parse_args()

    cfg = load_cfg()
    print(f"[smart_walker] Using pipeline config: {PIPELINE_CFG_PATH}")

    # Open source (image or video/webcam)
    src = 0 if args.source == "0" else args.source
    is_image = isinstance(src, str) and str(
        src).lower().endswith((".jpg", ".jpeg", ".png"))
    if is_image:
        frame = cv2.imread(src)
        if frame is None:
            raise RuntimeError(f"Could not read image {src}")
        cap = None
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source {args.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    model = YOLO(args.model)

    print("Press 'q' to quit. Press 'p' to print current facts_json.")
    frame_idx, last_json_print = 0, -1

    def process_frame(frame, frame_idx):
        h, w = frame.shape[:2]
        res = model.predict(source=frame, imgsz=args.imgsz,
                            conf=args.conf, verbose=False)[0]
        objs = to_objects(res, w, cfg)
        facts = {
            "frame_id": frame_idx,
            "timestamp_ms": int(time.time() * 1000),
            "caption": "",
            "risk": "",
            "objects": objs,
            "free_space": {"corridor_min_width_m": None, "nearest_obstacle_m": None},
            "hazards": [],
            "uncertainty": {"depth_std": None, "low_light": False},
            "source_depth": "pending",  # "rgbd" or "monocular" later
            "explain": {"rules_fired": [], "min_distance_m": None, "class_counts": {}}
        }
        return facts

    if is_image:
        facts = process_frame(frame, 0)
        vis = draw_overlay(
            frame.copy(), facts["objects"], caption="facts_json printed to console")
        print(json.dumps(facts, indent=2))
        cv2.imshow("smart_walker — per-frame facts", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of stream.")
            break
        facts = process_frame(frame, frame_idx)
        vis = draw_overlay(frame.copy(), facts["objects"])
        cv2.imshow("smart_walker — per-frame facts", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('p') or (args.print_json_every > 0 and frame_idx % args.print_json_every == 0 and frame_idx != last_json_print):
            print(json.dumps(facts, indent=2))
            last_json_print = frame_idx

        frame_idx += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
