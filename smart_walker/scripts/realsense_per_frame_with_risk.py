from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from ultralytics import YOLO
from rapidfuzz import process, fuzz

# ---------------- paths & config ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_DIR = Path(os.environ.get("SMART_WALKER_CONFIG_DIR", ROOT / "config"))
PIPELINE_CFG = CONFIG_DIR / "pipeline.yaml"
ONTOLOGY_CFG = CONFIG_DIR / "ontology.yaml"


def load_yaml(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------- tiny ontology mapper (inline) ----------------


@dataclass
class Mapped:
    canonical_class: str | None
    ontology_class: str


class OntologyMapper:
    def __init__(self, ontology_path: Path, sim_threshold: float = 80.0):
        cfg = load_yaml(ontology_path)
        self.sim_threshold = sim_threshold
        self.ontology_buckets = {}   # canonical -> ontology name
        self.prompts = []            # strings to search
        self.prompt2canon = {}       # string -> canonical
        self.synonyms = {k.lower(): v.lower()
                         for k, v in cfg.get("synonyms_to_canonical", {}).items()}

        for bucket in cfg["ontology"]:
            ont = bucket["name"]
            for c in bucket.get("canonical", []):
                c_low = c.lower()
                self.ontology_buckets[c_low] = ont
                self.prompts.append(c_low)
                self.prompt2canon[c_low] = c_low
            for p in bucket.get("prompts", []):
                p_low = p.lower()
                self.prompts.append(p_low)
                if bucket.get("canonical"):
                    self.prompt2canon[p_low] = bucket["canonical"][0].lower()

        self.hazard_set = set([c.lower() for b in cfg["ontology"]
                              if b["name"] == "hazard" for c in b.get("canonical", [])])

    def map_label(self, raw_label: str) -> Mapped:
        if not raw_label:
            return Mapped(None, "unknown_obstacle")
        s = raw_label.strip().lower()

        # synonym
        if s in self.synonyms:
            canon = self.synonyms[s]
            return Mapped(canon, self.ontology_buckets.get(canon, "unknown_obstacle"))

        # direct canonical
        if s in self.ontology_buckets:
            return Mapped(s, self.ontology_buckets[s])

        # fuzzy against prompts/canonicals
        match = process.extractOne(s, self.prompts, scorer=fuzz.WRatio)
        if match and match[1] >= self.sim_threshold:
            canon = self.prompt2canon.get(match[0], None)
            if canon:
                return Mapped(canon, self.ontology_buckets.get(canon, "unknown_obstacle"))

        return Mapped(None, "unknown_obstacle")

# ---------------- helpers ----------------


def bearing_from_bbox(box, img_w, bearing_cfg):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    frac = cx / float(img_w)
    if frac < bearing_cfg["left_max"]:
        return "left"
    if frac > bearing_cfg["right_min"]:
        return "right"
    return "centre"


def distance_bin_from_m(d_m, bins_cfg):
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
    vals = patch[np.nonzero(patch)]  # ignore zeros (no measurement)
    if vals.size == 0:
        return None
    return float(np.median(vals))


def draw_overlay(img, objects, risk, caption):
    vis = img.copy()
    color = (0, 255, 0) if risk == "safe" else (
        (0, 255, 255) if risk == "caution" else (0, 0, 255))
    # risk banner
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, f"RISK: {risk.upper()}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    # caption line
    if caption:
        cv2.putText(vis, caption[:80], (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    # boxes
    for o in objects:
        x1, y1, x2, y2 = map(int, o["bbox_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        dist_txt = "?" if o["distance_m"] is None else f"{o['distance_m']:.2f}m"
        role = o.get("ontology_class") or "unknown"
        canon = o.get("canonical_class") or o["raw_label"]
        label = f"{canon} [{role}] {o['conf']:.2f} {o['bearing']} {dist_txt} ({o['distance_bin']})"
        cv2.putText(vis, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return vis

# ---- baseline risk rules (inline) ----


def _nearest_obstacle_m(objs):
    ds = [o.get("distance_m")
          for o in objs if isinstance(o.get("distance_m"), (int, float))]
    return min(ds) if ds else None


def _multi_near_count(objs):
    return sum(1 for o in objs if o.get("distance_bin") in ("very_close", "near"))


def compute_baseline_risk(facts: dict, cfg: dict) -> dict:
    """Return {'risk': 'safe|caution|stop', 'rules_fired': [..]} based on config thresholds."""
    rf = []
    rules = cfg["risk_rules_baseline"]
    objs = facts.get("objects", [])
    nearest_m = _nearest_obstacle_m(objs)
    facts["explain"]["min_distance_m"] = nearest_m

    # STOP rules
    stop = rules["stop"]
    if nearest_m is not None and nearest_m < stop["nearest_obstacle_m_lt"]:
        rf.append("stop:nearest_obstacle")
    cmw = facts.get("free_space", {}).get("corridor_min_width_m")
    if cmw is not None and cmw < stop["corridor_min_width_m_lt"]:
        rf.append("stop:corridor_narrow")
    hazards = facts.get("hazards", [])
    if nearest_m is not None and hazards and nearest_m <= stop["hazard_within_m_lte"]:
        rf.append("stop:hazard_nearby")
    if rf:
        return {"risk": "stop", "rules_fired": rf}

    # CAUTION rules
    caut = rules["caution"]
    if nearest_m is not None and nearest_m < caut["nearest_obstacle_m_lt"]:
        rf.append("caution:nearest_obstacle")
    if _multi_near_count(objs) >= caut["multi_near_objects_count_gte"]:
        rf.append("caution:multi_near")
    if caut.get("high_uncertainty") and facts.get("uncertainty", {}).get("low_light"):
        rf.append("caution:uncertainty")
    if rf:
        return {"risk": "caution", "rules_fired": rf}

    return {"risk": "safe", "rules_fired": []}

# ---- tiny captioner (inline) ----


def _nearest_tuple(objs):
    ds = [(o["id"], o.get("raw_label"), o.get("distance_m"), o.get("bearing"))
          for o in objs if isinstance(o.get("distance_m"), (int, float))]
    if not ds:
        return None
    return min(ds, key=lambda x: x[2])


def make_caption(facts: dict) -> str:
    risk = facts.get("risk", "safe")
    objs = facts.get("objects", [])
    haz_ids = set(facts.get("hazards", []))

    if not objs:
        return f"{risk.capitalize()}. No obstacles detected."

    parts = [risk.capitalize() + "."]
    n = _nearest_tuple(objs)
    if n:
        _, lbl, d, bearing = n
        if d is not None:
            parts.append(f"Nearest {lbl} {d:.1f} m {bearing}.")
    hz = [o for o in objs if o["id"] in haz_ids]
    if hz:
        h = min(hz, key=lambda o: (o["distance_m"] if isinstance(
            o.get("distance_m"), (int, float)) else float('inf')))
        dtxt = f"{h['distance_m']:.1f} m " if isinstance(
            h.get("distance_m"), (int, float)) else ""
        parts.append(f"Hazard: {h['raw_label']} {dtxt}{h['bearing']}.")
    near_count = sum(1 for o in objs if o.get(
        "distance_bin") in ("very_close", "near"))
    if near_count >= 2:
        parts.append("Multiple nearby obstacles.")
    return " ".join(parts)

# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--print_json_every", type=int, default=15)
    args = ap.parse_args()

    cfg = load_yaml(PIPELINE_CFG)
    mapper = OntologyMapper(ONTOLOGY_CFG)

    # --- RealSense: color+depth, align depth->color, metres via depth_scale ---
    pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, args.width,
                         args.height, rs.format.bgr8, args.fps)
    rs_cfg.enable_stream(rs.stream.depth, args.width,
                         args.height, rs.format.z16, args.fps)
    profile = pipe.start(rs_cfg)

    # Official API: map z16 depth units to metres via get_depth_scale()
    depth_scale = float(profile.get_device(
    ).first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)  # align depth to color per SDK docs
    print(f"[smart_walker] depth_scale = {depth_scale:.6f} m per unit")

    model = YOLO(args.model)
    print("Press 'q' to quit. Press 'p' to print current facts_json.")
    frame_idx, last_json_print = 0, -1

    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue

            color_np = np.asanyarray(color.get_data())   # BGR8
            depth_u16 = np.asanyarray(depth.get_data())  # z16
            depth_m = depth_u16.astype(np.float32) * depth_scale

            res = model.predict(source=color_np, imgsz=args.imgsz,
                                conf=args.conf, verbose=False)[0]
            names = res.names
            H, W = depth_m.shape[:2]
            objs, hazards = [], []

            for i, b in enumerate(res.boxes):
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                raw_label = names[cls]
                mapped = mapper.map_label(raw_label)
                bearing = bearing_from_bbox(
                    (x1, y1, x2, y2), W, cfg["bearing"])
                d_m = median_depth_in_box(depth_m, x1, y1, x2, y2)
                d_bin = distance_bin_from_m(d_m, cfg["depth"]["metric_bins_m"])
                o = {
                    "id": i,
                    "raw_label": raw_label,
                    "canonical_class": mapped.canonical_class,
                    "ontology_class": mapped.ontology_class,
                    "conf": round(conf, 3),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "distance_m": None if d_m is None else round(d_m, 2),
                    "distance_bin": d_bin,
                    "bearing": bearing
                }
                objs.append(o)
                if mapped.ontology_class == "hazard":
                    hazards.append(o["id"])

            facts = {
                "frame_id": frame_idx,
                "timestamp_ms": int(time.time() * 1000),
                "caption": "",
                "risk": "",
                "objects": objs,
                "free_space": {"corridor_min_width_m": None, "nearest_obstacle_m": None},
                "hazards": hazards,
                "uncertainty": {"depth_std": None, "low_light": False},
                "source_depth": "rgbd",
                "explain": {"rules_fired": [], "min_distance_m": None, "class_counts": {}}
            }

            r = compute_baseline_risk(facts, cfg)
            facts["risk"] = r["risk"]
            facts["explain"]["rules_fired"] = r["rules_fired"]
            facts["caption"] = make_caption(facts)

            vis = draw_overlay(color_np, objs, facts["risk"], facts["caption"])
            cv2.imshow(
                "smart_walker — RealSense (ontology + baseline risk + caption)", vis)

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
