from __future__ import annotations
import argparse
import json
import time
import os
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from rapidfuzz import process, fuzz

# ---------------- paths & config ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_DIR = Path(os.environ.get("SMART_WALKER_CONFIG_DIR", ROOT / "config"))
PIPELINE_CFG = CONFIG_DIR / "pipeline.yaml"
ONTOLOGY_CFG = CONFIG_DIR / "ontology.yaml"

PRESETS = {
    "meters": 1.0,          # depth PNG already in metres
    # millimetres -> metres (what our bag export saved)
    "realsense_mm": 1/1000,
    "tum_5000": 1/5000,     # TUM: depth = metres * 5000
    "units": None           # raw z16 * depth_scale (pass --depth_scale)
}


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
    cx = (x1+x2)/2.0
    frac = cx/float(img_w)
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


def draw_overlay(rgb, objects):
    out = rgb.copy()
    for o in objects:
        x1, y1, x2, y2 = map(int, o["bbox_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        dist_txt = "?" if o["distance_m"] is None else f"{o['distance_m']:.2f}m"
        role = o.get("ontology_class") or "unknown"
        canon = o.get("canonical_class") or o["raw_label"]
        label = f"{canon} [{role}] {o['conf']:.2f} {o['bearing']} {dist_txt} ({o['distance_bin']})"
        cv2.putText(out, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return out

# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb",   required=True, help="Aligned color PNG/JPG")
    ap.add_argument("--depth", required=True, help="Aligned 16-bit depth PNG")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="realsense_mm",
                    help="How to convert depth PNG to metres")
    ap.add_argument("--depth_scale", type=float, default=None,
                    help="Required if --preset=units (metres per unit)")
    ap.add_argument("--model", default="yolov8n-oiv7.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    pipe_cfg = load_yaml(PPIPE := PIPELINE_CFG)
    mapper = OntologyMapper(ONTOLOGY_CFG)

    # convert depth units -> metres
    scale = PRESETS[args.preset]
    if args.preset == "units":
        if args.depth_scale is None:
            raise ValueError(
                "With --preset=units you must pass --depth_scale (metres per unit).")
        scale = args.depth_scale

    # load RGB
    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Failed to read RGB: {args.rgb}")

    # load depth robustly (expect uint16, single-channel)
    depth_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"Failed to read depth: {args.depth}")
    if depth_raw.ndim == 3 and depth_raw.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if depth_raw.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        depth_raw = cv2.cvtColor(depth_raw, code)
    if depth_raw.dtype != np.uint16:
        tmp = cv2.imread(args.depth, cv2.IMREAD_ANYDEPTH)
        if tmp is None or tmp.dtype != np.uint16 or tmp.ndim != 2:
            raise RuntimeError(
                f"Depth PNG must be 16-bit single-channel. Got dtype={depth_raw.dtype}, shape={depth_raw.shape}")
        depth_raw = tmp

    H, W = depth_raw.shape[:2]
    if (rgb.shape[0], rgb.shape[1]) != (H, W):
        raise RuntimeError(
            f"RGB ({rgb.shape[1]}x{rgb.shape[0]}) and depth ({W}x{H}) sizes differ; they must be aligned 1:1.")

    depth_m = depth_raw.astype(np.float32) * float(scale)

    # detect
    model = YOLO(args.model)
    res = model.predict(source=rgb, imgsz=args.imgsz,
                        conf=args.conf, verbose=False)[0]
    names = res.names

    objs, hazards = [], []
    for i, b in enumerate(res.boxes):
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        raw_label = names[cls]
        mapped = mapper.map_label(raw_label)
        bearing = bearing_from_bbox((x1, y1, x2, y2), W, pipe_cfg["bearing"])
        d_m = median_depth_in_box(depth_m, x1, y1, x2, y2)
        d_bin = distance_bin_from_m(d_m, pipe_cfg["depth"]["metric_bins_m"])
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
        "frame_id": 0,
        "timestamp_ms": int(time.time()*1000),
        "caption": "",
        "risk": "",
        "objects": objs,
        "free_space": {"corridor_min_width_m": None, "nearest_obstacle_m": None},
        "hazards": hazards,  # list of object ids in hazard role
        "uncertainty": {"depth_std": None, "low_light": False},
        "source_depth": "rgbd",
        "explain": {"rules_fired": [], "min_distance_m": None, "class_counts": {}}
    }

    vis = draw_overlay(rgb, objs)
    cv2.imshow("smart_walker — RGBD + ontology mapping (offline pair)", vis)
    print(json.dumps(facts, indent=2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
