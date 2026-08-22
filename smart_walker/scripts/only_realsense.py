# scripts/realsense_shared_control.py
from __future__ import annotations
import ctypes
import argparse
import json
import os
import sys
import time
import math
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from ultralytics import YOLO
from rapidfuzz import process, fuzz

# ----------------------------- Perf niceties -----------------------------
try:
    import torch
    torch.backends.cudnn.benchmark = True
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)  # avoid thread fights
except Exception:
    pass

# ----------------------------- Paths & Config ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_DIR = Path(os.environ.get("SMART_WALKER_CONFIG_DIR", ROOT / "config"))
PIPELINE_CFG = CONFIG_DIR / "pipeline.yaml"
ONTOLOGY_CFG = CONFIG_DIR / "ontology.yaml"

ACTIVE_KEY_WINDOW_MS = 150
PRINT_HOLD_INTERVAL_MS = 250


def load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------- Key Handling ------------------------------
user32 = ctypes.windll.user32 if os.name == "nt" else None
VK = {'W': 0x57, 'A': 0x41, 'S': 0x53, 'D': 0x44,
      'SPACE': 0x20, 'Q': 0x51, 'P': 0x50}


def key_down(vk: int) -> bool:
    if user32 is None:
        return False
    return (user32.GetAsyncKeyState(vk) & 0x8000) != 0


@dataclass
class IntentState:
    last_edge_dir: str = "idle"
    last_press_ms: int = 0
    level_dir: str = "idle"
    p_is_down: bool = False
    quit_requested: bool = False


class KeyListener:
    def __init__(self, poll_hz=120):
        self.prev = {k: False for k in VK}
        self.state = IntentState()
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.dt = 1.0 / float(max(1, poll_hz))
        self.t = threading.Thread(target=self._loop, daemon=True)

    def start(self): self.t.start(); return self

    def stop(self):
        self.stop_evt.set()
        try:
            self.t.join(timeout=1.0)
        except:
            pass

    def _edge(self, vk, name):
        isdn = key_down(vk)
        was = self.prev[name]
        self.prev[name] = isdn
        return (not was) and isdn

    def _loop(self):
        while not self.stop_evt.is_set():
            now_ms = int(time.time() * 1000)

            # Edge (tap)
            if self._edge(VK['W'], 'W'):
                self._set_edge("forward", now_ms)
            elif self._edge(VK['S'], 'S'):
                self._set_edge("backward", now_ms)
            elif self._edge(VK['A'], 'A'):
                self._set_edge("left", now_ms)
            elif self._edge(VK['D'], 'D'):
                self._set_edge("right", now_ms)
            elif self._edge(VK['SPACE'], 'SPACE'):
                self._set_edge("idle", now_ms)

            # Level (held)
            lvl = "idle"
            if key_down(VK['W']):
                lvl = "forward"
            elif key_down(VK['S']):
                lvl = "backward"
            elif key_down(VK['A']):
                lvl = "left"
            elif key_down(VK['D']):
                lvl = "right"

            with self.lock:
                self.state.level_dir = lvl
                self.state.p_is_down = key_down(VK['P'])
                if self._edge(VK['Q'], 'Q'):
                    self.state.quit_requested = True

            time.sleep(self.dt)

    def _set_edge(self, d, tms):
        with self.lock:
            self.state.last_edge_dir = d
            self.state.last_press_ms = tms

    def snapshot(self) -> IntentState:
        with self.lock:
            return IntentState(**self.state.__dict__)

# --------------------------- Ontology Mapper ----------------------------


@dataclass
class Mapped:
    canonical_class: Optional[str]
    ontology_class: str


class OntologyMapper:
    def __init__(self, ontology_path: Path, sim_threshold: float = 80.0):
        cfg = load_yaml(ontology_path)
        self.sim_threshold = sim_threshold
        self.ontology_buckets = {}
        self.prompts = []
        self.prompt2canon = {}
        self.synonyms = {k.lower(): v.lower()
                         for k, v in cfg.get("synonyms_to_canonical", {}).items()}
        for bucket in cfg["ontology"]:
            ont = bucket["name"]
            for c in bucket.get("canonical", []):
                cl = c.lower()
                self.ontology_buckets[cl] = ont
                self.prompts.append(cl)
                self.prompt2canon[cl] = cl
            for p in bucket.get("prompts", []):
                pl = p.lower()
                self.prompts.append(pl)
                if bucket.get("canonical"):
                    self.prompt2canon[pl] = bucket["canonical"][0].lower()

    def map_label(self, raw_label: str) -> Mapped:
        if not raw_label:
            return Mapped(None, "unknown_obstacle")
        s = raw_label.strip().lower()
        if s in self.synonyms:
            canon = self.synonyms[s]
            return Mapped(canon, self.ontology_buckets.get(canon, "unknown_obstacle"))
        if s in self.ontology_buckets:
            return Mapped(s, self.ontology_buckets[s])
        match = process.extractOne(s, self.prompts, scorer=fuzz.WRatio)
        if match and match[1] >= self.sim_threshold:
            canon = self.prompt2canon.get(match[0])
            if canon:
                return Mapped(canon, self.ontology_buckets.get(canon, "unknown_obstacle"))
        return Mapped(None, "unknown_obstacle")

# ----------------------- Detection / Depth helpers ----------------------


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
    if d_m is None or (isinstance(d_m, float) and (np.isnan(d_m) or np.isinf(d_m))):
        return "unknown"
    for name, (lo, hi) in bins_cfg.items():
        if lo <= d_m < hi:
            return name
    return "far"


def _safe_int_bounds(x1, y1, x2, y2, W, H):
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(W, int(np.ceil(x2)))
    y2 = min(H, int(np.ceil(y2)))
    return x1, y1, x2, y2


def _median_nonzero(a: np.ndarray) -> Optional[float]:
    vals = a[np.nonzero(a)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def median_depth_in_box(depth_m: np.ndarray, x1, y1, x2, y2, shrink_ratio=0.06, max_rand_samples=400):
    H, W = depth_m.shape[:2]
    dx, dy = shrink_ratio * (x2 - x1), shrink_ratio * (y2 - y1)
    rx1, ry1, rx2, ry2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    rx1, ry1, rx2, ry2 = _safe_int_bounds(rx1, ry1, rx2, ry2, W, H)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    patch = depth_m[ry1:ry2, rx1:rx2]
    m = _median_nonzero(patch)
    if m is not None:
        return m

    # centre fallback
    cx, cy = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)
    cx = min(W - 1, max(0, cx))
    cy = min(H - 1, max(0, cy))
    v = depth_m[cy, cx]
    if v > 0:
        return float(v)

    # random nonzero fallback
    fx1, fy1, fx2, fy2 = _safe_int_bounds(x1, y1, x2, y2, W, H)
    full = depth_m[fy1:fy2, fx1:fx2]
    nz = np.transpose(np.nonzero(full))
    if nz.shape[0] == 0:
        return None
    idx = np.random.choice(nz.shape[0], size=min(
        max_rand_samples, nz.shape[0]), replace=False)
    samples = full[nz[idx, 0], nz[idx, 1]]
    return float(np.median(samples)) if samples.size else None

# --------------------------- Risk computation ---------------------------


def _nearest_obstacle_m(objs):
    ds = [o.get("distance_m")
          for o in objs if isinstance(o.get("distance_m"), (int, float))]
    return min(ds) if ds else None


def _multi_near_count(objs):
    return sum(1 for o in objs if o.get("distance_bin") in ("very_close", "near"))


def compute_baseline_risk(facts: dict, cfg: dict) -> dict:
    rf = []
    rules = cfg["risk_rules_baseline"]
    objs = facts.get("objects", [])
    nearest_m = _nearest_obstacle_m(objs)
    facts["explain"]["min_distance_m"] = nearest_m

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

# --------------------------- Decision & captions ------------------------


def sector_for(bearing: str) -> str:
    return "L" if bearing == "left" else ("R" if bearing == "right" else "C")


def arbiter_decision(intent_raw: str, facts: dict, cfg: dict) -> tuple[str, str]:
    # intent_raw is one of: forward/backward/left/right (we provide effective dir)
    if facts.get("risk") == "stop":
        return "STOP", "obstacle ahead"
    sector = {"forward": "C", "backward": "C",
              "left": "L", "right": "R"}.get(intent_raw, "C")
    near_hi = cfg["depth"]["metric_bins_m"]["near"][1]
    nearest_sector_m = math.inf
    for o in facts.get("objects", []):
        dm = o.get("distance_m")
        if dm is None:
            continue
        if sector_for(o.get("bearing", "centre")) != sector:
            continue
        nearest_sector_m = min(nearest_sector_m, float(dm))
    if nearest_sector_m < near_hi:
        return "SLOW", "nearby obstacle"
    return "GO", "clear"


def _pretty_dir(d: str) -> str:
    return {"left": "left", "right": "right", "forward": "forward", "backward": "backward"}.get(d, "forward")


def make_user_caption(decision: str, intent_dir: str) -> str:
    dir_word = _pretty_dir(intent_dir)
    if decision == "GO":
        return f"Going {dir_word} as you intended."
    if decision == "SLOW":
        return f"Please be cautious going {dir_word} as you intended, as there are obstacles."
    # STOP
    return "We are stopping as there are obstacles nearby."


def draw_overlay(img, objects, risk, user_caption, intent_line, decision):
    vis = img.copy()
    color = (0, 255, 0) if decision == "GO" else (
        (0, 255, 255) if decision == "SLOW" else (0, 0, 255))
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 90), (0, 0, 0), -1)
    cv2.putText(vis, f"RISK: {risk.upper()}   DECISION: {decision}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(vis, intent_line[:100], (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(vis, user_caption[:95], (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    for o in objects:
        x1, y1, x2, y2 = map(int, o["bbox_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        dist_txt = "?" if o["distance_m"] is None else f"{o['distance_m']:.2f}m"
        role = o.get("ontology_class") or "unknown"
        base = o.get("display_label") or o.get("raw_label")
        canon = o.get("canonical_class") or base
        label = f"{canon} [{role}] {o['conf']:.2f} {o['bearing']} {dist_txt} ({o['distance_bin']})"
        cv2.putText(vis, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Objects: {len(objects)}", (vis.shape[1]-170, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2, cv2.LINE_AA)
    return vis

# ------------------------------- Packets --------------------------------


@dataclass
class FramePacket:
    color: np.ndarray
    depth_m: np.ndarray
    ts_ms: int


@dataclass
class InferPacket:
    ts_ms: int
    color: np.ndarray
    objects: List[Dict[str, Any]]
    hazards: List[int]

# ---------------------- Shared JSON snapshot (threadsafe) ----------------


class JsonSnapshot:
    def __init__(self):
        self.lock = threading.Lock()
        self.payload: Dict[str, Any] = {}

    def set(self, data: Dict[str, Any]):
        with self.lock:
            self.payload = data

    def get(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.payload)


def json_streamer(snapshot: JsonSnapshot, stop_evt: threading.Event, hz: float, pretty: bool):
    if hz <= 0:
        return
    period = 1.0 / float(hz)
    dumps_kwargs = {"ensure_ascii": False}
    if pretty:
        dumps_kwargs["indent"] = 2
    else:
        dumps_kwargs["separators"] = (",", ":")  # compact NDJSON
    try:
        while not stop_evt.is_set():
            data = snapshot.get()
            if data:
                s = json.dumps(data, **dumps_kwargs)
                if not pretty:
                    # NDJSON: one line per record
                    sys.stdout.write(s + "\n")
                else:
                    # pretty: add a blank line between updates for readability
                    sys.stdout.write(s + "\n\n")
                sys.stdout.flush()
            time.sleep(period)
    except Exception as e:
        print("[json_streamer] ERROR:", repr(e))
        traceback.print_exc()

# --------------------------- Worker Threads -----------------------------


def capture_thread(pipe, align, depth_scale, out_q: Queue, stop_evt: threading.Event):
    try:
        while not stop_evt.is_set():
            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            d = frames.get_depth_frame()
            c = frames.get_color_frame()
            if not d or not c:
                continue
            color_np = np.asanyarray(c.get_data())
            depth_u16 = np.asanyarray(d.get_data())
            depth_m = depth_u16.astype(np.float32) * depth_scale
            pkt = FramePacket(color=color_np, depth_m=depth_m,
                              ts_ms=int(time.time()*1000))
            while not out_q.empty():
                try:
                    out_q.get_nowait()
                except Empty:
                    break
            out_q.put(pkt)
    except Exception as e:
        print("[capture_thread] ERROR:", repr(e))
        traceback.print_exc()


def inference_thread(cfg, mapper: OntologyMapper, model: YOLO, in_q: Queue, out_q: Queue,
                     stop_evt: threading.Event, imgsz: int, conf: float, half: bool,
                     classes, use_track: bool, tracker_yaml: str):
    try:
        names_cache = None
        while not stop_evt.is_set():
            try:
                pkt: FramePacket = in_q.get(timeout=0.05)
            except Empty:
                continue

            color_np, depth_m, ts = pkt.color, pkt.depth_m, pkt.ts_ms

            if use_track:
                results = model.track(source=color_np, imgsz=imgsz, conf=conf, verbose=False,
                                      persist=True, tracker=tracker_yaml, half=half, classes=classes)
            else:
                results = model.predict(source=color_np, imgsz=imgsz, conf=conf, verbose=False,
                                        half=half, classes=classes)

            r0 = results[0]
            names = getattr(r0, "names", None) or getattr(
                results, "names", None) or getattr(model, "names", None)
            if names_cache is None and names is not None:
                names_cache = names
            boxes = getattr(r0, "boxes", None)

            objs: List[Dict[str, Any]] = []
            hazards: List[int] = []

            if boxes is not None and len(boxes) > 0:
                bx = boxes.xyxy.cpu().numpy()
                cf = boxes.conf.cpu().numpy() if hasattr(
                    boxes, "conf") else np.zeros((bx.shape[0],))
                cl = boxes.cls.cpu().numpy() if hasattr(
                    boxes, "cls") else np.zeros((bx.shape[0],))
                ids = boxes.id.int().cpu().tolist() if hasattr(
                    boxes, "id") and boxes.id is not None else [None] * len(bx)

                H, W = depth_m.shape[:2]
                for i, (xyxy, score, cls_idx, tid) in enumerate(zip(bx, cf, cl, ids)):
                    x1, y1, x2, y2 = map(float, xyxy.tolist())
                    raw_label = names_cache[int(
                        cls_idx)] if names_cache is not None else str(int(cls_idx))
                    mapped = mapper.map_label(raw_label)
                    bearing = bearing_from_bbox(
                        (x1, y1, x2, y2), W, cfg["bearing"])
                    d_m = median_depth_in_box(
                        depth_m, x1, y1, x2, y2, shrink_ratio=0.06)
                    d_bin = distance_bin_from_m(
                        d_m, cfg["depth"]["metric_bins_m"])
                    base = f"{raw_label} #{tid}" if tid is not None and raw_label != "person" else \
                        (f"person #{tid}" if tid is not None else raw_label)
                    obj = {
                        "id": int(tid) if tid is not None else i,
                        "raw_label": raw_label,
                        "display_label": base,
                        "canonical_class": mapped.canonical_class,
                        "ontology_class": mapped.ontology_class,
                        "conf": round(float(score), 3),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "distance_m": None if d_m is None else round(d_m, 2),
                        "distance_bin": d_bin,
                        "bearing": bearing
                    }
                    objs.append(obj)
                    if mapped.ontology_class == "hazard":
                        hazards.append(obj["id"])

            out = InferPacket(ts_ms=ts, color=color_np,
                              objects=objs, hazards=hazards)

            while not out_q.empty():
                try:
                    out_q.get_nowait()
                except Empty:
                    break
            out_q.put(out)

    except Exception as e:
        print("[inference_thread] ERROR:", repr(e))
        traceback.print_exc()

# ---------------------------------- Main --------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-oiv7.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--print_json_every", type=int, default=0,
                    help="Legacy: print every N frames (0=off)")
    ap.add_argument("--json_hz", type=float, default=10.0,
                    help="If >0, stream JSON to stdout at this rate")
    ap.add_argument("--json_pretty", action="store_true",
                    help="Pretty multi-line JSON instead of compact NDJSON")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="Ultralytics tracker config (e.g., bytetrack.yaml)")
    ap.add_argument("--half", action="store_true",
                    help="Run model in FP16 if CUDA is available")
    ap.add_argument("--track", action="store_true",
                    help="Use model.track() (IDs). Default off = predict()")
    args = ap.parse_args()

    cfg = load_yaml(PIPELINE_CFG)
    mapper = OntologyMapper(ONTOLOGY_CFG)

    # RealSense init (align depth -> color)
    pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, args.width,
                         args.height, rs.format.bgr8, args.fps)
    rs_cfg.enable_stream(rs.stream.depth, args.width,
                         args.height, rs.format.z16, args.fps)
    profile = pipe.start(rs_cfg)
    depth_scale = float(profile.get_device(
    ).first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)
    print(f"[smart_walker] depth_scale = {depth_scale:.6f} m/unit")
    print("Hold W/A/S/D to issue intent. SPACE=idle. P=print JSON. Q=quit.")

    model = YOLO(args.model)

    # Queues & workers
    cap_q = Queue(maxsize=1)
    inf_in = Queue(maxsize=1)
    inf_out = Queue(maxsize=1)
    stop_evt = threading.Event()

    threading.Thread(target=capture_thread, args=(
        pipe, align, depth_scale, cap_q, stop_evt), daemon=True).start()
    threading.Thread(
        target=inference_thread,
        args=(cfg, mapper, model, inf_in, inf_out, stop_evt,
              args.imgsz, args.conf, (args.half and HAS_TORCH), None,
              args.track, args.tracker),
        daemon=True
    ).start()

    # JSON snapshot + streamer
    snapshot = JsonSnapshot()
    threading.Thread(target=json_streamer, args=(
        snapshot, stop_evt, args.json_hz, args.json_pretty), daemon=True).start()

    WIN = "smart_walker — shared control (robust)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 720)

    kbd = KeyListener(poll_hz=120).start()

    frame_idx = 0
    last_manual_print_ms = 0
    last_json_print = -1
    latest_color = None

    # Persist last known inference + UI state
    last_objects: List[Dict[str, Any]] = []
    last_risk: str = "safe"
    last_decision: str = "GO"
    last_caption: str = "Going forward as you intended."
    last_intent_line: str = "Intent (level): idle   |   Intent (edge): idle"
    last_hazards: List[int] = []

    # remember last non-idle direction for captions when idle
    last_non_idle_dir: str = "forward"

    try:
        while True:
            # UI pump
            if hasattr(cv2, "pollKey"):
                cv2.pollKey()
            else:
                cv2.waitKey(1)

            # Pull freshest capture → push to inference
            try:
                pkt: FramePacket = cap_q.get_nowait()
                latest_color = pkt.color
                while not inf_in.empty():
                    try:
                        inf_in.get_nowait()
                    except Empty:
                        break
                inf_in.put(pkt)
            except Empty:
                pass

            # Keys
            s = kbd.snapshot()
            if s.quit_requested:
                break
            now_ms = int(time.time() * 1000)
            active = (now_ms - s.last_press_ms) <= ACTIVE_KEY_WINDOW_MS

            if s.last_edge_dir != "idle":
                last_non_idle_dir = s.last_edge_dir

            # Choose effective direction (always one of L/R/F/B)
            intent_for_caption = s.last_edge_dir if active else "idle"
            effective_dir = intent_for_caption if intent_for_caption != "idle" else last_non_idle_dir
            intent_line = f"Intent (level): {s.level_dir}   |   Intent (edge): {s.last_edge_dir}"

            # Try to consume a new inference result (non-blocking)
            got_new = False
            try:
                out: InferPacket = inf_out.get_nowait()
                got_new = True

                facts = {
                    "frame_id": frame_idx,
                    "timestamp_ms": out.ts_ms,
                    "objects": out.objects,
                    "free_space": {"corridor_min_width_m": None, "nearest_obstacle_m": None},
                    "hazards": out.hazards,
                    "uncertainty": {"depth_std": None, "low_light": False},
                    "source_depth": "rgbd",
                    "explain": {"rules_fired": [], "min_distance_m": None, "class_counts": {}}
                }

                r = compute_baseline_risk(facts, cfg)
                risk = r["risk"]
                decision, _ = arbiter_decision(
                    effective_dir, {"risk": risk, **facts}, cfg)
                caption = make_user_caption(decision, effective_dir)

                # Update persisted state
                last_objects = out.objects
                last_hazards = out.hazards
                last_risk = risk
                last_decision = decision
                last_caption = caption
                last_intent_line = intent_line

            except Empty:
                pass

            # Render with latest known state
            if latest_color is not None:
                vis = draw_overlay(
                    latest_color, last_objects, last_risk, last_caption, intent_line, last_decision)
                cv2.imshow(WIN, vis)

            # --------- Build JSON snapshot every frame (for streamer & optional prints) ----------
            payload = {
                "frame_id": frame_idx,
                "timestamp_ms": int(time.time() * 1000),
                "intent": {
                    "edge": s.last_edge_dir,
                    "level": s.level_dir,
                    "active": active,
                    "effective": effective_dir
                },
                "caption": last_caption,
                "decision": last_decision,
                "risk": last_risk,
                "hazards": last_hazards,
                "objects": last_objects
            }
            snapshot.set(payload)

            # Manual/periodic prints use same pretty/compact style as streamer
            if s.p_is_down and (now_ms - last_manual_print_ms) >= PRINT_HOLD_INTERVAL_MS:
                print(json.dumps(payload, indent=2) if args.json_pretty else json.dumps(
                    payload, separators=(",", ":"), ensure_ascii=False))
                last_manual_print_ms = now_ms
            if args.print_json_every > 0 and frame_idx % args.print_json_every == 0 and frame_idx != last_json_print:
                print(json.dumps(payload, indent=2) if args.json_pretty else json.dumps(
                    payload, separators=(",", ":"), ensure_ascii=False))
                last_json_print = frame_idx

            frame_idx += 1

    finally:
        stop_evt.set()
        kbd.stop()
        try:
            pipe.stop()
        except:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
