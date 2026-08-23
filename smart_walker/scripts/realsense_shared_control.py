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
import requests

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
IMU_MAX_DRAIN_PER_LOOP = 8  # cap how many motion frames to drain per loop


def load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------- Key Handling ------------------------------
user32 = ctypes.windll.user32 if os.name == "nt" else None
VK = {'W': 0x57, 'A': 0x41, 'S': 0x53, 'D': 0x44,
      'SPACE': 0x20, 'Q': 0x51, 'P': 0x50,
      'Y': 0x59, 'N': 0x4E, 'M': 0x4D, 'R': 0x52}


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
    yes_edge: bool = False
    no_edge: bool = False
    more_detail_edge: bool = False
    reassess_edge: bool = False


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
            elif self._edge(VK['Y'], 'Y'):
                with self.lock:
                    self.state.yes_edge = True
            elif self._edge(VK['N'], 'N'):
                with self.lock:
                    self.state.no_edge = True
            elif self._edge(VK['M'], 'M'):
                with self.lock:
                    self.state.more_detail_edge = True
            elif self._edge(VK['R'], 'R'):
                with self.lock:
                    self.state.reassess_edge = True

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
            snapshot = IntentState(**self.state.__dict__)
            self.state.yes_edge = False
            self.state.no_edge = False
            self.state.more_detail_edge = False
            self.state.reassess_edge = False
            return snapshot

# --------------------------- Ontology Mapper ----------------------------


@dataclass
class Mapped:
    canonical_class: Optional[str]
    ontology_class: str


class OntologyMapper:
    """Assigns a detector label to a safety bucket, by exact match only.

    The bucket is not cosmetic. `hazard` blocks a sector at 2.00 m where everything else blocks at
    0.70 m, and the canonical name it returns becomes the word the caption is permitted to use for
    the object.

    **Matching used to fall back to string similarity, and it must not.** The previous form called
    `rapidfuzz.WRatio` and accepted any match scoring 80 or better. Run over the detector's real
    vocabulary on 22 August 2026, that produced `stop sign -> stairs_up -> hazard`, so the walker
    stopped for a road sign and recorded a staircase; `tie -> person -> agent`, so a necktie became
    eligible to be reported as a moving person; and `carrot -> trolley`, `cat -> trolley`,
    `hair drier -> chair` and `dining table -> chair`, the last of which renamed a table in the Fact
    Packet that the whole safety argument treats as authoritative. A label the ontology does not
    name is now `unknown_obstacle`, which is what it is.

    A word appearing only in a bucket's `prompts` list yields the bucket and no canonical name, so
    the detector's own word survives into the caption. Mapping it to the bucket's first canonical,
    as the previous form did, turned every bench and desk into a chair.

    The ontology's `not_obstacles` list is read here but not applied by `map_label`, which has no
    way to say "no object". `is_not_obstacle` reports it separately and the detection loop drops the
    detection before it becomes a fact.
    """

    def __init__(self, ontology_path: Path):
        cfg = load_yaml(ontology_path)
        self.ontology_buckets: dict[str, str] = {}
        self.prompt_buckets: dict[str, str] = {}
        self.synonyms = {k.lower(): v.lower()
                         for k, v in cfg.get("synonyms_to_canonical", {}).items()}
        self.not_obstacles = {str(name).strip().lower()
                              for name in cfg.get("not_obstacles", []) or ()}
        for bucket in cfg["ontology"]:
            ont = bucket["name"]
            for c in bucket.get("canonical", []):
                self.ontology_buckets[c.lower()] = ont
            for p in bucket.get("prompts", []):
                self.prompt_buckets.setdefault(p.lower(), ont)

    def is_not_obstacle(self, raw_label: str) -> bool:
        """True where the label names something that cannot be an obstacle on the floor plane.

        A window, a shirt and a person's hand are all reported by the detector and none of them is a
        thing to be steered around. Without this the walker would stop for them, because a label the
        ontology does not name becomes `unknown_obstacle`, which still stops the walker at 0.70 m.
        """
        return bool(raw_label) and raw_label.strip().lower() in self.not_obstacles

    def map_label(self, raw_label: str) -> Mapped:
        if not raw_label:
            return Mapped(None, "unknown_obstacle")
        s = raw_label.strip().lower()
        canonical = self.synonyms.get(s, s)
        if canonical in self.ontology_buckets:
            return Mapped(canonical, self.ontology_buckets[canonical])
        bucket = self.prompt_buckets.get(s)
        if bucket:
            return Mapped(None, bucket)
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


def valid_depth_fraction(depth_m, top_fraction=0.55, bottom_fraction=0.95):
    """Returns the share of the reasoning region that carries a usable depth reading, 0.0 to 1.0.

    The measurement that replaced the low-light rule on 23 August 2026. That rule compared the mean
    grey level of the colour frame against 40 and could never fire, because nothing computed the
    mean and the flag was written as False on every frame. Measuring the three backlit staircase
    captures showed the threshold also pointed the wrong way: those frames average 101 and are
    brighter than the ordinary room frames at 73 to 94. The adverse condition is not darkness but
    high contrast, a window blown out at 250 with the staircase in shadow beneath it, and a mean
    cannot see that.

    What did separate them is how much depth came back. Over the same nine frames the backlit
    captures returned 34 to 54 per cent valid depth against a steady 67 to 68 per cent in the
    ordinary room. That measures the consequence rather than guessing at the cause, and it covers
    causes unrelated to light: a glossy floor, a dark absorbing surface, a wall closer than the
    stereo baseline resolves.

    The region is the same band `compute_lane_state` reasons over, rows 55 to 95 per cent of the
    frame, so the figure describes the depth the decision was actually made from rather than the
    whole image.

    A zero in a RealSense depth frame means no reading, not a surface at zero distance.
    """
    if depth_m is None or not isinstance(depth_m, np.ndarray) or depth_m.size == 0:
        return None
    height = depth_m.shape[0]
    band = depth_m[int(top_fraction * height):int(bottom_fraction * height), :]
    if band.size == 0:
        return None
    usable = np.isfinite(band) & (band > 0)
    return float(np.count_nonzero(usable)) / float(band.size)


def distance_bin_from_m(d_m, bins_cfg):
    """Labels a distance with the band it falls in, or "unknown" where no band contains it.

    The fallback was "far", so a distance below the lowest band, a negative reading among them,
    was labelled with the band that means the most room. A value the bands do not cover is a value
    the system has no opinion about, and "unknown" is what the rest of the pipeline already treats
    as not free space. A reading beyond the top of the far band, which the D455f does not produce
    at 99 metres, is the only case that loses a label it previously had, and calling that unknown
    is correct rather than a regression.
    """
    if d_m is None or (isinstance(d_m, float) and (np.isnan(d_m) or np.isinf(d_m))):
        return "unknown"
    for name, (lo, hi) in bins_cfg.items():
        if lo <= d_m < hi:
            return name
    return "unknown"


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


def median_depth_lower_band(
    depth_m: np.ndarray,
    x1,
    y1,
    x2,
    y2,
    band_frac: float = 0.33,
    horiz_shrink: float = 0.06,
    max_rand_samples: int = 400,
) -> Optional[float]:
    """
    Estimate object distance using only the lower portion of the bbox.

    Rationale: the lower band of an object's bbox is typically closer to the
    ground contact point (or closest visible point to the camera), producing a
    more conservative and realistic distance estimate for navigation.

    - band_frac: fraction of the bbox height used from the bottom (e.g. 0.33 = bottom third)
    - horiz_shrink: shrink the band horizontally to avoid edge artifacts
    """
    try:
        H, W = depth_m.shape[:2]
        # Compute a band covering the bottom band_frac of the bbox
        bh = max(1.0, float(y2 - y1))
        band_h = max(1.0, band_frac * bh)
        by1 = y2 - band_h
        by2 = y2
        # Horizontal shrink similar to median_depth_in_box
        dx = horiz_shrink * (x2 - x1)
        bx1, by1i, bx2, by2i = _safe_int_bounds(
            x1 + dx, by1, x2 - dx, by2, W, H)
        if bx2 <= bx1 or by2i <= by1i:
            return None
        patch = depth_m[by1i:by2i, bx1:bx2]
        m = _median_nonzero(patch)
        if m is not None:
            return m

        # If the band is entirely zeros (e.g., missing depth), try a sparse random sample
        nz = np.transpose(np.nonzero(patch))
        if nz.shape[0] == 0:
            return None
        idx = np.random.choice(nz.shape[0], size=min(
            max_rand_samples, nz.shape[0]), replace=False)
        samples = patch[nz[idx, 0], nz[idx, 1]]
        return float(np.median(samples)) if samples.size else None
    except Exception:
        return None

# --------------------------- Risk computation ---------------------------


def _nearest_obstacle_m(objs):
    ds = [o.get("distance_m")
          for o in objs if isinstance(o.get("distance_m"), (int, float))]
    return min(ds) if ds else None


def _multi_near_count(objs, caution_below_m):
    """Counts objects measured closer than the caution distance.

    Counted as `distance_bin in ("very_close", "near")` until 23 August 2026, which read a word
    describing the measurement rather than the measurement. The two agree, the label being derived
    from the same number by `distance_bin_from_m` moments earlier, and the count is unchanged:
    very_close and near together are everything below 1.50 m, which is the caution distance itself.

    The change is that a rule deciding motion no longer depends on a label it did not compute. It
    also makes the statement in `pipeline.yaml` true, that the distance bands are for presentation
    and nothing deciding motion reads them. That statement was written on 23 August 2026 and was
    false when written, this rule being the exception.
    """
    return sum(1 for o in objs
               if isinstance(o.get("distance_m"), (int, float))
               and o["distance_m"] < caution_below_m)


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
    if _multi_near_count(objs, caut["nearest_obstacle_m_lt"]) >= caut["multi_near_objects_count_gte"]:
        rf.append("caution:multi_near")
    # Depth coverage. `min_valid_depth_fraction` is absent from an older configuration, in which
    # case the rule is skipped rather than defaulted, so replaying an archived run does not apply a
    # rule that run was never subject to.
    min_valid = caut.get("min_valid_depth_fraction")
    valid_fraction = facts.get("uncertainty", {}).get("valid_depth_fraction")
    if min_valid is not None and valid_fraction is not None \
            and float(valid_fraction) < float(min_valid):
        rf.append("caution:depth_coverage")
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
    depth_m: np.ndarray
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


# --------------------------- LLM Integration ----------------------------


def _safe_min_distance(objects: List[Dict[str, Any]]) -> Optional[float]:
    dists = [o.get("distance_m")
             for o in objects if o.get("distance_m") is not None]
    if not dists:
        return None
    try:
        return float(min(dists))
    except Exception:
        return None


def build_scene_summary(payload: Dict[str, Any]) -> str:
    """Create a compact textual summary for the LLM.

    Keep this short to improve latency and stability.
    """
    intent = payload.get("intent", {})
    eff = intent.get("effective", "idle")
    objs = payload.get("objects", [])
    hazards = payload.get("hazards", [])
    motion = payload.get("motion") or {}
    yaw = motion.get("yaw_rate_deg_s")
    accel_g = motion.get("accel_norm_g")
    nearest = _safe_min_distance(objs)
    parts = [
        f"intent: {eff}",
        f"decision: {payload.get('decision', '')}",
        f"risk: {payload.get('risk', '')}",
        f"objects: {len(objs)}",
        f"hazards: {len(hazards)}",
    ]
    if nearest is not None:
        parts.append(f"nearest: {nearest:.2f}m")
    if yaw is not None:
        parts.append(f"yaw_rate: {yaw} deg/s")
    if accel_g is not None:
        parts.append(f"accel: {accel_g} g")
    # Dialog context (if any)
    dlg = payload.get("dialog") or {}
    if dlg.get("pending") and dlg.get("proposed_dir"):
        parts.append(f"ask: take {dlg['proposed_dir']}?")
    lr = dlg.get("last_reply")
    if lr:
        parts.append(f"user_reply: {lr}")
    return "; ".join(parts)


def call_llm_openai(
    endpoint: str,
    system_prompt: str,
    user_content: str,
    grammar: Optional[str],
    model_name: str = "qwen2.5-7b-instruct",
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 128,
    stop: Optional[List[str]] = None,
    timeout: int = 30,
):
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    base = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if stop:
        base["stop"] = stop

    attempts: List[Dict[str, Any]] = []
    if grammar is not None:
        p1 = dict(base)
        p1["grammar"] = {"type": "gbnf", "value": grammar}
        attempts.append(p1)
        p2 = dict(base)
        p2["grammar"] = grammar
        attempts.append(p2)
    attempts.append(dict(base))

    last_err = None
    for pl in attempts:
        try:
            r = requests.post(url, json=pl, timeout=timeout)
            if r.status_code < 400:
                j = r.json()
                return j["choices"][0]["message"]["content"]
            # Keep trying alternate payload formats if server rejects grammar field
            last_err = f"{r.status_code} {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"LLM call failed: {last_err}")


def llm_worker(snapshot: JsonSnapshot, out_state: Dict[str, Any], out_lock: threading.Lock,
               stop_evt: threading.Event,
               hz: float,
               endpoint: str,
               system_path: Path,
               grammar_path: Optional[Path],
               model_name: str,
               temperature: float,
               top_p: float,
               max_tokens: int,
               no_grammar: bool):
    # Rate control
    period = 1.0 / float(hz) if hz and hz > 0 else 0.5

    # Load prompts/grammar
    system_prompt = (
        system_path.read_text(encoding="utf-8")
        if system_path and system_path.exists()
        else "You are a helpful assistant. Output JSON."
    )
    grammar = None
    if not no_grammar and grammar_path and grammar_path.exists():
        grammar = grammar_path.read_text(encoding="utf-8")

    last_payload_ts: Optional[int] = None
    while not stop_evt.is_set():
        try:
            data = snapshot.get()
            if not data:
                time.sleep(period)
                continue

            # Build compact context and invoke the LLM
            ts = data.get("timestamp_ms")
            scene = build_scene_summary(data)
            content = call_llm_openai(
                endpoint=endpoint,
                system_prompt=system_prompt,
                user_content=scene,
                grammar=grammar,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            # Parse JSON content with a safe fallback
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {
                    "advisory": "caution",
                    "reason": "_unparsed_",
                    "suggestion": None,
                    "escalation": 1,
                    "speak": "Proceed with caution.",
                }

            # Safety guard: never contradict STOP from pipeline
            try:
                if str((data.get("risk") or "")).lower() == "stop" or str((data.get("decision") or "")).upper() == "STOP":
                    if parsed.get("advisory") != "stop":
                        parsed.update({
                            "advisory": "stop",
                            "reason": parsed.get("reason") or "Blocked path (safety guard)",
                            "escalation": max(1, int(parsed.get("escalation", 1))),
                        })
            except Exception:
                pass

            # Only accept responses for the latest snapshot timestamp
            if ts is None or last_payload_ts is None or int(ts) >= int(last_payload_ts):
                with out_lock:
                    out_state.clear()
                    out_state.update(parsed)
                    out_state["ts_ms"] = int(time.time() * 1000)
                    if ts is not None:
                        out_state["source_ts"] = int(ts)
                last_payload_ts = int(
                    ts) if ts is not None else last_payload_ts

        except Exception:
            # soft-fail and retry
            pass

        time.sleep(period)


@dataclass
class IMUState:
    gyro: np.ndarray  # rad/s, shape (3,)
    accel: np.ndarray  # m/s^2, shape (3,)
    ts_ms: int
    lock: threading.Lock

    @staticmethod
    def create():
        return IMUState(gyro=np.zeros(3, dtype=np.float32),
                        accel=np.zeros(3, dtype=np.float32),
                        ts_ms=0,
                        lock=threading.Lock())

    def update_from_motion_frame(self, f: rs.frame):
        md = f.as_motion_frame().get_motion_data()
        now_ms = int(time.time() * 1000)
        with self.lock:
            if f.get_profile().stream_type() == rs.stream.gyro:
                self.gyro[:] = (md.x, md.y, md.z)
            elif f.get_profile().stream_type() == rs.stream.accel:
                self.accel[:] = (md.x, md.y, md.z)
            self.ts_ms = now_ms


def capture_thread(pipe, align, depth_scale, out_q: Queue, stop_evt: threading.Event,
                   imu_enabled: bool, imu_state: Optional[IMUState], imu_max_drain: int):
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

            # Update IMU from this frameset if motion frames are included
            if imu_enabled and imu_state is not None:
                try:
                    for f in frames:
                        if f.is_motion_frame():
                            imu_state.update_from_motion_frame(f)
                except Exception:
                    pass

            # Drain a few IMU motion frames if enabled
            if imu_enabled and imu_state is not None:
                drained = 0
                while drained < max(1, imu_max_drain):
                    fs = pipe.poll_for_frames()
                    if not fs:
                        break
                    for f in fs:
                        if f.is_motion_frame():
                            imu_state.update_from_motion_frame(f)
                            drained += 1
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
                    if mapper.is_not_obstacle(raw_label):
                        # A window, a shirt or a person's hand. Dropped here rather than mapped,
                        # because every bucket including unknown_obstacle stops the walker.
                        continue
                    mapped = mapper.map_label(raw_label)
                    bearing = bearing_from_bbox(
                        (x1, y1, x2, y2), W, cfg["bearing"])
                    # Option B: Prefer median depth from the lower band of the bbox;
                    # fall back to the original centre-focused estimate.
                    d_lb = median_depth_lower_band(
                        depth_m, x1, y1, x2, y2, band_frac=0.33, horiz_shrink=0.06)
                    d_gen = median_depth_in_box(
                        depth_m, x1, y1, x2, y2, shrink_ratio=0.06)
                    d_m = d_lb if d_lb is not None else d_gen
                    distance_method = (
                        "D455F_LOWER_BBOX_MEDIAN" if d_lb is not None
                        else ("D455F_BBOX_MEDIAN" if d_gen is not None else None)
                    )
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
                        "distance_method": distance_method,
                        "distance_bin": d_bin,
                        "bearing": bearing
                    }
                    objs.append(obj)
                    if mapped.ontology_class == "hazard":
                        hazards.append(obj["id"])

            out = InferPacket(ts_ms=ts, color=color_np, depth_m=depth_m,
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
    # COCO, after `yolov8n-oiv7.pt` was tried and reverted on 22 August 2026. Open Images has the
    # vocabulary COCO lacks, including Stairs, and swapping to it made `hazard` reachable for the
    # first time. It also stopped finding furniture. Run over the same 1412 recorded frames, COCO
    # found bed 453 times, tv 294, laptop 243 and chair 69, and Open Images found none of them: its
    # detections were `Man`, `Human face`, `Clothing` and `Glasses`, and 926 frames of 1412 came
    # back empty against 349. Open Images nano scores 18.4 mAP against COCO nano's 37.3, spread over
    # 601 classes instead of 80.
    #
    # A caption cannot describe what the detector does not report, so the swap traded the furniture
    # a walker has to avoid for a hazard class that had never fired in a real run. Section 10.12 of
    # `HDSG_VERIFIED_GENERATION_POLICY.md` records the measurement. How the hazard class is detected
    # returns to being open, and is settled by the lab captures rather than by a change of weights.
    ap.add_argument("--model", default="yolov8n.pt")
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
    ap.add_argument("--imu", action="store_true",
                    help="Enable IMU (gyro/accel) ingestion if the device supports it")
    ap.add_argument("--imu_max_drain", type=int, default=IMU_MAX_DRAIN_PER_LOOP,
                    help="Max motion frames to drain per loop to update IMU state")
    ap.add_argument("--imu_yaw_axis", choices=["x", "y", "z"], default="z",
                    help="Which IMU gyro axis to treat as yaw (depends on camera mounting)")
    ap.add_argument("--imu_yaw_sign", type=int, choices=[-1, 1], default=1,
                    help="+1 or -1 to flip yaw direction if needed")
    # LLM integration
    ap.add_argument("--llm", action="store_true",
                    help="Enable LLM advisory via llama.cpp server")
    ap.add_argument("--llm_endpoint", default="http://localhost:8080",
                    help="llama.cpp server endpoint")
    ap.add_argument("--llm_hz", type=float, default=2.0,
                    help="Call LLM at this rate (Hz)")
    ap.add_argument("--llm_model_name", default="qwen2.5-7b-instruct",
                    help="Model name label for server")
    ap.add_argument("--llm_system", default=str(ROOT / "models" / "llm" /
                    "qwen2.5-7b-instruct" / "prompt_system.txt"), help="System prompt file path")
    ap.add_argument("--llm_grammar", default=str(ROOT / "models" / "llm" /
                    "qwen2.5-7b-instruct" / "grammar.gbnf"), help="GBNF grammar file path")
    ap.add_argument("--llm_no_grammar", action="store_true",
                    help="Disable grammar enforcement")
    ap.add_argument("--llm_max_tokens", type=int, default=128)
    ap.add_argument("--llm_temperature", type=float, default=0.2)
    ap.add_argument("--llm_top_p", type=float, default=0.9)
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
    imu_enabled = False
    if args.imu:
        try:
            rs_cfg.enable_stream(rs.stream.gyro)
            rs_cfg.enable_stream(rs.stream.accel)
            imu_enabled = True
        except Exception:
            imu_enabled = False
    profile = pipe.start(rs_cfg)
    depth_scale = float(profile.get_device(
    ).first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)
    print(f"[smart_walker] depth_scale = {depth_scale:.6f} m/unit")
    print("Hold W/A/S/D to issue intent. SPACE=idle. P=print JSON. Q=quit.")
    if imu_enabled:
        print("IMU enabled: summarizing gyro/accel into JSON motion block.")
    else:
        print("IMU disabled or unavailable. Run with --imu on D435i/D455 to enable motion cues.")

    model = YOLO(args.model)

    # Queues & workers
    cap_q = Queue(maxsize=1)
    inf_in = Queue(maxsize=1)
    inf_out = Queue(maxsize=1)
    stop_evt = threading.Event()

    imu_state = IMUState.create() if imu_enabled else None
    threading.Thread(target=capture_thread, args=(
        pipe, align, depth_scale, cap_q, stop_evt, imu_enabled, imu_state, args.imu_max_drain), daemon=True).start()
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

    # LLM worker state
    llm_state: Dict[str, Any] = {}
    llm_lock = threading.Lock()
    if args.llm:
        threading.Thread(
            target=llm_worker,
            args=(
                snapshot,
                llm_state,
                llm_lock,
                stop_evt,
                args.llm_hz,
                args.llm_endpoint,
                Path(args.llm_system),
                Path(args.llm_grammar) if not args.llm_no_grammar else None,
                args.llm_model_name,
                args.llm_temperature,
                args.llm_top_p,
                args.llm_max_tokens,
                args.llm_no_grammar,
            ),
            daemon=True,
        ).start()

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

    # Simple dialog state to support Y/N confirmation
    dialog = {
        "pending": False,
        "question": None,
        "proposed_dir": None,
        "since_ms": 0,
        "last_reply": None,  # 'yes' | 'no' | None
    }

    def _compute_clear_side(objs: List[Dict[str, Any]]) -> Optional[str]:
        """Return 'left' or 'right' with more clearance based on nearest object distance."""
        try:
            left_near = 0.0
            right_near = 0.0
            left_found = False
            right_found = False
            for o in objs:
                d = o.get("distance_m")
                b = o.get("bearing")
                if d is None or b is None:
                    continue
                side = str(b[0] if isinstance(b, (list, tuple)) else b).lower()
                if "left" in side:
                    left_near = d if not left_found else max(left_near, d)
                    left_found = True
                elif "right" in side:
                    right_near = d if not right_found else max(right_near, d)
                    right_found = True
            if not left_found and not right_found:
                return None
            if not left_found:
                return "right"
            if not right_found:
                return "left"
            return "left" if left_near >= right_near else "right"
        except Exception:
            return None

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

            # Dialogue: if user keeps pushing into STOP, propose a side and ask for Y/N
            try:
                if active and last_decision.upper() == "STOP" and s.level_dir != "idle":
                    if not dialog["pending"]:
                        side = _compute_clear_side(last_objects)
                        if side is not None:
                            dialog.update({
                                "pending": True,
                                "question": f"Path blocked. Take {side}? [Y/N]",
                                "proposed_dir": side,
                                "since_ms": now_ms,
                                "last_reply": None,
                            })
                # Capture Y/N answers (edge)
                if dialog["pending"] and (s.yes_edge or s.no_edge):
                    dialog["last_reply"] = "yes" if s.yes_edge else "no"
                    # Keep pending so LLM can see reply for one cycle; we'll clear it later
                # Auto-clear stale dialog after 6 seconds without interaction
                if dialog["pending"] and now_ms - dialog.get("since_ms", now_ms) > 6000:
                    dialog.update(
                        {"pending": False, "question": None, "proposed_dir": None, "last_reply": None})
            except Exception:
                pass

            # Render with latest known state
            if latest_color is not None:
                vis = draw_overlay(
                    latest_color, last_objects, last_risk, last_caption, intent_line, last_decision)
                # Optional LLM advisory line
                if args.llm:
                    try:
                        with llm_lock:
                            adv = llm_state.get("advisory")
                            reason = llm_state.get("reason")
                        if adv or reason:
                            # extend header area
                            cv2.rectangle(
                                vis, (0, 90), (vis.shape[1], 120), (0, 0, 0), -1)
                            txt = f"LLM: {adv or ''} - {reason or ''}"
                            cv2.putText(
                                vis, txt[:110], (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 2, cv2.LINE_AA)
                    except Exception:
                        pass
                # Dialog prompt line (below LLM)
                try:
                    if dialog["pending"] and dialog.get("question"):
                        cv2.rectangle(
                            vis, (0, 120), (vis.shape[1], 148), (0, 0, 0), -1)
                        qtxt = dialog["question"]
                        if dialog.get("last_reply"):
                            qtxt += f"  (You pressed {dialog['last_reply'].upper()})"
                            # clear after showing reply once
                            dialog.update(
                                {"pending": False, "question": None, "proposed_dir": None})
                        cv2.putText(
                            vis, qtxt[:120], (10, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 2, cv2.LINE_AA)
                except Exception:
                    pass
                cv2.imshow(WIN, vis)

            # --------- Build JSON snapshot every frame (for streamer & optional prints) ----------
            motion_block = None
            try:
                if imu_enabled and imu_state is not None:
                    with imu_state.lock:
                        ts_imu = imu_state.ts_ms
                        if ts_imu > 0:
                            axis_map = {"x": 0, "y": 1, "z": 2}
                            yaw_idx = axis_map.get(args.imu_yaw_axis, 2)
                            yaw_rate_deg_s = float(
                                args.imu_yaw_sign * imu_state.gyro[yaw_idx] * (180.0 / math.pi))
                            accel_norm_g = float(
                                np.linalg.norm(imu_state.accel) / 9.80665)
                            motion_block = {
                                "ts_ms": ts_imu,
                                "yaw_rate_deg_s": round(yaw_rate_deg_s, 2),
                                "accel_norm_g": round(accel_norm_g, 2),
                                "gyro_rad_s": [float(imu_state.gyro[0]), float(imu_state.gyro[1]), float(imu_state.gyro[2])],
                                "accel_m_s2": [float(imu_state.accel[0]), float(imu_state.accel[1]), float(imu_state.accel[2])]
                            }
            except Exception:
                motion_block = None

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
                "objects": last_objects,
                "motion": motion_block,
                "llm": (lambda: (llm_state.copy() if args.llm else None))()
            }
            # attach dialog state so LLM gets user reply context
            payload["dialog"] = dialog.copy()
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
