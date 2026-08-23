"""The perception layer: the camera, the detector, and the measurements taken from a frame.

Imported by `realsense_vlm_on_change_qwen.py` as `sw`. Nothing here decides what the walker does or
writes anything a person reads. It produces the measured facts that `hdsg_runtime` then reasons over:
sector clearances, an object's distance and bearing, the detector's labels mapped onto the ontology,
and the keyboard and IMU state.

This file also carried a standalone walker prototype until 24 August 2026, 660 lines reachable only
by running the file directly. It held a second decision policy returning GO, SLOW or STOP, a second
caption writer producing sentences such as "Going forward as you intended.", its own model client and
its own display. Nothing imported any of it, and it had drifted: its slow-down threshold was read
from `depth.metric_bins_m.near`, a presentation band, which is the same fault removed from
`_multi_near_count` on 23 August. Two decision policies in the file the live system imports invite a
reader to take the wrong one for what the walker does, and Chapter 3 describes one. Removed. The
prototype survives in the history and, almost identically, in `scripts/archive/only_realsense.py`.
"""

from __future__ import annotations
import ctypes
import os
import time
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

# ----------------------------- Perf niceties -----------------------------
try:
    # Imported for the side effect alone. cuDNN picks the fastest convolution algorithm for a fixed
    # input size on the first call and reuses it, which the detector benefits from because every
    # frame is the same shape. A HAS_TORCH flag was set beside it and read nowhere.
    import torch
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

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

# How many IMU samples the capture thread may take from the queue in one pass. The motion stream
# runs far faster than the frame loop, so without a cap a backlog is drained in full and the loop
# stalls behind it.
IMU_MAX_DRAIN_PER_LOOP = 8


def load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------- Key Handling ------------------------------
user32 = ctypes.windll.user32 if os.name == "nt" else None
# The keys the walker reads. Y, N and P were removed on 24 August 2026: Y and N were the prototype's
# yes and no confirmation and P its hold-to-print, and nothing has read any of the three since the
# canonical client replaced that interaction.
VK = {'W': 0x57, 'A': 0x41, 'S': 0x53, 'D': 0x44,
      'SPACE': 0x20, 'Q': 0x51, 'M': 0x4D, 'R': 0x52}


def key_down(vk: int) -> bool:
    if user32 is None:
        return False
    return (user32.GetAsyncKeyState(vk) & 0x8000) != 0


@dataclass
class IntentState:
    """What the keyboard has been doing, as the client reads it once per frame.

    A direction is reported as an edge and not as a level: `last_edge_dir` is the last direction
    tapped and `last_press_ms` says when, and the client treats a changed timestamp as a fresh
    expression of intent. A `level_dir` holding the direction currently held down sat here until
    24 August 2026 and was read by nothing, as were `p_is_down` and the yes and no edges.
    """

    last_edge_dir: str = "idle"
    last_press_ms: int = 0
    quit_requested: bool = False
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

    # The direction keys in the order they settle a tie, and the intent each expresses. Two pressed
    # within the same tick is a contradiction rather than two requests, so one wins.
    DIRECTION_KEYS = (("W", "forward"), ("S", "backward"), ("A", "left"), ("D", "right"),
                      ("SPACE", "idle"))

    def _loop(self):
        while not self.stop_evt.is_set():
            now_ms = int(time.time() * 1000)
            edges = self.edges()
            for name, direction in self.DIRECTION_KEYS:
                if edges[name]:
                    self._set_edge(direction, now_ms)
                    break
            with self.lock:
                # Independent of the direction keys and of each other. Asking for more detail while
                # expressing an intent is an ordinary thing to do.
                if edges["M"]:
                    self.state.more_detail_edge = True
                if edges["R"]:
                    self.state.reassess_edge = True
                if edges["Q"]:
                    self.state.quit_requested = True
            time.sleep(self.dt)

    def edges(self) -> Dict[str, bool]:
        """Which keys have just gone down, reading every one of them.

        `_edge` records the key's previous state as a side effect, so the keys have to be read
        together. They were tested in a short-circuiting chain until 24 August 2026: a tick in which
        a direction key fired left every key after it in the chain holding a stale state, and the
        next press of one was then read as no change and lost. It took a release and a re-press
        straddling one 8 ms tick, so it cost an occasional button press rather than producing a
        wrong one.
        """
        return {name: self._edge(vk, name) for name, vk in VK.items()}

    def _set_edge(self, d, tms):
        with self.lock:
            self.state.last_edge_dir = d
            self.state.last_press_ms = tms

    def snapshot(self) -> IntentState:
        with self.lock:
            snapshot = IntentState(**self.state.__dict__)
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

    A `prompts` list per bucket was read here until 24 August 2026, giving a word the bucket without
    a canonical name so that the detector's own word survived into the caption. Since the ontology
    became generated from `model.names` on 22 August 2026 no bucket carries one, and none can: the
    generator emits `canonical` alone and `--check` fails on a hand edit. The branch could not be
    reached and the loader read a key no file holds.

    The ontology's `not_obstacles` list is read here but not applied by `map_label`, which has no
    way to say "no object". `is_not_obstacle` reports it separately and the detection loop drops the
    detection before it becomes a fact.
    """

    def __init__(self, ontology_path: Path):
        cfg = load_yaml(ontology_path)
        self.ontology_buckets: dict[str, str] = {}
        self.synonyms = {k.lower(): v.lower()
                         for k, v in cfg.get("synonyms_to_canonical", {}).items()}
        self.not_obstacles = {str(name).strip().lower()
                              for name in cfg.get("not_obstacles", []) or ()}
        for bucket in cfg["ontology"]:
            ont = bucket["name"]
            for c in bucket.get("canonical", []):
                self.ontology_buckets[c.lower()] = ont

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


def median_depth_in_box(depth_m: np.ndarray, x1, y1, x2, y2, shrink_ratio=0.06,
                        min_coverage: float = 0.0):
    """The median depth inside a region, or None where the region has no usable reading.

    The region is shrunk by `shrink_ratio` on every side first. An edge picks up whatever lies
    behind the thing being measured, so the margin is excluded deliberately.

    `min_coverage` is the share of the region that must carry a reading before a measurement is
    reported. A median does not care how many samples it has, so without a floor a region in which
    almost nothing could be measured still yields a confident number: in the archived run of
    22 August 2026 a sector strip of 32,470 pixels carried 21 readings, and the median of those 21,
    54.65 m, was reported as the clearance and read as CLEAR.

    It defaults to zero, so a caller that has not chosen a floor is unaffected. The sector strips
    pass `sector.min_measured_fraction`. Object boxes do not, there being seven object detections in
    the archive, too few to choose a number from.

    Two fallbacks stood below this until 24 August 2026 and both are gone.

    The first read the single centre pixel. It sits inside the shrunk region at any shrink under a
    half, so it could only be reached once that region was known to hold nothing, and it was then
    one of those same absent readings. It could not fire.

    The second sampled the unshrunk region, which is to say the margin the shrink had just excluded,
    and returned its median as the measurement. That one did fire. Over the 1,480 archived frames of
    22 August 2026 it decided a sector clearance 12 times, returning 8.29 m and 65.535 m among
    others, and every such reading resolved to CLEAR. Reporting the background as the foreground's
    distance is worse than reporting nothing, because the sector logic treats an absent reading as
    not clear and treats a large number as room to move.
    """
    H, W = depth_m.shape[:2]
    dx, dy = shrink_ratio * (x2 - x1), shrink_ratio * (y2 - y1)
    rx1, ry1, rx2, ry2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    rx1, ry1, rx2, ry2 = _safe_int_bounds(rx1, ry1, rx2, ry2, W, H)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    patch = depth_m[ry1:ry2, rx1:rx2]
    if min_coverage > 0.0:
        measured = float(np.count_nonzero(np.isfinite(patch) & (patch > 0))) / float(patch.size)
        if measured < min_coverage:
            return None
    return _median_nonzero(patch)


def median_depth_lower_band(
    depth_m: np.ndarray,
    x1,
    y1,
    x2,
    y2,
    band_frac: float = 0.33,
    horiz_shrink: float = 0.06,
) -> Optional[float]:
    """An object's distance, measured from the bottom band of its box alone.

    The bottom of a detection is where the object meets the floor, or is otherwise its nearest
    visible part, so measuring there rather than over the whole box gives the shorter and safer
    estimate. `band_frac` is the share of the box height used, and `horiz_shrink` trims the sides
    for the reason `median_depth_in_box` shrinks its region.

    A sparse random sample stood below this until 24 August 2026, described as a fallback for a band
    holding no depth. It could not run: `_median_nonzero` returns None only when the patch has no
    reading, and the sample then asked the same patch for its readings.
    """
    H, W = depth_m.shape[:2]
    band_h = max(1.0, band_frac * max(1.0, float(y2 - y1)))
    dx = horiz_shrink * (x2 - x1)
    bx1, by1i, bx2, by2i = _safe_int_bounds(x1 + dx, y2 - band_h, x2 - dx, y2, W, H)
    if bx2 <= bx1 or by2i <= by1i:
        return None
    return _median_nonzero(depth_m[by1i:by2i, bx1:bx2])

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
            # A depth pixel holds millimetres, and the sensor has two ways of saying it has no
            # reading: zero, which every consumer already treats as absent, and the 16-bit ceiling.
            # The second was read as a distance of 65.535 m until 24 August 2026, which is fifty
            # times anything the D455f measures and always resolved to CLEAR.
            #
            # Measured over the 1,480 archived frames of 22 August 2026: 36 per cent of frames carry
            # at least one such pixel, and taking the median of a sector strip left 90 of 4,440
            # sector readings with the wrong status. Every one of the 90 erred the same way, calling
            # a strip clear when it was constrained or unmeasurable, once reporting clear where the
            # true clearance was 0.93 m. It also inflated the depth-coverage figure by up to 14
            # points on the frames where that figure matters most.
            #
            # Voided here, where the sensor's numbers become metres, rather than in each consumer:
            # the sector clearances, the object distances and the coverage figure all read from this
            # array. A voided pixel is indistinguishable from an unmeasurable one, and the sector
            # logic already treats a strip it cannot measure as not clear.
            depth_u16 = np.where(depth_u16 == 65535, 0, depth_u16)
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

