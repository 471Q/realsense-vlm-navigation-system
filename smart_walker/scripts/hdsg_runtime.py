"""Runtime contracts and release controls for the HDSG smart-walker prototype.

The generative layer composes its caption in its own words and declares the values it stated, which
`hdsg_composed` parses and checks. This module owns the deterministic layer: the fact packet, the
authority decision, the restricted prompt packet, and the release object in every case where no
caption is accepted.

An earlier design had the model select among controlled sentences the runtime had already written,
substituting measurements into placeholders. It was removed on 21 August 2026. Assembling the
sentence from a fixed predicate table left the model choosing between "has limited clearance" and
"is constrained", so the released wording was the runtime's rather than the model's, and the arm
measured a vision-language model doing clerical work. `hdsg.vlm_candidate.v1` and its grammar went
with it. The identifier is retained in the release and prompt packet enumerations because archives
recorded before that date carry it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Deque, Iterable, Mapping, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


# SOFTWARE_VERSION and RULE_SET_VERSION were literals here, "hdsg-v1" and "rules-v1", removed on
# 23 August 2026. Neither changed once through any rewrite, so a record carrying them named nothing
# that distinguished it from any other record, in a block whose whole purpose is that its values can
# be believed. `code_version` does their job from git rather than from a literal.

# Which rows of the frame the sector clearances are measured across: below the horizon, above the
# immediate foreground where the walker's own frame intrudes. Read from `sector` in
# `config/pipeline.yaml`, which is where a retune belongs. These are the fallbacks used when no
# configuration reaches the caller, and `tests/test_runtime_authority.py` fails if they disagree
# with the file. The pair was written out three times until 23 August 2026, here, in
# `valid_depth_fraction` and inside `compute_lane_state`, with nothing comparing them.
SECTOR_BAND_TOP_FRACTION = 0.55
SECTOR_BAND_BOTTOM_FRACTION = 0.95
# The fallback for `sector.min_measured_fraction`, used only where the configuration cannot be read.
# A strip measured over less of its area than this reports no clearance at all.
SECTOR_MIN_MEASURED_FRACTION = 0.05


_CODE_VERSION: Optional[dict] = None


def code_version() -> dict:
    """Identifies the code that produced a record.

    Resolved once and cached. It shells out to git three times, and a packet is built on every
    event: uncached it added ten seconds to the test suite and would put three subprocess calls on
    the walker's own event path. Caching is also correct rather than merely fast, since the code
    that produced a run cannot change while the run is in progress.

    `configuration_hash` covers `pipeline.yaml` and the other configuration files and none of the
    Python. Two literals, `software_version` "hdsg-v1" and `rule_set_version` "rules-v1", claimed to
    and did not: neither changed once through any rewrite, and both were removed on 23 August 2026. On 23 August 2026 the caution rule stopped counting distance bands and started
    counting metres, and the object band stopped being trusted and started being derived: two changes
    to what the walker does, in code, leaving every field in the record identical. A run from before
    and a run from after could not be told apart from their own evidence.

    The same defect was found for the model on 22 August, where the header recorded the name typed on
    the command line rather than the weights the endpoint had loaded, and was fixed by asking the
    endpoint. This asks git.

    `dirty` is true where the working tree carries uncommitted changes, which makes the commit
    identifier necessary but not sufficient to reproduce the run. Recording it is the point: a run
    made from an edited tree should say so rather than name a commit that does not contain the edit.
    Every field is None where git is unavailable, which is honest about not knowing rather than
    silently naming nothing.
    """
    global _CODE_VERSION
    if _CODE_VERSION is not None:
        return dict(_CODE_VERSION)

    import subprocess

    def git(*arguments):
        try:
            result = subprocess.run(
                ("git", *arguments), cwd=str(Path(__file__).resolve().parent),
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    _CODE_VERSION = {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }
    return dict(_CODE_VERSION)


# hdsg.schemas.v5. The record contracts have not changed since v2. v4 removed the templated
# candidate from the set and v5 revived RG_SUBJECT_MISMATCH for the composed attribution check, so
# both are changes to what the set describes rather than to the shape of any record. The retired
# templated candidate identifier survives in the release and prompt packet enumerations so that
# archives recorded before 21 August 2026 still validate, and nothing here emits it.
FACT_PACKET_SCHEMA = "hdsg.fact_packet.v2"
PROMPT_PACKET_SCHEMA = "hdsg.prompt_packet.v2"
CAPTION_SCHEMA = "hdsg.vlm_caption.v1"
RELEASE_SCHEMA = "hdsg.release.v2"

# The motion thresholds, in metres. One home each.
#
# These are the fallback values, used when a caller supplies nothing. `config/pipeline.yaml` is the
# configuration home and the live entry script reads them from there and passes them in; the
# constants exist so that a caller without a config file, a test or an offline replay, agrees with
# the shipped configuration rather than inventing its own number.
#
# They were written as literals in three places until 23 August 2026. `determine_authority` used
# 1.50 and 2.0 as parameter defaults, `build_fact_packet` wrote the same two as literals into the
# thresholds record a run is audited by, and the sector clear distance was 1.8 on the command line
# while every test packet recorded 2.0. A record claiming to state how a run was configured must
# not carry a number the run did not use.
OBJECT_STOP_BELOW_M = 0.70
OBJECT_CAUTION_BELOW_M = 1.50
HAZARD_STOP_AT_OR_BELOW_M = 2.0
# 1.8 rather than 2.0 because every run in the archive was recorded at 1.8, the command line's
# default, and moving the configuration would put the shipped value at odds with the evidence
# already collected. The value itself is not settled; it awaits a measurement at the laboratory.
SECTOR_CLEAR_AT_OR_ABOVE_M = 1.8

SEVERITY = {"SAFE": 0, "CAUTION": 1, "STOP": 2}
SECTORS = ("LEFT", "CENTRE", "RIGHT")
PROFILE_LIMITS = {
    "AUTOMATIC": (3, 0, False),
    "MORE_DETAIL": (4, 2, True),
    "REASSESSMENT": (3, 2, True),
}

REASON_CODE_ORDER = (
    "RG_NO_INTENT_EXPRESSED",
    "RG_GENERATION_PENDING",
    "RG_MODEL_UNAVAILABLE",
    "RG_GENERATION_TIMEOUT",
    "RG_CONSTRAINT_FAILURE",
    "RG_STALE_CANDIDATE",
    "RG_IDENTITY_MISMATCH",
    "RG_PARSE_FAILURE",
    "RG_SCHEMA_FAILURE",
    "RG_PROFILE_LIMIT_EXCEEDED",
    "RG_FACT_REFERENCE_INVALID",
    "RG_MEASUREMENT_REFERENCE_INVALID",
    "RG_OBJECT_REFERENCE_INVALID",
    # A clause states a number without naming the fact the number was declared against. Placed with
    # the reference failures because that is what it is: the number resolves and the subject does
    # not. It was absent from this ordering until 24 August 2026, and an unlisted code sorts last,
    # so a caption failing this and anything else reported the other as its primary reason. It sorted
    # behind RG_INTERNAL_GATE_ERROR, which exists to be last.
    "RG_SUBJECT_MISMATCH",
    # A declared value disagrees with the measurement it names. The central failure mode of the
    # composed-caption design and the one code it adds beyond the frozen v1 enumeration, which
    # hdsg.schemas.v4 must therefore carry.
    "RG_STATED_VALUE_MISMATCH",
    # The caption states none of the measurements it was given. The second code beyond the frozen v1
    # enumeration, and hdsg.schemas.v4 must carry it alongside RG_STATED_VALUE_MISMATCH.
    "RG_NO_MEASUREMENT_STATED",
    "RG_ACTION_LANGUAGE_DETECTED",
    "RG_DIRECT_NUMBER_DETECTED",
    "RG_VISIBLE_TEXT_CONTENT_DETECTED",
    "RG_MODEL_COMMENTARY_DETECTED",
    "RG_UNAPPROVED_LANGUAGE_DETECTED",
    "RG_INTERNAL_GATE_ERROR",
)

ACTION_TEMPLATES = {
    ("PROCEED", "LEFT"): ("proceed_left.v1", "Continue towards the left."),
    ("PROCEED", "CENTRE"): ("proceed_centre.v1", "Continue forward."),
    ("PROCEED", "RIGHT"): ("proceed_right.v1", "Continue towards the right."),
    ("SLOW", "LEFT"): ("slow_left.v1", "Slow down and continue towards the left."),
    ("SLOW", "CENTRE"): ("slow_centre.v1", "Slow down and continue forward."),
    ("SLOW", "RIGHT"): ("slow_right.v1", "Slow down and continue towards the right."),
    ("REDIRECT", "LEFT"): ("redirect_left.v1", "Change direction and continue towards the left."),
    ("REDIRECT", "CENTRE"): ("redirect_centre.v1", "Change direction and continue through the centre."),
    ("REDIRECT", "RIGHT"): ("redirect_right.v1", "Change direction and continue towards the right."),
    ("STOP", "NONE"): ("stop.v1", "Stop."),
}

# The restriction order from HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md section 6. A
# move to a higher rank is an escalation and triggers generation; a move to a lower or
# equal rank does not. The single source of truth replaces two inline copies that
# previously existed in the entry script.
RESTRICTION_ORDER = {"PROCEED": 0, "SLOW": 1, "REDIRECT": 2, "STOP": 3}

# The reason-line placeholder shown while a generation request is in flight and the
# deterministic interaction state is GUIDANCE_ACTIVE. HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md
# section 3.
PENDING_PLACEHOLDER_TEXT = "Assessing the environment."

# The remaining expressions screen a composed caption and the labels attached to its visual
# observations. `hdsg_composed` applies them; nothing in this module does.
# Instruction verbs.
#
# This list was extended on 23 August 2026 and reverted the same day. The extension added the verbs
# of walking, and a second expression refusing any second-person pronoun, on the reasoning that a
# question carrying the person's own wording into the prompt might produce an instruction. The
# reasoning was never tested before it was built.
#
# `Experiment_Question_Compliance_Probe.md` then tested it. Qwen3-VL-4B wrote no instruction in any
# of 32 answers, including sixteen questions written to provoke one, so the extension searched for
# something the shipped model does not produce. Qwen2.5-VL-3B did fail three times, and what its
# three failures have in common is not a vocabulary: they state no measurement at all. That is the
# condition `no_measurement_stated` now checks, and it decides the same three cases without reading
# the words.
#
# The list is kept as it stood before the extension. It is partial by construction and no longer
# carries the guarantee.
ACTION_RE = re.compile(
    r"\b(?:go|move|turn|continue|proceed|stop|avoid|choose|reverse|reorient)\b|slow\s+down|take\s+the\s+(?:left|right)|head\s+(?:left|right)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|metres?|meters?|centimetres?|centimeters?)\b",
    re.IGNORECASE,
)
COMMENTARY_RE = re.compile(
    r"\b(?:model|prompt|system instruction|image|json|schema|candidate|validator|error|unable)\b",
    re.IGNORECASE,
)
# Reading out writing that appears in the scene is prohibited, so the check looks for a quoted span
# and for the phrases that introduce one.
#
# A bare apostrophe was treated as a quotation mark. Under the templated design a caption was
# assembled from approved clauses and never contained one, but a composed caption is ordinary
# English prose, so "the room's centre is open" was rejected as visible text. That is a rejection for
# a possessive rather than for a safety violation, and it would have inflated the measured rejection
# rate for a reason unrelated to what the rate is reported to measure. A single-quoted span is still
# caught, because its quotes are bounded by non-letters where a possessive apostrophe is not.
VISIBLE_TEXT_RE = re.compile(
    r"\b(?:the sign says|text says|written text|reads [\"'])"
    r"|\""
    r"|(?<![A-Za-z])'[^']{2,}'(?![A-Za-z])",
    re.IGNORECASE,
)
VISUAL_LABEL_PROHIBITED_RE = re.compile(
    r"\b(?:ignore|follow|instruction|instructions|command|request|prompt|caption|read|write|say)\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    """Returns the current UTC time in the schema-compatible representation."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _reject_json_constant(token: str):
    """Refuses the three bare tokens Python accepts in JSON and the format does not define."""
    raise ValueError(f"{token} is not a JSON value")


def _finite_float(token: str) -> float:
    """Converts a JSON number, refusing one whose digits overflow to infinity."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"{token[:24]}... is not a finite number")
    return value


def loads_strict(raw: str):
    """Parses a model reply, refusing any value that is not a finite number.

    Two ways in, and both end with a non-finite float in the telemetry. Python's decoder accepts the
    bare tokens NaN, Infinity and -Infinity, which JSON does not define, and its encoder writes them
    out again. Separately, a number whose digits exceed what a float can hold converts to infinity
    without complaint, so "1" followed by four hundred zeros and a fractional part parses as inf.

    Either leaves a log line Python reads back and a strict reader cannot, and that file is the
    evidence Chapter 5 rests on. A non-finite declared value also poisons the event's
    `mean_absolute_error_m`, and one such event turns an average across events into NaN or drops it,
    depending on the tool.

    The bare tokens are reachable only where no grammar constrains the reply: the `--unconstrained`
    diagnostic mode, and the C0 and C1 baselines, which carry no gate either. The overflow is
    reachable everywhere, the caption grammar placing no bound on the digits of a number. Both are
    refused here rather than only in the scorer, so a reply the parser accepted holds no value the
    arithmetic downstream cannot use.

    Raises `ValueError`, which `json.JSONDecodeError` subclasses, so a caller already handling a
    parse failure handles this one without change.
    """
    return json.loads(raw, parse_constant=_reject_json_constant, parse_float=_finite_float)


def sha256_text(value: str) -> str:
    """Returns a labelled SHA-256 digest for a text value."""
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Returns a labelled SHA-256 digest or a deterministic missing-file digest."""
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return sha256_text(f"missing:{path.as_posix()}")


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _bearing(value: Any) -> str:
    text = str(value or "").lower()
    if "left" in text:
        return "LEFT"
    if "right" in text:
        return "RIGHT"
    return "CENTRE"


# The distance bands, in metres, as half-open intervals. These are the same bands as
# `depth.metric_bins_m` in `config/pipeline.yaml`, which is where a retune belongs, and
# `tests/test_runtime_authority.py` fails if the two disagree. They are repeated here rather than
# read because `normalise_objects` is called before the configuration path reaches the packet
# builder, and a band a detector supplied is no longer trusted, so this module has to be able to
# work one out on its own.
DISTANCE_BANDS_M = (
    ("VERY_CLOSE", 0.0, 0.7),
    ("NEAR", 0.7, 1.5),
    ("MID", 1.5, 3.0),
    ("FAR", 3.0, 99.0),
)


def _distance_bin(distance: Optional[float]) -> str:
    """Derives the band from the measured distance.

    The detector's own label was taken and merely checked against a list of permitted words until
    23 August 2026, so the packet carried a measurement and a word describing that measurement with
    nothing comparing them. They agreed, both coming from the same reading a moment apart, and no
    disagreement appears in any archived record. The weakness was structural: a description of a
    measurement was accepted rather than worked out from it, which is the arrangement that let the
    ontology drift away from the detector.

    The word reaches the person. It is the object's `state` in the permitted facts, so a caption may
    describe an object as near, and nothing checked that against the metres.

    A distance outside every band, which the D455f does not produce, is UNKNOWN rather than the band
    meaning the most room.
    """
    if not isinstance(distance, (int, float)) or isinstance(distance, bool):
        return "UNKNOWN"
    value = float(distance)
    if value != value or value in (float("inf"), float("-inf")):
        return "UNKNOWN"
    for name, low, high in DISTANCE_BANDS_M:
        if low <= value < high:
            return name
    return "UNKNOWN"


@dataclass
class _TrackObservation:
    timestamp_ms: float
    centre_x: float
    centre_y: float
    depth_m: Optional[float]


@dataclass
class MotionTracker:
    """Classifies tracked detections using compensated image motion and depth change."""

    movement_threshold_m: float = 0.12
    stationary_threshold_m: float = 0.05
    confirmation_observations: int = 4
    lost_after_observations: int = 12
    horizontal_fov_deg: float = 87.0
    histories: dict[int, Deque[_TrackObservation]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=12))
    )
    motion_scores: dict[int, Deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=12))
    )
    moving_hits: dict[int, int] = field(default_factory=dict)
    stationary_hits: dict[int, int] = field(default_factory=dict)
    confirmed_states: dict[int, str] = field(default_factory=dict)
    missed: dict[int, int] = field(default_factory=dict)
    previous_gray: Optional[np.ndarray] = None

    def _global_motion(self, gray: np.ndarray) -> tuple[float, float, bool]:
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV and NumPy are required for movement classification.")
        if self.previous_gray is None or self.previous_gray.shape != gray.shape:
            self.previous_gray = gray
            return 0.0, 0.0, False
        points = cv2.goodFeaturesToTrack(
            self.previous_gray, maxCorners=180, qualityLevel=0.01, minDistance=12
        )
        if points is None or len(points) < 12:
            self.previous_gray = gray
            return 0.0, 0.0, False
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None)
        self.previous_gray = gray
        if nxt is None or status is None:
            return 0.0, 0.0, False
        keep = status.reshape(-1).astype(bool)
        if int(keep.sum()) < 12:
            return 0.0, 0.0, False
        displacement = nxt.reshape(-1, 2)[keep] - points.reshape(-1, 2)[keep]
        dx, dy = np.median(displacement, axis=0)
        return float(dx), float(dy), True

    def update(self, frame_bgr: np.ndarray, objects: list[dict], timestamp_ms: float) -> list[dict]:
        """Adds movement fields to detections while retaining the original records."""
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV and NumPy are required for movement classification.")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        global_dx, global_dy, compensated = self._global_motion(gray)
        image_width = max(1, int(frame_bgr.shape[1]))
        focal_px = image_width / (2.0 * math.tan(math.radians(self.horizontal_fov_deg) / 2.0))
        seen: set[int] = set()
        enriched: list[dict] = []

        for index, source in enumerate(objects):
            item = dict(source)
            track_id = item.get("id")
            if not isinstance(track_id, int):
                track_id = index
            seen.add(track_id)
            x1, y1, x2, y2 = [float(v) for v in item.get("bbox_xyxy", [0, 0, 0, 0])]
            current = _TrackObservation(
                timestamp_ms=float(timestamp_ms),
                centre_x=(x1 + x2) / 2.0,
                centre_y=(y1 + y2) / 2.0,
                depth_m=_finite(item.get("distance_m")),
            )
            history = self.histories[track_id]
            score_m = 0.0
            state = "UNCONFIRMED"
            confidence: Optional[float] = None
            reasons: list[str] = []

            if compensated and history:
                previous = history[-1]
                dx_px = current.centre_x - previous.centre_x - global_dx
                dy_px = current.centre_y - previous.centre_y - global_dy
                reference_depth = current.depth_m or previous.depth_m
                lateral_m = 0.0 if reference_depth is None else math.hypot(dx_px, dy_px) * reference_depth / focal_px
                radial_m = 0.0
                if current.depth_m is not None and previous.depth_m is not None:
                    radial_m = abs(current.depth_m - previous.depth_m)
                score_m = math.hypot(lateral_m, radial_m)
                reasons.append("MOTION_CAMERA_COMPENSATED")
            elif not compensated:
                reasons.append("MOTION_COMPENSATION_UNAVAILABLE")

            history.append(current)
            self.motion_scores[track_id].append(score_m)
            self.missed[track_id] = 0
            previous_state = self.confirmed_states.get(track_id, "UNCONFIRMED")
            # Every tracked object is judged on the evidence, whatever class it belongs to.
            #
            # Motion used to require an ontology class of agent or rolling_obstacle, and an object
            # outside those two was not merely left unclassified: after enough observations it was
            # recorded as STATIONARY with a confidence of 1.0. A chair being pushed towards the user
            # was therefore asserted, with complete certainty, not to be moving, on the strength of
            # a prior about which kinds of thing move rather than any measurement. The objects that
            # matter are the ones behaving unexpectedly, so that prior was wrong in the direction
            # that costs most.
            #
            # What keeps sensor noise out is the evidence itself, and it is unchanged: optical flow
            # must have compensated for the camera's own motion, enough observations must have
            # accumulated, the displacement is converted to metres through the measured depth, it
            # must exceed movement_threshold_m, and it must persist for confirmation_observations
            # consecutive frames. If that is too loose it is a threshold to measure and tune, not a
            # class list to guess from.
            if compensated and len(history) >= self.confirmation_observations:
                recent = list(history)[-self.confirmation_observations:]
                recent_scores = list(self.motion_scores[track_id])[-(self.confirmation_observations - 1):]
                accumulated_motion = sum(recent_scores)
                depth_values = [entry.depth_m for entry in recent if entry.depth_m is not None]
                depth_span = max(depth_values) - min(depth_values) if len(depth_values) >= 2 else 0.0
                if accumulated_motion >= self.movement_threshold_m or depth_span >= self.movement_threshold_m:
                    self.moving_hits[track_id] = self.moving_hits.get(track_id, 0) + 1
                    self.stationary_hits[track_id] = 0
                    if self.moving_hits[track_id] >= self.confirmation_observations:
                        state = "MOVING"
                        confidence = min(1.0, max(accumulated_motion, depth_span) / max(self.movement_threshold_m, 1e-6))
                        self.confirmed_states[track_id] = state
                    else:
                        state = previous_state if previous_state == "MOVING" else "UNCONFIRMED"
                        reasons.append("MOTION_PERSISTENCE_INSUFFICIENT")
                    reasons.append("MOTION_THRESHOLD_EXCEEDED")
                elif accumulated_motion <= self.stationary_threshold_m and depth_span <= self.stationary_threshold_m:
                    self.stationary_hits[track_id] = self.stationary_hits.get(track_id, 0) + 1
                    self.moving_hits[track_id] = 0
                    if self.stationary_hits[track_id] >= self.confirmation_observations:
                        state = "STATIONARY"
                        confidence = min(1.0, 1.0 - max(accumulated_motion, depth_span) / max(self.movement_threshold_m, 1e-6))
                        self.confirmed_states[track_id] = state
                    else:
                        state = previous_state if previous_state == "MOVING" else "UNCONFIRMED"
                        reasons.append("MOTION_PERSISTENCE_INSUFFICIENT")
                    reasons.append("MOTION_WITHIN_STATIONARY_TOLERANCE")
                else:
                    state = previous_state if previous_state == "MOVING" else "UNCONFIRMED"
                    self.moving_hits[track_id] = 0
                    self.stationary_hits[track_id] = 0
                    reasons.append("MOTION_CLASSIFICATION_AMBIGUOUS")
            else:
                self.moving_hits[track_id] = 0
                self.stationary_hits[track_id] = 0
                reasons.append("MOTION_HISTORY_INSUFFICIENT")

            item.update({
                "track_id": track_id,
                "motion_state": state,
                "motion_confidence": None if confidence is None else round(confidence, 3),
                "motion_reason_codes": sorted(set(reasons)),
                "camera_motion_compensated": bool(compensated),
                "display_bounding_box": state == "MOVING",
            })
            enriched.append(item)

        for track_id in list(self.histories):
            if track_id in seen:
                continue
            self.missed[track_id] = self.missed.get(track_id, 0) + 1
            if self.missed[track_id] >= self.lost_after_observations:
                self.histories.pop(track_id, None)
                self.motion_scores.pop(track_id, None)
                self.moving_hits.pop(track_id, None)
                self.stationary_hits.pop(track_id, None)
                self.confirmed_states.pop(track_id, None)
                self.missed.pop(track_id, None)
        return enriched


def normalise_objects(objects: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Converts detector records into Full Fact Packet object records."""
    result: list[dict] = []
    for detection_id, source in enumerate(objects):
        distance = _finite(source.get("distance_m"))
        track_id = source.get("track_id", source.get("id"))
        if not isinstance(track_id, int):
            track_id = None
        stable_id = track_id if track_id is not None else detection_id
        canonical = source.get("canonical_class") or source.get("raw_label")
        motion_state = str(source.get("motion_state") or "UNCONFIRMED").upper()
        if motion_state not in {"UNCONFIRMED", "STATIONARY", "MOVING"}:
            motion_state = "UNCONFIRMED"
        result.append({
            "fact_id": f"object:{stable_id}",
            "detection_id": int(source.get("id", detection_id)),
            "track_id": track_id,
            "raw_label": str(source.get("raw_label") or "object"),
            "canonical_label": None if canonical is None else str(canonical),
            "ontology_class": None if source.get("ontology_class") is None else str(source.get("ontology_class")),
            "confidence": max(0.0, min(1.0, float(source.get("conf", 0.0)))),
            "bbox_xyxy": [max(0.0, float(v)) for v in source.get("bbox_xyxy", [0, 0, 0, 0])],
            "bearing": _bearing(source.get("bearing")),
            "distance_m": None if distance is None else round(distance, 3),
            "distance_valid": distance is not None,
            "distance_method": (
                str(source.get("distance_method"))
                if distance is not None and source.get("distance_method") in {
                    "D455F_LOWER_BBOX_MEDIAN", "D455F_BBOX_MEDIAN"
                }
                else ("D455F_LOWER_BBOX_MEDIAN" if distance is not None else None)
            ),
            "distance_bin": _distance_bin(distance),
            "grounding": "YOLO_D455F" if distance is not None else "YOLO_ONLY",
            "is_hazard": str(source.get("ontology_class") or "").lower() == "hazard",
            "motion_state": motion_state,
            "motion_confidence": source.get("motion_confidence"),
            "motion_reason_codes": list(source.get("motion_reason_codes") or []),
            "camera_motion_compensated": bool(source.get("camera_motion_compensated", False)),
            "display_bounding_box": motion_state == "MOVING",
        })
    return result


def sectors_from_lane_state(lane_state: Optional[Mapping[str, Any]]) -> dict[str, dict]:
    """Converts the three lane readings into authoritative sector facts."""
    depths = list((lane_state or {}).get("depths") or [])
    statuses = list((lane_state or {}).get("status") or [])
    output: dict[str, dict] = {}
    for index, sector in enumerate(("left", "centre", "right")):
        distance = _finite(depths[index]) if index < len(depths) else None
        status = str(statuses[index]).upper() if index < len(statuses) else "UNKNOWN"
        if distance is None or status not in {"CLEAR", "CONSTRAINED", "BLOCKED"}:
            distance = None
            status = "UNKNOWN"
        output[sector] = {
            "fact_id": f"sector:{sector}",
            "clearance_m": None if distance is None else round(distance, 3),
            "valid": distance is not None,
            "invalid_reason_codes": [] if distance is not None else ["DEPTH_SECTOR_INVALID"],
            "status": status,
        }
    return output


def _worst_sector(sectors: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    ranking = {"UNKNOWN": 4, "BLOCKED": 3, "CONSTRAINED": 2, "CLEAR": 1}
    ordered = sorted(
        sectors.items(),
        key=lambda pair: (
            -ranking.get(str(pair[1].get("status")), 4),
            pair[1].get("clearance_m") if pair[1].get("clearance_m") is not None else -1.0,
        ),
    )
    return ordered[0][0].upper() if ordered else None


def determine_authority(
    intent: str,
    object_advisory: str,
    sectors: Mapping[str, Mapping[str, Any]],
    objects: Optional[Iterable[Mapping[str, Any]]] = None,
    previous_selected_sector: Optional[str] = None,
    sector_choice_tolerance_m: float = 0.10,
    object_stop_below_m: float = OBJECT_STOP_BELOW_M,
    object_caution_below_m: float = OBJECT_CAUTION_BELOW_M,
    hazard_stop_at_or_below_m: float = HAZARD_STOP_AT_OR_BELOW_M,
) -> dict:
    """Applies the approved finite deterministic action and sector-selection policy."""
    intent = str(intent or "NONE").upper()
    object_advisory = str(object_advisory or "STOP").upper()
    if object_advisory not in SEVERITY:
        object_advisory = "STOP"
    statuses = {name.upper(): str(value.get("status") or "UNKNOWN").upper() for name, value in sectors.items()}
    effective_statuses = dict(statuses)
    binding_object_by_sector: dict[str, Mapping[str, Any]] = {}
    for item in objects or []:
        distance = _finite(item.get("distance_m"))
        if distance is None:
            continue
        sector = _bearing(item.get("bearing"))
        object_status = None
        if distance < object_stop_below_m or (item.get("is_hazard") and distance <= hazard_stop_at_or_below_m):
            object_status = "BLOCKED"
        elif distance < object_caution_below_m:
            object_status = "CONSTRAINED"
        if object_status is None:
            continue
        current = binding_object_by_sector.get(sector)
        if current is None or distance < float(current.get("distance_m", float("inf"))):
            binding_object_by_sector[sector] = item
        order = {"CLEAR": 0, "CONSTRAINED": 1, "BLOCKED": 2, "UNKNOWN": 3}
        if order[object_status] > order.get(effective_statuses.get(sector, "UNKNOWN"), 3):
            effective_statuses[sector] = object_status
        elif effective_statuses.get(sector) == "CLEAR":
            effective_statuses[sector] = object_status
    clear = [name for name in SECTORS if effective_statuses.get(name) == "CLEAR"]
    sector_advisory = "SAFE" if len(clear) == 3 else ("STOP" if not clear else "CAUTION")
    scene_advisory = max((object_advisory, sector_advisory), key=lambda item: SEVERITY[item])
    intended = {"FORWARD": "CENTRE", "LEFT": "LEFT", "RIGHT": "RIGHT"}.get(intent)
    rules: list[dict] = []

    def binding(fact_ids: list[str], source: str) -> dict:
        """Names the measured elements that produced this decision, primary first.

        **Why naming the cause is a separate requirement from telling the truth.** An output that
        names a real but non-causal fact is faithful and still misleading. Reporting a wall at
        1.2 metres when the system stopped for a chair at 0.5 metres is entailed by the Fact Packet,
        passes every check the gate applies to a measurement, and sends the person into the chair.
        The transparency metric therefore asks which element caused the behaviour, not only whether
        each statement is true.

        `accepted_fact_ids` carries every element a caption could name and be equally correct: two
        objects at the same distance, or a sector tied with the object standing in it. Scoring
        accepts any of them rather than privileging an arbitrary one.

        This paragraph was written for `identify_binding_fact` in the canonical client, a first
        implementation of the same idea that nothing called and that was deleted on 25 August 2026.
        It is kept here, beside the code that does the work, because the argument is about the
        metric rather than about either implementation.
        """
        return {
            "primary_fact_id": fact_ids[0] if fact_ids else None,
            "accepted_fact_ids": fact_ids,
            "source": source,
            "scoreable": bool(fact_ids),
            "unscoreable_reason": None if fact_ids else "No measured binding fact was available.",
        }

    if intent == "NONE":
        # No intent has been expressed, so there is no direction to assess and nothing to decide.
        #
        # This branch was absent until 24 August 2026, and the effect was not confined to the record.
        # An unexpressed intent fell through to the substitution below, which reads a missing intent
        # as CENTRE, and the whole guidance policy then ran against a direction the person had not
        # asked for. In front of a blocked centre with two comparably clear sides it reached
        # AWAITING_SECTOR_CHOICE and listed both, and the browser page unhides its two choice
        # buttons from that list, so a person who had pressed nothing was asked to pick a side. The
        # release said the opposite in the same breath, carrying IDLE_NO_INTENT alongside the two
        # options, which the release schema refuses.
        #
        # The scene is still measured and still reported: the advisories and the clear-sector list
        # are what the sector display draws, and they do not depend on an intent. Only the decision
        # is withheld.
        return {
            "object_advisory": object_advisory,
            "sector_advisory": sector_advisory,
            "composite_advisory": scene_advisory,
            "scene_advisory": scene_advisory,
            "motion_decision": "STOP",
            "selected_sector": "NONE",
            "interaction_state": "IDLE_NO_INTENT",
            "clear_sectors": clear,
            "selection_status": "UNAVAILABLE",
            "selection_options": [],
            "rules_fired": rules,
            # Stated as NONE rather than left to be derived. The source is otherwise worked out from
            # the fact identifiers, and an empty list reads as SECTOR.
            "scene_binding": binding([], "NONE"),
            "action_binding": binding([], "NONE"),
            "scene_fact_required": False,
        }

    if intent == "BACKWARD":
        condition = "condition:rear_unobserved"
        rules.append({"fact_id": condition, "rule_id": "rear.unobserved", "supporting_fact_ids": [condition]})
        return {
            "object_advisory": object_advisory,
            "sector_advisory": sector_advisory,
            "composite_advisory": scene_advisory,
            "scene_advisory": scene_advisory,
            "motion_decision": "STOP",
            "selected_sector": "NONE",
            "interaction_state": "REORIENTATION_REQUIRED",
            "clear_sectors": clear,
            "selection_status": "UNAVAILABLE",
            "selection_options": [],
            "rules_fired": rules,
            "scene_binding": binding([f"sector:{_worst_sector(sectors).lower()}"] if _worst_sector(sectors) else [], "SECTOR"),
            "action_binding": binding([condition], "CONDITION"),
            "scene_fact_required": True,
        }

    if intended is None:
        intended = "CENTRE"
    intended_status = effective_statuses.get(intended, "UNKNOWN")
    intended_fact = f"sector:{intended.lower()}"
    intended_object = binding_object_by_sector.get(intended)
    intended_binding_fact = intended_object.get("fact_id") if intended_object is not None else intended_fact
    interaction_state = "GUIDANCE_ACTIVE"
    selection_status = "SELECTED"
    selection_options: list[str] = []

    if intended_status == "CLEAR":
        decision, selected = "PROCEED", intended
        action_ids = [intended_binding_fact]
    else:
        # HDSG_DETERMINISTIC_ACTION_POLICY.md section 3, revised: a constrained intended sector
        # is still passable, so it no longer settles for SLOW without first checking whether a
        # fully clear alternative exists. It reaches the same candidate-selection logic as a
        # blocked or unknown intended sector; the two differ only in their fallback when that
        # selection cannot produce a redirect, since a constrained sector always has a safe
        # default (continue cautiously on it) that a blocked one does not.
        candidates = [candidate for candidate in clear if candidate != intended]
        resolved_as_slow = False
        if not candidates:
            if intended_status == "CONSTRAINED":
                decision, selected = "SLOW", intended
                action_ids = [intended_binding_fact]
                resolved_as_slow = True
            else:
                decision, selected = "STOP", "NONE"
                selection_status = "UNAVAILABLE"
                action_ids = ["condition:no_clear_sector"]
                rules.append({
                    "fact_id": "condition:no_clear_sector",
                    "rule_id": "sector.no_clear_alternative",
                    "supporting_fact_ids": [f"sector:{name.lower()}" for name in SECTORS],
                })
        else:
            order = {
                "LEFT": {"CENTRE": 1, "RIGHT": 2},
                "CENTRE": {"LEFT": 1, "RIGHT": 1},
                "RIGHT": {"CENTRE": 1, "LEFT": 2},
            }[intended]
            best_distance = min(order[candidate] for candidate in candidates)
            best = [candidate for candidate in candidates if order[candidate] == best_distance]
            if len(best) == 2:
                values = {candidate: sectors[candidate.lower()].get("clearance_m") for candidate in best}
                difference = abs(float(values[best[0]]) - float(values[best[1]]))
                if difference > sector_choice_tolerance_m:
                    selected = max(best, key=lambda candidate: float(values[candidate]))
                elif previous_selected_sector in best:
                    selected = str(previous_selected_sector)
                elif intended_status == "CONSTRAINED":
                    # The intended sector remains passable, so an unresolved tie between two
                    # equally clear alternatives is not worth forcing a closed-ended choice on
                    # the user; continue cautiously on the intended sector instead.
                    decision, selected = "SLOW", intended
                    action_ids = [intended_binding_fact]
                    resolved_as_slow = True
                else:
                    selected = "NONE"
                    decision = "STOP"
                    interaction_state = "AWAITING_SECTOR_CHOICE"
                    selection_status = "CHOICE_REQUIRED"
                    selection_options = best
                    action_ids = [intended_binding_fact] + [f"sector:{candidate.lower()}" for candidate in best]
            else:
                selected = best[0]
            if not resolved_as_slow and selected != "NONE":
                decision = "REDIRECT"
                action_ids = [intended_binding_fact, f"sector:{selected.lower()}"]

    worst = _worst_sector(sectors)
    nearest_binding_object = min(
        binding_object_by_sector.values(),
        key=lambda item: float(item["distance_m"]),
        default=None,
    )
    if SEVERITY[object_advisory] > SEVERITY[sector_advisory] and nearest_binding_object is not None:
        scene_ids = [str(nearest_binding_object["fact_id"])]
        scene_source = "OBJECT"
    elif (object_advisory == sector_advisory and nearest_binding_object is not None
          and worst is not None and scene_advisory != "SAFE"):
        scene_ids = [str(nearest_binding_object["fact_id"]), f"sector:{worst.lower()}"]
        scene_source = "BOTH"
    else:
        scene_ids = [f"sector:{worst.lower()}"] if worst else []
        scene_source = "SECTOR" if scene_ids else "NONE"
    scene_fact_required = SEVERITY[scene_advisory] > {
        "PROCEED": 0, "SLOW": 1, "REDIRECT": 1, "STOP": 2
    }[decision]
    return {
        "object_advisory": object_advisory,
        "sector_advisory": sector_advisory,
        "composite_advisory": scene_advisory,
        "scene_advisory": scene_advisory,
        "motion_decision": decision,
        "selected_sector": selected,
        "interaction_state": interaction_state,
        "clear_sectors": clear,
        "selection_status": selection_status,
        "selection_options": selection_options,
        "rules_fired": rules,
        "scene_binding": binding(scene_ids, scene_source),
        "action_binding": binding(
            action_ids,
            "SECTOR" if all(item.startswith("sector:") for item in action_ids)
            else ("OBJECT" if all(item.startswith("object:") for item in action_ids) else "BOTH"),
        ),
        "scene_fact_required": scene_fact_required,
    }


def guidance_signature(
    authority: Mapping[str, Any],
    measurement_state: str,
    objects: Optional[Iterable[Mapping[str, Any]]] = None,
) -> str:
    """Builds the material-change signature used by the hybrid caption policy."""
    object_tokens: dict[str, str] = {}
    for item in objects or ():
        fact_id = str(item.get("fact_id") or "")
        if not fact_id:
            continue
        label = str(item.get("canonical_label") or item.get("raw_label") or "object").lower()
        bearing = _bearing(item.get("bearing"))
        motion = "MOVING" if item.get("motion_state") == "MOVING" else "PRESENT"
        object_tokens[fact_id] = f"object:{label}:{bearing}:{motion}"

    def stable_tokens(binding: Mapping[str, Any]) -> list[str]:
        return [
            object_tokens.get(str(fact_id), str(fact_id))
            for fact_id in binding.get("accepted_fact_ids", [])
        ]

    scene_ids = stable_tokens(authority.get("scene_binding", {}))
    action_ids = stable_tokens(authority.get("action_binding", {}))
    moving_ids = [
        object_tokens.get(str(fact_id), str(fact_id))
        for fact_id in authority.get("moving_object_fact_ids", [])
    ]
    parts = [
        authority.get("scene_advisory", "STOP"),
        authority.get("motion_decision", "STOP"),
        authority.get("selected_sector", "NONE"),
        ",".join(scene_ids),
        ",".join(action_ids),
        str(measurement_state),
        ",".join(moving_ids),
    ]
    return "|".join(str(part) for part in parts)


def build_fact_packet(
    *,
    run_id: str,
    event_id: str,
    observation_id: str,
    ticket_id: Optional[str],
    timestamp_ms: float,
    intent: str,
    trigger_type: str,
    request_id: str,
    response_mode: str,
    previous_signature: Optional[str],
    objects: list[dict],
    sectors: dict[str, dict],
    authority: dict,
    mirror_view: bool,
    detector_model: str,
    detector_confidence: float,
    pipeline_config_path: Path,
    ontology_path: Path,
    clear_threshold_m: float,
    blocked_threshold_m: float,
    sector_choice_tolerance_m: float,
    motion_tracker: MotionTracker,
    object_caution_below_m: float = OBJECT_CAUTION_BELOW_M,
    hazard_stop_at_or_below_m: float = HAZARD_STOP_AT_OR_BELOW_M,
    # The geometry the sector clearances were measured with. Recorded rather than restated: the
    # caller reads these from `pipeline.yaml` and passes the same values to `compute_lane_state`, so
    # the record states the geometry that was used rather than the geometry that was intended.
    sector_band_top_fraction: float = SECTOR_BAND_TOP_FRACTION,
    sector_band_bottom_fraction: float = SECTOR_BAND_BOTTOM_FRACTION,
    sector_min_measured_fraction: float = SECTOR_MIN_MEASURED_FRACTION,
    sector_left_max_fraction: float = 0.33,
    sector_right_min_fraction: float = 0.66,
    depth_valid_fraction: Optional[float] = None,
    configuration_hash: Optional[str] = None,
    post_reorientation_stable_observations: int = 4,
    post_reorientation_max_variation_m: float = 0.10,
    reassessment_cooldown_ms: float = 1500.0,
    more_detail_freshness_ms: float = 5000.0,
    scenario_id: Optional[str] = None,
    input_method: str = "SYSTEM",
    control_id: Optional[str] = None,
    recording_dir: Optional[str] = None,
    source_mode: str = "LIVE",
) -> dict:
    """Builds the complete pre-VLM evidence record for one event.

    `recording_dir` names the directory holding the synchronised RGB and depth files for this
    run, relative to the telemetry log. When it is supplied the observation references point at
    those files rather than at memory, which is what satisfies the evaluation contract's
    requirement for an input file or recording identifier and what makes the event replayable.
    """
    depth_valid = all(bool(sector["valid"]) for sector in sectors.values())
    measurement_state = "VALID" if depth_valid else ("PARTIAL" if any(bool(sector["valid"]) for sector in sectors.values()) else "INVALID")
    authority = dict(authority)
    moving_ids = [item["fact_id"] for item in objects if item.get("motion_state") == "MOVING"]
    authority["moving_object_fact_ids"] = moving_ids
    signature = guidance_signature(authority, measurement_state, objects)
    if recording_dir:
        rgb_ref = f"{recording_dir}/{observation_id}.color.png"
        depth_ref = f"{recording_dir}/{observation_id}.depth_mm.png"
    else:
        rgb_ref = f"memory://{observation_id}/rgb"
        depth_ref = f"memory://{observation_id}/depth"
    packet = {
        "schema_version": FACT_PACKET_SCHEMA,
        "identity": {
            "run_id": run_id,
            "event_id": event_id,
            "observation_id": observation_id,
            "ticket_id": ticket_id,
            "scenario_id": scenario_id,
            "source_mode": source_mode,
        },
        "observation": {
            "captured_at_utc": utc_now(),
            "sensor_timestamp_ms": float(timestamp_ms),
            "camera_model": "Intel RealSense D455f",
            "device_id": "d455f_01",
            "rgb_ref": rgb_ref,
            "depth_ref": depth_ref,
            "mirror_view": bool(mirror_view),
            "depth_aligned_to_rgb": True,
            "rgb_valid": True,
            "depth_valid": depth_valid,
            # How much of the reasoning band carried a usable reading, recorded on every
            # observation whether or not it crosses any threshold. The point is the distribution:
            # the caution threshold is provisional and set from seven frames, and it can only be
            # settled from the spread the archive accumulates. None where no depth frame reached
            # the packet.
            "depth_valid_fraction": (
                None if depth_valid_fraction is None else round(float(depth_valid_fraction), 4)
            ),
            "measurement_state": measurement_state,
            "validity_reason_codes": [] if depth_valid else ["DEPTH_PARTIAL_OR_INVALID"],
            "rear_state": "UNOBSERVED",
        },
        "interaction": {
            "intent": str(intent).upper(),
            "trigger_type": trigger_type,
            "request_id": request_id,
            "input_method": input_method,
            "control_id": control_id,
            "response_mode": response_mode,
            "previous_guidance_signature": previous_signature,
            "current_guidance_signature": signature,
        },
        "objects": objects,
        "sectors": sectors,
        "deterministic": {key: value for key, value in authority.items() if key != "moving_object_fact_ids"},
        "configuration": {
            "code_version": code_version(),
            "configuration_hash": configuration_hash or sha256_file(pipeline_config_path),
            "ontology_hash": sha256_file(ontology_path),
            "detector_model": detector_model,
            "detector_confidence_threshold": float(detector_confidence),
            # `binding_tie_margin` was recorded here as a literal 0.15 until 23 August 2026. No line
            # of code read a 0.15 tie margin: it appeared only in this record, in the schema and in
            # one fixture, so the block stated two tie thresholds of which one was fiction. The
            # parameter that exists is `sector_choice_tolerance` below. Removed rather than nulled,
            # by decision, the archive being development data that can be recorded again.
            "thresholds_m": {
                "object_stop_below": float(blocked_threshold_m),
                "object_caution_below": float(object_caution_below_m),
                "hazard_stop_at_or_below": float(hazard_stop_at_or_below_m),
                "sector_blocked_below": float(blocked_threshold_m),
                "sector_clear_at_or_above": float(clear_threshold_m),
                "sector_choice_tolerance": float(sector_choice_tolerance_m),
            },
            "post_reorientation_stable_observations": int(post_reorientation_stable_observations),
            "post_reorientation_max_variation_m": float(post_reorientation_max_variation_m),
            # `horizontal_divisions: 3` said how many sectors there were and never where they were
            # divided. The boundaries lived only in `pipeline.yaml`, so reading an archived record
            # required still holding the file that hashes to its `configuration_hash`. They are
            # recorded here as well, so the block stands on its own.
            "sector_geometry": {
                "top_fraction": float(sector_band_top_fraction),
                "bottom_fraction": float(sector_band_bottom_fraction),
                "horizontal_divisions": 3,
                "left_max_fraction": float(sector_left_max_fraction),
                "right_min_fraction": float(sector_right_min_fraction),
                "min_measured_fraction": float(sector_min_measured_fraction),
            },
            "motion_tracking": {
                "enabled": True,
                "tracker_id": "botsort.v1",
                "camera_motion_compensation": "OPTICAL_FLOW_MEDIAN",
                "movement_threshold_m": float(motion_tracker.movement_threshold_m),
                "confirmation_observations": int(motion_tracker.confirmation_observations),
                "lost_after_observations": int(motion_tracker.lost_after_observations),
            },
            "timing_ms": {
                "restrictive_transition_persistence": 150,
                "recovery_transition_persistence": 500,
                "reassessment_cooldown": float(reassessment_cooldown_ms),
                "more_detail_freshness": float(more_detail_freshness_ms),
            },
        },
    }
    return packet


def _fact_from_packet(packet: Mapping[str, Any], fact_id: str) -> Optional[dict]:
    if fact_id.startswith("sector:"):
        return packet.get("sectors", {}).get(fact_id.split(":", 1)[1])
    if fact_id.startswith("object:"):
        return next((item for item in packet.get("objects", []) if item.get("fact_id") == fact_id), None)
    if fact_id.startswith("condition:"):
        return next((item for item in packet.get("deterministic", {}).get("rules_fired", []) if item.get("fact_id") == fact_id), None)
    return None


def _permitted_fact(packet: Mapping[str, Any], fact_id: str) -> Optional[dict]:
    fact = _fact_from_packet(packet, fact_id)
    if fact is None:
        return None
    if fact_id.startswith("sector:"):
        sector = fact_id.split(":", 1)[1]
        measurement = None
        if fact.get("valid"):
            measurement = {
                "measurement_id": f"m:sector:{sector}:clearance",
                "unit": "metre",
                "source_pointer": f"/sectors/{sector}/clearance_m",
            }
        return {
            "fact_id": fact_id,
            "fact_type": "SECTOR",
            "name": f"{sector} sector",
            "state": fact.get("status"),
            "bearing": None,
            "grounding": None,
            "motion_state": None,
            "supporting_fact_ids": [],
            "measurement": measurement,
        }
    if fact_id.startswith("object:"):
        measurement = None
        if fact.get("distance_valid"):
            token = fact_id.split(":", 1)[1]
            measurement = {
                "measurement_id": f"m:object:{token}:distance",
                "unit": "metre",
                "source_pointer": f"/objects/{packet.get('objects', []).index(fact)}/distance_m",
            }
        state = "MOVING" if fact.get("motion_state") == "MOVING" else (
            fact.get("distance_bin") if fact.get("distance_valid") else "PRESENT"
        )
        return {
            "fact_id": fact_id,
            "fact_type": "OBJECT",
            "name": fact.get("canonical_label") or fact.get("raw_label"),
            "state": state,
            "bearing": fact.get("bearing"),
            "grounding": fact.get("grounding"),
            "motion_state": fact.get("motion_state"),
            "supporting_fact_ids": [],
            "measurement": measurement,
        }
    return {
        "fact_id": fact_id,
        "fact_type": "CONDITION",
        "name": fact_id.split(":", 1)[1].replace("_", " "),
        "state": "ACTIVE",
        "bearing": None,
        "grounding": "DERIVED",
        "motion_state": None,
        "supporting_fact_ids": list(fact.get("supporting_fact_ids") or []),
        "measurement": None,
    }


def build_prompt_packet(
    fact_packet: Mapping[str, Any],
    *,
    prompt_id: str,
    model_id: str,
    model_hash: str,
    quantisation: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    system_prompt: str,
    constraint_hash: str,
    prompt_profile_id: Optional[str] = None,
    system_prompt_id: str = "hdsg.reason_only.v1",
    expected_response_schema: str = CAPTION_SCHEMA,
) -> dict:
    """Builds the restricted facts and response profile supplied to the VLM.

    A caller-supplied requirement set was accepted here until 23 August 2026, so that a routed
    question could narrow the permitted facts to its topic. The topic routing was withdrawn, and
    an answered question now uses the same More detail construction as any other expansion: a
    clause per sector and per detected object.
    """
    interaction = fact_packet["interaction"]
    deterministic = fact_packet["deterministic"]
    response_mode = interaction["response_mode"]
    max_reasons, max_visuals, allow_visuals = PROFILE_LIMITS[response_mode]
    requirements: list[dict] = []
    fact_ids: list[str] = []

    action_ids = list(deterministic["action_binding"]["accepted_fact_ids"])
    if action_ids:
        requirements.append({
            "requirement_id": "action_reason",
            "role": "ACTION_BINDING",
            "match": "ALL_OF" if len(action_ids) > 1 else "ANY_OF",
            "fact_ids": action_ids,
        })
        fact_ids.extend(action_ids)
    if deterministic.get("scene_fact_required"):
        scene_ids = list(deterministic["scene_binding"]["accepted_fact_ids"])
        if scene_ids:
            requirements.append({
                "requirement_id": "scene_reason",
                "role": "SCENE_BINDING",
                "match": "ANY_OF",
                "fact_ids": scene_ids,
            })
            fact_ids.extend(scene_ids)

    moving = [item["fact_id"] for item in fact_packet.get("objects", []) if item.get("motion_state") == "MOVING"]
    minimum_required_clauses = sum(
        len(item["fact_ids"]) if item["match"] == "ALL_OF" else 1
        for item in requirements
    )
    if moving and minimum_required_clauses < max_reasons:
        requirements.append({
            "requirement_id": "moving_object_alert",
            "role": "MOVING_OBJECT_ALERT",
            "match": "ANY_OF",
            "fact_ids": moving,
        })
        fact_ids.extend(moving)
        minimum_required_clauses += 1

    if response_mode == "MORE_DETAIL":
        # Every sector and every currently detected object earns its own clause, up to the
        # profile's remaining capacity, so More detail names the whole scene rather than adding
        # one fact picked from a list. Each gets its own single-fact requirement: a requirement
        # spanning several facts under ANY_OF is satisfied by naming only one of them, which is
        # exactly the terseness this mode exists to avoid.
        detail_count = 0
        for index, sector in enumerate(("sector:left", "sector:centre", "sector:right")):
            if sector in fact_ids or minimum_required_clauses + detail_count >= max_reasons:
                continue
            fact_ids.append(sector)
            requirements.append({
                "requirement_id": f"detail_sector_{index}",
                "role": "SCENE_BINDING",
                "match": "ANY_OF",
                "fact_ids": [sector],
            })
            detail_count += 1
        for item in fact_packet.get("objects", []):
            object_fact_id = item["fact_id"]
            if object_fact_id in fact_ids or minimum_required_clauses + detail_count >= max_reasons:
                continue
            fact_ids.append(object_fact_id)
            requirements.append({
                "requirement_id": f"detail_object_{object_fact_id.split(':', 1)[1]}",
                "role": "SCENE_BINDING",
                "match": "ANY_OF",
                "fact_ids": [object_fact_id],
            })
            detail_count += 1

    return _finish_prompt_packet(
        fact_packet, requirements, fact_ids,
        prompt_id=prompt_id, response_mode=response_mode,
        prompt_profile_id=prompt_profile_id, model_id=model_id, model_hash=model_hash,
        quantisation=quantisation, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, system_prompt=system_prompt, constraint_hash=constraint_hash,
        system_prompt_id=system_prompt_id, max_reasons=max_reasons,
        max_visuals=max_visuals, allow_visuals=allow_visuals,
        expected_response_schema=expected_response_schema,
    )


def _finish_prompt_packet(
    fact_packet: Mapping[str, Any],
    requirements: list[dict],
    fact_ids: list[str],
    *,
    prompt_id: str,
    response_mode: str,
    prompt_profile_id: Optional[str],
    model_id: str,
    model_hash: str,
    quantisation: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    system_prompt: str,
    constraint_hash: str,
    system_prompt_id: str,
    max_reasons: int,
    max_visuals: int,
    allow_visuals: bool,
    expected_response_schema: str = CAPTION_SCHEMA,
) -> dict:
    """Resolves the permitted facts and assembles the packet body.

    Shared by the automatic and question-routed requirement constructions so both produce a packet
    of identical shape and both pass through the same permitted-fact resolution.
    """
    interaction = fact_packet["interaction"]
    permitted = [item for item in (_permitted_fact(fact_packet, fact_id) for fact_id in dict.fromkeys(fact_ids)) if item is not None]
    routing = {
        "prompt_id": prompt_id,
        "event_id": fact_packet["identity"]["event_id"],
        "observation_id": fact_packet["identity"]["observation_id"],
        "request_id": interaction["request_id"],
        "response_mode": response_mode,
        "prompt_profile_id": prompt_profile_id or {
            "AUTOMATIC": "guidance_reason.v1",
            "MORE_DETAIL": "more_detail.v1",
            "REASSESSMENT": "reassessment.v1",
        }[response_mode],
    }
    return {
        "schema_version": PROMPT_PACKET_SCHEMA,
        "routing": routing,
        "image": {
            "included": True,
            "observation_id": fact_packet["identity"]["observation_id"],
            "rgb_ref": fact_packet["observation"]["rgb_ref"],
            "transform": {
                "longest_side_px": 448,
                "preserve_aspect_ratio": True,
                "encoding": "jpeg",
                "jpeg_quality": 70,
            },
        },
        "requirements": requirements,
        "permitted_facts": permitted,
        # Only what something reads. Seven further flags stood here until 23 August 2026 and no code
        # consulted any of them: the gate applies its own checks and the prompt text is written as
        # prose in `_COMPOSED_INSTRUCTION` rather than assembled from flags.
        #
        # Two of the seven were not merely inert. `authoritative_numbers_allowed: False` and
        # `measurement_placeholders_required: True` describe the templated contract removed on
        # 21 August 2026, under which the runtime wrote the sentences and a measurement appeared as a
        # placeholder the renderer filled in afterwards. The composed contract is the opposite: the
        # model writes the measured value into the prose and declares it, and no placeholder exists
        # anywhere in the system. Every prompt packet in the archive therefore states two rules that
        # were false of the run that wrote them.
        #
        # The other five were true of the design and read by nothing: `reason_only`,
        # `action_instruction_allowed`, `unlisted_fact_mentions_allowed`, `language` "en-GB", and
        # `max_reason_clauses`, whose name was wrong as well: it held the cap on requirements, and
        # nothing anywhere counts clauses.
        "response_constraints": {
            "visual_only_observations_allowed": allow_visuals,
            "max_visual_observations": max_visuals,
            "max_text_chars": None,
        },
        "generation": {
            "model_id": model_id,
            "model_hash": model_hash,
            "quantisation": quantisation,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "system_prompt_id": system_prompt_id,
            "system_prompt_hash": sha256_text(system_prompt),
            "constraint_id": expected_response_schema,
            "constraint_hash": constraint_hash,
        },
        "expected_response_schema": expected_response_schema,
    }


def _reason_sort(codes: Iterable[str]) -> list[str]:
    order = {code: index for index, code in enumerate(REASON_CODE_ORDER)}
    return sorted(set(codes), key=lambda code: order.get(code, len(order)))


# The precision every measurement is displayed at, whichever layer renders it. The gate compares a
# caption's numbers against measurements at this precision, so the constant is shared rather than
# repeated: a renderer showing two decimals while the gate compared three would reject a caption for
# stating exactly what the system was about to display.
MEASUREMENT_DECIMALS = 2


def display_value(value: Any) -> float:
    """Returns a measurement as it is displayed, which is the value the gate holds a caption to."""
    return float(f"{float(value):.{MEASUREMENT_DECIMALS}f}")


def _format_measurement(value: Any) -> str:
    return f"{float(value):.{MEASUREMENT_DECIMALS}f} metres"


def _fallback_reason(fact_packet: Mapping[str, Any]) -> tuple[str, list[str], list[str], list[dict]]:
    deterministic = fact_packet["deterministic"]
    action_ids = list(deterministic["action_binding"]["accepted_fact_ids"])
    scene_ids = list(deterministic["scene_binding"]["accepted_fact_ids"])
    decision = deterministic["motion_decision"]
    state = deterministic["interaction_state"]
    substitutions: list[dict] = []

    if state == "REORIENTATION_REQUIRED":
        return "The area behind the walker has not been observed.", action_ids, scene_ids, substitutions
    if state == "POST_REORIENTATION_STABILISING":
        return "The new view is not yet stable.", action_ids, scene_ids, substitutions
    if state == "AWAITING_SECTOR_CHOICE":
        intended = next((item for item in action_ids if _fact_from_packet(fact_packet, item) and _fact_from_packet(fact_packet, item).get("status") != "CLEAR"), "sector:centre")
        return f"The {intended.split(':')[1]} sector is blocked, while the left and right sectors are similarly clear.", action_ids, scene_ids, substitutions
    if decision == "STOP" and "condition:no_clear_sector" in action_ids:
        # The widest sector is named by its own measurement, not by a description of its role.
        #
        # This recorded `m:sector:best:clearance` until 24 August 2026. No sector is called best, so
        # the identifier resolved to nothing: the number shown was correct and the evidence behind it
        # could not be checked against the packet. It also distorted the contribution comparison,
        # which counts a measurement as named by matching identifiers between conditions. The model
        # can only declare identifiers it was offered, so the same clearance appeared under two
        # different names and was counted as two facts.
        valid = [(name, sector["clearance_m"]) for name, sector in fact_packet["sectors"].items()
                 if sector.get("clearance_m") is not None]
        if valid:
            # Ties settle on the sector order rather than on dictionary order, so the same scene
            # names the same sector on every run.
            widest = max(valid, key=lambda item: (item[1], -SECTORS.index(item[0].upper())))
            value = widest[1]
            substitutions.append({"measurement_id": f"m:sector:{widest[0]}:clearance",
                                  "formatted_value": _format_measurement(value)})
            return f"No forward sector is currently clear. The greatest measured clearance is {_format_measurement(value)}.", action_ids, scene_ids, substitutions
        return "Reliable depth measurements are not currently available.", action_ids, scene_ids, substitutions

    reason_parts: list[str] = []
    for fact_id in action_ids:
        fact = _fact_from_packet(fact_packet, fact_id)
        if not fact:
            continue
        if fact_id.startswith("object:"):
            label = fact.get("canonical_label") or fact.get("raw_label") or "object"
            bearing = str(fact.get("bearing") or "CENTRE").lower()
            bearing_text = "in the centre" if bearing == "centre" else f"on the {bearing}"
            distance = fact.get("distance_m")
            if distance is not None:
                token = fact_id.split(":", 1)[1]
                measurement_id = f"m:object:{token}:distance"
                formatted = _format_measurement(distance)
                substitutions.append({"measurement_id": measurement_id, "formatted_value": formatted})
                reason_parts.append(f"A {label} is detected {bearing_text} at {formatted}.")
            else:
                reason_parts.append(f"A {label} is detected {bearing_text}.")
            continue
        if not fact_id.startswith("sector:"):
            continue
        name = fact_id.split(":", 1)[1]
        value = fact.get("clearance_m")
        status = fact.get("status")
        if value is None:
            reason_parts.append(f"The {name} sector measurement is unavailable.")
            continue
        measurement_id = f"m:sector:{name}:clearance"
        formatted = _format_measurement(value)
        substitutions.append({"measurement_id": measurement_id, "formatted_value": formatted})
        predicate = {
            "CLEAR": f"is clear for {formatted}",
            "CONSTRAINED": f"has limited clearance at {formatted}",
            "BLOCKED": f"is blocked at {formatted}",
            "UNKNOWN": "has no reliable measurement",
        }[status]
        reason_parts.append(f"The {name} sector {predicate}.")
    if deterministic.get("scene_fact_required"):
        for fact_id in scene_ids:
            if fact_id in action_ids:
                continue
            fact = _fact_from_packet(fact_packet, fact_id)
            if fact and fact_id.startswith("sector:") and fact.get("clearance_m") is not None:
                name = fact_id.split(":", 1)[1]
                formatted = _format_measurement(fact["clearance_m"])
                substitutions.append({"measurement_id": f"m:sector:{name}:clearance", "formatted_value": formatted})
                reason_parts.append(f"The {name} sector has limited clearance at {formatted}." if fact["status"] == "CONSTRAINED" else f"The {name} sector is blocked at {formatted}.")
    return " ".join(reason_parts) or "Reliable depth measurements are not currently available.", action_ids, scene_ids, substitutions


def _interaction_text(state: str) -> Optional[str]:
    return {
        "AWAITING_SECTOR_CHOICE": "Select left or right.",
        "REORIENTATION_REQUIRED": "Reorient the walker or camera and select Reassess.",
        "POST_REORIENTATION_STABILISING": "Hold position while the new view is checked.",
    }.get(state)


def build_release(
    fact_packet: Mapping[str, Any],
    prompt_packet: Mapping[str, Any],
    *,
    release_id: str,
    failure_codes: Iterable[str] = (),
    no_intent: bool = False,
    pending: bool = False,
) -> dict:
    """Creates the deterministic release object.

    This builder renders the release for every case in which no generated caption is shown: the
    idle state, the pending state, and each rejection or unavailability. An accepted caption is
    rendered by `hdsg_composed.build_composed_release`, which calls this builder for the authority
    block, the identity and the action text, and then substitutes its own content. The deterministic
    account therefore has one implementation rather than two, and the action the user acts on is
    produced here in every case.

    `no_intent` renders the IDLE_NO_INTENT state from `HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md`:
    an empty caption while the sector display continues to update on its own, unrelated channel.

    `pending` renders that policy's GENERATION_PENDING state: the deterministic action line is
    available at once and the reason line carries a placeholder while a generation request is in
    flight. It has an effect only when the deterministic interaction state is `GUIDANCE_ACTIVE`.
    The three other temporary interaction states (`AWAITING_SECTOR_CHOICE`,
    `REORIENTATION_REQUIRED`, `POST_REORIENTATION_STABILISING`) already carry a complete
    deterministic account and are left unchanged by a pending generation, per that policy's
    interaction-state table.
    """
    deterministic = fact_packet["deterministic"]
    identity = fact_packet["identity"]
    action_template_id, action_text = ACTION_TEMPLATES[(deterministic["motion_decision"], deterministic["selected_sector"])]

    if no_intent:
        return {
            "schema_version": RELEASE_SCHEMA,
            "identity": {
                "release_id": release_id,
                "event_id": identity["event_id"],
                "observation_id": identity["observation_id"],
                "ticket_id": identity["ticket_id"],
                "request_id": fact_packet["interaction"]["request_id"],
                "released_at_utc": utc_now(),
            },
            "authority": {
                "scene_advisory": deterministic["scene_advisory"],
                "motion_decision": deterministic["motion_decision"],
                "selected_sector": deterministic["selected_sector"],
                "interaction_state": "IDLE_NO_INTENT",
                # Empty because this builder is what declares the state idle, and an idle release
                # offering a choice contradicts itself. It carried the packet's list until
                # 24 August 2026, which the release schema refuses. `determine_authority` now
                # withholds the decision when no intent has been expressed, so the list reaching
                # here is already empty on the live path; clearing it keeps the two fields agreeing
                # whatever is passed.
                "selection_options": [],
                "action_template_id": action_template_id,
            },
            "content": {
                "action_text": "",
                "reason_text": "",
                "interaction_text": None,
                "additional_detail_texts": [],
                "caption_text": "",
            },
            "evidence": {
                "action_binding_fact_ids": [],
                "scene_binding_fact_ids": [],
                "measurement_substitutions": [],
                "released_visual_observation_ids": [],
            },
            "verification": {
                "release_mode": "DETERMINISTIC_FALLBACK",
                "candidate_status": "UNAVAILABLE",
                "gate_outcome": "REJECTED_FALLBACK",
                "primary_reason_code": "RG_NO_INTENT_EXPRESSED",
                "reason_codes": ["RG_NO_INTENT_EXPRESSED"],
                "guidance_signature": fact_packet["interaction"]["current_guidance_signature"],
                "fact_packet_schema": FACT_PACKET_SCHEMA,
                "prompt_packet_schema": PROMPT_PACKET_SCHEMA,
                "candidate_schema": CAPTION_SCHEMA,
                "code_version": code_version(),
            },
        }

    pending_active = pending and deterministic["interaction_state"] == "GUIDANCE_ACTIVE"
    errors = _reason_sort(list(failure_codes) + (["RG_GENERATION_PENDING"] if pending else []))
    interaction_text = PENDING_PLACEHOLDER_TEXT if pending_active else _interaction_text(deterministic["interaction_state"])
    action_ids = list(deterministic["action_binding"]["accepted_fact_ids"])
    scene_ids = list(deterministic["scene_binding"]["accepted_fact_ids"])
    substitutions: list[dict] = []
    released_visual_ids: list[str] = []

    if pending_active:
        reason_text = ""
        additional = []
        candidate_status = "UNAVAILABLE"
        reason_codes = ["RG_GENERATION_PENDING"]
    else:
        reason_text, action_ids, scene_ids, substitutions = _fallback_reason(fact_packet)
        additional = []
        # UNAVAILABLE distinguishes a caption that never arrived from one the gate refused. The
        # composed builder passes its gate codes through to this builder, so the distinction is
        # drawn from the codes rather than from whether a candidate object was supplied.
        candidate_status = "UNAVAILABLE" if any(
            code in {"RG_MODEL_UNAVAILABLE", "RG_GENERATION_TIMEOUT", "RG_CONSTRAINT_FAILURE",
                     "RG_GENERATION_PENDING"}
            for code in errors
        ) else "REJECTED"
        reason_codes = errors or ["RG_INTERNAL_GATE_ERROR"]
    mode = "DETERMINISTIC_FALLBACK"
    gate_outcome = "REJECTED_FALLBACK"

    caption_parts = [action_text, reason_text]
    if interaction_text:
        caption_parts.append(interaction_text)
    caption_parts.extend(additional)
    caption_text = " ".join(part.strip() for part in caption_parts if part and part.strip())
    return {
        "schema_version": RELEASE_SCHEMA,
        "identity": {
            "release_id": release_id,
            "event_id": identity["event_id"],
            "observation_id": identity["observation_id"],
            "ticket_id": identity["ticket_id"],
            "request_id": fact_packet["interaction"]["request_id"],
            "released_at_utc": utc_now(),
        },
        "authority": {
            "scene_advisory": deterministic["scene_advisory"],
            "motion_decision": deterministic["motion_decision"],
            "selected_sector": deterministic["selected_sector"],
            "interaction_state": "GENERATION_PENDING" if pending_active else deterministic["interaction_state"],
            "selection_options": deterministic["selection_options"],
            "action_template_id": action_template_id,
        },
        "content": {
            "action_text": action_text,
            "reason_text": reason_text,
            "interaction_text": interaction_text,
            "additional_detail_texts": additional,
            "caption_text": caption_text,
        },
        "evidence": {
            "action_binding_fact_ids": action_ids,
            "scene_binding_fact_ids": scene_ids,
            "measurement_substitutions": list({item["measurement_id"]: item for item in substitutions}.values()),
            "released_visual_observation_ids": released_visual_ids,
        },
        "verification": {
            "release_mode": mode,
            "candidate_status": candidate_status,
            "gate_outcome": gate_outcome,
            "primary_reason_code": reason_codes[0],
            "reason_codes": reason_codes,
            "guidance_signature": fact_packet["interaction"]["current_guidance_signature"],
            "fact_packet_schema": FACT_PACKET_SCHEMA,
            "prompt_packet_schema": PROMPT_PACKET_SCHEMA,
            "candidate_schema": CAPTION_SCHEMA,
            "code_version": code_version(),
        },
    }


def build_generation_response_record(
    fact_packet: Mapping[str, Any],
    release: Mapping[str, Any],
    *,
    raw_response: Optional[str],
    candidate: Optional[Mapping[str, Any]],
    superseded: bool,
    queued_ms: int,
    started_ms: int,
    responded_ms: Optional[int],
    released_ms: int,
) -> dict:
    """Builds the per-generation telemetry record required by the evaluation contract.

    `HDSG_EVALUATION_CONTRACT.md` section 12 requires the raw model response and timing metadata
    on every evaluated event. The response is stored in full rather than as a digest: a digest
    supports integrity checking but not the failure investigation the contract asks for, and a
    rejected candidate is precisely the case where the raw text is the evidence. The digest is
    retained alongside it so integrity remains checkable.

    This record is written to the telemetry stream and never read back into the runtime, so the
    raw response still has no path to the display and the sole-release-path property is
    unaffected.

    Timing is diagnostic only; the same contract section excludes latency from the thesis
    outcomes. `pending` is the GENERATION_PENDING window from
    `HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md`: how long the reason line showed a placeholder
    while the action line was already correct and displayed.
    """
    verification = release.get("verification", {})
    return {
        "event_id": fact_packet["identity"]["event_id"],
        "observation_id": fact_packet["identity"]["observation_id"],
        "request_id": fact_packet["interaction"]["request_id"],
        "raw_response": raw_response,
        "response_sha256": None if raw_response is None else sha256_text(raw_response),
        "parsed": candidate is not None,
        "superseded_before_release": bool(superseded),
        "reason_codes": list(verification.get("reason_codes", [])),
        "release_mode": verification.get("release_mode"),
        "timing_ms": {
            "queue_wait": started_ms - queued_ms,
            "generation": None if responded_ms is None else responded_ms - started_ms,
            "gate_and_render": released_ms - (responded_ms if responded_ms is not None else started_ms),
            "pending": released_ms - queued_ms,
        },
    }


class GenerationTimeout(Exception):
    """The model server did not answer within the time allowed."""


class ConstraintUnavailable(Exception):
    """The approved generation constraint could not be supplied with the request."""


def generation_failure_code(error: Exception) -> str:
    """Maps generation failures to the stable release-gate catalogue.

    **By the type of the failure, not by the wording of it.** Until 25 August 2026 this searched
    `str(error)` for the words "timeout", "grammar" and "constraint". The client wraps up to 200
    characters of the model server's own error message inside the exception it raises, so the code
    recorded in the release, and reported in Chapter 5, was decided by a substring search over text
    written by llama.cpp. Two of six realistic failures came out wrong: an out-of-memory error was
    recorded as a timeout because the server's advice mentioned a `--timeout` flag, and a model that
    would not load was recorded as a constraint failure because the path contained the word
    "grammar".

    It is the same fault this pipeline has carried three times, a check reading a word that
    describes a thing rather than the thing itself, and this instance decided what a run is reported
    to have done.

    A server rejecting the constraint it was sent is not distinguishable here from any other server
    error, and is recorded as RG_MODEL_UNAVAILABLE. Telling the two apart means reading the server's
    error contract rather than guessing at its wording, which is recorded in
    `LAB_SESSION_CHECKLIST.md` section E together with the startup probe that would catch the case
    at the beginning of a session instead of once per event.
    """
    if isinstance(error, (GenerationTimeout, TimeoutError)):
        return "RG_GENERATION_TIMEOUT"
    if isinstance(error, ConstraintUnavailable):
        return "RG_CONSTRAINT_FAILURE"
    return "RG_MODEL_UNAVAILABLE"


def next_identifier(prefix: str, sequence: int) -> str:
    """Formats a stable local sequence identifier."""
    return f"{prefix}_{sequence:06d}"


def monotonic_time_ms() -> int:
    """Returns monotonic time in milliseconds for local freshness checks."""
    return int(time.monotonic() * 1000)
