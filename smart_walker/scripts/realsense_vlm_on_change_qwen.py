import argparse
import base64
from datetime import datetime
import json
import math
import time
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
try:
    import torch
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
except Exception:
    torch = None

# Reuse the smart walker pipeline components without modifying the original file
try:
    import pyrealsense2 as rs
except Exception as e:
    print("Failed to import pyrealsense2:", e, file=sys.stderr)
    rs = None

try:
    # Import from the robust shared-control pipeline
    import scripts.realsense_shared_control as sw
    import scripts.hdsg_runtime as hdsg
    import scripts.hdsg_questions as questions
    import scripts.hdsg_composed as hdsg_composed
    from scripts.hdsg_web_ui import WebInterface
    from scripts.hdsg_recording import ObservationRecorder
except Exception:
    # Fallback for running directly from scripts folder
    import realsense_shared_control as sw  # type: ignore
    import hdsg_runtime as hdsg  # type: ignore
    import hdsg_questions as questions  # type: ignore
    import hdsg_composed  # type: ignore
    from hdsg_web_ui import WebInterface  # type: ignore
    from hdsg_recording import ObservationRecorder  # type: ignore


def _encode_image(img_bgr: np.ndarray, fmt: str = "jpeg", quality: int = 75) -> Tuple[str, str]:
    """Encode an image and return (mime, base64_string).

    fmt: 'jpeg' or 'png'
    quality: JPEG quality (ignored for PNG)
    """
    fmt = (fmt or "jpeg").lower()
    if fmt == "png":
        ok, enc = cv2.imencode('.png', img_bgr)
        mime = 'image/png'
    else:
        ok, enc = cv2.imencode('.jpg', img_bgr, [int(
            cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        mime = 'image/jpeg'
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return mime, base64.b64encode(enc.tobytes()).decode('ascii')


def build_mm_chat_payload(model: str, system: str, text: str, b64_data: str, mime: str,
                          temperature: float, top_p: float, max_tokens: int):
    # OpenAI-compatible /v1/chat/completions with mixed content
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    # Put image first; some servers prefer this order for multimodal parsing
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{b64_data}"}},
                    {"type": "text", "text": text},
                ],
            },
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": False,
    }


class _NullKeyListener:
    """A key listener that reports nothing, used when the browser owns the keyboard.

    The main loop reads a snapshot every pass and does not care where the keys came from, so
    substituting this leaves the loop unchanged while the browser supplies the same shortcuts as
    explicit events.
    """

    def snapshot(self):
        # last_press_ms matches the main loop's initial baseline of -1, so the key-edge branch
        # never fires and the browser's intent events are never swallowed by it. A default of 0
        # would differ from -1 on the first pass and register one spurious idle press.
        return sw.IntentState(last_press_ms=-1)

    def stop(self) -> None:
        return None


def build_text_chat_payload(model: str, system: str, text: str, grammar: str,
                            temperature: float, top_p: float, max_tokens: int) -> dict:
    """Builds a text-only grammar-constrained request.

    Used by the question admission classifier. No image is attached: deciding whether a message is
    about the surroundings does not require the frame, and omitting it removes the scene as a
    channel into the decision.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "grammar": grammar,
        "stream": False,
    }


def _build_payload_from_image_array(img_bgr: np.ndarray, fmt: str, quality: int,
                                    args, *, system: Optional[str] = None,
                                    unconstrained: bool = False,
                                    max_tokens: Optional[int] = None,
                                    grammar: Optional[str] = None) -> tuple[dict, str]:
    """Encode image with requested format/quality and build chat payload.
    Returns (payload, desc) where desc is a short string for logging.

    `unconstrained` omits the generation constraint. It exists only for the diagnostic mode the
    --unconstrained flag selects, and every caller on the release path leaves it false, so the
    absence of a grammar remains an error there rather than a silent downgrade.
    """
    mime, b64 = _encode_image(img_bgr, fmt=fmt, quality=int(quality))
    payload = build_mm_chat_payload(
        model=args.model,
        system=system or args.system,
        text=args._user_txt_for_payload,  # type: ignore[attr-defined]
        b64_data=b64,
        mime=mime,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=int(max_tokens) if max_tokens else args.max_tokens,
    )
    if unconstrained:
        return payload, f"{fmt}/q{quality} unconstrained"
    constraint = grammar or getattr(args, '_hdsg_grammar', None)
    if not constraint:
        raise RuntimeError("The approved HDSG generation constraint is unavailable.")
    payload["grammar"] = constraint
    return payload, f"{fmt}/q{quality}"


def resize_for_vlm(img: np.ndarray, size: int) -> np.ndarray:
    """Scale so the longest side is `size`, keeping the frame's aspect ratio.

    A square resize of a 4:3 RealSense frame compresses the horizontal axis by
    25%, which distorts exactly the left/mid/right geometry the lane model reads
    off the image. Sides are left unrounded; Qwen3-VL aligns them to its patch
    grid internally, and snapping here would reintroduce aspect error at small
    sizes.
    """
    h, w = img.shape[:2]
    scale = float(size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) == (w, h):
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# Severity ordering over advisories. Higher is more restrictive. Kept as a
# module-level table rather than inline comparisons so the ordering is stated
# once, can be read at a glance, and is the single thing to change if a level is
# ever added.
ADVISORY_SEVERITY = {'safe': 0, 'caution': 1, 'stop': 2}
_SEVERITY_TO_ADVISORY = {v: k for k, v in ADVISORY_SEVERITY.items()}


def more_severe(a: Optional[str], b: Optional[str]) -> str:
    """Return whichever advisory is more restrictive.

    The deterministic layer measures obstruction two ways, and the two have
    complementary blind spots:

      - Per-object distance (YOLO plus lower-band median) is precise about the
        things it recognises, but YOLOv8's 80 COCO classes contain no wall, door,
        step, stair, glass, curtain or pole. Facing a blank wall it detects
        nothing and reports SAFE.
      - The lane median over an image third sees bulk geometry, so it catches
        walls and corridor narrowing, but a thin obstacle occupying a few percent
        of a band does not move that band's median. Facing a broom handle it
        reports clear.

    Neither is sound alone, and each is blind exactly where the other sees. Taking
    the more severe of the two yields a composite that is sound where neither
    component is: SAFE is reported only when both paths agree there is nothing
    there. An unrecognised advisory is treated as the most severe, so a parsing
    failure cannot manufacture a permissive result.
    """
    sa = ADVISORY_SEVERITY.get((a or '').lower(), ADVISORY_SEVERITY['stop'])
    sb = ADVISORY_SEVERITY.get((b or '').lower(), ADVISORY_SEVERITY['stop'])
    # Return a canonical level rather than whichever input string won. Echoing
    # the input back would propagate an unrecognised value downstream, where it
    # would score as severe here but fail to match 'stop' in every consumer that
    # compares against the advisory by name.
    return _SEVERITY_TO_ADVISORY[max(sa, sb)]


# Two measurements this close together are not meaningfully distinguishable at
# the sensor's stated accuracy, so a caption naming either one is treated as
# naming the cause. Widening this makes the transparency metric more permissive.
BINDING_TIE_MARGIN_M = 0.15

_LANE_NAMES = ('left', 'centre', 'right')
_LANE_STATUS_SEVERITY = {'clear': 0, 'constrained': 1, 'unknown': 2, 'blocked': 3}


def identify_binding_fact(objects: Optional[list], lane_state: Optional[dict],
                          binding: str, mirror_view: bool,
                          tie_margin_m: float = BINDING_TIE_MARGIN_M) -> dict:
    """Name the specific measured element that produced the composite advisory.

    The composition in more_severe() records which path was binding, objects or
    lanes, but not which element within that path. That distinction is required
    by the transparency metric: an output that names a real but non-causal fact
    is faithful and still misleading. Reporting a wall at 1.2 m when the system
    stopped for a chair at 0.5 m is entailed by the Fact Packet and sends the
    user into the chair.

    Within the object path the causal element is the nearest object carrying a
    finite distance, since every object rule in the baseline is a threshold on
    that distance. Within the lane path it is the most severely classified band.
    Elements within tie_margin_m of the nearest, or sharing the winning lane
    status, are returned as ties: naming any of them is equally correct, and
    scoring must accept all of them rather than privileging an arbitrary one.

    The returned distances are already in the user's frame of reference, because
    the object side labels and the lane depths are mirrored upstream when
    mirror_view is set. The flag is recorded so a log record can be interpreted
    without reference to the invocation.

    Returns a dict that is always populated, with 'kind' set to 'none' when no
    element can be identified, so downstream consumers need no null handling.
    A 'none' result is never filled in from the non-binding path: attributing the
    advisory to a path that did not produce it would be a fabricated cause, and
    the metric is better served by marking the record unscoreable.

    Two object rules are not threshold tests on the nearest distance and are
    therefore not fully described by the element returned here. 'caution:multi_near'
    is caused by a count rather than by any single object, and 'caution:uncertainty'
    is caused by the low-light flag rather than by an object at all. The rules that
    fired are recorded alongside this result in advisory_sources, and scoring
    consults them: when uncertainty is the only rule fired, the causal fact is the
    uncertainty flag and no object name can be correct.
    """
    result = {
        'kind': 'none',
        'binding': binding,
        'label': None,
        'distance_m': None,
        'side': None,
        'lane': None,
        'lane_status': None,
        'ties': [],
        'mirror_view': bool(mirror_view),
        'tie_margin_m': float(tie_margin_m),
    }

    def _object_candidates():
        cands = []
        for idx, o in enumerate(objects or []):
            d = o.get('distance_m')
            if not isinstance(d, (int, float)):
                continue
            try:
                d = float(d)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(d):
                continue
            name = (o.get('canonical_class') or o.get('display_label')
                    or o.get('raw_label') or 'obj')
            b = o.get('bearing')
            side_raw = str(b[0] if isinstance(b, (list, tuple)) else b).lower(
            ) if b is not None else ''
            side = 'left' if 'left' in side_raw else (
                'right' if 'right' in side_raw else 'center')
            if mirror_view and side in ('left', 'right'):
                side = 'right' if side == 'left' else 'left'
            cands.append({'index': idx, 'label': name,
                          'distance_m': d, 'side': side})
        cands.sort(key=lambda c: c['distance_m'])
        return cands

    def _lane_candidates():
        if not lane_state:
            return []
        status = list(lane_state.get('status') or [])
        depths = list(lane_state.get('depths') or [])
        cands = []
        for i, name in enumerate(_LANE_NAMES):
            if i >= len(status):
                continue
            st = str(status[i]).lower()
            d = depths[i] if i < len(depths) else None
            try:
                d = float(d) if isinstance(
                    d, (int, float)) and np.isfinite(d) else None
            except (TypeError, ValueError):
                d = None
            cands.append({'lane': name, 'lane_status': st, 'distance_m': d,
                          'severity': _LANE_STATUS_SEVERITY.get(st, 0)})
        # Most severe first; among equals the nearest reading leads, treating a
        # missing reading as farthest so it never displaces a measured band.
        cands.sort(key=lambda c: (-c['severity'],
                                  c['distance_m'] if c['distance_m'] is not None else float('inf')))
        return cands

    use_objects = binding in ('objects', 'both')
    use_lanes = binding in ('lanes', 'both')

    obj_cands = _object_candidates() if use_objects else []
    lane_cands = _lane_candidates() if use_lanes else []

    # When both paths agree the advisory, the object path is reported as the
    # primary cause because it carries a specific nameable entity, while the
    # lane band it falls in is retained as a tie. A caption naming either is
    # describing the same obstruction.
    if obj_cands:
        top = obj_cands[0]
        result.update({'kind': 'object', 'label': top['label'],
                       'distance_m': top['distance_m'], 'side': top['side'],
                       'index': top['index']})
        result['ties'] = [
            {'kind': 'object', 'label': c['label'],
             'distance_m': c['distance_m'], 'side': c['side'], 'index': c['index']}
            for c in obj_cands[1:]
            if c['distance_m'] - top['distance_m'] <= tie_margin_m
        ]
        if lane_cands and lane_cands[0]['severity'] > 0:
            result['ties'].append({'kind': 'lane', 'lane': lane_cands[0]['lane'],
                                   'lane_status': lane_cands[0]['lane_status'],
                                   'distance_m': lane_cands[0]['distance_m']})
        return result

    if lane_cands and lane_cands[0]['severity'] > 0:
        top = lane_cands[0]
        result.update({'kind': 'lane', 'lane': top['lane'],
                       'lane_status': top['lane_status'],
                       'distance_m': top['distance_m']})
        result['ties'] = [
            {'kind': 'lane', 'lane': c['lane'], 'lane_status': c['lane_status'],
             'distance_m': c['distance_m']}
            for c in lane_cands[1:] if c['severity'] == top['severity']
        ]
        return result

    return result


def compute_lane_state(depth_m: Optional[np.ndarray], mirror_view: bool,
                       clear_t: float, blocked_t: float,
                       left_max: float, right_min: float,
                       top_fraction: float = hdsg.SECTOR_BAND_TOP_FRACTION,
                       bottom_fraction: float = hdsg.SECTOR_BAND_BOTTOM_FRACTION) -> Optional[dict]:
    """Partition the lower field of view into three depth bands and classify each.

    The column boundaries come from `bearing` in pipeline.yaml, the same fractions
    `bearing_from_bbox` uses to decide which sector an object is in. They were hardcoded here as
    exact thirds while the config held 0.33 and 0.66, so an object between 0.33 and 0.3333 of the
    image width was called centre while its pixels were measured as part of the left band, and
    likewise between 0.66 and 0.6667 on the other side. Two and four pixels at 640 wide, but the
    caption could place an object in a sector whose clearance was measured without it.

    This is the single place lane clearance is computed. The risk override, the
    VLM prompt, the band overlay and the safe-path badge all read this result, so
    the bands the user sees are by construction the bands the system reasoned
    over. The logic previously existed as four independent copies, which drifted:
    two of them applied the mirror correction once and one applied it twice, so
    under --mirror_view the badge and the caption named opposite directions.

    The row band comes from `sector.band_top_fraction` and `sector.band_bottom_fraction` in
    pipeline.yaml: below the horizon, above the immediate foreground where the walker's own frame
    intrudes. It was hardcoded here as 0.55 and 0.95 until 23 August 2026, and written out twice more,
    in `valid_depth_fraction` and in the configuration record that states how a run was measured.
    The three agreed and nothing compared them.

    Returns None when depth is unavailable. Otherwise a dict carrying the band
    geometry, the median depth per band, a three-way status per band, the
    advisory implied by the clear/blocked pattern, and the deterministic
    suggestion token.
    """
    if depth_m is None or not isinstance(depth_m, np.ndarray):
        return None
    try:
        H, W = depth_m.shape[:2]
        y1 = int(top_fraction * H)
        y2 = int(bottom_fraction * H)
        # Derived from the predicate `bearing_from_bbox` applies, not from rounding the fraction.
        # A column is left where its fraction is below left_max, so the first centre column is
        # ceil(left_max * W); it is right where its fraction is strictly above right_min, so the
        # first right column is floor(right_min * W) + 1. Rounding both instead left two columns
        # out of 640 assigned to a different sector than the objects standing in them.
        xL1, xL2 = 0, min(W, math.ceil(left_max * W))
        xC1, xC2 = xL2, min(W, math.floor(right_min * W) + 1)
        xR1, xR2 = max(xC2, xL2), W
        d_L = sw.median_depth_in_box(depth_m, xL1, y1, xL2, y2)
        d_C = sw.median_depth_in_box(depth_m, xC1, y1, xC2, y2)
        d_R = sw.median_depth_in_box(depth_m, xR1, y1, xR2, y2)
        # Swapping here puts every downstream consumer in the user's frame of
        # reference. Do not mirror the derived token again later.
        if mirror_view:
            d_L, d_R = d_R, d_L

        def _status(d):
            """blocked / constrained / clear. Unreadable depth is not free space.

            A band with no valid reading resolves to 'unknown' and is treated as
            not clear, so missing data can never be mistaken for room to move.
            """
            if not isinstance(d, (int, float)) or not np.isfinite(d):
                return 'unknown'
            if d < blocked_t:
                return 'blocked'
            if d < clear_t:
                return 'constrained'
            return 'clear'

        st_L, st_C, st_R = _status(d_L), _status(d_C), _status(d_R)

        def _num(d):
            try:
                return float(d) if isinstance(d, (int, float)) and np.isfinite(d) else -1.0
            except Exception:
                return -1.0

        is_L = _num(d_L) >= clear_t
        is_C = _num(d_C) >= clear_t
        is_R = _num(d_R) >= clear_t
        clear_count = sum([is_L, is_C, is_R])

        # Advisory from the clear/blocked pattern alone.
        if clear_count == 3:
            advisory = 'SAFE'
        elif clear_count == 0:
            advisory = 'STOP'
        else:
            advisory = 'CAUTION'

        if clear_count == 0:
            auto_suggest = 'back'
        elif clear_count == 3:
            auto_suggest = 'continue'
        elif clear_count == 1:
            auto_suggest = 'continue' if is_C else ('left' if is_L else 'right')
        else:  # exactly two bands clear
            if is_C and is_L:
                auto_suggest = 'mid-left'
            elif is_C and is_R:
                auto_suggest = 'mid-right'
            else:
                auto_suggest = 'left-right'

        finite = [v for v in (_num(d_L), _num(d_C), _num(d_R)) if v >= 0]
        return {
            'rows': (y1, y2),
            'cols': ((xL1, xL2), (xC1, xC2), (xR1, xR2)),
            'depths': (d_L, d_C, d_R),
            'status': (st_L, st_C, st_R),
            'clear_count': clear_count,
            'advisory': advisory,
            'auto_suggest': auto_suggest,
            'finite_depths_m': finite,
            'clear_threshold_m': float(clear_t),
            'blocked_threshold_m': float(blocked_t),
        }
    except Exception:
        return None


# The encoding the image is sent in. JPEG at the configured quality is what the model receives, and
# it is not a command line choice. `--encode jpeg|png` stood here until 23 August 2026 and was never
# given a value other than its default: the two formats are not alternatives to be compared but a
# preferred encoding and a recovery one, and PNG remains available to `_call_vlm_with_fallbacks` for
# exactly that purpose.
PRIMARY_ENCODING = "jpeg"


def _image_transform(encoding: str, jpeg_quality: int, longest_side_px: int) -> dict:
    """How the image sent to the model was prepared, as the prompt packet records it.

    Takes the attempt that succeeded rather than reading the command line, because the two can
    differ: a rejected image is retried at a lower quality and then at a smaller size, and until
    23 August 2026 the packet stated the first attempt whichever one the model actually answered.

    `jpeg_quality` is null for a PNG. The frozen schema requires that, and the runtime wrote the
    configured number under either encoding, so a PNG attempt produced a packet the schema refuses.
    """
    return {
        "longest_side_px": int(longest_side_px),
        "encoding": str(encoding),
        "jpeg_quality": int(jpeg_quality) if encoding == "jpeg" else None,
    }


def _call_vlm_with_fallbacks(endpoint: str, base_img: np.ndarray, args, *,
                             system: Optional[str] = None, unconstrained: bool = False,
                             max_tokens: Optional[int] = None,
                             grammar: Optional[str] = None,
                             timeout_s: Optional[float] = None) -> tuple[str, dict]:
    """Try multiple encodings/sizes when server says 'failed to process image'.

    Returns the reply and a description of the attempt that produced it, in the shape the prompt
    packet's `image.transform` takes. The second value exists because a fallback changes both the
    encoding and, in the last case, the resolution, and the record must state the image the model
    answered rather than the one first offered.
    """
    attempts: list[tuple[str, int, int]] = []  # (fmt, quality, size)
    # Longest side; Qwen3-VL handles non-square input natively
    size0 = int(getattr(args, 'image_size', 448) or 448)
    # Primary attempt: the preferred encoding at the configured quality.
    attempts.append((PRIMARY_ENCODING, int(getattr(args, 'jpeg_quality', 70) or 70), size0))
    # Fallbacks: png same size, then jpeg qualities, then smaller size png
    attempts.extend([
        ('png', 0, size0),
        ('jpeg', 75, size0),
        ('jpeg', 60, size0),
        ('png', 0, 224),
    ])
    last_err: Optional[Exception] = None
    for fmt, q, sz in attempts:
        try:
            img = resize_for_vlm(base_img, sz)
            # Stash user text for payload build without threading globals
            # We attach it temporarily to args for _build_payload_from_image_array
            payload, desc = _build_payload_from_image_array(
                img, fmt, q, args,
                system=system, unconstrained=unconstrained, max_tokens=max_tokens,
                grammar=grammar,
            )
            content = call_vlm(
                endpoint, payload,
                timeout=float(timeout_s if timeout_s
                              else getattr(args, "vlm_timeout_s", 20.0)),
            )
            if desc != '':
                print(
                    f"[vlm_on_change_qwen] VLM accepted image encoding: {desc}")
            if (fmt, q, sz) != attempts[0]:
                print(f"[vlm_on_change_qwen] the first encoding was refused; the model answered on "
                      f"{fmt}/q{q}/sz{sz}, and the prompt packet records that.")
            return content, _image_transform(fmt, q, sz)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if 'failed to process image' in msg:
                print(
                    f"[vlm_on_change_qwen] VLM could not process image with {fmt}/q{q}/sz{sz}; trying next...")
                continue
            # Different error; re-raise immediately
            raise
    # Exhausted attempts
    if last_err:
        raise last_err
    raise RuntimeError(
        'VLM image processing failed with all encodings (unexpected)')


# One pooled connection for the whole run. A fresh connection per request costs
# ~2.2 s extra against "localhost" on Windows, which dwarfs inference itself.
_VLM_SESSION = requests.Session()


def call_vlm(endpoint: str, payload: dict, timeout: int | tuple = 45) -> str:
    url = endpoint.rstrip('/') + '/v1/chat/completions'
    r = _VLM_SESSION.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(
            f"VLM request failed: {r.status_code} {r.text[:200]}")
    j = r.json()
    return j["choices"][0]["message"]["content"]


def _wrap_text(text: str, width: int = 70):
    words = text.split()
    lines, cur, cur_len = [], [], 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) > width:
            lines.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += len(w) + (1 if len(cur) > 1 else 0)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _is_drawn(item: dict, args) -> bool:
    """Whether one detection gets a bounding box.

    Both display paths call this. The web interface and the OpenCV window applied the same condition
    written out twice, so a change to one silently produced two different pictures of the same scene.

    The default is objects confirmed as moving. Drawing every detection was tried on 22 August 2026
    and reverted the same day: under the Open Images vocabulary a room yields many more detections
    than under COCO, and a screen covered in boxes over stationary furniture hides the one box that
    matters, which is the thing that moved. A stationary object still reaches the description and
    still stops the walker; it simply is not outlined.

    `--boxes all` draws every detection, which is what a detector or ontology problem needs, since a
    label mapped to the wrong group is otherwise invisible. `--boxes none` draws nothing.

    `--debug_objects` is the older spelling of `--boxes all` and still works.
    """
    mode = getattr(args, "boxes", "moving")
    if getattr(args, "debug_objects", False):
        mode = "all"
    if mode == "none":
        return False
    if mode == "moving":
        return bool(item.get("display_bounding_box"))
    return item.get("bbox_xyxy") is not None


def _parse_bool_argument(value: str | bool) -> bool:
    """Parses an explicit command-line Boolean value."""
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"true", "1", "yes", "on"}:
        return True
    if normalised in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


# The blocks of the request catalogue that must be present and complete before a run may start, and
# the fields each must carry. Every one of them is read on a live path with a bare subscript, so an
# absent block is a crash rather than a degradation.
REQUIRED_CATALOGUE_REQUESTS = ("AUTO_GUIDANCE", "MORE_DETAIL", "REASSESS")
REQUIRED_CATALOGUE_BLOCKS = {
    "question_answer": ("prompt_profile_id", "fixed_instruction"),
    "unconstrained_diagnostic": ("system_prompt", "fixed_instruction"),
}


def load_request_catalogue(path) -> dict:
    """Reads the request catalogue and refuses an incomplete one before the run begins.

    Until 23 August 2026 this check named only the three request profiles. It was written when
    those three were the whole file: `unconstrained_diagnostic` arrived with the diagnostic mode
    and `question_answer` was rewritten when the topic routing was withdrawn, and neither addition
    extended the check. A catalogue missing either therefore started, ran, and failed on the first
    typed question with a KeyError, part way through a session with the camera running.

    Extracted from `main` at the same time so that the check can be tested. It could not be before,
    being one branch inside a function that opens a camera.

    Raises ValueError describing the first problem found. The caller wraps it with the path.
    """
    catalogue = json.loads(Path(path).read_text(encoding="utf-8"))
    if catalogue.get("catalogue_version") != "hdsg.request_catalogue.v1":
        raise ValueError("The catalogue version is not hdsg.request_catalogue.v1.")

    requests = catalogue.get("requests")
    if not isinstance(requests, dict) or not set(REQUIRED_CATALOGUE_REQUESTS).issubset(requests):
        raise ValueError("The required evaluated request profiles are absent.")
    for name in REQUIRED_CATALOGUE_REQUESTS:
        entry = requests[name]
        if not isinstance(entry, dict) or not all(
                str(entry.get(field) or "").strip()
                for field in ("response_mode", "prompt_profile_id", "fixed_instruction")):
            raise ValueError(f"The request profile is incomplete: {name}")

    for block, fields in REQUIRED_CATALOGUE_BLOCKS.items():
        entry = catalogue.get(block)
        if not isinstance(entry, dict) or not all(
                str(entry.get(field) or "").strip() for field in fields):
            raise ValueError(f"The catalogue block is absent or incomplete: {block}")

    for field in ("composed_system_prompt", "composed_system_prompt_id"):
        if not str(catalogue.get(field) or "").strip():
            raise ValueError(f"The catalogue field is absent or empty: {field}")

    return catalogue


def main():
    """Runs the authoritative HDSG sensing, generation, validation, and display path."""
    ap = argparse.ArgumentParser(
        description="HDSG RealSense D455f depth, YOLO grounding, and constrained VLM guidance"
    )
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen3-vl-4b-instruct")
    ap.add_argument("--model_hash", default=None)
    ap.add_argument("--quantisation", default="Q4_K_M")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--vlm_timeout_s", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--image_size", type=int, default=448)
    ap.add_argument("--jpeg_quality", type=int, default=70)
    ap.add_argument("--process_hz", type=float, default=8.0)
    ap.add_argument("--det_model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument(
        "--ui", choices=["web", "opencv"], default="web",
        help="display and input sink used when --show is set",
    )
    ap.add_argument("--ui_host", default="127.0.0.1")
    ap.add_argument("--ui_port", type=int, default=8321)
    ap.add_argument("--debug_lanes", action="store_true")
    ap.add_argument(
        "--boxes", choices=["all", "moving", "none"], default="moving",
        help="which detections are drawn: only those confirmed moving (the default), every one, "
             "or none",
    )
    # The older spelling of `--boxes all`, retained so existing command lines keep working.
    ap.add_argument("--debug_objects", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mirror_view", action="store_true")
    ap.add_argument("--no_mirror_tag", action="store_true")
    ap.add_argument("--clear_threshold_m", type=float, default=None,
                    help="override the sector clear distance; the default comes from "
                         "sector.clear_at_or_above_m in pipeline.yaml")
    ap.add_argument("--sector_choice_tolerance_m", type=float, default=0.10)
    ap.add_argument("--movement_threshold_m", type=float, default=0.12)
    ap.add_argument("--stationary_threshold_m", type=float, default=0.05)
    ap.add_argument("--movement_confirmation_observations", type=int, default=4)
    ap.add_argument("--more_detail_freshness_s", type=float, default=5.0)
    ap.add_argument("--reassessment_cooldown_s", type=float, default=1.5)
    ap.add_argument("--post_reorientation_stable_observations", type=int, default=4)
    ap.add_argument("--post_reorientation_max_variation_m", type=float, default=0.10)
    default_telemetry_dir = Path(__file__).resolve().parents[1] / "logs"
    ap.add_argument("--telemetry_dir", type=Path, default=default_telemetry_dir)
    ap.add_argument(
        "--evaluate", nargs="?", const=True, default=False, type=_parse_bool_argument,
        help="enable complete JSONL evaluation telemetry",
    )
    ap.add_argument(
        "--eval_name", "--eval-name", dest="eval_name", default=None,
        help="short scenario name used to group evaluation runs",
    )
    ap.add_argument(
        "--record_rgbd", nargs="?", const=True, default=True, type=_parse_bool_argument,
        help="write synchronised RGB and depth per observation during an evaluation run, "
             "so the run can be replayed under other conditions with identical input",
    )
    default_catalogue = Path(__file__).resolve().parents[1] / "config" / "hdsg_request_catalogue.v1.json"
    default_route_grammar = Path(__file__).resolve().parents[1] / "config" / "hdsg.question_route.v1.gbnf"
    ap.add_argument("--request_catalogue", type=Path, default=default_catalogue)
    ap.add_argument("--route_grammar", type=Path, default=default_route_grammar,
                    help="the question admission classifier constraint")
    default_caption_grammar = (
        Path(__file__).resolve().parents[1] / "config" / "hdsg.vlm_caption.v1.gbnf"
    )
    ap.add_argument("--caption_grammar", type=Path, default=default_caption_grammar,
                    help="the composed caption constraint")
    ap.add_argument("--caption_max_tokens", type=int, default=400,
                    help="token budget for a composed caption. The reply carries prose and a "
                         "declaration for every number in it. A budget too small truncates the "
                         "JSON and the reply is discarded as a parse failure. The deployed model "
                         "settles at around 240, so this is headroom rather than a target.")
    ap.add_argument("--caption_timeout_s", type=float, default=45.0,
                    help="request timeout for a composed caption. Larger than --vlm_timeout_s "
                         "because the detector runs continuously on the same GPU and the guidance "
                         "and question workers can be generating at once, so a call that takes "
                         "eight seconds on an idle device takes considerably longer in a live run.")
    ap.add_argument("--answer_questions", nargs="?", const=True, default=True,
                    type=_parse_bool_argument,
                    help="answer typed questions from the web UI chat box; when false a question "
                         "receives the out-of-scope reply and no model call is made")
    ap.add_argument("--unconstrained", action="store_true",
                    help="DIAGNOSTIC. Answer typed questions by sending the frame and the text "
                         "straight to the model with no routing, no permitted-fact packet, no "
                         "grammar and no entailment gate. Shows what the same model says without "
                         "the architecture. Answers are ungrounded by construction and a run "
                         "started with this flag is not an evaluation run. The guidance caption is "
                         "unaffected and stays deterministic.")
    args = ap.parse_args()

    if args.evaluate:
        if not args.eval_name:
            ap.error("--eval_name is required when --evaluate is true")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.eval_name):
            ap.error("--eval_name must contain only letters, numbers, underscores, or hyphens")

    if rs is None:
        raise RuntimeError("pyrealsense2 is required for the canonical D455f implementation.")
    if args.unconstrained:
        print("[hdsg] " + "=" * 68)
        print("[hdsg] UNCONSTRAINED DIAGNOSTIC MODE")
        print("[hdsg] Typed questions bypass admission, the permitted-fact packet, the grammar")
        print("[hdsg] and the entailment gate. Answers are ungrounded by construction.")
        print("[hdsg] This run is not evaluation evidence. The caption stays deterministic.")
        print("[hdsg] " + "=" * 68)

    # The caption grammar constrains the reply's structure while leaving the caption itself free.
    # It is the only constraint the release path applies, and the packet records its digest.
    try:
        caption_grammar_text: Optional[str] = args.caption_grammar.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"The composed caption constraint could not be loaded: {args.caption_grammar}"
        ) from error
    constraint_hash = hdsg.sha256_file(args.caption_grammar)
    print("[hdsg] generation: composed captions, values checked exactly against measurement")

    route_grammar_text: Optional[str] = None
    if args.answer_questions and not args.unconstrained:
        try:
            route_grammar_text = args.route_grammar.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(
                f"The question admission constraint could not be loaded: {args.route_grammar}"
            ) from error
    try:
        request_catalogue = load_request_catalogue(args.request_catalogue)
        # The catalogue is the single source for the text sent to the model, so the digest the
        # prompt packet records cannot name text the model never received.
        composed_system = str(request_catalogue["composed_system_prompt"])
        composed_system_id = str(request_catalogue["composed_system_prompt_id"])
        args.system = composed_system
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as error:
        raise RuntimeError(
            f"The approved request catalogue could not be loaded: {args.request_catalogue}"
        ) from error
    # Which weights the endpoint has actually loaded, asked of the endpoint rather than taken from
    # the command line.
    #
    # Added 23 August 2026, after a compliance probe was run against the wrong model: a server
    # started for earlier work was still listening on port 8080, and the run recorded
    # "qwen3-vl-4b-instruct" because that is what was typed. A log must state which weights produced
    # it, not which name someone passed, or a result cannot be attributed after the fact. The four
    # runs recorded on 22 August 2026 all used Qwen2.5-VL-3B, confirmed 23 August 2026 from
    # recollection rather than from any log.
    endpoint_model_path = None
    endpoint_model_hash = None
    try:
        import requests as _requests

        _reply = _requests.get(args.endpoint.rstrip("/") + "/v1/models", timeout=10.0)
        _reply.raise_for_status()
        _body = _reply.json()
        _entries = _body.get("models") or _body.get("data") or []
        if _entries:
            _first = _entries[0]
            endpoint_model_path = str(
                _first.get("model") or _first.get("id") or _first.get("name") or ""
            ) or None
    except Exception as error:
        print(f"[hdsg] warning: the endpoint did not report its model ({error}). "
              "The run header cannot state which weights answered.")
    if endpoint_model_path:
        # Hashed when the path names a file this machine can read. A remote endpoint reports a path
        # that does not resolve here, and the path alone is still worth recording.
        _candidate = Path(endpoint_model_path)
        if _candidate.is_file():
            endpoint_model_hash = hdsg.sha256_file(_candidate)
        print(f"[hdsg] endpoint model: {endpoint_model_path}")
        print("[hdsg] confirm this is the intended model before treating the run as evidence.")

    # `generation.model_hash` reaches every prompt packet in the run and is read as the fingerprint
    # of the weights that answered. It is therefore taken from the file the endpoint reported, when
    # that file is on this machine, in preference to anything supplied on the command line.
    #
    # The remaining two cases are worse and are marked as such. A digest given with --model_hash is
    # an assertion by the person starting the run, and if none is given the field falls back to a
    # digest of the model's name, which has the shape of a fingerprint and the content of a label.
    # Both were silent until 23 August 2026. The run header keeps `model_hash` and
    # `endpoint_model_sha256` separately so the two can still be compared.
    if endpoint_model_hash:
        args.model_hash = endpoint_model_hash
    elif args.model_hash:
        print("[hdsg] warning: the model digest is the one supplied on the command line. The "
              "endpoint's own weights were not readable from this machine, so nothing checked it.")
    else:
        args.model_hash = hdsg.sha256_text(f"unverified-model:{args.model}")
        print("[hdsg] warning: no model digest is available. The records carry a digest of the "
              "model's name, not of its weights, and cannot attribute the run to a set of weights.")

    cfg = sw.load_yaml(sw.PIPELINE_CFG)
    mapper = sw.OntologyMapper(sw.ONTOLOGY_CFG)
    # The distance at which a sector is declared BLOCKED, read from the named safety threshold.
    #
    # It was read from `depth.metric_bins_m.very_close[1]` until 23 August 2026. That band holds the
    # same number, 0.70, but it is a presentation band: it decides whether a caption says
    # "very close", and retuning it for readability silently moved the distance at which the walker
    # stops. The two are now separate, and the name here says what the number is for.
    try:
        blocked_threshold_m = float(
            cfg["risk_rules_baseline"]["stop"]["nearest_obstacle_m_lt"]
        )
    except Exception:
        blocked_threshold_m = 0.7
    # The same fractions that decide an object's bearing decide the sector columns, so the sector a
    # caption names is the sector whose depth was measured.
    try:
        sector_left_max = float(cfg["bearing"]["left_max"])
        sector_right_min = float(cfg["bearing"]["right_min"])
    except Exception:
        sector_left_max, sector_right_min = 1.0 / 3.0, 2.0 / 3.0
    # Which rows the clearances are measured across. Hardcoded in three places until 23 August 2026.
    try:
        sector_band_top = float(cfg["sector"]["band_top_fraction"])
        sector_band_bottom = float(cfg["sector"]["band_bottom_fraction"])
    except Exception:
        sector_band_top = hdsg.SECTOR_BAND_TOP_FRACTION
        sector_band_bottom = hdsg.SECTOR_BAND_BOTTOM_FRACTION
    # The remaining motion thresholds, read from the configuration and passed to every consumer, so
    # the numbers a run is judged by are the numbers the run was configured with. The command line
    # can still override the sector clear distance for a one-off; nothing else takes an override,
    # because a threshold that can be set two ways is a threshold that will be recorded wrongly.
    try:
        object_caution_below_m = float(
            cfg["risk_rules_baseline"]["caution"]["nearest_obstacle_m_lt"]
        )
    except Exception:
        object_caution_below_m = hdsg.OBJECT_CAUTION_BELOW_M
    try:
        hazard_stop_at_or_below_m = float(
            cfg["risk_rules_baseline"]["stop"]["hazard_within_m_lte"]
        )
    except Exception:
        hazard_stop_at_or_below_m = hdsg.HAZARD_STOP_AT_OR_BELOW_M
    if args.clear_threshold_m is None:
        try:
            args.clear_threshold_m = float(cfg["sector"]["clear_at_or_above_m"])
        except Exception:
            args.clear_threshold_m = hdsg.SECTOR_CLEAR_AT_OR_ABOVE_M
    print(f"[hdsg] thresholds: blocked below {blocked_threshold_m:.2f} m, "
          f"clear at or above {args.clear_threshold_m:.2f} m, "
          f"object caution below {object_caution_below_m:.2f} m, "
          f"hazard stop at or below {hazard_stop_at_or_below_m:.2f} m")
    runtime_configuration_hash = hdsg.sha256_text(json.dumps({
        "pipeline_hash": hdsg.sha256_file(sw.PIPELINE_CFG),
        "request_catalogue_hash": hdsg.sha256_file(args.request_catalogue),
        "constraint_hash": constraint_hash,
        # The grammar is loaded only when questions are answered and the diagnostic mode is off, so
        # both conditions are named here. Until 23 August 2026 this tested only the first, and a
        # diagnostic run recorded the digest of a constraint it had never loaded. A run header that
        # names a constraint not in force is the defect the run header exists to prevent.
        "route_constraint_hash": (
            hdsg.sha256_file(args.route_grammar)
            if args.answer_questions and not args.unconstrained else None
        ),
        "clear_threshold_m": args.clear_threshold_m,
        "sector_choice_tolerance_m": args.sector_choice_tolerance_m,
        "movement_threshold_m": args.movement_threshold_m,
        "stationary_threshold_m": args.stationary_threshold_m,
        "movement_confirmation_observations": args.movement_confirmation_observations,
        "more_detail_freshness_s": args.more_detail_freshness_s,
        "reassessment_cooldown_s": args.reassessment_cooldown_s,
        "post_reorientation_stable_observations": args.post_reorientation_stable_observations,
        "post_reorientation_max_variation_m": args.post_reorientation_max_variation_m,
    }, sort_keys=True))

    from queue import Empty, Full, Queue
    import threading
    from ultralytics import YOLO

    pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    rs_cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipe.start(rs_cfg)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)
    print(f"[hdsg] D455f depth scale = {depth_scale:.6f} m/unit")

    cap_q: "Queue[sw.FramePacket]" = Queue(maxsize=1)
    inf_in: "Queue[sw.FramePacket]" = Queue(maxsize=1)
    inf_out: "Queue[sw.InferPacket]" = Queue(maxsize=1)
    generation_q: "Queue[dict]" = Queue(maxsize=1)
    # Questions queue separately from guidance so that asking one never delays a guidance update.
    # A small backlog is allowed because a user can reasonably type a second question while the
    # first is still being answered; beyond that the main loop declines rather than blocking.
    question_q: "Queue[dict]" = Queue(maxsize=3)
    stop_evt = threading.Event()
    state_lock = threading.Lock()

    model = YOLO(args.det_model)
    # The detector's whole class vocabulary. A caption may not use the detector's own label for a
    # class it can recognise but did not report in the current observation. The bound is lexical: it
    # covers those labels and their enumerated plural forms, and a synonym passes, so a caption may
    # say "steps" where "stairs" would be refused. Section 10.10 of
    # `HDSG_VERIFIED_GENERATION_POLICY.md` measures the gap. A word outside this vocabulary is
    # ordinary language and is not policed.
    detector_classes = [str(name) for name in (getattr(model, "names", None) or {}).values()]
    if not detector_classes:
        # An empty vocabulary makes the object check a no-operation, so every class becomes
        # nameable and the property stops holding without anything failing. Treated as a startup
        # failure for the same reason a missing grammar is: silently bypassing the mechanism the
        # architectural claim rests on is worse than not running.
        raise RuntimeError(
            "The detector reported no class vocabulary, so a caption could name any object. "
            f"Check the weights given by --det_model: {args.det_model}."
        )
    print(f"[hdsg] detector vocabulary: {len(detector_classes)} classes")
    use_cuda = bool(torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available())
    if use_cuda:
        model.to("cuda")
    half_flag = bool(args.half and use_cuda)
    threading.Thread(
        target=sw.capture_thread,
        args=(pipe, align, depth_scale, cap_q, stop_evt, False, None, sw.IMU_MAX_DRAIN_PER_LOOP),
        daemon=True,
    ).start()
    threading.Thread(
        target=sw.inference_thread,
        args=(cfg, mapper, model, inf_in, inf_out, stop_evt, args.imgsz, args.conf,
              half_flag, None, True, "botsort.yaml"),
        daemon=True,
    ).start()
    # sw.KeyListener polls GetAsyncKeyState, which reports whether a key is physically down
    # regardless of which window has focus. That is correct for the OpenCV sink, which has no
    # text entry, and wrong for the web sink: typing "what is on my left" into the question box
    # would fire forward, left and backward from its w, a and s, "my" would fire More detail, and
    # the q in "question" would set quit_requested and end the run. The browser implements the
    # same shortcuts itself and guards them on the question box having focus, so with the web
    # sink the operating-system hook is both redundant and harmful.
    keyboard = (
        _NullKeyListener() if (args.show and args.ui == "web")
        else sw.KeyListener(poll_hz=120).start()
    )

    motion_tracker = hdsg.MotionTracker(
        movement_threshold_m=args.movement_threshold_m,
        stationary_threshold_m=args.stationary_threshold_m,
        confirmation_observations=max(2, args.movement_confirmation_observations),
    )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{run_stamp}"
    telemetry_path: Optional[Path] = None
    recorder: Optional[ObservationRecorder] = None
    recording_dir_ref: Optional[str] = None
    if args.evaluate:
        evaluation_dir = args.telemetry_dir / str(args.eval_name)
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = evaluation_dir / f"{run_id}.jsonl"
        if args.record_rgbd:
            # Kept beside the telemetry file and named after the run, so a log and its frames
            # travel together. The reference stored in each Fact Packet is relative to the
            # telemetry file's directory, which keeps a recording portable between machines.
            recording_dir_ref = run_id
            recorder = ObservationRecorder(
                evaluation_dir / run_id, depth_scale=depth_scale
            ).start()
            print(f"[hdsg] recording RGB-D to {(evaluation_dir / run_id)}")
    telemetry_lock = threading.Lock()

    def record(record_type: str, value: dict):
        if telemetry_path is None:
            return
        envelope = {
            "evaluation_name": args.eval_name,
            "record_type": record_type,
            "recorded_at_utc": hdsg.utc_now(),
            "record": value,
        }
        # Stamped only when the diagnostic mode is active, so the envelope shape of a normal run is
        # unchanged and its absence means a gated run. A log carrying this key is not evaluation
        # evidence: some of its answers never passed the entailment gate.
        if args.unconstrained:
            envelope["unconstrained_diagnostic_run"] = True
        with telemetry_lock:
            with telemetry_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(envelope, separators=(",", ":"), ensure_ascii=True) + "\n")

    # The first record in every evaluation log, so a result can be attributed to the software and
    # the weights that produced it without asking anyone what was running at the time.
    #
    # `model_id` is the name passed on the command line and `endpoint_model_path` is what the server
    # reported having loaded. Both are kept because they can disagree, and on 23 August 2026 they
    # did: a probe was run against Qwen2.5-VL-3B while every record said qwen3-vl-4b-instruct. Only
    # the second of the two is evidence.
    record("run_header", {
        "run_id": run_id,
        "started_at_utc": hdsg.utc_now(),
        "evaluation_name": args.eval_name,
        "model_id": args.model,
        "model_hash": args.model_hash,
        "endpoint": args.endpoint,
        "endpoint_model_path": endpoint_model_path,
        "endpoint_model_sha256": endpoint_model_hash,
        "detector_model": args.det_model,
        "detector_confidence": args.conf,
        "runtime_configuration_hash": runtime_configuration_hash,
        "request_catalogue_hash": hdsg.sha256_file(args.request_catalogue),
        "caption_constraint_hash": constraint_hash,
        "unconstrained_diagnostic_run": bool(args.unconstrained),
        "recording_dir": recording_dir_ref,
    })

    latest_release: Optional[dict] = None
    active_request_key: Optional[tuple[str, str, str]] = None
    generation_inflight = False
    generation_request_id: Optional[str] = None
    queued_request_id: Optional[str] = None
    reassessment_inflight = False
    reassessment_cooldown_until = 0.0

    def publish_release(release: dict, record_type: str = "authoritative_release"):
        nonlocal latest_release
        with state_lock:
            latest_release = release
        record(record_type, release)

    def generate_candidate(fact_packet: dict, prompt_packet: dict, image,
                           fixed_instruction: str) -> tuple[Optional[dict], list[str],
                                                            Optional[str], list[dict]]:
        """Runs one generation and gates the caption it returns.

        Returns the candidate, its reason codes, the raw reply and the scored assertions. Both call
        sites share this so the generation path has one implementation.

        The prompt packet's `image.transform` is corrected here, in place, to the attempt the model
        actually answered on. Both callers therefore record the packet after this returns, not
        before, so that a run in which an image was refused and retried says so.
        """
        raw_response: Optional[str] = None
        scored: list[dict] = []
        candidate: Optional[dict] = None
        codes: list[str] = []
        try:
            args._user_txt_for_payload = hdsg_composed.build_composed_prompt(
                prompt_packet, fact_packet, fixed_instruction
            )
            raw_response, sent_transform = _call_vlm_with_fallbacks(
                args.endpoint, image, args,
                system=composed_system,
                grammar=caption_grammar_text,
                max_tokens=args.caption_max_tokens,
                timeout_s=args.caption_timeout_s,
            )
            prompt_packet["image"]["transform"].update(sent_transform)
            candidate, codes = hdsg_composed.parse_caption_candidate(raw_response)
            if candidate is None and not str(raw_response).rstrip().endswith("}"):
                # A grammar-constrained reply that stops before its closing brace ran out of
                # tokens rather than being malformed. The two are indistinguishable in the
                # reason code, so the distinction is drawn here.
                print(f"[hdsg] caption truncated at {len(raw_response or '')} characters; "
                      f"raise --caption_max_tokens above {args.caption_max_tokens}")
            if candidate is not None:
                record("vlm_caption", candidate)
                gate_codes, scored = hdsg_composed.validate_caption_candidate(
                    candidate, prompt_packet, fact_packet,
                    detector_classes=detector_classes,
                )
                codes = list(codes) + list(gate_codes)
                if gate_codes:
                    # A gate rejection is otherwise silent: the user sees the deterministic
                    # fallback, which for several sector states is worded identically to an
                    # accepted caption, so nothing on screen indicates a rejection occurred.
                    print(f"[hdsg] caption rejected {gate_codes}: "
                          f"{str(candidate.get('caption'))[:160]}")
        except Exception as error:
            print(f"[hdsg] constrained generation failed: {error}")
            return None, [hdsg.generation_failure_code(error)], raw_response, scored
        finally:
            if hasattr(args, "_user_txt_for_payload"):
                delattr(args, "_user_txt_for_payload")
        return candidate, codes, raw_response, scored

    def make_release(fact_packet: dict, prompt_packet: dict, release_id: str,
                     candidate: Optional[dict], failure_codes: list[str],
                     scored: list[dict], **kwargs) -> dict:
        """Builds the release from an accepted caption or from the deterministic layer."""
        return hdsg_composed.build_composed_release(
            fact_packet, prompt_packet, release_id=release_id, candidate=candidate,
            scored_assertions=scored, failure_codes=failure_codes, **kwargs
        )

    def generation_worker():
        nonlocal generation_inflight, generation_request_id, queued_request_id
        nonlocal reassessment_inflight, reassessment_cooldown_until
        while not stop_evt.is_set():
            try:
                request = generation_q.get(timeout=0.05)
            except Empty:
                continue
            fact_packet = request["fact_packet"]
            request_id = fact_packet["interaction"]["request_id"]
            with state_lock:
                generation_inflight = True
                generation_request_id = request_id
                if queued_request_id == request_id:
                    queued_request_id = None
            prompt_packet = request["prompt_packet"]
            request_key = request["request_key"]
            candidate = None
            failure_codes: list[str] = []
            raw_response: Optional[str] = None
            started_ms = hdsg.monotonic_time_ms()
            responded_ms: Optional[int] = None
            catalogue_entry = request_catalogue["requests"][fact_packet["interaction"]["request_id"]]
            candidate, failure_codes, raw_response, scored_assertions = generate_candidate(
                fact_packet, prompt_packet, request["image"],
                str(catalogue_entry["fixed_instruction"]),
            )
            responded_ms = hdsg.monotonic_time_ms()
            # Recorded here rather than when the request was queued, so that `image.transform`
            # states the encoding and size the model answered on. `generate_candidate` corrects it
            # in place, and the packet is the same object the main loop built. A request dropped
            # from a full queue is now not recorded at all, which is the accurate outcome: no
            # generation was attempted on it.
            record("restricted_prompt_packet", prompt_packet)
            # Recorded whenever the gate scored anything, which now includes a caption that
            # declared no measurement and named an object the detector had not reported. That case
            # previously wrote nothing at all, so the evidence for it existed only as a reason code
            # on the release and never reached the assertions log the evaluation reads.
            if scored_assertions:
                record("declared_assertions", {
                    "event_id": fact_packet["identity"]["event_id"],
                    "assertions": scored_assertions,
                    "summary": hdsg_composed.assertion_summary(scored_assertions),
                })

            # HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md section 7. An AUTO_GUIDANCE candidate
            # is stale only when the guidance signature confirmed at response time differs from
            # the one the request was made against, not merely because a newer event has since
            # been allocated. MORE_DETAIL and REASSESS keep the previous event-identity check,
            # since a closed request's response is tied to the specific request that produced it
            # rather than to a held explanation.
            with state_lock:
                if fact_packet["interaction"]["request_id"] == "AUTO_GUIDANCE":
                    still_active = fact_packet["interaction"]["current_guidance_signature"] == confirmed_signature
                else:
                    still_active = active_request_key == request_key
            if not still_active:
                failure_codes = ["RG_STALE_CANDIDATE"]
                candidate = None
            release = make_release(
                fact_packet, prompt_packet, request["release_id"],
                candidate, failure_codes, scored_assertions,
            )
            record("generation_response", hdsg.build_generation_response_record(
                fact_packet,
                release,
                raw_response=raw_response,
                candidate=candidate,
                superseded=not still_active,
                queued_ms=request["queued_ms"],
                started_ms=started_ms,
                responded_ms=responded_ms,
                released_ms=hdsg.monotonic_time_ms(),
            ))
            if still_active:
                publish_release(release)
            else:
                record("stale_release_not_displayed", release)
            if fact_packet["interaction"]["request_id"] == "REASSESS":
                reassessment_inflight = False
                reassessment_cooldown_until = time.monotonic() + args.reassessment_cooldown_s
            with state_lock:
                generation_inflight = False
                generation_request_id = None

    threading.Thread(target=generation_worker, daemon=True).start()

    def answer_question(request: dict) -> tuple[str, Optional[str], str, bool, str]:
        """Admits one question and produces its answer.

        Returns the answer text, the admission outcome, the stage that settled it, whether the
        answer reached generation, and the release mode behind it.

        The release mode is reported because the approved templates and the deterministic fallback
        are word for word identical for several sector states, so the answer alone does not reveal
        whether the model contributed to it.
        """
        question = request["question"]
        route = request["route"]
        resolved_by = request["resolved_by"]

        # Diagnostic mode. Nothing below the model call applies: no routing, no permitted facts, no
        # grammar, no gate. The reply is shown as the model produced it, which is the point of the
        # comparison. The caption path is untouched and still carries only released text.
        if args.unconstrained:
            entry = request_catalogue["unconstrained_diagnostic"]
            try:
                args._user_txt_for_payload = (
                    f"{entry['fixed_instruction']}\n\nThe person asked: {question}"
                )
                # The diagnostic mode builds no prompt packet, so the attempt that succeeded has
                # nowhere to be recorded and is discarded.
                raw, _ = _call_vlm_with_fallbacks(
                    args.endpoint, request["image"], args,
                    system=str(entry["system_prompt"]),
                    unconstrained=True,
                    max_tokens=int(entry.get("max_tokens") or args.max_tokens),
                )
            except Exception as error:
                print(f"[hdsg] unconstrained answer failed: {error}")
                return (f"The model did not answer: {error}", "UNCONSTRAINED", "UNCONSTRAINED", False,
                        "UNGATED_FAILED")
            finally:
                if hasattr(args, "_user_txt_for_payload"):
                    delattr(args, "_user_txt_for_payload")
            record("unconstrained_response", {
                "question_text_sha256": hdsg.sha256_text(question),
                "system_prompt_hash": hdsg.sha256_text(str(entry["system_prompt"])),
                "observation_id": request["fact_packet"]["identity"]["observation_id"],
                "raw_response": raw,
                "gated": False,
            })
            return str(raw).strip(), "UNCONSTRAINED", "UNCONSTRAINED", True, "UNGATED"

        # The admission classifier. Reached unless the keyword filter already recognised a request
        # for a fresh look. The grammar admits three tokens and nothing else, so a crafted question
        # can at worst be admitted when it should have been declined, and an admitted question is
        # still answered through the unchanged gate.
        if route is None:
            # The classifier is safe to point at the person's raw text only because the decoder
            # cannot emit anything outside the three tokens. Without the constraint the call is an
            # unconstrained model reading untrusted input, which is a different thing entirely.
            # `or ""` stood here until 23 August 2026 and would have sent no constraint at all. The
            # caption path already refuses a missing constraint rather than degrading; this is the
            # same guarantee and is now held to the same standard.
            if not route_grammar_text:
                raise RuntimeError("The approved question admission constraint is unavailable.")
            payload = build_text_chat_payload(
                model=args.model,
                system=questions.CLASSIFIER_SYSTEM_PROMPT,
                text=questions.build_classifier_prompt(question),
                grammar=route_grammar_text,
                temperature=0.0,
                top_p=1.0,
                max_tokens=48,
            )
            try:
                raw_route = call_vlm(args.endpoint, payload,
                                     timeout=float(getattr(args, "vlm_timeout_s", 20.0)))
            except Exception as error:
                print(f"[hdsg] question classification failed: {error}")
                return questions.OUT_OF_SCOPE_TEXT, None, "ADMISSION_CLASSIFIER", False, "CATALOGUE_REPLY"
            route, parse_error = questions.parse_route(raw_route)
            resolved_by = "ADMISSION_CLASSIFIER"
            if route is None:
                # A reply the grammar should have made impossible. Declining is the conservative
                # outcome, since an unreadable admission decision is no decision at all.
                print(f"[hdsg] admission reply unreadable: {parse_error}")
                return questions.OUT_OF_SCOPE_TEXT, None, resolved_by, False, "CATALOGUE_REPLY"

        if route == "OUT_OF_SCOPE":
            return questions.OUT_OF_SCOPE_TEXT, route, resolved_by, False, "CATALOGUE_REPLY"
        if route == "REASSESS":
            # Handed to the existing control rather than answered. The main loop owns the
            # reassessment state machine, so the worker only reports the redirection.
            return questions.REASSESS_TEXT, route, resolved_by, False, "CATALOGUE_REPLY"

        fact_packet = request["fact_packet"]
        answer_entry = request_catalogue["question_answer"]

        # No requirement set is supplied, so build_prompt_packet runs its own More detail
        # construction: a clause per sector and per detected object, which is the whole measured
        # scene. Selecting a subset in advance is what the withdrawn topic routing did, and the
        # entailment gate checks the same property afterwards without having to guess first.
        prompt_id = allocate("prompt", "prompt")
        prompt_packet = hdsg.build_prompt_packet(
            fact_packet,
            prompt_id=prompt_id,
            model_id=args.model,
            model_hash=args.model_hash,
            quantisation=args.quantisation,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.caption_max_tokens,
            system_prompt=composed_system,
            constraint_hash=constraint_hash,
            prompt_profile_id=str(answer_entry["prompt_profile_id"]),
            system_prompt_id=composed_system_id,
            expected_response_schema=hdsg.CAPTION_SCHEMA,
        )
        prompt_packet["image"]["transform"].update(
            _image_transform(PRIMARY_ENCODING, args.jpeg_quality, args.image_size))

        candidate, failure_codes, raw_response, scored_assertions = generate_candidate(
            fact_packet, prompt_packet, request["image"],
            questions.build_answer_instruction(
                question, str(answer_entry["fixed_instruction"])
            ),
        )
        # Recorded after the call, so that `image.transform` states the encoding and size the model
        # answered on rather than the first one offered. See `generate_candidate`.
        record("restricted_prompt_packet", prompt_packet)
        if scored_assertions:
            record("declared_assertions", {
                "event_id": fact_packet["identity"]["event_id"],
                "assertions": scored_assertions,
                "summary": hdsg_composed.assertion_summary(scored_assertions),
            })

        # The answer passes through the unchanged entailment gate and the unchanged release
        # builder, so a candidate citing a fact it was not given is rejected here exactly as a
        # guidance candidate would be, and a rejection still yields the deterministic text.
        release = make_release(
            fact_packet, prompt_packet, allocate("release", "release"),
            candidate, failure_codes, scored_assertions,
        )
        record("question_release", release)
        if release["verification"]["gate_outcome"] == "ACCEPTED":
            answer = questions.answer_text_from_release(release)
        else:
            # A rejected candidate falls back to the permitted facts rather than to the release
            # builder's own fallback, which describes only the action binding and would answer a
            # narrower question than the one asked.
            answer = questions.deterministic_answer(fact_packet, prompt_packet["requirements"])
        answer = questions.with_action_prefix(release, answer)
        # The primary reason code accompanies the mode, because "fell back" without saying why is
        # not enough to tell a rejected caption from an unavailable model while testing.
        verification = release["verification"]
        mode = verification["release_mode"]
        if mode != "VLM_ACCEPTED":
            mode = f"{mode} · {verification['primary_reason_code']}"
        return answer, route, resolved_by, True, mode

    def question_worker():
        """Answers typed questions on their own thread.

        Questions run on a separate queue from guidance generation so that asking one never delays
        a guidance update and never displaces the guidance caption. The answer is delivered to the
        chat panel; the caption line continues to show the deterministic action and its reason,
        which is the transparency guarantee the caption exists to provide.
        """
        while not stop_evt.is_set():
            try:
                request = question_q.get(timeout=0.05)
            except Empty:
                continue
            try:
                answer, route, resolved_by, reached, mode = answer_question(request)
            except Exception as error:
                print(f"[hdsg] question handling failed: {error}")
                answer, route, resolved_by, reached, mode = (
                    questions.OUT_OF_SCOPE_TEXT, request["route"], request["resolved_by"],
                    False, "DETERMINISTIC_FALLBACK",
                )
            record("question_record", questions.build_route_record(
                request["question"], route, resolved_by, reached
            ))
            if web_ui is not None:
                web_ui.publish_chat_turn(request["question"], answer, route=route, mode=mode)

    # Started further down, once web_ui and allocate exist. The worker closes over both.

    sequence = {"event": 0, "observation": 0, "ticket": 0, "prompt": 0, "release": 0}

    def allocate(name: str, prefix: str) -> str:
        sequence[name] += 1
        return hdsg.next_identifier(prefix, sequence[name])

    def enqueue_request(fact_packet: dict, image: np.ndarray):
        nonlocal active_request_key, queued_request_id
        prompt_id = allocate("prompt", "prompt")
        request_id = fact_packet["interaction"]["request_id"]
        catalogue_entry = request_catalogue["requests"].get(request_id)
        # An `enabled` flag was tested here until 23 August 2026. It was never once set false in any
        # revision of the catalogue, nothing wrote it, and the two blocks added later never carried
        # it, so it described a capability the system did not have. Channels are switched with the
        # command line flags that are actually used, `--answer_questions` and `--unconstrained`, and
        # those are recorded in the run header. The profile must still exist.
        if not catalogue_entry:
            raise RuntimeError(f"The request profile is absent from the catalogue: {request_id}")
        if catalogue_entry.get("response_mode") != fact_packet["interaction"]["response_mode"]:
            raise RuntimeError(f"The request profile mode does not match the Fact Packet: {request_id}")
        prompt_packet = hdsg.build_prompt_packet(
            fact_packet,
            prompt_id=prompt_id,
            model_id=args.model,
            model_hash=args.model_hash,
            quantisation=args.quantisation,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.caption_max_tokens,
            system_prompt=composed_system,
            constraint_hash=constraint_hash,
            prompt_profile_id=str(catalogue_entry["prompt_profile_id"]),
            system_prompt_id=composed_system_id,
            expected_response_schema=hdsg.CAPTION_SCHEMA,
        )
        prompt_packet["image"]["transform"].update(
            _image_transform(PRIMARY_ENCODING, args.jpeg_quality, args.image_size))
        immediate_release_id = allocate("release", "release")
        final_release_id = allocate("release", "release")
        request_key = (
            fact_packet["identity"]["event_id"],
            fact_packet["identity"]["observation_id"],
            fact_packet["interaction"]["request_id"],
        )
        with state_lock:
            active_request_key = request_key
        record("full_fact_packet", fact_packet)

        immediate = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id=immediate_release_id,
            pending=True,
        )
        publish_release(immediate, "authoritative_interim_release")
        item = {
            "fact_packet": fact_packet,
            "prompt_packet": prompt_packet,
            "image": resize_for_vlm(image, args.image_size),
            "request_key": request_key,
            "release_id": final_release_id,
            # Marks the start of the pending window, which is when the interim release publishes
            # the deterministic action and the reason line becomes a placeholder.
            "queued_ms": hdsg.monotonic_time_ms(),
        }
        if generation_q.full():
            try:
                generation_q.get_nowait()
            except Empty:
                pass
        generation_q.put_nowait(item)
        with state_lock:
            queued_request_id = request_id

    current_ticket_id: Optional[str] = None
    active_intent = "NONE"
    last_key_event_ms = -1
    latest_color: Optional[np.ndarray] = None
    latest_depth: Optional[np.ndarray] = None
    display_color: Optional[np.ndarray] = None
    latest_objects: list[dict] = []
    latest_sector_facts: Optional[dict] = None
    latest_authority: Optional[dict] = None
    # None until the first depth frame arrives, and recorded as None in any packet built before it,
    # which is the honest value: no frame was measured, rather than a coverage of zero.
    latest_depth_valid_fraction: Optional[float] = None
    latest_observation_id: Optional[str] = None
    observation_images: dict[str, np.ndarray] = {}
    previous_selected_sector: Optional[str] = None
    # The most recently confirmed guidance signature and authority under the intent-triggered
    # explanation policy (HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md section 6). "Confirmed"
    # means the persistence period for the current rank transition has elapsed; it is not the
    # same thing as the latest raw per-frame authority, which updates every frame regardless.
    confirmed_signature: Optional[str] = None
    confirmed_authority: Optional[dict] = None
    pending_signature: Optional[str] = None
    pending_since = 0.0
    # The guidance signature of the live scene as of the most recent inference tick, and the
    # monotonic time that tick was captured. Updated every inference frame regardless of whether
    # it triggers generation, so a More detail request always sees the true current scene rather
    # than the signature of whichever event last called the model.
    latest_material_signature: Optional[str] = None
    latest_observation_captured_monotonic = 0.0
    # The signature for which a More detail request has already been issued. While the scene
    # remains at this signature, pressing More detail again does not call the model a second
    # time; it reports that the detailed description already covers the current scene.
    more_detail_signature: Optional[str] = None
    ui_notice: Optional[str] = None
    ui_notice_until = 0.0
    pending_reassessment = False
    pending_reassessment_input_method = "KEYBOARD_SHORTCUT"
    reorientation_required = False
    stabilising = False
    stable_statuses: Optional[tuple[str, str, str]] = None
    stable_depths: Optional[tuple[float, float, float]] = None
    stable_count = 0
    last_process = 0.0
    process_period = 1.0 / max(args.process_hz, 0.1)
    control_regions: dict[str, tuple[int, int, int, int]] = {}
    mouse_events = {"more_detail": False, "reassess": False, "choice": None}
    mouse_lock = threading.Lock()

    def on_mouse(event, x, y, _flags, _parameter):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        with mouse_lock:
            for control_id, (x1, y1, x2, y2) in control_regions.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if control_id == "MORE_DETAIL":
                        mouse_events["more_detail"] = True
                    elif control_id == "REASSESS":
                        mouse_events["reassess"] = True
                    elif control_id in {"LEFT", "RIGHT"}:
                        mouse_events["choice"] = control_id
                    break

    web_ui: Optional[WebInterface] = None
    use_web_ui = bool(args.show and args.ui == "web")
    use_opencv_ui = bool(args.show and args.ui == "opencv")
    if use_web_ui:
        web_ui = WebInterface(host=args.ui_host, port=args.ui_port).start()
        print(f"[hdsg] interface: {web_ui.url}")
    if use_opencv_ui:
        cv2.namedWindow("HDSG smart walker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("HDSG smart walker", on_mouse)

    if args.answer_questions and web_ui is not None:
        threading.Thread(target=question_worker, daemon=True).start()

    def trigger_type(previous: Optional[dict], current: dict) -> str:
        if previous is None:
            return "MOTION_INTENT_STARTED"
        for field, name in (
            ("motion_decision", "MOTION_DECISION_CHANGED"),
            ("selected_sector", "SELECTED_SECTOR_CHANGED"),
            ("scene_advisory", "SCENE_ADVISORY_CHANGED"),
        ):
            if previous.get(field) != current.get(field):
                return name
        if previous.get("action_binding") != current.get("action_binding"):
            return "BINDING_FACT_CHANGED"
        if previous.get("_measurement_state") != current.get("_measurement_state"):
            return "MEASUREMENT_VALIDITY_CHANGED"
        return "MOVING_OBJECT_CHANGED"

    def create_fact_packet(
        response_mode: str,
        request_id: str,
        event_trigger: str,
        observation_id: str,
        timestamp_ms: float,
        objects: list[dict],
        sectors: dict,
        authority: dict,
        input_method: str = "SYSTEM",
        control_id: Optional[str] = None,
    ) -> dict:
        event_id = allocate("event", "evt")
        return hdsg.build_fact_packet(
            run_id=run_id,
            event_id=event_id,
            observation_id=observation_id,
            ticket_id=current_ticket_id,
            timestamp_ms=timestamp_ms,
            intent=active_intent,
            trigger_type=event_trigger,
            request_id=request_id,
            response_mode=response_mode,
            previous_signature=confirmed_signature,
            objects=objects,
            sectors=sectors,
            authority=authority,
            mirror_view=args.mirror_view,
            detector_model=args.det_model,
            detector_confidence=args.conf,
            pipeline_config_path=sw.PIPELINE_CFG,
            ontology_path=sw.ONTOLOGY_CFG,
            clear_threshold_m=args.clear_threshold_m,
            blocked_threshold_m=blocked_threshold_m,
            # The geometry the clearances were actually measured with, not the intended geometry.
            sector_band_top_fraction=sector_band_top,
            sector_band_bottom_fraction=sector_band_bottom,
            sector_left_max_fraction=sector_left_max,
            sector_right_min_fraction=sector_right_min,
            sector_choice_tolerance_m=args.sector_choice_tolerance_m,
            motion_tracker=motion_tracker,
            object_caution_below_m=object_caution_below_m,
            hazard_stop_at_or_below_m=hazard_stop_at_or_below_m,
            depth_valid_fraction=latest_depth_valid_fraction,
            configuration_hash=runtime_configuration_hash,
            post_reorientation_stable_observations=args.post_reorientation_stable_observations,
            post_reorientation_max_variation_m=args.post_reorientation_max_variation_m,
            reassessment_cooldown_ms=args.reassessment_cooldown_s * 1000.0,
            more_detail_freshness_ms=args.more_detail_freshness_s * 1000.0,
            scenario_id=args.eval_name if args.evaluate else None,
            input_method=input_method,
            control_id=control_id,
            recording_dir=recording_dir_ref,
        )

    def publish_idle_release():
        """Publishes the IDLE_NO_INTENT release when no movement intent is expressed.

        HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md section 4: the caption is empty while the
        sector display continues on its own, unrelated channel. Before the first inference frame
        has arrived there is no sector or authority state to build a Fact Packet from; latest_release
        is still at its initial None in that case, so there is nothing to clear.
        """
        if latest_authority is None or latest_sector_facts is None or latest_observation_id is None:
            return
        packet = create_fact_packet(
            "AUTOMATIC", "AUTO_GUIDANCE", "MOTION_INTENT_STARTED", latest_observation_id,
            time.monotonic() * 1000.0, latest_objects, latest_sector_facts, latest_authority,
        )
        record("full_fact_packet", packet)
        release = hdsg.build_release(
            packet, {}, release_id=allocate("release", "release"), no_intent=True,
        )
        publish_release(release)

    print("[hdsg] running. W/A/S/D set intent, Space clears intent, M requests more detail, R reassesses, Q quits.")
    if telemetry_path is None:
        print("[hdsg] evaluation telemetry is disabled.")
    else:
        print(f"[hdsg] evaluation '{args.eval_name}': {telemetry_path}")
    try:
        while True:
            if use_opencv_ui:
                if hasattr(cv2, "pollKey"):
                    cv2.pollKey()
                else:
                    cv2.waitKey(1)
            else:
                # The OpenCV sink's waitKey doubles as the loop's pacing. Without a window there
                # is nothing to pump, so yield briefly instead of spinning on the input queues.
                time.sleep(0.005)

            try:
                frame_packet = cap_q.get_nowait()
                latest_color = frame_packet.color
                latest_depth = frame_packet.depth_m
                now = time.monotonic()
                if now - last_process >= process_period:
                    while not inf_in.empty():
                        try:
                            inf_in.get_nowait()
                        except Empty:
                            break
                    inf_in.put(frame_packet)
                    last_process = now
            except Empty:
                pass

            keys = keyboard.snapshot()
            with mouse_lock:
                mouse_more_detail = bool(mouse_events["more_detail"])
                mouse_reassess = bool(mouse_events["reassess"])
                mouse_choice = mouse_events["choice"]
                mouse_events.update({"more_detail": False, "reassess": False, "choice": None})

            # Browser input joins the same variables the OpenCV sink's mouse callback feeds, so
            # everything downstream is unaware of which sink a press came from. Intent arrives as
            # an explicit direction rather than as a key edge, so it is applied here directly.
            browser_intent: Optional[str] = None
            browser_questions: list[str] = []
            browser_quit = False
            if web_ui is not None:
                for event in web_ui.poll_events():
                    kind = event.get("type")
                    if kind == "quit":
                        browser_quit = True
                    elif kind == "control":
                        control_id = event.get("control_id")
                        if control_id == "MORE_DETAIL":
                            mouse_more_detail = True
                        elif control_id == "REASSESS":
                            mouse_reassess = True
                        elif control_id in {"LEFT", "RIGHT"}:
                            mouse_choice = control_id
                    elif kind == "intent":
                        direction = str(event.get("direction", "")).upper()
                        if direction in {"FORWARD", "LEFT", "RIGHT", "BACKWARD", "NONE"}:
                            browser_intent = direction
                    elif kind == "chat_question":
                        text = str(event.get("text", "")).strip()
                        if text:
                            browser_questions.append(text[:200])

            if keys.quit_requested or browser_quit:
                break
            if (mouse_choice in {"LEFT", "RIGHT"} and latest_authority is not None
                    and latest_authority.get("interaction_state") == "AWAITING_SECTOR_CHOICE"
                    and mouse_choice in latest_authority.get("selection_options", [])):
                previous_selected_sector = str(mouse_choice)
            mapped_intent: Optional[str] = None
            if keys.last_press_ms != last_key_event_ms:
                last_key_event_ms = keys.last_press_ms
                mapped_intent = {
                    "forward": "FORWARD",
                    "left": "LEFT",
                    "right": "RIGHT",
                    "backward": "BACKWARD",
                    "idle": "NONE",
                }[keys.last_edge_dir]
            elif browser_intent is not None:
                mapped_intent = browser_intent

            if mapped_intent is not None:
                # Every intent change resets the confirmed guidance state, so the next inference
                # tick is treated as a fresh T1 intent expression rather than compared against
                # the signature confirmed under the previous intent.
                active_intent = mapped_intent if mapped_intent != "NONE" else "NONE"
                current_ticket_id = None if mapped_intent == "NONE" else allocate("ticket", "ticket")
                confirmed_signature = None
                confirmed_authority = None
                more_detail_signature = None
                if mapped_intent == "NONE":
                    publish_idle_release()

            more_detail_requested = bool(keys.more_detail_edge or mouse_more_detail)
            if more_detail_requested:
                if reassessment_inflight or pending_reassessment:
                    ui_notice = "More detail is unavailable while reassessment is active."
                    ui_notice_until = time.monotonic() + 2.0
                    continue
                # Built from the live pipeline state (latest_authority, latest_sector_facts,
                # latest_observation_id), which updates on every inference tick regardless of
                # whether that tick triggered generation, rather than from latest_fact_packet,
                # which under the intent-triggered policy is only touched by an actual VLM
                # request and can be several scene changes behind the frame currently on screen.
                pipeline_stale = (
                    latest_observation_id is None or latest_authority is None
                    or latest_sector_facts is None
                    or latest_observation_id not in observation_images
                    or time.monotonic() - latest_observation_captured_monotonic > args.more_detail_freshness_s
                )
                if pipeline_stale:
                    ui_notice = "Current information is too old. Select Reassess."
                    ui_notice_until = time.monotonic() + 3.0
                elif more_detail_signature is not None and more_detail_signature == latest_material_signature:
                    # The scene has not moved enough since the last time it was elaborated:
                    # answer from what is already on screen rather than asking the model again.
                    with state_lock:
                        still_pending = generation_inflight and generation_request_id == "MORE_DETAIL"
                    ui_notice = (
                        "More detail is already being generated for the current scene."
                        if still_pending else
                        "Already showing the detailed description for the current scene."
                    )
                    ui_notice_until = time.monotonic() + 2.0
                else:
                    packet = create_fact_packet(
                        "MORE_DETAIL", "MORE_DETAIL", "USER_REQUESTED", latest_observation_id,
                        time.monotonic() * 1000.0, latest_objects, latest_sector_facts, latest_authority,
                        input_method="ONSCREEN_CONTROL" if mouse_more_detail else "KEYBOARD_SHORTCUT",
                        control_id="MORE_DETAIL",
                    )
                    enqueue_request(packet, observation_images[latest_observation_id].copy())
                    more_detail_signature = latest_material_signature
                    ui_notice = "More detail requested."
                    ui_notice_until = time.monotonic() + 2.0

            reassess_requested = bool(keys.reassess_edge or mouse_reassess)
            if reassess_requested:
                if reassessment_inflight or pending_reassessment or time.monotonic() < reassessment_cooldown_until:
                    ui_notice = "Reassessment is already active or cooling down."
                    ui_notice_until = time.monotonic() + 2.0
                else:
                    pending_reassessment = True
                    pending_reassessment_input_method = (
                        "ONSCREEN_CONTROL" if mouse_reassess else "KEYBOARD_SHORTCUT"
                    )
                    reassessment_inflight = True
                    ui_notice = "Reassessment requested."
                    ui_notice_until = time.monotonic() + 2.0
                    if reorientation_required:
                        stabilising = True
                        stable_count = 0
                        stable_statuses = None
                        stable_depths = None

            # Question admission. The measurement pre-check and the keyword filter run here, on the
            # main loop, because both are deterministic and both can settle a question without a
            # model call. Only what remains is handed to the worker.
            for question in browser_questions:
                question = questions.normalise_question(question)
                if not question:
                    continue
                if not args.answer_questions:
                    web_ui.publish_chat_turn(question, questions.OUT_OF_SCOPE_TEXT, route=None)
                    # Recorded like any other outcome. A question the channel declined without a
                    # model call is a result, and a log that omits it does not account for every
                    # question the run was asked.
                    record("question_record", questions.build_route_record(
                        question, None, "CHANNEL_DISABLED", False
                    ))
                    continue

                # Checked before the classifier, so an unmeasurable scene costs zero model calls
                # rather than two. The condition is the one the More detail path already uses.
                observation_age_ms = (
                    None if latest_observation_id is None
                    else (time.monotonic() - latest_observation_captured_monotonic) * 1000.0
                )
                have_frame = (
                    latest_authority is not None and latest_sector_facts is not None
                    and latest_observation_id in observation_images
                )
                # The measurement pre-check does not apply in the diagnostic mode: that path reads
                # the frame rather than the measurements, so refusing it for want of a valid
                # clearance would withhold the very comparison the mode exists to show.
                answerable = have_frame and (
                    args.unconstrained or questions.measurement_is_answerable(
                        latest_sector_facts, observation_age_ms,
                        args.more_detail_freshness_s * 1000.0,
                    )
                )
                if not answerable:
                    web_ui.publish_chat_turn(question, questions.NO_MEASUREMENT_TEXT, route=None)
                    record("question_record", questions.build_route_record(
                        question, None, "MEASUREMENT_PRECHECK", False
                    ))
                    continue

                keyword_route = questions.classify_keywords(question)
                resolved_by = "KEYWORD_FILTER" if keyword_route else "ADMISSION_CLASSIFIER"
                packet = create_fact_packet(
                    "MORE_DETAIL", "MORE_DETAIL", "USER_REQUESTED", latest_observation_id,
                    time.monotonic() * 1000.0, latest_objects, latest_sector_facts,
                    latest_authority,
                    # hdsg.fact_packet.v2 carries TYPED_QUESTION, so a question is recorded as
                    # what it is rather than as the button whose profile it borrows.
                    #
                    # `control_id` names the on-screen control that was activated, and typing a
                    # question activates none. It said "MORE_DETAIL" until 23 August 2026, so every
                    # question event in the archive records a button nobody pressed. It was also
                    # redundant: `request_id` already carries MORE_DETAIL, for the frozen-enumeration
                    # reason given in the request catalogue.
                    input_method="TYPED_QUESTION", control_id=None,
                )
                record("full_fact_packet", packet)
                try:
                    question_q.put_nowait({
                        "question": question,
                        "route": keyword_route,
                        "resolved_by": resolved_by,
                        "fact_packet": packet,
                        "image": observation_images[latest_observation_id].copy(),
                    })
                except Full:
                    web_ui.publish_chat_turn(
                        question,
                        "I am still working through the previous questions. Ask again in a moment.",
                        route=None,
                    )
                    record("question_record", questions.build_route_record(
                        question, None, "QUEUE_FULL", False
                    ))

            try:
                inference = inf_out.get_nowait()
            except Empty:
                inference = None

            if inference is not None:
                display_color = inference.color.copy()
                observation_id = allocate("observation", "obs")
                latest_observation_id = observation_id
                latest_observation_captured_monotonic = time.monotonic()
                observation_images[observation_id] = inference.color.copy()
                if recorder is not None:
                    # Queued to a writer thread, so the sensing loop never waits on disk. A frame
                    # that could not be queued is counted, and the run reports itself incomplete
                    # at shutdown rather than yielding a recording with silent gaps.
                    recorder.record(
                        observation_id, inference.color.copy(), inference.depth_m, inference.ts_ms
                    )
                while len(observation_images) > 20:
                    observation_images.pop(next(iter(observation_images)))
                tracked = motion_tracker.update(inference.color, list(inference.objects), inference.ts_ms)
                latest_objects = hdsg.normalise_objects(tracked)
                lane_state = compute_lane_state(
                    inference.depth_m,
                    mirror_view=args.mirror_view,
                    clear_t=args.clear_threshold_m,
                    blocked_t=blocked_threshold_m,
                    left_max=sector_left_max,
                    right_min=sector_right_min,
                    top_fraction=sector_band_top,
                    bottom_fraction=sector_band_bottom,
                )
                latest_sector_facts = hdsg.sectors_from_lane_state(lane_state)
                # Recorded on every observation, whether or not it crosses the caution threshold.
                # The threshold is provisional and set from seven frames; only the distribution the
                # archive accumulates can settle it.
                latest_depth_valid_fraction = sw.valid_depth_fraction(
                    inference.depth_m, sector_band_top, sector_band_bottom)
                baseline_facts = {
                    "objects": inference.objects,
                    "free_space": {"corridor_min_width_m": None},
                    "hazards": inference.hazards,
                    "uncertainty": {"valid_depth_fraction": latest_depth_valid_fraction},
                    "explain": {},
                }
                object_result = sw.compute_baseline_risk(baseline_facts, cfg)
                latest_authority = hdsg.determine_authority(
                    active_intent,
                    str(object_result["risk"]).upper(),
                    latest_sector_facts,
                    objects=latest_objects,
                    previous_selected_sector=previous_selected_sector,
                    sector_choice_tolerance_m=args.sector_choice_tolerance_m,
                    object_stop_below_m=blocked_threshold_m,
                    object_caution_below_m=object_caution_below_m,
                    hazard_stop_at_or_below_m=hazard_stop_at_or_below_m,
                )
                if latest_authority["selected_sector"] in hdsg.SECTORS:
                    previous_selected_sector = latest_authority["selected_sector"]
                reorientation_required = latest_authority["interaction_state"] == "REORIENTATION_REQUIRED"

                if stabilising:
                    statuses = tuple(latest_sector_facts[name]["status"] for name in ("left", "centre", "right"))
                    valid = all(latest_sector_facts[name]["valid"] for name in ("left", "centre", "right"))
                    depths = tuple(
                        float(latest_sector_facts[name]["clearance_m"])
                        for name in ("left", "centre", "right")
                    ) if valid else None
                    within_variation = bool(
                        valid and stable_depths is not None and depths is not None
                        and max(
                            abs(value - reference)
                            for value, reference in zip(depths, stable_depths)
                        ) <= args.post_reorientation_max_variation_m
                    )
                    if valid and statuses == stable_statuses and within_variation:
                        stable_count += 1
                    elif valid:
                        stable_statuses = statuses
                        stable_depths = depths
                        stable_count = 1
                    else:
                        stable_statuses = None
                        stable_depths = None
                        stable_count = 0
                    if stable_count < max(1, args.post_reorientation_stable_observations):
                        latest_authority = dict(latest_authority)
                        latest_authority.update({
                            "motion_decision": "STOP",
                            "selected_sector": "NONE",
                            "interaction_state": "POST_REORIENTATION_STABILISING",
                            "selection_status": "UNAVAILABLE",
                            "selection_options": [],
                        })
                    else:
                        stabilising = False
                        reorientation_required = False
                        active_intent = "FORWARD"
                        latest_authority = hdsg.determine_authority(
                            active_intent,
                            str(object_result["risk"]).upper(),
                            latest_sector_facts,
                            objects=latest_objects,
                            previous_selected_sector=None,
                            sector_choice_tolerance_m=args.sector_choice_tolerance_m,
                            object_stop_below_m=blocked_threshold_m,
                            object_caution_below_m=object_caution_below_m,
                            hazard_stop_at_or_below_m=hazard_stop_at_or_below_m,
                        )

                measurement_state = "VALID" if all(item["valid"] for item in latest_sector_facts.values()) else "PARTIAL"
                signature_authority = dict(latest_authority)
                signature_authority["moving_object_fact_ids"] = [
                    item["fact_id"] for item in latest_objects
                    if item.get("motion_state") == "MOVING"
                ]
                signature_authority["_measurement_state"] = measurement_state
                material_signature = hdsg.guidance_signature(
                    signature_authority, measurement_state, latest_objects
                )
                latest_material_signature = material_signature

                handled_reassessment = False
                if pending_reassessment:
                    packet = create_fact_packet(
                        "REASSESSMENT", "REASSESS", "USER_REQUESTED", observation_id,
                        inference.ts_ms, latest_objects, latest_sector_facts, latest_authority,
                        input_method=pending_reassessment_input_method, control_id="REASSESS",
                    )
                    enqueue_request(packet, inference.color.copy())
                    pending_reassessment = False
                    handled_reassessment = True

                if active_intent != "NONE":
                    # HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md sections 5 and 6. Generation
                    # is triggered only by intent expression (T1, confirmed_signature is None
                    # the first time under the current intent) or by a restrictive escalation
                    # (T2, a move to a higher rank in RESTRICTION_ORDER). A de-escalation or a
                    # lateral change at an unchanged rank still updates the confirmed action and
                    # reason immediately, from the deterministic templates, but does not call
                    # the model. The persistence period selects how long a new rank must hold
                    # before it is confirmed, asymmetric in the same way as before: short for a
                    # move to a more restrictive rank, longer for recovery.
                    now = time.monotonic()
                    current_rank = hdsg.RESTRICTION_ORDER.get(latest_authority["motion_decision"], 3)
                    if confirmed_signature is None:
                        ready = True
                        is_escalation = True
                    else:
                        if material_signature != pending_signature:
                            pending_signature = material_signature
                            pending_since = now
                        previous_rank = hdsg.RESTRICTION_ORDER.get(
                            (confirmed_authority or {}).get("motion_decision"), 0
                        )
                        persistence = 0.15 if current_rank > previous_rank else 0.50
                        ready = material_signature != confirmed_signature and now - pending_since >= persistence
                        is_escalation = ready and current_rank > previous_rank
                    if (ready and not pending_reassessment and not handled_reassessment
                            and not reassessment_inflight):
                        event_trigger = trigger_type(confirmed_authority, signature_authority)
                        packet = create_fact_packet(
                            "AUTOMATIC", "AUTO_GUIDANCE", event_trigger, observation_id,
                            inference.ts_ms, latest_objects, latest_sector_facts, latest_authority,
                        )
                        confirmed_signature = packet["interaction"]["current_guidance_signature"]
                        confirmed_authority = json.loads(json.dumps(signature_authority))
                        pending_signature = material_signature
                        pending_since = now
                        if is_escalation:
                            enqueue_request(packet, inference.color.copy())
                        else:
                            record("full_fact_packet", packet)
                            release = hdsg.build_release(
                                packet, {}, release_id=allocate("release", "release"),
                            )
                            publish_release(release)

            if web_ui is not None and display_color is not None:
                # The browser receives the unannotated frame on one channel and the structured
                # state on another, and draws the overlays itself. Nothing is composited here.
                if web_ui.should_publish_frame():
                    encoded, buffer = cv2.imencode(
                        ".jpg", display_color, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    if encoded:
                        web_ui.publish_frame(buffer.tobytes())

                with state_lock:
                    release = latest_release
                    display_generation_inflight = generation_inflight
                    display_generation_request_id = generation_request_id
                    display_queued_request_id = queued_request_id
                if reassessment_inflight:
                    worker_state = "REASSESSING"
                elif display_queued_request_id == "MORE_DETAIL":
                    worker_state = "MORE DETAIL QUEUED"
                elif display_generation_inflight:
                    worker_state = {
                        "MORE_DETAIL": "GENERATING MORE DETAIL",
                        "REASSESS": "REASSESSING",
                        "AUTO_GUIDANCE": "GENERATING GUIDANCE",
                    }.get(display_generation_request_id, "GENERATING")
                else:
                    worker_state = "MONITORING"
                height, width = display_color.shape[:2]
                web_ui.publish_state({
                    "frame_width": int(width),
                    "frame_height": int(height),
                    "intent": active_intent,
                    "worker_state": worker_state,
                    # Drives the banner. A screenshot of the diagnostic mode has to be
                    # distinguishable from one of the release path.
                    "unconstrained": bool(args.unconstrained),
                    "caption_text": (release or {}).get("content", {}).get("caption_text") or "",
                    "release_mode": (release or {}).get("verification", {}).get("release_mode"),
                    "clear_sectors": list((latest_authority or {}).get("clear_sectors", [])),
                    "selection_options": list((latest_authority or {}).get("selection_options", [])),
                    "sectors": {
                        name: {
                            "status": item["status"],
                            "clearance_m": item["clearance_m"],
                        }
                        for name, item in (latest_sector_facts or {}).items()
                    },
                    "objects": [
                        {
                            "label": item.get("canonical_label") or item.get("raw_label") or "object",
                            "distance_m": item.get("distance_m"),
                            "bbox_xyxy": item.get("bbox_xyxy"),
                            "bearing": item.get("bearing"),
                            "motion_state": item.get("motion_state"),
                            # Forwarded so the web interface can draw what the OpenCV window draws.
                            # Without the bucket a test cannot tell a hazard from a chair on screen.
                            "ontology_class": item.get("ontology_class"),
                            "is_hazard": bool(item.get("is_hazard")),
                            "raw_label": item.get("raw_label"),
                        }
                        for item in latest_objects
                        if _is_drawn(item, args)
                    ],
                    "controls_enabled": {
                        "more_detail": not reassessment_inflight and not pending_reassessment,
                        "reassess": not reassessment_inflight and not pending_reassessment,
                    },
                    "notice": ui_notice if ui_notice and time.monotonic() < ui_notice_until else None,
                })

            if use_opencv_ui and display_color is not None:
                vis = display_color.copy()
                if latest_sector_facts is not None and args.debug_lanes:
                    height, width = vis.shape[:2]
                    y1, y2 = int(0.55 * height), int(0.95 * height)
                    colours = {"CLEAR": (0, 180, 0), "CONSTRAINED": (0, 190, 255), "BLOCKED": (0, 0, 200), "UNKNOWN": (120, 120, 120)}
                    for index, name in enumerate(("left", "centre", "right")):
                        x1, x2 = int(index * width / 3), int((index + 1) * width / 3)
                        sector = latest_sector_facts[name]
                        colour = colours[sector["status"]]
                        overlay = vis.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
                        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
                        value = sector["clearance_m"]
                        value_text = "?" if value is None else f"{value:.2f}m"
                        cv2.putText(vis, f"{name.upper()} {value_text} {sector['status']}", (x1 + 4, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2, cv2.LINE_AA)

                for item in latest_objects:
                    if not _is_drawn(item, args):
                        continue
                    x1, y1, x2, y2 = map(int, item["bbox_xyxy"])
                    moving = item.get("motion_state") == "MOVING"
                    hazard = bool(item.get("is_hazard"))
                    # A hazard stops the walker at 2.00 m where everything else stops it at 0.70 m,
                    # so which detections are in that bucket is the thing being watched during a
                    # test. Magenta for a hazard, red for a confirmed moving object, yellow
                    # otherwise. The border is thicker for a hazard so it survives a screenshot.
                    colour = (255, 0, 255) if hazard else ((0, 0, 255) if moving else (255, 255, 0))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3 if hazard else 2)
                    label = item.get("canonical_label") or item.get("raw_label") or "object"
                    distance = item.get("distance_m")
                    distance_text = "?" if distance is None else f"{distance:.2f}m"
                    suffix = " MOVING" if moving else f" {item.get('motion_state', 'UNCONFIRMED')}"
                    # The bucket is shown for every object, because a detection mapping to
                    # unknown_obstacle when it should have mapped to a named class is invisible
                    # otherwise and is the failure the Open Images swap is most likely to produce.
                    bucket = item.get("ontology_class") or "unmapped"
                    text = f"{label} [{bucket}] {distance_text}{suffix}"
                    cv2.putText(vis, text, (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)

                with state_lock:
                    release = latest_release
                    display_generation_inflight = generation_inflight
                    display_generation_request_id = generation_request_id
                    display_queued_request_id = queued_request_id
                caption = release["content"]["caption_text"] if release else None
                if caption:
                    lines = _wrap_text(caption, width=74)
                    height, width = vis.shape[:2]
                    line_height = 20
                    block_height = 16 + line_height * len(lines)
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (0, height - block_height), (width, height), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)
                    y = height - block_height + 18
                    for line in lines:
                        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        y += line_height

                height, width = vis.shape[:2]
                cv2.rectangle(vis, (0, 0), (width, 28), (30, 30, 30), -1)
                if reassessment_inflight:
                    worker_state = "REASSESSING"
                elif display_queued_request_id == "MORE_DETAIL":
                    worker_state = "MORE DETAIL QUEUED"
                elif display_generation_inflight:
                    worker_state = {
                        "MORE_DETAIL": "GENERATING MORE DETAIL",
                        "REASSESS": "REASSESSING",
                        "AUTO_GUIDANCE": "GENERATING GUIDANCE",
                    }.get(display_generation_request_id, "GENERATING")
                else:
                    worker_state = "MONITORING"
                cv2.putText(vis, f"HDSG | intent:{active_intent} | {worker_state} | M: more detail | R: reassess",
                            (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
                if latest_authority is not None:
                    clear_text = " + ".join(latest_authority["clear_sectors"]) or "NONE"
                    badge = f"CLEAR SECTORS: {clear_text}"
                    colour = (0, 170, 0) if clear_text != "NONE" else (0, 0, 220)
                    (text_width, text_height), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    x1, y1 = width - text_width - 24, 36
                    cv2.rectangle(vis, (x1, y1), (width - 8, y1 + text_height + 14), colour, -1)
                    cv2.putText(vis, badge, (x1 + 8, y1 + text_height + 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 2, cv2.LINE_AA)
                if ui_notice and time.monotonic() < ui_notice_until:
                    cv2.putText(vis, ui_notice, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 215, 255), 2, cv2.LINE_AA)
                if args.mirror_view and not args.no_mirror_tag:
                    cv2.putText(vis, "MIRROR VIEW", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 215, 255), 1, cv2.LINE_AA)
                next_regions: dict[str, tuple[int, int, int, int]] = {}

                def draw_control(control_id: str, label: str, bounds: tuple[int, int, int, int], enabled: bool = True):
                    x1, y1, x2, y2 = bounds
                    colour = (55, 105, 55) if enabled else (75, 75, 75)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), colour, -1)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (220, 220, 220), 1)
                    cv2.putText(vis, label, (x1 + 8, y1 + 19), cv2.FONT_HERSHEY_SIMPLEX,
                                0.46, (255, 255, 255), 1, cv2.LINE_AA)
                    if enabled:
                        next_regions[control_id] = bounds

                if (latest_authority is not None
                        and latest_authority.get("interaction_state") == "AWAITING_SECTOR_CHOICE"):
                    options = set(latest_authority.get("selection_options", []))
                    choice_y1, choice_y2 = 68, 98
                    draw_control("LEFT", "Select left", (10, choice_y1, 122, choice_y2), "LEFT" in options)
                    draw_control("RIGHT", "Select right", (132, choice_y1, 254, choice_y2), "RIGHT" in options)
                with mouse_lock:
                    control_regions.clear()
                    control_regions.update(next_regions)
                cv2.imshow("HDSG smart walker", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        keyboard.stop()
        pipe.stop()
        if web_ui is not None:
            web_ui.stop()
        if recorder is not None:
            stats = recorder.stop()
            record("recording_summary", stats)
            state = "complete" if stats["complete"] else "INCOMPLETE"
            print(
                f"[hdsg] RGB-D recording {state}: {stats['written']} written, "
                f"{stats['dropped']} dropped, {stats['failed']} failed"
            )
            if not stats["complete"]:
                print("[hdsg] warning: this recording has gaps and is not a faithful replay source.")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
