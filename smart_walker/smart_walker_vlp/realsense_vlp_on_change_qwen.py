import argparse
import base64
import time
import re
import sys
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
except Exception:
    # Fallback for running directly from scripts folder
    import realsense_shared_control as sw  # type: ignore


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


def _build_payload_from_image_array(img_bgr: np.ndarray, fmt: str, quality: int,
                                    args) -> tuple[dict, str]:
    """Encode image with requested format/quality and build chat payload.
    Returns (payload, desc) where desc is a short string for logging.
    """
    mime, b64 = _encode_image(img_bgr, fmt=fmt, quality=int(quality))
    payload = build_mm_chat_payload(
        model=args.model,
        system=args.system,
        text=args._user_txt_for_payload,  # type: ignore[attr-defined]
        b64_data=b64,
        mime=mime,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
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


def _call_vlm_with_fallbacks(endpoint: str, base_img: np.ndarray, args) -> str:
    """Try multiple encodings/sizes when server says 'failed to process image'."""
    attempts: list[tuple[str, int, int]] = []  # (fmt, quality, size)
    # Longest side; Qwen3-VL handles non-square input natively
    size0 = int(getattr(args, 'image_size', 448) or 448)
    # Primary attempt: user choice
    attempts.append((str(getattr(args, 'encode', 'jpeg') or 'jpeg'), int(
        getattr(args, 'jpeg_quality', 70) or 70), size0))
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
            payload, desc = _build_payload_from_image_array(img, fmt, q, args)
            content = call_vlm(endpoint, payload, timeout=60)
            if desc != '':
                print(
                    f"[vlm_on_change_qwen] VLM accepted image encoding: {desc}")
            return content
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


def _parse_advice(content: str) -> tuple[Optional[str], Optional[str]]:
    """Parse Advice line from VLM content. Returns (ADVICE, suggested_dir).
    ADVICE is one of SAFE|CAUTION|STOP (upper) if found. suggested_dir is 'left'|'right' when present in advice text.
    """
    try:
        # Find the Advice line
        for ln in content.splitlines():
            # Accept either a colon or a semicolon after the level
            m = re.match(
                r"\s*Advice:\s*(SAFE|CAUTION|STOP)\s*[:;]\s*(.*)", ln, flags=re.IGNORECASE)
            if not m:
                m = re.match(
                    r"\s*(SAFE|CAUTION|STOP)\s*[:;]\s*(.*)", ln, flags=re.IGNORECASE)
            if m:
                adv = m.group(1).upper()
                text = (m.group(2) or '').lower()
                sug = None
                if 'suggest:' in text:
                    ms = re.search(r"suggest\s*[:]\s*([a-z\-]+)", text)
                    if ms:
                        token = ms.group(1).strip()
                        if token in ('left', 'right', 'back', 'continue', 'stay-left', 'stay-right', 'mid-left', 'mid-right', 'left-right'):
                            sug = token
                else:
                    if 'left' in text and 'right' not in text:
                        sug = 'left'
                    elif 'right' in text and 'left' not in text:
                        sug = 'right'
                return adv, sug
    except Exception:
        pass
    return None, None


def _strip_suggest(text: str) -> str:
    """Remove any trailing 'Suggest: <dir>' clause from a caption."""
    try:
        return re.sub(r"\s*Suggest:\s*(left|right|back)\.?\s*$", "", text, flags=re.IGNORECASE)
    except Exception:
        return text


def _humanize_lane_token(token: str) -> str:
    try:
        t = (token or '').lower()
        mapping = {
            'continue': 'mid',
            'left': 'left',
            'right': 'right',
            'stay-left': 'left',
            'stay-right': 'right',
            'mid-left': 'mid-left',
            'mid-right': 'mid-right',
            'left-right': 'left-right',
            'back': 'back',
        }
        return mapping.get(t, t)
    except Exception:
        return token


def _caption_with_human_lane(content: str) -> str:
    """Replace the Suggest token in the caption with a human-readable lane name matching the badge."""
    try:
        if not content:
            return content
        # Replace only the token part after 'Suggest:' while keeping punctuation

        def _repl(m):
            tok = m.group(1)
            human = _humanize_lane_token(tok)
            return f"Suggest: {human}"
        return re.sub(r"(?i)Suggest:\s*([a-z\-]+)", _repl, content)
    except Exception:
        return content


def _humanize_lane_mentions(content: str) -> str:
    """Rewrite technical lane references like 'Lane L1/L2/L3' or 'L1' to human terms.
    - L1 -> left, L2 -> mid, L3 -> right
    - 'Lane C'/'Lane M' -> middle
    - 'Lane L'/'Lane R' -> left/right
    Conservative patterns to avoid touching numbers in distances.
    """
    try:
        s = content
        # Common forms: 'Lane L1', 'lane l2', 'lane l3'
        s = re.sub(r"(?i)\blane\s*[-_ ]*l?1\b", "left lane", s)
        s = re.sub(r"(?i)\blane\s*[-_ ]*l?2\b", "middle lane", s)
        s = re.sub(r"(?i)\blane\s*[-_ ]*l?3\b", "right lane", s)
        # 'Lane C' or 'Lane M'
        s = re.sub(r"(?i)\blane\s*[cm]\b", "middle lane", s)
        # 'Lane L' / 'Lane R'
        s = re.sub(r"(?i)\blane\s*l\b", "left lane", s)
        s = re.sub(r"(?i)\blane\s*r\b", "right lane", s)
        # Standalone L1/L2/L3 tokens
        s = re.sub(r"(?i)\bL1\b", "left", s)
        s = re.sub(r"(?i)\bL2\b", "mid", s)
        s = re.sub(r"(?i)\bL3\b", "right", s)
        return s
    except Exception:
        return content


def _intent_phrase(intent: Optional[str]) -> str:
    try:
        i = (intent or '').lower()
        if i in ('forward', 'fwd', 'up'):
            return 'forward'
        if i in ('back', 'backward', 'down'):
            return 'back'
        if i == 'left':
            return 'left'
        if i == 'right':
            return 'right'
        return 'ahead'
    except Exception:
        return 'ahead'


def _finalize_caption(content: str, adv: Optional[str], intent: Optional[str]) -> str:
    """Make the caption one clean human sentence.
    - For SAFE: drop technical 'near:' details; concise path-clear phrasing; keep Suggest token.
    - For CAUTION/STOP: keep obstacle distances (near: ...); append 'for <intent>' after Suggest.
    - Fix duplicate punctuation and ensure trailing period.
    """
    try:
        s = content.strip()
        # Normalize GO -> SAFE
        if re.search(r"(?i)\bAdvice:\s*GO\b", s):
            s = re.sub(r"(?i)\bAdvice:\s*GO\b", "Advice: SAFE", s)
            adv = 'SAFE'
        # Extract suggest token (already humanized elsewhere)
        m = re.search(r"(?i)Suggest:\s*([^.;]+)", s)
        suggest_txt = m.group(1).strip() if m else None
        # Intent phrase
        ip = _intent_phrase(intent)
        # Remove any duplicated punctuation like ';;' or ':;'
        s = re.sub(r";{2,}", ";", s)
        s = re.sub(r":;", ";", s)
        # For SAFE, remove near: ... section and restate succinctly
        if (adv or '').upper() == 'SAFE':
            s = re.sub(r"(?i)\s*;?\s*near\s*:[^;\.]*", "", s)
            # Replace prefix with human phrasing
            s = re.sub(r"(?i)^Advice:\s*SAFE\s*[:;]?", "SAFE:", s)
            if suggest_txt:
                # Keep Suggest token as machine-readable, but make sentence human
                s = re.sub(r"(?i)Suggest:\s*[^.;]+",
                           f"Suggest: {suggest_txt}", s)
                # Prepend concise clause before Suggest
                base = f"SAFE: path looks clear to move {ip}; "
                # Remove any leftover reason before Suggest:
                idx = s.lower().rfind('suggest:')
                s = base + s[idx:]
        else:
            # CAUTION or STOP (or unknown): ensure near: is present; attach 'for <intent>' after Suggest
            if 'near:' not in s.lower():
                # leave as-is; _ensure_scene_detail already tried to add one
                pass
            if suggest_txt:
                s = re.sub(r"(?i)Suggest:\s*([^.;]+)",
                           f"Suggest: {suggest_txt} for {ip}", s)
        # Ensure single sentence end with a period
        s = s.rstrip()
        if not s.endswith(('.', '!', '?')):
            s = s + '.'
        return s
    except Exception:
        return content


def _ensure_suggest(adv: Optional[str], sug: Optional[str], content: str, fallback: Optional[str]) -> str:
    """Ensure caption contains a 'Suggest: ...' clause; append fallback when missing."""
    try:
        if 'suggest:' in (content or '').lower():
            # Optionally normalize tokens for CAUTION/SAFE
            try:
                adv2, sug2 = _parse_advice(content)
                if adv2 == 'CAUTION' and sug2 in ('left', 'right'):
                    return re.sub(r"(?i)\bSuggest:\s*(left|right)\b",
                                  lambda m: f"Suggest: {'stay-left' if m.group(1).lower() == 'left' else 'stay-right'}",
                                  content)
                if adv2 == 'SAFE' and sug2 in ('left', 'right'):
                    return re.sub(r"(?i)\bSuggest:\s*(left|right)\b", "Suggest: continue", content)
                if adv2 == 'STOP' and sug2 in ('stay-left', 'stay-right'):
                    return re.sub(r"(?i)\bSuggest:\s*stay\-(left|right)\b",
                                  lambda m: f"Suggest: {m.group(1).lower()}", content)
                # New: avoid contradictory STOP + continue → use fallback or 'back'
                if adv2 == 'STOP' and sug2 in ('continue',):
                    tok = (fallback or 'back')
                    tok = tok.lower()
                    if tok in ('stay-left', 'stay-right'):
                        tok = 'left' if tok == 'stay-left' else 'right'
                    if tok not in ('left', 'right', 'back'):
                        tok = 'back'
                    return re.sub(r"(?i)\bSuggest:\s*continue\b", f"Suggest: {tok}", content)
            except Exception:
                pass
            return content
        add = sug or fallback
        if not add:
            return content
        token = add.lower()
        # Normalize by Advice for consistency
        if adv == 'CAUTION' and token in ('left', 'right'):
            token = 'stay-left' if token == 'left' else 'stay-right'
        if adv == 'SAFE' and token in ('left', 'right'):
            token = 'continue'
        if adv == 'STOP' and token in ('stay-left', 'stay-right'):
            token = 'left' if token == 'stay-left' else 'right'
        if token in ('left', 'right', 'back', 'continue', 'stay-left', 'stay-right', 'mid-left', 'mid-right', 'left-right'):
            return content.rstrip() + f" Suggest: {token}"
        return content
    except Exception:
        return content


def _ensure_scene_detail(content: str, detail: Optional[str]) -> str:
    """Guarantee a minimal scene description before 'Suggest:'.
    Rule: if the sentence does NOT contain an explicit distance (e.g., '1.2 m'/'1.2m'),
    we inject our compact detail (near: objs / lanes L/C/R / nearest x.xm).
    Direction words alone (left/right) are NOT considered sufficient.
    """
    try:
        if not detail:
            return content
        s = (content or "").lower()
        has_distance = bool(re.search(r"\b\d+(?:\.\d+)?\s*m\b", s))
        # Prefer to always include specific nearby object detail if provided
        if detail and detail.lower().startswith('near:'):
            if 'near:' not in s:
                idx = s.rfind('suggest:')
                if idx != -1:
                    pre = content[:idx].rstrip().rstrip('.')
                    post = content[idx:]
                    return f"{pre}; {detail} {post}"
                return content.rstrip().rstrip('.') + f"; {detail}"
            return content
        # Otherwise, if no explicit distance is present, add the generic detail
        if has_distance:
            return content
        # Insert detail just before 'Suggest:' to keep a single sentence
        idx = s.rfind('suggest:')
        if idx != -1:
            pre = content[:idx].rstrip().rstrip('.')
            post = content[idx:]
            return f"{pre}; {detail} {post}"
        return content.rstrip().rstrip('.') + f"; {detail}"
    except Exception:
        return content


def _is_system_echo(content: str) -> bool:
    s = (content or "").lower()
    return (
        "you are a safety-first vision assistant" in s
        or "output format (no extra text)" in s
        or s.startswith("you are a safety-first vision assistant")
    )


def _is_valid_caption(content: str) -> bool:
    try:
        if not content:
            return False
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines:
            return False
        if len(lines) == 1:
            s0 = lines[0].lower()
            if s0.startswith('advice:'):
                return True
            if s0.startswith('safe:') or s0.startswith('caution:') or s0.startswith('stop:'):
                return True
            return False
        if len(lines) >= 2:
            return (lines[0].lower().startswith('scene:') and lines[1].lower().startswith('advice:'))
        return False
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Event-driven RealSense -> Qwen3-VL (trigger on risk/intent change)")

    # VLM parameters (defaults adapted for Qwen3-VL 4B)
    ap.add_argument('--endpoint', type=str, default='http://127.0.0.1:8080',
                    help='OpenAI-compatible VLM server base URL (llama.cpp)')
    ap.add_argument('--model', type=str, default='qwen3-vl-4b-instruct',
                    help='VLM model name/id on server')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--max_tokens', type=int, default=120)
    ap.add_argument('--system', type=str, default=(
        "You are a safety-first vision assistant for an elderly user's smart walker. "
        "Use only the provided scene facts; do not invent.\n\n"
        "Output exactly ONE natural, human-sounding sentence that briefly describes the scene and clearly states the safety decision. "
        "Always include at least one short concrete detail drawn from the facts (e.g., a specific object with distance and side, or lane clearance). "
        "Begin with the decision token in uppercase, formatted as either 'Advice: <SAFE|CAUTION|STOP>:' or '<SAFE|CAUTION|STOP>:' followed by a very short reason. "
        "Always append a concise 'Suggest: ...' at the end. Allowed tokens: 'continue', 'stay-left', 'stay-right', 'left', 'right', 'mid-left', 'mid-right', 'left-right', 'back'. "
        "Lane model: only three depth bands (LEFT / MID / RIGHT). If two adjacent bands are clear, use a combined token (mid-left or mid-right). If left & right clear but mid blocked, use 'left-right'. If only MID clear, 'continue'. If none clear, 'back'. "
        "Guidance: for SAFE prefer 'continue' (or the best clear combo); for CAUTION use 'stay-left'/'stay-right' or a combo if two are clear; for STOP use 'left' or 'right' if exactly one side offers slightly more space, otherwise 'back'. "
        "If a memory_list is provided for this ticket, prefer consistent object names based on it; do not switch names unless clearly contradicted. "
        "Plain language only; no labels, no lists, no JSON, no line breaks, no extra text."),
        help="System: one-sentence scene assessment + decision; end with 'Suggest: <...>' using allowed 3-band tokens and combos.")
    ap.add_argument('--vlm_timeout_s', type=float, default=20.0,
                    help='Max seconds to wait for a single VLM response')
    ap.add_argument('--late_accept_s', type=float, default=1.5,
                    help='Grace window (seconds) to accept late VLM results after a ticket switch for overlay display')
    ap.add_argument('--strict_caption', action='store_true',
                    help="Only display captions that start with 'Advice:' or 'SAFE:/CAUTION:/STOP:'; otherwise show raw text")

    # Trigger policy
    ap.add_argument('--min_interval_s', type=float, default=0.9,
                    help='Minimum seconds between VLM calls (debounce)')

    # Camera / encoding
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--image_size', type=int, default=448,
                    help='Longest side before sending to VLM; aspect ratio is preserved')
    ap.add_argument('--jpeg_quality', type=int, default=70)
    ap.add_argument('--encode', choices=['jpeg', 'png'], default='jpeg',
                    help='Image encoding format for VLM payload (default: jpeg)')
    ap.add_argument('--process_hz', type=float, default=0.0,
                    help='Max processing rate for detector/risk. 0 = unlimited.')
    ap.add_argument('--warmup', action='store_true',
                    help='Warm up YOLO on CUDA and print CUDA device info')

    # Object detector (reuses shared pipeline)
    ap.add_argument('--det_model', default='yolov8n.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--track', action='store_true', help='Use tracker IDs')
    ap.add_argument('--half', action='store_true',
                    help='Use FP16 if available')

    # UI
    ap.add_argument('--show', action='store_true',
                    help='Show preview with overlay')
    ap.add_argument('--clear_threshold_m', type=float, default=1.8,
                    help='Minimum clear distance to consider a lane navigable (depth-driven lanes)')
    ap.add_argument('--debug_lanes', action='store_true',
                    help='Overlay depth lane ROIs and median values for debugging')
    ap.add_argument('--mirror_view', action='store_true',
                    help='Swap left/right semantics to match a mirrored camera view')
    ap.add_argument('--no_mirror_tag', action='store_true',
                    help='Suppress the on-screen "MIRROR VIEW" notice while mirror_view is enabled')
    args = ap.parse_args()

    # Load config & ontology
    cfg = sw.load_yaml(sw.PIPELINE_CFG)
    mapper = sw.OntologyMapper(sw.ONTOLOGY_CFG)

    # RealSense init
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
    print(f"[vlm_on_change_qwen] depth_scale = {depth_scale:.6f} m/unit")

    # Queues & threads (reusing shared pipeline workers)
    from queue import Queue, Empty
    cap_q: 'Queue[sw.FramePacket]' = Queue(maxsize=1)
    inf_in: 'Queue[sw.FramePacket]' = Queue(maxsize=1)
    inf_out: 'Queue[sw.InferPacket]' = Queue(maxsize=1)

    stop_evt = __import__('threading').Event()

    # Detector
    from ultralytics import YOLO
    model = YOLO(args.det_model)
    # Prefer GPU for detector if available
    use_cuda = bool(torch is not None and hasattr(
        torch, 'cuda') and torch.cuda.is_available())
    if use_cuda:
        try:
            model.to('cuda')
            if args.half:
                print("[vlm_on_change_qwen] YOLO on CUDA (FP16)")
            else:
                print("[vlm_on_change_qwen] YOLO on CUDA (FP32)")
        except Exception as e:
            print(
                "[vlm_on_change_qwen] Failed to move YOLO to CUDA, staying on CPU:", e)
    else:
        if args.half:
            print(
                "[vlm_on_change_qwen] --half requested but CUDA is not available; running on CPU FP32.")

    __import__('threading').Thread(target=sw.capture_thread, args=(
        pipe, align, depth_scale, cap_q, stop_evt, False, None, sw.IMU_MAX_DRAIN_PER_LOOP), daemon=True).start()
    half_flag = bool(args.half and (torch is not None) and hasattr(
        torch, 'cuda') and torch.cuda.is_available())
    __import__('threading').Thread(
        target=sw.inference_thread,
        args=(cfg, mapper, model, inf_in, inf_out, stop_evt,
              args.imgsz, args.conf, half_flag, None, args.track, 'bytetrack.yaml'),
        daemon=True
    ).start()

    # Keyboard intent (reuse W/A/S/D scheme)
    kbd = sw.KeyListener(poll_hz=120).start()

    # --- Async VLM worker to avoid blocking the main loop ---
    from queue import Queue as _Q
    vlm_q: '_Q[tuple[np.ndarray, str, int]]' = _Q(maxsize=1)
    vlm_inflight: bool = False

    dialog = {
        "pending": False,
        "proposed_dir": None,  # 'left' | 'right'
        "since_ms": 0,
        "last_reply": None,
    }

    def _vlm_worker():
        nonlocal last_caption, last_vlm_ts, vlm_inflight, active_ticket_id, current_ticket
        while not stop_evt.is_set():
            try:
                item = vlm_q.get(timeout=0.05)
            except Empty:
                continue
            try:
                send_np, user_txt, t_id = item
                vlm_inflight = True
                mime, b64 = _encode_image(
                    send_np, fmt=args.encode, quality=int(args.jpeg_quality))
                setattr(args, '_user_txt_for_payload', user_txt)
                try:
                    content = _call_vlm_with_fallbacks(
                        args.endpoint, send_np, args)
                finally:
                    if hasattr(args, '_user_txt_for_payload'):
                        try:
                            delattr(args, '_user_txt_for_payload')
                        except Exception:
                            pass

                accept_for_ticket = (int(t_id) == int(active_ticket_id))
                accept_unbound = (int(t_id) == 0)
                try:
                    grace = float(getattr(args, 'late_accept_s', 1.5))
                except Exception:
                    grace = 1.5
                recently_switched = (
                    time.time() - last_ticket_switch_ts) <= grace

                if accept_for_ticket or accept_unbound or recently_switched:
                    if _is_system_echo(content):
                        print(
                            f"[warn] VLM returned system prompt text; ignoring overlay. Raw len={len(content)}")
                    elif not _is_valid_caption(content):
                        if getattr(args, 'strict_caption', False):
                            print(
                                f"[warn] VLM returned non-minimal caption; ignoring overlay. First 80 chars: {content[:80]!r}")
                        else:
                            print(
                                f"[info] Accepting non-minimal caption for overlay (strict_caption=off). First 80 chars: {content[:80]!r}")
                            last_caption = content
                    else:
                        adv_tmp, sug_tmp = _parse_advice(content)
                        # Keep model's suggestion if present; we'll compute a fallback if missing
                        try:
                            fallback = (current_ticket or {}).get(
                                'last_auto_suggest')
                        except Exception:
                            fallback = None
                        content = _ensure_suggest(
                            adv_tmp, sug_tmp, content, fallback)
                        content = _caption_with_human_lane(content)
                        # Ensure a minimal concrete detail is present; fall back to our scene snippet
                        try:
                            scene_fallback = (current_ticket or {}).get(
                                'last_scene_snippet')
                        except Exception:
                            scene_fallback = None
                        content = _ensure_scene_detail(content, scene_fallback)
                        adv_tmp2, _sug_tmp2 = _parse_advice(content)
                        intent_dir = (current_ticket or {}).get(
                            'intent_dir') if current_ticket else None
                        content = _finalize_caption(
                            content, adv_tmp2, intent_dir)
                        last_caption = _humanize_lane_mentions(content)
                    last_vlm_ts = time.time()

                    if accept_for_ticket:
                        try:
                            if current_ticket is not None:
                                current_ticket['last_llm_ts'] = last_vlm_ts
                        except Exception:
                            pass
                        adv, sug = _parse_advice(content)
                        try:
                            if current_ticket is not None:
                                current_ticket['last_advice'] = adv
                                if adv == 'SAFE':
                                    current_ticket['status'] = 'moving'
                                elif adv == 'CAUTION':
                                    current_ticket['status'] = 'moving'
                                elif adv == 'STOP':
                                    current_ticket['status'] = 'blocked'
                        except Exception:
                            pass
                        try:
                            # Only allow interactive direction proposal for STOP with left/right
                            if adv == 'STOP' and sug in ('left', 'right'):
                                dialog.update({
                                    'pending': True,
                                    'proposed_dir': sug,
                                    'since_ms': int(time.time()*1000),
                                    'last_reply': None,
                                })
                            else:
                                if dialog.get('pending'):
                                    dialog['pending'] = False
                                    dialog['proposed_dir'] = None
                                    dialog['last_reply'] = None
                        except Exception:
                            pass
                        print(
                            f"[{int(last_vlm_ts*1000)}] VLM (ticket {t_id}) accepted; overlay updated")
                        # Ensure human lane names after any late normalization
                        last_caption = _caption_with_human_lane(last_caption)
                else:
                    if not getattr(args, 'strict_caption', False):
                        if _is_system_echo(content):
                            print(
                                f"[warn] Stale VLM result echoed system prompt; overlay not updated. ticket={t_id} active={active_ticket_id}")
                        else:
                            print(
                                f"[info] Accepting late/stale VLM caption for overlay (strict_caption=off). ticket={t_id} active={active_ticket_id} First 80 chars: {content[:80]!r}")
                            adv_tmp3, _sug_tmp3 = _parse_advice(content)
                            intent_dir2 = (current_ticket or {}).get(
                                'intent_dir') if current_ticket else None
                            content2 = _finalize_caption(
                                content, adv_tmp3, intent_dir2)
                            last_caption = _humanize_lane_mentions(content2)
                            last_vlm_ts = time.time()
                    else:
                        print(
                            f"[info] Dropped stale VLM result for ticket {t_id}; active={active_ticket_id}")
                vlm_inflight = False
            except Exception as e:
                print("[vlm_on_change_qwen] VLM worker failed:", e)
                # Helpful hint for common server error when mmproj is missing
                try:
                    msg = str(e).lower()
                    if 'mmproj' in msg or 'image input is not supported' in msg:
                        print("[hint] The llama.cpp server reported missing image support. "
                              "Start the server with a Qwen3-VL mmproj via -MmprojPath, e.g.:\n"
                              "  & .\\scripts\\start_llama_server.ps1 -ModelPath \"...Qwen3VL-4B-Instruct-*.gguf\" -MmprojPath \"...mmproj-Qwen3VL-4B-Instruct-*.gguf\" ...")
                except Exception:
                    pass
                last_vlm_ts = time.time()
                vlm_inflight = False

    __import__('threading').Thread(target=_vlm_worker, daemon=True).start()

    prev_risk: Optional[str] = None
    prev_dir: Optional[str] = None
    last_vlm_ts = 0.0
    last_caption: Optional[str] = None
    latest_color: Optional[np.ndarray] = None
    last_decision: str = "GO"
    latest_depth: Optional[np.ndarray] = None
    last_nearest_m: Optional[float] = None
    next_ticket_id: int = 1
    active_ticket_id: int = 0
    current_ticket = None
    TICKET_BANNER_UNTIL = 0.0
    TICKET_BANNER_TEXT: Optional[str] = None
    CLOSE_BANNER_UNTIL = 0.0
    CLOSE_BANNER_TEXT: Optional[str] = None
    DECLINE_COOLDOWN_UNTIL = 0.0
    last_ticket_switch_ts = 0.0
    last_objects = []
    last_hazards = []

    print("[vlm_on_change_qwen] running; triggers on risk/intent change. Q to quit. W/A/S/D for intents.")
    try:
        last_proc_t = 0.0
        proc_period = (1.0 / float(args.process_hz)
                       ) if args.process_hz and args.process_hz > 0 else 0.0
        while True:
            if hasattr(cv2, 'pollKey'):
                cv2.pollKey()
            else:
                cv2.waitKey(1)

            try:
                pkt = cap_q.get_nowait()
                latest_color = pkt.color
                try:
                    latest_depth = pkt.depth_m
                except Exception:
                    latest_depth = None
                now = time.time()
                due_proc = (proc_period <= 0.0) or (
                    (now - last_proc_t) >= proc_period)
                if due_proc:
                    while not inf_in.empty():
                        try:
                            inf_in.get_nowait()
                        except Empty:
                            break
                    inf_in.put(pkt)
                    last_proc_t = now
            except Empty:
                pass

            out: Optional[sw.InferPacket] = None
            try:
                out = inf_out.get_nowait()
            except Empty:
                pass

            s = kbd.snapshot()
            if s.quit_requested:
                break
            now_ms = int(time.time() * 1000)
            active = (now_ms - s.last_press_ms) <= sw.ACTIVE_KEY_WINDOW_MS
            if dialog.get("pending"):
                if s.yes_edge:
                    dialog["last_reply"] = "yes"
                    dialog["pending"] = False
                elif s.no_edge:
                    dialog["last_reply"] = "no"
                    dialog["pending"] = False
            if dialog.get("last_reply") == "yes" and dialog.get("proposed_dir"):
                current_ticket = {
                    'id': next_ticket_id,
                    'intent_dir': dialog.get('proposed_dir'),
                    'status': 'open',
                    'last_llm_ts': 0.0,
                    'proposed_dir': None,
                    'last_user_reply': None,
                    'start_ts': time.time(),
                    'mem': {'counts': {}, 'opened_ts': time.time()},
                }
                active_ticket_id = next_ticket_id
                next_ticket_id += 1
                TICKET_BANNER_TEXT = f"Ticket #{active_ticket_id} opened: going {current_ticket['intent_dir']}"
                TICKET_BANNER_UNTIL = time.time() + 2.5
                print(
                    f"[ticket] opened #{active_ticket_id} going {current_ticket['intent_dir']}")
                dialog.update({'pending': False, 'proposed_dir': None,
                              'last_reply': None, 'since_ms': now_ms})
                last_vlm_ts = 0.0
                last_ticket_switch_ts = time.time()
            elif dialog.get("last_reply") == "no" and dialog.get("proposed_dir") and not dialog.get("pending"):
                dialog.update({'pending': False, 'proposed_dir': None,
                              'since_ms': now_ms, 'last_reply': None})
                DECLINE_COOLDOWN_UNTIL = time.time() + 4.0

            # Space-to-standby: close any active ticket and return to idle
            if s.last_edge_dir == 'idle' and active:
                if current_ticket and current_ticket.get('status') != 'closed':
                    current_ticket['status'] = 'closed'
                    current_ticket['result'] = 'standby'
                    current_ticket['end_ts'] = time.time()
                    CLOSE_BANNER_TEXT = f"Ticket #{active_ticket_id} closed: standby"
                    CLOSE_BANNER_UNTIL = time.time() + 2.5
                    print(
                        f"[ticket] closed #{active_ticket_id} result=standby")
                # Reset to idle (no active ticket)
                current_ticket = None
                active_ticket_id = 0
                dialog.update(
                    {'pending': False, 'proposed_dir': None, 'last_reply': None})
                last_vlm_ts = 0.0
                last_ticket_switch_ts = time.time()
                last_non_idle_dir = 'idle'
            elif s.last_edge_dir != 'idle':
                last_non_idle_dir = s.last_edge_dir
                if current_ticket is None or (current_ticket and current_ticket.get('intent_dir') != s.last_edge_dir):
                    if current_ticket and current_ticket.get('status') != 'closed':
                        current_ticket['status'] = 'closed'
                        current_ticket['result'] = 'switched'
                        current_ticket['end_ts'] = time.time()
                        CLOSE_BANNER_TEXT = f"Ticket #{active_ticket_id} closed: switched"
                        CLOSE_BANNER_UNTIL = time.time() + 2.5
                        print(
                            f"[ticket] closed #{active_ticket_id} result=switched")
                    current_ticket = {
                        'id': next_ticket_id,
                        'intent_dir': s.last_edge_dir,
                        'status': 'open',
                        'last_llm_ts': 0.0,
                        'proposed_dir': None,
                        'last_user_reply': None,
                        'start_ts': time.time(),
                        'moving_since': None,
                        'mem': {'counts': {}, 'opened_ts': time.time()},
                    }
                    active_ticket_id = next_ticket_id
                    next_ticket_id += 1
                    TICKET_BANNER_TEXT = f"Ticket #{active_ticket_id} opened: going {s.last_edge_dir}"
                    TICKET_BANNER_UNTIL = time.time() + 2.5
                    print(
                        f"[ticket] opened #{active_ticket_id} going {s.last_edge_dir}")
                    dialog.update(
                        {'pending': False, 'proposed_dir': None, 'last_reply': None, 'since_ms': now_ms})
                    last_vlm_ts = 0.0
                    last_ticket_switch_ts = time.time()
            else:
                last_non_idle_dir = prev_dir or 'idle'
            pressing_dir = getattr(s, 'level_dir', 'idle') or 'idle'
            intent_for_caption = s.last_edge_dir if active else 'idle'
            effective_dir = intent_for_caption if intent_for_caption != 'idle' else (
                last_non_idle_dir or 'idle')

            if out is not None:
                facts = {
                    "frame_id": 0,
                    "timestamp_ms": out.ts_ms,
                    "objects": out.objects,
                    "free_space": {"corridor_min_width_m": None, "nearest_obstacle_m": None},
                    "hazards": out.hazards,
                    "uncertainty": {"depth_std": None, "low_light": False},
                    "source_depth": "rgbd",
                    "explain": {"rules_fired": [], "min_distance_m": None, "class_counts": {}}
                }
                r = sw.compute_baseline_risk(facts, cfg)
                risk = r["risk"]
                facts_with_risk = dict(facts)
                facts_with_risk["risk"] = risk
                decision, _why = sw.arbiter_decision(
                    effective_dir, facts_with_risk, cfg)
                last_decision = decision
                try:
                    last_nearest_m = sw._safe_min_distance(out.objects)
                except Exception:
                    last_nearest_m = None
                try:
                    last_objects = list(out.objects)
                    last_hazards = list(out.hazards)
                except Exception:
                    pass

                # Update per-ticket memory of nearby objects (label/side/min_dist/last_seen/count)
                try:
                    if current_ticket and current_ticket.get('status') != 'closed':
                        mem = current_ticket.setdefault(
                            'mem', {'counts': {}, 'opened_ts': time.time()})
                        counts = mem.setdefault('counts', {})
                        now_ts = time.time()
                        for o in out.objects:
                            name = o.get("canonical_class") or o.get(
                                "display_label") or o.get("raw_label") or "obj"
                            dist = o.get("distance_m")
                            b = o.get("bearing")
                            side_raw = str(b[0] if isinstance(
                                b, (list, tuple)) else b).lower() if b is not None else ""
                            side = "left" if "left" in side_raw else (
                                "right" if "right" in side_raw else "center")
                            # Mirror semantics if requested
                            if getattr(args, 'mirror_view', False):
                                if side == 'left':
                                    side = 'right'
                                elif side == 'right':
                                    side = 'left'
                            if isinstance(dist, (int, float)) and dist <= 3.5:
                                key = (name, side)
                                ent = counts.get(key) or {
                                    "count": 0, "min_dist": float('inf'), "last_seen": 0.0}
                                ent["count"] = int(ent.get("count", 0)) + 1
                                try:
                                    ent["min_dist"] = min(
                                        float(ent.get("min_dist", float('inf'))), float(dist))
                                except Exception:
                                    ent["min_dist"] = float(dist)
                                ent["last_seen"] = now_ts
                                counts[key] = ent
                        # Optional: drop very stale entries (> 15s)
                        try:
                            ttl = 15.0
                            stale = [k for k, v in counts.items() if (
                                now_ts - float(v.get('last_seen', 0.0))) > ttl]
                            for k in stale:
                                counts.pop(k, None)
                        except Exception:
                            pass
                except Exception:
                    pass

                if str(risk).lower() != "stop" and dialog.get("pending"):
                    dialog["pending"] = False

                changed = (risk != prev_risk) or (effective_dir != prev_dir)
                continuing_push = (str(risk).lower(
                ) == 'stop' and pressing_dir != 'idle' and pressing_dir == effective_dir)
                due = (time.time() - last_vlm_ts) >= float(args.min_interval_s)
                last_llm_ts_val = float(
                    (current_ticket or {}).get('last_llm_ts') or 0.0)
                reconfirm_due = bool(current_ticket) and (current_ticket.get(
                    'status') != 'closed') and (time.time() - last_llm_ts_val >= 10.0)
                stop_case = (str(risk).lower() == 'stop' and (
                    continuing_push or dialog.get('pending')))
                base_send = ((changed or reconfirm_due)
                             and due) or (stop_case and due)
                in_hold = bool(current_ticket) and (last_llm_ts_val > 0.0) and (
                    (time.time() - last_llm_ts_val) < 10.0)
                ticket_active = bool(current_ticket) and (
                    current_ticket.get('status') != 'closed')
                should_send = ticket_active and (
                    base_send and (not in_hold or stop_case))

                if should_send and latest_color is not None:
                    send_np = resize_for_vlm(latest_color, args.image_size)
                    objs_n = len(out.objects)
                    haz_n = len(out.hazards)
                    nearest_txt = f"{last_nearest_m:.2f}m" if isinstance(
                        last_nearest_m, (int, float)) else "?"
                    obj_summ: list[tuple[float, str]] = []
                    try:
                        for o in out.objects:
                            name = o.get("canonical_class") or o.get(
                                "display_label") or o.get("raw_label") or "obj"
                            dist = o.get("distance_m")
                            b = o.get("bearing")
                            side_raw = str(b[0] if isinstance(
                                b, (list, tuple)) else b).lower() if b is not None else ""
                            side = "left" if "left" in side_raw else (
                                "right" if "right" in side_raw else "center")
                            if getattr(args, 'mirror_view', False):
                                if side == 'left':
                                    side = 'right'
                                elif side == 'right':
                                    side = 'left'
                            if isinstance(dist, (int, float)):
                                obj_summ.append(
                                    (float(dist), f"{name}:{float(dist):.2f}m {side}"))
                        obj_summ.sort(key=lambda x: x[0])
                    except Exception:
                        obj_summ = []
                    top_objs = ", ".join(
                        [s for _, s in obj_summ[:3]]) if obj_summ else ""
                    scene_snippet: Optional[str] = None
                    lines = [
                        f"intent: {effective_dir}; decision: {last_decision}; risk: {risk}",
                        f"objects: {objs_n}; hazards: {haz_n}; nearest: {nearest_txt}",
                        f"braking: {'yes' if str(risk).lower() == 'stop' else 'no'}; pushing: {'yes' if continuing_push else 'no'}; push_dir: {pressing_dir}",
                    ]
                    if top_objs:
                        lines.append(f"objects_list: {top_objs}")
                        scene_snippet = f"near: {top_objs}"
                    # Depth-driven lane clearance (3-band: L, C, R)
                    try:
                        if latest_depth is not None and isinstance(latest_depth, np.ndarray):
                            H, W = latest_depth.shape[:2]
                            y1 = int(0.55 * H)
                            y2 = int(0.95 * H)
                            xL1, xL2 = 0, int(W/3)
                            xC1, xC2 = int(W/3), int(2*W/3)
                            xR1, xR2 = int(2*W/3), W
                            d_L = sw.median_depth_in_box(
                                latest_depth, xL1, y1, xL2, y2)
                            d_C = sw.median_depth_in_box(
                                latest_depth, xC1, y1, xC2, y2)
                            d_R = sw.median_depth_in_box(
                                latest_depth, xR1, y1, xR2, y2)
                            if getattr(args, 'mirror_view', False):
                                d_L, d_R = d_R, d_L
                            CLEAR_T = float(
                                getattr(args, 'clear_threshold_m', 1.8))

                            def _fmt(v):
                                return (f"{v:.2f}m" if isinstance(v, (int, float)) and np.isfinite(v) else "?")
                            lines.append(
                                f"lanes3: L:{_fmt(d_L)}, C:{_fmt(d_C)}, R:{_fmt(d_R)}")
                            # status categories
                            try:
                                near_hi = float(sw.load_yaml(sw.PIPELINE_CFG)[
                                                'depth']['metric_bins_m']['very_close'][1])
                            except Exception:
                                near_hi = 0.7

                            def _status(d):
                                if not isinstance(d, (int, float)) or not np.isfinite(d):
                                    return 'unknown'
                                if d < near_hi:
                                    return 'blocked'
                                if d < CLEAR_T:
                                    return 'constrained'
                                return 'clear'
                            st_L, st_C, st_R = _status(
                                d_L), _status(d_C), _status(d_R)
                            lines.append(
                                f"lane_status3: L:{st_L}, C:{st_C}, R:{st_R}")
                            # auto suggest based on combinations

                            def _num(d):
                                try:
                                    return float(d) if isinstance(d, (int, float)) and np.isfinite(d) else -1.0
                                except Exception:
                                    return -1.0
                            is_L = _num(d_L) >= CLEAR_T
                            is_C = _num(d_C) >= CLEAR_T
                            is_R = _num(d_R) >= CLEAR_T
                            auto_suggest = None
                            clear_count = sum([is_L, is_C, is_R])
                            if clear_count == 0:
                                auto_suggest = 'back'
                            elif clear_count == 3:
                                auto_suggest = 'continue'
                            elif clear_count == 1:
                                if is_L:
                                    auto_suggest = 'left'
                                elif is_C:
                                    auto_suggest = 'continue'
                                else:
                                    auto_suggest = 'right'
                            else:  # two clear
                                if is_C and is_L and not is_R:
                                    auto_suggest = 'mid-left'
                                elif is_C and is_R and not is_L:
                                    auto_suggest = 'mid-right'
                                elif is_L and is_R and not is_C:
                                    auto_suggest = 'left-right'
                                else:
                                    # fallback
                                    auto_suggest = 'continue' if is_C else (
                                        'left' if is_L else 'right')
                            # Detect "stuck" cases: intent pushing into blocked direction repeatedly
                            stuck_flag = False
                            if str(risk).lower() == 'stop' and pressing_dir != 'idle':
                                # Heuristic: if pressing forward into a blocked center lane
                                if pressing_dir == 'forward' and st_C == 'blocked':
                                    stuck_flag = True
                                # Or pressing left/right into a blocked side lane while others are constrained/blocked
                                if pressing_dir == 'left' and st_L == 'blocked' and st_C != 'clear' and st_R != 'clear':
                                    stuck_flag = True
                                if pressing_dir == 'right' and st_R == 'blocked' and st_C != 'clear' and st_L != 'clear':
                                    stuck_flag = True
                            if stuck_flag:
                                lines.append("stuck: true")
                            # mirror suggestion tokens if requested (swap left/right within combos)
                            if getattr(args, 'mirror_view', False):
                                mirror_map = {
                                    'left': 'right', 'right': 'left',
                                    'stay-left': 'stay-right', 'stay-right': 'stay-left',
                                    'mid-left': 'mid-right', 'mid-right': 'mid-left',
                                    'left-right': 'left-right'
                                }
                                auto_suggest = mirror_map.get(
                                    auto_suggest, auto_suggest)
                            lines.append(f"auto_suggest: {auto_suggest}")
                            if current_ticket is not None:
                                current_ticket['last_auto_suggest'] = auto_suggest
                            if scene_snippet is None:
                                try:
                                    if all(np.isfinite([_num(d_L), _num(d_C), _num(d_R)])):
                                        scene_snippet = f"lanes3 {(_num(d_L)):.1f}/{(_num(d_C)):.1f}/{(_num(d_R)):.1f}m"
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Append compact per-ticket memory summary to help VLM stay consistent within a ticket
                    try:
                        mem_counts = ((current_ticket or {}).get(
                            'mem') or {}).get('counts', {})
                        if mem_counts:
                            now_ts = time.time()
                            cand = []
                            for (name, side), v in mem_counts.items():
                                if (now_ts - float(v.get('last_seen', 0.0))) <= 10.0:
                                    md = v.get('min_dist')
                                    cnt = int(v.get('count', 0))
                                    cand.append((cnt, md if isinstance(
                                        md, (int, float)) else 999.0, f"{name}:{(md if isinstance(md, (int, float)) else 0.0):.2f}m {side}"))
                            if cand:
                                cand.sort(key=lambda x: (-x[0], x[1]))
                                mem_top = ", ".join([c[2] for c in cand[:3]])
                                if mem_top:
                                    lines.append(f"memory_list: {mem_top}")
                    except Exception:
                        pass
                    # If no other snippet, fall back to nearest distance
                    if scene_snippet is None and nearest_txt != "?":
                        scene_snippet = f"nearest {nearest_txt}"
                    # Stash snippet for VLM fallback augmentation
                    try:
                        if current_ticket is not None:
                            current_ticket['last_scene_snippet'] = scene_snippet
                    except Exception:
                        pass
                    if dialog.get("pending") and dialog.get("proposed_dir"):
                        lines.append(f"ask: take {dialog['proposed_dir']}?")
                    if dialog.get("last_reply"):
                        lines.append(f"user_reply: {dialog['last_reply']}")
                    # If we have detected a stuck case, explicitly ask the VLM for help to get unstuck
                    try:
                        if 'stuck: true' in lines:
                            lines.append(
                                "help_request: user appears stuck; propose how to get unstuck using available space or suggest small camera movements to reassess unseen areas.")
                    except Exception:
                        pass
                    user_txt = "\n".join(lines)
                    try:
                        can_enqueue = (not vlm_inflight) and vlm_q.empty()
                        if can_enqueue:
                            vlm_q.put_nowait(
                                (send_np, user_txt, active_ticket_id))
                    except Exception as e:
                        print(
                            "[vlm_on_change_qwen] Failed to enqueue VLM request:", e)

                prev_risk = risk
                prev_dir = effective_dir

            if args.show and latest_color is not None:
                vis = latest_color.copy()
                # Optional debug overlay for depth lanes (display only 3 macro bands L/C/R)
                if getattr(args, 'debug_lanes', False) and latest_depth is not None:
                    try:
                        H, W = latest_depth.shape[:2]
                        y1 = int(0.55 * H)
                        y2 = int(0.95 * H)
                        xL1, xL2 = 0, int(W/3)
                        xC1, xC2 = int(W/3), int(2*W/3)
                        xR1, xR2 = int(2*W/3), W
                        d_L = sw.median_depth_in_box(
                            latest_depth, xL1, y1, xL2, y2)
                        d_C = sw.median_depth_in_box(
                            latest_depth, xC1, y1, xC2, y2)
                        d_R = sw.median_depth_in_box(
                            latest_depth, xR1, y1, xR2, y2)
                        if getattr(args, 'mirror_view', False):
                            d_L, d_R = d_R, d_L
                        CLEAR_T = float(
                            getattr(args, 'clear_threshold_m', 1.8))

                        def draw_band(x1, y1b, x2, y2b, d, label):
                            col = (0, 200, 0) if (isinstance(
                                d, (int, float)) and d >= CLEAR_T) else (0, 0, 200)
                            overlay = vis.copy()
                            cv2.rectangle(overlay, (x1, y1b),
                                          (x2, y2b), col, -1)
                            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
                            txt = '?' if not isinstance(
                                d, (int, float)) or not np.isfinite(d) else f"{d:.2f}m"
                            cv2.putText(vis, f"{label}:{txt}", (x1+4, y1b-6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2, cv2.LINE_AA)
                        draw_band(xL1, y1, xL2, y2, d_L, 'L')
                        draw_band(xC1, y1, xC2, y2, d_C, 'C')
                        draw_band(xR1, y1, xR2, y2, d_R, 'R')
                        cv2.putText(vis, f"CLR>{CLEAR_T:.1f}m", (
                            10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                if last_objects:
                    for o in last_objects:
                        try:
                            x1, y1, x2, y2 = map(
                                int, o.get("bbox_xyxy", [0, 0, 0, 0]))
                            is_hazard = (o.get("id") in last_hazards) or (
                                o.get("ontology_class") == "hazard")
                            color = (0, 0, 255) if is_hazard else (255, 255, 0)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                            base = o.get("canonical_class") or o.get(
                                "display_label") or o.get("raw_label") or "obj"
                            dist = o.get("distance_m")
                            dtxt = "?" if dist is None else f"{float(dist):.2f}m"
                            label = f"{base} {dtxt}"
                            cv2.putText(vis, label, (x1, max(
                                20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                        except Exception:
                            continue
                if 'vlm_inflight' in locals() and vlm_inflight:
                    h, w = vis.shape[:2]
                    overlay = vis.copy()
                    bar_w = max(60, int(w * 0.25))
                    phase = (time.time() * 1.5) % 1.0
                    start = int(phase * (w + bar_w)) - bar_w
                    x1 = max(0, start)
                    x2 = min(w, start + bar_w)
                    y1 = h - 6 - 18
                    y2 = h - 6
                    cv2.rectangle(overlay, (x1, y1),
                                  (x2, y2), (0, 215, 255), -1)
                    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
                    intent_txt = (current_ticket or {}).get(
                        'intent_dir') if current_ticket else effective_dir
                    txt = f"VLM is assessing the environment for going {intent_txt}..."
                    cv2.putText(vis, txt, (10, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 215, 255), 1, cv2.LINE_AA)
                if last_caption:
                    overlay = vis.copy()
                    # Show caption with human-readable lane naming aligned to the badge
                    cap_show = _caption_with_human_lane(last_caption)
                    lines = _wrap_text(cap_show, width=70)
                    pad, lh = 8, 20
                    block_h = pad*2 + lh*len(lines)
                    h, w = vis.shape[:2]
                    cv2.rectangle(overlay, (0, h - block_h),
                                  (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                    intent_show = pressing_dir if pressing_dir != 'idle' else effective_dir
                    nearest_txt = f"{last_nearest_m:.2f}m" if isinstance(
                        last_nearest_m, (int, float)) else "?"
                    ticket_txt = f" ticket:{active_ticket_id}" if active_ticket_id else ""
                    info_txt = f"intent:{intent_show}  going:{effective_dir}  nearest:{nearest_txt}  risk:{prev_risk or '?'}  decision:{last_decision}{ticket_txt}"
                    cv2.putText(vis, info_txt, (10, h - block_h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                    y = h - block_h + pad + 14
                    for ln in lines:
                        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        y += lh
                else:
                    h, w = vis.shape[:2]
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (0, h - 26), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
                    intent_show = pressing_dir if pressing_dir != 'idle' else effective_dir
                    nearest_txt = f"{last_nearest_m:.2f}m" if isinstance(
                        last_nearest_m, (int, float)) else "?"
                    ticket_txt = f" ticket:{active_ticket_id}" if active_ticket_id else ""
                    info_txt = f"intent:{intent_show}  going:{effective_dir}  nearest:{nearest_txt}  risk:{prev_risk or '?'}  decision:{last_decision}{ticket_txt}"
                    cv2.putText(vis, info_txt, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (230, 230, 230), 1, cv2.LINE_AA)
                htop, wtop = vis.shape[:2]
                header = vis.copy()
                cv2.rectangle(header, (0, 0), (wtop, 24), (30, 30, 30), -1)
                intent_txt = (current_ticket or {}).get(
                    'intent_dir') if current_ticket else '?'
                status_txt = (current_ticket or {}).get(
                    'status') if current_ticket else 'idle'
                last_llm_ts_val = float(
                    (current_ticket or {}).get('last_llm_ts') or 0.0)
                if vlm_inflight:
                    next_txt = f"assessing for {intent_txt}..."
                elif not vlm_q.empty():
                    next_txt = 'queued...'
                elif last_llm_ts_val > 0.0:
                    rem = max(0, int(10 - (time.time() - last_llm_ts_val)))
                    next_txt = f"VLM updating in {rem}s"
                else:
                    next_txt = 'next:—'
                head_txt = f"Ticket #{active_ticket_id} | intent:{intent_txt} | status:{status_txt} | {next_txt}"
                cv2.putText(header, head_txt, (10, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (240, 240, 240), 1, cv2.LINE_AA)
                # Indicate mirrored view unless user suppresses the tag
                if getattr(args, 'mirror_view', False) and not getattr(args, 'no_mirror_tag', False):
                    tag = "MIRROR VIEW: left/right swapped"
                    cv2.putText(header, tag, (wtop - 10 - 250, 16), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 215, 255), 1, cv2.LINE_AA)
                cv2.addWeighted(header, 0.8, vis, 0.2, 0, vis)
                # Top-right safe path badge showing chosen 3-band (or combo) suggestion token (compute every frame)
                try:
                    def _num(v):
                        try:
                            return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else -1.0
                        except Exception:
                            return -1.0
                    frame_safe_token = None
                    if latest_depth is not None:
                        H2, W2 = latest_depth.shape[:2]
                        yy1, yy2 = int(0.55 * H2), int(0.95 * H2)
                        xL1, xL2 = 0, int(W2/3)
                        xC1, xC2 = int(W2/3), int(2*W2/3)
                        xR1, xR2 = int(2*W2/3), W2
                        d_L = sw.median_depth_in_box(
                            latest_depth, xL1, yy1, xL2, yy2)
                        d_C = sw.median_depth_in_box(
                            latest_depth, xC1, yy1, xC2, yy2)
                        d_R = sw.median_depth_in_box(
                            latest_depth, xR1, yy1, xR2, yy2)
                        if getattr(args, 'mirror_view', False):
                            d_L, d_R = d_R, d_L
                        CLEAR_T = float(
                            getattr(args, 'clear_threshold_m', 1.8))
                        is_L = _num(d_L) >= CLEAR_T
                        is_C = _num(d_C) >= CLEAR_T
                        is_R = _num(d_R) >= CLEAR_T
                        cc = sum([is_L, is_C, is_R])
                        if cc == 0:
                            frame_safe_token = 'back'
                        elif cc == 3:
                            frame_safe_token = 'continue'
                        elif cc == 1:
                            if is_C:
                                frame_safe_token = 'continue'
                            elif is_L:
                                frame_safe_token = 'left'
                            else:
                                frame_safe_token = 'right'
                        else:  # two
                            if is_C and is_L and not is_R:
                                frame_safe_token = 'mid-left'
                            elif is_C and is_R and not is_L:
                                frame_safe_token = 'mid-right'
                            elif is_L and is_R and not is_C:
                                frame_safe_token = 'left-right'
                            else:
                                frame_safe_token = 'continue'

                    # Prefer per-frame token; fall back to stored or caption-based if unavailable
                    safe_token = frame_safe_token
                    if safe_token is None and current_ticket and 'last_auto_suggest' in current_ticket:
                        safe_token = current_ticket.get('last_auto_suggest')
                    if safe_token is None and last_caption:
                        m = re.search(
                            r"Suggest:\s*([a-z\-]+)", last_caption, flags=re.IGNORECASE)
                        if m:
                            safe_token = m.group(1).lower()

                    if safe_token:
                        mapping = {
                            'continue': 'MID',
                            'left': 'LEFT',
                            'right': 'RIGHT',
                            'stay-left': 'LEFT',
                            'stay-right': 'RIGHT',
                            'mid-left': 'MID-LEFT',
                            'mid-right': 'MID-RIGHT',
                            'left-right': 'LEFT-RIGHT',
                            'back': 'NONE'
                        }
                        band_name = mapping.get(safe_token, safe_token.upper())
                        txt = f"SAFE PATH: {band_name}"
                        (tw, th), _ = cv2.getTextSize(
                            txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        pad = 6
                        bw, bh = tw + pad*2, th + pad*2
                        x2, y1 = wtop - 10, 6
                        x1, y2 = x2 - bw, y1 + bh
                        col = (0, 180, 0) if 'none' not in band_name.lower() else (
                            0, 0, 215)
                        overlay2 = vis.copy()
                        cv2.rectangle(overlay2, (x1, y1), (x2, y2), col, -1)
                        cv2.addWeighted(overlay2, 0.85, vis, 0.15, 0, vis)
                        cv2.putText(vis, txt, (x1 + pad, y2 - pad - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
                cv2.imshow('VLM on change (Qwen3-VL)', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        try:
            kbd.stop()
        except Exception:
            pass
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
