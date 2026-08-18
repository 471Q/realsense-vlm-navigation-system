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


def _call_vlm_with_fallbacks(endpoint: str, base_img: np.ndarray, args) -> str:
    """Try multiple encodings/sizes when server says 'failed to process image'."""
    attempts: list[tuple[str, int, int]] = []  # (fmt, quality, size)
    size0 = int(getattr(args, 'image_size', 336) or 336)
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
            img = base_img
            if img.shape[0] != sz or img.shape[1] != sz:
                img = cv2.resize(base_img, (sz, sz),
                                 interpolation=cv2.INTER_AREA)
            # Stash user text for payload build without threading globals
            # We attach it temporarily to args for _build_payload_from_image_array
            payload, desc = _build_payload_from_image_array(img, fmt, q, args)
            content = call_vlm(endpoint, payload, timeout=60)
            if desc != '':
                print(f"[vlm_on_change] VLM accepted image encoding: {desc}")
            return content
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if 'failed to process image' in msg:
                print(
                    f"[vlm_on_change] VLM could not process image with {fmt}/q{q}/sz{sz}; trying next…")
                continue
            # Different error; re-raise immediately
            raise
    # Exhausted attempts
    if last_err:
        raise last_err
    raise RuntimeError(
        'VLM image processing failed with all encodings (unexpected)')


def call_vlm(endpoint: str, payload: dict, timeout: int | tuple = 45) -> str:
    url = endpoint.rstrip('/') + '/v1/chat/completions'
    r = requests.post(url, json=payload, timeout=timeout)
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
            # Accept either a colon or a semicolon after the level to handle
            # variants like "Advice: STOP; Suggest: back" or "STOP: ..."
            m = re.match(
                r"\s*Advice:\s*(SAFE|CAUTION|STOP)\s*[:;]\s*(.*)", ln, flags=re.IGNORECASE)
            if not m:
                m = re.match(
                    r"\s*(SAFE|CAUTION|STOP)\s*[:;]\s*(.*)", ln, flags=re.IGNORECASE)
            if m:
                adv = m.group(1).upper()
                text = (m.group(2) or '').lower()
                sug = None
                # Accept broader suggestion vocabulary
                if 'suggest:' in text:
                    # capture tokens after Suggest:
                    ms = re.search(r"suggest\s*[:]\s*([a-z\-]+)", text)
                    if ms:
                        token = ms.group(1).strip()
                        if token in ('left', 'right', 'back', 'continue', 'stay-left', 'stay-right'):
                            sug = token
                else:
                    # backward-compat: infer from presence of words
                    if 'left' in text and 'right' not in text:
                        sug = 'left'
                    elif 'right' in text and 'left' not in text:
                        sug = 'right'
                return adv, sug
    except Exception:
        pass
    return None, None


def _ensure_suggest(adv: Optional[str], sug: Optional[str], content: str, fallback: Optional[str]) -> str:
    """Ensure caption contains a 'Suggest: ...' clause; append fallback when missing."""
    try:
        if 'suggest:' in (content or '').lower():
            # Normalize tokens to align with Advice semantics
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
                # New: conflicting STOP + continue → replace with fallback or 'back'
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
        # normalize tokens
        token = add.lower()
        if adv == 'CAUTION' and token in ('left', 'right'):
            token = 'stay-left' if token == 'left' else 'stay-right'
        if adv == 'SAFE' and token in ('left', 'right'):
            token = 'continue'
        if adv == 'STOP' and token in ('stay-left', 'stay-right'):
            token = 'left' if token == 'stay-left' else 'right'
        if token in ('left', 'right', 'back', 'continue', 'stay-left', 'stay-right'):
            return content.rstrip() + f" Suggest: {token}"
        return content
    except Exception:
        return content


def _ensure_scene_detail(content: str, detail: Optional[str]) -> str:
    """Guarantee a minimal scene description before 'Suggest:'.
    We require an explicit distance (like '1.2 m'/'1.2m') or an existing lane snippet; otherwise
    we inject our compact detail (near: objs / lanes L/C/R / nearest x.xm).
    """
    try:
        if not detail:
            return content
        s = (content or "").lower()
        has_distance = bool(re.search(r"\b\d+(?:\.\d+)?\s*m\b", s))
        has_lanes = 'lanes' in s
        if has_distance or has_lanes:
            return content
        idx = s.rfind('suggest:')
        if idx != -1:
            pre = content[:idx].rstrip().rstrip('.')
            post = content[idx:]
            return f"{pre}; {detail} {post}"
        return content.rstrip().rstrip('.') + f"; {detail}"
    except Exception:
        return content


def _is_system_echo(content: str) -> bool:
    """Detect if the model echoed our system prompt; hide it from overlay."""
    s = (content or "").lower()
    return (
        "you are a safety-first vision assistant" in s
        or "output format (no extra text)" in s
        or s.startswith("you are a safety-first vision assistant")
    )


def _is_valid_caption(content: str) -> bool:
    """Strict caption check used when --strict_caption is passed.
    Accept either:
      - Single-line starting with 'Advice:' (preferred now), or
      - Two-line where first starts with 'Scene:' and second with 'Advice:' (legacy support).
    """
    try:
        if not content:
            return False
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines:
            return False
        # Preferred: single line starting with Advice: or a bare decision token
        if len(lines) == 1:
            s0 = lines[0].lower()
            if s0.startswith('advice:'):
                return True
            if s0.startswith('safe:') or s0.startswith('caution:') or s0.startswith('stop:'):
                return True
            return False
        # Legacy: first Scene:, second Advice:
        if len(lines) >= 2:
            return (lines[0].lower().startswith('scene:') and lines[1].lower().startswith('advice:'))
        return False
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Event-driven RealSense → VLM (trigger on risk/intent change)")

    # VLM parameters
    ap.add_argument('--endpoint', type=str, default='http://localhost:8080',
                    help='OpenAI-compatible VLM server base URL (llama.cpp)')
    ap.add_argument('--model', type=str, default='llava-1.5-7b',
                    help='VLM model name/id on server')
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--max_tokens', type=int, default=160)
    ap.add_argument('--system', type=str, default=(
        "You are a safety-first vision assistant for an elderly user's smart walker. "
        "Use only the provided scene facts; do not invent.\n\n"
        "Output exactly ONE natural, human-sounding sentence that briefly describes the scene and clearly states the safety decision. "
        "Always include at least one short concrete detail drawn from the facts (e.g., a specific object with distance and side, or lane clearance). "
        "Begin with the decision token in uppercase, formatted as either 'Advice: <SAFE|CAUTION|STOP>:' or '<SAFE|CAUTION|STOP>:' followed by a very short reason. "
        "Always append a concise 'Suggest: ...' at the end: for SAFE use 'continue'; for CAUTION use 'stay-left' or 'stay-right' (keep to the clearer side); for STOP use 'left' or 'right' if plausible, otherwise 'back'. "
        "If a memory_list is provided for this ticket, prefer consistent object names based on it; do not switch 'refrigerator' to 'door' unless the new view clearly contradicts prior context. If uncertain, keep the earlier name or say 'object'. "
        "Plain language only; no labels (like intent/objects), no lists, no JSON, no line breaks, no extra text."),
        help="System: one-sentence scene assessment + decision; always end with 'Suggest: <continue|stay-left|stay-right|left|right|back>'.")
    ap.add_argument('--vlm_timeout_s', type=float, default=25.0,
                    help='Max seconds to wait for a single VLM response before timing out; set 0 to disable (patient mode). UI shows assessing timer while waiting')
    ap.add_argument('--late_accept_s', type=float, default=1.5,
                    help='Grace window (seconds) to accept late VLM results after a ticket switch for overlay display')
    ap.add_argument('--strict_caption', action='store_true',
                    help="Only display captions that start with 'Advice:' or 'SAFE:/CAUTION:/STOP:'; otherwise show raw text")

    # Trigger policy
    ap.add_argument('--min_interval_s', type=float, default=2.0,
                    help='Minimum seconds between VLM calls (debounce)')

    # Camera / encoding
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--image_size', type=int, default=336,
                    help='Resize square dimension before sending to VLM')
    ap.add_argument('--jpeg_quality', type=int, default=70)
    ap.add_argument('--encode', choices=['jpeg', 'png'], default='jpeg',
                    help='Image encoding format for VLM payload (default: jpeg)')
    ap.add_argument('--process_hz', type=float, default=0.0,
                    help='Max processing rate for detector/risk. 0 = unlimited. Use to emulate e.g. 2 FPS.')
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
    print(f"[vlm_on_change] depth_scale = {depth_scale:.6f} m/unit")

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
                print("[vlm_on_change] YOLO on CUDA (FP16)")
            else:
                print("[vlm_on_change] YOLO on CUDA (FP32)")
                # Print quick CUDA memory stats for assurance
                try:
                    alloc = torch.cuda.memory_allocated(0) / (1024**2)
                    reserv = torch.cuda.memory_reserved(0) / (1024**2)
                    print(
                        f"[vlm_on_change] CUDA mem: allocated={alloc:.1f}MB reserved={reserv:.1f}MB")
                except Exception:
                    pass
            # Optional warmup to verify CUDA path and reduce first-frame latency
            if getattr(args, 'warmup', False):
                try:
                    dev_name = torch.cuda.get_device_name(0)
                    print(
                        f"[vlm_on_change] CUDA device: {dev_name}; torch {getattr(torch, '__version__', '?')}; cuda {getattr(torch.version, 'cuda', None)}")
                    sz = int(max(64, min(getattr(args, 'imgsz', 320), 640)))
                    dummy = np.zeros((sz, sz, 3), dtype=np.uint8)
                    t0 = time.time()
                    _ = model.predict(
                        source=dummy, imgsz=sz, conf=0.25, verbose=False, half=bool(args.half))
                    torch.cuda.synchronize()
                    dt = (time.time() - t0) * 1000.0
                    print(
                        f"[vlm_on_change] YOLO CUDA warmup: {dt:.1f} ms (imgsz={sz})")
                except Exception as we:
                    print("[vlm_on_change] YOLO warmup failed:", we)
        except Exception as e:
            print("[vlm_on_change] Failed to move YOLO to CUDA, staying on CPU:", e)
    else:
        if args.half:
            print(
                "[vlm_on_change] --half requested but CUDA is not available; running on CPU FP32.")

    __import__('threading').Thread(target=sw.capture_thread, args=(
        pipe, align, depth_scale, cap_q, stop_evt, False, None, sw.IMU_MAX_DRAIN_PER_LOOP), daemon=True).start()
    # Decide FP16 only if running on CUDA
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
    # Attach ticket_id to VLM work items to avoid publishing stale captions
    vlm_q: '_Q[tuple[np.ndarray, str, int]]' = _Q(maxsize=1)
    vlm_inflight: bool = False

    # Simple dialog state to support Y/N confirmation of alternative path
    dialog = {
        "pending": False,
        "proposed_dir": None,  # 'left' | 'right'
        "since_ms": 0,
        "last_reply": None,  # 'yes' | 'no' | None
    }

    def _compute_clear_side(objs):
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

    def _vlm_worker():
        nonlocal last_caption, last_vlm_ts, vlm_inflight, active_ticket_id, current_ticket
        while not stop_evt.is_set():
            try:
                item = vlm_q.get(timeout=0.05)
            except Empty:
                continue
            try:
                send_np, user_txt, t_id = item
                # Mark inflight at the moment we start the actual request
                vlm_inflight = True
                mime, b64 = _encode_image(
                    send_np, fmt=args.encode, quality=int(args.jpeg_quality))
                # Make user text accessible to the fallback builder in this scope only
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
                # Decide acceptance policy for caption
                accept_for_ticket = (int(t_id) == int(active_ticket_id))
                accept_unbound = (int(t_id) == 0)
                try:
                    grace = float(getattr(args, 'late_accept_s', 1.5))
                except Exception:
                    grace = 1.5
                recently_switched = (
                    time.time() - last_ticket_switch_ts) <= grace

                if accept_for_ticket or accept_unbound or recently_switched:
                    # Guard: ignore echo of system prompt; otherwise prefer showing what the model said
                    if _is_system_echo(content):
                        print(
                            f"[warn] VLM returned system prompt text; ignoring overlay. Raw len={len(content)}")
                    elif not _is_valid_caption(content):
                        if getattr(args, 'strict_caption', False):
                            print(
                                f"[warn] VLM returned non-minimal caption; ignoring overlay. First 80 chars: {content[:80]!r}")
                        else:
                            # Show the raw caption even if it doesn't match the strict template
                            print(
                                f"[info] Accepting non-minimal caption for overlay (strict_caption=off). First 80 chars: {content[:80]!r}")
                            last_caption = content
                    else:
                        # Keep model's suggestion if present; we'll compute a fallback if missing
                        adv_tmp, sug_tmp = _parse_advice(content)
                        # Ensure a suggestion is present; if model omitted, append our auto_suggest
                        try:
                            fallback = (current_ticket or {}).get(
                                'last_auto_suggest')
                        except Exception:
                            fallback = None
                        content = _ensure_suggest(
                            adv_tmp, sug_tmp, content, fallback)
                        # Ensure a minimal concrete detail is present; fall back to our scene snippet
                        try:
                            scene_fallback = (current_ticket or {}).get(
                                'last_scene_snippet')
                        except Exception:
                            scene_fallback = None
                        content = _ensure_scene_detail(content, scene_fallback)
                        last_caption = content
                    last_vlm_ts = time.time()

                    if accept_for_ticket:
                        # Mark the time we actually received an LLM response (drives 10s reconfirm window)
                        try:
                            if current_ticket is not None:
                                current_ticket['last_llm_ts'] = last_vlm_ts
                        except Exception:
                            pass
                        # Update ticket status from Advice
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
                        # Populate dialog only from LLM suggestion (remove rule-based suggestion)
                        try:
                            if adv == 'STOP' and sug in ('left', 'right'):
                                dialog.update({
                                    'pending': True,
                                    'proposed_dir': sug,
                                    'since_ms': int(time.time()*1000),
                                    'last_reply': None,
                                })
                            else:
                                # Clear pending suggestion if not applicable
                                if dialog.get('pending'):
                                    dialog['pending'] = False
                                    dialog['proposed_dir'] = None
                                    dialog['last_reply'] = None
                        except Exception:
                            pass
                        # If SAFE/CAUTION, set moving_since if not set
                        try:
                            if current_ticket is not None and current_ticket.get('status') in ('moving',):
                                if not current_ticket.get('moving_since'):
                                    current_ticket['moving_since'] = time.time(
                                    )
                            else:
                                # risk STOP or other: reset moving_since
                                if current_ticket is not None:
                                    current_ticket['moving_since'] = None
                        except Exception:
                            pass
                        print(
                            f"[{int(last_vlm_ts*1000)}] VLM (ticket {t_id}) accepted; overlay updated")
                else:
                    # Too stale to accept for the current ticket. In permissive mode, still update the overlay
                    # so the user can see what the model said, but do not update ticket timers/state.
                    if not getattr(args, 'strict_caption', False):
                        if _is_system_echo(content):
                            print(
                                f"[warn] Stale VLM result echoed system prompt; overlay not updated. ticket={t_id} active={active_ticket_id}")
                        else:
                            print(
                                f"[info] Accepting late/stale VLM caption for overlay (strict_caption=off). ticket={t_id} active={active_ticket_id} First 80 chars: {content[:80]!r}")
                            last_caption = content
                            last_vlm_ts = time.time()
                    else:
                        # Strict mode: drop stale caption to avoid confusing guidance
                        print(
                            f"[info] Dropped stale VLM result for ticket {t_id}; active={active_ticket_id}")
                # Clear inflight on completion
                vlm_inflight = False
            except Exception as e:
                print("[vlm_on_change] VLM worker failed:", e)
                # Back off according to min_interval by marking last_vlm_ts now
                last_vlm_ts = time.time()
                vlm_inflight = False

    __import__('threading').Thread(target=_vlm_worker, daemon=True).start()

    prev_risk: Optional[str] = None
    prev_dir: Optional[str] = None
    last_vlm_ts = 0.0
    last_caption: Optional[str] = None
    latest_color: Optional[np.ndarray] = None
    latest_depth: Optional[np.ndarray] = None
    last_decision: str = "GO"
    last_nearest_m: Optional[float] = None
    # Ticketing state
    next_ticket_id: int = 1
    active_ticket_id: int = 0
    # dict with keys: id, intent_dir, status, last_llm_ts, proposed_dir, last_user_reply
    current_ticket = None
    # Proceed banner removed; suggestions are LLM-driven and shown in ticket header area
    # Toast for ticket opened/closed and nag-suppression after decline
    TICKET_BANNER_UNTIL = 0.0
    TICKET_BANNER_TEXT: Optional[str] = None
    CLOSE_BANNER_UNTIL = 0.0
    CLOSE_BANNER_TEXT: Optional[str] = None
    DECLINE_COOLDOWN_UNTIL = 0.0
    # Track last ticket switch/open time to allow brief acceptance of late results
    last_ticket_switch_ts = 0.0
    last_objects = []
    last_hazards = []
    SUCCESS_HOLD_S = 1.5  # seconds to consider a move successful after SAFE/CAUTION

    print("[vlm_on_change] running; triggers on risk/intent change. Q to quit. W/A/S/D for intents.")
    try:
        last_proc_t = 0.0
        proc_period = (1.0 / float(args.process_hz)
                       ) if args.process_hz and args.process_hz > 0 else 0.0
        while True:
            # UI pump
            if hasattr(cv2, 'pollKey'):
                cv2.pollKey()
            else:
                cv2.waitKey(1)

            # Move capture → inference
            try:
                pkt = cap_q.get_nowait()
                latest_color = pkt.color
                try:
                    latest_depth = pkt.depth_m
                except Exception:
                    latest_depth = None
                # Rate limit forwarding into inference to achieve effective process_hz
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
                else:
                    # Skip forwarding; keep latest_color for optional preview and VLM image
                    pass
            except Empty:
                pass

            # Latest inference
            out: Optional[sw.InferPacket] = None
            try:
                out = inf_out.get_nowait()
            except Empty:
                pass

            # Snapshot intent
            s = kbd.snapshot()
            if s.quit_requested:
                break
            now_ms = int(time.time() * 1000)
            active = (now_ms - s.last_press_ms) <= sw.ACTIVE_KEY_WINDOW_MS
            # Dialog Y/N capture (non-blocking)
            if dialog.get("pending"):
                if s.yes_edge:
                    dialog["last_reply"] = "yes"
                    dialog["pending"] = False
                elif s.no_edge:
                    dialog["last_reply"] = "no"
                    dialog["pending"] = False
            # If user accepted LLM suggestion, open a new ticket with that direction
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
                # Ticket opened toast
                TICKET_BANNER_TEXT = f"Ticket #{active_ticket_id} opened: going {current_ticket['intent_dir']}"
                TICKET_BANNER_UNTIL = time.time() + 2.5
                print(
                    f"[ticket] opened #{active_ticket_id} going {current_ticket['intent_dir']}")
                # reset dialog
                dialog.update({'pending': False, 'proposed_dir': None,
                              'last_reply': None, 'since_ms': now_ms})
                last_vlm_ts = 0.0
                last_ticket_switch_ts = time.time()
            elif dialog.get("last_reply") == "no" and dialog.get("proposed_dir") and not dialog.get("pending"):
                # Declined suggestion: clear and apply anti-nag cooldown, do not auto-offer alternate
                dialog.update({'pending': False, 'proposed_dir': None,
                              'since_ms': now_ms, 'last_reply': None})
                # anti-nag cooldown after decline
                DECLINE_COOLDOWN_UNTIL = time.time() + 4.0
            # remember last non-idle direction
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
                current_ticket = None
                active_ticket_id = 0
                dialog.update(
                    {'pending': False, 'proposed_dir': None, 'last_reply': None})
                last_vlm_ts = 0.0
                last_ticket_switch_ts = time.time()
                last_non_idle_dir = 'idle'
            elif s.last_edge_dir != 'idle':
                last_non_idle_dir = s.last_edge_dir
                # Ticket open on new intent edge (start of navigation)
                if current_ticket is None or (current_ticket and current_ticket.get('intent_dir') != s.last_edge_dir):
                    # Close previous ticket if it exists and isn't closed
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
                    # Ticket opened toast
                    TICKET_BANNER_TEXT = f"Ticket #{active_ticket_id} opened: going {s.last_edge_dir}"
                    TICKET_BANNER_UNTIL = time.time() + 2.5
                    print(
                        f"[ticket] opened #{active_ticket_id} going {s.last_edge_dir}")
                    # Reset dialog and allow immediate caption
                    dialog.update(
                        {'pending': False, 'proposed_dir': None, 'last_reply': None, 'since_ms': now_ms})
                    last_vlm_ts = 0.0
                    last_ticket_switch_ts = time.time()
            else:
                # Default to idle until the first directed key press
                last_non_idle_dir = prev_dir or 'idle'
            pressing_dir = getattr(s, 'level_dir', 'idle') or 'idle'
            intent_for_caption = s.last_edge_dir if active else 'idle'
            effective_dir = intent_for_caption if intent_for_caption != 'idle' else (
                last_non_idle_dir or 'idle')

            # Compute risk when we have inference
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
                # enrich facts with risk for decision
                facts_with_risk = dict(facts)
                facts_with_risk["risk"] = risk
                decision, _why = sw.arbiter_decision(
                    effective_dir, facts_with_risk, cfg)
                last_decision = decision
                try:
                    # nearest distance helper available in shared control
                    last_nearest_m = sw._safe_min_distance(out.objects)
                except Exception:
                    last_nearest_m = None
                # cache latest objects/hazards for overlay drawing
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
                            if getattr(args, 'mirror_view', False):
                                if side == 'left':
                                    side = 'right'
                                elif side == 'right':
                                    side = 'left'
                            # Focus memory on reasonably near items
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

                # If hard STOP: do not suggest a direction here; braking-first then wait for LLM assessment
                # If no longer STOP, clear any pending dialog
                if str(risk).lower() != "stop" and dialog.get("pending"):
                    dialog["pending"] = False

                # Consider success/close conditions for current ticket
                try:
                    if current_ticket and current_ticket.get('status') in ('moving',) and (str(risk).lower() != 'stop'):
                        mv_since = current_ticket.get('moving_since') or 0.0
                        if mv_since and (getattr(s, 'level_dir', 'idle') == 'idle') and (time.time() - mv_since >= SUCCESS_HOLD_S):
                            current_ticket['status'] = 'closed'
                            current_ticket['result'] = 'success'
                            current_ticket['end_ts'] = time.time()
                            CLOSE_BANNER_TEXT = f"Ticket #{active_ticket_id} closed: success"
                            CLOSE_BANNER_UNTIL = time.time() + 2.5
                            print(
                                f"[ticket] closed #{active_ticket_id} result=success")
                except Exception:
                    pass

                # Trigger: change in risk/intent OR 10s reconfirm within active ticket OR persistence under STOP while pushing
                changed = (risk != prev_risk) or (effective_dir != prev_dir)
                continuing_push = (str(risk).lower(
                ) == 'stop' and pressing_dir != 'idle' and pressing_dir == effective_dir)
                due = (time.time() - last_vlm_ts) >= float(args.min_interval_s)
                last_llm_ts_val = float(
                    (current_ticket or {}).get('last_llm_ts') or 0.0)
                reconfirm_due = bool(current_ticket) and (current_ticket.get('status') != 'closed') and (
                    time.time() - last_llm_ts_val >= 10.0)
                stop_case = (str(risk).lower() == 'stop' and (
                    continuing_push or dialog.get('pending')))
                base_send = ((changed or reconfirm_due)
                             and due) or (stop_case and due)
                # Hold for 10s after last LLM response: don't send again until reconfirm window, unless STOP case
                in_hold = bool(current_ticket) and (last_llm_ts_val > 0.0) and (
                    (time.time() - last_llm_ts_val) < 10.0)
                # Gate: do not start VLM until the user has expressed intent (i.e., we have an active ticket)
                ticket_active = bool(current_ticket) and (
                    current_ticket.get('status') != 'closed')
                should_send = ticket_active and (
                    base_send and (not in_hold or stop_case))

                if should_send and latest_color is not None:
                    # Downscale to model-preferred size before sending
                    send_np = cv2.resize(
                        latest_color, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
                    # Build a compact, structured scene summary for the VLM
                    objs_n = len(out.objects)
                    haz_n = len(out.hazards)
                    nearest_txt = f"{last_nearest_m:.2f}m" if isinstance(
                        last_nearest_m, (int, float)) else "?"
                    # Include a compact summary of the nearest few detected objects to ground the VLM
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
                    # Depth-driven lane clearance (temporary trial)
                    try:
                        if latest_depth is not None and isinstance(latest_depth, np.ndarray):
                            H, W = latest_depth.shape[:2]
                            y1 = int(0.55 * H)
                            y2 = int(0.95 * H)
                            xL1, xL2 = 0, int(W/3)
                            xC1, xC2 = int(W/3), int(2*W/3)
                            xR1, xR2 = int(2*W/3), W
                            Ld = sw.median_depth_in_box(
                                latest_depth, xL1, y1, xL2, y2)
                            Cd = sw.median_depth_in_box(
                                latest_depth, xC1, y1, xC2, y2)
                            Rd = sw.median_depth_in_box(
                                latest_depth, xR1, y1, xR2, y2)
                            # Mirror semantics for lanes if requested (swap left/right values)
                            if getattr(args, 'mirror_view', False):
                                Ld, Rd = Rd, Ld

                            def _fmt(v):
                                return (f"{v:.2f}m" if isinstance(v, (int, float)) and np.isfinite(v) else "?")
                            lines.append(
                                f"lanes_depth: left:{_fmt(Ld)}, center:{_fmt(Cd)}, right:{_fmt(Rd)}")
                            CLEAR_T = float(
                                getattr(args, 'clear_threshold_m', 1.8))
                            obs_sides = []
                            if isinstance(Ld, (int, float)) and Ld < CLEAR_T:
                                obs_sides.append('left')
                            if isinstance(Cd, (int, float)) and Cd < CLEAR_T:
                                obs_sides.append('center')
                            if isinstance(Rd, (int, float)) and Rd < CLEAR_T:
                                obs_sides.append('right')
                            if obs_sides:
                                lines.append(
                                    f"obstacles_sides_depth: {','.join(obs_sides)}")
                            # Auto-suggest from depth lanes
                            auto_suggest = None
                            if isinstance(Cd, (int, float)) and Cd >= CLEAR_T:
                                auto_suggest = 'continue'
                            else:
                                l_ok = isinstance(
                                    Ld, (int, float)) and Ld >= CLEAR_T
                                r_ok = isinstance(
                                    Rd, (int, float)) and Rd >= CLEAR_T
                                if l_ok or r_ok:
                                    # Prefer cautionary 'stay-<side>' when center is not clear
                                    auto_suggest = 'stay-left' if (
                                        float(Ld or 0) >= float(Rd or 0)) else 'stay-right'
                                else:
                                    # Neither side meets threshold; conservative fallback
                                    auto_suggest = 'back'
                            # Mirror suggestion tokens if requested
                            if getattr(args, 'mirror_view', False):
                                if auto_suggest == 'left':
                                    auto_suggest = 'right'
                                elif auto_suggest == 'right':
                                    auto_suggest = 'left'
                                elif auto_suggest == 'stay-left':
                                    auto_suggest = 'stay-right'
                                elif auto_suggest == 'stay-right':
                                    auto_suggest = 'stay-left'
                            lines.append(f"auto_suggest: {auto_suggest}")
                            if current_ticket is not None:
                                current_ticket['last_auto_suggest'] = auto_suggest
                            if scene_snippet is None:
                                try:
                                    if isinstance(Cd, (int, float)) and np.isfinite(Cd) and isinstance(Ld, (int, float)) and isinstance(Rd, (int, float)):
                                        scene_snippet = f"lanes {Ld:.1f}/{Cd:.1f}/{Rd:.1f}m"
                                        if obs_sides:
                                            scene_snippet += f", obstacles: {','.join(obs_sides)}"
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
                            # Keep only recently seen entries (<= 10s) to avoid stale context
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
                    if scene_snippet is None and nearest_txt != "?":
                        scene_snippet = f"nearest {nearest_txt}"
                    try:
                        if current_ticket is not None:
                            current_ticket['last_scene_snippet'] = scene_snippet
                    except Exception:
                        pass
                    if dialog.get("pending") and dialog.get("proposed_dir"):
                        lines.append(f"ask: take {dialog['proposed_dir']}?")
                    if dialog.get("last_reply"):
                        lines.append(f"user_reply: {dialog['last_reply']}")
                    user_txt = "\n".join(lines)
                    # Enqueue VLM request asynchronously, but do NOT reset countdown while one is active or queued
                    try:
                        can_enqueue = (not vlm_inflight) and vlm_q.empty()
                        if can_enqueue:
                            vlm_q.put_nowait(
                                (send_np, user_txt, active_ticket_id))
                        else:
                            # Skip enqueue to avoid restarting countdown; an assessment is active or queued
                            pass
                    except Exception as e:
                        print("[vlm_on_change] Failed to enqueue VLM request:", e)

                prev_risk = risk
                prev_dir = effective_dir

                # Per-inference compact log line
                nn = (f"{last_nearest_m:.2f}m" if isinstance(
                    last_nearest_m, (int, float)) else "?")
                rules = ",".join(r.get("rules_fired", []))
                print(
                    f"[infer {out.ts_ms}] intent={effective_dir} nearest={nn} risk={risk} decision={last_decision} rules=[{rules}]")
            # Optional preview
            if args.show and latest_color is not None:
                vis = latest_color.copy()
                # Optional debug overlay for depth lanes
                if getattr(args, 'debug_lanes', False) and latest_depth is not None:
                    try:
                        H, W = latest_depth.shape[:2]
                        y1 = int(0.55 * H)
                        y2 = int(0.95 * H)
                        xL1, xL2 = 0, int(W/3)
                        xC1, xC2 = int(W/3), int(2*W/3)
                        xR1, xR2 = int(2*W/3), W
                        Ld = sw.median_depth_in_box(
                            latest_depth, xL1, y1, xL2, y2)
                        Cd = sw.median_depth_in_box(
                            latest_depth, xC1, y1, xC2, y2)
                        Rd = sw.median_depth_in_box(
                            latest_depth, xR1, y1, xR2, y2)
                        CLEAR_T = float(
                            getattr(args, 'clear_threshold_m', 1.8))

                        def draw_band(x1, y1b, x2, y2b, d):
                            col = (0, 200, 0) if (isinstance(
                                d, (int, float)) and d >= CLEAR_T) else (0, 0, 200)
                            overlay = vis.copy()
                            cv2.rectangle(overlay, (x1, y1b),
                                          (x2, y2b), col, -1)
                            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
                            txt = '?' if not isinstance(
                                d, (int, float)) or not np.isfinite(d) else f"{d:.2f}m"
                            cv2.putText(
                                vis, txt, (x1+6, y1b-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
                        draw_band(xL1, y1, xL2, y2, Ld)
                        draw_band(xC1, y1, xC2, y2, Cd)
                        draw_band(xR1, y1, xR2, y2, Rd)
                        cv2.putText(vis, f"CLR>{CLEAR_T:.1f}m", (
                            10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                # Draw detections (lightweight overlay)
                if last_objects:
                    # Color scheme: hazards red, others cyan
                    for o in last_objects:
                        try:
                            x1, y1, x2, y2 = map(
                                int, o.get("bbox_xyxy", [0, 0, 0, 0]))
                            is_hazard = (o.get("id") in last_hazards) or (
                                o.get("ontology_class") == "hazard")
                            color = (0, 0, 255) if is_hazard else (255, 255, 0)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                            # Compact label: canonical or raw + distance
                            base = o.get("canonical_class") or o.get(
                                "display_label") or o.get("raw_label") or "obj"
                            dist = o.get("distance_m")
                            dtxt = "?" if dist is None else f"{float(dist):.2f}m"
                            label = f"{base} {dtxt}"
                            cv2.putText(vis, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, color, 2, cv2.LINE_AA)
                        except Exception:
                            continue
                # Loading bar overlay while VLM request is in-flight
                if 'vlm_inflight' in locals() and vlm_inflight:
                    h, w = vis.shape[:2]
                    overlay = vis.copy()
                    bar_w = max(60, int(w * 0.25))
                    phase = (time.time() * 1.5) % 1.0
                    start = int(phase * (w + bar_w)) - bar_w
                    x1 = max(0, start)
                    x2 = min(w, start + bar_w)
                    y1 = h - 6 - 18  # just above caption block
                    y2 = h - 6
                    cv2.rectangle(overlay, (x1, y1),
                                  (x2, y2), (0, 215, 255), -1)
                    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
                    # Assessing indicator with intended direction
                    intent_txt = (current_ticket or {}).get(
                        'intent_dir') if current_ticket else effective_dir
                    txt = f"VLM is assessing the environment for going {intent_txt}..."
                    cv2.putText(vis, txt, (10, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 215, 255), 1, cv2.LINE_AA)
                # Draw caption block (if any) and always show a compact status line
                if last_caption:
                    overlay = vis.copy()
                    lines = _wrap_text(last_caption, width=70)
                    pad, lh = 8, 20
                    block_h = pad*2 + lh*len(lines)
                    h, w = vis.shape[:2]
                    cv2.rectangle(overlay, (0, h - block_h),
                                  (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                    # Info line just above caption: show ticket/intent and going direction with key context
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
                    # No caption yet: draw a single status strip at the bottom
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
                # Top-of-screen ticket panel and suggestion (LLM-driven)
                htop, wtop = vis.shape[:2]
                # Header: ticket id, intent, status, countdown
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
                # Append result if closed
                res_txt = (current_ticket or {}).get(
                    'result') if current_ticket else None
                if status_txt == 'closed' and res_txt:
                    head_txt = f"Ticket #{active_ticket_id} | intent:{intent_txt} | status:{status_txt}({res_txt}) | {next_txt}"
                else:
                    head_txt = f"Ticket #{active_ticket_id} | intent:{intent_txt} | status:{status_txt} | {next_txt}"
                cv2.putText(header, head_txt, (10, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (240, 240, 240), 1, cv2.LINE_AA)
                if getattr(args, 'mirror_view', False):
                    tag = "MIRROR VIEW: left/right swapped"
                    cv2.putText(header, tag, (wtop - 10 - 250, 16), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 215, 255), 1, cv2.LINE_AA)
                cv2.addWeighted(header, 0.8, vis, 0.2, 0, vis)
                # Top-right risk badge (SAFE/CAUTION/STOP)
                try:
                    risk_s = (prev_risk or '?').lower()
                    if risk_s in ('safe', 'caution', 'stop'):
                        badge_col = (0, 180, 0) if risk_s == 'safe' else (
                            (0, 215, 255) if risk_s == 'caution' else (0, 0, 215))
                        txt = risk_s.upper()
                        (tw, th), _ = cv2.getTextSize(
                            txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        pad = 8
                        bw, bh = tw + pad*2, th + pad*2
                        x2, y1 = wtop - 8, 8
                        x1, y2 = x2 - bw, y1 + bh
                        cv2.rectangle(vis, (x1, y1), (x2, y2), badge_col, -1)
                        cv2.putText(vis, txt, (x1 + pad, y2 - pad - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
                # STOP braking-first message
                if (prev_risk or '').lower() == 'stop':
                    cv2.rectangle(vis, (0, 24), (wtop, 44), (0, 0, 0), -1)
                    cv2.putText(vis, "STOP - we are braking. Waiting for LLM assessment...", (
                        10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)
                # LLM-driven suggestion after assessment
                if dialog.get('pending') and dialog.get('proposed_dir'):
                    cv2.rectangle(vis, (0, 44), (wtop, 68), (0, 0, 0), -1)
                    cv2.putText(vis, f"Suggestion: Take {dialog['proposed_dir']} (Y/N)", (
                        10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
                # If ticket opened toast is active, show it stacked below other banners
                if TICKET_BANNER_TEXT and time.time() < TICKET_BANNER_UNTIL:
                    h4, w4 = vis.shape[:2]
                    cv2.rectangle(vis, (0, 56), (w4, 84), (40, 40, 40), -1)
                    cv2.putText(vis, TICKET_BANNER_TEXT, (10, 78), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2, cv2.LINE_AA)
                elif TICKET_BANNER_TEXT and time.time() >= TICKET_BANNER_UNTIL:
                    TICKET_BANNER_TEXT = None
                # Closed ticket toast stacked below
                if CLOSE_BANNER_TEXT and time.time() < CLOSE_BANNER_UNTIL:
                    h5, w5 = vis.shape[:2]
                    cv2.rectangle(vis, (0, 84), (w5, 112), (30, 30, 30), -1)
                    cv2.putText(vis, CLOSE_BANNER_TEXT, (10, 106), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2, cv2.LINE_AA)
                elif CLOSE_BANNER_TEXT and time.time() >= CLOSE_BANNER_UNTIL:
                    CLOSE_BANNER_TEXT = None
                cv2.imshow('VLM on change', vis)
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
