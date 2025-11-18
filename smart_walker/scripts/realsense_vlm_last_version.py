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
                if 'left' in text and 'right' not in text:
                    sug = 'left'
                elif 'right' in text and 'left' not in text:
                    sug = 'right'
                return adv, sug
    except Exception:
        pass
    return None, None


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
        "Begin with the decision token in uppercase, formatted as either 'Advice: <SAFE|CAUTION|STOP>:' or '<SAFE|CAUTION|STOP>:' followed by a very short reason. "
        "If STOP and a safer alternative looks possible, append 'Suggest: left' or 'Suggest: right' or 'Suggest: back' at the end. "
        "Plain language only; no labels (like intent/objects), no lists, no JSON, no line breaks, no extra text."),
        help="System: one-sentence scene assessment + decision; start with Advice: <LEVEL>: or <LEVEL>:; optional 'Suggest: <dir>' for STOP only.")
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
                    # Too stale to accept; drop and log
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
            if s.last_edge_dir != 'idle':
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
                should_send = base_send and (not in_hold or stop_case)

                if should_send and latest_color is not None:
                    # Downscale to model-preferred size before sending
                    send_np = cv2.resize(
                        latest_color, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
                    # Build a compact, structured scene summary for the VLM
                    objs_n = len(out.objects)
                    haz_n = len(out.hazards)
                    nearest_txt = f"{last_nearest_m:.2f}m" if isinstance(
                        last_nearest_m, (int, float)) else "?"
                    lines = [
                        f"intent: {effective_dir}; decision: {last_decision}; risk: {risk}",
                        f"objects: {objs_n}; hazards: {haz_n}; nearest: {nearest_txt}",
                        f"braking: {'yes' if str(risk).lower() == 'stop' else 'no'}; pushing: {'yes' if continuing_push else 'no'}; push_dir: {pressing_dir}",
                    ]
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
                    # Simple assessing indicator (no count-up)
                    txt = "LLM assessing…"
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
                    next_txt = 'assessing…'
                elif not vlm_q.empty():
                    next_txt = 'queued…'
                elif last_llm_ts_val > 0.0:
                    rem = max(0, int(10 - (time.time() - last_llm_ts_val)))
                    next_txt = f"next:{rem}s"
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
                cv2.addWeighted(header, 0.8, vis, 0.2, 0, vis)
                # STOP braking-first message
                if (prev_risk or '').lower() == 'stop':
                    cv2.rectangle(vis, (0, 24), (wtop, 44), (0, 0, 0), -1)
                    cv2.putText(vis, "STOP — we are braking. Waiting for LLM assessment…", (
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
