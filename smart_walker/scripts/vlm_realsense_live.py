import argparse
import base64
import json
import sys
import time
from typing import Optional

import cv2
import numpy as np
import requests

try:
    import pyrealsense2 as rs
except Exception as e:
    print("Failed to import pyrealsense2: ", e, file=sys.stderr)
    rs = None


def rs_pipeline(width: int = 640, height: int = 480, fps: int = 30):
    if rs is None:
        raise RuntimeError("pyrealsense2 not available")
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipe = rs.pipeline()
    pipe.start(cfg)
    return pipe


def frame_to_jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode('.jpg', img_bgr, encode_param)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(enc.tobytes()).decode('ascii')


def build_mm_chat_payload(model: str, system: str, text: str, b64_jpeg: str,
                          temperature: float, top_p: float, max_tokens: int):
    # OpenAI-compatible chat/completions with image part
    # Many servers accept: messages[].content as array of content parts
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}
                    }
                ]
            }
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": False,
    }


def call_vlm(endpoint: str, payload: dict, timeout: int = 30) -> str:
    url = endpoint.rstrip('/') + '/v1/chat/completions'
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(
            f"VLM request failed: {r.status_code} {r.text[:200]}")
    j = r.json()
    return j["choices"][0]["message"]["content"]


def _wrap_text(text: str, width: int = 60):
    # simple word-wrap for overlay rendering
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) > width:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            if cur:
                cur_len += 1 + len(w)
            else:
                cur_len = len(w)
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def main():
    ap = argparse.ArgumentParser(
        description="RealSense -> VLM live client (image+text)")
    ap.add_argument('--endpoint', type=str, default='http://localhost:8080',
                    help='OpenAI-compatible server base URL')
    ap.add_argument('--model', type=str, default='llava-1.5-7b',
                    help='Model name/id on server')
    ap.add_argument('--hz', type=float, default=1.0,
                    help='Frames per second to send')
    ap.add_argument('--width', type=int, default=640,
                    help='Stream width (RealSense color)')
    ap.add_argument('--height', type=int, default=480,
                    help='Stream height (RealSense color)')
    ap.add_argument('--image_size', type=int, default=336,
                    help='Downscale square size sent to VLM (recommend 336 for LLaVA)')
    ap.add_argument('--jpeg_quality', type=int, default=75,
                    help='JPEG quality for encoded frame (lower = smaller/faster)')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--max_tokens', type=int, default=128)
    ap.add_argument('--system', type=str, default=(
        "You are a real-time vision-language assistant. "
        "Given a live camera frame, describe the scene briefly, mention any hazards, "
        "and provide a short suggestion for safe navigation."),
        help='System prompt')
    ap.add_argument('--show', action='store_true',
                    help='Show a preview window with last caption overlaid')
    ap.add_argument('--once', action='store_true',
                    help='Capture one frame, get a response, and exit')
    args = ap.parse_args()

    period = 1.0 / float(args.hz) if args.hz > 0 else 1.0

    pipe = rs_pipeline(args.width, args.height, fps=30)
    align = None

    print("[vlm] starting; press Ctrl+C to stop.")
    last_t = 0.0
    last_out = None
    try:
        while True:
            t0 = time.time()
            # Grab latest color frame
            fs = pipe.wait_for_frames()
            c = fs.get_color_frame()
            if not c:
                continue
            color_np = np.asanyarray(c.get_data())

            # Optional resize for bandwidth
            if color_np.shape[1] != args.width:
                color_np = cv2.resize(
                    color_np, (args.width, args.height), interpolation=cv2.INTER_AREA)

            # Downscale to model-friendly size for sending
            send_np = cv2.resize(
                color_np, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

            # Encode frame
            b64 = frame_to_jpeg_b64(send_np, quality=int(args.jpeg_quality))

            # Build and send request
            text = "Live frame. Describe and advise for safe movement in one or two sentences."
            payload = build_mm_chat_payload(
                model=args.model,
                system=args.system,
                text=text,
                b64_jpeg=b64,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            try:
                out = call_vlm(args.endpoint, payload, timeout=60)
            except Exception as e:
                print(f"[vlm] request failed: {e}")
                out = None

            ts = int(time.time() * 1000)
            if out:
                last_out = out
                print(f"[{ts}] VLM: {out}")

            # Optional preview window
            if args.show:
                vis = color_np.copy()
                overlay = vis.copy()
                if last_out:
                    lines = _wrap_text(last_out, width=70)
                    pad = 8
                    lh = 20
                    block_h = pad*2 + lh*len(lines)
                    h, w = vis.shape[:2]
                    # semi-transparent black box at bottom
                    cv2.rectangle(overlay, (0, h - block_h),
                                  (w, h), (0, 0, 0), -1)
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
                    y = h - block_h + pad + 14
                    for ln in lines:
                        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        y += lh
                cv2.imshow('RealSense VLM', vis)
                # press q to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # pace
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
            if args.once and out is not None:
                break
    except KeyboardInterrupt:
        pass
    finally:
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
