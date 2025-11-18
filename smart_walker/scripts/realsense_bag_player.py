# scripts/realsense_bag_player.py
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to RealSense .bag file")
    ap.add_argument("--realtime", type=str, default="false",
                    help="Playback in real-time? (true/false). Try both if one glitches.")
    ap.add_argument("--timeout_ms", type=int, default=1000,
                    help="wait_for_frames timeout")
    ap.add_argument("--loop", action="store_true",
                    help="Loop the bag when it ends")
    ap.add_argument("--save_one_pair", action="store_true",
                    help="Save one aligned color+depth pair to outputs/bag_test/")
    args = ap.parse_args()

    bag_path = Path(args.bag).resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    # --- Configure pipeline for FILE playback (do NOT force stream profiles) ---
    cfg = rs.config()
    # Let the bag dictate available streams; don't call cfg.enable_stream with fixed sizes here
    rs.config.enable_device_from_file(
        # we manage looping ourselves
        cfg, str(bag_path), repeat_playback=False)

    pipe = rs.pipeline()
    profile = pipe.start(cfg)

    pb = profile.get_device().as_playback()
    # Toggle real-time vs non-real-time. Some bags behave better in one or the other.
    realtime = str(args.realtime).lower() in ("1", "true", "yes", "y", "on")
    # real-time: may drop frames; non-real-time: no drops, app controls pacing
    pb.set_real_time(realtime)
    print(f"[bag_player] file={pb.file_name()} realtime={pb.is_real_time()}")

    # depth units -> metres
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[bag_player] depth_scale={depth_scale:.6f} m per unit")

    # Align depth -> color if both exist
    align = rs.align(rs.stream.color)

    saved = False
    print("Press 'q' to quit.")
    try:
        while True:
            # --- Early-exit if playback stopped or reached duration ---
            status = pb.current_status()  # pyrealsense2.playback_status
            # Optional defensive EOF check via position/duration:
            pos_ns = pb.get_position()            # nanoseconds
            dur_ns = pb.get_duration().total_seconds() * \
                1e9 if hasattr(pb.get_duration(), "total_seconds") else None
            if status.name.lower() == "stopped" or (dur_ns and pos_ns >= dur_ns):
                if args.loop:
                    pb.seek(dt.timedelta(0))
                    pb.resume()
                    saved = False  # allow saving a pair again after loop
                    continue
                else:
                    print("[bag_player] End of file.")
                    break

            # --- Try to get frames with a short timeout; on timeout, just continue loop ---
            try:
                frames = pipe.wait_for_frames(args.timeout_ms)
            except RuntimeError as e:
                # Typical at EOF or when a stream temporarily has no frames
                # Check status again; either loop, continue, or exit gracefully
                if pb.current_status().name.lower() == "stopped":
                    if args.loop:
                        pb.seek(dt.timedelta(0))
                        pb.resume()
                        continue
                    print("[bag_player] Reached EOF (timeout).")
                    break
                # Otherwise, transient gap: continue trying
                continue

            # Align (if color stream exists)
            try:
                aligned = align.process(frames)
                depth = aligned.get_depth_frame()
                color = aligned.get_color_frame()
            except Exception:
                # If no color in the bag, fall back to raw frames
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()

            if not depth and not color:
                continue

            # To numpy
            depth_np = np.asanyarray(depth.get_data()) if depth else None
            color_np = np.asanyarray(color.get_data()) if color else None

            # Visualise
            if depth_np is not None:
                depth_vis = cv2.convertScaleAbs(depth_np, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            else:
                # Make a blank panel if no depth
                h, w = color_np.shape[:2]
                depth_vis = np.zeros((h, w, 3), dtype=np.uint8)

            if color_np is None:
                # make a 3-channel view from depth only
                color_np = np.zeros_like(depth_vis)

            panel = np.hstack((color_np, depth_vis))

            # Centre pixel distance
            if depth_np is not None:
                h, w = depth_np.shape
                centre_units = depth_np[h//2, w//2]
                centre_m = centre_units * depth_scale if centre_units > 0 else 0.0
            else:
                centre_m = 0.0

            cv2.putText(panel, f"centre depth: {centre_m:.2f} m (real_time={pb.is_real_time()})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("RealSense bag (color | depth)", panel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if args.save_one_pair and not saved and (depth_np is not None) and (color_np is not None):
                outdir = Path("outputs/bag_test")
                outdir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(outdir / "color.png"), color_np)
                cv2.imwrite(str(outdir / "depth_z16.png"), depth_np)
                depth_mm = np.clip(depth_np.astype(
                    np.float32) * depth_scale * 1000.0, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(outdir / "depth_mm.png"), depth_mm)
                print(
                    f"[bag_player] Saved color.png, depth_z16.png, depth_mm.png to {outdir.resolve()}")
                saved = True

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
