# scripts/realsense_depth_test.py
from __future__ import annotations
import numpy as np
import cv2
import pyrealsense2 as rs


def main():
    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get depth scale (device-specific units → metres)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[smart_walker] depth_scale = {depth_scale:.6f} metres per unit")

    # Align depth to color
    align_to_color = rs.align(rs.stream.color)

    try:
        print("Press 'q' to quit.")
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align_to_color.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue

            # Convert to numpy
            depth_np = np.asanyarray(depth.get_data())
            color_np = np.asanyarray(color.get_data())

            # Visualise depth as colour map (for sanity)
            depth_vis = cv2.convertScaleAbs(
                depth_np, alpha=0.03)  # scale for display
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Show side-by-side
            both = np.hstack((color_np, depth_vis))

            # Read centre pixel distance in metres
            h, w = depth_np.shape
            centre_d_units = depth_np[h//2, w//2]
            centre_m = centre_d_units * depth_scale if centre_d_units > 0 else 0.0
            cv2.putText(
                both,
                f"centre depth: {centre_m:.2f} m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("RealSense smoke test (color | depth)", both)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
