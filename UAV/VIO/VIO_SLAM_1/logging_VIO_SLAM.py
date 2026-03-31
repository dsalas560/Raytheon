"""
logging_VIO_SLAM.py

- VIO baseline or SLAM (loop-closure) mode via USE_SLAM flag
- Prompts for run label (loop/snake/etc.)
- Saves ONE timestamped CSV + 3 timestamped plot PNGs per run (no overwriting)
- Saves outputs next to this script (NOT VS Code's working directory)
- Displays "jump flagged" if pose delta between consecutive samples exceeds TELEPORT_METERS
- Logs mapping stats (keyframes/mappoints) if available in your SpectacularAI build

Output files (saved next to this script):
- trajectory_<label>_<VIO|SLAM>_<YYYY-mm-dd_HH-MM-SS>.csv
- plot_<label>_<VIO|SLAM>_<YYYY-mm-dd_HH-MM-SS>_1_xy.png
- plot_<label>_<VIO|SLAM>_<YYYY-mm-dd_HH-MM-SS>_2_z.png
- plot_<label>_<VIO|SLAM>_<YYYY-mm-dd_HH-MM-SS>_3_dist_from_start.png
"""

import os
import csv
import time
import math
from datetime import datetime

import cv2
import depthai as dai
import spectacularAI
import spectacularAI.depthai as sad


# =========================
# User settings
# =========================
USE_SLAM = False               # False = VIO-only baseline, True = VISLAM/loop-closure enabled
SHOW_PREVIEW = True           # Show RGB preview window
SAVE_PLOTS = True             # Save plot PNGs at end
TRY_SHOW_PLOTS = False        # If True, will attempt plt.show() after saving (optional)

PREVIEW_W, PREVIEW_H = 640, 360
FPS = 30

# "Teleport" heuristic: if pose jumps more than this between consecutive outputs, flag it.
# We DO NOT "apply" corrections here; we only observe and log.
TELEPORT_METERS = 0.75

METERS_TO_FEET = 3.28084


def safe_run_label(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "run"
    raw = raw.replace(" ", "_")
    keep = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
    out = "".join(keep)
    return out if out else "run"


def dist3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def quat_to_yaw_deg(q) -> float:
    """Convert quaternion (w,x,y,z) to yaw (deg)."""
    w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def plot_from_csv(csv_path: str, plot_base_path: str):
    """
    Reads CSV and saves 3 PNG plots:
    1) XY trajectory
    2) Z vs time
    3) distance-from-start vs time
    """
    import matplotlib.pyplot as plt

    t, x, y, z, d = [], [], [], [], []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("x_m", "") == "" or row.get("y_m", "") == "" or row.get("z_m", "") == "":
                continue
            try:
                tt = float(row["t_sec"])
                xx = float(row["x_m"])
                yy = float(row["y_m"])
                zz = float(row["z_m"])
            except Exception:
                continue
            t.append(tt)
            x.append(xx)
            y.append(yy)
            z.append(zz)
            d.append(dist3(xx, yy, zz))

    if len(t) < 2:
        print("[WARN] Not enough TRACKING samples to plot.")
        return

    # 1) XY
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("XY Trajectory (relative to start)")
    plt.axis("equal")
    plt.grid(True)
    out1 = plot_base_path + "_1_xy.png"
    plt.savefig(out1, dpi=160)
    print("Saved:", out1)

    # 2) Z vs time
    plt.figure()
    plt.plot(t, z)
    plt.xlabel("Time (s)")
    plt.ylabel("Z (m)")
    plt.title("Z vs Time")
    plt.grid(True)
    out2 = plot_base_path + "_2_z.png"
    plt.savefig(out2, dpi=160)
    print("Saved:", out2)

    # 3) Distance-from-start vs time
    plt.figure()
    plt.plot(t, d)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance from start (m)")
    plt.title("Distance-from-start vs Time")
    plt.grid(True)
    out3 = plot_base_path + "_3_dist_from_start.png"
    plt.savefig(out3, dpi=160)
    print("Saved:", out3)

    if TRY_SHOW_PLOTS:
        plt.show(block=True)
    else:
        plt.close("all")


def main():
    # Always save outputs next to this script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    run_label = safe_run_label(input("Enter run label (e.g., loop, snake): "))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode = "SLAM" if USE_SLAM else "VIO"

    csv_path = os.path.join(SCRIPT_DIR, f"trajectory_{run_label}_{mode}_{stamp}.csv")
    plot_base_path = os.path.join(SCRIPT_DIR, f"plot_{run_label}_{mode}_{stamp}")

    # ---------------------------
    # DepthAI pipeline (RGB preview)
    # ---------------------------
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(PREVIEW_W, PREVIEW_H)
    cam.setInterleaved(False)
    cam.setFps(FPS)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    # ---------------------------
    # SpectacularAI config
    # ---------------------------
    config = sad.Configuration()

    # Try to enable SLAM if requested. (If your installed version uses a different knob,
    # this may be ignored; we still log gracefully either way.)
    try:
        config.useSlam = bool(USE_SLAM)
    except Exception:
        pass

    # Optional SLAM/mapping stats (best-effort)
    slam_stats = {"keyframes": "", "mappoints": ""}

    def on_mapper_output(mapper_output):
        # Best-effort mapping stats; safe if API differs.
        try:
            m = mapper_output.map
            slam_stats["keyframes"] = str(len(m.keyFrames))
            slam_stats["mappoints"] = str(len(m.mapPoints))
        except Exception:
            pass

    # Build SpectacularAI pipeline wrapper
    if USE_SLAM:
        # Some builds accept mapper callback in constructor; if not, fall back cleanly.
        try:
            vio_pipeline = sad.Pipeline(pipeline, config, on_mapper_output)
        except TypeError:
            vio_pipeline = sad.Pipeline(pipeline, config)
    else:
        vio_pipeline = sad.Pipeline(pipeline, config)

    # ---------------------------
    # Run + log
    # ---------------------------
    LOG_HZ = 30.0
    LOG_DT = 1.0 / LOG_HZ
    last_log_t = 0.0

    start_pose = None
    prev_pose = None
    last_dist_from_start = None

    print(f"\nMode: {mode}")
    print(f"Saving CSV: {csv_path}")
    print("Press 'q' in the preview window to stop.\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wall_time_iso",
            "t_sec",
            "status",
            "x_m", "y_m", "z_m",
            "yaw_deg",
            "dpos_m",
            "teleport_flag",
            "slam_keyframes",
            "slam_mappoints",
        ])
        f.flush()

        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue("rgb", maxSize=2, blocking=False)
            vio = vio_pipeline.startSession(device)

            t0 = time.time()

            while True:
                out = vio.waitForOutput()
                now = time.time()
                t_sec = now - t0
                wall_iso = datetime.now().isoformat(timespec="milliseconds")

                status = out.status

                teleport_flag = 0
                dpos = ""

                # Compute relative pose if tracking
                if status == spectacularAI.TrackingStatus.TRACKING:
                    p = out.pose.position
                    q = out.pose.orientation

                    x, y, z = float(p.x), float(p.y), float(p.z)
                    yaw_deg = quat_to_yaw_deg(q)

                    if start_pose is None:
                        start_pose = (x, y, z)

                    rx, ry, rz = x - start_pose[0], y - start_pose[1], z - start_pose[2]
                    last_dist_from_start = dist3(rx, ry, rz)

                    if prev_pose is not None:
                        dx = x - prev_pose[0]
                        dy = y - prev_pose[1]
                        dz = z - prev_pose[2]
                        d = dist3(dx, dy, dz)
                        dpos = f"{d:.6f}"
                        if d > TELEPORT_METERS:
                            teleport_flag = 1

                    prev_pose = (x, y, z)

                else:
                    rx = ry = rz = ""
                    yaw_deg = ""
                    # keep prev_pose as-is; don't reset it on brief tracking loss

                # Log at fixed rate
                if (now - last_log_t) >= LOG_DT:
                    last_log_t = now
                    writer.writerow([
                        wall_iso,
                        f"{t_sec:.6f}",
                        str(status),
                        rx if rx == "" else f"{rx:.6f}",
                        ry if ry == "" else f"{ry:.6f}",
                        rz if rz == "" else f"{rz:.6f}",
                        yaw_deg if yaw_deg == "" else f"{yaw_deg:.3f}",
                        dpos,
                        teleport_flag,
                        slam_stats["keyframes"] if USE_SLAM else "",
                        slam_stats["mappoints"] if USE_SLAM else "",
                    ])
                    if int(t_sec) % 2 == 0:
                        f.flush()

                # Preview overlay
                if SHOW_PREVIEW:
                    in_rgb = q_rgb.tryGet()
                    if in_rgb is not None:
                        frame = in_rgb.getCvFrame()

                        cv2.putText(frame, f"Mode: {mode}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        if status == spectacularAI.TrackingStatus.TRACKING and start_pose is not None:
                            rx_ft = float(rx) * METERS_TO_FEET
                            ry_ft = float(ry) * METERS_TO_FEET
                            rz_ft = float(rz) * METERS_TO_FEET
                            dist_ft = (last_dist_from_start * METERS_TO_FEET) if last_dist_from_start is not None else 0.0

                            cv2.putText(frame, f"X: {rx_ft:+.2f} ft", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, f"Y: {ry_ft:+.2f} ft", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, f"Z: {rz_ft:+.2f} ft", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, f"Yaw: {float(yaw_deg):+.1f} deg", (10, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, f"Dist from start: {last_dist_from_start:.3f} m ({dist_ft:.2f} ft)",
                                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                            if USE_SLAM:
                                cv2.putText(frame, f"KFs: {slam_stats['keyframes']}  Pts: {slam_stats['mappoints']}",
                                            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                            if teleport_flag:
                                cv2.putText(frame, "JUMP FLAGGED", (10, 240),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                        else:
                            cv2.putText(frame, f"Tracking: {status}", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        cv2.imshow("VIO/SLAM Logging (observe-only)", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                time.sleep(0.001)

    cv2.destroyAllWindows()

    print(f"\nSaved CSV: {csv_path}")

    # Plot & save PNGs (same folder)
    if SAVE_PLOTS:
        try:
            plot_from_csv(csv_path, plot_base_path)
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()
