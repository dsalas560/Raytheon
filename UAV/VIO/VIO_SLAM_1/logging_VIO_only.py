"""
logging_vio_with_plot_and_drift.py

What it does:
- Runs SpectacularAI VIO (no SLAM/loop-closure)
- Creates a NEW CSV per run with a timestamp in the filename
- Logs: wall-clock timestamp, t_sec, status, x/y/z (meters, relative to start), yaw_deg
- Shows live RGB preview with overlay (XYZ, yaw, drift-from-start)
- Displays drift continuously (distance from start)
- At the end: prints final drift and saves a plot PNG + shows the plots

How to use:
- Run:  python logging_vio_with_plot_and_drift.py
- Enter a run label like: loop  or  snake
- Walk your path, press 'q' to quit
- Outputs:
    trajectory_<label>_<YYYY-mm-dd_HH-MM-SS>.csv
    plot_<label>_<YYYY-mm-dd_HH-MM-SS>.png
"""

import time
import csv
import math
from datetime import datetime

import cv2
import depthai as dai
import spectacularAI
import spectacularAI.depthai as sad

# Matplotlib is only used after the run (for plotting)
import matplotlib.pyplot as plt


METERS_TO_FEET = 3.28084


def quat_to_yaw_deg(q) -> float:
    """Convert quaternion (w,x,y,z) to yaw (deg)."""
    w, x, y, z = q.w, q.x, q.y, q.z
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw)


def dist3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def safe_run_label(raw: str) -> str:
    """
    Sanitize run label for filenames.
    Keep alnum, dash, underscore. Replace spaces with underscore.
    """
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


def plot_trajectory(csv_path: str, plot_path: str):
    """
    Read CSV and plot:
    - XY path
    - Z over time
    - Drift over time (distance from start)
    Saves plot to plot_path and also shows it.
    """
    t = []
    x = []
    y = []
    z = []
    drift = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # only plot TRACKING rows with position values
            if row["x_m"] == "" or row["y_m"] == "" or row["z_m"] == "":
                continue
            try:
                tt = float(row["t_sec"])
                xx = float(row["x_m"])
                yy = float(row["y_m"])
                zz = float(row["z_m"])
            except ValueError:
                continue

            t.append(tt)
            x.append(xx)
            y.append(yy)
            z.append(zz)
            drift.append(dist3(xx, yy, zz))

    if len(t) < 2:
        print("Not enough TRACKING data to plot (did tracking fail or was it too short?).")
        return

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("XY Trajectory (relative to start)")
    plt.axis("equal")
    plt.grid(True)

    plt.figure()
    plt.plot(t, z)
    plt.xlabel("Time (s)")
    plt.ylabel("Z (m)")
    plt.title("Z vs Time")
    plt.grid(True)

    plt.figure()
    plt.plot(t, drift)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance from start (m)")
    plt.title("Drift-from-start vs Time")
    plt.grid(True)

    # Save one combined PNG by saving the current figures individually is messy;
    # simplest: save the last figure, and also save all figures via a loop.
    # We'll save a single PNG for the last figure and rely on "show" for the rest.
    plt.savefig(plot_path, dpi=160)
    print(f"Saved plot snapshot: {plot_path}")

    plt.show()


def main():
    # ---------------------------
    # Run label + timestamped filenames
    # ---------------------------
    run_label = safe_run_label(input("Enter run label (e.g., loop, snake): "))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    csv_path = f"trajectory_{run_label}_{stamp}.csv"
    plot_path = f"plot_{run_label}_{stamp}.png"

    # ---------------------------
    # Build DepthAI pipeline (RGB preview)
    # ---------------------------
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 360)
    cam.setInterleaved(False)
    cam.setFps(30)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    # ---------------------------
    # SpectacularAI VIO setup
    # ---------------------------
    config = sad.Configuration()
    vio_pipeline = sad.Pipeline(pipeline, config)

    # ---------------------------
    # Logging / rate control
    # ---------------------------
    LOG_HZ = 30.0
    LOG_DT = 1.0 / LOG_HZ
    last_log_t = 0.0

    # Relative origin for this run
    start_pose = None

    # We’ll store last known drift for display even if tracking momentarily drops
    last_drift_m = None

    print(f"Starting run '{run_label}'")
    print(f"Logging to: {csv_path}")
    print("Press 'q' in the preview window to stop.\n")

    # ---------------------------
    # Open CSV and start device
    # ---------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wall_time_iso",
            "t_sec",
            "status",
            "x_m", "y_m", "z_m",
            "yaw_deg"
        ])
        f.flush()

        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue("rgb", maxSize=2, blocking=False)
            vio = vio_pipeline.startSession(device)

            t0 = time.time()

            while True:
                out = vio.waitForOutput()  # VIO output tick
                now = time.time()
                t_sec = now - t0
                wall_iso = datetime.now().isoformat(timespec="milliseconds")

                status = out.status

                # Extract pose if tracking
                if status == spectacularAI.TrackingStatus.TRACKING:
                    p = out.pose.position
                    q = out.pose.orientation

                    x, y, z = p.x, p.y, p.z
                    yaw_deg = quat_to_yaw_deg(q)

                    if start_pose is None:
                        start_pose = (x, y, z)

                    rx, ry, rz = x - start_pose[0], y - start_pose[1], z - start_pose[2]
                    last_drift_m = dist3(rx, ry, rz)
                else:
                    rx = ry = rz = None
                    yaw_deg = None

                # Log at fixed rate (so files are comparable)
                if (now - last_log_t) >= LOG_DT:
                    last_log_t = now
                    writer.writerow([
                        wall_iso,
                        f"{t_sec:.6f}",
                        str(status),
                        "" if rx is None else f"{rx:.6f}",
                        "" if ry is None else f"{ry:.6f}",
                        "" if rz is None else f"{rz:.6f}",
                        "" if yaw_deg is None else f"{yaw_deg:.3f}",
                    ])
                    # flush periodically
                    if int(t_sec) % 2 == 0:
                        f.flush()

                # RGB overlay
                in_rgb = q_rgb.tryGet()
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()

                    if status == spectacularAI.TrackingStatus.TRACKING and start_pose is not None:
                        rx_ft, ry_ft, rz_ft = rx * METERS_TO_FEET, ry * METERS_TO_FEET, rz * METERS_TO_FEET
                        drift_ft = (last_drift_m * METERS_TO_FEET) if last_drift_m is not None else float("nan")

                        cv2.putText(frame, f"X: {rx_ft:+.2f} ft", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Y: {ry_ft:+.2f} ft", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Z: {rz_ft:+.2f} ft", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Yaw: {yaw_deg:+.1f} deg", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # "Drift during flight" = distance from start (not loop-closure drift)
                        cv2.putText(frame, f"Dist from start: {last_drift_m:.3f} m ({drift_ft:.2f} ft)", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        msg = f"VIO: {status}"
                        if last_drift_m is not None:
                            msg += f" | last dist-from-start: {last_drift_m:.3f} m"
                        cv2.putText(frame, msg, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    cv2.imshow("RGB Preview (VIO + logging)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                time.sleep(0.001)

    cv2.destroyAllWindows()

    # ---------------------------
    # Compute final drift metric (from start)
    # ---------------------------
    # IMPORTANT: For "loop drift" you must return physically near the start.
    # This metric always reports final distance-from-start, which becomes "drift" only for loops.
    final_drift_m = None
    final_wall = None

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
        # find last row that has x/y/z
        for row in reversed(rows):
            if row["x_m"] != "" and row["y_m"] != "" and row["z_m"] != "":
                try:
                    xx = float(row["x_m"])
                    yy = float(row["y_m"])
                    zz = float(row["z_m"])
                    final_drift_m = dist3(xx, yy, zz)
                    final_wall = row["wall_time_iso"]
                except ValueError:
                    pass
                break

    if final_drift_m is None:
        print(f"\nSaved: {csv_path}")
        print("No final drift computed (no TRACKING samples with position).")
        return

    print(f"\nSaved: {csv_path}")
    print(f"Final distance-from-start: {final_drift_m:.4f} m ({final_drift_m * METERS_TO_FEET:.2f} ft)")
    if final_wall:
        print(f"Final sample time: {final_wall}")
    print("\nNote: If you physically returned to your start point (closed loop),")
    print("this final distance-from-start is a good estimate of accumulated drift.\n")

    # ---------------------------
    # Plot after run
    # ---------------------------
    plot_trajectory(csv_path, plot_path)


if __name__ == "__main__":
    main()
