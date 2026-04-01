"""
test_vio_simple.py
OAK-D-S2 + SpectacularAI VIO test (USB2 workaround)

NOTE:
- Your laptop is negotiating USB "HIGH" (USB2). SpectacularAI normally blocks VIO.
- This workaround disables the USB speed check so the pipeline can start.
- Expect lower FPS, possible frame drops, and more tracking loss on USB2.

Run:
  python test_vio_simple.py
Stop:
  Ctrl+C
"""

import time
import depthai as dai
import spectacularAI
import spectacularAI.depthai as sad


def main():
    print("=" * 70)
    print("OAK-D S2 VIO Test (SpectacularAI) - USB2 Workaround")
    print("=" * 70)
    print("\nInitializing camera + VIO...")
    print("Press Ctrl+C to stop\n")

    # 1) Create DepthAI pipeline (do NOT manually add nodes)
    dai_pipeline = dai.Pipeline()

    # 2) SpectacularAI config
    config = sad.Configuration()

    # --- WORKAROUND: allow running even if USB speed is only HIGH (USB2) ---
    # This prevents SpectacularAI from throwing "LOW USB SPEED!" at startup.
    config.ensureSufficientUsbSpeed = False

    # Optional: reduce load a bit (keep defaults unless you still get drops)
    # Some SDK builds expose these, some don't. Safe-guard with hasattr checks.
    if hasattr(config, "stereoFps"):
        config.stereoFps = 30  # lower bandwidth vs 60
    if hasattr(config, "imuHz"):
        config.imuHz = 200     # lower IMU rate if available

    # 3) Create SpectacularAI wrapper pipeline using correct constructor
    vio_pipeline = sad.Pipeline(dai_pipeline, config)

    # 4) Connect device (same pipeline)
    print("Connecting to OAK-D S2...")
    device = dai.Device(dai_pipeline)

    # Show negotiated USB speed (will likely be HIGH on your laptop)
    try:
        print("USB speed:", device.getUsbSpeed())
    except Exception:
        pass

    print("✓ Camera connected!")

    # 5) Start VIO session
    print("\nStarting SpectacularAI VIO (USB2 allowed)...")
    vio_session = vio_pipeline.startSession(device)
    print("✓ VIO started\n")

    print("=" * 70)
    print("Tips:")
    print("- Move SLOWLY, keep motion smooth (USB2 causes drops).")
    print("- Use textured surfaces (posters, carpet, keyboard).")
    print("- Avoid blank walls and bright windows.")
    print("=" * 70)

    start_pose = None
    frame_count = 0
    tracking_count = 0

    while True:
        out = vio_session.waitForOutput()
        frame_count += 1

        status = out.status

        if status == spectacularAI.TrackingStatus.TRACKING:
            tracking_count += 1
            pos = out.pose.position
            x, y, z = pos.x, pos.y, pos.z

            if start_pose is None:
                start_pose = (x, y, z)
                print("\n✓ Origin set at first TRACKING pose (0,0,0)\n")

            rx = x - start_pose[0]
            ry = y - start_pose[1]
            rz = z - start_pose[2]
            dist = (rx * rx + ry * ry + rz * rz) ** 0.5
            tracking_percent = (tracking_count / frame_count) * 100.0

            print(
                f"\r[TRACKING] X={rx:+.3f}m  Y={ry:+.3f}m  Z={rz:+.3f}m"
                f"  | Dist={dist:.3f}m  | Track={tracking_percent:5.1f}%",
                end="",
                flush=True,
            )

        elif status == spectacularAI.TrackingStatus.LOST_TRACKING:
            print(
                "\r[LOST] Tracking lost — slow down / add texture (USB2 drops frames)          ",
                end="",
                flush=True,
            )

        else:
            print(
                "\r[INIT] Initializing — move slowly, slight sideways motion, textured surface         ",
                end="",
                flush=True,
            )

        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
