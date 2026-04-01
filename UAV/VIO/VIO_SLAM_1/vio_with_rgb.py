import time
import cv2
import depthai as dai
import spectacularAI
import spectacularAI.depthai as sad

METERS_TO_FEET = 3.28084

def main():
    # ---------------------------
    # 1) Build DepthAI pipeline
    # ---------------------------
    pipeline = dai.Pipeline()

    # RGB camera (preview is low bandwidth and fast)
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 360)
    cam.setInterleaved(False)
    cam.setFps(30)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    # ---------------------------
    # 2) SpectacularAI config + wrapper pipeline
    # ---------------------------
    config = sad.Configuration()
    # If you ever need to bypass USB check:
    # config.ensureSufficientUsbSpeed = False

    vio_pipeline = sad.Pipeline(pipeline, config)

    # ---------------------------
    # 3) Start device
    # ---------------------------
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=2, blocking=False)

        # Start VIO session on the same device
        vio = vio_pipeline.startSession(device)

        start_pose = None

        while True:
            # ---- VIO output ----
            out = vio.waitForOutput()

            if out.status == spectacularAI.TrackingStatus.TRACKING:
                p = out.pose.position
                x, y, z = p.x, p.y, p.z

                if start_pose is None:
                    start_pose = (x, y, z)

                rx, ry, rz = x - start_pose[0], y - start_pose[1], z - start_pose[2]
                rx_ft, ry_ft, rz_ft = rx * METERS_TO_FEET, ry * METERS_TO_FEET, rz * METERS_TO_FEET

            # ---- RGB frame (non-blocking) ----
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

                # Overlay pose text (if tracking)
                if start_pose is not None and out.status == spectacularAI.TrackingStatus.TRACKING:
                    cv2.putText(frame, f"X: {rx_ft:+.2f} ft", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(frame, f"Y: {ry_ft:+.2f} ft", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(frame, f"Z: {rz_ft:+.2f} ft", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                else:
                    cv2.putText(frame, f"VIO: {out.status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.imshow("RGB Preview (with VIO overlay)", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.001)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
