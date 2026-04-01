"""
vo_depthai_lk_loop_softcorr_fast_RGB_MAIN.py  (DepthAI v2.x, NO SpectacularAI)

Based on your working vo_depthai_lk_loop_softcorr_fast.py, preserving:
- Run labels + CSV logging
- All overlay info
- Green tracking dots
- IMU yaw overlay
- Loop banner
- Soft correction
- Lag-optimized loop closure

Changes:
- Adds ColorCamera RGB stream
- Uses RGB frame for:
  (1) Loop closure ORB
  (2) Main visualization window (all info drawn on RGB)

Units:
- METERS (internally + CSV + overlay)

Controls:
- Press 'q' to quit
"""

import time
import csv
import math
from datetime import datetime

import cv2
import numpy as np
import depthai as dai


# -----------------------
# Settings
# -----------------------
FPS = 30
W, H = 640, 400
MONO_RES = dai.MonoCameraProperties.SensorResolution.THE_400_P

# Feature tracking (LK)
MAX_CORNERS = 300
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3

# VO thresholds
MIN_PNP_POINTS = 25
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 12.0
REDETECT_EVERY = 10

# Logging
LOG_DT = 1.0 / 30.0

# IMU
IMU_HZ = 200

# Loop closure (fast)
ENABLE_LOOP = True
KEYFRAME_INTERVAL = 30        # store keyframe ~1 Hz at 30 FPS
LOOP_CHECK_INTERVAL = 0.5     # check ~2 Hz
MIN_LOOP_SEPARATION = 10
MATCH_THRESHOLD = 45          # lowered slightly because we downscale ORB
MAX_KEYFRAMES = 300
MAX_MATCH_CANDIDATES = 80     # only check most recent N keyframes (big speedup)

# ORB speed knobs
ORB_NFEATURES = 400
ORB_SCALE = 0.5               # downscale before ORB (0.5 => 320x200)

# Soft drift correction
ENABLE_SOFT_CORRECTION = True
SOFT_CORR_ALPHA = 0.15        # 0.05–0.25 typical; higher = more aggressive
SOFT_CORR_COOLDOWN = 1.0      # seconds between applying corrections
MIN_DRIFT_TO_CORRECT_M = 0.20 # ignore tiny drift
MAX_CORR_STEP_M = 1.0         # clamp per correction event to avoid crazy jumps


# -----------------------
# Helpers
# -----------------------
def safe_run_label(raw: str) -> str:
    raw = (raw or "").strip().replace(" ", "_")
    return "".join(ch for ch in raw if ch.isalnum() or ch in "-_") or "run"


def wrap_deg180(a: float) -> float:
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a


def rotmat_to_yaw_deg(R: np.ndarray) -> float:
    return wrap_deg180(math.degrees(math.atan2(R[1, 0], R[0, 0])))


def clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-9 or n <= max_norm:
        return vec
    return vec * (max_norm / n)


# -----------------------
# DepthAI Pipeline (v2 API)
# -----------------------
def build_pipeline() -> dai.Pipeline:
    p = dai.Pipeline()

    monoL = p.create(dai.node.MonoCamera)
    monoR = p.create(dai.node.MonoCamera)
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoL.setResolution(MONO_RES)
    monoR.setResolution(MONO_RES)
    monoL.setFps(FPS)
    monoR.setFps(FPS)

    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)

    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, IMU_HZ)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # ✅ RGB camera (for loop closure + display)
    camRgb = p.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(W, H)     # IMPORTANT: match mono resolution for 1:1 dot overlay
    camRgb.setInterleaved(False)
    camRgb.setFps(FPS)

    xleft = p.create(dai.node.XLinkOut)
    xleft.setStreamName("left")
    stereo.rectifiedLeft.link(xleft.input)

    xdep = p.create(dai.node.XLinkOut)
    xdep.setStreamName("depth")
    stereo.depth.link(xdep.input)

    ximu = p.create(dai.node.XLinkOut)
    ximu.setStreamName("imu")
    imu.out.link(ximu.input)

    xrgb = p.create(dai.node.XLinkOut)
    xrgb.setStreamName("rgb")
    camRgb.preview.link(xrgb.input)

    return p


# -----------------------
# Loop Closure (fast + bounded) — now accepts RGB or Gray
# -----------------------
class LoopClosureORB:
    """
    Keyframes: ORB descriptors + pose at capture time.
    Fast optimizations:
    - ORB on downscaled frames
    - fewer features
    - only match last MAX_MATCH_CANDIDATES keyframes
    """

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.keyframes = []  # dict: {'id','des','pose','t_sec'}
        self.last_check_wall = 0.0
        self.last_detect = None  # (kf_id, score, drift_m)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(frame)
        if gray is None:
            return None
        if ORB_SCALE != 1.0:
            new_w = max(1, int(gray.shape[1] * ORB_SCALE))
            new_h = max(1, int(gray.shape[0] * ORB_SCALE))
            return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray

    def add_keyframe(self, frame: np.ndarray, pose_xyz: np.ndarray, frame_id: int, t_sec: float):
        small = self._prep(frame)
        if small is None:
            return
        _, des = self.orb.detectAndCompute(small, None)
        if des is None or len(des) == 0:
            return

        self.keyframes.append({
            "id": frame_id,
            "t_sec": t_sec,
            "des": des,
            "pose": pose_xyz.copy(),
        })

        if len(self.keyframes) > MAX_KEYFRAMES:
            self.keyframes.pop(0)

    def check_loop(self, frame: np.ndarray, current_pose_xyz: np.ndarray, frame_id: int):
        now = time.time()
        if now - self.last_check_wall < LOOP_CHECK_INTERVAL:
            return None
        self.last_check_wall = now

        if len(self.keyframes) < (MIN_LOOP_SEPARATION + 1):
            return None

        small = self._prep(frame)
        if small is None:
            return None
        _, des = self.orb.detectAndCompute(small, None)
        if des is None or len(des) == 0:
            return None

        candidates = self.keyframes[:-MIN_LOOP_SEPARATION]
        if len(candidates) > MAX_MATCH_CANDIDATES:
            candidates = candidates[-MAX_MATCH_CANDIDATES:]

        best = None
        best_score = 0

        for kf in candidates:
            if frame_id - kf["id"] < MIN_LOOP_SEPARATION:
                continue
            try:
                matches = self.bf.match(kf["des"], des)
                score = len(matches)
                if score > best_score:
                    best_score = score
                    best = kf
            except Exception:
                continue

        if best is None or best_score < MATCH_THRESHOLD:
            self.last_detect = None
            return None

        drift_vec = best["pose"] - current_pose_xyz
        drift_m = float(np.linalg.norm(drift_vec))

        self.last_detect = (best["id"], best_score, drift_m)
        return {
            "matched_kf_id": best["id"],
            "score": best_score,
            "drift_m": drift_m,
            "kf_time": best["t_sec"],
            "matched_pose": best["pose"].copy(),
        }


# -----------------------
# VO with LK + PnP + optional soft correction
# -----------------------
class VO_LK:
    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)
        self.dist = np.zeros((4, 1), dtype=np.float64)

        # world <- camera pose
        self.T_w_c = np.eye(4, dtype=np.float64)

        self.prev_gray = None
        self.prev_depth = None
        self.prev_pts = None

        self.status = "INIT"
        self.num_tracked = 0
        self.num_used_pnp = 0
        self.inliers = 0

        # IMU yaw (display only)
        self.yaw_imu_deg = 0.0
        self._last_imu_wall = None
        self.frame_idx = 0

        # Soft correction
        self._last_corr_wall = 0.0

    def update_imu(self, gyro_z_rad_s: float):
        now = time.time()
        if self._last_imu_wall is None:
            self._last_imu_wall = now
            return
        dt = now - self._last_imu_wall
        self._last_imu_wall = now
        if 0 < dt < 0.2:
            self.yaw_imu_deg = wrap_deg180(self.yaw_imu_deg + math.degrees(gyro_z_rad_s * dt))

    def _detect(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE,
            blockSize=7,
            useHarrisDetector=False,
        )

    def process(self, gray, depth_mm):
        self.frame_idx += 1

        if self.prev_gray is None or self.prev_depth is None or self.prev_pts is None:
            self.prev_gray = gray
            self.prev_depth = depth_mm
            self.prev_pts = self._detect(gray)
            self.status = "WARMUP"
            self.num_tracked = self.num_used_pnp = self.inliers = 0
            return

        if self.prev_pts is None or len(self.prev_pts) < MIN_PNP_POINTS:
            self.prev_gray = gray
            self.prev_depth = depth_mm
            self.prev_pts = self._detect(gray)
            self.status = "REDETECT"
            self.num_tracked = self.num_used_pnp = self.inliers = 0
            return

        next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or st is None:
            self.prev_gray = gray
            self.prev_depth = depth_mm
            self.prev_pts = self._detect(gray)
            self.status = "LK_FAIL"
            self.num_tracked = self.num_used_pnp = self.inliers = 0
            return

        st = st.reshape(-1)
        prev_good = self.prev_pts[st == 1].reshape(-1, 2)
        curr_good = next_pts[st == 1].reshape(-1, 2)
        self.num_tracked = len(prev_good)

        if self.num_tracked < MIN_PNP_POINTS:
            self.prev_gray = gray
            self.prev_depth = depth_mm
            self.prev_pts = self._detect(gray)
            self.status = f"LOW_TRACK({self.num_tracked})"
            self.num_used_pnp = self.inliers = 0
            return

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        obj_pts, img_pts = [], []
        for (u0, v0), (u1, v1) in zip(prev_good, curr_good):
            x0, y0 = int(round(u0)), int(round(v0))
            if not (0 <= x0 < W and 0 <= y0 < H):
                continue

            z_m = float(self.prev_depth[y0, x0]) / 1000.0
            if z_m < DEPTH_MIN_M or z_m > DEPTH_MAX_M:
                continue

            X = (u0 - cx) * z_m / fx
            Y = (v0 - cy) * z_m / fy
            obj_pts.append([X, Y, z_m])
            img_pts.append([u1, v1])

        self.num_used_pnp = len(obj_pts)
        if self.num_used_pnp < MIN_PNP_POINTS:
            self.prev_gray = gray
            self.prev_depth = depth_mm
            self.prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)
            self.status = f"DEPTH_FILTER({self.num_used_pnp})"
            self.inliers = 0
            return

        obj_pts = np.asarray(obj_pts, dtype=np.float64)
        img_pts = np.asarray(img_pts, dtype=np.float64)

        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=150
        )
        if not ok or inl is None or len(inl) < 12:
            self.prev_gray = gray
            self.prev_depth = depth_mm
            self.prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)
            self.status = "PNP_FAIL"
            self.inliers = 0
            return

        self.inliers = int(len(inl))

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        R_inv = R.T
        t_inv = -R_inv @ t

        T_prev_cur = np.eye(4, dtype=np.float64)
        T_prev_cur[:3, :3] = R_inv
        T_prev_cur[:3, 3:] = t_inv

        self.T_w_c = self.T_w_c @ T_prev_cur
        self.status = "TRACKING"

        if (self.frame_idx % REDETECT_EVERY) == 0:
            pts = self._detect(gray)
            self.prev_pts = pts if pts is not None and len(pts) >= MIN_PNP_POINTS else curr_good.reshape(-1, 1, 2).astype(np.float32)
        else:
            self.prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)

        self.prev_gray = gray
        self.prev_depth = depth_mm

    def apply_soft_correction(self, target_pose_xyz: np.ndarray) -> dict | None:
        if not ENABLE_SOFT_CORRECTION:
            return None

        now = time.time()
        if now - self._last_corr_wall < SOFT_CORR_COOLDOWN:
            return None

        cur = self.T_w_c[:3, 3].reshape(3)
        drift = target_pose_xyz.reshape(3) - cur
        drift_mag = float(np.linalg.norm(drift))

        if drift_mag < MIN_DRIFT_TO_CORRECT_M:
            return None

        step = clamp_norm(drift, MAX_CORR_STEP_M)
        corr = step * SOFT_CORR_ALPHA

        self.T_w_c[:3, 3] = (cur + corr).reshape(3)

        self._last_corr_wall = now
        return {
            "drift_m": drift_mag,
            "alpha": SOFT_CORR_ALPHA,
            "corr_step_m": float(np.linalg.norm(corr)),
        }

    def pose(self):
        p = self.T_w_c[:3, 3].copy()
        yaw_vis = rotmat_to_yaw_deg(self.T_w_c[:3, :3])
        return p, yaw_vis


# -----------------------
# Main
# -----------------------
def main():
    run_label = safe_run_label(input("Enter run label: "))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = f"vo_depthai_lk_loop_softcorr_fast_{run_label}_{stamp}.csv"

    print("\nDepthAI VO (StereoDepth + OpenCV LK) + Fast Loop + Soft Correction (RGB MAIN)")
    print("Units: METERS (internally + CSV + overlay).")
    print(f"CSV: {csv_path}\n")

    pipeline = build_pipeline()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "wall_time_iso", "t_sec", "status",
            "x_m", "y_m", "z_m",
            "yaw_vis_deg", "yaw_imu_deg",
            "tracked_pts", "pnp_pts", "inliers",
            "loop_detected", "loop_kf_id", "loop_score", "loop_drift_m",
            "softcorr_applied", "softcorr_step_m"
        ])
        f.flush()

        with dai.Device(pipeline) as device:
            calib = device.readCalibration()
            K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, W, H), dtype=np.float64)
            print("K (LEFT):\n", K, "\n")

            vo = VO_LK(K)
            loop = LoopClosureORB() if ENABLE_LOOP else None

            q_left = device.getOutputQueue("left", maxSize=4, blocking=True)
            q_dep = device.getOutputQueue("depth", maxSize=4, blocking=True)
            q_imu = device.getOutputQueue("imu", maxSize=50, blocking=False)
            q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=True)

            t0 = time.time()
            last_log = 0.0
            frame_id = 0

            banner_until = 0.0
            banner_text = ""

            while True:
                now = time.time()
                t_sec = now - t0
                wall_iso = datetime.now().isoformat(timespec="milliseconds")

                # IMU
                for msg in q_imu.tryGetAll():
                    for pkt in msg.packets:
                        vo.update_imu(pkt.gyroscope.z)

                # Frames (blocking)
                gray = q_left.get().getCvFrame()
                depth_mm = q_dep.get().getFrame()
                rgb = q_rgb.get().getCvFrame()

                # VO
                vo.process(gray, depth_mm)
                pos, yaw_vis = vo.pose()

                # Loop + soft correction (on RGB)
                loop_detected = 0
                loop_kf_id = ""
                loop_score = ""
                loop_drift_m = ""
                softcorr_applied = 0
                softcorr_step_m = ""

                if ENABLE_LOOP and loop is not None and vo.status == "TRACKING":
                    if (frame_id % KEYFRAME_INTERVAL) == 0:
                        loop.add_keyframe(rgb, pos, frame_id, t_sec)

                    info = loop.check_loop(rgb, pos, frame_id)
                    if info is not None:
                        loop_detected = 1
                        loop_kf_id = str(info["matched_kf_id"])
                        loop_score = str(info["score"])
                        loop_drift_m = f"{info['drift_m']:.3f}"

                        corr_info = vo.apply_soft_correction(info["matched_pose"])
                        if corr_info is not None:
                            softcorr_applied = 1
                            softcorr_step_m = f"{corr_info['corr_step_m']:.3f}"

                        banner_until = time.time() + 1.0
                        banner_text = f"LOOP! kf={info['matched_kf_id']} score={info['score']} drift={info['drift_m']:.2f}m"

                # Refresh pose after correction
                pos, yaw_vis = vo.pose()

                # Log
                if (now - last_log) >= LOG_DT:
                    last_log = now
                    w.writerow([
                        wall_iso,
                        f"{t_sec:.6f}",
                        vo.status,
                        f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                        f"{yaw_vis:.3f}",
                        f"{vo.yaw_imu_deg:.3f}",
                        vo.num_tracked,
                        vo.num_used_pnp,
                        vo.inliers,
                        loop_detected,
                        loop_kf_id,
                        loop_score,
                        loop_drift_m,
                        softcorr_applied,
                        softcorr_step_m
                    ])
                    if int(t_sec) % 2 == 0:
                        f.flush()

                # ✅ Visualize on RGB (full original overlay)
                vis = rgb.copy()

                if vo.prev_pts is not None:
                    for p in vo.prev_pts.reshape(-1, 2):
                        u, v = int(p[0]), int(p[1])
                        if 0 <= u < W and 0 <= v < H:
                            cv2.circle(vis, (u, v), 2, (0, 255, 0), -1)

                cv2.putText(vis, "VO: DepthAI Stereo + OpenCV LK (FAST) [RGB MAIN]", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis, f"Status: {vo.status}", (10, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis, f"Pos(m): x={pos[0]:+.2f} y={pos[1]:+.2f} z={pos[2]:+.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, f"Yaw(vis)={yaw_vis:+.1f}  Yaw(imu)={vo.yaw_imu_deg:+.1f}", (10, 94),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, f"tracked={vo.num_tracked} pnp={vo.num_used_pnp} inliers={vo.inliers}", (10, 118),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if ENABLE_LOOP and loop is not None:
                    cv2.putText(vis, f"keyframes={len(loop.keyframes)} cand<= {MAX_MATCH_CANDIDATES}", (10, 142),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    if time.time() < banner_until:
                        cv2.putText(vis, banner_text, (10, 172),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                        if softcorr_applied:
                            cv2.putText(vis, f"SoftCorr applied (alpha={SOFT_CORR_ALPHA}) step={softcorr_step_m}m",
                                        (10, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                cv2.imshow("VO + Loop + SoftCorr (FAST) - RGB MAIN", vis)

                frame_id += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
    print(f"\nSaved CSV: {csv_path}\n[DONE]")


if __name__ == "__main__":
    main()
