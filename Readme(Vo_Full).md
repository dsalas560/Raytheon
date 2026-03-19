# 🚁 Visual Odometry + Loop Closure + MAVLink External Vision (DepthAI + Raspberry Pi 5)

## 📌 Overview

This project implements a **real-time visual odometry (VO) and lightweight SLAM system** using an **OAK-D-S2 stereo camera** and a **Raspberry Pi 5**, designed for **GPS-denied drone navigation**.

The system:

* Estimates **camera motion in 3D (x, y, z)** using stereo vision
* Tracks visual features across frames using optical flow
* Uses depth to recover **metric scale (meters)**
* Solves camera motion using **Perspective-n-Point (PnP)**
* Applies **loop closure detection** to reduce drift
* Sends position to a Pixhawk flight controller using **MAVLink `VISION_POSITION_ESTIMATE`**

This allows integration with **ArduPilot EKF** for **external vision-based navigation**.

---

## 🧠 Core Idea (Theory)

The system replaces GPS by estimating position using vision:

1. Detect visual features in an image
2. Track those features across frames
3. Use stereo depth to convert pixels → 3D points
4. Solve camera motion using PnP
5. Accumulate motion over time → global position
6. Detect revisited areas (loop closure)
7. Correct accumulated drift
8. Send position to Pixhawk via MAVLink

---

## 🧱 System Architecture

```
OAK-D-S2 Camera
   ├── Left Mono (grayscale)
   ├── Right Mono (depth)
   ├── RGB (visualization + loop closure)
   └── IMU (gyro only, display/logging)

          ↓

Raspberry Pi 5 (Python)
   ├── Visual Odometry (LK + PnP)
   ├── Loop Closure (ORB)
   ├── Soft Drift Correction
   └── MAVLink (External Vision)

          ↓

Pixhawk 6X (ArduPilot)
   ├── EKF (sensor fusion)
   ├── IMU + Barometer
   └── External Vision Input

          ↓

Drone Navigation (No GPS)
```

---

## ⚙️ Key Components

### 1. Feature Detection & Tracking

* Detects strong visual points (corners)
* Tracks them across frames

### 2. Depth-Based 3D Reconstruction

* Uses stereo depth to convert 2D points → 3D coordinates (meters)

### 3. Motion Estimation (PnP)

* Solves camera movement between frames using 3D → 2D correspondences

### 4. Pose Accumulation

* Chains frame-to-frame motion into a continuous trajectory

### 5. Loop Closure

* Recognizes previously seen locations using ORB features
* Estimates drift and corrects it

### 6. MAVLink Integration

* Sends position to Pixhawk using:

```
VISION_POSITION_ESTIMATE
```

---

## 📦 Libraries Used

| Library                           | Purpose                               |
| --------------------------------- | ------------------------------------- |
| `depthai`                         | Camera pipeline, stereo depth, IMU    |
| `opencv-python (cv2)`             | Feature detection, tracking, PnP, ORB |
| `numpy`                           | Matrix math, transformations          |
| `pymavlink`                       | MAVLink communication with Pixhawk    |
| `time`, `math`, `csv`, `datetime` | Logging, timing, math utilities       |

---

## 🔍 Key Functions and Their Roles

---

### 📷 DepthAI (Camera & Sensors)

#### `dai.Pipeline()`

* Creates a processing pipeline on the OAK device

#### `dai.node.MonoCamera`

* Left + right grayscale cameras for stereo depth

#### `dai.node.StereoDepth`

* Computes depth map from stereo pair

#### `dai.node.ColorCamera`

* RGB stream used for:

  * visualization
  * loop closure

#### `dai.node.IMU`

* Provides gyroscope data (used for yaw display only)

---

### 🧮 OpenCV (Core Vision Algorithms)

#### `cv2.goodFeaturesToTrack()`

* Detects strong corner features
* These appear as **green dots** on screen
* Used for tracking

#### `cv2.calcOpticalFlowPyrLK()`

* Tracks features between frames using Lucas-Kanade optical flow

#### `cv2.solvePnPRansac()`

* Solves for camera motion using:

  * 3D points (from depth)
  * 2D projections (current frame)
* Returns rotation + translation

#### `cv2.Rodrigues()`

* Converts rotation vector → rotation matrix

#### `cv2.ORB_create()`

* Extracts descriptors for loop closure

#### `cv2.BFMatcher()`

* Matches ORB descriptors between frames

#### `cv2.cvtColor()`

* Converts RGB → grayscale

---

### 🧠 Visual Odometry Class (`VO_LK`)

#### `process(gray, depth)`

Core function that:

* Tracks features
* Builds 3D → 2D correspondences
* Runs PnP
* Updates pose

#### `update_imu()`

* Integrates gyro Z-axis
* Used for **display only (NOT fused into pose)**

#### `apply_soft_correction()`

* Applies gradual drift correction after loop detection

#### `pose()`

* Returns current position and yaw

---

### 🔁 Loop Closure (`LoopClosureORB`)

#### `add_keyframe()`

* Stores:

  * ORB descriptors
  * pose
  * frame ID

#### `check_loop()`

* Matches current frame against past keyframes
* If match is strong:

  * computes drift
  * returns correction data

---

### 📡 MAVLink Integration (`MavlinkVisionPublisher`)

#### `vision_position_estimate_send()`

(from `pymavlink`)

Sends:

```
position (x, y, z)
orientation (roll, pitch, yaw)
timestamp
```

#### `vo_pose_to_ned()`

* Converts VO coordinates → NED frame (required by Pixhawk)

#### `align_if_needed()`

* Aligns VO heading with Pixhawk heading at startup

---

## 🟢 What the Green Dots Represent

The green dots are:

* Features detected using `cv2.goodFeaturesToTrack()`
* Tracked across frames using optical flow

They represent:

* stable points in the environment
* used for motion estimation

More dots = better tracking
Fewer dots = degraded pose estimation

---

## 📉 Drift & Correction

### Drift Sources

* noisy depth
* imperfect tracking
* PnP estimation errors

### Correction Method

* detect revisited locations (loop closure)
* compute drift vector
* apply small correction:

```
new_pose = current_pose + alpha * drift
```

---

## 🧭 IMU Usage

### Camera IMU (OAK-D)

* Used for:

  * yaw visualization
  * logging
* NOT used for pose estimation

### Pixhawk IMU

* Used inside ArduPilot EKF
* Handles:

  * stabilization
  * orientation
  * velocity estimation

---

## ❗ Important Note (External Vision)

This code:
✔ Sends position to Pixhawk
❌ Does NOT force Pixhawk to use it

To use external vision, ArduPilot must be configured:

* EKF source set to **ExternalNav**
* GPS disabled or deprioritized

---

## ▶️ How to Run

```bash
MAVLINK_DIALECT=ardupilotmega python3 vo_full.py
```

---

## 📊 Output

### Live Window Displays:

* Position (x, y, z)
* Yaw (vision vs IMU)
* Feature points (green dots)
* Loop closure events
* Tracking status

### CSV Log:

* timestamp
* position
* tracking stats
* loop closure data
* correction info

---

## 🚀 Applications

* GPS-denied UAV navigation
* Indoor drone flight
* Autonomous robotics
* SLAM research
* Vision-based localization

---

## 📌 Summary

This system implements a **real-time, stereo-based visual odometry pipeline** with:

* feature tracking
* depth-based reconstruction
* PnP motion estimation
* loop closure correction
* MAVLink external vision integration

It provides a complete foundation for **autonomous navigation without GPS**.

---

