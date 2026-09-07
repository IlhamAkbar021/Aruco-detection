<div align="center">

# 🎯 ArUco Accuracy & Registration Benchmark Tool

**A visual benchmarking and evaluation tool for analyzing approach trajectories, accuracy, and repeatability across different robotic navigation methods.**

[![ROS](https://img.shields.io/badge/ROS-Noetic%20%7C%20ROS%202-22314E?style=for-the-badge&logo=ros)](#)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](#)

<img width="777" height="413" alt="Screenshot from 2026-09-07 11-43-21" src="https://github.com/user-attachments/assets/3a51278e-884a-4304-85b3-b1511da49578" />

</div>

---

> **Overview for Product Performance Teams**  
> Designed as a lightweight, repeatable visual baseline tool to compare different robot navigation approaches. Rather than replacing high-precision reference sensors (such as lasers or motion capture), this tool provides a practical empirical reference to evaluate approach trajectories and verify that alternative navigation strategies yield consistent results within acceptable millimeter-level error margins.

## ⚙️ Core Tracking Engine

| Feature | Description |
| :--- | :--- |
| 🎯 **Sniper Mode (Dynamic ROI)** | Reduces compute load by searching for tags in a downscaled stream, then instantly locking onto a localized, full-resolution bounding box upon detection to ensure accurate corner extraction. |
| ⚡ **Dynamic Snap EMA Filter** | Applies an adaptive filter to establish a steady baseline. Smooths micro-jitter when still, but shifts to zero-lag mode during movements (>5cm) to prevent tracking latency during dynamic runs. |
| 📡 **Direct TF Broadcasting** | Converts detected tag poses and quaternions directly into the ROS TF tree, enabling real-time comparison between internal robot odometry and visual target estimation. |
| 🔒 **Dictionary Locking** | Locked specifically to the `DICT_APRILTAG_36h11` family to eliminate CPU spikes from multi-dictionary cycling during evaluation runs. |

## 🎛️ Evaluation & Testing Controls

*   **Live Telemetry Panel:** Real-time display of Tag ID, Distance (m), X/Y offsets (m), and Heading angle (degrees) relative to the camera lens.
*   **Record:** Logs instant pose metrics into the session table to sample positional accuracy at specific approach waypoints.
*   **Register (Zeroing):** Saves the tag's current pose as an absolute zero baseline. Essential for testing docking approaches, allowing teams to measure relative deviation from a target regardless of minor tag mounting angles.
*   **Unregister:** Resets the active zero baseline back to raw camera frame coordinates for uncompensated testing.
*   **Headless Mode:** Disables video stream rendering to save memory and CPU cycles, ensuring the benchmark tool itself does not interfere with the robot's onboard navigation performance.

## 📊 Data Management & Reporting

*   **Metrics Table:** Interactive data logging grid that converts distance and lateral offsets to centimeters for straightforward error analysis.
*   **Export Data:** Saves benchmark sessions directly to `.csv` format for statistical comparison, standard deviation calculations, and cross-approach error analysis.
*   **Data Scrubbing:** Allows quick deletion of invalid or interrupted trial runs to keep evaluation datasets clean.

---

<details>
<summary><b>📦 System Dependencies (Click to Expand)</b></summary>
<br>

Ensure the following packages are installed in your environment:
*   `ros-noetic-desktop` / ROS 2
*   `python3`
*   `opencv-python` (with ArUco module)
*   `cv_bridge`
*   `PyQt5`
*   `numpy`
*   `tf` / `tf.transformations`

</details>

---

## 🚀 Quick Start

Ensure ROS core is active and your camera topic is publishing, then run the script directly:

```bash
# Navigate to workspace source directory
cd /var/aeolus/data_ws/src

# Launch the tool directly via Python
python april_tag.py
