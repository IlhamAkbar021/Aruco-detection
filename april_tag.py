#!/usr/bin/env python3

import csv
import math
import os
import signal
import sys
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QPushButton, QShortcut,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)
import rospy
from sensor_msgs.msg import CameraInfo, Image
import tf
from tf.transformations import quaternion_from_matrix

# Optimize OpenCV threading to prevent high CPU switching overhead
cv2.setNumThreads(2)

# ====================================================
# CONFIGURATION
# ====================================================
CFG = {
    "image_topic": "/head_camera/image_rect",
    "info_topic": "/head_camera/camera_info",
    "camera_frame": "rgb_optical_frame",
    
    # Physical camera offset compensation (meters)
    "x_offset": -0.040,
    
    "display_width": 320,
    "display_height": 240,
    
    # ------------------------------------------------
    # PERFORMANCE & PRECISION SETTINGS
    # ------------------------------------------------
    "processing_scale": 0.5,            # Low-res scale used ONLY for searching to save CPU
    "detection_interval_gui": 0.05,     # ~20 FPS interval when GUI is rendering
    "detection_interval_headless": 0.0, # Real-time processing (0ms delay) in Headless mode
    "gui_interval": 0.05,               # GUI update frequency
    "marker_size": 0.145,               # MUST match physical tag size exactly for precise distance
    "ema_alpha": 0.08,                  # Heavy filter for stable positional locking when still
    
    "report_file": "aruco_accuracy_report.csv",
    "baseline_file": "aruco_registered_tags.csv",
}

# Set initial detection interval
CFG["detection_interval"] = CFG["detection_interval_gui"]

# ====================================================
# GLOBAL STATE & RESOURCES
# ====================================================
state = {
    "latest_input_frame": None,
    "latest_processed_frame": None,
    "processing_running": True,
    "headless_mode": False,
    "K": None,
    "dist_coeffs": np.zeros((5, 1), dtype=np.float32),
    
    "roi_box": None,
    
    "locked_tag_id": None,
    "locked_tvec": None,
    "locked_rvec": None,
    
    "c_tag": None, "c_dist": 0.0, "c_yaw": 0.0, "c_abs_yaw": 0.0,
    "c_cx": 0.0, "c_cz": 0.0, "c_raw_cx": 0.0, "c_raw_cz": 0.0,
    "is_zeroed": False,
    
    "last_detection_time": 0.0,
    "detection_fps": 0.0,
    "last_detection_timestamp": None,
    
    "baselines": {},
    "recorded_data_history": []
}

ui = {}
locks = {"data": threading.Lock(), "frame": threading.Lock()}
bridge = CvBridge()
tf_broadcaster = None

# 3D Marker Object Points
obj_pts = np.float32([
    [-CFG["marker_size"] / 2,  CFG["marker_size"] / 2, 0],
    [ CFG["marker_size"] / 2,  CFG["marker_size"] / 2, 0],
    [ CFG["marker_size"] / 2, -CFG["marker_size"] / 2, 0],
    [-CFG["marker_size"] / 2, -CFG["marker_size"] / 2, 0]
])

# Optimized Detector Parameters for max speed and precision
aruco_params = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters()
aruco_params.polygonalApproxAccuracyRate = 0.05
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# Tightened thresholding search windows to cut CPU load in half
aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 15
aruco_params.adaptiveThreshWinSizeStep = 10 

def get_aruco_dict(dict_enum):
    return cv2.aruco.Dictionary_get(dict_enum) if hasattr(cv2.aruco, "Dictionary_get") else cv2.aruco.getPredefinedDictionary(dict_enum)

# Locked to single tag family to prevent dictionary-cycling lag
adict = get_aruco_dict(cv2.aruco.DICT_APRILTAG_36h11)

# ====================================================
# UTILITY FUNCTIONS
# ====================================================
def init_storage_files():
    if not os.path.exists(CFG["report_file"]):
        try:
            with open(CFG["report_file"], "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Tag_ID", "Distance(cm)", "X(cm)", "Y(cm)", "Heading(deg)"])
        except Exception: pass

    if os.path.exists(CFG["baseline_file"]):
        try:
            with open(CFG["baseline_file"], "r") as f:
                for r in csv.reader(f):
                    if not r: continue
                    t_id = int(r[0])
                    state["baselines"][t_id] = (float(r[1]), float(r[2]), float(r[3])) if len(r) >= 4 else (float(r[1]), 0.0, 0.50)
        except Exception: pass

def save_baselines():
    try:
        with open(CFG["baseline_file"], "w", newline="") as f:
            writer = csv.writer(f)
            for k, v in state["baselines"].items():
                writer.writerow([k, v[0], v[1], v[2]])
    except Exception: pass

def set_base():
    with locks["data"]:
        if state["c_tag"] is not None:
            state["baselines"][state["c_tag"]] = (state["c_abs_yaw"], state["c_raw_cx"], state["c_raw_cz"])
            state["c_yaw"] = state["c_cx"] = state["c_cz"] = 0.0
            state["is_zeroed"] = True
    save_baselines()

def clear_base():
    with locks["data"]:
        if state["c_tag"] in state["baselines"]:
            del state["baselines"][state["c_tag"]]
    save_baselines()

def toggle_headless():
    state["headless_mode"] = not state["headless_mode"]
    if state["headless_mode"]:
        CFG["detection_interval"] = CFG["detection_interval_headless"]
        # Keeping SUBPIX enabled in headless mode to ensure distance precision matches streaming
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  
        ui["btn_headless"].setText("Show Stream")
        ui["btn_headless"].setStyleSheet("background-color: #C53030; color: white; font-weight: bold; border-radius: 4px;")
        ui["video_label"].setText("HEADLESS MODE ACTIVE")
    else:
        CFG["detection_interval"] = CFG["detection_interval_gui"]
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        ui["btn_headless"].setText("Headless")
        ui["btn_headless"].setStyleSheet("background-color: #2D3748; color: white; font-weight: bold; border-radius: 4px;")

def record_metrics():
    with locks["data"]:
        if state["c_tag"] is None: return
        row = [
            rospy.get_time(), state["c_tag"],
            f"{state['c_dist'] * 100:.2f}", f"{state['c_cx'] * 100:.2f}",
            f"{state['c_cz'] * 100:.2f}", f"{state['c_yaw']:.2f}"
        ]
    state["recorded_data_history"].append(row)
    try:
        with open(CFG["report_file"], "a", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception: pass
    
    r_idx = ui["table"].rowCount()
    ui["table"].insertRow(r_idx)
    for i, val in enumerate(row[1:]):
        ui["table"].setItem(r_idx, i, QTableWidgetItem(str(val)))
    ui["table"].scrollToBottom()

def delete_row():
    rows = sorted(list(set(item.row() for item in ui["table"].selectedItems())), reverse=True)
    for r in rows:
        ui["table"].removeRow(r)
        if r < len(state["recorded_data_history"]):
            state["recorded_data_history"].pop(r)

def export_csv():
    path, _ = QFileDialog.getSaveFileName(None, "Export Table", "aruco_metrics.csv", "CSV Files (*.csv)")
    if not path: return
    if not path.endswith(".csv"): path += ".csv"
    try:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ui["table"].horizontalHeaderItem(i).text() for i in range(ui["table"].columnCount())])
            for r in range(ui["table"].rowCount()):
                writer.writerow([ui["table"].item(r, c).text() if ui["table"].item(r, c) else "" for c in range(ui["table"].columnCount())])
    except Exception: pass

def copy_table_selection():
    if "table" not in ui: return
    clipboard_str = "\n".join([
        "\t".join([ui["table"].item(r, c).text() for c in range(ui["table"].columnCount()) if ui["table"].item(r, c) and ui["table"].item(r, c).isSelected()])
        for r in range(ui["table"].rowCount())
    ])
    QApplication.clipboard().setText(clipboard_str)

# ====================================================
# ROS & PROCESSING FUNCTIONS
# ====================================================
def info_cb(msg):
    if state["K"] is None:
        state["K"] = np.array(msg.P, dtype=np.float32).reshape(3, 4)[:3, :3]

def img_cb(msg):
    try:
        frame_gray = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
    except Exception: return
    with locks["frame"]:
        state["latest_input_frame"] = frame_gray

def process_frame(gray_frame):
    global adict
    if state["K"] is None or gray_frame is None: return None

    h_img, w_img = gray_frame.shape[:2]
    proc_scale = CFG["processing_scale"]
    offset_x, offset_y = 0, 0
    ran_full_res = False

    if state["roi_box"] is not None:
        # SNIPER MODE: Tag is locked. Use full resolution image for precise tracking.
        x1, y1, x2, y2 = state["roi_box"]
        search_frame = gray_frame[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
        ran_full_res = True
        corners, ids, _ = cv2.aruco.detectMarkers(search_frame, adict, parameters=aruco_params)
    else:
        # SEARCH MODE: Tag is lost. Downscale whole image to search quickly.
        if proc_scale < 1.0:
            search_frame = cv2.resize(gray_frame, (0, 0), fx=proc_scale, fy=proc_scale, interpolation=cv2.INTER_NEAREST)
        else:
            search_frame = gray_frame
            ran_full_res = True
            
        corners, ids, _ = cv2.aruco.detectMarkers(search_frame, adict, parameters=aruco_params)

    t_tag, t_dist, t_yaw, t_abs, t_cx, t_cz = None, 0.0, 0.0, 0.0, 0.0, 0.0
    z_flag, raw_cx, raw_cz = False, 0.0, 0.0
    detected_corners = None

    if ids is not None and len(corners) > 0:
        idx = max(range(len(corners)), key=lambda i: cv2.contourArea(corners[i][0]))
        t_id = int(ids[idx][0])

        selected_corners = corners[idx][0].copy()

        # Re-scale corners only if we found the tag in the downscaled search mode
        if not ran_full_res and proc_scale < 1.0:
            selected_corners /= proc_scale

        selected_corners[:, 0] += offset_x
        selected_corners[:, 1] += offset_y
        detected_corners = selected_corners.astype(np.int32)

        # Update ROI box for the next frame using full-resolution coordinates
        pad = 50
        min_x = max(0, int(np.min(selected_corners[:, 0])) - pad)
        max_x = min(w_img, int(np.max(selected_corners[:, 0])) + pad)
        min_y = max(0, int(np.min(selected_corners[:, 1])) - pad)
        max_y = min(h_img, int(np.max(selected_corners[:, 1])) + pad)
        state["roi_box"] = (min_x, min_y, max_x, max_y)

        # Solve pose using original full camera intrinsic matrix K
        success, rvec, tvec = cv2.solvePnP(obj_pts, selected_corners, state["K"], state["dist_coeffs"], flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if success:
            if state["locked_tag_id"] != t_id or state["locked_tvec"] is None:
                state["locked_tag_id"] = t_id
                state["locked_rvec"] = rvec.copy()
                state["locked_tvec"] = tvec.copy()
            else:
                # Menghitung seberapa jauh tag berpindah
                jump_distance = math.sqrt(
                    (state["locked_tvec"][0][0] - tvec[0][0])**2 +
                    (state["locked_tvec"][1][0] - tvec[1][0])**2 +
                    (state["locked_tvec"][2][0] - tvec[2][0])**2
                )
                
                # LOMPATAN BESAR (> 5cm): Pindah instan (Snap 100%)
                if jump_distance > 0.05:
                    state["locked_tvec"] = tvec.copy()
                    state["locked_rvec"] = rvec.copy()
                else:
                    # LOMPATAN KECIL (< 5cm): Dynamic Alpha (Mencegah lag lambat)
                    dynamic_alpha = min(1.0, jump_distance * 20.0) 
                    alpha = max(CFG["ema_alpha"], dynamic_alpha)
                    
                    state["locked_tvec"] = alpha * tvec + (1.0 - alpha) * state["locked_tvec"]
                    state["locked_rvec"] = alpha * rvec + (1.0 - alpha) * state["locked_rvec"]

            rmat, _ = cv2.Rodrigues(state["locked_rvec"])
            t_mat = np.eye(4)
            t_mat[:3, :3] = rmat
            quat = quaternion_from_matrix(t_mat)
            
            tx = float(state["locked_tvec"][0][0])
            ty = float(state["locked_tvec"][1][0])
            tz = float(state["locked_tvec"][2][0])

            tx = tx + CFG["x_offset"]
            
            if tf_broadcaster:
                tf_broadcaster.sendTransform((tx, ty, tz), quat, rospy.Time.now(), f"aruco_tag_{t_id}", CFG["camera_frame"])

            cx, cz = tx, tz
            raw_cx, raw_cz = cx, cz
            
            # Yaw (Bearing) diambil dari CX dan CZ
            # NOTE: Jika arah kemudi robot Kiri/Kanan terbalik, ubah menjadi raw_yaw = -math.degrees(...)
            raw_yaw = math.degrees(math.atan2(cx, cz))

            if t_id in state["baselines"]:
                z_flag = True
                b_yaw, b_x, b_z = state["baselines"][t_id]
            else:
                b_yaw, b_x, b_z = 0.0, 0.0, 0.0

            t_abs = raw_yaw
            t_cx = -(cx - b_x) if z_flag else -cx
            t_cz = -(cz - b_z) if z_flag else cz

            # Perhitungan Yaw Final secara instan tanpa delay dari buffer list
            t_yaw = (t_abs - b_yaw + 180.0) % 360.0 - 180.0
            
            t_dist = cz if abs(cx) <= 0.2 else math.sqrt(cx * cx + cz * cz)
            t_tag = state["locked_tag_id"]
    else:
        state["roi_box"] = None

    with locks["data"]:
        state["c_tag"], state["c_dist"], state["c_yaw"], state["c_abs_yaw"] = t_tag, t_dist, t_yaw, t_abs
        state["c_cx"], state["c_cz"], state["c_raw_cx"], state["c_raw_cz"] = t_cx, t_cz, raw_cx, raw_cz
        state["is_zeroed"] = z_flag

    return (gray_frame, detected_corners, state["roi_box"])

def processing_loop():
    while not rospy.is_shutdown() and state["processing_running"]:
        now = time.monotonic()
        if CFG["detection_interval"] > 0 and (now - state["last_detection_time"] < CFG["detection_interval"]):
            time.sleep(0.001)
            continue

        with locks["frame"]:
            frame = state["latest_input_frame"]
            state["latest_input_frame"] = None

        if frame is None:
            time.sleep(0.001)
            continue

        state["last_detection_time"] = now
        processed = process_frame(frame)

        d_now = time.monotonic()
        if state["last_detection_timestamp"] is not None:
            dt = d_now - state["last_detection_timestamp"]
            if dt > 0: state["detection_fps"] = 0.8 * state["detection_fps"] + 0.2 * (1.0 / dt)
        state["last_detection_timestamp"] = d_now

        # In headless mode, bypass storing processed rendering frames to save CPU memory
        if not state["headless_mode"]:
            with locks["frame"]:
                state["latest_processed_frame"] = processed

# ====================================================
# GUI FUNCTIONS
# ====================================================
def update_gui():
    if rospy.is_shutdown():
        shutdown()
        QApplication.quit()
        return

    with locks["data"]:
        tag, dist, cx, cz, yaw, z_flag = state["c_tag"], state["c_dist"], state["c_cx"], state["c_cz"], state["c_yaw"], state["is_zeroed"]

    # Only process and render video pixmap if NOT in headless mode
    if not state["headless_mode"]:
        with locks["frame"]:
            frame_data = state["latest_processed_frame"]
            state["latest_processed_frame"] = None

        if frame_data is not None:
            gray_frame, detected_corners, roi = frame_data
            small_display = cv2.resize(gray_frame, (CFG["display_width"], CFG["display_height"]), interpolation=cv2.INTER_NEAREST)
            display_bgr = cv2.cvtColor(small_display, cv2.COLOR_GRAY2BGR)

            sx, sy = CFG["display_width"] / float(gray_frame.shape[1]), CFG["display_height"] / float(gray_frame.shape[0])

            if detected_corners is not None:
                scaled = detected_corners.astype(np.float32)
                scaled[:, 0] *= sx
                scaled[:, 1] *= sy
                cv2.polylines(display_bgr, [scaled.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)

            if roi is not None:
                cv2.rectangle(display_bgr, (int(roi[0]*sx), int(roi[1]*sy)), (int(roi[2]*sx), int(roi[3]*sy)), (255, 100, 0), 1, cv2.LINE_AA)

            cv2.putText(display_bgr, f"Detect: {state['detection_fps']:.1f} FPS", (CFG["display_width"] - 130, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            ui["video_label"].setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)))

    if tag is not None:
        z_str = "<span style='color: #3182CE;'>(Registered)</span>" if z_flag else ""
        ui["lbl_pose"].setText(f"<div style='font-size: 13px; color: #2D3748;'><b>ID:</b> {tag} {z_str} &nbsp;|&nbsp; <b>Dist:</b> {dist:.3f} m<br><b>X:</b> {cx:.2f} m &nbsp;|&nbsp; <b>Y:</b> {cz:.2f} m &nbsp;|&nbsp; <b>Heading:</b> {yaw:.1f}°</div>")
    else:
        ui["lbl_pose"].setText("<div style='color:#A0AEC0; font-size: 13px;'>No Tag Detected</div>")

    ui["btn_record"].setEnabled(bool(tag))
    ui["btn_remove_zero"].setEnabled(z_flag)
    ui["btn_zero"].setText("Registered" if z_flag else "Register")

def build_ui(window):
    window.setWindowTitle("ArUco Accuracy & Registration Tool")
    window.setGeometry(100, 100, 850, 420)
    window.setStyleSheet("background-color: #F7FAFC; QPushButton { background-color: #2D3748; color: white; padding: 6px; border-radius: 4px; border: none; font-weight: bold; } QPushButton:hover { background-color: #4A5568; } QPushButton:disabled { background-color: #E2E8F0; color: #A0AEC0; }")

    main_widget = QWidget()
    window.setCentralWidget(main_widget)
    layout = QHBoxLayout(main_widget)
    
    left, right = QVBoxLayout(), QVBoxLayout()
    
    ui["video_label"] = QLabel("Waiting for Camera Frame...")
    ui["video_label"].setFixedSize(CFG["display_width"], CFG["display_height"])
    ui["video_label"].setStyleSheet("background:#1A202C; color:white; border-radius:4px;")
    ui["video_label"].setAlignment(Qt.AlignCenter)
    left.addWidget(ui["video_label"], alignment=Qt.AlignCenter)

    metrics = QFrame()
    metrics.setStyleSheet("background:white; border:1px solid #E2E8F0; border-radius: 4px;")
    metrics_layout = QVBoxLayout(metrics)
    ui["lbl_pose"] = QLabel()
    metrics_layout.addWidget(ui["lbl_pose"])
    left.addWidget(metrics)

    btn_grid = QGridLayout()
    ui["btn_record"] = QPushButton("Record")
    ui["btn_zero"] = QPushButton("Register")
    ui["btn_remove_zero"] = QPushButton("Unregister")
    ui["btn_headless"] = QPushButton("Headless")

    ui["btn_record"].clicked.connect(record_metrics)
    ui["btn_zero"].clicked.connect(set_base)
    ui["btn_remove_zero"].clicked.connect(clear_base)
    ui["btn_headless"].clicked.connect(toggle_headless)

    btn_grid.addWidget(ui["btn_record"], 0, 0)
    btn_grid.addWidget(ui["btn_zero"], 0, 1)
    btn_grid.addWidget(ui["btn_remove_zero"], 1, 0)
    btn_grid.addWidget(ui["btn_headless"], 1, 1)
    left.addLayout(btn_grid)

    ui["table"] = QTableWidget(0, 5)
    ui["table"].setHorizontalHeaderLabels(["ID", "Dist (cm)", "X (cm)", "Y (cm)", "Heading (°)"])
    ui["table"].horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    ui["table"].setSelectionBehavior(QTableWidget.SelectRows)
    ui["table"].setStyleSheet("background:white; border:1px solid #E2E8F0;")
    QShortcut(QKeySequence.Copy, ui["table"]).activated.connect(copy_table_selection)
    right.addWidget(ui["table"])

    t_btns = QHBoxLayout()
    btn_del = QPushButton("Delete")
    btn_exp = QPushButton("Export")
    btn_del.clicked.connect(delete_row)
    btn_exp.clicked.connect(export_csv)
    t_btns.addWidget(btn_del)
    t_btns.addWidget(btn_exp)
    right.addLayout(t_btns)

    layout.addLayout(left, stretch=0)
    layout.addLayout(right, stretch=1)

def shutdown():
    state["processing_running"] = False

# ====================================================
# MAIN ENTRY POINT
# ====================================================
def main():
    global tf_broadcaster
    rospy.init_node("aruco_accuracy_tool", disable_signals=True, anonymous=True)
    tf_broadcaster = tf.TransformBroadcaster()

    init_storage_files()

    app = QApplication(sys.argv)
    window = QMainWindow()
    build_ui(window)
    window.show()

    app.aboutToQuit.connect(shutdown)
    signal.signal(signal.SIGINT, lambda sig, frame: [shutdown(), app.quit()])

    rospy.Subscriber(CFG["info_topic"], CameraInfo, info_cb, queue_size=1)
    rospy.Subscriber(CFG["image_topic"], Image, img_cb, queue_size=1, buff_size=2**24)

    threading.Thread(target=processing_loop, daemon=True).start()

    gui_timer = QTimer()
    gui_timer.timeout.connect(update_gui)
    gui_timer.start(int(CFG["gui_interval"] * 1000))

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
