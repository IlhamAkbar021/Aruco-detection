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
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QShortcut,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import rospy
from sensor_msgs.msg import CameraInfo, Image


FLAT_BTN = """
QPushButton {
    background-color: #2D3748;
    color: white;
    font-weight: bold;
    padding: 6px 10px;
    border-radius: 4px;
    border: none;
}
QPushButton:hover {
    background-color: #4A5568;
}
QPushButton:pressed {
    background-color: #1A202C;
}
QPushButton:disabled {
    background-color: #E2E8F0;
    color: #A0AEC0;
}
"""

HEADLESS_BTN_STYLE = """
QPushButton {
    background-color: #C53030;
    color: white;
    font-weight: bold;
    padding: 6px 10px;
    border-radius: 4px;
    border: none;
}
QPushButton:hover {
    background-color: #E53E3E;
}
"""


class ArucoTagNode(QMainWindow):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # ROS NODE & PARAMETERS
        # ----------------------------------------------------
        rospy.init_node("patrol_lite_optimized", disable_signals=True, anonymous=True)

        self.image_topic = rospy.get_param("~image_topic", "/head_camera/image_rect")
        self.info_topic = rospy.get_param("~camera_info_topic", "/head_camera/camera_info")

        # Camera display resolution (320x240)
        self.display_width = rospy.get_param("~display_width", 320)
        self.display_height = rospy.get_param("~display_height", 240)

        self.ema_alpha = rospy.get_param("~ema_alpha", 0.05)
        self.detection_interval = rospy.get_param("~detection_interval", 0.10)  # Default 10 Hz
        self.gui_interval = rospy.get_param("~gui_interval", 0.10)              # Max 10 Hz
        self.max_yaw_jump = rospy.get_param("~max_yaw_jump", 35.0)

        self.marker_size = rospy.get_param("~marker_size", 0.145)
        self.axis_length = rospy.get_param("~axis_length", 0.0725)

        self.report_file = rospy.get_param("~report_file", "patrol_accuracy_report.csv")
        self.baseline_file = rospy.get_param("~baseline_file", "patrol_baselines.csv")

        # ----------------------------------------------------
        # THREADING & LOCKS
        # ----------------------------------------------------
        self.data_lock = threading.Lock()
        self.frame_lock = threading.Lock()

        self.latest_input_frame = None
        self.latest_processed_frame = None
        self.processing_running = True
        self.headless_mode = False

        # ----------------------------------------------------
        # CAMERA & PNP SETUP
        # ----------------------------------------------------
        self.K = None
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.bridge = CvBridge()

        # Physical marker point definitions
        self.obj_pts = np.float32([
            [-self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0]
        ])

        # ----------------------------------------------------
        # ARUCO DETECTOR INITIALIZATION
        # ----------------------------------------------------
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.params = cv2.aruco.DetectorParameters_create()
        else:
            self.params = cv2.aruco.DetectorParameters()

        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        if hasattr(cv2.aruco, "Dictionary_get"):
            self.adict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        else:
            self.adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

        # ----------------------------------------------------
        # STATE VARIABLES
        # ----------------------------------------------------
        self.locked_tag_id = None
        self.locked_tvec = None
        self.locked_rvec = None

        self.c_tag = None
        self.c_dist = 0.0
        self.c_yaw = 0.0
        self.c_abs_yaw = 0.0
        self.c_cx = 0.0
        self.c_cz = 0.0
        self.c_raw_cx = 0.0
        self.c_raw_cz = 0.0
        self.is_zeroed = False

        # 10-Frame Averaging Buffer
        self.yaw_buffer = []

        # Metrics
        self.last_detection_time = 0.0
        self.camera_fps = 0.0
        self.detection_fps = 0.0
        self.last_camera_time = None
        self.last_detection_timestamp = None

        # Storage
        self.baselines = {}
        self.recorded_data_history = []

        # ----------------------------------------------------
        # INITIALIZE DATA & UI
        # ----------------------------------------------------
        self.init_storage_files()
        self.init_ui()

        # ----------------------------------------------------
        # ROS SUBSCRIBERS
        # ----------------------------------------------------
        self.info_sub = rospy.Subscriber(self.info_topic, CameraInfo, self.info_cb, queue_size=1)
        self.img_sub = rospy.Subscriber(self.image_topic, Image, self.img_cb, queue_size=1, buff_size=2**24)

        # ----------------------------------------------------
        # WORKER THREAD & TIMERS
        # ----------------------------------------------------
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        self.gui_timer = QTimer()
        self.gui_timer.timeout.connect(self.update_gui)
        self.gui_timer.start(int(self.gui_interval * 1000))

    # ============================================================
    # HEADLESS MODE & DYNAMIC THROTTLING
    # ============================================================

    def toggle_headless(self):
        """Toggles headless mode, disabling rendering and throttling detection rate."""
        self.headless_mode = not self.headless_mode

        if self.headless_mode:
            self.detection_interval = 0.33  # Throttle detection to ~3 Hz
            self.btn_headless.setText("Show Stream")
            self.btn_headless.setStyleSheet(HEADLESS_BTN_STYLE)
            self.video_label.setText("HEADLESS MODE ACTIVE\n\n(Detection at 3 Hz)")
            rospy.loginfo("Headless mode enabled: Render disabled, detection at 3 Hz.")
        else:
            self.detection_interval = 0.10  # Restore detection to 10 Hz
            self.btn_headless.setText("Headless")
            self.btn_headless.setStyleSheet(FLAT_BTN)
            rospy.loginfo("Headless mode disabled: Stream rendering restored at 10 Hz.")

    # ============================================================
    # STORAGE INITIALIZATION & BASELINES
    # ============================================================

    def init_storage_files(self):
        if not os.path.exists(self.report_file):
            try:
                with open(self.report_file, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "Timestamp", "Tag_ID", "Distance(cm)", "X(cm)", "Y(cm)", "Heading(deg)"
                    ])
            except Exception as e:
                rospy.logerr(f"Failed to create report file: {e}")

        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, "r") as f:
                    for r in csv.reader(f):
                        if not r:
                            continue
                        tag_id = int(r[0])
                        if len(r) >= 4:
                            self.baselines[tag_id] = (float(r[1]), float(r[2]), float(r[3]))
                        else:
                            self.baselines[tag_id] = (float(r[1]), 0.0, 0.50)
            except Exception as e:
                rospy.logwarn(f"Failed to load baseline file: {e}")

    def save_baselines(self):
        try:
            with open(self.baseline_file, "w", newline="") as f:
                writer = csv.writer(f)
                for k, v in self.baselines.items():
                    writer.writerow([k, v[0], v[1], v[2]])
        except Exception as e:
            rospy.logerr(f"Failed to save baseline: {e}")

    def set_base(self):
        with self.data_lock:
            if self.c_tag is not None:
                self.baselines[self.c_tag] = (self.c_abs_yaw, self.c_raw_cx, self.c_raw_cz)
                # Force active metrics to zero immediately to eliminate UI lag
                self.c_yaw = 0.0
                self.c_cx = 0.0
                self.c_cz = 0.0
                self.is_zeroed = True
                self.yaw_buffer.clear() # Clear buffer so old heading doesn't skew average
        self.save_baselines()

    def clear_base(self):
        with self.data_lock:
            if self.c_tag in self.baselines:
                del self.baselines[self.c_tag]
        self.save_baselines()

    # ============================================================
    # UI SETUP & ACTIONS
    # ============================================================

    def init_ui(self):
        self.setWindowTitle("ArUco Recorder - Compact")
        self.setGeometry(100, 100, 850, 420)
        self.setStyleSheet("background-color: #F7FAFC;")

        main = QWidget()
        self.setCentralWidget(main)

        main_layout = QHBoxLayout(main)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left = QVBoxLayout()
        left.setSpacing(8)

        right = QVBoxLayout()
        right.setSpacing(8)

        # 1. Camera Frame (320x240 Resolution)
        self.video_label = QLabel("Waiting for Camera Frame...")
        self.video_label.setFixedSize(self.display_width, self.display_height)
        self.video_label.setStyleSheet("background:#1A202C; color:white; border-radius:4px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # 2. Lower Metrics Card
        metrics = QFrame()
        metrics.setStyleSheet("background:white; border-radius:4px; border:1px solid #E2E8F0; padding: 4px;")
        metrics_layout = QVBoxLayout(metrics)
        metrics_layout.setContentsMargins(6, 6, 6, 6)
        
        self.lbl_aruco_pose = QLabel()
        self.lbl_aruco_pose.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        metrics_layout.addWidget(self.lbl_aruco_pose)
        left.addWidget(metrics)

        # 3. Control Action Buttons (2x2 Grid)
        btn_grid = QGridLayout()
        btn_grid.setSpacing(6)

        self.btn_record = QPushButton("Record")
        self.btn_zero = QPushButton("Register")
        self.btn_remove_zero = QPushButton("Unregister")
        self.btn_headless = QPushButton("Headless")

        buttons = [
            (self.btn_record, self.record_metrics, 0, 0),
            (self.btn_zero, self.set_base, 0, 1),
            (self.btn_remove_zero, self.clear_base, 1, 0),
            (self.btn_headless, self.toggle_headless, 1, 1)
        ]

        for button, callback, row, col in buttons:
            button.setStyleSheet(FLAT_BTN)
            button.clicked.connect(callback)
            btn_grid.addWidget(button, row, col)

        left.addLayout(btn_grid)

        # 4. Right Side Data Table
        self.data_table = QTableWidget(0, 5)
        self.data_table.setHorizontalHeaderLabels(
            ["ID", "Dist (cm)", "X (cm)", "Y (cm)", "Heading (°)"]
        )
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.setStyleSheet("background:white; border:1px solid #E2E8F0;")

        copy_shortcut = QShortcut(QKeySequence.Copy, self.data_table)
        copy_shortcut.activated.connect(self.copy_table_selection)

        right.addWidget(self.data_table)

        table_buttons = QHBoxLayout()
        btn_del = QPushButton("Delete")
        btn_exp = QPushButton("Export")

        btn_del.setStyleSheet(FLAT_BTN)
        btn_exp.setStyleSheet(FLAT_BTN)

        btn_del.clicked.connect(self.delete_row)
        btn_exp.clicked.connect(self.export_csv)

        table_buttons.addWidget(btn_del)
        table_buttons.addWidget(btn_exp)
        right.addLayout(table_buttons)

        main_layout.addLayout(left, stretch=0)
        main_layout.addLayout(right, stretch=1)

    def add_table_row(self, t_id, dist, cx, cz, yaw):
        row = self.data_table.rowCount()
        self.data_table.insertRow(row)
        values = [str(t_id), str(dist), str(cx), str(cz), str(yaw)]
        for i, value in enumerate(values):
            self.data_table.setItem(row, i, QTableWidgetItem(value))
        self.data_table.scrollToBottom()

    def record_metrics(self):
        with self.data_lock:
            if self.c_tag is None:
                return
            row = [
                rospy.get_time(),
                self.c_tag,
                f"{self.c_dist * 100:.2f}",
                f"{self.c_cx * 100:.2f}",
                f"{self.c_cz * 100:.2f}",
                f"{self.c_yaw:.2f}"
            ]

        self.recorded_data_history.append(row)

        try:
            with open(self.report_file, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            rospy.logerr(f"Failed to write CSV: {e}")

        self.add_table_row(row[1], row[2], row[3], row[4], row[5])

    def delete_row(self):
        rows = sorted(
            list(set(item.row() for item in self.data_table.selectedItems())),
            reverse=True
        )

        for r in rows:
            self.data_table.removeRow(r)
            if r < len(self.recorded_data_history):
                self.recorded_data_history.pop(r)

        try:
            with open(self.report_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Tag_ID", "Distance(cm)", "X(cm)", "Y(cm)", "Heading(deg)"
                ])
                writer.writerows(self.recorded_data_history)
        except Exception as e:
            rospy.logerr(f"Failed to rewrite CSV: {e}")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Table", "aruco_metrics.csv", "CSV Files (*.csv)"
        )

        if not path:
            return
        if not path.endswith(".csv"):
            path += ".csv"

        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.data_table.horizontalHeaderItem(i).text()
                    for i in range(self.data_table.columnCount())
                ])
                for r in range(self.data_table.rowCount()):
                    writer.writerow([
                        self.data_table.item(r, c).text() if self.data_table.item(r, c) else ""
                        for c in range(self.data_table.columnCount())
                    ])
        except Exception as e:
            rospy.logerr(f"Failed to export CSV: {e}")

    def copy_table_selection(self):
        if self.data_table is None:
            return
        clipboard_str = "\n".join([
            "\t".join([
                self.data_table.item(r, c).text()
                for c in range(self.data_table.columnCount())
                if self.data_table.item(r, c) and self.data_table.item(r, c).isSelected()
            ])
            for r in range(self.data_table.rowCount())
        ])
        QApplication.clipboard().setText(clipboard_str)

    # ============================================================
    # ROS CALLBACKS
    # ============================================================

    def info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.P, dtype=np.float32).reshape(3, 4)[:3, :3]
            rospy.loginfo("Camera matrix initialized successfully.")
            self.info_sub.unregister()

    def img_cb(self, msg):
        now = time.monotonic()
        if self.last_camera_time is not None:
            dt = now - self.last_camera_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.camera_fps = 0.9 * self.camera_fps + 0.1 * instant_fps
        self.last_camera_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"CvBridge error: {e}")
            return

        with self.frame_lock:
            self.latest_input_frame = frame

    # ============================================================
    # OPTIMIZED VISION PROCESSING & COMPUTATION
    # ============================================================

    @staticmethod
    def angle_diff_deg(a, b):
        return (a - b + 180.0) % 360.0 - 180.0

    def process_frame(self, frame):
        if self.K is None:
            return frame if not self.headless_mode else None

        # Downscale detection frame by 50% to reduce pixel searches by 75%
        scale_factor = 0.5
        small_frame = cv2.resize(
            frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST
        )
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.adict, parameters=self.params)

        t_tag, t_dist, t_yaw, t_abs, t_cx, t_cz = None, 0.0, 0.0, 0.0, 0.0, 0.0
        z_flag = False
        raw_cx, raw_cz = 0.0, 0.0

        if ids is not None and len(corners) > 0:
            idx = max(range(len(corners)), key=lambda i: cv2.contourArea(corners[i][0]))
            t_id = int(ids[idx][0])

            # Rescale detected corners back to full resolution scale for accurate solvePnP
            selected_corners = corners[idx][0] / scale_factor

            if not self.headless_mode:
                cv2.polylines(
                    frame, [selected_corners.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA
                )

            # 1. Prevent "comeback/flipping" ambiguity by using the previous pose as a guess
            if self.locked_rvec is not None and self.locked_tvec is not None and self.locked_tag_id == t_id:
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_pts, selected_corners, self.K, self.dist_coeffs,
                    rvec=self.locked_rvec.copy(), tvec=self.locked_tvec.copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_pts, selected_corners, self.K, self.dist_coeffs, 
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

            if not success:
                return frame if not self.headless_mode else None

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Invert the Z-axis projection (-rotation_matrix[2, 2]) so facing straight = 0.0
            # Invert the X-axis projection (-rotation_matrix[0, 2]) so Right = Positive, Left = Negative
            raw_yaw = math.degrees(math.atan2(-rotation_matrix[0, 2], -rotation_matrix[2, 2]))
            
            # NOTE: If your physical camera setup is mounted upside down or reversed, 
            # and you still need to flip the direction, just add a minus sign like this:
            # raw_yaw = -math.degrees(math.atan2(rotation_matrix[0, 2], rotation_matrix[2, 2]))

            if t_id in self.baselines:
                z_flag = True
                b_yaw, b_x, b_z = self.baselines[t_id]
            else:
                b_yaw, b_x, b_z = 0.0, 0.0, 0.0

            if self.locked_tag_id != t_id:
                self.locked_tag_id = t_id
                self.locked_rvec = rvec.copy()
                self.locked_tvec = tvec.copy()
                t_abs = raw_yaw
                self.yaw_buffer.clear() # Reset rolling buffer on newly locked tag
            else:
                if abs(self.angle_diff_deg(raw_yaw, self.c_abs_yaw)) <= self.max_yaw_jump:
                    if self.locked_tvec is None:
                        self.locked_tvec = tvec.copy()
                    else:
                        self.locked_tvec = self.ema_alpha * tvec + (1.0 - self.ema_alpha) * self.locked_tvec

                    if self.locked_rvec is None:
                        self.locked_rvec = rvec.copy()
                    else:
                        self.locked_rvec = self.ema_alpha * rvec + (1.0 - self.ema_alpha) * self.locked_rvec

                    t_abs = self.c_abs_yaw + self.ema_alpha * self.angle_diff_deg(raw_yaw, self.c_abs_yaw)
                else:
                    t_abs = self.c_abs_yaw

            if not self.headless_mode and self.locked_rvec is not None and self.locked_tvec is not None:
                try:
                    cv2.drawFrameAxes(
                        frame, self.K, self.dist_coeffs, self.locked_rvec, self.locked_tvec, self.axis_length, 2
                    )
                except Exception:
                    pass

            if self.locked_tvec is None:
                return frame if not self.headless_mode else None

            cx = float(self.locked_tvec[0][0])
            cz = float(self.locked_tvec[2][0])
            raw_cx, raw_cz = cx, cz

            t_cx = -(cx - b_x) if z_flag else -cx
            t_cz = (cz - b_z) if z_flag else cz
            
            # Retrieve current un-averaged heading
            t_yaw_raw = self.angle_diff_deg(t_abs, b_yaw)

            # ==========================================
            # 10-FRAME AVERAGING FOR HEADING (t_yaw)
            # ==========================================
            self.yaw_buffer.append(t_yaw_raw)
            if len(self.yaw_buffer) > 10:
                self.yaw_buffer.pop(0) # Keep buffer strictly at 10 items
            
            # Override t_yaw with the averaged value
            t_yaw = sum(self.yaw_buffer) / len(self.yaw_buffer)
            # ==========================================

            t_dist = cz if abs(cx) <= 0.2 else math.sqrt(cx * cx + cz * cz)
            t_tag = self.locked_tag_id
        else:
            # Bypass clearing locked_tag_id to prevent EMA filter reset on 1-2 dropped frames
            pass

        # HUD Overlay (Skipped in Headless Mode)
        if not self.headless_mode:
            img_h, img_w = frame.shape[:2]
            scale = img_w / 320.0
            font_scale = 0.45 * scale
            thickness = max(1, int(1.5 * scale))

            if t_tag is not None:
                id_str = f"ID: {t_tag}"
                (id_w, id_h), _ = cv2.getTextSize(id_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                pad = int(4 * scale)
                cv2.rectangle(frame, (pad, pad), (pad * 2 + id_w, pad * 2 + id_h), (20, 20, 20), -1)
                cv2.putText(
                    frame, id_str, (pad + int(3 * scale), pad + id_h + int(1 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
                )

            det_str = f"Detect: {self.detection_fps:.1f} FPS"
            (det_w, det_h), _ = cv2.getTextSize(det_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x_pos = img_w - det_w - int(8 * scale)
            y_pos = int(18 * scale)

            cv2.putText(frame, det_str, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, det_str, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        with self.data_lock:
            # Only update GUI metrics if a valid tag was matched on this frame
            if t_tag is not None:
                self.c_tag = t_tag
                self.c_dist = t_dist
                self.c_yaw = t_yaw # This is now the 10-frame averaged value
                self.c_abs_yaw = t_abs
                self.c_cx = t_cx
                self.c_cz = t_cz
                self.c_raw_cx = raw_cx
                self.c_raw_cz = raw_cz
                self.is_zeroed = z_flag
            else:
                self.c_tag = None

        return frame if not self.headless_mode else None

    # ============================================================
    # LOOP & GUI REFRESH
    # ============================================================

    def processing_loop(self):
        rospy.loginfo("ArUco processing thread started.")

        while not rospy.is_shutdown() and self.processing_running:
            now = time.monotonic()

            if now - self.last_detection_time < self.detection_interval:
                time.sleep(0.005)
                continue

            with self.frame_lock:
                if self.latest_input_frame is None:
                    frame = None
                else:
                    frame = self.latest_input_frame
                    self.latest_input_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            self.last_detection_time = now
            processed = self.process_frame(frame)

            detection_now = time.monotonic()
            if self.last_detection_timestamp is not None:
                dt = detection_now - self.last_detection_timestamp
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self.detection_fps = 0.8 * self.detection_fps + 0.2 * instant_fps
            self.last_detection_timestamp = detection_now

            with self.frame_lock:
                self.latest_processed_frame = processed

        rospy.loginfo("ArUco processing thread stopped.")

    def update_gui(self):
        if rospy.is_shutdown():
            self.shutdown()
            QApplication.quit()
            return

        with self.frame_lock:
            frame = self.latest_processed_frame
            self.latest_processed_frame = None

        with self.data_lock:
            tag = self.c_tag
            dist = self.c_dist
            cx = self.c_cx
            cz = self.c_cz
            yaw = self.c_yaw
            z_flag = self.is_zeroed

        # Render video frame (320x240) only when NOT in headless mode
        if not self.headless_mode and frame is not None:
            display = cv2.resize(
                frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

        # Compact metric output format
        if tag is not None:
            z_str = "<span style='color: #3182CE;'>(Registered)</span>" if z_flag else ""
            self.lbl_aruco_pose.setText(
                f"""
                <div style='font-size: 13px; color: #2D3748; line-height: 1.3;'>
                <b>ID:</b> {tag} {z_str} &nbsp;|&nbsp; <b>Dist:</b> {dist:.3f} m<br>
                <b>X:</b> {cx:.2f} m &nbsp;|&nbsp; <b>Y:</b> {cz:.2f} m &nbsp;|&nbsp; <b>Heading:</b> {yaw:+.1f}°
                </div>
                """
            )
        else:
            self.lbl_aruco_pose.setText(
                "<div style='color:#A0AEC0; font-size: 13px;'>No Tag Detected</div>"
            )

        self.btn_record.setEnabled(bool(tag))
        self.btn_remove_zero.setEnabled(z_flag)
        self.btn_zero.setText("Registered" if z_flag else "Register")

    # ============================================================
    # SHUTDOWN CLEANUP
    # ============================================================

    def shutdown(self):
        self.processing_running = False
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.gui_timer.stop()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()


# ============================================================
# MAIN APPLICATION ENTRY
# ============================================================

def main():
    app = QApplication(sys.argv)
    node = ArucoTagNode()

    def sigint_handler(sig, frame):
        node.shutdown()
        QApplication.quit()

    signal.signal(signal.SIGINT, sigint_handler)

    node.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()