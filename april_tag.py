#!/usr/bin/env python3
import sys, math, csv, signal, threading, os, time
import numpy as np
import cv2

# --- MAXIMUM CPU OPTIMIZATION: Restrict OpenCV Threading ---
cv2.setNumThreads(1)

# ROS & Messages
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QKeySequence

# --- CONSTANTS & GLOBAL STATE ---
DISPLAY_WIDTH = 480         # Configurable camera display width (e.g., 640, 480, 320)
DISPLAY_HEIGHT = 320        # Configurable camera display height (e.g., 480, 360, 240)

MARKER_SIZE = 0.145         
l = MARKER_SIZE / 2.0       
obj_pts = np.float32([[-l, l, 0], [l, l, 0], [l, -l, 0], [-l, -l, 0]])
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

try:
    adict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
except AttributeError:
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

params = cv2.aruco.DetectorParameters_create()
params.minMarkerPerimeterRate = 0.01
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE 

# Thread Lock
data_lock = threading.Lock()

# Global State Data
K = None
recorded_data_history = []
last_frame_time = 0.0  
current_fps = 0.0      

# BASELINE OFFSETS
BASELINES_FILE = "patrol_baselines.csv"
baselines = {}

def load_baselines():
    global baselines
    if os.path.exists(BASELINES_FILE):
        with open(BASELINES_FILE, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    if len(row) == 2:
                        baselines[int(row[0])] = (float(row[1]), 0.0, 0.50)
                    elif len(row) == 4:
                        baselines[int(row[0])] = (float(row[1]), float(row[2]), float(row[3]))
                except ValueError:
                    pass

def save_baselines():
    with open(BASELINES_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        for tag_id, (yaw, b_x, b_z) in baselines.items():
            writer.writerow([tag_id, yaw, b_x, b_z])

load_baselines()

# --- STABILIZATION STATE ---
locked_tag_id = None
locked_tvec = None
locked_rvec = None
EMA_ALPHA = 0.10  # Heavy smoothing to freeze live feed noise completely

# Final Output State
current_tag_id, current_display_dist, current_yaw = None, 0.0, 0.0
current_abs_yaw = 0.0
current_raw_cx = 0.0  
current_raw_cz = 0.0  
current_cx, current_cy, current_cz = 0.0, 0.0, 0.0
current_is_zeroed = False  
latest_processed_frame = None

bridge = CvBridge()

csv_file = open("patrol_accuracy_report.csv", "a", newline='')
csv_writer = csv.writer(csv_file)
if os.stat("patrol_accuracy_report.csv").st_size == 0:
    csv_writer.writerow(["Timestamp", "Tag_ID", "Distance(cm)", "X(cm)", "Y(cm)", "Heading(deg)"])

class CopyableTableWidget(QTableWidget):
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            copied_text = ""
            for r in range(self.rowCount()):
                row_data = []
                for c in range(self.columnCount()):
                    item = self.item(r, c)
                    if item and item.isSelected():
                        row_data.append(item.text())
                if row_data:
                    copied_text += "\t".join(row_data) + "\n"
            QApplication.clipboard().setText(copied_text)
        else:
            super().keyPressEvent(event)

def main_thread_update():
    global latest_processed_frame
    if rospy.is_shutdown(): return

    with data_lock:
        t_id_curr = current_tag_id
        dist_curr = current_display_dist
        yaw_curr = current_yaw
        cx, cy, cz = current_cx, current_cy, current_cz
        is_zeroed_curr = current_is_zeroed
        
        frame_to_show = latest_processed_frame
        latest_processed_frame = None  

    if frame_to_show is not None:
        frame_resized = cv2.resize(frame_to_show, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        video_label.setPixmap(QPixmap.fromImage(q_img))

    if t_id_curr is not None:
        zero_status_text = "<span style='color: #007BFF;'>[Zeroed]</span>" if is_zeroed_curr else ""
        metric_text = (
            f"<div style='line-height: 1.2;'>"
            f"<b>Tag ID:</b> {t_id_curr} {zero_status_text}<br>"
            f"<b>Distance:</b> {dist_curr:.3f} m<br>"
            f"<b>X:</b> {cx:.2f} m<br>"
            f"<b>Y:</b> {cz:.2f} m<br>"
            f"<b>Heading:</b> {yaw_curr:+.1f}°"
            f"</div>"
        )
        lbl_aruco_pose.setText(metric_text)
        btn_zero.setText("Update Tag Baseline" if is_zeroed_curr else "Set Tag Baseline")
        btn_remove_zero.setEnabled(is_zeroed_curr)
        btn_record.setEnabled(True)
    else:
        lbl_aruco_pose.setText("<b>Tag Searching...</b>")
        btn_zero.setText("Set Tag Baseline")
        btn_remove_zero.setEnabled(False)
        btn_record.setEnabled(False)

def set_zero_heading():
    global baselines
    with data_lock:
        if current_tag_id is None: return
        t_id = current_tag_id
        abs_yaw = current_abs_yaw
        raw_x = current_raw_cx
        raw_z = current_raw_cz  
        
    baselines[t_id] = (abs_yaw, raw_x, raw_z)
    save_baselines()
    rospy.loginfo(f"Baseline updated. Tag: {t_id}")

def remove_baseline():
    global baselines
    with data_lock:
        if current_tag_id is None: return
        t_id = current_tag_id

    if t_id in baselines:
        del baselines[t_id]
        save_baselines()

# --- INSTANT RECORDING FEATURE ---
def record_instant_metrics():
    with data_lock:
        if current_tag_id is None:
            rospy.logwarn("Cannot record metrics: No ArUco Tag in sight!")
            return
        t_id = current_tag_id
        dist_cm = current_display_dist * 100.0
        cx_cm = current_cx * 100.0
        cz_cm = current_cz * 100.0
        yaw_deg = current_yaw

    timestamp = rospy.get_time()
    row_data = [timestamp, t_id, f"{dist_cm:.2f}", f"{cx_cm:.2f}", f"{cz_cm:.2f}", f"{yaw_deg:.2f}"]
    
    recorded_data_history.append(row_data)
    csv_writer.writerow(row_data)
    csv_file.flush()
    
    QMetaObject.invokeMethod(window, "add_table_row", Qt.QueuedConnection, 
                             Q_ARG(str, str(t_id)), Q_ARG(str, f"{dist_cm:.1f}"), 
                             Q_ARG(str, f"{cx_cm:.1f}"), Q_ARG(str, f"{cz_cm:.1f}"), Q_ARG(str, f"{yaw_deg:+.1f}"))
    rospy.loginfo(f"Instant Data Saved for Tag {t_id}: Dist={dist_cm:.1f}cm, X={cx_cm:.1f}cm, Y={cz_cm:.1f}cm, Heading={yaw_deg:+.1f}°")

def delete_selected_row():
    global csv_file, csv_writer
    selected_items = data_table.selectedItems()
    if not selected_items: return
    
    rows = list(set([item.row() for item in selected_items]))
    rows.sort(reverse=True)
    for row in rows:
        data_table.removeRow(row)
        if row < len(recorded_data_history):
            recorded_data_history.pop(row)
            
    csv_file.close()
    csv_file = open("patrol_accuracy_report.csv", "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "Tag_ID", "Distance(cm)", "X(cm)", "Y(cm)", "Heading(deg)"])
    for r_data in recorded_data_history:
        csv_writer.writerow(r_data)
    csv_file.flush()

def export_table_to_csv():
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getSaveFileName(window, "Export Table", "aruco_metrics.csv", "CSV Files (*.csv);;All Files (*)", options=options)
    if file_path:
        if not file_path.endswith('.csv'): file_path += '.csv'
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [data_table.horizontalHeaderItem(i).text() for i in range(data_table.columnCount())]
                writer.writerow(headers)
                for row in range(data_table.rowCount()):
                    row_data = [data_table.item(row, col).text() if data_table.item(row, col) else "" for col in range(data_table.columnCount())]
                    writer.writerow(row_data)
        except Exception as e:
            rospy.logerr(f"Failed to export CSV: {e}")

def info_callback(msg):
    global K
    if K is None:
        with data_lock:
            K = np.array(msg.P).reshape(3, 4)[:3, :3]

def image_callback(msg):
    global current_tag_id, current_display_dist, current_yaw, current_abs_yaw
    global current_raw_cx, current_raw_cz, current_cx, current_cy, current_cz
    global current_is_zeroed, latest_processed_frame
    global locked_tag_id, locked_tvec, locked_rvec
    global last_frame_time, current_fps
    
    if rospy.is_shutdown(): return 
    
    # 5 FPS Throttle (0.2s)
    curr_time = time.time()
    dt = curr_time - last_frame_time
    if dt < 0.20: return  
    
    if last_frame_time > 0:
        raw_fps = 1.0 / dt
        current_fps = (0.8 * current_fps) + (0.2 * raw_fps) if current_fps > 0 else raw_fps
    last_frame_time = curr_time

    with data_lock: local_K = K
    if local_K is None: return

    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)

    c_tag_id, c_display_dist, c_yaw, c_abs_yaw = None, 0.0, 0.0, 0.0
    c_raw_cx, c_raw_cz = 0.0, 0.0
    c_cx, c_cy, c_cz = 0.0, 0.0, 0.0
    c_is_zeroed = False

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        target_idx = -1
        if locked_tag_id is not None:
            for i, t_id in enumerate(ids):
                if t_id[0] == locked_tag_id:
                    target_idx = i
                    break
                    
        if target_idx == -1:
            max_area = 0
            for i, c in enumerate(corners):
                area = cv2.contourArea(c[0])
                if area > max_area:
                    max_area = area
                    target_idx = i

        if target_idx != -1:
            t_id = ids[target_idx][0]
            c = corners[target_idx]

            # FIX 1: Extrinsic Guess Warm-Start to lock 3D orientation
            if locked_tag_id == t_id and locked_rvec is not None and locked_tvec is not None:
                rvec = locked_rvec.copy()
                tvec = locked_tvec.copy()
                cv2.solvePnP(obj_pts, c[0], local_K, dist_coeffs, rvec, tvec, 
                            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                _, rvec, tvec = cv2.solvePnP(obj_pts, c[0], local_K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

            R, _ = cv2.Rodrigues(rvec)
            abs_yaw_deg = math.degrees(math.atan2(R[0, 2], -R[2, 2]))

            c_is_zeroed = t_id in baselines
            b_yaw, b_x, b_z = baselines[t_id] if c_is_zeroed else (0.0, 0.0, 0.0)
            rel_yaw_deg = (abs_yaw_deg - b_yaw + 180.0) % 360.0 - 180.0

            # FIX 2: Outlier Gate - Reject sudden single-frame jumps larger than 15 degrees
            if locked_tag_id == t_id and current_yaw is not None:
                yaw_jump = abs((rel_yaw_deg - current_yaw + 180.0) % 360.0 - 180.0)
                if yaw_jump > 15.0:
                    return  # Skip glitch frame entirely

            if locked_tag_id is None or locked_tag_id != t_id:
                locked_tag_id = t_id
                locked_rvec = rvec
                locked_tvec = tvec
            else:
                # Strong EMA filter
                locked_rvec = (EMA_ALPHA * rvec) + ((1.0 - EMA_ALPHA) * locked_rvec)
                locked_tvec = (EMA_ALPHA * tvec) + ((1.0 - EMA_ALPHA) * locked_tvec)

            cv2.drawFrameAxes(frame, local_K, dist_coeffs, locked_rvec, locked_tvec, l)
            cx, cy, cz = locked_tvec.flatten()
            
            c_raw_cx, c_raw_cz = cx, cz
            c_cy = cy  

            R_smooth, _ = cv2.Rodrigues(locked_rvec)
            smooth_abs_yaw = math.degrees(math.atan2(R_smooth[0, 2], -R_smooth[2, 2]))

            # X OUTPUT REVERSED (Left = +, Right = -)
            c_cx = -(cx - b_x) if c_is_zeroed else -cx       
            c_cz = cz - b_z if c_is_zeroed else cz

            c_yaw = (smooth_abs_yaw - b_yaw + 180.0) % 360.0 - 180.0
            c_display_dist = max(0.0, cz) if abs(cx) > 0.2 else math.sqrt(cx**2 + cz**2)
            c_abs_yaw = smooth_abs_yaw  
            c_tag_id = locked_tag_id

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            z_ind = " [Zeroed]" if c_is_zeroed else ""
            cv2.putText(frame, f"ID:{locked_tag_id} | D:{c_display_dist:.3f}m | X:{c_cx:.2f} Y:{c_cz:.2f} | H:{c_yaw:+.1f}{z_ind}", 
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.27, (0, 255, 0), 3, cv2.LINE_AA)
              
    else:
        locked_tag_id, locked_tvec, locked_rvec = None, None, None

    cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 220, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.27, (0, 255, 255), 3, cv2.LINE_AA)
    
    with data_lock:
        current_tag_id, current_display_dist, current_yaw, current_abs_yaw = c_tag_id, c_display_dist, c_yaw, c_abs_yaw
        current_raw_cx, current_raw_cz = c_raw_cx, c_raw_cz
        current_cx, current_cy, current_cz = c_cx, c_cy, c_cz
        current_is_zeroed, latest_processed_frame = c_is_zeroed, frame

# --- APPLICATION SETUP ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArUco Accuracy Evaluator - Ultra Lightweight")
        self.setGeometry(100, 100, 1100, 500)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QVBoxLayout()
        global video_label, lbl_aruco_pose, btn_zero, btn_remove_zero, btn_record, data_table
        
        video_label = QLabel("Waiting for Camera...")
        video_label.setMinimumSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        video_label.setMaximumSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        video_label.setStyleSheet("background-color: black; color: white;")
        video_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(video_label)

        bottom_ui_layout = QHBoxLayout()
        metrics_group = QGroupBox("Live Metrics")
        metrics_layout = QVBoxLayout()
        lbl_aruco_pose = QLabel("<b>Tag Searching...</b>")
        lbl_aruco_pose.setStyleSheet("color: #008800; font-size: 14px;")
        metrics_layout.addWidget(lbl_aruco_pose)
        metrics_group.setLayout(metrics_layout)
        bottom_ui_layout.addWidget(metrics_group, stretch=3)

        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        btn_record = QPushButton("Record Current Metrics")
        btn_record.setStyleSheet("background-color: #E0A800; color: white; font-weight: bold; padding: 6px;")
        btn_record.clicked.connect(record_instant_metrics)
        control_layout.addWidget(btn_record)

        btn_zero = QPushButton("Set Baseline")
        btn_zero.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold; padding: 4px;")
        btn_zero.clicked.connect(set_zero_heading)
        control_layout.addWidget(btn_zero)

        btn_remove_zero = QPushButton("Remove Baseline")
        btn_remove_zero.setStyleSheet("background-color: #6C757D; color: white; font-weight: bold; padding: 4px;")
        btn_remove_zero.clicked.connect(remove_baseline)
        control_layout.addWidget(btn_remove_zero)

        control_group.setLayout(control_layout)
        bottom_ui_layout.addWidget(control_group, stretch=2)
        left_panel.addLayout(bottom_ui_layout)
        main_layout.addLayout(left_panel, stretch=1)

        right_panel = QVBoxLayout()
        table_group = QGroupBox("Recorded Metrics")
        table_layout = QVBoxLayout()
        data_table = CopyableTableWidget(0, 5)
        data_table.setHorizontalHeaderLabels(["Tag ID", "Distance (cm)", "X (cm)", "Y (cm)", "Heading (°)"])
        data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        data_table.setSelectionBehavior(QTableWidget.SelectRows)
        table_layout.addWidget(data_table)

        btn_delete = QPushButton("Delete Row")
        btn_delete.setStyleSheet("background-color: #DC3545; color: white; font-weight: bold; padding: 6px;")
        btn_delete.clicked.connect(delete_selected_row)
        table_layout.addWidget(btn_delete)

        btn_export = QPushButton("Export CSV")
        btn_export.setStyleSheet("background-color: #28A745; color: white; font-weight: bold; padding: 6px;")
        btn_export.clicked.connect(export_table_to_csv)
        table_layout.addWidget(btn_export)

        table_group.setLayout(table_layout)
        right_panel.addWidget(table_group)
        main_layout.addLayout(right_panel, stretch=1)

    @pyqtSlot(str, str, str, str, str)
    def add_table_row(self, t_id, dist, cx, cz, yaw):
        row_idx = data_table.rowCount()
        data_table.insertRow(row_idx)
        data_table.setItem(row_idx, 0, QTableWidgetItem(t_id))
        data_table.setItem(row_idx, 1, QTableWidgetItem(dist))
        data_table.setItem(row_idx, 2, QTableWidgetItem(cx))
        data_table.setItem(row_idx, 3, QTableWidgetItem(cz))
        data_table.setItem(row_idx, 4, QTableWidgetItem(yaw))
        data_table.scrollToBottom()


rospy.init_node('patrol_benchmark_gui', anonymous=True)
app = QApplication(sys.argv)
signal.signal(signal.SIGINT, lambda *args: app.quit())

window = MainWindow()

sub_info = rospy.Subscriber("/head_camera/camera_info", CameraInfo, info_callback)
sub_image = rospy.Subscriber("/head_camera/image_rect", Image, image_callback, queue_size=1)

gui_timer = QTimer()
gui_timer.timeout.connect(main_thread_update)
gui_timer.start(200)

def on_exit():
    gui_timer.stop()
    sub_info.unregister()
    sub_image.unregister()
    if not csv_file.closed:
        csv_file.close()

app.aboutToQuit.connect(on_exit)
window.show()
sys.exit(app.exec_())
