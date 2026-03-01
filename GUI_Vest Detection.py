import sys
import os
import json
import ssl

# ═══════════════════════════════════════════════════════════════════════════
# STARTUP DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

print("="*60)
print("Safety Vest Detection System - Smooth Tracking")
print("="*60)

print(f"\n✓ Python version: {sys.version}")

print("\nChecking dependencies...")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")
    print("  Run: pip install opencv-python")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                                 QFrame, QComboBox, QFileDialog, QMessageBox,
                                 QGraphicsDropShadowEffect, QSizePolicy,
                                 QCheckBox, QLineEdit)
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
    print(f"✓ PyQt6: OK")
except ImportError as e:
    print(f"✗ PyQt6: {e}")
    print("  Run: pip install PyQt6")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print(f"✓ Ultralytics: OK")
except ImportError as e:
    print(f"✗ Ultralytics: {e}")
    print("  Run: pip install ultralytics")
    input("Press Enter to exit...")
    sys.exit(1)



# MQTT (publish stats)
try:
    import paho.mqtt.client as mqtt
    print("û paho-mqtt: OK")
    MQTT_AVAILABLE = True
except ImportError as e:
    print(f"? paho-mqtt: {e}")
    print("  Run: pip install paho-mqtt")
    MQTT_AVAILABLE = False

# DeepSORT tracking disabled
DEEPSORT_AVAILABLE = False

# CPU Monitoring
try:
    import psutil
    print(f"✓ psutil: OK")
    PSUTIL_AVAILABLE = True
except ImportError:
    print("⚠ psutil not installed (CPU monitoring disabled)")
    print("  Run: pip install psutil")
    PSUTIL_AVAILABLE = False

# GPU Monitoring
GPU_AVAILABLE = False
GPU_LIBRARY = None
try:
    import GPUtil
    print(f"✓ GPUtil: OK")
    GPU_AVAILABLE = True
    GPU_LIBRARY = "gputil"
except ImportError:
    try:
        import pynvml
        pynvml.nvmlInit()
        print(f"✓ pynvml: OK")
        GPU_AVAILABLE = True
        GPU_LIBRARY = "pynvml"
    except Exception:
        # fallback: if CUDA is available via torch, use PyTorch APIs for memory stats
        try:
            if torch.cuda.is_available():
                print("✓ CUDA available but GPUtil/pynvml not installed — using PyTorch for GPU stats")
                GPU_AVAILABLE = True
                GPU_LIBRARY = "pytorch"
            else:
                print("⚠ No GPU monitoring library (GPU monitoring disabled)")
                print("  Run: pip install GPUtil  OR  pip install pynvml")
        except Exception:
            print("⚠ No GPU monitoring library (GPU monitoring disabled)")
            print("  Run: pip install GPUtil  OR  pip install pynvml")

import time
import threading
import queue
import urllib.request
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════
# GPU CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"✓ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Optimize PyTorch for inference where appropriate
    try:
        torch.backends.cudnn.benchmark = True  # Auto-tune for speed
        torch.backends.cudnn.enabled = True
    except Exception:
        pass
    try:
        torch.cuda.set_per_process_memory_fraction(0.90)  # Use up to 90% of GPU memory
    except Exception:
        pass
    torch.cuda.empty_cache()
else:
    print("⚠ CUDA not available - using CPU (slower inference)")

print("\n✓ All core dependencies loaded!")
print("="*60)

# MQTT CONFIG
MQTT_HOST = "a3eb353511ab48eda8f1365c60928f57.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "lequangvan2003"
MQTT_PASSWORD = "Cockeo20062003%"
MQTT_TOPIC = "safetyvest/status"
MQTT_PUBLISH_INTERVAL_SEC = 1.0
DASHBOARD_UPDATE_URL = "http://127.0.0.1:5000/api/update"


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM MONITOR
# ═══════════════════════════════════════════════════════════════════════════

class SystemMonitor:
    def __init__(self):
        self.cpu_percent = 0.0
        self.gpu_percent = 0.0
        self.gpu_memory_percent = 0.0
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        while self._running:
            try:
                if PSUTIL_AVAILABLE:
                    cpu = psutil.cpu_percent(interval=0.5)
                else:
                    cpu = 0.0
                
                gpu = 0.0
                gpu_mem = 0.0
                
                if GPU_AVAILABLE:
                    try:
                        if GPU_LIBRARY == "gputil":
                            import GPUtil
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu = gpus[0].load * 100
                                gpu_mem = gpus[0].memoryUtil * 100
                        elif GPU_LIBRARY == "pynvml":
                            import pynvml
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu = util.gpu
                            gpu_mem = util.memory
                        elif GPU_LIBRARY == "pytorch":
                            # fallback: approximate GPU usage via allocated memory
                            try:
                                total = float(torch.cuda.get_device_properties(0).total_memory)
                                used = float(torch.cuda.memory_allocated(0))
                                gpu_mem = (used / total) * 100 if total > 0 else 0.0
                                # use memory percent as proxy for compute load when no tooling available
                                gpu = gpu_mem
                            except Exception:
                                pass
                    except:
                        pass
                
                with self._lock:
                    self.cpu_percent = cpu
                    self.gpu_percent = gpu
                    self.gpu_memory_percent = gpu_mem
                
            except Exception as e:
                print(f"Monitor error: {e}")
            
            time.sleep(0.5)
    
    def get_stats(self):
        with self._lock:
            return {
                'cpu': self.cpu_percent,
                'gpu': self.gpu_percent,
                'gpu_memory': self.gpu_memory_percent
            }


# ═══════════════════════════════════════════════════════════════════════════
# SMOOTH BOUNDING BOX TRACKER - PREVENTS FLICKERING
# ═══════════════════════════════════════════════════════════════════════════

class MqttPublisher:
    def __init__(self):
        self.client = None
        self.connected = False
    
    def connect(self):
        if not MQTT_AVAILABLE:
            return False
        try:
            self.client = mqtt.Client(protocol=mqtt.MQTTv311)
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            self.client.tls_insecure_set(False)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"? MQTT connect error: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        self.connected = (rc == 0)
        if self.connected:
            print("✓ MQTT connected")
        else:
            print(f"? MQTT connect failed: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
    
    def publish(self, payload):
        if not self.client or not self.connected:
            return
        try:
            self.client.publish(MQTT_TOPIC, json.dumps(payload), qos=0, retain=False)
        except Exception:
            pass
    
    def close(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
            self.client = None


class MqttSubscriber:
    """Simple MQTT client that subscribes to a topic and calls a callback
    whenever a message arrives. Runs the Paho loop in a background thread.
    """

    def __init__(self, topic, message_callback=None):
        self.client = None
        self.connected = False
        self.topic = topic
        self.message_callback = message_callback

    def connect(self):
        if not MQTT_AVAILABLE:
            return False
        try:
            self.client = mqtt.Client(protocol=mqtt.MQTTv311)
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            self.client.tls_insecure_set(False)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"? MQTT subscribe error: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        self.connected = (rc == 0)
        if self.connected:
            client.subscribe(self.topic)
            print(f"✓ Subscribed to MQTT topic: {self.topic}")
        else:
            print(f"? MQTT subscribe failed: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
        except Exception:
            payload = str(msg.payload)
        if self.message_callback:
            self.message_callback(msg.topic, payload)

    def close(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
            self.client = None

class SmoothBoundingBox:
    def __init__(self, smoothing_factor=0.6, max_age=20):
        self.smoothing_factor = smoothing_factor
        self.tracks = {}
        self.max_age = max_age
    
    def update(self, track_id, new_box, det_class):
        new_box = [float(x) for x in new_box]
        
        if track_id in self.tracks:
            old_box = self.tracks[track_id]['box']
            smoothed_box = [
                old_box[0] * self.smoothing_factor + new_box[0] * (1 - self.smoothing_factor),
                old_box[1] * self.smoothing_factor + new_box[1] * (1 - self.smoothing_factor),
                old_box[2] * self.smoothing_factor + new_box[2] * (1 - self.smoothing_factor),
                old_box[3] * self.smoothing_factor + new_box[3] * (1 - self.smoothing_factor),
            ]
            self.tracks[track_id]['box'] = smoothed_box
            self.tracks[track_id]['class'] = det_class
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['visible'] = True
            self.tracks[track_id]['frames_seen'] += 1
        else:
            self.tracks[track_id] = {
                'box': new_box,
                'class': det_class,
                'age': 0,
                'visible': True,
                'frames_seen': 1
            }
    
    def age_tracks(self, active_ids):
        to_remove = []
        for track_id in self.tracks:
            if track_id not in active_ids:
                self.tracks[track_id]['age'] += 1
                self.tracks[track_id]['visible'] = False
                if self.tracks[track_id]['age'] > self.max_age:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_all_tracks(self):
        return self.tracks.copy()
    
    def reset(self):
        self.tracks = {}


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSING THREAD
# ═══════════════════════════════════════════════════════════════════════════

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    detection_signal = pyqtSignal(list)
    stats_signal = pyqtSignal(int, int, int, int, int, float)
    image_saved_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(str)
    
    def __init__(self, camera_source=0, model_path='yolo11n.pt', is_rtsp=False, save_folder='captured_images'):
        super().__init__()
        self._running = False
        self.model = None
        self.person_model = None
        self.model_path = model_path
        self.cap = None
        self.camera_source = camera_source
        self.is_rtsp = is_rtsp
        
        self.frame_queue = None
        self.rtsp_thread = None
        self.rtsp_running = False
        
        self.system_monitor = SystemMonitor()
        self.mqtt = MqttPublisher()
        self.last_mqtt_publish = 0.0
        self._mqtt_connected = False
        
        self.smooth_boxes = SmoothBoundingBox(smoothing_factor=0.85, max_age=40)
        self.unique_persons_with_vest = set()
        self.unique_persons_without_vest = set()
        
        self.current_persons = 0
        self.current_with_vest = 0
        self.current_without_vest = 0
        
        self.vest_classes = []
        self.no_vest_classes = []
        
        self.save_folder = save_folder
        self.last_save_time = 0
        self.save_cooldown = 3
        self.auto_save_enabled = True
        
        self._create_save_folder()
        self.debug = True
        
    def _create_save_folder(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            print(f"✓ Created save folder: {self.save_folder}")
        
    def debug_print(self, msg):
        if self.debug:
            print(f"[VideoThread] {msg}")
    
    def _classify_model_classes(self):
        if self.model is None:
            return
        
        self.vest_classes = []
        self.no_vest_classes = []
        
        for cls_id, cls_name in self.model.names.items():
            cls_lower = cls_name.lower()
            
            no_vest_patterns = ['no_vest', 'no-vest', 'no vest', 'novest', 'without_vest', 
                               'without-vest', 'without vest', 'missing_vest', 'missing-vest',
                               'missing vest', 'no_safety', 'no-safety', 'no safety',
                               'unsafe', 'violation', 'no ppe', 'no-ppe']
            
            is_no_vest = False
            for pattern in no_vest_patterns:
                if pattern in cls_lower:
                    self.no_vest_classes.append(cls_id)
                    is_no_vest = True
                    self.debug_print(f"Class '{cls_name}' (ID: {cls_id}) -> NO VEST")
                    break
            
            if is_no_vest:
                continue
            
            vest_patterns = ['vest', 'safety', 'ppe', 'jacket', 'high-vis', 'highvis',
                            'reflective', 'protective', 'safe']
            
            for pattern in vest_patterns:
                if pattern in cls_lower:
                    self.vest_classes.append(cls_id)
                    self.debug_print(f"Class '{cls_name}' (ID: {cls_id}) -> WITH VEST")
                    break
        
        self.debug_print(f"Vest classes: {self.vest_classes}")
        self.debug_print(f"No-vest classes: {self.no_vest_classes}")
        
    def load_model(self):
        try:
            self.debug_print(f"Loading vest model: {self.model_path}")
            self.debug_print(f"Device: {DEVICE}")
            self.connection_status_signal.emit("Loading vest detection model...")
            
            if not os.path.exists(self.model_path):
                self.error_signal.emit(f"Model not found: {self.model_path}")
                return False
            
            # Load with GPU optimization
            self.model = YOLO(self.model_path)
            self.model.to(DEVICE)  # Explicitly set device
            self.debug_print("Vest model loaded successfully")
            self.debug_print(f"Vest model classes: {self.model.names}")
            
            self._classify_model_classes()
            
            self.connection_status_signal.emit("Loading person detection model...")
            self.debug_print("Loading person detection model (yolov8n.pt)...")
            self.person_model = YOLO('yolov8n.pt')
            self.person_model.to(DEVICE)  # Explicitly set device
            self.debug_print("Person model loaded successfully")
            

            
            self.connection_status_signal.emit("All models loaded!")
            return True
        except Exception as e:
            self.error_signal.emit(f"Model error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _rtsp_reader(self):
        self.debug_print("RTSP reader started")
        failures = 0
        
        while self.rtsp_running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    failures = 0
                    if not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put(frame)
                else:
                    failures += 1
                    if failures >= 30:
                        self.debug_print("Reconnecting...")
                        self._reconnect()
                        failures = 0
                    time.sleep(0.01)
            except Exception as e:
                self.debug_print(f"RTSP error: {e}")
                time.sleep(0.1)
    
    def _reconnect(self):
        if self.cap:
            self.cap.release()
        
        for i in range(3):
            time.sleep(2)
            self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    self.debug_print("Reconnected!")
                    return True
        return False
    
    def _setup_capture(self):
        self.debug_print(f"Setting up capture: {self.camera_source} (RTSP: {self.is_rtsp})")
        
        if self.is_rtsp:
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_source, backend)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.debug_print(f"Success! Frame: {frame.shape}")
                        self.frame_queue = queue.Queue(maxsize=2)
                        self.rtsp_running = True
                        self.rtsp_thread = threading.Thread(target=self._rtsp_reader, daemon=True)
                        self.rtsp_thread.start()
                        return True
                    self.cap.release()
            
            self.error_signal.emit(f"Failed to connect to RTSP:\n{self.camera_source}")
            return False
        else:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_source, backend)
                
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.debug_print(f"Success! Frame: {frame.shape}")
                        return True
                    self.cap.release()
            
            self.error_signal.emit(f"Failed to open camera {self.camera_source}")
            return False
    
    def _save_image(self, frame, detection_type="vest"):
        current_time = time.time()
        
        if current_time - self.last_save_time < self.save_cooldown:
            return None
        
        self.last_save_time = current_time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{detection_type}_{timestamp}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            self.debug_print(f"Image saved: {filepath}")
            return filepath
        except Exception as e:
            self.debug_print(f"Failed to save image: {e}")
            return None

    def _send_dashboard_update(self, payload):
        try:
            req = urllib.request.Request(
                DASHBOARD_UPDATE_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=0.25):
                pass
        except Exception:
            pass

    def _vest_inside_person(self, vest_box, person_box, threshold=0.3):
        vx1, vy1, vx2, vy2 = vest_box
        px1, py1, px2, py2 = person_box
        
        ix1 = max(vx1, px1)
        iy1 = max(vy1, py1)
        ix2 = min(vx2, px2)
        iy2 = min(vy2, py2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        vest_area = (vx2 - vx1) * (vy2 - vy1)
        
        return (intersection / vest_area) >= threshold if vest_area > 0 else False
    
    def _draw_smooth_box(self, frame, box, label, color, alpha=1.0):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x2 <= x1 or y2 <= y1:
            return
        
        overlay = frame.copy()
        
        if alpha < 1.0:
            faded_color = tuple(int(c * alpha) for c in color)
        else:
            faded_color = color
        
        thickness = 3 if alpha > 0.7 else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), faded_color, thickness, cv2.LINE_AA)
        
        corner_len = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
        corner_thickness = thickness + 1
        
        cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), faded_color, corner_thickness, cv2.LINE_AA)
        cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), faded_color, corner_thickness, cv2.LINE_AA)
        cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), faded_color, corner_thickness, cv2.LINE_AA)
        cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), faded_color, corner_thickness, cv2.LINE_AA)
        cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), faded_color, corner_thickness, cv2.LINE_AA)
        cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), faded_color, corner_thickness, cv2.LINE_AA)
        cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), faded_color, corner_thickness, cv2.LINE_AA)
        cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), faded_color, corner_thickness, cv2.LINE_AA)
        
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(0, y1 - th - 12)
        label_y2 = y1
        label_x2 = min(w, x1 + tw + 12)
        
        cv2.rectangle(overlay, (x1, label_y1), (label_x2, label_y2), faded_color, -1, cv2.LINE_AA)
        cv2.putText(overlay, label, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            frame[:] = overlay
    
    def _draw_tracking_info(self, frame):
        h, w = frame.shape[:2]
        info_x = w - 230
        info_y = 10
        box_height = 110
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x - 10, info_y), (w - 10, info_y + box_height), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (info_x - 10, info_y), (w - 10, info_y + box_height), (100, 100, 120), 2, cv2.LINE_AA)
        
        cv2.putText(frame, "TRACKING INFO", (info_x, info_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (info_x, info_y + 30), (w - 20, info_y + 30), (80, 80, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Persons in Frame: {self.current_persons}", (info_x, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Unique w/ Vest: {len(self.unique_persons_with_vest)}", (info_x, info_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Unique no Vest: {len(self.unique_persons_without_vest)}", (info_x, info_y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)
    
    def _draw_text_with_outline(self, frame, text, pos, font_scale, color, thickness=2):
        x, y = pos
        cv2.putText(frame, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    def _get_color(self, value):
        if value < 50:
            return (0, 255, 100)
        elif value < 80:
            return (0, 220, 255)
        else:
            return (0, 100, 255)
    
    def _draw_system_stats(self, frame, fps, cpu, gpu, vram):
        x = 15
        y_start = 30
        y_spacing = 25
        font_scale = 0.55
        thickness = 2
        
        fps_color = (0, 255, 100) if fps >= 20 else (0, 220, 255) if fps >= 10 else (0, 100, 255)
        self._draw_text_with_outline(frame, f"FPS: {fps:.1f}", (x, y_start), font_scale, fps_color, thickness)
        
        cpu_color = self._get_color(cpu)
        self._draw_text_with_outline(frame, f"CPU: {cpu:.0f}%", (x, y_start + y_spacing), font_scale, cpu_color, thickness)
        
        if GPU_AVAILABLE:
            gpu_color = self._get_color(gpu)
            self._draw_text_with_outline(frame, f"GPU: {gpu:.0f}%", (x, y_start + y_spacing * 2), font_scale, gpu_color, thickness)
            vram_color = self._get_color(vram)
            self._draw_text_with_outline(frame, f"VRAM: {vram:.0f}%", (x, y_start + y_spacing * 3), font_scale, vram_color, thickness)
        else:
            self._draw_text_with_outline(frame, "GPU: N/A", (x, y_start + y_spacing * 2), font_scale, (150, 150, 150), thickness)
        
        if self.is_rtsp:
            h, w = frame.shape[:2]
            self._draw_text_with_outline(frame, "RTSP", (w - 70, 30), font_scale, (0, 255, 255), thickness)
        
    def run(self):
        self.debug_print("Thread started")
        self.system_monitor.start()
        self._mqtt_connected = self.mqtt.connect()
        
        if not self.load_model():
            self.system_monitor.stop()
            self._mqtt_connected = False
            self.mqtt.close()
            return
        
        if not self._setup_capture():
            self.system_monitor.stop()
            self._mqtt_connected = False
            self.mqtt.close()
            return
        
        self._running = True
        frame_count = 0
        start_time = datetime.now()
        
        self.debug_print("Processing loop started")
        
        while self._running:
            if self.is_rtsp and self.frame_queue:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    ret = True
                except:
                    continue
            else:
                ret, frame = self.cap.read()
            
            if not ret or frame is None:
                continue
            frame_started_ts = time.time()
            
            frame_count += 1
            original_frame = frame.copy()
            annotated_frame = frame.copy()
            
            # GPU Inference with Half-Precision (FP16) for speed
            # Detect persons - optimized for GPU
            person_results = self.person_model(
                frame, 
                conf=0.45,
                classes=[0], 
                verbose=False,
                device=DEVICE,
                half=(DEVICE == 'cuda'),
                imgsz=416              # smaller for faster CPU processing
            )
            person_boxes = []
            
            for box in person_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                person_boxes.append({
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': conf,
                    'has_vest': False
                })
            
            # Detect vests - optimized for GPU
            vest_results = self.model(
                frame, 
                conf=0.45,
                verbose=False,
                device=DEVICE,
                half=(DEVICE == 'cuda'),
                imgsz=416              # smaller for faster CPU processing
            )
            vest_detections = []
            detection_list = []
            
            for box in vest_results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                detection_list.append((class_name, conf))
                
                if cls_id in self.vest_classes:
                    vest_detections.append([float(x1), float(y1), float(x2), float(y2), conf])
            
            # Associate vests with persons
            for person in person_boxes:
                person_box = person['box']
                for vest in vest_detections:
                    if self._vest_inside_person(vest[:4], person_box, threshold=0.3):
                        person['has_vest'] = True
                        break
            
            self.current_persons = len(person_boxes)
            self.current_with_vest = sum(1 for p in person_boxes if p['has_vest'])
            self.current_without_vest = sum(1 for p in person_boxes if not p['has_vest'])
            
            vest_detected_this_frame = self.current_with_vest > 0
            
            # Simple ID-less tracking: count vests per person
            for i, person in enumerate(person_boxes):
                track_id = i  # Simple index-based ID
                det_class = 'with_vest' if person['has_vest'] else 'no_vest'
                self.smooth_boxes.update(track_id, person['box'], det_class)
                
                if det_class == 'with_vest':
                    self.unique_persons_with_vest.add(track_id)
                    self.unique_persons_without_vest.discard(track_id)
                else:
                    if track_id not in self.unique_persons_with_vest:
                        self.unique_persons_without_vest.add(track_id)
            
            # Age out old tracks
            active_track_ids = set(range(len(person_boxes)))
            self.smooth_boxes.age_tracks(active_track_ids)
            
            # Draw smooth boxes (every frame for smooth appearance)
            all_tracks = self.smooth_boxes.get_all_tracks()
            
            for track_id, track_data in all_tracks.items():
                box = track_data['box']
                det_class = track_data['class']
                age = track_data['age']
                visible = track_data['visible']
                
                if visible:
                    alpha = 1.0
                else:
                    alpha = max(0.2, 1.0 - (age / self.smooth_boxes.max_age) * 0.8)
                
                if det_class == 'with_vest':
                    color = (0, 255, 0)
                    label = f"ID:{track_id} SAFE"
                else:
                    color = (0, 0, 255)
                    label = f"ID:{track_id} VIOLATION"
                
                self._draw_smooth_box(annotated_frame, box, label, color, alpha)
            
            # Draw vest indicators
            for vest in vest_detections:
                x1, y1, x2, y2 = map(int, vest[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(annotated_frame, (cx, cy), 7, (0, 200, 0), 2, cv2.LINE_AA)
            
            # Save image
            if vest_detected_this_frame and self.auto_save_enabled:
                saved_path = self._save_image(original_frame, "safety_vest")
                if saved_path:
                    self.image_saved_signal.emit(saved_path)
            
            # FPS
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            stats = self.system_monitor.get_stats()
            
            self._draw_system_stats(annotated_frame, fps, stats['cpu'], stats['gpu'], stats['gpu_memory'])
            self._draw_tracking_info(annotated_frame)
            
            rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            
            self.change_pixmap_signal.emit(qt_image)
            self.detection_signal.emit(detection_list)
            self.stats_signal.emit(
                self.current_persons,
                self.current_with_vest,
                self.current_without_vest,
                len(self.unique_persons_with_vest),
                len(self.unique_persons_without_vest),
                fps
            )
            
            # Periodic GPU memory cleanup (every 100 frames)
            if DEVICE == 'cuda' and frame_count % 100 == 0:
                torch.cuda.empty_cache()
            
            now = time.time()
            if now - self.last_mqtt_publish >= MQTT_PUBLISH_INTERVAL_SEC:
                payload = {
                    "people_in_frame": self.current_persons,
                    "compliant": self.current_with_vest,
                    "violated": self.current_without_vest,
                    "cpu_percent": round(stats.get("cpu", 0.0), 2),
                    "gpu_percent": round(stats.get("gpu", 0.0), 2),
                    "gpu_memory_percent": round(stats.get("gpu_memory", 0.0), 2),
                    "fps": round(fps, 2),
                    "frame_processing_ms": round((time.time() - frame_started_ts) * 1000.0, 2),
                    "source_timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "source_epoch_ms": int(time.time() * 1000),
                    "camera_source": str(self.camera_source),
                }
                self._send_dashboard_update(payload)
                self.mqtt.publish(payload)
                self.last_mqtt_publish = now
        
        self.system_monitor.stop()
        self._mqtt_connected = False
        self.mqtt.close()
        self.rtsp_running = False
        if self.cap:
            self.cap.release()
        self.debug_print("Thread ended")
    
    def reset_counters(self):
        self.current_persons = 0
        self.current_with_vest = 0
        self.current_without_vest = 0
        self.unique_persons_with_vest = set()
        self.unique_persons_without_vest = set()
        self.smooth_boxes.reset()
    
    def set_auto_save(self, enabled):
        self.auto_save_enabled = enabled
    
    def stop(self):
        self.debug_print("Stopping...")
        self._running = False
        self.rtsp_running = False
        self.system_monitor.stop()
        
        # Clean up GPU memory
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            self.debug_print("GPU memory cleared")
        
        self.wait(3000)
        if self.isRunning():
            self.terminate()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN GUI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class SafetyVestDetectorGUI(QMainWindow):
    # signal used to safely marshal MQTT messages from background thread into GUI
    mqtt_message_signal = pyqtSignal(str, str)  # topic, payload

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦺 Safety Vest Detection - Smooth Tracking")
        self.setGeometry(50, 50, 1400, 850)
        self.setMinimumSize(1000, 600)
        
        self.video_thread = None
        self.current_model_path = r'C:\Users\kckes\OneDrive\Desktop\NOKIA TEMI\Yolo Detection\Safety Vests.v14-rf-detr-medium-576x576.yolov11\trained_model2\weights\best.pt'
        self.save_folder = os.path.join(os.path.expanduser("~"), "Desktop", "SafetyVest_Captures")
        
        self.rtsp_presets = [
            "rtsp://192.168.1.100:554/live",
            "rtsp://192.168.1.100:554/stream1",
            "rtsp://192.168.1.100:8554/stream",
        ]
        
        self.images_saved = 0
        self.init_ui()
        self.setup_animations()
        # start a SystemMonitor for visible CPU/GPU stats in the GUI
        self.system_monitor = SystemMonitor()
        try:
            self.system_monitor.start()
        except Exception:
            pass
        
        # start MQTT subscriber for dashboard logging
        self.mqtt_subscriber = None
        if MQTT_AVAILABLE:
            self.mqtt_subscriber = MqttSubscriber(MQTT_TOPIC, self.handle_mqtt_message)
            connected = self.mqtt_subscriber.connect()
            if connected:
                self.log_message("MQTT dashboard: subscribed to topic")
            else:
                self.log_message("MQTT dashboard: failed to subscribe")
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(main_layout)
        
        # Title
        title_frame = QFrame()
        title_frame.setFixedHeight(40)
        title_layout = QHBoxLayout(title_frame)
        title_label = QLabel("🦺 SAFETY VEST DETECTION - SMOOTH TRACKING")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: white;")
        title_layout.addWidget(title_label)
        title_frame.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2); border-radius: 8px;")
        main_layout.addWidget(title_frame)
        
        # Status bar
        self.connection_label = QLabel("📡 Status: Ready")
        self.connection_label.setFont(QFont("Consolas", 9))
        self.connection_label.setStyleSheet("color: #00d4ff; background: rgba(0,50,100,0.5); padding: 5px; border-radius: 4px;")
        main_layout.addWidget(self.connection_label)
        
        # Content
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        main_layout.addLayout(content_layout, 1)
        
        # Video panel
        video_frame = QFrame()
        video_frame.setStyleSheet("background: #1a1a2e; border-radius: 10px; border: 1px solid #3d3d5c;")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setSpacing(5)
        video_layout.setContentsMargins(8, 8, 8, 8)
        
        # Controls row 1
        ctrl1 = QHBoxLayout()
        ctrl1.addWidget(QLabel("📷"))
        self.camera_combo = QComboBox()
        self.camera_combo.setFixedSize(100, 25)
        self.detect_cameras()
        ctrl1.addWidget(self.camera_combo)
        
        self.rtsp_checkbox = QCheckBox("RTSP")
        self.rtsp_checkbox.stateChanged.connect(self.toggle_rtsp)
        ctrl1.addWidget(self.rtsp_checkbox)
        ctrl1.addStretch()
        
        ctrl1.addWidget(QLabel("🤖"))
        self.model_label = QLabel("Model...")
        self.model_label.setStyleSheet("color: #00ff88;")
        ctrl1.addWidget(self.model_label)
        
        browse_btn = QPushButton("📁")
        browse_btn.setFixedSize(30, 25)
        browse_btn.clicked.connect(self.browse_model)
        ctrl1.addWidget(browse_btn)
        video_layout.addLayout(ctrl1)
        
        # Controls row 2 - RTSP
        ctrl2 = QHBoxLayout()
        ctrl2.addWidget(QLabel("📡"))
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://IP:port/stream")
        self.rtsp_input.setEnabled(False)
        ctrl2.addWidget(self.rtsp_input)
        
        self.rtsp_preset_combo = QComboBox()
        self.rtsp_preset_combo.setFixedWidth(100)
        self.rtsp_preset_combo.addItem("Presets")
        for preset in self.rtsp_presets:
            self.rtsp_preset_combo.addItem(preset.split('/')[-1], preset)
        self.rtsp_preset_combo.setEnabled(False)
        self.rtsp_preset_combo.currentIndexChanged.connect(self.apply_preset)
        ctrl2.addWidget(self.rtsp_preset_combo)
        
        self.test_btn = QPushButton("Test")
        self.test_btn.setFixedSize(45, 25)
        self.test_btn.setEnabled(False)
        self.test_btn.clicked.connect(self.test_rtsp)
        ctrl2.addWidget(self.test_btn)
        video_layout.addLayout(ctrl2)
        
        # Controls row 3 - Save
        ctrl3 = QHBoxLayout()
        ctrl3.addWidget(QLabel("💾"))
        self.save_folder_label = QLabel(os.path.basename(self.save_folder))
        self.save_folder_label.setStyleSheet("color: #00d4ff;")
        ctrl3.addWidget(self.save_folder_label, 1)
        
        change_btn = QPushButton("📂")
        change_btn.setFixedSize(30, 25)
        change_btn.clicked.connect(self.change_save_folder)
        ctrl3.addWidget(change_btn)
        
        open_btn = QPushButton("📁")
        open_btn.setFixedSize(30, 25)
        open_btn.clicked.connect(self.open_save_folder)
        ctrl3.addWidget(open_btn)
        
        self.auto_save_checkbox = QCheckBox("Auto")
        self.auto_save_checkbox.setChecked(True)
        self.auto_save_checkbox.stateChanged.connect(self.toggle_auto_save)
        ctrl3.addWidget(self.auto_save_checkbox)
        video_layout.addLayout(ctrl3)
        
        # Video display
        self.video_label = QLabel("📺 Click START to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background: #0a0a0f; border: 2px solid #3d3d5c; border-radius: 8px; color: #666;")
        video_layout.addWidget(self.video_label, 1)
        
        content_layout.addWidget(video_frame, 1)
        
        # Info panel
        info_frame = QFrame()
        info_frame.setFixedWidth(300)
        info_frame.setStyleSheet("background: #1a1a2e; border-radius: 10px; border: 1px solid #3d3d5c;")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(10, 10, 10, 10)
        
        # Status
        status_row = QHBoxLayout()
        self.status_indicator = QLabel("● OFFLINE")
        self.status_indicator.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.status_indicator.setStyleSheet("color: #ff4757;")
        status_row.addWidget(self.status_indicator)
        status_row.addStretch()
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #00ff88;")
        status_row.addWidget(self.fps_label)
        # GPU status label (updated periodically)
        self.gpu_status_label = QLabel("GPU: N/A")
        self.gpu_status_label.setStyleSheet("color: #00d4ff;")
        status_row.addWidget(self.gpu_status_label)
        info_layout.addLayout(status_row)
        
        # Current stats
        info_layout.addWidget(QLabel("👁 CURRENT IN FRAME"))
        
        current_row = QHBoxLayout()
        
        self.current_persons_label = QLabel("0")
        self.current_persons_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.current_persons_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_persons_label.setStyleSheet("color: white; background: #252540; border-radius: 6px; padding: 10px;")
        current_row.addWidget(self.current_persons_label)
        
        self.current_vest_label = QLabel("0")
        self.current_vest_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.current_vest_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_vest_label.setStyleSheet("color: #00ff88; background: rgba(0,255,136,0.15); border: 1px solid #00ff88; border-radius: 6px; padding: 10px;")
        current_row.addWidget(self.current_vest_label)
        
        self.current_no_vest_label = QLabel("0")
        self.current_no_vest_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.current_no_vest_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_no_vest_label.setStyleSheet("color: #ff6b6b; background: rgba(255,107,107,0.15); border: 1px solid #ff6b6b; border-radius: 6px; padding: 10px;")
        current_row.addWidget(self.current_no_vest_label)
        
        info_layout.addLayout(current_row)
        
        # Unique stats
        info_layout.addWidget(QLabel("📊 UNIQUE PERSONS (TOTAL)"))
        
        self.unique_vest_label = QLabel("0")
        self.unique_vest_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.unique_vest_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unique_vest_label.setStyleSheet("color: #00ff88; background: rgba(0,255,136,0.2); border: 2px solid #00ff88; border-radius: 8px; padding: 15px;")
        info_layout.addWidget(self.unique_vest_label)
        
        self.unique_no_vest_label = QLabel("0")
        self.unique_no_vest_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.unique_no_vest_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unique_no_vest_label.setStyleSheet("color: #ff6b6b; background: rgba(255,107,107,0.2); border: 2px solid #ff6b6b; border-radius: 8px; padding: 15px;")
        info_layout.addWidget(self.unique_no_vest_label)
        
        # Saved count
        saved_row = QHBoxLayout()
        saved_row.addWidget(QLabel("💾 Images Saved:"))
        self.saved_count_label = QLabel("0")
        self.saved_count_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.saved_count_label.setStyleSheet("color: #00d4ff;")
        saved_row.addWidget(self.saved_count_label)
        saved_row.addStretch()
        info_layout.addLayout(saved_row)
        
        # Log
        info_layout.addWidget(QLabel("📋 LOG"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #0a0a0f; color: #00ff88; font-family: Consolas; font-size: 9px; border-radius: 6px;")
        info_layout.addWidget(self.log_text, 1)

        # connect MQTT signal to log updater (thread-safe)
        self.mqtt_message_signal.connect(self.on_mqtt_received)
        
        content_layout.addWidget(info_frame)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.start_btn = QPushButton("▶ START")
        self.start_btn.setFixedSize(100, 35)
        self.start_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.start_btn.setStyleSheet("background: #00b09b; color: white; border-radius: 6px;")
        self.start_btn.clicked.connect(self.start_detection)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setFixedSize(100, 35)
        self.stop_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.stop_btn.setStyleSheet("background: #eb3349; color: white; border-radius: 6px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        btn_layout.addWidget(self.stop_btn)
        
        self.mqtt_test_btn = QPushButton("📡 MQTT TEST")
        self.mqtt_test_btn.setFixedSize(120, 35)
        self.mqtt_test_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.mqtt_test_btn.setStyleSheet("background: #00b09b; color: white; border-radius: 6px;")
        self.mqtt_test_btn.clicked.connect(self.test_mqtt_publish)
        btn_layout.addWidget(self.mqtt_test_btn)
        
        self.reset_btn = QPushButton("🔄 RESET")
        self.reset_btn.setFixedSize(100, 35)
        self.reset_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.reset_btn.setStyleSheet("background: #667eea; color: white; border-radius: 6px;")
        self.reset_btn.clicked.connect(self.reset_stats)
        btn_layout.addWidget(self.reset_btn)
        
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        
        # Update model label
        if os.path.exists(self.current_model_path):
            name = os.path.basename(self.current_model_path)
            self.model_label.setText(name[:20] + "..." if len(name) > 20 else name)
        
        # Style
        self.setStyleSheet("""
            QMainWindow { background: #16213e; }
            QLabel { color: #b8b8d1; }
            QLineEdit { background: #1a1a2e; color: #00d4ff; border: 1px solid #4d4d6d; border-radius: 4px; padding: 5px; }
            QComboBox { background: #2d2d44; color: white; border: 1px solid #4d4d6d; border-radius: 4px; padding: 3px; }
            QCheckBox { color: #00d4ff; }
            QPushButton { background: #4d4d6d; color: white; border: none; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background: #00d4ff; }
        """)
    
    def setup_animations(self):
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.pulse)
        self.pulse_state = True
        # periodic update for system stats (CPU/GPU)
        self._sys_timer = QTimer()
        self._sys_timer.timeout.connect(self.update_system_stats)
        self._sys_timer.start(500)
    
    def pulse(self):
        color = "#00ff00" if self.pulse_state else "#00aa00"
        self.status_indicator.setStyleSheet(f"color: {color};")
        self.pulse_state = not self.pulse_state
    
    def detect_cameras(self):
        """Detect a single local webcam to simplify selection.

        Tries indexes 0..2 and adds only the first working device as
        `Webcam (index N)`. If none work, shows `None`.
        """
        self.camera_combo.clear()
        found = False
        # try only a few common indices and stop on first working camera
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.camera_combo.addItem(f"Webcam (index {i})", i)
                    found = True
                    break
        if not found:
            self.camera_combo.addItem("None", -1)
    
    def toggle_rtsp(self, state):
        is_rtsp = state == Qt.CheckState.Checked.value
        self.camera_combo.setEnabled(not is_rtsp)
        self.rtsp_input.setEnabled(is_rtsp)
        self.rtsp_preset_combo.setEnabled(is_rtsp)
        self.test_btn.setEnabled(is_rtsp)
    
    def apply_preset(self, index):
        if index > 0:
            preset = self.rtsp_preset_combo.itemData(index)
            if preset:
                self.rtsp_input.setText(preset)
    
    def toggle_auto_save(self, state):
        if self.video_thread:
            self.video_thread.set_auto_save(state == Qt.CheckState.Checked.value)
    
    def change_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.save_folder)
        if folder:
            self.save_folder = folder
            self.save_folder_label.setText(os.path.basename(folder))
    
    def open_save_folder(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        if sys.platform == 'win32':
            os.startfile(self.save_folder)
        elif sys.platform == 'darwin':
            os.system(f'open "{self.save_folder}"')
        else:
            os.system(f'xdg-open "{self.save_folder}"')
    
    def log_message(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

    # --- MQTT dashboard helpers ------------------------------------------------
    def handle_mqtt_message(self, topic, payload):
        """Called from MQTT subscriber thread."""
        # forward to Qt thread via signal
        self.mqtt_message_signal.emit(topic, payload)

    def on_mqtt_received(self, topic, payload):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] MQTT ← {topic}: {payload}")

    def update_system_stats(self):
        try:
            if hasattr(self, 'system_monitor') and self.system_monitor:
                stats = self.system_monitor.get_stats()
                gpu = stats.get('gpu', 0.0)
                vram = stats.get('gpu_memory', 0.0)
                # format nicely
                self.gpu_status_label.setText(f"GPU: {gpu:.0f}%  VRAM: {vram:.0f}%")
        except Exception:
            pass
    
    def test_rtsp(self):
        url = self.rtsp_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Enter RTSP URL!")
            return
        
        self.connection_label.setText("📡 Testing...")
        QApplication.processEvents()
        
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            start = time.time()
            success = False
            while time.time() - start < 10:
                ret, frame = cap.read()
                if ret and frame is not None:
                    success = True
                    break
                time.sleep(0.1)
            cap.release()
            
            if success:
                self.connection_label.setText("✅ Connected!")
                QMessageBox.information(self, "Success", "Connection successful!")
            else:
                self.connection_label.setText("❌ Failed")
                QMessageBox.warning(self, "Failed", "No frames received")
        except Exception as e:
            self.connection_label.setText("❌ Error")
            QMessageBox.critical(self, "Error", str(e))
    
    def test_mqtt_publish(self):
        if not MQTT_AVAILABLE:
            QMessageBox.warning(self, "Error", "paho-mqtt not installed")
            return
        
        payload = {
            "type": "test",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": "mqtt test"
        }
        
        publisher = MqttPublisher()
        if not publisher.connect():
            QMessageBox.critical(self, "Error", "MQTT connect failed")
            return
        
        start = time.time()
        while time.time() - start < 3:
            if publisher.connected:
                break
            QApplication.processEvents()
            time.sleep(0.1)
        
        if not publisher.connected:
            publisher.close()
            QMessageBox.critical(self, "Error", "MQTT connection timeout")
            return
        
        publisher.publish(payload)
        publisher.close()
        QMessageBox.information(self, "Success", "Test message published")

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model (*.pt)")
        if path and os.path.exists(path):
            self.current_model_path = path
            name = os.path.basename(path)
            self.model_label.setText(name[:20] + "..." if len(name) > 20 else name)
    
    def start_detection(self):
        if self.rtsp_checkbox.isChecked():
            source = self.rtsp_input.text().strip()
            if not source:
                QMessageBox.warning(self, "Error", "Enter RTSP URL!")
                return
            is_rtsp = True
        else:
            source = self.camera_combo.currentData()
            if source == -1:
                QMessageBox.warning(self, "Error", "No camera!")
                return
            is_rtsp = False
        
        if not os.path.exists(self.current_model_path):
            QMessageBox.critical(self, "Error", "Model not found!")
            return
        
        self.video_thread = VideoThread(source, self.current_model_path, is_rtsp, self.save_folder)
        self.video_thread.set_auto_save(self.auto_save_checkbox.isChecked())
        self.video_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_thread.stats_signal.connect(self.update_stats)
        self.video_thread.image_saved_signal.connect(self.on_image_saved)
        self.video_thread.error_signal.connect(self.show_error)
        self.video_thread.connection_status_signal.connect(lambda s: self.connection_label.setText(f"📡 {s}"))
        self.video_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_indicator.setText("● ONLINE")
        self.status_indicator.setStyleSheet("color: #00ff00;")
        self.pulse_timer.start(500)
        self.log_message("Started detection")
    
    def stop_detection(self):
        self.pulse_timer.stop()
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_indicator.setText("● OFFLINE")
        self.status_indicator.setStyleSheet("color: #ff4757;")
        self.video_label.setText("📺 Stopped")
        self.log_message("Stopped")
    
    def reset_stats(self):
        self.images_saved = 0
        self.current_persons_label.setText("0")
        self.current_vest_label.setText("0")
        self.current_no_vest_label.setText("0")
        self.unique_vest_label.setText("0")
        self.unique_no_vest_label.setText("0")
        self.saved_count_label.setText("0")
        self.fps_label.setText("FPS: 0")
        self.log_text.clear()
        
        if self.video_thread:
            self.video_thread.reset_counters()
        self.log_message("Counters reset")
    
    def update_frame(self, image):
        scaled = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)
    
    def update_stats(self, curr_persons, curr_vest, curr_no_vest, unique_vest, unique_no_vest, fps):
        self.current_persons_label.setText(str(curr_persons))
        self.current_vest_label.setText(str(curr_vest))
        self.current_no_vest_label.setText(str(curr_no_vest))
        self.unique_vest_label.setText(str(unique_vest))
        self.unique_no_vest_label.setText(str(unique_no_vest))
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def on_image_saved(self, filepath):
        self.images_saved += 1
        self.saved_count_label.setText(str(self.images_saved))
        self.log_message(f"💾 {os.path.basename(filepath)[:25]}")
    
    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.stop_detection()
    
    def closeEvent(self, event):
        self.pulse_timer.stop()
        if self.video_thread:
            self.video_thread.stop()
        # shut down MQTT subscriber as well
        if hasattr(self, 'mqtt_subscriber') and self.mqtt_subscriber:
            try:
                self.mqtt_subscriber.close()
            except Exception:
                pass
        # stop system monitor and timer
        try:
            if hasattr(self, '_sys_timer') and self._sys_timer:
                self._sys_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'system_monitor') and self.system_monitor:
                self.system_monitor.stop()
        except Exception:
            pass
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Safety Vest Detection...")
    print("="*60 + "\n")
    
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        window = SafetyVestDetectorGUI()
        window.show()
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        
