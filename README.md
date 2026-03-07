🦺 Real-Time AI Safety Vest Detection System

A production-grade PPE (Personal Protective Equipment) compliance monitoring system built for Nokia 5G Futures Lab. Uses dual YOLO models with smooth tracking, MQTT telemetry over TLS, and a real-time web dashboard — designed for deployment on 5G-connected edge devices with Nokia 360° cameras.

📋 Table of Contents
Overview
System Architecture
How It Works
Project Structure
Features
Installation
Usage
Model Training
Model Performance
Web Dashboard
Nokia 5G Camera Integration
API Reference
Performance Benchmarks
Key Design Decisions
Author
Overview

This system detects whether workers on construction sites are wearing safety vests in real-time. It uses a dual-model inference pipeline — YOLOv8n for person detection and a custom-trained YOLOv11n for vest classification — combined with smooth bounding box tracking for persistent identity monitoring.

The system integrates with Nokia 5G 360° cameras via RTSP streaming and the Nokia RXRM platform, making it suitable for deployment on 5G-connected edge devices in industrial environments.

System Architecture
┌─────────────────────────────────────────────────────────────────────────┐
│                        NOKIA 5G 360° CAMERA                            │
└──────────────┬──────────────────────────────────┬───────────────────────┘
               │ OSC Protocol                     │ 360° Video Feed
               ▼                                  ▼
┌──────────────────────────┐    ┌──────────────────────────────────────┐
│  camera_osc_client.py    │    │     rxrm_api_client.py               │
│  Camera Control Layer    │    │     Stream Management Layer          │
│                          │    │                                      │
│  • Set 8K/6K/4K res      │    │  • JWT authentication                │
│  • Start RTSP streaming  │    │  • Create 4 directional viewports    │
│  • Enable AI features    │    │    (0°, 90°, 180°, 270°)             │
│  • Monitor battery/temp  │    │  • Publish as RTSP streams           │
│  • Control face blur     │    │  • Output: 4 RTSP URLs               │
└──────────────────────────┘    └──────────────┬───────────────────────┘
                                               │
                              4 RTSP streams    │
                              rtsp://...        │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  GUI_Vest Detection.py (MAIN APPLICATION)               │
│                                                                         │
│  Video Input ──→ Person Detection (YOLOv8n) ──→ Vest Detection (v11)   │
│       │              │                               │                  │
│       │              └── Association Logic ───────────┘                  │
│       │                        │                                        │
│       │              Smooth Bounding Box Tracking                       │
│       │                        │                                        │
│       ▼                        ▼                                        │
│  PyQt6 GUI            MQTT + HTTP Publish                               │
│  (Live Video +        (every 1 second)                                  │
│   Stats Panel)                                                          │
└────────────┬────────────────────────────────┬───────────────────────────┘
             │ MQTT (TLS)                     │ HTTP POST
             ▼                                ▼
┌────────────────────────┐    ┌───────────────────────────────────────────┐
│   HiveMQ Cloud Broker  │    │   mqtt_web_dashboard.py                   │
│                        │    │   Flask Web Dashboard                     │
│  Topic:                │    │                                           │
│  safetyvest/status     │    │  • Real-time KPI cards                    │
│                        │    │  • Violation event logging                │
│  Connects to:          │    │  • CSV export                             │
│  • Grafana             │    │  • Prometheus /metrics endpoint           │
│  • Node-RED            │    │  • http://localhost:5000                  │
│  • Any MQTT client     │    │                                           │
└────────────────────────┘    └───────────────────────────────────────────┘

How It Works
Step	File	Description
1. Train	YOlO training .py	One-time offline training — downloads Roboflow Safety Vests dataset (v14), trains YOLOv11n for 50 epochs at 576×576, outputs best.pt
2. Configure Camera	camera_osc_client.py	Sets up Nokia 360° camera — resolution, streaming mode, AI features, face blur
3. Create Streams	rxrm_api_client.py	Connects to Nokia RXRM platform, slices 360° video into 4 directional RTSP streams
4. Detect	GUI_Vest Detection.py	Reads RTSP streams, runs dual YOLO inference on every frame, associates vests to persons, tracks identities
5. Monitor	mqtt_web_dashboard.py	Receives telemetry via HTTP + MQTT, displays real-time dashboard in browser
Project Structure
YOLO_detection_model/
├── GUI_Vest Detection.py       # Main application — PyQt6 GUI + dual YOLO inference + MQTT
├── YOlO training .py           # Model training script (YOLOv11n on Roboflow dataset)
├── camera_osc_client.py        # Nokia 5G 360° camera control via OSC protocol
├── rxrm_api_client.py          # Nokia RXRM platform API — viewport creation + RTSP publishing
├── mqtt_web_dashboard.py       # Flask web dashboard + REST API + Prometheus metrics
├── LEDtest.py                  # Hardware LED indicator test script
├── data.yaml                   # YOLO dataset configuration
├── data_fixed.yaml             # Fixed dataset paths configuration
├── violation_events.csv        # Sample violation event log
├── train/                      # Training dataset images & labels
├── valid/                      # Validation dataset images & labels
├── test/                       # Test dataset images & labels
└── README.md

Features
🎯 Detection & Tracking
Dual-model inference: YOLOv8n for person detection + custom YOLOv11n for vest classification
Vest-person association: Geometric overlap matching (IoU-based) to determine if a vest belongs to a person
Smooth bounding boxes: Exponential moving average (factor=0.85) eliminates box flickering
Track aging: Tracks fade out after 40 frames of no detection with visual fade effect
Unique person counting: Separate counters for compliant vs. violated individuals
⚡ GPU Optimization
FP16 half-precision inference on CUDA GPUs (~2× speedup)
cuDNN benchmark auto-tuning for optimal convolution algorithms
90% GPU memory allocation for maximum throughput
Periodic VRAM cleanup every 100 frames via torch.cuda.empty_cache()
Multi-threaded RTSP capture with producer-consumer queue (buffer size = 1)
📡 Communication & IoT
MQTT over TLS (HiveMQ Cloud, port 8883) for real-time telemetry publishing
MQTT subscriber built into GUI for dashboard message logging
HTTP POST to Flask dashboard (/api/update) every 1 second
Auto-reconnect on RTSP stream failure (3 retries with backoff)
🖥️ User Interface (PyQt6)
Dark-themed desktop GUI with live video feed
Real-time overlays: corner-style bounding boxes, tracking info panel, system stats
Green = SAFE (has vest), Red = VIOLATION (no vest)
RTSP connection testing and preset management
Auto-save violation images to disk with cooldown
📊 Web Dashboard (Flask)
Real-time KPI cards: people in frame, compliant, violated, FPS, CPU/GPU usage
Violation event table with timestamps
CSV export of violation history
Prometheus-compatible /metrics endpoint for Grafana integration
Installation
Prerequisites
Python 3.9+
NVIDIA GPU with CUDA support (recommended)
RTSP camera source (optional — webcam also supported)
Setup
# Clone the repository
git clone https://github.com/lequangvan2019-wq/YOLO_detection_model.git
cd YOLO_detection_model

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install ultralytics opencv-python PyQt6 flask
pip install paho-mqtt psutil GPUtil pynvml

Usage
1. Train the Model (one-time)
python "YOlO training .py"


This downloads the Roboflow Safety Vests v14 dataset and trains a YOLOv11n model. The trained weights (best.pt) are saved automatically.

2. Start the Web Dashboard
python mqtt_web_dashboard.py

Dashboard: http://localhost:5000
Prometheus metrics: GET /metrics
CSV download: GET /api/violations.csv
3. Run the Detection GUI
python "GUI_Vest Detection.py"

Select camera source (webcam index or RTSP URL)
Browse and load your trained model (best.pt)
Click START to begin real-time detection
Model Training
Configuration
Parameter	Value
Base Model	YOLOv11n (nano)
Dataset	Safety Vests v14 (Roboflow)
Image Size	576 × 576
Epochs	50
Classes	safety_vest, no_safety_vest
Model Performance
Confusion Matrix
	Predicted: no_safety_vest	Predicted: safety_vest	Predicted: background
True: no_safety_vest	316	13	23
True: safety_vest	21	1,164	102
True: background	68	189	—
Normalized Accuracy
Class	Precision
no_safety_vest	90%
safety_vest	90%

The model achieves 90% accuracy on both classes with low false positive rates, making it reliable for real-time safety monitoring.

Web Dashboard

The Flask-based dashboard provides real-time monitoring:

KPI Cards: People in frame, compliant count, violation count, FPS, CPU/GPU usage, latency
Violation Event Table: Timestamped log of all compliance state transitions
System Performance: CPU, GPU, VRAM, frame processing time
CSV Export: Download full violation history
Prometheus Endpoint: /metrics for Grafana/alerting integration
Nokia 5G Camera Integration
Camera Control (camera_osc_client.py)

Controls the Nokia 5G 360° camera via OSC protocol:

Set resolution (8K / 6K / 4K)
Start/stop RTSP live streaming
Enable/disable on-camera AI features
Monitor battery level and temperature
Control privacy features (face blur)
Stream Management (rxrm_api_client.py)

Interfaces with Nokia's RXRM (Real-time eXtended Reality Media) platform:

JWT-authenticated login
Discovers connected cameras
Creates 4 directional viewports from 360° feed (Front 0°, Right 90°, Back 180°, Left 270°)
Publishes each viewport as an independent RTSP stream
Supports custom viewport parameters (azimuth, elevation, FoV)
API Reference
MQTT Telemetry Payload

Published every 1 second to safetyvest/status:

{
  "people_in_frame": 5,
  "compliant": 3,
  "violated": 2,
  "cpu_percent": 45.2,
  "gpu_percent": 78.1,
  "gpu_memory_percent": 62.3,
  "fps": 24.5,
  "frame_processing_ms": 41.2,
  "source_timestamp": "2026-02-15T14:30:00.123",
  "source_epoch_ms": 1771234567890,
  "camera_source": "rtsp://192.168.1.100:554/live"
}

Dashboard REST API
Endpoint	Method	Description
/	GET	Web dashboard UI
/api/state	GET	Current system state (JSON)
/api/update	POST	Push detection results from GUI
/api/violations.csv	GET	Download violation event log
/metrics	GET	Prometheus-format metrics
Performance Benchmarks

Tested on NVIDIA GPU with CUDA:

Metric	Value
Inference Latency	< 250ms end-to-end
FPS (GPU)	20–30 FPS
FPS (CPU)	5–10 FPS
MQTT Publish Rate	1 message/second
Dashboard Refresh	1 second polling
RTSP Buffer	1 frame (minimal latency)
Key Design Decisions
Dual-model pipeline — Separates person detection (YOLOv8n) from vest classification (YOLOv11n) for higher accuracy than single-model approaches
Threaded RTSP capture — Decouples network I/O from inference to prevent frame blocking
Frame dropping strategy — Always processes the most recent frame, prioritizing latency over completeness for real-time safety
Smooth bounding boxes — Exponential moving average (α=0.85) prevents visual flickering common in frame-by-frame detection
Violation transition logging — Only logs when state changes from compliant → violated (avoids log spam)
FP16 inference — Halves GPU memory usage and nearly doubles throughput with negligible accuracy loss
Dual output channels — MQTT for IoT integration + HTTP for local dashboard, ensuring data reaches both cloud and local monitoring
Tech Stack
Category	Technology
AI/ML	YOLOv8, YOLOv11, PyTorch, Ultralytics
Computer Vision	OpenCV
GUI	PyQt6
Web Dashboard	Flask, HTML/CSS/JS
IoT Protocol	MQTT over TLS (HiveMQ Cloud)
Camera Integration	Nokia RXRM API, OSC Protocol
Monitoring	Prometheus, psutil, GPUtil/pynvml
GPU	CUDA, cuDNN, FP16 half-precision
License

This project was developed as part of an engineering internship at Nokia 5G Futures Lab (Nov 2025 – Feb 2026).

Author

Quang Van Le
Bachelor of Engineering (Electrical) — Western Sydney University
Nokia 5G Futures Lab Intern
