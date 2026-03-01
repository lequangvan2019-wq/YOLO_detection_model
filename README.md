# 🦺 Real-Time AI Safety Vest Detection System

A real-time PPE (Personal Protective Equipment) compliance monitoring system using dual YOLO models, DeepSORT tracking, and MQTT communication. Built for construction site safety monitoring with Nokia 5G infrastructure integration.

![System Architecture](docs/architecture.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Dashboard](#web-dashboard)
- [API Reference](#api-reference)
- [Performance](#performance)
- [License](#license)

---

## Overview

This system detects whether workers on construction sites are wearing safety vests in real-time. It uses a **dual-model inference pipeline** — YOLOv8 for person detection and YOLOv11 for vest classification — combined with DeepSORT tracking for persistent identity monitoring.

The system supports both local webcams and **RTSP streams** (e.g., Nokia 360° cameras), making it suitable for deployment on 5G-connected edge devices.

---

## System Architecture

```
┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
│  INPUT SOURCES  │     │    VIDEO PIPELINE        │     │   AI INFERENCE      │
│                 │     │                          │     │                     │
│ Nokia 360° Cam ─┼──── │ Multi-threaded Capture   │──── │ YOLOv8 (Person)     │
│ (RTSP Stream)   │     │ Producer-Consumer Queue  │     │ YOLOv11 (Vest)      │
│                 │     │ Frame Dropping Strategy  │     │ FP16 GPU Inference  │
│                ─┼──── │ Buffer Size = 1          │     │ DeepSORT Tracking   │
│ Local Webcam    │     │                          │     │ Smooth Bounding Box │
└─────────────────┘     └──────────────────────────┘     └────────┬────────────┘
                                                                  │
                         ┌────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                      │
│                                                                             │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐   │
│  │ PyQt6 GUI    │   │ MQTT over TLS    │   │ Flask Web Dashboard        │   │
│  │ Live Video   │   │ HiveMQ Cloud     │   │ Real-time Metrics          │   │
│  │ Stats Panel  │   │ Status Publish   │   │ Violation Event Log        │   │
│  │ Tracking Info│   │                  │   │ Prometheus /metrics        │   │
│  └──────────────┘   └──────────────────┘   │ CSV Export                 │   │
│                                             └────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Violation Event Logging (CSV) + Auto Image Capture                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 🎯 Detection & Tracking
- **Dual-model inference**: YOLOv8 for person detection + YOLOv11 for vest classification
- **DeepSORT tracking**: Persistent identity tracking across frames
- **Smooth bounding boxes**: Exponential smoothing eliminates box flickering
- **Vest-person association**: Geometric overlap matching (IoU-based)

### ⚡ Performance Optimization
- **FP16 half-precision inference** on CUDA GPUs (~2x speedup)
- **Multi-threaded RTSP capture** with producer-consumer architecture
- **Intelligent frame dropping** — always processes the most recent frame
- **Periodic GPU memory cleanup** (every 100 frames)
- **Buffer size = 1** to minimize RTSP stream latency

### 📡 Communication & Monitoring
- **MQTT over TLS** (HiveMQ Cloud) for real-time status publishing (this one is to control the robot (not included in this code))
- **Flask web dashboard** with live metrics and violation event log
- **Prometheus-compatible `/metrics` endpoint** for Grafana integration
- **REST API** (`POST /api/update`) for system state updates
- **CSV violation logging** with automatic export

### 🖥️ User Interface
- **PyQt6 desktop GUI** with dark theme
- Real-time video display with tracking overlays
- System performance monitoring (CPU, GPU, VRAM, FPS)
- RTSP connection testing and preset management
- Auto image capture on violation detection

## Project Structure

```
safety-vest-detection/
├── GUI_Vest Detection.py      # Main application — PyQt6 GUI + AI inference
├── mqtt_web_dashboard.py      # Flask web dashboard + REST API + Prometheus
├── YOlO training.py           # Model training script (YOLOv11)
├── confusion_matrix.png       # Model validation results
├── confusion_matrix_normalized.png
├── captured_images/           # Auto-saved violation images
├── violation_events.csv       # Logged violation events
├── docs/
│   └── architecture.png       # System architecture diagram
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- RTSP camera source (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/safety-vest-detection.git
cd safety-vest-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python PyQt6 flask
pip install deep-sort-realtime paho-mqtt psutil GPUtil pynvml
```

---

## Usage

### 1. Run the Detection GUI

```bash
python "GUI_Vest Detection.py"
```

- Select camera source (webcam or RTSP)
- Browse and load your trained YOLO model (`.pt` file)
- Click **START** to begin real-time detection

### 2. Run the Web Dashboard

```bash
python mqtt_web_dashboard.py
```

- Dashboard: [http://localhost:5000](http://localhost:5000)
- API endpoint: `POST /api/update`
- Prometheus metrics: `GET /metrics`
- CSV download: `GET /api/violations.csv`

### 3. RTSP Stream (Nokia 360° Camera)

```
rtsp://192.168.1.100:554/live
```

Enable the **RTSP** checkbox in the GUI and enter the stream URL.

---

## Model Training

The vest detection model is trained on the [Roboflow Safety Vests dataset](https://universe.roboflow.com/roboflow-universe-projects/safety-vests) using YOLOv11.

```bash
python "YOlO training.py"
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv11n |
| Dataset | Safety Vests v14 (Roboflow) |
| Image Size | 576×576 |
| Epochs | 50 |
| Classes | `safety_vest`, `no_safety_vest` |

### Model Performance

| Metric | Value |
|--------|-------|
| safety_vest True Positives | 1,164 |
| no_safety_vest True Positives | 316 |
| Overall Accuracy | High precision with low false positives |

---

## Web Dashboard

The Flask-based web dashboard provides:

- **Real-time KPI cards**: People in frame, compliant count, violation count, FPS, CPU/GPU usage, latency
- **Violation event table**: Timestamped log of all violation transitions
- **System performance panel**: CPU, GPU, VRAM, frame processing time
- **CSV export**: Download full violation history
- **Prometheus endpoint**: `/metrics` for Grafana/alerting integration

### Dashboard API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard UI |
| `/api/state` | GET | Current system state (JSON) |
| `/api/update` | POST | Push detection results |
| `/api/violations.csv` | GET | Download violation log |
| `/metrics` | GET | Prometheus metrics |

---

## Performance

Tested on NVIDIA GPU with CUDA:

| Metric | Value |
|--------|-------|
| Inference Latency | < 250ms end-to-end |
| FPS (GPU) | 20-30 FPS |
| FPS (CPU) | 5-10 FPS |
| MQTT Publish Rate | 1 message/second |
| Dashboard Refresh | 1 second polling |

---

## Key Design Decisions

1. **Threaded RTSP capture** — Decouples network I/O from inference to prevent blocking
2. **Frame dropping** — Prioritizes latency over completeness for real-time safety
3. **Dual-model pipeline** — Separates person detection from vest classification for accuracy
4. **Smooth bounding boxes** — Exponential moving average prevents visual flickering
5. **Violation transition logging** — Only logs when state changes from compliant to violated (avoids spam)

---

## License

This project was developed as part of an engineering internship at **Nokia 5G Futures Lab**.

---

## Author

**Quang Van Le**  
Electrical Engineer — Western Sydney University  
Nokia 5G Futures Lab Intern (11-2025 – 2-2026)
