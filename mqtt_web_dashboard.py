"""HTTP dashboard for safety vest model output (no MQTT dependency).

Run:
    python mqtt_web_dashboard.py

Dashboard:
    http://localhost:5000

GUI sender endpoint:
    POST /api/update

Grafana/Prometheus scrape endpoint:
    GET /metrics
"""

from datetime import datetime
import csv
import os
import threading
import time

from flask import Flask, Response, jsonify, render_template_string, request, send_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLATION_CSV_PATH = os.path.join(BASE_DIR, "violation_events.csv")
MAX_EVENTS = 300

STATE_LOCK = threading.Lock()
STATE = {
    "people_in_frame": 0,
    "compliant": 0,
    "violated": 0,
    "cpu_percent": 0.0,
    "gpu_percent": 0.0,
    "gpu_memory_percent": 0.0,
    "fps": 0.0,
    "frame_processing_ms": 0.0,
    "source_timestamp": None,
    "source_epoch_ms": None,
    "camera_source": "",
    "updated_at": None,
    "dashboard_latency_ms": None,
    "last_violation_at": None,
    "violation_events": [],
    "total_violation_events": 0,
}


def _ensure_csv_header():
    if os.path.exists(VIOLATION_CSV_PATH):
        return
    with open(VIOLATION_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_time",
                "people_in_frame",
                "compliant",
                "violated",
                "cpu_percent",
                "gpu_percent",
                "gpu_memory_percent",
                "fps",
                "frame_processing_ms",
                "dashboard_latency_ms",
                "camera_source",
            ]
        )


def _append_violation_csv(event):
    _ensure_csv_header()
    with open(VIOLATION_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                event["event_time"],
                event["people_in_frame"],
                event["compliant"],
                event["violated"],
                event["cpu_percent"],
                event["gpu_percent"],
                event["gpu_memory_percent"],
                event["fps"],
                event["frame_processing_ms"],
                event["dashboard_latency_ms"],
                event["camera_source"],
            ]
        )


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Safety Vest Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel2: #1f2937;
      --line: #334155;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --ok: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 16px;
      font-family: "Segoe UI", Tahoma, Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top, #1e293b, var(--bg));
    }
    .container { max-width: 1200px; margin: 0 auto; display: grid; gap: 12px; }
    .panel {
      background: rgba(17,24,39,.92);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
    }
    h1 { margin: 0 0 8px 0; font-size: 24px; }
    .meta { color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 10px; }
    .cards { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; }
    .card { background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .value { font-size: 28px; font-weight: 700; line-height: 1; }
    .people { color: #93c5fd; } .ok { color: #86efac; } .bad { color: #fca5a5; }
    .perf { color: #fde68a; } .lat { color: #c4b5fd; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    .btn {
      background: #2563eb; color: white; border: none; border-radius: 8px;
      padding: 8px 12px; cursor: pointer; font-size: 13px;
    }
    .btn:hover { background: #1d4ed8; }
    .status-dot {
      display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 6px;
      background: var(--bad);
    }
    .status-dot.live { background: var(--ok); }
    @media (max-width: 980px) { .cards { grid-template-columns: repeat(3, minmax(120px, 1fr)); } .row { grid-template-columns: 1fr; } }
    @media (max-width: 640px) { .cards { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <div class="container">
    <div class="panel">
      <h1>Safety Vest Dashboard</h1>
      <div class="meta">
        <span><span id="liveDot" class="status-dot"></span><span id="liveText">No recent updates</span></span>
        <span>Last update: <strong id="updatedAt">-</strong></span>
        <span>Last violation: <strong id="lastViolation">-</strong></span>
        <span>Total violation events: <strong id="totalEvents">0</strong></span>
        <button class="btn" onclick="downloadCsv()">Download CSV</button>
      </div>
    </div>

    <div class="panel">
      <div class="cards">
        <div class="card"><div class="label">People In Frame</div><div id="people" class="value people">0</div></div>
        <div class="card"><div class="label">Compliant</div><div id="compliant" class="value ok">0</div></div>
        <div class="card"><div class="label">Violated</div><div id="violated" class="value bad">0</div></div>
        <div class="card"><div class="label">FPS</div><div id="fps" class="value perf">0</div></div>
        <div class="card"><div class="label">CPU / GPU %</div><div id="cpuGpu" class="value perf">0 / 0</div></div>
        <div class="card"><div class="label">Latency ms</div><div id="latency" class="value lat">0</div></div>
      </div>
    </div>

    <div class="row">
      <div class="panel">
        <div class="label" style="margin-bottom:8px;">System Performance</div>
        <table>
          <tbody>
            <tr><th>CPU %</th><td id="cpu">0</td></tr>
            <tr><th>GPU %</th><td id="gpu">0</td></tr>
            <tr><th>GPU Memory %</th><td id="vram">0</td></tr>
            <tr><th>Frame Processing ms</th><td id="procMs">0</td></tr>
            <tr><th>Dashboard Latency ms</th><td id="dashMs">0</td></tr>
            <tr><th>Camera Source</th><td id="cameraSource">-</td></tr>
          </tbody>
        </table>
      </div>
      <div class="panel">
        <div class="label" style="margin-bottom:8px;">Recent Violation Events</div>
        <table>
          <thead><tr><th>Time</th><th>People</th><th>Compliant</th><th>Violated</th></tr></thead>
          <tbody id="eventsBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    function setText(id, value) { document.getElementById(id).textContent = value; }

    function downloadCsv() {
      window.location.href = '/api/violations.csv';
    }

    function renderEvents(events) {
      const body = document.getElementById('eventsBody');
      body.innerHTML = '';
      if (!events || events.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="4">No violations logged yet</td>';
        body.appendChild(row);
        return;
      }
      events.slice().reverse().forEach(ev => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${ev.event_time}</td><td>${ev.people_in_frame}</td><td>${ev.compliant}</td><td>${ev.violated}</td>`;
        body.appendChild(row);
      });
    }

    async function refresh() {
      try {
        const res = await fetch('/api/state');
        const s = await res.json();
        setText('people', s.people_in_frame ?? 0);
        setText('compliant', s.compliant ?? 0);
        setText('violated', s.violated ?? 0);
        setText('fps', (s.fps ?? 0).toFixed(1));
        setText('cpuGpu', `${(s.cpu_percent ?? 0).toFixed(0)} / ${(s.gpu_percent ?? 0).toFixed(0)}`);
        setText('latency', (s.dashboard_latency_ms ?? 0).toFixed(1));
        setText('updatedAt', s.updated_at || '-');
        setText('lastViolation', s.last_violation_at || '-');
        setText('totalEvents', s.total_violation_events ?? 0);

        setText('cpu', (s.cpu_percent ?? 0).toFixed(2));
        setText('gpu', (s.gpu_percent ?? 0).toFixed(2));
        setText('vram', (s.gpu_memory_percent ?? 0).toFixed(2));
        setText('procMs', (s.frame_processing_ms ?? 0).toFixed(2));
        setText('dashMs', (s.dashboard_latency_ms ?? 0).toFixed(2));
        setText('cameraSource', s.camera_source || '-');

        const nowMs = Date.now();
        const updateMs = Date.parse(s.updated_at || '') || 0;
        const live = (nowMs - updateMs) < 5000;
        const dot = document.getElementById('liveDot');
        dot.classList.toggle('live', live);
        setText('liveText', live ? 'Live updates active' : 'No recent updates');

        renderEvents(s.violation_events || []);
      } catch (e) {
        setText('liveText', 'Dashboard connection issue');
      }
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/api/state", methods=["GET"])
def api_state():
    with STATE_LOCK:
        return jsonify(STATE)


@app.route("/api/update", methods=["POST"])
def api_update():
    data = request.get_json(silent=True) or {}
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    now_ms = int(time.time() * 1000)

    with STATE_LOCK:
        prev_violated = int(STATE.get("violated", 0))

        STATE["people_in_frame"] = int(data.get("people_in_frame", 0) or 0)
        STATE["compliant"] = int(data.get("compliant", 0) or 0)
        STATE["violated"] = int(data.get("violated", 0) or 0)
        STATE["cpu_percent"] = float(data.get("cpu_percent", 0.0) or 0.0)
        STATE["gpu_percent"] = float(data.get("gpu_percent", 0.0) or 0.0)
        STATE["gpu_memory_percent"] = float(data.get("gpu_memory_percent", 0.0) or 0.0)
        STATE["fps"] = float(data.get("fps", 0.0) or 0.0)
        STATE["frame_processing_ms"] = float(data.get("frame_processing_ms", 0.0) or 0.0)
        STATE["source_timestamp"] = data.get("source_timestamp")
        STATE["source_epoch_ms"] = data.get("source_epoch_ms")
        STATE["camera_source"] = str(data.get("camera_source", ""))
        STATE["updated_at"] = now_str

        source_epoch_ms = data.get("source_epoch_ms")
        if isinstance(source_epoch_ms, (int, float)):
            STATE["dashboard_latency_ms"] = max(0.0, float(now_ms - int(source_epoch_ms)))
        else:
            STATE["dashboard_latency_ms"] = None

        # Log only on transition into violation state.
        if prev_violated <= 0 and STATE["violated"] > 0:
            event = {
                "event_time": now_str,
                "people_in_frame": STATE["people_in_frame"],
                "compliant": STATE["compliant"],
                "violated": STATE["violated"],
                "cpu_percent": round(STATE["cpu_percent"], 2),
                "gpu_percent": round(STATE["gpu_percent"], 2),
                "gpu_memory_percent": round(STATE["gpu_memory_percent"], 2),
                "fps": round(STATE["fps"], 2),
                "frame_processing_ms": round(STATE["frame_processing_ms"], 2),
                "dashboard_latency_ms": round(STATE["dashboard_latency_ms"] or 0.0, 2),
                "camera_source": STATE["camera_source"],
            }
            STATE["last_violation_at"] = now_str
            STATE["violation_events"].append(event)
            if len(STATE["violation_events"]) > MAX_EVENTS:
                STATE["violation_events"] = STATE["violation_events"][-MAX_EVENTS:]
            STATE["total_violation_events"] += 1
            _append_violation_csv(event)

    return jsonify({"ok": True})


@app.route("/api/violations.csv", methods=["GET"])
def api_violations_csv():
    _ensure_csv_header()
    return send_file(
        VIOLATION_CSV_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="violation_events.csv",
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    with STATE_LOCK:
        lines = [
            "# HELP safety_people_in_frame Number of people in current frame",
            "# TYPE safety_people_in_frame gauge",
            f"safety_people_in_frame {STATE['people_in_frame']}",
            "# HELP safety_compliant Number of compliant people in current frame",
            "# TYPE safety_compliant gauge",
            f"safety_compliant {STATE['compliant']}",
            "# HELP safety_violated Number of violated people in current frame",
            "# TYPE safety_violated gauge",
            f"safety_violated {STATE['violated']}",
            "# HELP safety_cpu_percent CPU usage percentage",
            "# TYPE safety_cpu_percent gauge",
            f"safety_cpu_percent {STATE['cpu_percent']}",
            "# HELP safety_gpu_percent GPU usage percentage",
            "# TYPE safety_gpu_percent gauge",
            f"safety_gpu_percent {STATE['gpu_percent']}",
            "# HELP safety_gpu_memory_percent GPU memory usage percentage",
            "# TYPE safety_gpu_memory_percent gauge",
            f"safety_gpu_memory_percent {STATE['gpu_memory_percent']}",
            "# HELP safety_fps Processing FPS",
            "# TYPE safety_fps gauge",
            f"safety_fps {STATE['fps']}",
            "# HELP safety_frame_processing_ms Frame processing latency in ms",
            "# TYPE safety_frame_processing_ms gauge",
            f"safety_frame_processing_ms {STATE['frame_processing_ms']}",
            "# HELP safety_dashboard_latency_ms Dashboard receive latency in ms",
            "# TYPE safety_dashboard_latency_ms gauge",
            f"safety_dashboard_latency_ms {STATE['dashboard_latency_ms'] or 0.0}",
            "# HELP safety_total_violation_events Total number of violation events",
            "# TYPE safety_total_violation_events counter",
            f"safety_total_violation_events {STATE['total_violation_events']}",
        ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain; version=0.0.4")


if __name__ == "__main__":
    _ensure_csv_header()
    print("Starting dashboard on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
