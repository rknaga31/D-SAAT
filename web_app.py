"""
D-SAAT — Dedicated Local Web Server & Cockpit HUD (FastAPI + HTML5/CSS/JS)

Runs entirely locally without Streamlit!
Launches the full multimodal driver safety pipeline and serves an ultra-low latency,
dark obsidian cockpit interface on http://localhost:8000.

Usage:
  python web_app.py
  python web_app.py --port 8080
  python web_app.py --demo
"""

import os
import sys
import time
import math
import argparse
import threading
import webbrowser
from pathlib import Path
from typing import Generator, Optional

import base64
from pydantic import BaseModel
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.utils.buffer import SharedState
from src.utils.logger import get_logger, configure_from_config
from src.capture.video_capture import VideoCapture
from src.capture.audio_capture import AudioCapture
from src.features.visual_features import VisualFeatureExtractor
from src.features.rppg_features import RPPGExtractor
from src.features.audio_features import AudioFeatureExtractor
from src.features.smartwatch_features import SmartwatchFeatureExtractor
from src.fusion.multimodal_fusion import MultimodalFusion
from src.prediction.risk_scorer import RiskScorer
from src.alert.alert_manager import AlertManager

log = get_logger("D-SAAT.Web")

# ── Shared State & Global Flags ───────────────────────────────────────────────
SHARED_STATE = SharedState()
_shutdown = threading.Event()
_DEMO_MODE = False
_PIPELINE_STARTED = False

app = FastAPI(title="D-SAAT Cockpit", docs_url=None, redoc_url=None)

# Mount static web directory for CSS and JS
web_dir = _ROOT / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


# ── Demo Telemetry & Synthetic Frame Generator ────────────────────────────────
def get_demo_snapshot() -> dict:
    t = time.time() % 60
    # Natural cyclic drowsiness oscillation
    score = 0.12 + 0.32 * abs(math.sin(t / 16.0))
    rppg_hr = 70.0 + 10.0 * math.sin(t / 14.0)
    watch_hr = rppg_hr + 1.2 * math.cos(t / 5.0)
    cv_diff = abs(rppg_hr - watch_hr)
    cv_status = "AGREE" if cv_diff < 4.0 else "MODERATE"

    level = 0 if score < 0.25 else 1 if score < 0.50 else 2 if score < 0.75 else 3
    labels = ["SAFE", "WARNING", "DANGER", "CRITICAL"]
    colors = ["#10b981", "#f59e0b", "#f97316", "#ef4444"]
    messages = [
        "Nominal driver alertness maintained. All systems nominal.",
        "Early ocular drowsiness signs detected. Stay attentive.",
        "Significant microsleep fatigue detected! Prepare to rest.",
        "EMERGENCY: Immediate driver intervention required!"
    ]

    return {
        "demo_mode": True,
        "pipeline_running": True,
        "session_duration": time.time() % 3600,
        "actual_fps": 30.0,
        "face_detected": True,
        "ear": round(0.28 - 0.08 * abs(math.sin(t / 3.5)), 2),
        "mar": round(0.14 + 0.42 * max(0, math.sin(t / 10.0)), 2),
        "perclos": round(score * 0.35, 3),
        "blink_rate": round(15.0 + 3.0 * math.sin(t / 6.0), 1),
        "heart_rate": round(rppg_hr, 1),
        "hrv_rmssd": round(36.0 - 12.0 * score, 1),
        "breathing_rate": round(15.5 + 2.5 * math.sin(t / 20.0), 1),
        "visual_score": round(score * 0.88, 3),
        "rppg_score": round(score * 0.72, 3),
        "audio_score": round(score * 0.48, 3),
        "smartwatch_score": round(score * 0.62, 3),
        "smartwatch_hr": round(watch_hr, 1),
        "smartwatch_spo2": round(98.6 - 2.0 * score, 1),
        "smartwatch_stress": round(24.0 + 40.0 * score, 1),
        "smartwatch_hrv": round(42.0 - 16.0 * score, 1),
        "cross_val_status": cv_status,
        "fused_score": round(score, 3),
        "smoothed_score": round(score, 3),
        "alert_level": level,
        "alert_label": labels[level],
        "alert_color": colors[level],
        "alert_message": messages[level],
        "pitch": round(4.5 * math.sin(t / 8.0), 1),
        "yaw": round(8.0 * math.sin(t / 10.0), 1),
        "roll": round(2.5 * math.cos(t / 12.0), 1),
    }


def generate_synthetic_frame() -> bytes:
    """Renders a futuristic dark cockpit telemetry simulation frame."""
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Ambient deep cockpit gradient
    t = time.time()
    for y in range(h):
        val = int(8 + (y / h) * 16)
        frame[y, :] = (val, val + 2, val + 6)

    # Grid background lines
    for gx in range(0, w, 80):
        cv2.line(frame, (gx, 0), (gx, h), (18, 25, 40), 1)
    for gy in range(0, h, 60):
        cv2.line(frame, (0, gy), (w, gy), (18, 25, 40), 1)

    # Simulated driver facial bounding box
    cx = int(w / 2 + 15 * math.sin(t / 4.0))
    cy = int(h / 2 + 10 * math.cos(t / 5.0))
    bw, bh = 180, 230
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2

    # Draw face box with neon cyan brackets
    cv2.rectangle(frame, (x1, y1), (x2, y2), (248, 189, 56), 1)
    k = 18
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + k, y1), (248, 189, 56), 3)
    cv2.line(frame, (x1, y1), (x1, y1 + k), (248, 189, 56), 3)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - k, y1), (248, 189, 56), 3)
    cv2.line(frame, (x2, y1), (x2, y1 + k), (248, 189, 56), 3)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + k, y2), (248, 189, 56), 3)
    cv2.line(frame, (x1, y2), (x1, y2 - k), (248, 189, 56), 3)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - k, y2), (248, 189, 56), 3)
    cv2.line(frame, (x2, y2), (x2, y2 - k), (248, 189, 56), 3)

    # Forehead rPPG region of interest
    rx1, ry1 = cx - 40, cy - 80
    rx2, ry2 = cx + 40, cy - 50
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (180, 50, 240), 1)
    cv2.putText(frame, "rPPG ROI", (rx1, ry1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 50, 240), 1)

    # Simulated eye landmark points
    eye_l = (cx - 35, cy - 20)
    eye_r = (cx + 35, cy - 20)
    cv2.circle(frame, eye_l, 4, (56, 189, 248), -1)
    cv2.circle(frame, eye_r, 4, (56, 189, 248), -1)

    # Top Alert Bar
    snap = get_demo_snapshot()
    level = snap["alert_level"]
    colors = [(16, 185, 129), (11, 158, 245), (22, 115, 249), (68, 68, 239)]  # BGR
    bar_color = colors[level]
    cv2.rectangle(frame, (0, 0), (w, 36), bar_color, -1)
    cv2.putText(frame, f"D-SAAT DEMO  |  ALERT: {snap['alert_label']}  |  RISK: {snap['smoothed_score']*100:.1f}%",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()


# ── Processing Pipeline Background Worker ─────────────────────────────────────
class ClientFramePayload(BaseModel):
    image: str


class WebPipelineWorker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.video_cap = None
        self.audio_cap = None
        self.proc_thread = None
        self.pipeline = None

    def start(self, no_audio: bool = False) -> bool:
        global _DEMO_MODE
        from main import ProcessingPipeline

        if not no_audio:
            self.audio_cap = AudioCapture(self.cfg)
            if not self.audio_cap.start():
                log.warning("Audio device unavailable. Running video-only.")
                self.audio_cap = None

        self.video_cap = VideoCapture(self.cfg)
        video_started = self.video_cap.start()

        # Always initialize pipeline so client browser frames can be processed immediately
        self.pipeline = ProcessingPipeline(
            self.cfg,
            self.video_cap if video_started else None,
            self.audio_cap,
            SHARED_STATE
        )

        if not video_started:
            log.warning("Webcam not accessible on host. Client browser webcam and Demo mode are active.")
            _DEMO_MODE = True
            return False

        self.proc_thread = threading.Thread(target=self.pipeline.run, name="WebProcessing", daemon=True)
        self.proc_thread.start()
        return True

    def stop(self):
        _shutdown.set()
        if self.video_cap:
            self.video_cap.stop()
        if self.audio_cap:
            self.audio_cap.stop()


worker: Optional[WebPipelineWorker] = None


def sanitize_telemetry(raw: dict) -> dict:
    """Sanitizes shared state dictionary into JSON-serializable primitives."""
    clean = {}
    for k, v in raw.items():
        if k in ("frame_jpg", "rppg_filtered"):
            continue
        if isinstance(v, (np.floating, float)):
            clean[k] = None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            clean[k] = bool(v)
        elif isinstance(v, (np.ndarray, bytes)):
            continue
        else:
            clean[k] = v
    return clean


# ── HTTP API Routes ───────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = web_dir / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/telemetry")
def get_telemetry():
    global _DEMO_MODE
    if _DEMO_MODE:
        return JSONResponse(get_demo_snapshot())

    snap = SHARED_STATE.snapshot()
    if not snap or not snap.get("pipeline_running"):
        return JSONResponse(get_demo_snapshot())

    clean_snap = sanitize_telemetry(snap)
    clean_snap["demo_mode"] = False
    return JSONResponse(clean_snap)


@app.post("/api/demo_mode")
def toggle_demo_mode():
    global _DEMO_MODE
    _DEMO_MODE = not _DEMO_MODE
    return JSONResponse({"demo_mode": _DEMO_MODE})


@app.post("/api/process_client_frame")
async def process_client_frame(payload: ClientFramePayload):
    """Processes a video frame streamed from the client's browser webcam."""
    global _DEMO_MODE, worker
    if worker is None or worker.pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)

    try:
        data_str = payload.image
        if "," in data_str:
            data_str = data_str.split(",", 1)[1]
        img_bytes = base64.b64decode(data_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"error": "Invalid image payload"}, status_code=400)

        # Process frame through full D-SAAT AI pipeline
        telemetry_update, annotated_bgr = worker.pipeline.process_single_frame(frame)
        _DEMO_MODE = False

        # Encode annotated frame with face mesh / reticle overlay
        _, jpg = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("utf-8")

        # Direct telemetry from this processed client frame
        clean_snap = sanitize_telemetry(telemetry_update)
        clean_snap["demo_mode"] = False

        # Acute eye closure / microsleep guarantee
        if clean_snap.get("eye_closed") or (0.0 < (clean_snap.get("ear") or 0.0) < 0.24):
            clean_snap["alert_level"] = 3
            clean_snap["alert_label"] = "CRITICAL"
            clean_snap["alert_message"] = "MICROSLEEP ALERT: Driver eyes closed!"
            clean_snap["smoothed_score"] = max(clean_snap.get("smoothed_score") or 0.0, 0.85)

        return JSONResponse({
            "status": "ok",
            "telemetry": clean_snap,
            "annotated_frame": annotated_b64
        })
    except Exception as exc:
        log.error(f"Error processing client frame: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


def frame_stream_generator() -> Generator[bytes, None, None]:
    """Generates continuous MJPEG multipart stream for <img> tag."""
    while not _shutdown.is_set():
        if _DEMO_MODE:
            frame_bytes = generate_synthetic_frame()
        else:
            frame_bytes = SHARED_STATE.get("frame_jpg")
            if frame_bytes is None:
                frame_bytes = generate_synthetic_frame()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033)  # ~30 FPS


@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(
        frame_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── Launcher ──────────────────────────────────────────────────────────────────
def launch_server(host: str = "0.0.0.0", port: int = 8000, demo: bool = False,
                  no_audio: bool = False, open_browser: bool = True):
    global _DEMO_MODE, worker
    _DEMO_MODE = demo

    # Load configuration
    import yaml
    cfg_path = _ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    configure_from_config(cfg)

    # Initialize pipeline worker (ready for both hardware and client browser camera)
    worker = WebPipelineWorker(cfg)
    if not demo:
        worker.start(no_audio=no_audio)
    else:
        from main import ProcessingPipeline
        worker.pipeline = ProcessingPipeline(cfg, None, None, SHARED_STATE)

    url = f"http://{host}:{port}" if host != "0.0.0.0" else f"http://localhost:{port}"
    log.info("=" * 60)
    log.info(f"  D-SAAT Cockpit is LIVE at: {url}")
    log.info(f"  Mode: {'DEMO SIMULATION' if _DEMO_MODE else 'HARDWARE CAMERA & SENSORS'}")
    log.info("  Press Ctrl+C to stop.")
    log.info("=" * 60)

    # Don't pop up browser if headless or explicitly disabled
    is_headless = os.environ.get("HEADLESS", "0") == "1"
    if open_browser and not is_headless:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown.set()
        if worker:
            worker.stop()
        log.info("D-SAAT server stopped cleanly.")


if __name__ == "__main__":
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", 8000))

    parser = argparse.ArgumentParser(description="D-SAAT Localhost Web Server")
    parser.add_argument("--host", default=default_host, help=f"Host address (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port (default: {default_port})")
    parser.add_argument("--demo", action="store_true", help="Start directly in demo mode")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio capture")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    launch_server(
        host=args.host,
        port=args.port,
        demo=args.demo,
        no_audio=args.no_audio,
        open_browser=not args.no_browser
    )
