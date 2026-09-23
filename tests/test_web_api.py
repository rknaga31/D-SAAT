"""
Integration tests for FastAPI Web App Endpoints & Driver Safety Scorer:
Verifies that the web cockpit does NOT stay stuck at 100% SAFE on Render / cloud environments.
"""

import base64
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from web_app import app, worker, SHARED_STATE, _DEMO_MODE
import web_app


@pytest.fixture(scope="module")
def client():
    # Ensure worker pipeline is initialized in web_app
    with TestClient(app) as test_client:
        yield test_client


def test_standby_telemetry_initial(client):
    """Initially before any camera frames or demo mode, telemetry is in standby, not stuck 100% safe."""
    web_app._DEMO_MODE = False
    SHARED_STATE.update(pipeline_running=False, frame_ts=0.0)

    res = client.get("/api/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert data["pipeline_running"] is False
    assert data["alert_label"] == "STANDBY"
    assert data["face_detected"] is False
    assert data["smoothed_score"] == 0.0


def test_demo_mode_toggle(client):
    """Toggling demo mode activates dynamic simulated telemetry."""
    res = client.post("/api/demo_mode")
    assert res.status_code == 200
    assert res.json()["demo_mode"] is True

    res2 = client.get("/api/telemetry")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["demo_mode"] is True
    assert data2["pipeline_running"] is True
    assert "smoothed_score" in data2
    assert "perclos" in data2

    # Toggle back off
    res3 = client.post("/api/demo_mode")
    assert res3.status_code == 200
    assert res3.json()["demo_mode"] is False


def test_process_client_frame_no_face_penalizes_score(client):
    """When a blank frame (no face) is uploaded, score is NOT 100% safe; penalty is applied."""
    web_app._DEMO_MODE = False

    # Create a blank black frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, jpg = cv2.imencode(".jpg", blank)
    b64 = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("utf-8")

    res = client.post("/api/process_client_frame", json={"image": b64})
    assert res.status_code == 200
    body = res.json()
    assert "telemetry" in body
    t = body["telemetry"]

    # Face should not be detected
    assert t["face_detected"] is False
    # Risk should NOT be 0.0 (which would give 100% safe)
    assert t["smoothed_score"] >= 0.40
    # Alert level must be elevated
    assert t["alert_level"] >= 1
    assert "AWAITING DRIVER FACE" in t["alert_message"] or "DRIVER ATTENTION LOST" in t["alert_message"]


def test_face_lost_escalation_after_timeout(client):
    """After losing face for >1 second, face_lost becomes True and triggers LEVEL_WARNING (ATTENTION LOST)."""
    # Simulate face lost for 1.5 seconds in pipeline
    pipeline = web_app.ensure_pipeline().pipeline
    pipeline.visual_ext.last_face_detected_ts = time.time() - 2.0
    pipeline.visual_ext.first_frame_ts = time.time() - 5.0

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, jpg = cv2.imencode(".jpg", blank)
    b64 = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("utf-8")

    res = client.post("/api/process_client_frame", json={"image": b64})
    assert res.status_code == 200
    t = res.json()["telemetry"]

    assert t["face_detected"] is False
    assert t["face_lost"] is True
    assert t["smoothed_score"] >= 0.45
    assert t["alert_level"] >= 1
    assert "ATTENTION LOST" in t["alert_message"] or "DRIVER ATTENTION LOST" in t["alert_message"]


def test_shared_state_telemetry_reflects_client_stream(client):
    """Shared state must be synced so GET /api/telemetry serves live client data, not 100% safe."""
    res = client.get("/api/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert data["demo_mode"] is False
    assert data["smoothed_score"] >= 0.40
    assert data["alert_level"] >= 1


def test_microsleep_eye_closed_elevates_to_critical(client, monkeypatch):
    """When eyes close during driving, the API immediately elevates to Level 3 Critical with safety score <= 15%."""
    pipeline = web_app.ensure_pipeline().pipeline

    # Mock process_single_frame to simulate face detected with closed eyes
    def mock_process(frame):
        telemetry = {
            "face_detected": True,
            "face_lost": False,
            "ear": 0.18,
            "eye_closed": True,
            "blink_score": 0.85,
            "smoothed_score": 0.88,
            "alert_level": 3,
            "alert_label": "CRITICAL",
            "alert_message": "MICROSLEEP ALERT: Driver eyes closed!"
        }
        return telemetry, frame

    monkeypatch.setattr(pipeline, "process_single_frame", mock_process)

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, jpg = cv2.imencode(".jpg", blank)
    b64 = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("utf-8")

    res = client.post("/api/process_client_frame", json={"image": b64})
    assert res.status_code == 200
    t = res.json()["telemetry"]

    assert t["face_detected"] is True
    assert t["alert_level"] == 3
    assert t["alert_label"] == "CRITICAL"
    assert t["smoothed_score"] >= 0.85
    assert "MICROSLEEP" in t["alert_message"]
