"""
VirtualPhysio-Driver — Clean Telemetry & Monitoring Dashboard

Reads from the SharedState populated by main.py's ProcessingPipeline
and renders a real-time, clean multimodal driver monitoring HUD.

Run with:
  # Terminal 1:  python main.py --headless
  # Terminal 2:  streamlit run dashboard/app.py
"""

import sys
import time
import math
from pathlib import Path
from collections import deque
from typing import Optional

# Add project root to Python path so we can import src.*
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import io

# ── Import shared state from main.py ──────────────────────────────────────────
try:
    from main import SHARED_STATE
    _LIVE_PIPELINE = True
except Exception:
    _LIVE_PIPELINE = False
    from src.utils.buffer import SharedState
    SHARED_STATE = SharedState()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VirtualPhysio-Driver · Cockpit Telemetry",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for Ultra-Clean Obsidian / Slate Telemetry UI ──────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  /* Global root resets & dark canvas */
  :root {
    --bg-base: #080c14;
    --bg-card: rgba(15, 23, 42, 0.72);
    --border-card: rgba(255, 255, 255, 0.08);
    --border-subtle: rgba(255, 255, 255, 0.04);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #38bdf8;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #ef4444;
    --accent-purple: #a855f7;
  }

  html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #080c14 !important;
    color: #f8fafc !important;
  }

  /* Remove default Streamlit top header whitespace */
  .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 98% !important;
  }

  /* Header Bar */
  .vp-appbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 15, 29, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 12px 22px;
    margin-bottom: 14px;
  }

  .vp-brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .vp-brand-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 2px 10px rgba(2, 132, 199, 0.35);
  }

  .vp-brand-title {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .vp-brand-subtitle {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 400;
  }

  /* Status Pill & Alert Capsule */
  .vp-status-capsule {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    transition: all 0.3s ease;
  }

  .status-safe {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
  }

  .status-warning {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
  }

  .status-danger {
    background: rgba(249, 115, 22, 0.15);
    border: 1px solid rgba(249, 115, 22, 0.35);
    color: #fb923c;
  }

  .status-critical {
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid rgba(239, 68, 68, 0.45);
    color: #f87171;
  }

  .dot-indicator {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
  }

  .dot-safe { background: #10b981; }
  .dot-warning { background: #f59e0b; }
  .dot-danger { background: #f97316; }
  .dot-critical { background: #ef4444; }

  /* ── Kill all Streamlit rerun opacity dimming / brightness flickering ── */
  div[data-testid="stAppViewContainer"],
  div[data-testid="stAppViewContainer"] *,
  div[data-testid="stVerticalBlock"] > div,
  div[data-testid="element-container"],
  .element-container,
  .stElementContainer,
  div[data-testid="stImage"],
  div.stPlotlyChart {
    opacity: 1 !important;
    transition: none !important;
    animation: none !important;
  }

  /* Hide Streamlit running man spinner in header to eliminate header twitch */
  div[data-testid="stStatusWidget"],
  .stStatusWidget {
    display: none !important;
    visibility: hidden !important;
  }

  /* Keep top header static and non-distracting */
  header[data-testid="stHeader"] {
    background: transparent !important;
  }

  /* Clean HUD Cards */
  .vp-card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .vp-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }

  .vp-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .vp-card-badge {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  /* Metric Pill Items (No word wrap bugs) */
  .vp-metric-tile {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  .vp-tile-label {
    font-size: 0.68rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .vp-tile-value {
    font-size: 1.25rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    color: #f8fafc;
    white-space: nowrap;
    line-height: 1.2;
  }

  .vp-tile-unit {
    font-size: 0.68rem;
    color: #64748b;
    margin-top: 1px;
    white-space: nowrap;
  }

  /* Progress Bars */
  .vp-progress-track {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 9999px;
    height: 6px;
    width: 100%;
    overflow: hidden;
    margin-top: 5px;
  }

  .vp-progress-bar {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
  }

  /* Streamlit Tabs Polish */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    padding-bottom: 4px;
    margin-top: 14px;
  }

  .stTabs [data-baseweb="tab"] {
    background-color: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    color: #94a3b8;
    padding: 8px 18px;
    font-size: 0.8rem;
    font-weight: 500;
  }

  .stTabs [aria-selected="true"] {
    background-color: rgba(56, 189, 248, 0.12) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    color: #38bdf8 !important;
  }

  /* Clean Scrollbars */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #080c14; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# ── Session history buffers ───────────────────────────────────────────────────

def _init_history():
    if "history" not in st.session_state:
        st.session_state.history = {
            "time":           deque(maxlen=300),
            "smoothed_score": deque(maxlen=300),
            "ear":            deque(maxlen=300),
            "mar":            deque(maxlen=300),
            "heart_rate":     deque(maxlen=300),
            "watch_hr":       deque(maxlen=300),
            "breathing_rate": deque(maxlen=300),
            "perclos":        deque(maxlen=300),
            "visual_score":   deque(maxlen=300),
            "rppg_score":     deque(maxlen=300),
            "audio_score":    deque(maxlen=300),
            "watch_score":    deque(maxlen=300),
            "watch_spo2":     deque(maxlen=300),
            "watch_stress":   deque(maxlen=300),
        }
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = not _LIVE_PIPELINE


_init_history()


def _update_history(snap: dict):
    """Append snapshot values to session history."""
    h = st.session_state.history
    t = time.time() - st.session_state.start_time
    h["time"].append(round(t, 1))
    h["smoothed_score"].append(snap.get("smoothed_score", 0.0))
    h["ear"].append(snap.get("ear", 0.0))
    h["mar"].append(snap.get("mar", 0.0))
    h["heart_rate"].append(snap.get("heart_rate", 0.0))
    h["watch_hr"].append(snap.get("smartwatch_hr", 0.0))
    h["breathing_rate"].append(snap.get("breathing_rate", 0.0))
    h["perclos"].append(snap.get("perclos", 0.0) * 100)
    h["visual_score"].append(snap.get("visual_score", 0.0))
    h["rppg_score"].append(snap.get("rppg_score", 0.0))
    h["audio_score"].append(snap.get("audio_score", 0.0))
    h["watch_score"].append(snap.get("smartwatch_score", 0.0))
    h["watch_spo2"].append(snap.get("smartwatch_spo2", 98.0))
    h["watch_stress"].append(snap.get("smartwatch_stress", 20.0))


# ── Demo / simulation mode ────────────────────────────────────────────────────

def _get_demo_snapshot() -> dict:
    """Generate realistic synthetic telemetry for demo mode."""
    t = time.time() % 60
    # Natural cyclic drowsiness oscillation
    score = 0.12 + 0.32 * abs(math.sin(t / 16.0))
    rppg_hr = 70.0 + 10.0 * math.sin(t / 14.0)
    watch_hr = rppg_hr + 1.2 * math.cos(t / 5.0)
    cv_diff = abs(rppg_hr - watch_hr)
    cv_status = "AGREE" if cv_diff < 4.0 else "MODERATE"

    return {
        "face_detected": True,
        "ear": 0.28 - 0.07 * abs(math.sin(t / 3.5)),
        "ear_left": 0.28, "ear_right": 0.27,
        "mar": 0.14 + 0.45 * max(0, math.sin(t / 10.0)),
        "perclos": score * 0.35,
        "blink_rate": 15.0 + 3.0 * math.sin(t / 6.0),
        "blink_count": int(t * 0.28),
        "eye_closed": score > 0.65,
        "yawn_detected_visual": score > 0.60,
        "yawn_count_visual": int(t / 18),
        "pitch": 4.5 * math.sin(t / 8.0),
        "yaw": 8.0 * math.sin(t / 10.0),
        "roll": 2.5 * math.cos(t / 12.0),
        "head_pose_alert": abs(8.0 * math.sin(t / 10.0)) > 24,
        "heart_rate": round(rppg_hr, 1),
        "hrv_rmssd": 36.0 - 12.0 * score,
        "rppg_quality": 0.92,
        "breathing_rate": 15.5 + 2.5 * math.sin(t / 20.0),
        "yawn_score_audio": score * 0.55,
        "audio_drowsiness": score * 0.45,
        "visual_score": score * 0.88,
        "rppg_score": score * 0.72,
        "audio_score": score * 0.48,
        "smartwatch_score": score * 0.62,
        "smartwatch_hr": round(watch_hr, 1),
        "smartwatch_spo2": round(98.6 - 2.0 * score, 1),
        "smartwatch_stress": round(24.0 + 40.0 * score, 1),
        "smartwatch_hrv": round(42.0 - 16.0 * score, 1),
        "smartwatch_connected": True,
        "cross_val_status": cv_status,
        "cross_val_diff": round(cv_diff, 1),
        "cross_val_confidence": 0.97,
        "fused_score": score,
        "smoothed_score": score,
        "w_visual": 0.30, "w_rppg": 0.20, "w_audio": 0.25, "w_smartwatch": 0.25,
        "alert_level": 0 if score < 0.25 else 1 if score < 0.50 else 2 if score < 0.75 else 3,
        "alert_label": "SAFE" if score < 0.25 else "WARNING" if score < 0.50 else "DANGER" if score < 0.75 else "CRITICAL",
        "alert_color": "#10b981" if score < 0.25 else "#f59e0b" if score < 0.50 else "#f97316" if score < 0.75 else "#ef4444",
        "alert_emoji": "●",
        "alert_message": "Nominal driver alertness maintained" if score < 0.25 else "Early micro-sleep signs detected" if score < 0.50 else "Significant ocular fatigue detected" if score < 0.75 else "EMERGENCY: Immediate driver intervention required",
        "session_duration": t,
        "drowsy_events": int(t / 24),
        "critical_events": 0,
        "score_trend": 0.01 * math.sin(t / 5.0),
        "actual_fps": 30.0,
        "pipeline_running": True,
        "frame_jpg": None,
    }


# ── Chart Styling & Helpers ───────────────────────────────────────────────────

CLEAN_CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.45)",
    font=dict(color="#94a3b8", family="Inter, sans-serif", size=11),
    margin=dict(l=38, r=16, t=28, b=28),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.04)",
        linecolor="rgba(255, 255, 255, 0.08)",
        tickfont=dict(color="#64748b", size=9),
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.04)",
        linecolor="rgba(255, 255, 255, 0.08)",
        tickfont=dict(color="#64748b", size=9),
        showgrid=True,
    ),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.06)",
        font=dict(color="#94a3b8", size=9),
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="right",
        x=1,
    ),
    height=210,
    uirevision="telemetry",
)


def _render_clean_gauge(score: float, label: str, color: str):
    """Render a clean, modern semi-circular radial gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={
            "suffix": "%",
            "font": {"color": "#f8fafc", "size": 32, "family": "JetBrains Mono"},
            "valueformat": ".1f"
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.1)",
                "tickfont": {"color": "#64748b", "size": 9},
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["0", "25", "50", "75", "100"],
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(30, 41, 59, 0.4)",
            "borderwidth": 1,
            "bordercolor": "rgba(255, 255, 255, 0.06)",
            "steps": [
                {"range": [0, 25], "color": "rgba(16, 185, 129, 0.12)"},
                {"range": [25, 50], "color": "rgba(245, 158, 11, 0.12)"},
                {"range": [50, 75], "color": "rgba(249, 115, 22, 0.15)"},
                {"range": [75, 100], "color": "rgba(239, 68, 68, 0.18)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", family="Inter"),
        margin=dict(l=24, r=24, t=20, b=10),
        height=175,
        uirevision="gauge",
    )
    return fig


def _render_clean_radar(visual: float, rppg: float, audio: float, smartwatch: float = 0.0):
    """Render clean radar chart with ample padding to prevent text clipping."""
    categories = ["Visual", "rPPG HR", "Audio", "Smartwatch"]
    values = [visual, rppg, audio, smartwatch]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line=dict(color="#38bdf8", width=2),
        fillcolor="rgba(56, 189, 248, 0.18)",
        marker=dict(size=4, color="#38bdf8"),
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.6)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(255, 255, 255, 0.06)",
                linecolor="rgba(255, 255, 255, 0.06)",
                tickfont=dict(size=8, color="#64748b"),
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.08)",
                linecolor="rgba(255, 255, 255, 0.08)",
                tickfont=dict(size=10, color="#94a3b8", family="Inter"),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=45, t=35, b=25),
        height=210,
        showlegend=False,
        uirevision="radar",
    )
    return fig


def _make_spline_trace(times, values, name, color, fill=False):
    """Create a sleek line trace with soft gradients."""
    return go.Scatter(
        x=list(times),
        y=list(values),
        name=name,
        line=dict(color=color, width=2, shape="spline", smoothing=0.8),
        fill="tozeroy" if fill else "none",
        fillcolor=f"{color}14" if fill else None,
        mode="lines",
        hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
    )


def _generate_stylized_driver_feed():
    """Generates a clean, modern simulated cockpit visual HUD."""
    import cv2 as _cv2
    w, h = 480, 310
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Modern deep dark background with subtle cockpit horizon
    img[:] = (18, 14, 10)  # BGR
    
    # Subtle cockpit grid lines
    _cv2.line(img, (0, 200), (w, 200), (35, 28, 20), 1)
    _cv2.line(img, (w//2, 140), (w//2, h), (28, 22, 16), 1)

    # Clean driver head bounding box & target reticle
    cx, cy = w // 2, 130
    _cv2.rectangle(img, (cx - 75, cy - 75), (cx + 75, cy + 85), (60, 48, 30), 1)
    
    # Sleek cyan corner target brackets
    bracket_len = 16
    for x_corner, y_corner, dx, dy in [
        (cx - 75, cy - 75, 1, 1),
        (cx + 75, cy - 75, -1, 1),
        (cx - 75, cy + 85, 1, -1),
        (cx + 75, cy + 85, -1, -1),
    ]:
        _cv2.line(img, (x_corner, y_corner), (x_corner + dx * bracket_len, y_corner), (248, 189, 56), 2)
        _cv2.line(img, (x_corner, y_corner), (x_corner, y_corner + dy * bracket_len), (248, 189, 56), 2)

    # Stylized facial landmark points (eyes, nose, mouth)
    _cv2.circle(img, (cx - 30, cy - 20), 6, (248, 189, 56), 1)
    _cv2.circle(img, (cx + 30, cy - 20), 6, (248, 189, 56), 1)
    _cv2.circle(img, (cx, cy + 10), 3, (160, 140, 100), -1)
    _cv2.ellipse(img, (cx, cy + 42), (24, 10), 0, 0, 180, (160, 211, 52), 2)

    # Telemetry HUD text
    _cv2.putText(img, "FACEMESH 468 LANDMARKS: LOCKED", (16, 26),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 180, 140), 1, _cv2.LINE_AA)
    _cv2.putText(img, "RPPG ROI: FOREHEAD + CHEEKS", (16, 46),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 100, 80), 1, _cv2.LINE_AA)
    _cv2.putText(img, "SYNTHETIC DRIVER FEED", (w - 175, h - 14),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 140, 200), 1, _cv2.LINE_AA)

    return Image.fromarray(_cv2.cvtColor(img, _cv2.COLOR_BGR2RGB))


# ── Main Render Entry Point ───────────────────────────────────────────────────

def render():
    # ── Refresh state ──
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 1000

    # Auto-refresh with calm, flicker-free cadence
    if st.session_state.auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=st.session_state.refresh_interval, key="vpd_clean_refresh")
        except ImportError:
            pass

    # ── Fetch Telemetry Snapshot ──────────────────────────────────────────────
    demo = st.session_state.demo_mode
    if demo:
        snap = _get_demo_snapshot()
    else:
        snap = SHARED_STATE.snapshot()
        if not snap or not snap.get("pipeline_running"):
            st.session_state.demo_mode = True
            snap = _get_demo_snapshot()
            demo = True

    _update_history(snap)
    h = st.session_state.history

    level = snap.get("alert_level", 0)
    label = snap.get("alert_label", "SAFE")
    score = snap.get("smoothed_score", 0.0)
    color = snap.get("alert_color", "#10b981")
    alert_msg = snap.get("alert_message", "Driver Alertness Normal")

    # Map status class
    status_classes = ["status-safe", "status-warning", "status-danger", "status-critical"]
    dot_classes = ["dot-safe", "dot-warning", "dot-danger", "dot-critical"]
    status_cls = status_classes[min(level, 3)]
    dot_cls = dot_classes[min(level, 3)]

    # ── Top Application Bar ───────────────────────────────────────────────────
    dur = snap.get("session_duration", 0.0)
    mins, secs = divmod(int(dur), 60)
    fps = snap.get("actual_fps", 30.0)

    mode_badge = (
        '<span style="background:rgba(56, 189, 248, 0.12); color:#38bdf8; border:1px solid rgba(56,189,248,0.25);'
        ' padding:3px 10px; border-radius:9999px; font-size:0.72rem; font-weight:600;">LIVE FEED</span>'
        if not demo else
        '<span style="background:rgba(245, 158, 11, 0.12); color:#fbbf24; border:1px solid rgba(245,158,11,0.25);'
        ' padding:3px 10px; border-radius:9999px; font-size:0.72rem; font-weight:600;">DEMO SIMULATION</span>'
    )

    st.markdown(f"""
    <div class="vp-appbar">
      <div class="vp-brand">
        <div class="vp-brand-icon">🚘</div>
        <div>
          <div class="vp-brand-title">
            <span>VirtualPhysio</span>
            <span style="color:#38bdf8; font-weight:400;">Driver</span>
            {mode_badge}
          </div>
          <div class="vp-brand-subtitle">
            Multimodal Edge Physiological Telemetry · Real-Time Micro-Sleep Guard
          </div>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:16px;">
        <div style="text-align:right; font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#94a3b8;">
          SESSION <span style="color:#f8fafc; font-weight:600;">{mins:02d}:{secs:02d}</span>
          &nbsp;·&nbsp; <span style="color:#f8fafc; font-weight:600;">{fps:.1f}</span> FPS
        </div>
        <div class="vp-status-capsule {status_cls}">
          <span class="dot-indicator {dot_cls}"></span>
          <span>{label}</span>
          <span style="opacity:0.65;">·</span>
          <span>{score * 100:.1f}%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Cockpit HUD Grid (3 Balanced Columns) ─────────────────────────────────
    col_vision, col_risk, col_vitals = st.columns([1.1, 1.0, 1.1], gap="medium")

    # ── Column 1: Driver Vision Feed & Ocular Dynamics ────────────────────────
    with col_vision:
        st.markdown("""
        <div class="vp-card">
          <div class="vp-card-header">
            <div class="vp-card-title">Driver Vision Stream</div>
            <div class="vp-card-badge" style="background:rgba(16,185,129,0.12); color:#34d399;">
              FaceMesh Active
            </div>
          </div>
        """, unsafe_allow_html=True)

        frame_jpg = snap.get("frame_jpg")
        if frame_jpg and not demo:
            img = Image.open(io.BytesIO(frame_jpg))
            st.image(img, use_container_width=True)
        else:
            sim_img = _generate_stylized_driver_feed()
            st.image(sim_img, use_container_width=True)

        # 4 Ocular / Behavioral Metric Tiles
        ear = snap.get("ear", 0.28)
        mar = snap.get("mar", 0.15)
        perclos = snap.get("perclos", 0.0) * 100
        pitch = snap.get("pitch", 0.0)

        ear_alert = ear < 0.21
        mar_alert = mar > 0.65
        perclos_alert = perclos > 15.0

        st.markdown(f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:10px;">
          <div class="vp-metric-tile">
            <div class="vp-tile-label">Eye Aspect (EAR)</div>
            <div class="vp-tile-value" style="color:{'#f87171' if ear_alert else '#f8fafc'};">{ear:.3f}</div>
            <div class="vp-tile-unit">Threshold: 0.210</div>
          </div>
          <div class="vp-metric-tile">
            <div class="vp-tile-label">Mouth Aspect (MAR)</div>
            <div class="vp-tile-value" style="color:{'#f87171' if mar_alert else '#f8fafc'};">{mar:.3f}</div>
            <div class="vp-tile-unit">Yawn: &gt; 0.650</div>
          </div>
          <div class="vp-metric-tile">
            <div class="vp-tile-label">PERCLOS</div>
            <div class="vp-tile-value" style="color:{'#fbbf24' if perclos_alert else '#f8fafc'};">{perclos:.1f}%</div>
            <div class="vp-tile-unit">Eye closure ratio</div>
          </div>
          <div class="vp-metric-tile">
            <div class="vp-tile-label">Head Pitch</div>
            <div class="vp-tile-value">{pitch:+.1f}&deg;</div>
            <div class="vp-tile-unit">Nodding angle</div>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Column 2: Central Drowsiness Risk Engine ──────────────────────────────
    with col_risk:
        st.markdown(f"""
        <div class="vp-card">
          <div class="vp-card-header">
            <div class="vp-card-title">Drowsiness Risk Engine</div>
            <div class="vp-card-badge {status_cls}">
              Level {level} · {label}
            </div>
          </div>
        """, unsafe_allow_html=True)

        # Clean Radial Gauge
        st.plotly_chart(_render_clean_gauge(score, "Fused Risk", color),
                        use_container_width=True, config={"displayModeBar": False})

        # Alert Advisory Message Box
        st.markdown(f"""
        <div style="background:rgba(30, 41, 59, 0.45); border-left:3px solid {color}; border-radius:6px;
                    padding:8px 12px; margin-bottom:12px; font-size:0.75rem; color:#e2e8f0;">
          <b>Advisory:</b> {alert_msg}
        </div>
        """, unsafe_allow_html=True)

        # 4 Modality Progress Meters
        vs = snap.get("visual_score", 0.0)
        rs = snap.get("rppg_score", 0.0)
        aus = snap.get("audio_score", 0.0)
        ws = snap.get("smartwatch_score", 0.0)

        modalities = [
            ("Visual Ocular", vs, snap.get("w_visual", 0.30), "#38bdf8"),
            ("rPPG Contactless", rs, snap.get("w_rppg", 0.20), "#f43f5e"),
            ("Acoustic / Audio", aus, snap.get("w_audio", 0.25), "#34d399"),
            ("Smartwatch PPG", ws, snap.get("w_smartwatch", 0.25), "#a855f7"),
        ]

        st.markdown('<div style="font-size:0.7rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Modality Contributions</div>', unsafe_allow_html=True)

        for name, m_score, weight, hex_col in modalities:
            st.markdown(f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#cbd5e1;">
                <span>{name} <span style="color:#64748b; font-size:0.66rem;">(w={weight:.2f})</span></span>
                <span style="font-family:'JetBrains Mono', monospace; font-weight:600;">{m_score * 100:.1f}%</span>
              </div>
              <div class="vp-progress-track">
                <div class="vp-progress-bar" style="background:{hex_col}; width:{min(100, max(2, m_score * 100)):.1f}%;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Column 3: Smartwatch & Physiological Vitals ───────────────────────────
    with col_vitals:
        cv_status = snap.get("cross_val_status", "AGREE")
        cv_diff = snap.get("cross_val_diff", 0.0)
        cv_colors = {
            "AGREE": ("#10b981", "rgba(16, 185, 129, 0.12)"),
            "MODERATE": ("#f59e0b", "rgba(245, 158, 11, 0.12)"),
            "DISAGREE": ("#ef4444", "rgba(239, 68, 68, 0.15)"),
            "WATCH ONLY": ("#38bdf8", "rgba(56, 189, 248, 0.12)"),
            "CAMERA ONLY": ("#a855f7", "rgba(168, 85, 247, 0.12)"),
        }
        text_c, bg_c = cv_colors.get(cv_status, ("#94a3b8", "rgba(148, 163, 184, 0.12)"))

        st.markdown(f"""
        <div class="vp-card">
          <div class="vp-card-header">
            <div class="vp-card-title">Physiological Vitals</div>
            <div class="vp-card-badge" style="background:{bg_c}; color:{text_c}; border:1px solid {text_c}40;">
              HR Cross-Val: {cv_status} (&Delta;{cv_diff:.1f})
            </div>
          </div>
        """, unsafe_allow_html=True)

        watch_hr = snap.get("smartwatch_hr", 0.0)
        rppg_hr = snap.get("heart_rate", 0.0)
        spo2 = snap.get("smartwatch_spo2", 98.0)
        stress = snap.get("smartwatch_stress", 24.0)
        hrv = snap.get("smartwatch_hrv", 36.0)
        br = snap.get("breathing_rate", 15.5)

        # Dual Heart Rate Comparison Banner
        st.markdown(f"""
        <div style="background:rgba(30, 41, 59, 0.5); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:12px; margin-bottom:12px;">
          <div style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
            Dual Heart Rate Cross-Validation
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <div style="font-size:0.7rem; color:#a855f7; font-weight:600;">⌚ Smartwatch PPG</div>
              <div style="font-size:1.6rem; font-weight:700; font-family:'JetBrains Mono', monospace; color:#f8fafc;">
                {watch_hr:.0f} <span style="font-size:0.8rem; font-weight:400; color:#64748b;">BPM</span>
              </div>
            </div>
            <div style="height:36px; width:1px; background:rgba(255,255,255,0.08);"></div>
            <div>
              <div style="font-size:0.7rem; color:#f43f5e; font-weight:600;">📷 Camera rPPG</div>
              <div style="font-size:1.6rem; font-weight:700; font-family:'JetBrains Mono', monospace; color:#f8fafc;">
                {rppg_hr:.0f} <span style="font-size:0.8rem; font-weight:400; color:#64748b;">BPM</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Vital Stat Tiles
        st.markdown(f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
          <div class="vp-metric-tile">
            <div class="vp-tile-label">Blood Oxygen (SpO2)</div>
            <div class="vp-tile-value" style="color:{'#f87171' if spo2 < 94 else '#f8fafc'};">{spo2:.1f}%</div>
            <div class="vp-tile-unit">Pulse Oximetry</div>
          </div>
          <div class="vp-metric-tile">
            <div class="vp-tile-label">Respiration Rate</div>
            <div class="vp-tile-value">{br:.1f}</div>
            <div class="vp-tile-unit">Breaths / min</div>
          </div>
          <div class="vp-metric-tile">
            <div class="vp-tile-label">Heart Rate Var (HRV)</div>
            <div class="vp-tile-value">{hrv:.1f}</div>
            <div class="vp-tile-unit">RMSSD (ms)</div>
          </div>
          <div class="vp-metric-tile">
            <div class="vp-tile-label">Stress Index</div>
            <div class="vp-tile-value" style="color:{'#f87171' if stress > 75 else '#f8fafc'};">{stress:.0f}</div>
            <div class="vp-tile-unit">Normalized 0–100</div>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Secondary Telemetry & Analytics Tabs ──────────────────────────────────
    tab_charts, tab_radar, tab_intel = st.tabs([
        "📈 Live Telemetry Curves",
        "🔺 Modality Radar & Weights",
        "📋 Session Intelligence & Diagnostics"
    ])

    times = list(h["time"])

    with tab_charts:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown('<div class="vp-tile-label" style="font-size:0.76rem; color:#e2e8f0; font-weight:600; margin-bottom:2px;">Drowsiness Risk & Modality Breakdown</div>', unsafe_allow_html=True)
            # Fused Drowsiness Timeline
            fig_risk = go.Figure()
            fig_risk.add_trace(_make_spline_trace(times, h["smoothed_score"], "Fused Risk", "#f43f5e", fill=True))
            fig_risk.add_trace(_make_spline_trace(times, h["visual_score"], "Visual", "#38bdf8"))
            fig_risk.add_trace(_make_spline_trace(times, h["rppg_score"], "rPPG", "#ec4899"))
            fig_risk.add_trace(_make_spline_trace(times, h["audio_score"], "Audio", "#34d399"))
            fig_risk.add_trace(_make_spline_trace(times, h["watch_score"], "Smartwatch", "#a855f7"))
            fig_risk.update_layout(**CLEAN_CHART_THEME)
            fig_risk.update_yaxes(range=[0, 1.05], tickvals=[0, 0.25, 0.5, 0.75, 1.0])
            fig_risk.update_xaxes(title_text="Session Time (s)")
            st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="vp-tile-label" style="font-size:0.76rem; color:#e2e8f0; font-weight:600; margin-top:8px; margin-bottom:2px;">Ocular & Facial Dynamics (EAR / MAR)</div>', unsafe_allow_html=True)
            # EAR & MAR Dynamics
            fig_ocular = go.Figure()
            fig_ocular.add_trace(_make_spline_trace(times, h["ear"], "Eye Aspect Ratio (EAR)", "#38bdf8"))
            fig_ocular.add_trace(_make_spline_trace(times, h["mar"], "Mouth Aspect Ratio (MAR)", "#f59e0b"))
            if times:
                fig_ocular.add_hline(y=0.21, line=dict(color="#ef4444", dash="dash", width=1),
                                     annotation_text="EAR Threshold (0.21)", annotation_font=dict(color="#ef4444", size=9))
            fig_ocular.update_layout(**CLEAN_CHART_THEME)
            fig_ocular.update_xaxes(title_text="Session Time (s)")
            st.plotly_chart(fig_ocular, use_container_width=True, config={"displayModeBar": False})

        with c2:
            st.markdown('<div class="vp-tile-label" style="font-size:0.76rem; color:#e2e8f0; font-weight:600; margin-bottom:2px;">Dual Heart Rate Signal Tracking (BPM)</div>', unsafe_allow_html=True)
            # Dual Heart Rate Overlay (Watch vs Camera)
            fig_hr = go.Figure()
            fig_hr.add_trace(_make_spline_trace(times, h["watch_hr"], "⌚ Smartwatch PPG", "#a855f7"))
            fig_hr.add_trace(_make_spline_trace(times, h["heart_rate"], "📷 Camera rPPG", "#f43f5e"))
            if times:
                fig_hr.add_hrect(y0=60, y1=100, fillcolor="rgba(56, 189, 248, 0.05)",
                                line_width=0, annotation_text="Resting Range", annotation_font_color="#38bdf8")
            fig_hr.update_layout(**CLEAN_CHART_THEME)
            fig_hr.update_yaxes(title_text="Heart Rate (BPM)")
            fig_hr.update_xaxes(title_text="Session Time (s)")
            st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="vp-tile-label" style="font-size:0.76rem; color:#e2e8f0; font-weight:600; margin-top:8px; margin-bottom:2px;">Respiration, Stress & PERCLOS Trends</div>', unsafe_allow_html=True)
            # Respiration & PERCLOS
            fig_resp = go.Figure()
            fig_resp.add_trace(_make_spline_trace(times, h["perclos"], "PERCLOS (%)", "#fbbf24", fill=True))
            fig_resp.add_trace(_make_spline_trace(times, h["breathing_rate"], "Breathing (rpm)", "#34d399"))
            fig_resp.add_trace(_make_spline_trace(times, h["watch_stress"], "Stress (0-100)", "#60a5fa"))
            fig_resp.update_layout(**CLEAN_CHART_THEME)
            fig_resp.update_xaxes(title_text="Session Time (s)")
            st.plotly_chart(fig_resp, use_container_width=True, config={"displayModeBar": False})

    with tab_radar:
        r_col1, r_col2 = st.columns([1, 1], gap="large")
        with r_col1:
            st.markdown('<div class="vp-card-title" style="margin-bottom:8px;">4-Modality Balance Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(
                _render_clean_radar(vs, rs, aus, ws),
                use_container_width=True,
                config={"displayModeBar": False}
            )
        with r_col2:
            st.markdown('<div class="vp-card-title" style="margin-bottom:8px;">Dynamic Fusion Weights & Sensor Status</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.4); border-radius:10px; padding:14px; border:1px solid rgba(255,255,255,0.06);">
              <table style="width:100%; font-size:0.78rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06); color:#94a3b8;">
                  <th style="padding:6px 0; text-align:left;">Sensor Modality</th>
                  <th style="text-align:right;">Weight</th>
                  <th style="text-align:right;">Status</th>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.03);">
                  <td style="padding:7px 0; color:#38bdf8;">Visual (FaceMesh)</td>
                  <td style="text-align:right; font-family:'JetBrains Mono';">{snap.get('w_visual', 0.30):.2f}</td>
                  <td style="text-align:right; color:#10b981;">● Online</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.03);">
                  <td style="padding:7px 0; color:#f43f5e;">rPPG Pulse (Camera)</td>
                  <td style="text-align:right; font-family:'JetBrains Mono';">{snap.get('w_rppg', 0.20):.2f}</td>
                  <td style="text-align:right; color:#10b981;">● Active (Q={snap.get('rppg_quality', 0.9):.2f})</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.03);">
                  <td style="padding:7px 0; color:#34d399;">Acoustic (Microphone)</td>
                  <td style="text-align:right; font-family:'JetBrains Mono';">{snap.get('w_audio', 0.25):.2f}</td>
                  <td style="text-align:right; color:#10b981;">● Calibrated</td>
                </tr>
                <tr>
                  <td style="padding:7px 0; color:#a855f7;">Smartwatch PPG</td>
                  <td style="text-align:right; font-family:'JetBrains Mono';">{snap.get('w_smartwatch', 0.25):.2f}</td>
                  <td style="text-align:right; color:#10b981;">● Connected (Gated)</td>
                </tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

    with tab_intel:
        i_col1, i_col2 = st.columns([1, 1], gap="medium")
        with i_col1:
            st.markdown('<div class="vp-card-title" style="margin-bottom:8px;">Session Telemetry Log</div>', unsafe_allow_html=True)
            drowsy_ev = snap.get("drowsy_events", 0)
            crit_ev = snap.get("critical_events", 0)
            blink_c = snap.get("blink_count", 0)
            yawn_c = snap.get("yawn_count_visual", 0)

            st.markdown(f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
              <div class="vp-metric-tile">
                <div class="vp-tile-label">Drowsy Events</div>
                <div class="vp-tile-value" style="color:{'#fbbf24' if drowsy_ev > 0 else '#f8fafc'};">{drowsy_ev}</div>
                <div class="vp-tile-unit">Cumulative instances</div>
              </div>
              <div class="vp-metric-tile">
                <div class="vp-tile-label">Critical Alerts</div>
                <div class="vp-tile-value" style="color:{'#f87171' if crit_ev > 0 else '#f8fafc'};">{crit_ev}</div>
                <div class="vp-tile-unit">Over-threshold triggers</div>
              </div>
              <div class="vp-metric-tile">
                <div class="vp-tile-label">Total Blinks</div>
                <div class="vp-tile-value">{blink_c}</div>
                <div class="vp-tile-unit">Detected blinks</div>
              </div>
              <div class="vp-metric-tile">
                <div class="vp-tile-label">Yawn Count</div>
                <div class="vp-tile-value">{yawn_c}</div>
                <div class="vp-tile-unit">Visual instances</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with i_col2:
            st.markdown('<div class="vp-card-title" style="margin-bottom:8px;">Pipeline Architecture</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(30,41,59,0.3); border-radius:8px; padding:12px; font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#94a3b8; line-height:1.6;">
              <div>Webcam Stream ──→ VideoCapture ──→ Visual & rPPG Feature Extractors</div>
              <div>Audio Stream  ──→ AudioCapture ──→ Spectral & Yawn Acoustic Extractor</div>
              <div>Smartwatch    ──→ Wearable API ──→ PPG HR, SpO2 & Stress Pipeline</div>
              <div style="color:#38bdf8;">Fusion Engine   ──→ Cross-Validation ──→ Adaptive EMA Scoring</div>
              <div style="color:#34d399;">Safety Engine   ──→ Real-Time Alerts ──→ Audio & Visual Interventions</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="vp-card-title" style="margin-bottom:12px;">System Controls</div>', unsafe_allow_html=True)
        st.session_state.demo_mode = st.toggle("Simulation Demo Mode", value=demo)
        st.session_state.auto_refresh = st.toggle("Live Telemetry Stream", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.select_slider(
                "Refresh Cadence",
                options=[1000, 2000, 3000, 5000],
                value=st.session_state.refresh_interval,
                format_func=lambda x: f"{x/1000:.1f}s"
            )
        st.divider()

        st.markdown('<div class="vp-card-title" style="margin-bottom:8px;">Safety Thresholds</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem; display:flex; flex-direction:column; gap:6px;">
          <div style="display:flex; justify-content:space-between; padding:4px 8px; border-radius:6px; background:rgba(16,185,129,0.1); color:#34d399;">
            <span>SAFE</span> <span>0 &ndash; 25%</span>
          </div>
          <div style="display:flex; justify-content:space-between; padding:4px 8px; border-radius:6px; background:rgba(245,158,11,0.1); color:#fbbf24;">
            <span>WARNING</span> <span>25 &ndash; 50%</span>
          </div>
          <div style="display:flex; justify-content:space-between; padding:4px 8px; border-radius:6px; background:rgba(249,115,22,0.1); color:#fb923c;">
            <span>DANGER</span> <span>50 &ndash; 75%</span>
          </div>
          <div style="display:flex; justify-content:space-between; padding:4px 8px; border-radius:6px; background:rgba(239,68,68,0.1); color:#f87171;">
            <span>CRITICAL</span> <span>75 &ndash; 100%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="vp-card-title" style="margin-bottom:8px;">Diagnostics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.72rem; color:#94a3b8; font-family:'JetBrains Mono', monospace; line-height:1.7;">
          STATE: <span style="color:#34d399;">ACTIVE</span><br/>
          PIPELINE: {'LIVE' if not demo else 'SYNTHETIC'}<br/>
          FPS: {fps:.1f}<br/>
          BUFFER: 300 SLOTS<br/>
          REFRESH: 300 ms
        </div>
        """, unsafe_allow_html=True)


# ── Run ───────────────────────────────────────────────────────────────────────
render()
