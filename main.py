"""
D-SAAT — Main Pipeline Entry Point

Orchestrates all system threads:
  • VideoCapture    → frame buffer
  • AudioCapture    → audio chunk buffer
  • VisualFeatureExtractor → runs in processing thread
  • RPPGExtractor          → accumulates signal per frame
  • AudioFeatureExtractor  → processes audio chunks
  • MultimodalFusion       → combines all scores
  • RiskScorer             → maps to alert levels
  • AlertManager           → fires alerts
  • SharedState            → bridge to Streamlit dashboard

Usage:
  python main.py [--config config.yaml] [--no-audio] [--headless]
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

# ── Local imports ─────────────────────────────────────────────────────────────
from src.capture.video_capture import VideoCapture
from src.capture.audio_capture import AudioCapture
from src.features.visual_features import VisualFeatureExtractor
from src.features.rppg_features import RPPGExtractor
from src.features.audio_features import AudioFeatureExtractor
from src.features.smartwatch_features import SmartwatchFeatureExtractor
from src.fusion.multimodal_fusion import MultimodalFusion
from src.prediction.risk_scorer import RiskScorer
from src.alert.alert_manager import AlertManager
from src.utils.buffer import SharedState
from src.utils.logger import get_logger, configure_from_config

log = get_logger(__name__)

# Global shared state — read by Streamlit dashboard
SHARED_STATE = SharedState()

# Shutdown flag
_shutdown = threading.Event()


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        log.warning("Config file not found at '%s'. Using defaults.", path)
        return {}
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info("Config loaded from '%s'.", path)
    return cfg


# ── Processing Thread ─────────────────────────────────────────────────────────

class ProcessingPipeline:
    """
    Runs in a dedicated thread. Reads from capture buffers, runs all
    feature extractors, fuses scores, and pushes results to SharedState.
    """

    def __init__(self, cfg: dict, video_cap: Optional[VideoCapture],
                 audio_cap: Optional[AudioCapture],
                 shared_state: SharedState):
        self.cfg = cfg
        self.video_cap = video_cap
        self.audio_cap = audio_cap
        self.state = shared_state

        # Feature extractors
        self.visual_ext = VisualFeatureExtractor(cfg)
        self.rppg_ext   = RPPGExtractor(cfg)
        self.audio_ext  = AudioFeatureExtractor(cfg)
        self.smartwatch_ext = SmartwatchFeatureExtractor(cfg)

        # Fusion & scoring
        self.fusion  = MultimodalFusion(cfg)
        self.scorer  = RiskScorer(cfg)
        self.alerter = AlertManager(cfg)

        # Timing
        self.target_fps: float = cfg.get("video", {}).get("fps", 30)
        self._frame_count: int = 0
        self._audio_process_interval: int = 6  # process audio every N frames

    def process_single_frame(self, frame: np.ndarray, frame_ts: Optional[float] = None,
                             audio_chunk: Optional[np.ndarray] = None) -> Tuple[dict, np.ndarray]:
        """
        Process an arbitrary frame (from local webcam, file, or browser stream).
        Updates state and returns (telemetry_dict, annotated_frame).
        """
        if frame_ts is None:
            frame_ts = time.time()
        self._frame_count += 1

        # ── Visual features ────────────────────────────────
        vis = self.visual_ext.process(frame, frame_ts)

        # ── rPPG ───────────────────────────────────────────
        if vis.face_roi is not None:
            self.rppg_ext.update(vis.face_roi, frame_ts)
        rppg = self.rppg_ext.get_features()

        # ── Audio (if provided or from cap) ────────────────
        audio_feat = None
        if audio_chunk is not None:
            self.audio_ext.update(audio_chunk)
            audio_feat = self.audio_ext.get_features(audio_chunk)
        elif self.audio_cap and self._frame_count % self._audio_process_interval == 0:
            chunk, _ = self.audio_cap.get_latest_chunk()
            if chunk is not None:
                self.audio_ext.update(chunk)
                audio_feat = self.audio_ext.get_features(chunk)

        # ── Smartwatch (4th Modality) ──────────────────────
        smartwatch_feat = self.smartwatch_ext.get_features()

        # ── Fusion ─────────────────────────────────────────
        fused = self.fusion.fuse(vis, rppg, audio_feat, smartwatch_feat)

        # ── Risk scoring ───────────────────────────────────
        assessment = self.scorer.score(fused)

        # ── Alert management ───────────────────────────────
        alert_msg = self.alerter.process(assessment)

        # ── Push to SharedState for dashboard ──────────────
        annotated = vis.annotated_frame if vis.annotated_frame is not None else frame

        # Encode frame as JPEG bytes
        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

        current_fps = self.video_cap.actual_fps if self.video_cap else getattr(self.visual_ext, "fps_est", 30.0)

        telemetry_update = dict(
            # Frame
            frame_jpg=jpg.tobytes(),
            frame_ts=frame_ts,
            frame_count=self._frame_count,

            # Visual
            face_detected=vis.face_detected,
            face_lost=getattr(fused, "face_lost", False),
            ear=vis.ear,
            ear_left=vis.ear_left,
            ear_right=vis.ear_right,
            blink_score=getattr(vis, "blink_score", 0.0),
            eye_blink_left=getattr(vis, "eye_blink_left", 0.0),
            eye_blink_right=getattr(vis, "eye_blink_right", 0.0),
            mar=vis.mar,
            perclos=vis.perclos,
            blink_rate=vis.blink_rate_per_min,
            blink_count=vis.blink_count,
            eye_closed=vis.eye_closed,
            yawn_detected_visual=vis.yawn_detected,
            yawn_count_visual=vis.yawn_count,
            pitch=vis.pitch,
            yaw=vis.yaw,
            roll=vis.roll,
            head_pose_alert=vis.head_pose_alert,

            # rPPG
            heart_rate=rppg.heart_rate_bpm,
            hrv_rmssd=rppg.hrv_rmssd,
            rppg_quality=rppg.signal_quality,
            rppg_filtered=rppg.filtered_signal,
            hr_alert=rppg.hr_alert,

            # Audio
            breathing_rate=fused.breathing_rate if audio_feat else 0.0,
            yawn_score_audio=fused.yawn_score if audio_feat else 0.0,
            audio_drowsiness=audio_feat.audio_drowsiness_score if audio_feat else 0.0,

            # Smartwatch (4th Modality)
            smartwatch_hr=smartwatch_feat.heart_rate_bpm if smartwatch_feat else 0.0,
            smartwatch_spo2=smartwatch_feat.spo2 if smartwatch_feat else 98.0,
            smartwatch_stress=smartwatch_feat.stress_level if smartwatch_feat else 20.0,
            smartwatch_hrv=smartwatch_feat.hrv_rmssd if smartwatch_feat else 0.0,
            smartwatch_connected=smartwatch_feat.is_connected if smartwatch_feat else False,
            smartwatch_confidence=smartwatch_feat.confidence if smartwatch_feat else 0.0,
            cross_val_status=fused.cross_validation_status,
            cross_val_diff=fused.cross_validation_diff,
            cross_val_confidence=fused.cross_validation_confidence,

            # Fusion
            visual_score=fused.visual_score,
            rppg_score=fused.rppg_score,
            audio_score=fused.audio_score,
            smartwatch_score=fused.smartwatch_score,
            fused_score=fused.fused_score,
            smoothed_score=fused.smoothed_score,
            w_visual=fused.w_visual,
            w_rppg=fused.w_rppg,
            w_audio=fused.w_audio,
            w_smartwatch=fused.w_smartwatch,

            # Assessment
            alert_level=assessment.alert_level,
            alert_label=assessment.alert_label,
            alert_color=assessment.color,
            alert_emoji=assessment.emoji,
            alert_message=alert_msg or assessment.alert_reason or "",
            session_duration=assessment.session_duration_sec,
            drowsy_events=assessment.total_drowsy_events,
            critical_events=assessment.critical_event_count,
            score_trend=assessment.score_trend,

            # System
            actual_fps=current_fps,
            pipeline_running=True,
        )

        self.state.update(**telemetry_update)
        return telemetry_update, annotated

    def run(self) -> None:
        """Main processing loop."""
        log.info("ProcessingPipeline started.")
        self.state.update(pipeline_running=True, startup_time=time.time())

        while not _shutdown.is_set():
            loop_start = time.time()

            if self.video_cap is None:
                time.sleep(0.05)
                continue

            # ── Grab latest video frame ────────────────────────
            frame, frame_ts = self.video_cap.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            self.process_single_frame(frame, frame_ts)

            # ── Frame rate control ─────────────────────────────
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / self.target_fps) - elapsed)
            time.sleep(sleep_time)

        log.info("ProcessingPipeline stopped after %d frames.", self._frame_count)



# ── Signal handling ───────────────────────────────────────────────────────────

def _handle_shutdown(sig, frame):
    log.info("Shutdown signal received (%s). Stopping...", sig)
    _shutdown.set()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="D-SAAT Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio capture")
    parser.add_argument("--headless", action="store_true",
                        help="Run without OpenCV window (Streamlit mode)")
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path
    cfg = load_config(str(cfg_path))
    configure_from_config(cfg)

    # ── Register signal handlers ───────────────────────────────
    signal.signal(signal.SIGINT,  _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # ── Start video capture ────────────────────────────────────
    video_cap = VideoCapture(cfg)
    if not video_cap.start():
        log.critical("Cannot open webcam. Exiting.")
        sys.exit(1)

    # ── Start audio capture ────────────────────────────────────
    audio_cap = None
    if not args.no_audio:
        audio_cap = AudioCapture(cfg)
        if not audio_cap.start():
            log.warning("Audio capture failed. Running video-only mode.")
            audio_cap = None

    # ── Start processing pipeline thread ───────────────────────
    pipeline = ProcessingPipeline(cfg, video_cap, audio_cap, SHARED_STATE)
    proc_thread = threading.Thread(target=pipeline.run, name="Processing",
                                   daemon=True)
    proc_thread.start()

    log.info("=" * 60)
    log.info("  D-SAAT is RUNNING")
    log.info("  Dashboard: streamlit run dashboard/app.py")
    log.info("  Press Ctrl+C to stop.")
    log.info("=" * 60)

    # ── Optional: headless mode – no OpenCV window ──────────────
    if args.headless:
        log.info("Running headless (no OpenCV window). Use Streamlit dashboard.")
        _shutdown.wait()  # Block until signal
    else:
        # Show live OpenCV preview window
        log.info("Showing live preview window. Press 'q' to quit.")
        while not _shutdown.is_set():
            frame_jpg = SHARED_STATE.get("frame_jpg")
            if frame_jpg is not None:
                frame = cv2.imdecode(np.frombuffer(frame_jpg, dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is not None:
                    # Draw alert overlay
                    level = SHARED_STATE.get("alert_level", 0)
                    label = SHARED_STATE.get("alert_label", "SAFE")
                    score = SHARED_STATE.get("smoothed_score", 0.0)
                    colors = {0: (0, 200, 0), 1: (0, 180, 255),
                              2: (0, 100, 255), 3: (0, 0, 220)}
                    color = colors.get(level, (0, 200, 0))
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)
                    cv2.putText(frame,
                                f"ALERT: {label}  |  Score: {score:.3f}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                    cv2.imshow("D-SAAT", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                _shutdown.set()
                break

        cv2.destroyAllWindows()

    # ── Cleanup ────────────────────────────────────────────────
    _shutdown.set()
    pipeline.smartwatch_ext.stop()
    video_cap.stop()
    if audio_cap:
        audio_cap.stop()
    proc_thread.join(timeout=5.0)
    log.info("D-SAAT shut down cleanly.")


if __name__ == "__main__":
    main()
