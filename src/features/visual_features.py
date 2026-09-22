"""
Visual Feature Extractor — MediaPipe FaceLandmarker (Tasks API v1.0+)

Supports MediaPipe >= 0.10 (Tasks API) with automatic model download.

Computes:
  • EAR  (Eye Aspect Ratio)    → blink detection
  • MAR  (Mouth Aspect Ratio)  → yawn detection
  • PERCLOS                    → % eye closure in rolling window
  • Blink rate                 → blinks per minute
  • Head pose                  → pitch / yaw / roll via solvePnP
  • Annotated frame            → landmarks + ROI overlay

All results are returned as a VisualFeatures dataclass each call to process().
"""

import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── MediaPipe Task model ──────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"


def _ensure_model() -> str:
    """Download the FaceLandmarker model if not already present."""
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        log.info("Downloading FaceLandmarker model (~30 MB)…")
        try:
            urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
            log.info("Model saved to %s", _MODEL_PATH)
        except Exception as exc:
            log.error("Model download failed: %s", exc)
            raise RuntimeError(
                f"Cannot download MediaPipe model from {_MODEL_URL}. "
                "Please download it manually and place it at: "
                f"{_MODEL_PATH}"
            ) from exc
    return str(_MODEL_PATH)


# ── MediaPipe Tasks API imports ───────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import RunningMode
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    log.error("MediaPipe not found — visual features will be unavailable.")

# ── Landmark indices (MediaPipe 478-point FaceMesh compatible) ────────────────
# These are valid for the standard 468 + 10 iris landmarks model
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE  = [33,  160, 158, 133, 153, 144]

# Head pose 3D model (mm, generic)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # Nose tip
    (0.0,  -330.0, -65.0),   # Chin
    (-225.0, 170.0,-135.0),  # Left eye corner
    (225.0,  170.0,-135.0),  # Right eye corner
    (-150.0,-150.0,-125.0),  # Left mouth
    (150.0, -150.0,-125.0),  # Right mouth
], dtype=np.float64)

POSE_LM_IDX = [1, 152, 263, 33, 287, 57]


@dataclass
class VisualFeatures:
    # ── Eye metrics ──────────────────────────────────
    ear: float = 0.0
    ear_left: float = 0.0
    ear_right: float = 0.0
    eye_closed: bool = False
    blink_detected: bool = False
    blink_count: int = 0
    blink_rate_per_min: float = 0.0
    perclos: float = 0.0

    # ── Mouth metrics ────────────────────────────────
    mar: float = 0.0
    yawn_detected: bool = False
    yawn_count: int = 0

    # ── Head pose ────────────────────────────────────
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    head_pose_alert: bool = False

    # ── Face ─────────────────────────────────────────
    face_detected: bool = False
    confidence: float = 0.0

    # ── Outputs ──────────────────────────────────────
    annotated_frame: Optional[np.ndarray] = field(default=None, repr=False)
    face_roi: Optional[np.ndarray] = field(default=None, repr=False)
    face_bbox: Optional[Tuple[int, int, int, int]] = None


class VisualFeatureExtractor:
    """
    Processes BGR frames using MediaPipe FaceLandmarker Tasks API.
    Auto-downloads the required model file on first instantiation.
    """

    def __init__(self, cfg: dict):
        vc = cfg.get("visual", {})
        self.ear_thresh: float = vc.get("ear_threshold", 0.24)
        self.ear_consec: int   = vc.get("ear_consecutive_frames", 2)
        self.mar_thresh: float = vc.get("mar_threshold", 0.65)
        self.mar_consec: int   = vc.get("mar_consecutive_frames", 15)
        self.perclos_win: float = vc.get("perclos_window_sec", 60.0)
        self.blink_win: float   = vc.get("blink_window_sec", 60.0)
        self.head_yaw_thr: float   = vc.get("head_yaw_threshold", 30.0)
        self.head_pitch_thr: float = vc.get("head_pitch_threshold", 25.0)
        self.head_roll_thr: float  = vc.get("head_roll_threshold", 20.0)

        self._detector = None
        self._available = False

        if _HAS_MEDIAPIPE:
            try:
                model_path = _ensure_model()
                base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._detector = mp_vision.FaceLandmarker.create_from_options(options)
                self._available = True
                log.info("VisualFeatureExtractor initialised (MediaPipe Tasks API).")
            except Exception as exc:
                log.error("FaceLandmarker init failed: %s", exc)
                log.warning("Visual features will be unavailable.")
        else:
            log.warning("MediaPipe unavailable — visual features disabled.")

        # State
        self._ear_consec_count: int = 0
        self._mar_consec_count: int = 0
        self._blink_count: int = 0
        self._yawn_count: int = 0
        self._eye_was_closed: bool = False
        self._eye_closed_events: deque = deque()
        self._blink_events: deque = deque()

    # ── Main entry point ──────────────────────────────────────────

    def process(self, frame: np.ndarray, ts: Optional[float] = None) -> VisualFeatures:
        """Process a BGR frame and return extracted visual features."""
        if ts is None:
            ts = time.time()

        feat = VisualFeatures()
        h, w = frame.shape[:2]
        annotated = frame.copy()

        if not self._available or self._detector is None:
            feat.annotated_frame = annotated
            return feat

        try:
            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = self._detector.detect(mp_image)
        except Exception as exc:
            log.debug("FaceLandmarker detection error: %s", exc)
            feat.annotated_frame = annotated
            return feat

        if not detection_result.face_landmarks:
            feat.face_detected = False
            feat.annotated_frame = annotated
            return feat

        face_lm = detection_result.face_landmarks[0]
        feat.face_detected = True
        feat.confidence = 1.0

        # Convert normalized landmarks to pixel coords
        lm = [(int(p.x * w), int(p.y * h)) for p in face_lm]

        # Safety check — we need at least 478 landmarks
        if len(lm) < 400:
            feat.annotated_frame = annotated
            return feat

        # ── EAR ───────────────────────────────────────────────
        feat.ear_left  = self._compute_ear(lm, LEFT_EYE)
        feat.ear_right = self._compute_ear(lm, RIGHT_EYE)
        feat.ear = (feat.ear_left + feat.ear_right) / 2.0
        feat.eye_closed = feat.ear < self.ear_thresh

        if feat.eye_closed:
            self._ear_consec_count += 1
            self._eye_closed_events.append(ts)
        else:
            if self._ear_consec_count >= self.ear_consec and self._eye_was_closed:
                self._blink_count += 1
                self._blink_events.append(ts)
                feat.blink_detected = True
            self._ear_consec_count = 0

        self._eye_was_closed = feat.eye_closed
        feat.blink_count = self._blink_count

        # Prune old events
        cutoff = ts - self.perclos_win
        while self._eye_closed_events and self._eye_closed_events[0] < cutoff:
            self._eye_closed_events.popleft()
        cutoff_b = ts - self.blink_win
        while self._blink_events and self._blink_events[0] < cutoff_b:
            self._blink_events.popleft()

        feat.perclos = min(1.0, len(self._eye_closed_events) / max(1, self.perclos_win * 30))
        feat.blink_rate_per_min = len(self._blink_events) * (60.0 / self.blink_win)

        # ── MAR ───────────────────────────────────────────────
        feat.mar = self._compute_mar(lm)
        if feat.mar > self.mar_thresh:
            self._mar_consec_count += 1
        else:
            if self._mar_consec_count >= self.mar_consec:
                self._yawn_count += 1
                feat.yawn_detected = True
            self._mar_consec_count = 0
        feat.yawn_count = self._yawn_count

        # ── Head Pose ─────────────────────────────────────────
        feat.pitch, feat.yaw, feat.roll = self._compute_head_pose(lm, w, h)
        feat.head_pose_alert = (
            abs(feat.yaw) > self.head_yaw_thr or
            abs(feat.pitch) > self.head_pitch_thr or
            abs(feat.roll) > self.head_roll_thr
        )

        # ── Face ROI ──────────────────────────────────────────
        feat.face_roi, feat.face_bbox = self._extract_face_roi(lm, frame, w, h)

        # ── Annotation ────────────────────────────────────────
        annotated = self._annotate(annotated, lm, feat, w, h)
        feat.annotated_frame = annotated

        return feat

    # ── EAR ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_ear(lm: list, eye_idx: list) -> float:
        def dist(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))
        p = [np.array(lm[i]) for i in eye_idx]
        return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]) + 1e-7)

    # ── MAR ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_mar(lm: list) -> float:
        if len(lm) < 400:
            return 0.0
        top    = np.array(lm[13])
        bottom = np.array(lm[14])
        left   = np.array(lm[78])
        right  = np.array(lm[308])
        return float(np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-7))

    # ── Head Pose ─────────────────────────────────────────────────

    @staticmethod
    def _compute_head_pose(lm: list, w: int, h: int) -> Tuple[float, float, float]:
        try:
            pts = np.array([lm[i] for i in POSE_LM_IDX], dtype=np.float64)
            fl  = float(w)
            cam = np.array([[fl, 0, w/2], [0, fl, h/2], [0, 0, 1]], dtype=np.float64)
            ok, rv, _ = cv2.solvePnP(MODEL_POINTS, pts, cam,
                                     np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return 0.0, 0.0, 0.0
            R, _ = cv2.Rodrigues(rv)
            sy = float(np.sqrt(R[0,0]**2 + R[1,0]**2))
            if sy > 1e-6:
                x = np.arctan2(R[2,1], R[2,2])
                y = np.arctan2(-R[2,0], sy)
                z = np.arctan2(R[1,0], R[0,0])
            else:
                x = np.arctan2(-R[1,2], R[1,1])
                y = np.arctan2(-R[2,0], sy)
                z = 0.0
            return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))
        except Exception:
            return 0.0, 0.0, 0.0

    # ── Face ROI ──────────────────────────────────────────────────

    @staticmethod
    def _extract_face_roi(lm: list, frame: np.ndarray,
                          w: int, h: int):
        xs = [p[0] for p in lm]
        ys = [p[1] for p in lm]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return None, None
        return frame[y1:y2, x1:x2], (x1, y1, bw, bh)

    # ── Annotation ────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, lm: list,
                  feat: VisualFeatures, w: int, h: int) -> np.ndarray:
        for idx in LEFT_EYE + RIGHT_EYE:
            if idx < len(lm):
                cv2.circle(frame, lm[idx], 2, (0, 255, 0), -1)
        for idx in [78, 308, 13, 14]:
            if idx < len(lm):
                cv2.circle(frame, lm[idx], 2, (255, 255, 0), -1)

        if feat.face_bbox:
            x, y, bw, bh = feat.face_bbox
            c = (0, 255, 0) if not feat.head_pose_alert else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), c, 2)

        if feat.eye_closed:
            cv2.putText(frame, "EYES CLOSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if feat.yawn_detected:
            cv2.putText(frame, "YAWN", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        if feat.head_pose_alert:
            cv2.putText(frame, f"HEAD DRIFT P{feat.pitch:.0f} Y{feat.yaw:.0f} R{feat.roll:.0f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        metrics = [f"EAR:{feat.ear:.3f}", f"MAR:{feat.mar:.3f}",
                   f"PERCLOS:{feat.perclos*100:.1f}%", f"Blink/m:{feat.blink_rate_per_min:.1f}"]
        for i, txt in enumerate(metrics):
            cv2.putText(frame, txt, (w-200, 25+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

    def release(self) -> None:
        if self._detector:
            self._detector.close()
