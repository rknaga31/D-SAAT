"""
Multimodal Fusion Engine

Combines drowsiness sub-scores from three modalities:
  • Visual  (EAR, MAR, PERCLOS, head pose)
  • rPPG    (heart rate, HRV)
  • Audio   (yawn, breathing rate, MFCC)

Fusion strategy:
  1. Convert each modality's features to a 0–1 drowsiness sub-score
  2. Adaptive weighted average — downweights low-confidence modalities
  3. Exponential Moving Average smoothing for temporal consistency
  4. Output: fused score (0 = fully alert, 1 = critically drowsy)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.features.visual_features import VisualFeatures
from src.features.rppg_features import RPPGFeatures
from src.features.audio_features import AudioFeatures
from src.features.smartwatch_features import SmartwatchFeatures
from src.fusion.cross_validation import (
    cross_validate_heart_rate,
    CrossValidationResult,
    STATUS_NONE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FusedState:
    # ── Sub-scores (0–1) ──────────────────────────────────────────
    visual_score: float     = 0.0
    rppg_score: float       = 0.0
    audio_score: float      = 0.0
    smartwatch_score: float = 0.0

    # ── Effective weights (after confidence adjustment & gating) ──
    w_visual: float     = 0.30
    w_rppg: float       = 0.20
    w_audio: float      = 0.25
    w_smartwatch: float = 0.25

    # ── Fused output ─────────────────────────────────────────────
    fused_score: float = 0.0           # Raw (unsmoothed)
    smoothed_score: float = 0.0        # After EMA

    # ── Confidence ───────────────────────────────────────────────
    overall_confidence: float = 0.0

    # ── Component details for dashboard ──────────────────────────
    ear: float = 0.0
    eye_closed: bool = False
    blink_score: float = 0.0
    eye_blink_left: float = 0.0
    eye_blink_right: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    blink_rate: float = 0.0
    heart_rate: float = 0.0
    hrv_rmssd: float = 0.0
    breathing_rate: float = 0.0
    yawn_score: float = 0.0
    head_pose_alert: bool = False
    face_detected: bool = False
    face_lost: bool = False

    # ── Smartwatch (4th Modality) ────────────────────────────────
    smartwatch_hr: float = 0.0
    smartwatch_spo2: float = 98.0
    smartwatch_stress: float = 20.0
    smartwatch_hrv: float = 0.0
    smartwatch_connected: bool = False

    # ── Cross-Validation Details ─────────────────────────────────
    cross_validation_status: str = STATUS_NONE
    cross_validation_diff: float = 0.0
    cross_validation_confidence: float = 0.0


class MultimodalFusion:
    """
    Fuses visual, rPPG, and audio sub-scores into a single drowsiness score.
    """

    def __init__(self, cfg: dict):
        fc = cfg.get("fusion", {})
        self.base_w_visual: float = fc.get("weight_visual", 0.30)
        self.base_w_rppg: float   = fc.get("weight_rppg",   0.20)
        self.base_w_audio: float  = fc.get("weight_audio",  0.25)
        self.base_w_smartwatch: float = fc.get("weight_smartwatch", 0.25)
        self.alpha: float         = fc.get("smoothing_alpha", 0.3)

        self.vis_conf_thr: float  = fc.get("visual_confidence_threshold", 0.5)
        self.rppg_conf_thr: float = fc.get("rppg_confidence_threshold", 0.3)
        self.audio_conf_thr: float = fc.get("audio_confidence_threshold", 0.3)
        self.watch_conf_thr: float = fc.get("smartwatch_confidence_threshold", 0.4)

        # Visual feature thresholds from config
        vc = cfg.get("visual", {})
        self.ear_thresh: float  = vc.get("ear_threshold", 0.24)
        self.mar_thresh: float  = vc.get("mar_threshold", 0.65)
        self.perclos_warn: float = vc.get("perclos_threshold_warning", 0.15)
        self.perclos_dng: float  = vc.get("perclos_threshold_danger", 0.30)
        self.blink_min: float   = vc.get("blink_rate_min_normal", 10.0)
        self.blink_max: float   = vc.get("blink_rate_max_normal", 20.0)

        # rPPG thresholds
        rc = cfg.get("rppg", {})
        self.hr_min: float = rc.get("normal_hr_min", 55.0)
        self.hr_max: float = rc.get("normal_hr_max", 100.0)

        # Smartwatch thresholds
        sc = cfg.get("smartwatch", {})
        self.watch_hr_min: float = sc.get("low_hr_threshold", 50.0)
        self.watch_hr_max: float = sc.get("high_hr_threshold", 120.0)
        self.watch_spo2_min: float = sc.get("low_spo2_threshold", 92.0)
        self.watch_stress_max: float = sc.get("high_stress_threshold", 80.0)

        # EMA state
        self._smoothed: float = 0.0
        self._first: bool = True
        self._no_face_start: Optional[float] = None

        log.info("MultimodalFusion initialised (w=%.2f/%.2f/%.2f/%.2f, α=%.2f).",
                 self.base_w_visual, self.base_w_rppg, self.base_w_audio,
                 self.base_w_smartwatch, self.alpha)

    # ── Main entry point ──────────────────────────────────────────

    def fuse(self,
             visual: Optional[VisualFeatures],
             rppg: Optional[RPPGFeatures],
             audio: Optional[AudioFeatures],
             smartwatch: Optional[SmartwatchFeatures] = None) -> FusedState:
        """
        Compute fused drowsiness score from four modality feature objects.
        Any modality can be None (e.g., if pipeline is still warming up or disconnected).
        """
        state = FusedState()

        # ── Visual sub-score ───────────────────────────────────
        vis_conf = 0.0
        if visual is not None:
            state.ear = visual.ear
            state.eye_closed = visual.eye_closed
            state.mar = visual.mar
            state.perclos = visual.perclos
            state.blink_rate = visual.blink_rate_per_min
            state.head_pose_alert = visual.head_pose_alert
            state.blink_score = getattr(visual, "blink_score", 0.0)
            state.eye_blink_left = getattr(visual, "eye_blink_left", 0.0)
            state.eye_blink_right = getattr(visual, "eye_blink_right", 0.0)

            if visual.face_detected:
                self._no_face_start = None
                vis_conf = visual.confidence
                state.visual_score = self._visual_subscore(visual)
                state.face_detected = True
            else:
                import time as _t
                now = _t.time()
                if self._no_face_start is None:
                    self._no_face_start = now
                face_lost_dur = now - self._no_face_start
                state.face_detected = False
                state.face_lost = True
                # If face is lost during active tracking, it is an attention hazard / distraction
                if face_lost_dur > 1.0:
                    state.visual_score = min(0.65, 0.40 + 0.10 * (face_lost_dur - 1.0))
                    vis_conf = 0.85
                else:
                    state.visual_score = 0.20
                    vis_conf = 0.50

        # ── rPPG sub-score ─────────────────────────────────────
        rppg_conf = 0.0
        rppg_hr = 0.0
        if rppg is not None:
            rppg_conf = rppg.signal_quality
            state.rppg_score = self._rppg_subscore(rppg)
            rppg_hr = rppg.heart_rate_bpm
            state.heart_rate = rppg.heart_rate_bpm
            state.hrv_rmssd  = rppg.hrv_rmssd

        # ── Audio sub-score ────────────────────────────────────
        audio_conf = 0.0
        if audio is not None:
            audio_conf = audio.confidence
            state.audio_score = audio.audio_drowsiness_score
            state.breathing_rate = audio.breathing_rate_bpm
            state.yawn_score = audio.yawn_score

        # ── Smartwatch (4th Modality) ──────────────────────────
        watch_conf = 0.0
        watch_connected = False
        if smartwatch is not None and smartwatch.is_connected:
            watch_connected = True
            watch_conf = smartwatch.confidence
            state.smartwatch_score = self._smartwatch_subscore(smartwatch)
            state.smartwatch_hr = smartwatch.heart_rate_bpm
            state.smartwatch_spo2 = smartwatch.spo2
            state.smartwatch_stress = smartwatch.stress_level
            state.smartwatch_hrv = smartwatch.hrv_rmssd
            state.smartwatch_connected = True

        # ── Cross-Modal Validation: rPPG vs Smartwatch ────────
        cv_res = cross_validate_heart_rate(
            rppg_hr=rppg_hr,
            rppg_quality=rppg_conf,
            watch_hr=state.smartwatch_hr,
            watch_connected=watch_connected,
            watch_confidence=watch_conf,
        )
        state.cross_validation_status = cv_res.status
        state.cross_validation_diff = cv_res.diff_bpm
        state.cross_validation_confidence = cv_res.confidence

        if cv_res.consensus_hr > 0.0:
            state.heart_rate = cv_res.consensus_hr

        # ── Adaptive weighting with Gating & Redistribution ────
        w_vis = self.base_w_visual if vis_conf >= self.vis_conf_thr else self.base_w_visual * vis_conf
        w_rppg = self.base_w_rppg if rppg_conf >= self.rppg_conf_thr else self.base_w_rppg * rppg_conf
        w_audio = self.base_w_audio if audio_conf >= self.audio_conf_thr else self.base_w_audio * audio_conf

        # Gating: if smartwatch disconnected or below confidence threshold, redistribute its weight
        if watch_connected and watch_conf >= self.watch_conf_thr:
            w_watch = self.base_w_smartwatch
        elif watch_connected and watch_conf > 0.1:
            w_watch = self.base_w_smartwatch * watch_conf
        else:
            w_watch = 0.0  # Gated out; weight redistributed

        total_w = w_vis + w_rppg + w_audio + w_watch
        if total_w < 1e-6:
            # No data at all
            state.fused_score = 0.0
            state.overall_confidence = 0.0
            state.w_visual = state.w_rppg = state.w_audio = state.w_smartwatch = 0.0
        else:
            w_vis   /= total_w
            w_rppg  /= total_w
            w_audio /= total_w
            w_watch /= total_w
            state.w_visual = w_vis
            state.w_rppg   = w_rppg
            state.w_audio  = w_audio
            state.w_smartwatch = w_watch

            state.fused_score = (
                w_vis   * state.visual_score +
                w_rppg  * state.rppg_score +
                w_audio * state.audio_score +
                w_watch * state.smartwatch_score
            )

            # If smartwatch data is simulated/mock or disconnected, ensure real visual fatigue isn't diluted
            is_mock_watch = not watch_connected or "simulated" in getattr(smartwatch, "provider", "").lower() or getattr(smartwatch, "provider", "").lower() == "mock"
            if is_mock_watch and state.visual_score > 0.15:
                state.fused_score = max(state.fused_score, state.visual_score * 0.90)

            state.overall_confidence = (
                vis_conf * w_vis + rppg_conf * w_rppg +
                audio_conf * w_audio + watch_conf * w_watch
            )

        # ── EMA smoothing ──────────────────────────────────────
        if self._first:
            self._smoothed = state.fused_score
            self._first = False
        else:
            # Acute eye closure overrides normal EMA lag
            if visual is not None and (visual.eye_closed or (0 < visual.ear < max(self.ear_thresh, 0.255))):
                self._smoothed = max(self._smoothed, state.fused_score, 0.70)
            else:
                self._smoothed = (self.alpha * state.fused_score +
                                  (1 - self.alpha) * self._smoothed)

        state.smoothed_score = float(np.clip(self._smoothed, 0.0, 1.0))
        return state

    # ── Visual sub-scorer ─────────────────────────────────────────

    def _visual_subscore(self, v: VisualFeatures) -> float:
        """Map visual features to a [0, 1] drowsiness score."""
        score = 0.0

        # Immediate eye closure & microsleep (acute driver risk)
        if v.eye_closed:
            score += 0.85
        elif v.ear < self.ear_thresh:
            score += 0.70
        elif v.ear < self.ear_thresh * 1.15:
            score += 0.35
        elif v.ear < self.ear_thresh * 1.30:
            score += 0.15

        # PERCLOS — cumulative eye closure over window
        if v.perclos > self.perclos_dng:
            score += 0.35
        elif v.perclos > self.perclos_warn:
            score += 0.20
        else:
            score += v.perclos / self.perclos_warn * 0.10

        # EAR — immediate eye closure
        if v.ear < self.ear_thresh:
            score += 0.25
        elif v.ear < self.ear_thresh * 1.2:
            score += 0.10

        # Blink rate — too slow or too fast is a signal
        if v.blink_rate_per_min > 0:
            if v.blink_rate_per_min < self.blink_min:
                score += 0.15   # Infrequent blinking
            elif v.blink_rate_per_min < self.blink_min * 0.5:
                score += 0.25   # Very rare blinks

        # MAR — yawn from video
        if v.mar > self.mar_thresh:
            score += 0.15
        elif v.mar > self.mar_thresh * 0.8:
            score += 0.05

        # Head pose (distraction or head slump)
        if v.head_pose_alert:
            score += 0.20

        # Natural subtle vigilance variation for alert driver:
        # Avoid flat 0.00000; alert drivers exhibit living micro-variations (~95-98% safe)
        if score == 0.0 and v.face_detected:
            import time as _t
            score = 0.03 + 0.02 * abs(np.sin(_t.time() / 8.0))

        return float(min(1.0, score))

    # ── rPPG sub-scorer ───────────────────────────────────────────

    def _rppg_subscore(self, r: RPPGFeatures) -> float:
        """Map rPPG features to a [0, 1] drowsiness score."""
        if r.signal_quality < 0.1:
            return 0.0

        score = 0.0
        hr = r.heart_rate_bpm

        # Low heart rate → drowsy
        if 0 < hr < self.hr_min:
            deficit = (self.hr_min - hr) / self.hr_min
            score += min(0.4, deficit * 0.8)

        # High heart rate can indicate stress (also a warning)
        if hr > self.hr_max:
            excess = (hr - self.hr_max) / self.hr_max
            score += min(0.2, excess * 0.5)

        # Low HRV → less adaptive, potential drowsiness
        if r.hrv_rmssd < 20.0 and r.hrv_rmssd > 0:
            score += 0.2
        elif r.hrv_rmssd < 10.0 and r.hrv_rmssd > 0:
            score += 0.35

        return float(min(1.0, score))

    # ── Smartwatch sub-scorer ──────────────────────────────────────

    def _smartwatch_subscore(self, sw: SmartwatchFeatures) -> float:
        """Map smartwatch physiological features to a [0, 1] drowsiness/risk score."""
        score = 0.0
        hr = sw.heart_rate_bpm

        # Low heart rate (< 50 BPM) indicates extreme drowsiness / bradycardia
        if 0 < hr < self.watch_hr_min:
            deficit = (self.watch_hr_min - hr) / self.watch_hr_min
            score += min(0.40, deficit * 0.80)
        # Elevated heart rate (> 120 BPM) indicates acute stress or physiological strain
        elif hr > self.watch_hr_max:
            excess = (hr - self.watch_hr_max) / self.watch_hr_max
            score += min(0.30, excess * 0.60)

        # Low SpO2 (< 92% critical hypoxia, < 95% warning)
        if 0 < sw.spo2 < self.watch_spo2_min:
            score += 0.40
        elif 0 < sw.spo2 < 95.0:
            score += 0.20

        # Elevated stress metric (> 80 high risk, > 65 elevated)
        if sw.stress_level > self.watch_stress_max:
            score += 0.25
        elif sw.stress_level > 65.0:
            score += 0.10

        # Autonomic fatigue from suppressed HRV RMSSD
        if 0 < sw.hrv_rmssd < 15.0:
            score += 0.20
        elif 0 < sw.hrv_rmssd < 25.0:
            score += 0.10

        return float(min(1.0, score))

    def reset(self) -> None:
        self._smoothed = 0.0
        self._first = True
