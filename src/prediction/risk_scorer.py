"""
Drowsiness Risk Scorer

Maps fused score → discrete ALERT_LEVEL and computes confidence-weighted
overall health risk. Maintains session statistics for the dashboard.
"""

import time
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

from src.fusion.multimodal_fusion import FusedState
from src.utils.logger import get_logger

log = get_logger(__name__)

# Alert level constants
LEVEL_SAFE     = 0   # Green  — fully alert
LEVEL_WARNING  = 1   # Yellow — mildly drowsy
LEVEL_DANGER   = 2   # Orange — moderately drowsy
LEVEL_CRITICAL = 3   # Red    — severely drowsy / dangerous


@dataclass
class RiskAssessment:
    alert_level: int = LEVEL_SAFE        # 0–3
    alert_label: str = "SAFE"
    fused_score: float = 0.0             # 0–1
    smoothed_score: float = 0.0

    # Per-modality sub-scores
    visual_score: float     = 0.0
    rppg_score: float       = 0.0
    audio_score: float      = 0.0
    smartwatch_score: float = 0.0

    # Smartwatch metrics
    smartwatch_hr: float = 0.0
    smartwatch_spo2: float = 98.0
    smartwatch_stress: float = 20.0
    smartwatch_connected: bool = False
    cross_validation_status: str = ""

    # Specific diagnostic reason for escalated alerts
    alert_reason: str = ""

    # Session statistics
    session_duration_sec: float = 0.0
    total_drowsy_events: int = 0
    critical_event_count: int = 0

    # Trend
    score_trend: float = 0.0            # positive = getting worse
    timestamp: float = field(default_factory=time.time)

    @property
    def color(self) -> str:
        return {0: "#00C851", 1: "#ffbb33", 2: "#ff8800", 3: "#CC0000"}[self.alert_level]

    @property
    def emoji(self) -> str:
        return {0: "✅", 1: "⚠️", 2: "🟠", 3: "🔴"}[self.alert_level]


class RiskScorer:
    """
    Converts a FusedState into a RiskAssessment with alert levels,
    session-level statistics, and smartwatch rule-based overrides.
    """

    def __init__(self, cfg: dict):
        ac = cfg.get("alert", {})
        self.level_0_max: float = ac.get("level_0_max", 0.25)
        self.level_1_max: float = ac.get("level_1_max", 0.50)
        self.level_2_max: float = ac.get("level_2_max", 0.75)

        # Smartwatch alert rules thresholds
        sc = cfg.get("smartwatch", {})
        self.watch_low_hr: float = sc.get("low_hr_threshold", 50.0)
        self.watch_high_hr: float = sc.get("high_hr_threshold", 120.0)
        self.watch_low_spo2: float = sc.get("low_spo2_threshold", 92.0)
        self.watch_high_stress: float = sc.get("high_stress_threshold", 80.0)

        self._session_start: float = time.time()
        self._score_history: deque = deque(maxlen=300)  # ~10s at 30fps
        self._drowsy_events: int = 0
        self._critical_events: int = 0
        self._prev_level: int = LEVEL_SAFE

        log.info("RiskScorer initialised (thresholds: 0=%.2f, 1=%.2f, 2=%.2f, 3=1.0).",
                 self.level_0_max, self.level_1_max, self.level_2_max)

    def score(self, state: FusedState) -> RiskAssessment:
        """Convert FusedState → RiskAssessment with physiological rules."""
        s = state.smoothed_score
        self._score_history.append(s)

        # Baseline alert level based on continuous fused score
        if s <= self.level_0_max:
            level = LEVEL_SAFE
            label = "SAFE"
        elif s <= self.level_1_max:
            level = LEVEL_WARNING
            label = "WARNING"
        elif s <= self.level_2_max:
            level = LEVEL_DANGER
            label = "DANGER"
        else:
            level = LEVEL_CRITICAL
            label = "CRITICAL"

        reason = ""

        # ── Smartwatch Physiological Rule Overrides ───────────────────
        # Rule 1: HR < 50 or > 120 AND rPPG agrees → Level 3 (CRITICAL)
        if state.smartwatch_connected and state.smartwatch_hr > 0:
            hr_abnormal = (state.smartwatch_hr < self.watch_low_hr or
                           state.smartwatch_hr > self.watch_high_hr)
            rppg_agrees = (state.cross_validation_status == "AGREE" or
                           (state.heart_rate > 0 and state.cross_validation_diff < 5.0))
            if hr_abnormal and rppg_agrees:
                level = LEVEL_CRITICAL
                label = "CRITICAL"
                reason = f"Abnormal HR ({state.smartwatch_hr:.0f} BPM) verified by rPPG"

        # Rule 2: SpO2 < 92% → Level 3 (CRITICAL)
        if state.smartwatch_connected and 0 < state.smartwatch_spo2 < self.watch_low_spo2:
            level = LEVEL_CRITICAL
            label = "CRITICAL"
            reason = f"Dangerous hypoxemia (SpO2: {state.smartwatch_spo2:.1f}% < 92%)"

        # Rule 3: Stress > 80 AND visual fatigue high → Level 2 (DANGER)
        if state.smartwatch_connected and state.smartwatch_stress > self.watch_high_stress:
            visual_fatigue = (state.visual_score >= 0.25 or
                              state.perclos >= 0.15 or
                              state.head_pose_alert)
            if visual_fatigue and level < LEVEL_DANGER:
                level = LEVEL_DANGER
                label = "DANGER"
                if not reason:
                    reason = f"High stress ({state.smartwatch_stress:.0f}/100) with visual fatigue"

        # Count events (level transitions upward)
        if level > self._prev_level:
            self._drowsy_events += 1
            if level == LEVEL_CRITICAL:
                self._critical_events += 1

        self._prev_level = level

        # Trend: compare last 30 samples vs previous 30
        history = list(self._score_history)
        if len(history) >= 60:
            recent = sum(history[-30:]) / 30
            prev   = sum(history[-60:-30]) / 30
            trend  = recent - prev
        else:
            trend = 0.0

        ra = RiskAssessment(
            alert_level=level,
            alert_label=label,
            fused_score=state.fused_score,
            smoothed_score=s,
            visual_score=state.visual_score,
            rppg_score=state.rppg_score,
            audio_score=state.audio_score,
            smartwatch_score=state.smartwatch_score,
            smartwatch_hr=state.smartwatch_hr,
            smartwatch_spo2=state.smartwatch_spo2,
            smartwatch_stress=state.smartwatch_stress,
            smartwatch_connected=state.smartwatch_connected,
            cross_validation_status=state.cross_validation_status,
            alert_reason=reason,
            session_duration_sec=time.time() - self._session_start,
            total_drowsy_events=self._drowsy_events,
            critical_event_count=self._critical_events,
            score_trend=trend,
            timestamp=time.time(),
        )
        return ra

    def reset_session(self) -> None:
        self._session_start = time.time()
        self._score_history.clear()
        self._drowsy_events = 0
        self._critical_events = 0
        self._prev_level = LEVEL_SAFE
