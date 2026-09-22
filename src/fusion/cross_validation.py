"""
Cross-Validation Engine for Heart Rate Modalities

Compares camera-based rPPG heart rate against direct smartwatch heart rate:
  • diff < 5 BPM   → High confidence, status: "AGREE"
  • diff > 15 BPM  → Anomaly flagged, investigate, status: "DISAGREE"
  • 5 <= diff <= 15 → Moderate confidence, status: "MODERATE"
  • One missing    → Use available modality with reduced confidence,
                      status: "WATCH ONLY" or "CAMERA ONLY"
  • Neither valid  → status: "NONE"
"""

from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

STATUS_AGREE       = "AGREE"
STATUS_DISAGREE    = "DISAGREE"
STATUS_MODERATE    = "MODERATE"
STATUS_WATCH_ONLY  = "WATCH ONLY"
STATUS_CAMERA_ONLY = "CAMERA ONLY"
STATUS_NONE        = "NONE"


@dataclass
class CrossValidationResult:
    """Result of cross-validating rPPG against Smartwatch heart rate."""
    status: str = STATUS_NONE              # "AGREE", "DISAGREE", "WATCH ONLY", "CAMERA ONLY", "NONE"
    confidence: float = 0.0                # Overall HR trust score [0.0, 1.0]
    diff_bpm: float = 0.0                  # Absolute difference between modalities (BPM)
    consensus_hr: float = 0.0              # Fused/best-estimate heart rate (BPM)
    anomaly_flag: bool = False             # True if disagreement exceeds threshold (>15 BPM)
    message: str = ""
    rppg_hr: float = 0.0
    smartwatch_hr: float = 0.0


def cross_validate_heart_rate(
    rppg_hr: float,
    rppg_quality: float,
    watch_hr: float,
    watch_connected: bool,
    watch_confidence: float = 1.0,
    agree_threshold_bpm: float = 5.0,
    disagree_threshold_bpm: float = 15.0,
) -> CrossValidationResult:
    """
    Compare rPPG heart rate with smartwatch heart rate.

    Parameters
    ----------
    rppg_hr : float
        Camera rPPG heart rate estimate (BPM).
    rppg_quality : float
        rPPG signal quality in [0.0, 1.0].
    watch_hr : float
        Smartwatch PPG heart rate (BPM).
    watch_connected : bool
        Whether smartwatch is online and reporting data.
    watch_confidence : float
        Confidence of smartwatch sensor reading [0.0, 1.0].
    agree_threshold_bpm : float
        Max difference for "AGREE" status (default 5 BPM).
    disagree_threshold_bpm : float
        Min difference for "DISAGREE" anomaly status (default 15 BPM).

    Returns
    -------
    CrossValidationResult
    """
    res = CrossValidationResult(rppg_hr=rppg_hr, smartwatch_hr=watch_hr)

    rppg_valid = rppg_hr > 30.0 and rppg_quality >= 0.15
    watch_valid = watch_connected and watch_hr > 30.0 and watch_confidence > 0.15

    # ── Case 1: Both modalities available ──────────────────────────
    if rppg_valid and watch_valid:
        diff = abs(rppg_hr - watch_hr)
        res.diff_bpm = round(diff, 2)

        if diff < agree_threshold_bpm:
            # Both agree (< 5 BPM) → High confidence
            res.status = STATUS_AGREE
            res.confidence = min(1.0, 0.90 + 0.10 * (1.0 - diff / agree_threshold_bpm))
            # Weighted consensus (watch sensor has higher SNR)
            res.consensus_hr = round(0.65 * watch_hr + 0.35 * rppg_hr, 1)
            res.anomaly_flag = False
            res.message = f"High confidence agreement (diff: {diff:.1f} BPM)"

        elif diff > disagree_threshold_bpm:
            # Disagree (> 15 BPM) → Flag anomaly, investigate
            res.status = STATUS_DISAGREE
            res.confidence = max(0.20, 0.35 - 0.01 * (diff - disagree_threshold_bpm))
            # Prefer smartwatch as primary reference during sensor disagreement
            res.consensus_hr = watch_hr if watch_confidence >= rppg_quality else rppg_hr
            res.anomaly_flag = True
            res.message = f"Anomaly: rPPG and Smartwatch disagree (diff: {diff:.1f} BPM > 15 BPM)"
            log.warning("HR Cross-Validation Anomaly! rPPG=%.1f BPM vs Watch=%.1f BPM (diff=%.1f)",
                        rppg_hr, watch_hr, diff)

        else:
            # Moderate agreement (5 to 15 BPM)
            res.status = STATUS_MODERATE
            ratio = (diff - agree_threshold_bpm) / (disagree_threshold_bpm - agree_threshold_bpm)
            res.confidence = float(0.90 - ratio * 0.40)  # [0.90 -> 0.50]
            res.consensus_hr = round(0.60 * watch_hr + 0.40 * rppg_hr, 1)
            res.anomaly_flag = False
            res.message = f"Moderate agreement (diff: {diff:.1f} BPM)"

        return res

    # ── Case 2: Smartwatch only ────────────────────────────────────
    if watch_valid:
        res.status = STATUS_WATCH_ONLY
        res.confidence = float(watch_confidence * 0.85)  # Reduced confidence without second source
        res.consensus_hr = watch_hr
        res.anomaly_flag = False
        res.message = "Smartwatch only; camera rPPG unverified"
        return res

    # ── Case 3: Camera rPPG only ───────────────────────────────────
    if rppg_valid:
        res.status = STATUS_CAMERA_ONLY
        res.confidence = float(rppg_quality * 0.70)      # Reduced confidence without watch
        res.consensus_hr = rppg_hr
        res.anomaly_flag = False
        res.message = "Camera rPPG only; smartwatch disconnected"
        return res

    # ── Case 4: Neither valid ──────────────────────────────────────
    res.status = STATUS_NONE
    res.confidence = 0.0
    res.consensus_hr = 0.0
    res.anomaly_flag = False
    res.message = "No physiological heart rate data available"
    return res


class CrossValidator:
    """Stateful wrapper for cross-validating heart rate modalities."""

    def __init__(self, agree_threshold_bpm: float = 5.0,
                 disagree_threshold_bpm: float = 15.0):
        self.agree_threshold = agree_threshold_bpm
        self.disagree_threshold = disagree_threshold_bpm
        self._last_result = CrossValidationResult()

    def validate(
        self,
        rppg_hr: float,
        rppg_quality: float,
        watch_hr: float,
        watch_connected: bool,
        watch_confidence: float = 1.0,
    ) -> CrossValidationResult:
        result = cross_validate_heart_rate(
            rppg_hr=rppg_hr,
            rppg_quality=rppg_quality,
            watch_hr=watch_hr,
            watch_connected=watch_connected,
            watch_confidence=watch_confidence,
            agree_threshold_bpm=self.agree_threshold,
            disagree_threshold_bpm=self.disagree_threshold,
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> CrossValidationResult:
        return self._last_result
