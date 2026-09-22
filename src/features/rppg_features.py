"""
Remote Photoplethysmography (rPPG) Feature Extractor

Estimates heart rate and HRV from the subtle colour changes in facial skin
caused by blood pulsation — NO wearable required.

Algorithm (CHROM method):
  1. Extract mean RGB from forehead/cheek ROI each frame
  2. Normalise channels
  3. Project onto chrominance signals X = 3R-2G, Y = 1.5R+G-1.5B
  4. Bandpass filter 0.75–3.0 Hz (45–180 BPM)
  5. FFT → dominant frequency → heart rate
  6. RMSSD from inter-beat intervals (IBI) → HRV
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
from scipy import signal as scipy_signal

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RPPGFeatures:
    heart_rate_bpm: float = 0.0          # Beats per minute
    hrv_rmssd: float = 0.0               # Root Mean Square Successive Differences (ms)
    signal_quality: float = 0.0          # 0–1 confidence in rPPG estimate
    raw_signal: List[float] = field(default_factory=list)
    filtered_signal: List[float] = field(default_factory=list)
    ibi_series: List[float] = field(default_factory=list)   # Inter-beat intervals (ms)
    hr_alert: bool = False               # True if HR out of normal range


class RPPGExtractor:
    """
    CHROM-based rPPG processor.

    Call update(roi) per frame, then get_features() to retrieve estimates.
    """

    def __init__(self, cfg: dict):
        rp = cfg.get("rppg", {})
        self.window_sec: float = rp.get("window_sec", 10.0)
        self.min_hz: float     = rp.get("min_hz", 0.75)
        self.max_hz: float     = rp.get("max_hz", 3.0)
        self.fps_est: float    = 30.0  # Will be updated dynamically
        self.normal_hr_min: float = rp.get("normal_hr_min", 55.0)
        self.normal_hr_max: float = rp.get("normal_hr_max", 100.0)
        self.low_hr_thr: float    = rp.get("low_hr_threshold", 50.0)
        self.high_hr_thr: float   = rp.get("high_hr_threshold", 110.0)

        self._rgb_buffer: deque = deque()       # (r, g, b, ts)
        self._hr_history: deque = deque(maxlen=10)
        self._last_hr: float = 0.0
        self._last_features = RPPGFeatures()
        self._update_count: int = 0

        log.info("RPPGExtractor initialised (window=%.1fs, %.2f–%.2f Hz).",
                 self.window_sec, self.min_hz, self.max_hz)

    # ── Update ────────────────────────────────────────────────────

    def update(self, face_roi: Optional[np.ndarray], timestamp: Optional[float] = None) -> None:
        """
        Feed a new face ROI frame.

        Parameters
        ----------
        face_roi  : np.ndarray (H×W×3 BGR) or None if face not detected
        timestamp : Unix timestamp
        """
        if face_roi is None or face_roi.size == 0:
            return

        ts = timestamp if timestamp is not None else time.time()

        # Extract mean RGB from the central 40% of the ROI (forehead/cheek)
        h, w = face_roi.shape[:2]
        cy, cx = h // 2, w // 2
        rh, rw = max(1, int(h * 0.4)), max(1, int(w * 0.4))
        patch = face_roi[cy - rh // 2: cy + rh // 2,
                         cx - rw // 2: cx + rw // 2]

        if patch.size == 0:
            return

        # BGR → RGB means
        b_mean = float(np.mean(patch[:, :, 0]))
        g_mean = float(np.mean(patch[:, :, 1]))
        r_mean = float(np.mean(patch[:, :, 2]))

        self._rgb_buffer.append((r_mean, g_mean, b_mean, ts))
        self._update_count += 1

        # Remove samples older than window
        cutoff = ts - self.window_sec
        while self._rgb_buffer and self._rgb_buffer[0][3] < cutoff:
            self._rgb_buffer.popleft()

    def get_features(self) -> RPPGFeatures:
        """Compute and return rPPG features from current buffer."""
        feat = RPPGFeatures()

        n = len(self._rgb_buffer)
        if n < int(self.fps_est * 3):  # Need at least 3 seconds of data
            feat.signal_quality = 0.0
            feat.heart_rate_bpm = self._last_hr  # Return last known
            return feat

        data = np.array(self._rgb_buffer)  # shape (n, 4): R, G, B, ts
        r, g, b, timestamps = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # Estimate actual FPS from timestamps
        if len(timestamps) >= 2:
            self.fps_est = max(10.0, (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))

        # ── CHROM algorithm ────────────────────────────────────
        # Normalise each channel by its mean
        r_n = r / (np.mean(r) + 1e-7)
        g_n = g / (np.mean(g) + 1e-7)
        b_n = b / (np.mean(b) + 1e-7)

        # Chrominance signals
        xs = 3 * r_n - 2 * g_n        # X chrominance
        ys = 1.5 * r_n + g_n - 1.5 * b_n   # Y chrominance

        # Standard deviation based combination
        std_xs = np.std(xs) + 1e-7
        std_ys = np.std(ys) + 1e-7
        pleth = xs / std_xs - ys / std_ys

        # ── Bandpass filter ────────────────────────────────────
        nyq = self.fps_est / 2.0
        low  = self.min_hz / nyq
        high = min(0.99, self.max_hz / nyq)

        try:
            sos = scipy_signal.butter(4, [low, high], btype="band", output="sos")
            filtered = scipy_signal.sosfiltfilt(sos, pleth)
        except Exception:
            filtered = pleth.copy()

        feat.raw_signal = pleth.tolist()[-100:]          # last 100 samples
        feat.filtered_signal = filtered.tolist()[-100:]

        # ── FFT → Heart Rate ───────────────────────────────────
        n_fft = len(filtered)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.fps_est)
        fft_mag = np.abs(np.fft.rfft(filtered * np.hanning(n_fft)))

        # Mask to valid heart-rate band
        band_mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        if not np.any(band_mask):
            feat.heart_rate_bpm = self._last_hr
            return feat

        peak_idx = np.argmax(fft_mag[band_mask])
        dominant_freq = freqs[band_mask][peak_idx]
        hr_bpm = dominant_freq * 60.0

        # ── Signal Quality ─────────────────────────────────────
        total_power = np.sum(fft_mag ** 2) + 1e-7
        band_power  = np.sum(fft_mag[band_mask] ** 2)
        feat.signal_quality = min(1.0, float(band_power / total_power))

        # Smooth HR using exponential moving average
        alpha = 0.3
        if self._last_hr > 0:
            hr_bpm = alpha * hr_bpm + (1 - alpha) * self._last_hr

        self._last_hr = hr_bpm
        self._hr_history.append(hr_bpm)
        feat.heart_rate_bpm = float(np.mean(self._hr_history))

        # ── HRV (RMSSD) ────────────────────────────────────────
        feat.hrv_rmssd = self._compute_hrv(filtered, feat.heart_rate_bpm)

        # ── Alerts ─────────────────────────────────────────────
        feat.hr_alert = (
            feat.heart_rate_bpm < self.low_hr_thr or
            feat.heart_rate_bpm > self.high_hr_thr
        ) and feat.signal_quality > 0.3

        return feat

    # ── HRV ───────────────────────────────────────────────────────

    def _compute_hrv(self, filtered_signal: np.ndarray, hr_bpm: float) -> float:
        """
        Estimate HRV-RMSSD from the filtered rPPG signal by finding
        successive peaks (proxy for R-R intervals).
        """
        if hr_bpm <= 0:
            return 0.0
        try:
            # Expected peak distance in samples
            peak_dist = int(self.fps_est * 60.0 / (hr_bpm + 1e-7))
            peak_dist = max(5, peak_dist)
            peaks, _ = scipy_signal.find_peaks(
                filtered_signal,
                distance=peak_dist,
                prominence=np.std(filtered_signal) * 0.5
            )
            if len(peaks) < 3:
                return 0.0

            ibis_s = np.diff(peaks) / self.fps_est          # seconds
            ibis_ms = ibis_s * 1000.0                       # milliseconds
            rmssd = float(np.sqrt(np.mean(np.diff(ibis_ms) ** 2)))
            return rmssd
        except Exception:
            return 0.0

    def reset(self) -> None:
        self._rgb_buffer.clear()
        self._hr_history.clear()
        self._last_hr = 0.0
