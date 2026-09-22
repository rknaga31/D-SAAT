"""
Audio Feature Extractor

Processes microphone audio chunks to extract:
  • MFCC features (13 coefficients)       → acoustic fingerprint
  • Yawn detection                         → sustained low-freq vocalisation
  • Breathing rate estimation              → ZCR + envelope
  • Speech activity detection             → energy thresholding
  • Acoustic drowsiness score             → composite audio metric

Dependencies: librosa, noisereduce, scipy, numpy
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

from src.utils.logger import get_logger

log = get_logger(__name__)

# Optional imports — degrade gracefully if not installed
try:
    import noisereduce as nr
    _HAS_NR = True
except ImportError:
    _HAS_NR = False
    log.warning("noisereduce not found — audio denoising disabled.")

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False
    log.warning("librosa not found — MFCC features disabled.")


@dataclass
class AudioFeatures:
    # ── MFCC ──────────────────────────────────────────────────────
    mfcc: List[float] = field(default_factory=lambda: [0.0] * 13)
    mfcc_delta: List[float] = field(default_factory=lambda: [0.0] * 13)

    # ── Energy / Activity ─────────────────────────────────────────
    rms_energy: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_centroid: float = 0.0
    speech_active: bool = False

    # ── Yawn detection ────────────────────────────────────────────
    yawn_score: float = 0.0             # 0–1 acoustic yawn probability
    yawn_detected: bool = False
    yawn_count: int = 0

    # ── Breathing ─────────────────────────────────────────────────
    breathing_rate_bpm: float = 0.0    # breaths per minute
    breathing_regular: bool = True

    # ── Drowsiness score (audio-only) ────────────────────────────
    audio_drowsiness_score: float = 0.0   # 0–1

    # ── Quality ───────────────────────────────────────────────────
    signal_quality: float = 0.0
    confidence: float = 0.0


class AudioFeatureExtractor:
    """
    Stateful audio feature extractor.

    Call update(audio_chunk) each time a new audio chunk arrives,
    then get_features() to retrieve computed features.
    """

    def __init__(self, cfg: dict):
        ac = cfg.get("audio", {})
        af = cfg.get("audio_features", {})

        self.sample_rate: int = ac.get("sample_rate", 22050)
        self.n_mfcc: int = af.get("n_mfcc", 13)
        self.yawn_energy_thresh: float = af.get("yawn_energy_threshold", 0.02)
        self.breathing_win: float = af.get("breathing_window_sec", 10.0)
        self.br_min: float = af.get("breathing_rate_normal_min", 12.0)
        self.br_max: float = af.get("breathing_rate_normal_max", 20.0)

        # Rolling audio buffer for longer-window analyses
        self._audio_window: deque = deque()   # (chunk, ts)
        self._yawn_count: int = 0
        self._breathing_env: deque = deque()  # envelope samples

        # Yawn detection state
        self._yawn_consec: int = 0
        self._yawn_consec_thresh: int = 3     # chunks of yawn-like audio
        self._in_yawn: bool = False

        log.info("AudioFeatureExtractor initialised (sr=%d, mfcc=%d).",
                 self.sample_rate, self.n_mfcc)

    # ── Update ────────────────────────────────────────────────────

    def update(self, chunk: np.ndarray, ts: Optional[float] = None) -> None:
        """Push a new audio chunk into the processing buffer."""
        if chunk is None or len(chunk) == 0:
            return
        if ts is None:
            ts = time.time()
        self._audio_window.append((chunk, ts))

        # Prune old chunks beyond window
        cutoff = ts - self.breathing_win
        while self._audio_window and self._audio_window[0][1] < cutoff:
            self._audio_window.popleft()

    def get_features(self, chunk: Optional[np.ndarray] = None) -> AudioFeatures:
        """
        Compute features for the most recent chunk (and window).

        Parameters
        ----------
        chunk : np.ndarray float32 — the latest audio chunk (mono)
        """
        feat = AudioFeatures()

        if chunk is None or len(chunk) < 64:
            return feat

        # ── Denoise ────────────────────────────────────────────
        if _HAS_NR and len(chunk) > 256:
            try:
                chunk = nr.reduce_noise(y=chunk, sr=self.sample_rate,
                                        stationary=True, prop_decrease=0.75)
            except Exception:
                pass  # Use raw chunk if denoising fails

        # ── Signal quality ─────────────────────────────────────
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        feat.rms_energy = rms
        feat.signal_quality = min(1.0, rms / 0.05)   # rough normalisation

        # ── Zero Crossing Rate ─────────────────────────────────
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
        feat.zero_crossing_rate = zcr

        # ── Speech Activity ────────────────────────────────────
        feat.speech_active = rms > 0.005 and zcr > 0.01

        # ── MFCC ───────────────────────────────────────────────
        if _HAS_LIBROSA and len(chunk) > 512:
            try:
                mfcc = librosa.feature.mfcc(
                    y=chunk.astype(np.float32),
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc
                )
                mfcc_mean = mfcc.mean(axis=1).tolist()
                feat.mfcc = mfcc_mean

                # Spectral centroid
                centroid = librosa.feature.spectral_centroid(
                    y=chunk, sr=self.sample_rate
                )
                feat.spectral_centroid = float(centroid.mean())

                # Delta MFCC
                if mfcc.shape[1] > 2:
                    delta = librosa.feature.delta(mfcc)
                    feat.mfcc_delta = delta.mean(axis=1).tolist()
            except Exception as exc:
                log.debug("MFCC error: %s", exc)

        # ── Yawn Detection ─────────────────────────────────────
        feat.yawn_score, feat.yawn_detected = self._detect_yawn(chunk, rms, zcr)
        if feat.yawn_detected:
            self._yawn_count += 1
        feat.yawn_count = self._yawn_count

        # ── Breathing Rate ─────────────────────────────────────
        feat.breathing_rate_bpm = self._estimate_breathing_rate(chunk)
        feat.breathing_regular = self.br_min <= feat.breathing_rate_bpm <= self.br_max

        # ── Audio Drowsiness Score ─────────────────────────────
        feat.audio_drowsiness_score = self._compute_drowsiness(feat)
        feat.confidence = min(1.0, rms / 0.02)

        return feat

    # ── Yawn detection ─────────────────────────────────────────────

    def _detect_yawn(self, chunk: np.ndarray,
                     rms: float, zcr: float) -> tuple:
        """
        Heuristic yawn detector using:
          - Medium energy (not silence, not loud speech)
          - Low ZCR (yawn is a smooth low-frequency sound)
          - Low spectral centroid (fundamental < 500 Hz)
          - Duration: sustained for multiple consecutive chunks
        """
        # Yawn signature: medium energy, low ZCR, low-frequency content
        is_yawn_like = (
            0.008 < rms < 0.15 and    # Energy band
            zcr < 0.08 and             # Low zero-crossing (smooth)
            self._low_freq_dominance(chunk) > 0.55   # Mostly low freq
        )

        if is_yawn_like:
            self._yawn_consec += 1
        else:
            self._in_yawn = False
            self._yawn_consec = max(0, self._yawn_consec - 1)

        yawn_score = min(1.0, self._yawn_consec / (self._yawn_consec_thresh * 2))
        yawn_detected = False

        if self._yawn_consec >= self._yawn_consec_thresh and not self._in_yawn:
            yawn_detected = True
            self._in_yawn = True

        return yawn_score, yawn_detected

    def _low_freq_dominance(self, chunk: np.ndarray,
                             cutoff_hz: float = 600.0) -> float:
        """Return the fraction of energy below cutoff_hz."""
        if len(chunk) < 64:
            return 0.0
        try:
            fft_mag = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), d=1.0 / self.sample_rate)
            low_mask = freqs < cutoff_hz
            total_energy = np.sum(fft_mag ** 2) + 1e-9
            low_energy   = np.sum(fft_mag[low_mask] ** 2)
            return float(low_energy / total_energy)
        except Exception:
            return 0.0

    # ── Breathing rate ─────────────────────────────────────────────

    def _estimate_breathing_rate(self, chunk: np.ndarray) -> float:
        """
        Estimate breaths per minute from the audio envelope of the
        windowed signal.

        Method: RMS envelope → lowpass filter at 0.5 Hz → count peaks
        """
        try:
            if not self._audio_window:
                return 0.0

            # Concatenate window
            chunks = [c for c, _ in self._audio_window]
            full = np.concatenate(chunks)
            if len(full) < self.sample_rate:
                return 0.0

            # Envelope via Hilbert transform
            analytic = scipy_signal.hilbert(full)
            envelope = np.abs(analytic)

            # Lowpass filter (keep only 0.1–0.5 Hz = 6–30 breaths/min)
            nyq = self.sample_rate / 2.0
            sos = scipy_signal.butter(2, [0.1 / nyq, 0.5 / nyq],
                                       btype="band", output="sos")
            env_filtered = scipy_signal.sosfiltfilt(sos, envelope)

            # Count peaks — each peak ≈ one breath
            min_dist = int(self.sample_rate * 2.0)  # min 2s between breaths
            peaks, _ = scipy_signal.find_peaks(env_filtered, distance=min_dist)

            duration_min = len(full) / self.sample_rate / 60.0
            if duration_min <= 0:
                return 0.0

            return float(len(peaks) / duration_min)
        except Exception as exc:
            log.debug("Breathing rate error: %s", exc)
            return 0.0

    # ── Audio drowsiness score ─────────────────────────────────────

    def _compute_drowsiness(self, feat: AudioFeatures) -> float:
        """
        Rule-based audio drowsiness score (0 = alert, 1 = very drowsy).
        """
        score = 0.0

        # Yawn contribution
        score += feat.yawn_score * 0.4

        # Abnormal breathing rate
        if feat.breathing_rate_bpm > 0:
            if feat.breathing_rate_bpm < self.br_min:
                score += 0.3  # Slow breathing → drowsy
            elif feat.breathing_rate_bpm > self.br_max:
                score += 0.1  # Fast → slightly aroused but could be stress

        # Low energy — very quiet speech → drowsy
        if feat.rms_energy < 0.003 and not feat.speech_active:
            score += 0.2

        return min(1.0, score)
