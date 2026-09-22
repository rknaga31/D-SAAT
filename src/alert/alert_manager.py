"""
Alert Manager — Graded 3+1 level alert system

Level 0 — SAFE    : No action
Level 1 — WARNING : Visual flash + soft beep tone
Level 2 — DANGER  : Persistent visual + alarm tone
Level 3 — CRITICAL: Visual + alarm + TTS voice warning

Uses a cooldown timer to prevent alert spam.
Generates beep tones via numpy without external sound files.
"""

import threading
import time
from typing import Optional

import numpy as np

from src.prediction.risk_scorer import (
    RiskAssessment,
    LEVEL_SAFE, LEVEL_WARNING, LEVEL_DANGER, LEVEL_CRITICAL
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Try to import audio output for beeps
try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

# Try to import TTS
try:
    import pyttsx3
    _HAS_TTS = True
except ImportError:
    _HAS_TTS = False
    log.warning("pyttsx3 not found — TTS alerts disabled.")


class AlertManager:
    """
    Manages graded alerts based on the current RiskAssessment.

    Thread-safe: alert sounds are dispatched to a daemon thread.
    """

    # Human-readable messages for each level
    MESSAGES = {
        LEVEL_SAFE:     "Driver is alert. All systems normal.",
        LEVEL_WARNING:  "Drowsiness detected — please stay focused!",
        LEVEL_DANGER:   "Significant drowsiness — take a break soon!",
        LEVEL_CRITICAL: "CRITICAL — Stop the vehicle immediately and rest!",
    }

    BEEP_FREQS = {
        LEVEL_WARNING:  440.0,   # A4 — gentle
        LEVEL_DANGER:   880.0,   # A5 — urgent
        LEVEL_CRITICAL: 1200.0,  # High — alarm
    }

    def __init__(self, cfg: dict):
        ac = cfg.get("alert", {})
        self.cooldown_sec: float = ac.get("cooldown_sec", 3.0)
        self.tts_enabled: bool = ac.get("tts_enabled", True) and _HAS_TTS

        self._last_alert_time: dict = {
            LEVEL_WARNING:  0.0,
            LEVEL_DANGER:   0.0,
            LEVEL_CRITICAL: 0.0,
        }
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        self._current_level: int = LEVEL_SAFE
        self._alert_active: bool = False

        if self.tts_enabled:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 160)
                self._tts_engine.setProperty("volume", 1.0)
            except Exception as exc:
                log.warning("TTS init failed: %s", exc)
                self.tts_enabled = False

        log.info("AlertManager initialised (cooldown=%.1fs, tts=%s).",
                 self.cooldown_sec, self.tts_enabled)

    # ── Main entry point ──────────────────────────────────────────

    def process(self, assessment: RiskAssessment) -> Optional[str]:
        """
        Evaluate the risk assessment and fire appropriate alerts.

        Returns the alert message string if an alert was fired, else None.
        """
        level = assessment.alert_level
        self._current_level = level
        self._alert_active = level >= LEVEL_WARNING

        if level == LEVEL_SAFE:
            return None

        now = time.time()
        if now - self._last_alert_time.get(level, 0) < self.cooldown_sec:
            return None  # Cooldown — suppress alert

        self._last_alert_time[level] = now
        msg = assessment.alert_reason if assessment.alert_reason else self.MESSAGES[level]

        log.warning("ALERT [%s] score=%.3f | %s",
                    assessment.alert_label, assessment.smoothed_score, msg)

        # Dispatch sound in background thread to avoid blocking pipeline
        threading.Thread(
            target=self._fire_alert,
            args=(level, msg),
            daemon=True,
            name=f"Alert-L{level}"
        ).start()

        return msg

    # ── Alert dispatch ─────────────────────────────────────────────

    def _fire_alert(self, level: int, msg: Optional[str] = None) -> None:
        """Play beep tone and optionally TTS in a background thread."""
        try:
            # Generate and play beep
            if level in self.BEEP_FREQS:
                self._play_beep(
                    freq=self.BEEP_FREQS[level],
                    duration=0.3 if level == LEVEL_WARNING else
                             0.6 if level == LEVEL_DANGER else 1.0,
                    volume=0.4 if level == LEVEL_WARNING else
                           0.6 if level == LEVEL_DANGER else 0.9,
                )

            # TTS for critical only
            if level == LEVEL_CRITICAL and self.tts_enabled:
                speech_text = msg or self.MESSAGES[LEVEL_CRITICAL]
                self._speak(speech_text)
        except Exception as exc:
            log.debug("Alert dispatch error: %s", exc)

    @staticmethod
    def _play_beep(freq: float, duration: float, volume: float,
                   sample_rate: int = 22050) -> None:
        """Generate a pure sine-wave beep safely using winsound on Windows or sounddevice."""
        # On Windows, winsound.Beep is 100% crash-proof and directly communicates with OS
        try:
            import winsound
            f = int(np.clip(freq, 37, 32767))
            ms = int(max(50, duration * 1000))
            winsound.Beep(f, ms)
            return
        except Exception:
            pass

        if _HAS_SD:
            try:
                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                wave = volume * np.sin(2 * np.pi * freq * t)
                fade_samples = int(0.02 * sample_rate)
                fade = np.linspace(0, 1, fade_samples)
                wave[:fade_samples] *= fade
                wave[-fade_samples:] *= fade[::-1]
                sd.play(wave.astype(np.float32), samplerate=sample_rate)
                sd.wait()
            except Exception as exc:
                log.debug("sd.play error: %s", exc)

    def _speak(self, text: str) -> None:
        """Thread-safe TTS speech."""
        if not self.tts_enabled or self._tts_engine is None:
            return
        with self._tts_lock:
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as exc:
                log.debug("TTS speak failed: %s", exc)

    # ── Properties ────────────────────────────────────────────────

    @property
    def current_level(self) -> int:
        return self._current_level

    @property
    def alert_active(self) -> bool:
        return self._alert_active

    def get_status_text(self) -> str:
        return self.MESSAGES.get(self._current_level, "Unknown")
