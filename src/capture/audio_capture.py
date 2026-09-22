"""
Microphone audio capture thread — continuously records audio chunks and
pushes them into a ThreadSafeBuffer with precise timestamps.
"""

import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from src.utils.buffer import ThreadSafeBuffer
from src.utils.logger import get_logger

log = get_logger(__name__)


class AudioCapture:
    """
    Background thread that reads from the microphone and fills a ring buffer.

    Each buffer entry is a 1-D float32 numpy array (mono, normalised to [-1, 1]).

    Usage
    -----
    audio = AudioCapture(cfg)
    audio.start()
    chunk, ts = audio.get_latest_chunk()
    audio.stop()
    """

    def __init__(self, cfg: dict):
        ac = cfg.get("audio", {})
        self.sample_rate: int = ac.get("sample_rate", 22050)
        self.channels: int = ac.get("channels", 1)
        self.block_duration: float = ac.get("block_duration", 0.5)   # seconds
        self.device_index = ac.get("device_index", None)             # None = default
        buf_size: int = ac.get("buffer_size", 60)

        self.block_size = int(self.sample_rate * self.block_duration)
        self.buffer: ThreadSafeBuffer = ThreadSafeBuffer(maxlen=buf_size)

        self._stream: Optional[sd.InputStream] = None
        self._stop_event = threading.Event()
        self._running = False
        self._chunk_count = 0

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the microphone stream. Returns True on success."""
        try:
            self._stop_event.clear()
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.block_size,
                device=self.device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            log.info("AudioCapture started (%d Hz, %d ch, %.2fs blocks).",
                     self.sample_rate, self.channels, self.block_duration)
            return True
        except Exception as exc:
            log.error("AudioCapture failed to start: %s", exc)
            self._running = False
            return False

    def stop(self) -> None:
        """Stop and close the microphone stream."""
        self._stop_event.set()
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("AudioCapture stopped. Captured %d chunks.", self._chunk_count)

    # ── Internal callback ─────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int,
                         time_info, status) -> None:
        """Called by sounddevice in a high-priority thread."""
        if status:
            log.debug("AudioCapture status: %s", status)
        ts = time.time()
        # Convert to mono if stereo, then flatten
        chunk = indata.copy()
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        self.buffer.put(chunk.astype(np.float32), ts)
        self._chunk_count += 1

    # ── Public API ────────────────────────────────────────────────

    def get_latest_chunk(self):
        """Return (chunk: np.ndarray, timestamp: float) or (None, None)."""
        item = self.buffer.get_latest()
        if item is None:
            return None, None
        return item.data, item.timestamp

    def get_audio_window(self, seconds: float) -> np.ndarray:
        """
        Return a concatenated audio array covering the last *seconds* seconds.
        Returns an empty array if the buffer is empty.
        """
        items = self.buffer.get_window(seconds)
        if not items:
            return np.array([], dtype=np.float32)
        return np.concatenate([it.data for it in items])

    @property
    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()
