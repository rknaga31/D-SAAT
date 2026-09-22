"""
Webcam capture thread — continuously grabs frames and pushes them into
a ThreadSafeBuffer with precise timestamps.
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.utils.buffer import ThreadSafeBuffer
from src.utils.logger import get_logger

log = get_logger(__name__)


class VideoCapture:
    """
    Background thread that reads from a webcam and fills a ring buffer.

    Usage
    -----
    cap = VideoCapture(cfg)
    cap.start()
    frame, ts = cap.get_latest_frame()
    cap.stop()
    """

    def __init__(self, cfg: dict):
        vc = cfg.get("video", {})
        self.device_index: int = vc.get("device_index", 0)
        self.width: int = vc.get("width", 640)
        self.height: int = vc.get("height", 480)
        self.target_fps: int = vc.get("fps", 30)
        buf_size: int = vc.get("buffer_size", 90)

        self.buffer: ThreadSafeBuffer = ThreadSafeBuffer(maxlen=buf_size)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Performance stats
        self.actual_fps: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the webcam and start the capture thread. Returns True on success."""
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Try without backend hint (Linux / macOS)
            self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            log.error("Cannot open webcam at device index %d", self.device_index)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency

        self._stop_event.clear()
        self._running = True
        self._start_time = time.time()
        self._frame_count = 0

        self._thread = threading.Thread(target=self._capture_loop,
                                        name="VideoCapture", daemon=True)
        self._thread.start()
        log.info("VideoCapture started (device=%d, %dx%d @ %dfps)",
                 self.device_index, self.width, self.height, self.target_fps)
        return True

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera."""
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        log.info("VideoCapture stopped. Captured %d frames (%.1f fps avg).",
                 self._frame_count, self.actual_fps)

    # ── Internal capture loop ─────────────────────────────────────

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                log.warning("VideoCapture: dropped frame.")
                time.sleep(0.01)
                continue

            ts = time.time()
            self.buffer.put(frame, ts)

            self._frame_count += 1
            elapsed = ts - self._start_time
            if elapsed > 0:
                self.actual_fps = self._frame_count / elapsed

    # ── Public API ────────────────────────────────────────────────

    def get_latest_frame(self):
        """Return (frame: np.ndarray, timestamp: float) or (None, None)."""
        item = self.buffer.get_latest()
        if item is None:
            return None, None
        return item.data, item.timestamp

    def get_frames_window(self, seconds: float):
        """Return list of (frame, ts) within the last *seconds* seconds."""
        items = self.buffer.get_window(seconds)
        return [(it.data, it.timestamp) for it in items]

    @property
    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()
