"""
Thread-safe timestamped ring buffer for multimodal data streams.
Supports concurrent producers (capture threads) and consumers (feature extractors).
"""

import threading
import time
from collections import deque
from typing import Any, Generic, List, NamedTuple, Optional, Tuple, TypeVar

T = TypeVar("T")


class TimestampedItem(NamedTuple):
    timestamp: float  # Unix time
    data: Any


class ThreadSafeBuffer(Generic[T]):
    """
    Fixed-size circular buffer that is safe for concurrent read/write.

    Attributes
    ----------
    maxlen : int
        Maximum number of items before oldest is discarded.
    """

    def __init__(self, maxlen: int = 100):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    # ── Producers ────────────────────────────────────────────────

    def put(self, data: T, timestamp: Optional[float] = None) -> None:
        """Append item with optional timestamp (defaults to now)."""
        ts = timestamp if timestamp is not None else time.time()
        item = TimestampedItem(ts, data)
        with self._not_empty:
            self._buf.append(item)
            self._not_empty.notify_all()

    # ── Consumers ────────────────────────────────────────────────

    def get_latest(self) -> Optional[TimestampedItem]:
        """Return the most recently added item (non-blocking)."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def get_last_n(self, n: int) -> List[TimestampedItem]:
        """Return up to *n* most recent items (oldest first)."""
        with self._lock:
            items = list(self._buf)
        return items[-n:] if len(items) >= n else items

    def get_window(self, seconds: float) -> List[TimestampedItem]:
        """Return all items timestamped within the last *seconds* seconds."""
        cutoff = time.time() - seconds
        with self._lock:
            items = list(self._buf)
        return [it for it in items if it.timestamp >= cutoff]

    def wait_for_item(self, timeout: float = 1.0) -> Optional[TimestampedItem]:
        """Block until a new item is available or timeout expires."""
        with self._not_empty:
            self._not_empty.wait_for(lambda: len(self._buf) > 0, timeout=timeout)
            return self._buf[-1] if self._buf else None

    # ── Inspection ───────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buf) == 0

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    @property
    def maxlen(self) -> int:
        return self._buf.maxlen


class SharedState:
    """
    Thread-safe key-value store for sharing computed metrics between
    the processing pipeline and the Streamlit dashboard.
    """

    def __init__(self):
        self._state: dict = {}
        self._lock = threading.RLock()

    def update(self, **kwargs) -> None:
        with self._lock:
            self._state.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def snapshot(self) -> dict:
        """Return a shallow copy of the entire state dict."""
        with self._lock:
            return dict(self._state)
