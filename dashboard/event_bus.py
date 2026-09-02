#!/usr/bin/env python3
"""
Verit NIDS - Dashboard Event Bus
------------------------------------------------------------------
A thread-safe, in-process pub-sub broadcaster. The detection loop calls
`publish()` once per completed flow; each connected dashboard browser
tab holds its own subscriber queue (via the Flask SSE endpoint) so slow
or disconnected clients never back up the detection thread.

Also keeps a rolling history (for a new tab's initial page load) and
running summary stats (for the badge counters).
"""

import queue
import threading
import time
from collections import deque

import numpy as np


def _jsonable(value):
    """Flatten numpy scalar types (common in a DataFrame row.to_dict())
    into plain Python types so json.dumps() never chokes on them."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if v != v else v  # NaN -> null
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


class EventBus:
    def __init__(self, history_size=300):
        self._subscribers = []  # list[queue.Queue]
        self._lock = threading.Lock()
        self._history = deque(maxlen=history_size)
        self._stats = {
            "total_flows": 0,
            "benign": 0,
            "known_attacks": 0,
            "zero_day": 0,
            "packets_note": None,
        }
        self._start_time = time.time()

    def publish(self, event: dict):
        clean = {k: _jsonable(v) for k, v in event.items()}
        clean.setdefault("received_at", time.time())

        with self._lock:
            self._history.append(clean)
            self._update_stats(clean)
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(clean)
                except queue.Full:
                    dead.append(q)  # a stalled/disconnected client -- drop it, don't block on it
            for q in dead:
                self._subscribers.remove(q)

    def _update_stats(self, event):
        self._stats["total_flows"] += 1
        verdict = event.get("verdict", "")
        if verdict == "BENIGN":
            self._stats["benign"] += 1
        elif verdict == "ZERO_DAY_SUSPECTED":
            self._stats["zero_day"] += 1
        elif verdict:
            self._stats["known_attacks"] += 1

    def subscribe(self, maxsize=1000):
        q = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def get_recent(self, n=150):
        with self._lock:
            return list(self._history)[-n:]

    def get_stats(self):
        with self._lock:
            stats = dict(self._stats)
        stats["uptime_seconds"] = time.time() - self._start_time
        stats["connected_clients"] = len(self._subscribers)
        return stats
