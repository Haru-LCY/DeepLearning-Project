"""Thread-safe progress fanout for Server-Sent Events."""

from __future__ import annotations

from datetime import datetime, timezone
import queue
import threading
from typing import Any


class ProgressBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, set[queue.Queue[dict[str, Any]]]] = {}

    def subscribe(self, job_id: str) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.setdefault(job_id, set()).add(subscriber)
        return subscriber

    def unsubscribe(self, job_id: str, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(job_id)
            if not subscribers:
                return
            subscribers.discard(subscriber)
            if not subscribers:
                self._subscribers.pop(job_id, None)

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        event = {**event, "job_id": job_id, "timestamp": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            subscribers = list(self._subscribers.get(job_id, set()))
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(event)
                except queue.Empty:
                    pass


def queue_get_with_timeout(subscriber: queue.Queue[dict[str, Any]], timeout: float) -> dict[str, Any] | None:
    try:
        return subscriber.get(timeout=timeout)
    except queue.Empty:
        return None
