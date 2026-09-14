"""Bounded queues that preserve recent speech when processing falls behind."""

import logging
import queue
import time

log = logging.getLogger(__name__)


class LatestQueue(queue.Queue):
    """Nonblocking producer queue; evict the oldest waiting item on overload."""

    def __init__(self, maxsize: int, name: str):
        super().__init__(maxsize=max(1, int(maxsize)))
        self.name = name
        self.dropped = 0

    def put_nowait(self, item):
        # Keep eviction and insertion atomic relative to consumers/producers.
        with self.not_full:
            if self._qsize() >= self.maxsize:
                self._get()
                self.unfinished_tasks -= 1
                self.dropped += 1
                log.warning("%s overloaded: skipped oldest chunk (%d total)", self.name, self.dropped)
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


def expired(item: dict, max_age: float) -> bool:
    """Age is measured from approximate speech start, including segmentation."""
    start = item.get("segment_start")
    if start is not None and time.monotonic() - start > max_age:
        log.warning("Skipping speech older than %.1fs to recover live translation", max_age)
        return True
    return False
