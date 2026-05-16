"""Audio broadcast: one producer → many WebSocket listeners.

Uses asyncio queues per client. The TTSWorker puts WAV chunks into
a sync queue; a bridge task picks them up and fans out to all
connected client queues.
"""

from __future__ import annotations

import asyncio
import logging
import queue as sync_queue
import time

log = logging.getLogger(__name__)


class AudioBroadcaster:
    """
    Bridges the sync TTS audio queue with multiple async WebSocket clients.

    Call `run(loop)` from the asyncio event loop to start the bridge task.
    """

    def __init__(self, audio_queue: sync_queue.Queue, max_queue: int = 20):
        self.audio_queue = audio_queue
        self.max_queue = max_queue
        self._clients: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    def add_client(self, q: asyncio.Queue) -> None:
        self._clients.add(q)
        log.debug("Broadcaster: client added. total=%d", len(self._clients))

    def remove_client(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)
        log.debug("Broadcaster: client removed. total=%d", len(self._clients))

    async def run(self) -> None:
        """Async loop — run as asyncio task inside the FastAPI event loop."""
        log.info("AudioBroadcaster running.")
        loop = asyncio.get_event_loop()
        while True:
            # Poll the sync queue in a thread-pool executor so we don't block
            try:
                item = await loop.run_in_executor(
                    None, lambda: self.audio_queue.get(timeout=0.1)
                )
            except sync_queue.Empty:
                await asyncio.sleep(0)
                continue
            except Exception:
                await asyncio.sleep(0.05)
                continue

            wav: bytes = item.get("wav", b"")
            if not wav:
                continue

            dead: list[asyncio.Queue] = []
            for client_q in list(self._clients):
                if client_q.full():
                    # Drop oldest chunk for this slow client
                    try:
                        client_q.get_nowait()
                        log.debug("Broadcaster: dropped old chunk for slow client.")
                    except asyncio.QueueEmpty:
                        pass
                try:
                    client_q.put_nowait(wav)
                except asyncio.QueueFull:
                    dead.append(client_q)

            for dq in dead:
                self._clients.discard(dq)
