"""Global mutable pipeline state shared across all modules."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PipelineStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class LatencyRecord:
    segment_duration: float = 0.0
    stt_duration: float = 0.0
    translation_duration: float = 0.0
    tts_duration: float = 0.0
    total_pipeline: float = 0.0
    estimated_user_delay: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class PipelineState:
    status: PipelineStatus = PipelineStatus.STOPPED
    audio_device_name: str = ""
    audio_device_index: Optional[int] = None
    listeners: int = 0
    last_spanish_text: str = ""
    last_german_text: str = ""
    last_error: str = ""
    last_latency: Optional[LatencyRecord] = None
    stt_queue_depth: int = 0
    tts_queue_depth: int = 0
    stream_queue_depth: int = 0
    segment_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_status(self, status: PipelineStatus) -> None:
        with self._lock:
            self.status = status

    def is_running(self) -> bool:
        with self._lock:
            return self.status == PipelineStatus.RUNNING

    def is_paused(self) -> bool:
        with self._lock:
            return self.status == PipelineStatus.PAUSED

    def is_active(self) -> bool:
        """True when pipeline should process audio (running but not paused/stopped)."""
        with self._lock:
            return self.status == PipelineStatus.RUNNING

    def to_dict(self) -> dict:
        with self._lock:
            lr = self.last_latency
            return {
                "status": self.status.value,
                "audio_device": self.audio_device_name,
                "audio_device_index": self.audio_device_index,
                "listeners": self.listeners,
                "last_spanish": self.last_spanish_text,
                "last_german": self.last_german_text,
                "last_error": self.last_error,
                "stt_queue_depth": self.stt_queue_depth,
                "tts_queue_depth": self.tts_queue_depth,
                "stream_queue_depth": self.stream_queue_depth,
                "segment_count": self.segment_count,
                "latency": {
                    "segment_s": round(lr.segment_duration, 2),
                    "stt_s": round(lr.stt_duration, 2),
                    "translation_s": round(lr.translation_duration, 2),
                    "tts_s": round(lr.tts_duration, 2),
                    "pipeline_s": round(lr.total_pipeline, 2),
                    "user_delay_s": round(lr.estimated_user_delay, 2),
                } if lr else None,
            }


# Singleton accessed by all modules
pipeline_state = PipelineState()
