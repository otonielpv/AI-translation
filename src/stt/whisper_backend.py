"""Spanish speech-to-text using faster-whisper.

Runs in a dedicated worker thread. Reads speech segments from
`segment_queue` and puts transcription results into `text_queue`.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class WhisperBackend:
    """
    Transcribes Spanish audio segments using faster-whisper.

    Input:  segment_queue items: {"audio": np.ndarray, "duration": float}
    Output: text_queue items:    {"text": str, "segment_duration": float, "stt_duration": float}
    """

    def __init__(
        self,
        segment_queue: queue.Queue,
        text_queue: queue.Queue,
        model_name: str = "medium",
        device: str = "auto",
        compute_type: str = "int8_float16",
        beam_size: int = 5,
        language: str = "es",
    ):
        self.segment_queue = segment_queue
        self.text_queue = text_queue
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model = None
        self._stopped = False

    def _resolve_device(self) -> tuple[str, str]:
        if self.device != "auto":
            return self.device, self.compute_type
        try:
            import torch
            if torch.cuda.is_available():
                log.info("CUDA available — using GPU for STT.")
                return "cuda", self.compute_type
        except ImportError:
            pass
        log.info("CUDA not available — using CPU for STT (compute_type=int8).")
        return "cpu", "int8"

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        device, compute_type = self._resolve_device()
        log.info(
            "Loading Whisper model '%s' on %s (%s) …",
            self.model_name, device, compute_type,
        )
        t0 = time.monotonic()
        try:
            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:
            log.warning("Failed to load model on %s: %s. Retrying on CPU with int8.", device, exc)
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        log.info("Whisper model loaded in %.1fs.", time.monotonic() - t0)

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._load_model()

    def transcribe(self, audio: np.ndarray, segment_duration: float) -> Optional[str]:
        if self._model is None:
            return None
        t0 = time.monotonic()
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # we handle VAD ourselves
        )
        text_parts = [s.text.strip() for s in segments if s.text.strip()]
        text = " ".join(text_parts).strip()
        stt_dur = time.monotonic() - t0
        log.info("[STT] %.2fs → %r  (stt=%.2fs)", segment_duration, text, stt_dur)
        return text, stt_dur

    def stop(self) -> None:
        self._stopped = True

    def run(self) -> None:
        """Blocking loop — run in a dedicated thread."""
        self._load_model()

        while not self._stopped:
            try:
                item = self.segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            audio: np.ndarray = item["audio"]
            segment_duration: float = item["duration"]

            try:
                result = self.transcribe(audio, segment_duration)
                if result is None:
                    continue
                text, stt_dur = result
                if not text:
                    log.debug("STT returned empty text, skipping.")
                    continue
                self.text_queue.put_nowait({
                    "text": text,
                    "segment_duration": segment_duration,
                    "stt_duration": stt_dur,
                })
            except queue.Full:
                log.warning("STT text queue full, dropping result.")
            except Exception as exc:
                log.exception("STT error: %s", exc)

        log.info("WhisperBackend stopped.")
