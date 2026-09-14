"""Pipeline orchestrator.

Wires audio capture → VAD → STT → Translation → TTS → Broadcaster.
Provides PipelineController with start/pause/resume/stop.

All CPU/GPU-heavy work runs in daemon threads.
The broadcaster bridge runs as an asyncio task inside the FastAPI loop.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from src.state import PipelineState, PipelineStatus, pipeline_state

log = logging.getLogger(__name__)


class PipelineController:
    """
    High-level controller. Created once at startup; the FastAPI app and MIDI
    controller both hold a reference.
    """

    def __init__(self, cfg: dict, state: PipelineState, broadcaster):
        self.cfg = cfg
        self.state = state
        self.broadcaster = broadcaster

        # Inter-thread queues
        self._frame_queue: queue.Queue = queue.Queue(
            maxsize=int(
                cfg.get("audio", {}).get("ring_buffer_seconds", 30)
                * cfg.get("audio", {}).get("sample_rate", 16000)
                / 1024
                * 2
            )
        )
        self._segment_queue: queue.Queue = queue.Queue(maxsize=50)
        self._text_queue: queue.Queue = queue.Queue(maxsize=50)
        self._translation_queue: queue.Queue = queue.Queue(maxsize=50)
        # audio_queue is owned by broadcaster
        self._audio_queue: queue.Queue = broadcaster.audio_queue

        self._capture = None
        self._vad = None
        self._stt = None
        self._translator = None
        self._tts_worker = None
        self._midi = None

        self._threads: list[threading.Thread] = []
        self._translation_thread: Optional[threading.Thread] = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._stopping = False  # instance variable, not class variable
        self._start_lock = threading.Lock()
        self._file_player = None  # set when using a file as source instead of mic

    # ------------------------------------------------------------------
    # Load models (blocking — call before start, or on first start)
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            self._do_load()

    def _do_load(self) -> None:
        cfg = self.cfg
        sr = cfg.get("audio", {}).get("sample_rate", 16000)

        # Audio capture — device is already resolved in main.py before uvicorn starts.
        # cfg["audio"]["device"] is guaranteed to be an integer index at this point.
        from src.audio.capture import AudioCapture

        dev_idx = cfg.get("audio", {}).get("device")
        if dev_idx is None or not isinstance(dev_idx, int):
            raise RuntimeError(
                "Audio device index not set. This is a bug — device should be resolved before pipeline start."
            )

        import sounddevice as sd
        dev_info = sd.query_devices(dev_idx)
        self.state.audio_device_index = dev_idx
        self.state.audio_device_name = dev_info["name"]

        self._capture = AudioCapture(
            device_index=dev_idx,
            target_sample_rate=sr,
            channels=cfg.get("audio", {}).get("channels", 1),
            ring_buffer_seconds=cfg.get("audio", {}).get("ring_buffer_seconds", 30),
            frame_queue=self._frame_queue,
        )

        # VAD
        from src.audio.vad import VADSegmenter

        vad_cfg = cfg.get("vad", {})
        debug_cfg = cfg.get("debug", {})
        self._vad = VADSegmenter(
            frame_queue=self._frame_queue,
            segment_queue=self._segment_queue,
            sample_rate=sr,
            pre_roll_ms=vad_cfg.get("pre_roll_ms", 300),
            post_roll_ms=vad_cfg.get("post_roll_ms", 400),
            min_speech_ms=vad_cfg.get("min_speech_ms", 250),
            min_silence_ms=vad_cfg.get("min_silence_ms", 700),
            max_segment_seconds=vad_cfg.get("max_segment_seconds", 8.0),
            threshold=vad_cfg.get("threshold", 0.3),
            input_gain=vad_cfg.get("input_gain", 1.0),
            dump_segments=debug_cfg.get("dump_segments", False),
            dump_dir=Path("debug"),
        )

        # STT
        from src.stt.whisper_backend import WhisperBackend

        stt_cfg = cfg.get("stt", {})
        self._stt = WhisperBackend(
            segment_queue=self._segment_queue,
            text_queue=self._text_queue,
            model_name=stt_cfg.get("model_name", "medium"),
            device=stt_cfg.get("device", "auto"),
            compute_type=stt_cfg.get("compute_type", "int8_float16"),
            beam_size=stt_cfg.get("beam_size", 5),
            language=stt_cfg.get("language", "es"),
        )

        # Load Whisper model now (it's only called via transcribe() in _stt_loop, not via run())
        self._stt.ensure_loaded()

        # Translation
        from src.translation.helsinki import build_translator

        self._translator = build_translator(cfg)

        # TTS
        from src.tts.piper_backend import TTSWorker, build_tts

        tts_engine = build_tts(cfg)
        self._tts_worker = TTSWorker(
            translation_queue=self._translation_queue,
            audio_queue=self._audio_queue,
            tts=tts_engine,
            state=self.state,
            dump_tts=debug_cfg.get("dump_tts", False),
            dump_dir=Path("debug"),
        )

        self._loaded = True
        log.info("All pipeline components loaded.")

    # ------------------------------------------------------------------
    # Translation bridge (runs in a thread)
    # ------------------------------------------------------------------
    def _translation_loop(self) -> None:
        while True:
            try:
                item = self._text_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stopping:
                    break
                continue

            if not self.state.is_active():
                log.debug("Pipeline paused/stopped — dropping STT result.")
                continue

            text_es: str = item["text"]
            seg_dur: float = item.get("segment_duration", 0.0)
            stt_dur: float = item.get("stt_duration", 0.0)
            seg_start: float = item.get("segment_start", time.monotonic())

            with self.state._lock:
                self.state.last_spanish_text = text_es
                self.state.segment_count += 1

            try:
                t0 = time.monotonic()
                text_de, tr_dur = self._translator.translate(text_es)
                with self.state._lock:
                    self.state.last_german_text = text_de
            except Exception as exc:
                log.exception("Translation error: %s", exc)
                continue

            if not text_de:
                continue

            try:
                self._translation_queue.put_nowait({
                    "text_de": text_de,
                    "text_es": text_es,
                    "segment_duration": seg_dur,
                    "stt_duration": stt_dur,
                    "translation_duration": tr_dur,
                    "segment_start": seg_start,
                })
            except queue.Full:
                log.warning("Translation queue full, dropping.")

        log.info("Translation loop exited.")

    # ------------------------------------------------------------------
    # VAD/STT wrapper that injects segment_start timestamp
    # ------------------------------------------------------------------
    def _stt_loop(self) -> None:
        """Wraps WhisperBackend.run() to inject segment_start timestamps."""
        import numpy as np

        while True:
            try:
                item = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stopping:
                    break
                continue

            if not self.state.is_active():
                continue

            item["segment_start"] = time.monotonic()
            audio = item["audio"]
            seg_dur = item["duration"]

            with self.state._lock:
                self.state.stt_queue_depth = self._segment_queue.qsize()

            log.info("[STT] transcribing %.2fs segment …", seg_dur)
            try:
                result = self._stt.transcribe(audio, seg_dur)
                if result is None:
                    log.warning("[STT] transcribe() returned None (model not loaded?)")
                    continue
                text, stt_dur = result
                if not text:
                    log.info("[STT] empty result for %.2fs segment", seg_dur)
                    continue
                log.info("[STT] → %r  (%.2fs)", text, stt_dur)
                self._text_queue.put_nowait({
                    "text": text,
                    "segment_duration": seg_dur,
                    "stt_duration": stt_dur,
                    "segment_start": item["segment_start"],
                })
            except queue.Full:
                log.warning("Text queue full, dropping STT result.")
            except Exception as exc:
                log.exception("STT error in pipeline: %s", exc)

        log.info("STT loop exited.")

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    def start(self, source_file: Optional[str] = None) -> None:
        """
        Start the pipeline.

        If source_file is given, audio is read from that media file (any
        ffmpeg-supported format) instead of the live microphone. Useful for
        testing translation against a recorded preach.
        """
        with self._start_lock:
            if self.state.status != PipelineStatus.STOPPED:
                log.warning("Pipeline already running or paused — ignoring start.")
                return
            # Mark as running immediately inside the lock to block concurrent starts
            self.state.set_status(PipelineStatus.RUNNING)

        try:
            self._ensure_loaded()
        except Exception as exc:
            log.error("Pipeline load error: %s", exc)
            with self.state._lock:
                self.state.last_error = str(exc)
                self.state.status = PipelineStatus.STOPPED
            return

        self._stopping = False

        # Reset stopped flags on workers so they can run again after a stop/start cycle
        self._vad._stopped = False
        self._tts_worker._stopped = False
        self._stt._stopped = False

        # Start audio source: either the mic or a file player
        if source_file:
            from src.audio.file_player import FilePlayer
            from pathlib import Path as _Path
            try:
                self._file_player = FilePlayer(
                    file_path=_Path(source_file),
                    frame_queue=self._frame_queue,
                    target_sample_rate=self.cfg.get("audio", {}).get("sample_rate", 16000),
                )
                self._file_player.start()
                self.state.audio_device_name = f"[FILE] {_Path(source_file).name}"
            except Exception as exc:
                log.error("Failed to start file player: %s", exc)
                with self.state._lock:
                    self.state.last_error = str(exc)
                    self.state.status = PipelineStatus.STOPPED
                return
        else:
            self._capture.start()

        # Start worker threads
        self._threads = [
            threading.Thread(target=self._vad.run, name="vad", daemon=True),
            threading.Thread(target=self._stt_loop, name="stt", daemon=True),
            threading.Thread(target=self._translation_loop, name="translation", daemon=True),
            threading.Thread(target=self._tts_worker.run, name="tts", daemon=True),
        ]
        for t in self._threads:
            t.start()

        log.info("Pipeline started.")

    def pause(self) -> None:
        if self.state.status != PipelineStatus.RUNNING:
            return
        self.state.set_status(PipelineStatus.PAUSED)
        log.info("Pipeline paused.")

    def resume(self) -> None:
        if self.state.status != PipelineStatus.PAUSED:
            return
        # Discard queued audio accumulated during pause
        self._drain_queue(self._frame_queue)
        self._drain_queue(self._segment_queue)
        self._drain_queue(self._text_queue)
        self._drain_queue(self._translation_queue)
        self.state.set_status(PipelineStatus.RUNNING)
        log.info("Pipeline resumed.")

    def toggle_pause(self) -> None:
        if self.state.status == PipelineStatus.RUNNING:
            self.pause()
        elif self.state.status == PipelineStatus.PAUSED:
            self.resume()

    def set_device(self, device_index: int) -> None:
        """Change the audio input device. Must be called while pipeline is stopped."""
        if self.state.status != PipelineStatus.STOPPED:
            log.warning("Cannot change device while pipeline is running — stop first.")
            return
        import sounddevice as sd
        try:
            info = sd.query_devices(device_index)
        except Exception as exc:
            log.error("Invalid device index %d: %s", device_index, exc)
            return
        self.cfg.setdefault("audio", {})["device"] = device_index
        self.state.audio_device_index = device_index
        self.state.audio_device_name = info["name"]
        # Force reload of AudioCapture on next start
        self._capture = None
        self._loaded = False
        log.info("Audio device changed to [%d] %s", device_index, info["name"])

    def stop(self) -> None:
        if self.state.status == PipelineStatus.STOPPED:
            return
        self._stopping = True
        self.state.set_status(PipelineStatus.STOPPED)

        if self._file_player:
            self._file_player.stop()
            self._file_player = None
        if self._capture:
            self._capture.stop()
        if self._vad:
            self._vad.stop()
        if self._tts_worker:
            self._tts_worker.stop()

        # Drain queues so threads unblock
        for q in [self._frame_queue, self._segment_queue, self._text_queue, self._translation_queue]:
            self._drain_queue(q)

        log.info("Pipeline stopped.")

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
