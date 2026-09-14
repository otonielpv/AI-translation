"""German TTS using Piper (local ONNX-based TTS engine).

Piper is invoked as a subprocess since it ships as a standalone binary
or Python package. We use the `piper` Python package if available,
otherwise fall back to calling the piper binary.

Output: WAV bytes (PCM 16-bit, mono) suitable for browser decoding.
"""

from __future__ import annotations

import io
import logging
import queue
import struct
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from src.realtime import expired

log = logging.getLogger(__name__)

_INTER_CHUNK_SILENCE_MS_DEFAULT = 200


class TTSBackend:
    """Abstract interface for TTS backends."""

    def synthesize(self, text: str) -> tuple[bytes, float]:
        """Return (wav_bytes, duration_seconds)."""
        raise NotImplementedError


class PiperTTS(TTSBackend):
    """
    Synthesizes German speech using Piper TTS.

    Tries the `piper` Python package first; falls back to subprocess.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str = "",
        inter_chunk_silence_ms: int = _INTER_CHUNK_SILENCE_MS_DEFAULT,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.inter_chunk_silence_ms = inter_chunk_silence_ms
        self._piper = None  # piper Python object if available
        self._use_subprocess = False
        self._sample_rate = 22050  # default; updated after model load

    def _auto_config(self) -> Path:
        # Try exact <model>.onnx.json first
        candidate = Path(str(self.model_path) + ".json")
        if candidate.is_file():
            return candidate
        # Downloads may prefix the voice name with their repository path.
        # Never pick an unrelated voice's JSON or an arbitrary first match.
        jsons = sorted(
            p for p in self.model_path.parent.glob("*.json")
            if p.is_file() and p.name.endswith("_" + candidate.name)
        )
        if len(jsons) == 1:
            return jsons[0]
        if len(jsons) > 1:
            raise ValueError(
                "Multiple Piper configurations match this voice: "
                + ", ".join(str(p) for p in jsons)
                + ". Set tts.piper_config to the correct file in config.yaml."
            )
        raise FileNotFoundError(
            f"Piper voice configuration not found: {candidate}. "
            "Download the .onnx.json file for this exact voice alongside the .onnx model, "
            "or set tts.piper_config to its existing path in config.yaml."
        )

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Piper model not found: {self.model_path}\n"
                "Download a German voice from:\n"
                "  https://huggingface.co/rhasspy/piper-voices\n"
                "Recommended: de_DE-thorsten-medium.onnx\n"
                "Place it in models/piper/"
            )
        if self.config_path is None or not self.config_path.is_file():
            configured_path = self.config_path
            self.config_path = self._auto_config()
            if configured_path is not None:
                log.warning("Piper configuration %s missing; using %s", configured_path, self.config_path)
        log.info("Piper voice configuration: %s", self.config_path)
        try:
            from piper import PiperVoice  # type: ignore
            cfg = str(self.config_path)
            self._piper = PiperVoice.load(str(self.model_path), config_path=cfg, use_cuda=False)
            self._sample_rate = self._piper.config.sample_rate
            log.info("Piper loaded via Python package. sample_rate=%d", self._sample_rate)
        except ImportError:
            log.warning(
                "piper Python package not found: subprocess fallback reloads the voice for each chunk. "
                "Install piper-tts in this environment to reduce live translation latency."
            )
            self._use_subprocess = True
            self._detect_sample_rate()

    def _detect_sample_rate(self) -> None:
        """Try to read sample_rate from the .json config."""
        cfg_path = self.config_path
        if not cfg_path.exists():
            cfg_path = Path(str(self.model_path) + ".json")
        if cfg_path.exists():
            import json
            try:
                data = json.loads(cfg_path.read_text())
                self._sample_rate = data.get("audio", {}).get("sample_rate", 22050)
                log.info("Piper config sample_rate=%d (from %s)", self._sample_rate, cfg_path)
            except Exception:
                pass

    def _silence_bytes(self) -> bytes:
        """Return PCM silence for inter_chunk_silence_ms."""
        n_frames = int(self._sample_rate * self.inter_chunk_silence_ms / 1000)
        return b"\x00" * n_frames * 2  # 16-bit mono

    def _make_wav(self, pcm_bytes: bytes) -> bytes:
        """Wrap raw 16-bit mono PCM in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def _synthesize_python(self, text: str) -> bytes:
        # piper.PiperVoice.synthesize() returns an iterable of AudioChunk objects.
        # Each chunk has audio_int16_bytes (raw 16-bit PCM) and sample_rate.
        pcm_chunks: list[bytes] = []
        for chunk in self._piper.synthesize(text):
            pcm_chunks.append(chunk.audio_int16_bytes)
            if self._sample_rate != chunk.sample_rate:
                self._sample_rate = chunk.sample_rate
        if not pcm_chunks:
            log.warning("Piper synthesize() returned no audio chunks for: %r", text)
            return b""
        return self._make_wav(b"".join(pcm_chunks))

    def _synthesize_subprocess(self, text: str) -> bytes:
        cmd = ["piper", "--model", str(self.model_path), "--output_raw"]
        if self.config_path.exists():
            cmd += ["--config", str(self.config_path)]
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Piper subprocess failed: {result.stderr.decode()}")
        # result.stdout is raw PCM (16-bit mono)
        pcm = result.stdout
        silence = self._silence_bytes()
        return self._make_wav(pcm + silence)

    def synthesize(self, text: str) -> tuple[bytes, float]:
        if not text.strip():
            return b"", 0.0
        t0 = time.monotonic()
        if self._use_subprocess:
            wav = self._synthesize_subprocess(text)
        else:
            base_wav = self._synthesize_python(text)
            if not base_wav:
                return b"", 0.0
            # Extract PCM from the WAV, append silence, re-wrap
            with io.BytesIO(base_wav) as f:
                with wave.open(f, "rb") as wf:
                    pcm = wf.readframes(wf.getnframes())
            wav = self._make_wav(pcm + self._silence_bytes())
        duration = time.monotonic() - t0
        log.info("[TTS] synthesized %d bytes in %.2fs", len(wav), duration)
        return wav, duration


class TTSWorker:
    """
    Reads from `translation_queue`, calls TTS, puts WAV bytes into `audio_queue`.

    Input:  {"text_de": str, "text_es": str, "segment_duration": float,
              "stt_duration": float, "translation_duration": float, "segment_start": float}
    Output: {"wav": bytes, "latency": LatencyRecord-dict}
    """

    def __init__(
        self,
        translation_queue: queue.Queue,
        audio_queue: queue.Queue,
        tts: TTSBackend,
        state,  # PipelineState
        dump_tts: bool = False,
        dump_dir: Path = Path("debug"),
        max_segment_age: float = 7.0,
    ):
        self.translation_queue = translation_queue
        self.audio_queue = audio_queue
        self.tts = tts
        self.state = state
        self.dump_tts = dump_tts
        self.dump_dir = dump_dir
        self.max_segment_age = max_segment_age
        self._stopped = False
        self._tts_idx = 0

    def stop(self) -> None:
        self._stopped = True

    def run(self) -> None:
        while not self._stopped:
            try:
                item = self.translation_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            text_de: str = item["text_de"]
            if not text_de or not self.state.is_active() or expired(item, self.max_segment_age):
                continue

            try:
                wav, tts_dur = self.tts.synthesize(text_de)
            except Exception as exc:
                log.exception("TTS error: %s", exc)
                continue
            if not wav or not self.state.is_active() or expired(item, self.max_segment_age):
                continue

            if self.dump_tts and wav:
                self.dump_dir.mkdir(exist_ok=True)
                path = self.dump_dir / f"tts_{self._tts_idx:04d}_de.wav"
                path.write_bytes(wav)
                self._tts_idx += 1

            now = time.monotonic()
            seg_dur = item.get("segment_duration", 0.0)
            stt_dur = item.get("stt_duration", 0.0)
            tr_dur  = item.get("translation_duration", 0.0)
            seg_start = item.get("segment_start", now)

            total_pipeline = stt_dur + tr_dur + tts_dur
            user_delay = now - seg_start  # segment collection and processing queues included

            log.info(
                "[latency] segment=%.2fs stt=%.2fs translation=%.2fs tts=%.2fs "
                "total_pipeline=%.2fs estimated_user_delay=%.2fs",
                seg_dur, stt_dur, tr_dur, tts_dur, total_pipeline, user_delay,
            )

            from src.state import LatencyRecord
            lr = LatencyRecord(
                segment_duration=seg_dur,
                stt_duration=stt_dur,
                translation_duration=tr_dur,
                tts_duration=tts_dur,
                total_pipeline=total_pipeline,
                estimated_user_delay=user_delay,
            )
            with self.state._lock:
                self.state.last_latency = lr
                self.state.tts_queue_depth = self.audio_queue.qsize()

            try:
                self.audio_queue.put_nowait({"wav": wav, "latency": lr})
            except queue.Full:
                log.warning("Audio broadcast queue full, dropping TTS chunk.")

        log.info("TTSWorker stopped.")


def build_tts(cfg: dict) -> TTSBackend:
    backend = cfg.get("tts", {}).get("backend", "piper")
    if backend == "piper":
        model = cfg.get("tts", {}).get("piper_model", "models/piper/de_DE-thorsten-medium.onnx")
        config = cfg.get("tts", {}).get("piper_config", "")
        silence_ms = cfg.get("tts", {}).get("inter_chunk_silence_ms", 200)
        t = PiperTTS(model_path=model, config_path=config, inter_chunk_silence_ms=silence_ms)
        t.load()
        return t
    raise ValueError(f"Unknown TTS backend: {backend!r}")
