"""Voice Activity Detection using Silero VAD.

Reads float32 mono 16 kHz frames from frame_queue and emits complete
speech segments into segment_queue.
"""

from __future__ import annotations

import logging
import queue
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_SILERO_SR = 16000
_CHUNK = 512  # Silero v5 requires exactly 512 samples at 16 kHz


class VADSegmenter:
    """
    Consumes resampled 16 kHz mono frames, applies Silero VAD per 512-sample
    chunk, and emits speech segments as dicts:
      {"audio": np.ndarray float32, "duration": float}
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        segment_queue: queue.Queue,
        sample_rate: int = 16000,
        pre_roll_ms: int = 300,
        post_roll_ms: int = 400,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
        max_segment_seconds: float = 8.0,
        threshold: float = 0.3,
        input_gain: float = 1.0,
        dump_segments: bool = False,
        dump_dir: Path = Path("debug"),
    ):
        assert sample_rate == _SILERO_SR, f"VAD requires 16000 Hz input, got {sample_rate}"
        self.frame_queue = frame_queue
        self.segment_queue = segment_queue
        self.sr = sample_rate
        self.pre_roll = int(pre_roll_ms * sample_rate / 1000)
        self.post_roll = int(post_roll_ms * sample_rate / 1000)
        self.min_speech = int(min_speech_ms * sample_rate / 1000)
        self.min_silence = int(min_silence_ms * sample_rate / 1000)
        self.max_segment = int(max_segment_seconds * sample_rate)
        self.threshold = threshold
        self.input_gain = float(input_gain)
        self.dump_segments = dump_segments
        self.dump_dir = dump_dir
        self._model = None
        self._seg_idx = 0
        self._stopped = False

    def _load(self) -> None:
        import torch

        # Load from local hub cache without hitting the network.
        # Falls back to a full download if the cache doesn't exist yet.
        try:
            model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad",
                force_reload=False, trust_repo=True,
                skip_validation=True,
            )
        except TypeError:
            # Older torch versions don't have skip_validation
            model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad",
                force_reload=False, trust_repo=True,
            )
        model.reset_states()
        self._model = model
        log.info("Silero VAD model loaded.")

    def _prob(self, chunk: np.ndarray) -> float:
        import torch
        gained = chunk * self.input_gain if self.input_gain != 1.0 else chunk
        # Clip to [-1, 1] so we don't feed extreme values to the model
        gained = np.clip(gained, -1.0, 1.0)
        t = torch.from_numpy(gained.astype(np.float32))
        with torch.no_grad():
            return self._model(t, self.sr).item()

    def _emit(self, frames: np.ndarray) -> None:
        dur = len(frames) / self.sr
        if len(frames) < self.min_speech:
            log.debug("VAD: segment too short (%.2fs), discarding", dur)
            return
        log.info("VAD: segment %.2fs → STT queue", dur)

        if self.dump_segments:
            self.dump_dir.mkdir(exist_ok=True)
            p = self.dump_dir / f"seg_{self._seg_idx:04d}_es.wav"
            pcm = (frames * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(str(p), "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2)
                wf.setframerate(self.sr); wf.writeframes(pcm.tobytes())
            log.debug("VAD: saved segment to %s", p)

        self._seg_idx += 1
        # Reset model state between utterances
        self._model.reset_states()
        # Apply gain to the audio sent to STT (helps Whisper on quiet mics)
        if self.input_gain != 1.0:
            frames = np.clip(frames * self.input_gain, -1.0, 1.0)
        try:
            self.segment_queue.put_nowait({"audio": frames, "duration": dur})
        except queue.Full:
            log.warning("VAD: segment queue full, dropping")

    def stop(self) -> None:
        self._stopped = True

    def run(self) -> None:
        """Blocking loop — run in a dedicated daemon thread."""
        self._load()

        in_speech = False
        silence_frames = 0

        pre_buf: list[np.ndarray] = []
        pre_len = 0
        active: list[np.ndarray] = []
        active_len = 0

        leftover = np.empty(0, dtype=np.float32)
        chunks_processed = 0
        speech_chunks = 0

        while not self._stopped:
            # Collect available frames
            batch: list[np.ndarray] = []
            try:
                batch.append(self.frame_queue.get(timeout=0.05))
            except queue.Empty:
                continue

            while True:
                try:
                    batch.append(self.frame_queue.get_nowait())
                except queue.Empty:
                    break

            audio = np.concatenate([leftover] + batch)
            pos = 0

            while pos + _CHUNK <= len(audio):
                chunk = audio[pos: pos + _CHUNK]
                pos += _CHUNK
                chunks_processed += 1

                prob = self._prob(chunk)
                is_speech = prob >= self.threshold

                if is_speech:
                    speech_chunks += 1

                # Periodic debug so operator can see VAD is alive
                if chunks_processed % 200 == 0:
                    log.debug(
                        "VAD: %d chunks processed, %d speech (%.1f%%), in_speech=%s",
                        chunks_processed, speech_chunks,
                        100 * speech_chunks / chunks_processed, in_speech,
                    )

                if not in_speech:
                    pre_buf.append(chunk)
                    pre_len += _CHUNK
                    # Keep only the last pre_roll frames
                    while pre_len - len(pre_buf[0]) >= self.pre_roll:
                        removed = pre_buf.pop(0)
                        pre_len -= len(removed)

                    if is_speech:
                        log.debug("VAD: speech start (prob=%.2f)", prob)
                        in_speech = True
                        silence_frames = 0
                        active = list(pre_buf)
                        active_len = pre_len
                        pre_buf = []
                        pre_len = 0
                else:
                    active.append(chunk)
                    active_len += _CHUNK

                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += _CHUNK

                    if active_len >= self.max_segment:
                        log.debug("VAD: max segment reached, force-emitting")
                        self._emit(np.concatenate(active))
                        in_speech = False
                        silence_frames = 0
                        active = []
                        active_len = 0
                        pre_buf = []
                        pre_len = 0

                    elif silence_frames >= self.min_silence:
                        log.debug("VAD: speech end (silence=%.2fs)", silence_frames / self.sr)
                        segment = np.concatenate(active)
                        # Keep post_roll of the trailing silence
                        keep = max(active_len - silence_frames + min(silence_frames, self.post_roll), 0)
                        self._emit(segment[:keep])
                        in_speech = False
                        silence_frames = 0
                        active = []
                        active_len = 0
                        pre_buf = []
                        pre_len = 0

            leftover = audio[pos:]

        log.info("VAD segmenter stopped. total_chunks=%d speech_chunks=%d", chunks_processed, speech_chunks)
