"""Play a video/audio file into the pipeline as if it were live mic input.

Uses ffmpeg to decode any container (.mp4, .mkv, .wav, .mp3, .m4a, …) to
16 kHz mono float32 PCM, then pushes 512-sample chunks into the frame_queue
at real-time speed so VAD/STT/translation/TTS run normally.

Requires ffmpeg.exe in PATH. Install on Windows with:
    winget install Gyan.FFmpeg
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000
CHUNK_FRAMES = 512  # matches VAD's expected chunk size


class FilePlayer:
    """
    Decodes a media file with ffmpeg and feeds its audio into frame_queue
    at real-time rate. A drop-in replacement for AudioCapture during testing.
    """

    def __init__(
        self,
        file_path: Path,
        frame_queue: queue.Queue,
        target_sample_rate: int = TARGET_SR,
        realtime: bool = True,
    ):
        self.file_path = Path(file_path)
        self.frame_queue = frame_queue
        self.target_sample_rate = target_sample_rate
        self.realtime = realtime
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    def start(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Media file not found: {self.file_path}")
        if not self.ffmpeg_available():
            raise RuntimeError(
                "ffmpeg not found in PATH. Install with: winget install Gyan.FFmpeg"
            )

        log.info("FilePlayer starting: %s", self.file_path)
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="file-player", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        log.info("FilePlayer stopped.")

    def _run(self) -> None:
        """Spawn ffmpeg, read PCM, feed into queue at real-time rate."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(self.file_path),
            "-f", "f32le",  # 32-bit float PCM little-endian
            "-acodec", "pcm_f32le",
            "-ac", "1",  # mono
            "-ar", str(self.target_sample_rate),
            "-",  # stdout
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception as exc:
            log.error("Failed to start ffmpeg: %s", exc)
            return

        bytes_per_sample = 4  # float32
        chunk_bytes = CHUNK_FRAMES * bytes_per_sample
        chunk_period = CHUNK_FRAMES / self.target_sample_rate  # seconds per chunk

        next_deadline = time.monotonic()
        total_samples = 0

        try:
            while not self._stopped:
                raw = self._proc.stdout.read(chunk_bytes)
                if not raw:
                    break

                # Pad last chunk with zeros if file ended mid-chunk
                if len(raw) < chunk_bytes:
                    raw = raw + b"\x00" * (chunk_bytes - len(raw))

                samples = np.frombuffer(raw, dtype=np.float32).copy()
                total_samples += len(samples)

                try:
                    self.frame_queue.put_nowait(samples)
                except queue.Full:
                    # Drop the oldest chunk to keep up
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(samples)
                    except queue.Empty:
                        pass

                if self.realtime:
                    next_deadline += chunk_period
                    sleep_for = next_deadline - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        # We've fallen behind — reset deadline so we don't burst
                        next_deadline = time.monotonic()
        finally:
            if self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2)
                except Exception:
                    pass
            stderr = b""
            try:
                stderr = self._proc.stderr.read() or b""
            except Exception:
                pass
            if stderr:
                log.debug("ffmpeg stderr: %s", stderr.decode(errors="replace"))
            duration_s = total_samples / self.target_sample_rate
            log.info("FilePlayer finished. fed %.2fs of audio into pipeline.", duration_s)
