"""Audio input capture using sounddevice with a ring buffer.

Captures at the device's native sample rate, then resamples to 16 kHz mono
float32 before enqueuing. WASAPI devices only support their native rate
(typically 48000 Hz); we never ask the device for 16 kHz directly.

The sounddevice callback does only the minimal work: copy + resample.
All heavy processing happens in downstream threads.
"""

from __future__ import annotations

import logging
import queue
import wave
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000  # Rate expected by VAD and STT


def list_devices(wasapi_only: bool = True) -> list[dict]:
    """
    Return available audio input devices.

    With wasapi_only=True (default), only Windows WASAPI devices are shown.
    Windows exposes each physical device through MME, DirectSound, WASAPI,
    and WDM-KS. WASAPI is the modern API and avoids duplicates.
    Falls back to all APIs if no WASAPI devices are found.
    """
    import sounddevice as sd

    all_devices = sd.query_devices()
    host_apis = sd.query_hostapis()

    result = []
    for i, d in enumerate(all_devices):
        if d["max_input_channels"] <= 0:
            continue
        api_name = host_apis[d["hostapi"]]["name"] if d["hostapi"] < len(host_apis) else ""
        result.append({
            "index": i,
            "name": d["name"],
            "channels": d["max_input_channels"],
            "hostapi": api_name,
            "default_sr": int(d["default_samplerate"]),
        })

    if wasapi_only:
        wasapi = [d for d in result if "wasapi" in d["hostapi"].lower()]
        if wasapi:
            return wasapi

    return result


def get_device_native_rate(device_index: int) -> int:
    """Return the device's default (native) sample rate."""
    import sounddevice as sd
    info = sd.query_devices(device_index)
    return int(info["default_samplerate"])


def find_device(name_or_index) -> Optional[int]:
    """
    Find a device index by partial name (case-insensitive) or integer index.
    Searches WASAPI-only list. Returns None if not found.
    """
    if name_or_index is None:
        return None
    if isinstance(name_or_index, int):
        return name_or_index
    needle = str(name_or_index).lower()
    for d in list_devices():
        if needle in d["name"].lower():
            return d["index"]
    return None


def select_device_interactive(prefer_name: str = "behringer") -> int:
    """Print deduplicated device list and prompt the user to select one."""
    devices = list_devices()
    if not devices:
        raise RuntimeError("No audio input devices found.")

    # Auto-select preferred device
    for d in devices:
        if prefer_name.lower() in d["name"].lower():
            log.info("Auto-selected preferred device: [%d] %s", d["index"], d["name"])
            return d["index"]

    print("\nAvailable audio input devices:")
    for d in devices:
        print(f"  [{d['index']}] {d['name']}  ({d['channels']} ch, {d['default_sr']} Hz)")

    valid = {d["index"] for d in devices}
    while True:
        raw = input("\nEnter device index: ").strip()
        try:
            idx = int(raw)
            if idx in valid:
                return idx
        except ValueError:
            pass
        print(f"Invalid index. Choose from: {sorted(valid)}")


def _find_mme_fallback(device_name: str) -> Optional[int]:
    """Find an MME device index whose name matches device_name (partial, case-insensitive)."""
    import sounddevice as sd
    all_d = sd.query_devices()
    apis = sd.query_hostapis()
    needle = device_name.lower()
    for i, d in enumerate(all_d):
        if d["max_input_channels"] <= 0:
            continue
        api = apis[d["hostapi"]]["name"].lower()
        if "mme" in api and needle[:20] in d["name"].lower():
            return i
    return None


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a float32 mono array from src_rate to dst_rate."""
    if src_rate == dst_rate:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


class AudioCapture:
    """
    Captures audio from an input device at its native sample rate,
    resamples to TARGET_SR (16 kHz) mono float32, and enqueues chunks.

    The frame_queue receives numpy arrays of shape (N,) at TARGET_SR.
    No processing beyond mono-mix and resample happens in the callback.
    """

    # Blocksize at native rate; after resample this becomes ~640 frames at 16 kHz
    # for a 48 kHz device (1920 * 16000/48000 = 640).
    CHUNK_FRAMES_NATIVE = 1920

    def __init__(
        self,
        device_index: int,
        target_sample_rate: int = TARGET_SR,
        channels: int = 1,
        ring_buffer_seconds: float = 30.0,
        frame_queue: Optional[queue.Queue] = None,
    ):
        self.device_index = device_index
        self.target_sample_rate = target_sample_rate
        self.channels = channels
        self.ring_buffer_seconds = ring_buffer_seconds

        # Detect native rate; fall back to target if query fails
        try:
            self.native_rate = get_device_native_rate(device_index)
        except Exception:
            self.native_rate = target_sample_rate

        self.frame_queue: queue.Queue[np.ndarray] = frame_queue or queue.Queue(
            maxsize=int(ring_buffer_seconds * target_sample_rate / 512 * 2)
        )
        self._stream = None
        self._overruns = 0
        self._underruns = 0
        self._leftover = np.empty(0, dtype=np.float32)

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            log.warning("Audio callback status: %s", status)
            self._underruns += 1

        # Mix to mono
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()

        # Resample to target rate if needed
        if self.native_rate != self.target_sample_rate:
            mono = _resample(mono, self.native_rate, self.target_sample_rate)

        try:
            self.frame_queue.put_nowait(mono)
        except queue.Full:
            self._overruns += 1
            if self._overruns % 50 == 1:
                log.warning(
                    "Audio queue full — overrun #%d (queue size %d)",
                    self._overruns, self.frame_queue.qsize(),
                )

    def start(self) -> None:
        import sounddevice as sd

        apis = sd.query_hostapis()
        dev_info = sd.query_devices(self.device_index)
        api_name = apis[dev_info["hostapi"]]["name"] if dev_info["hostapi"] < len(apis) else ""
        is_wasapi = "wasapi" in api_name.lower()

        # WASAPI shared-mode settings avoid exclusive-mode conflicts with
        # other running apps (OBS, Voicemeeter, etc.)
        extra = None
        if is_wasapi:
            try:
                extra = sd.WasapiSettings(exclusive=False)
            except Exception:
                pass  # older sounddevice versions may not support this

        try:
            self._stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.native_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.CHUNK_FRAMES_NATIVE,
                callback=self._callback,
                extra_settings=extra,
            )
            self._stream.start()
        except Exception:
            # WASAPI failed — fall back to MME index for the same device name
            if is_wasapi:
                log.warning(
                    "WASAPI open failed for device %d, falling back to MME...", self.device_index
                )
                mme_idx = _find_mme_fallback(dev_info["name"])
                if mme_idx is not None:
                    self.device_index = mme_idx
                    self.native_rate = get_device_native_rate(mme_idx)
                    self._stream = sd.InputStream(
                        device=mme_idx,
                        samplerate=self.native_rate,
                        channels=self.channels,
                        dtype="float32",
                        blocksize=self.CHUNK_FRAMES_NATIVE,
                        callback=self._callback,
                    )
                    self._stream.start()
                else:
                    raise
            else:
                raise

        log.info(
            "Audio capture started: device=%d native_sr=%d target_sr=%d ch=%d api=%s",
            self.device_index, self.native_rate, self.target_sample_rate, self.channels, api_name,
        )

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info(
                "Audio capture stopped. overruns=%d underruns=%d",
                self._overruns, self._underruns,
            )

    def record_test_wav(
        self,
        duration_seconds: float = 10.0,
        output_path: Path = Path("debug/test_capture.wav"),
    ) -> None:
        """Record duration_seconds of audio and save as 16 kHz mono WAV."""
        import sounddevice as sd

        output_path.parent.mkdir(exist_ok=True)
        total_native = int(self.native_rate * duration_seconds)
        log.info("Recording %.0fs from device %d (native %d Hz) to %s …",
                 duration_seconds, self.device_index, self.native_rate, output_path)
        data = sd.rec(
            total_native,
            samplerate=self.native_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device_index,
        )
        sd.wait()
        mono = data[:, 0] if data.ndim > 1 else data.flatten()
        if self.native_rate != self.target_sample_rate:
            mono = _resample(mono, self.native_rate, self.target_sample_rate)
        pcm16 = (mono * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(output_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(pcm16.tobytes())
        log.info("Test WAV saved: %s  (%d samples at %d Hz)", output_path, len(pcm16), self.target_sample_rate)
