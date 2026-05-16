# AI-Translation — Setup Guide

## System Requirements

- Windows 10/11 (64-bit)
- Python 3.11 or 3.12
- Behringer X32 (or any Windows audio input)
- Local WiFi network
- Optional: NVIDIA GPU (RTX 3050 or better) for faster transcription

---

## 1. Python Environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

---

## 2. Install PyTorch (GPU recommended)

**With CUDA (RTX 3050):**
```bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bat
pip install torch
```

---

## 3. Install dependencies

```bat
pip install -r requirements.txt
```

---

## 4. Download a Piper German voice

1. Go to: https://huggingface.co/rhasspy/piper-voices/tree/main/de/de_DE/thorsten/medium
2. Download both files:
   - `de_DE-thorsten-medium.onnx`
   - `de_DE-thorsten-medium.onnx.json`
3. Place them in: `models/piper/`

Your `config.yaml` already points to `models/piper/de_DE-thorsten-medium.onnx`.

**Install Piper as a Python package (recommended for Windows):**
```bat
pip install piper-tts
```

If the package is unavailable, download the Piper binary from:
https://github.com/rhasspy/piper/releases
and place `piper.exe` somewhere on your PATH.

---

## 5. Configure

Edit `config.yaml`:

- `audio.device`: Set to `"Behringer"` (auto-detect) or a device index.
  Leave as `null` to be prompted on startup.
- `stt.model_name`: Start with `"medium"`. Use `"large-v3-turbo"` on GPU.
- `stt.device`: `"cuda"` (GPU) or `"cpu"`.
- `midi.enabled`: Set to `true` if you want MIDI control from FreeShow.

---

## 6. Run

Double-click `run.bat` or run:

```bat
python main.py
```

The server will:
1. Print available audio devices and select the Behringer X32 if found.
2. Start the web server.
3. Print a QR code for the listener URL.
4. Open the operator page at: `http://<your-LAN-IP>:8000/operator`

---

## 7. Listener access

Mobile users scan the QR code or navigate to:
```
http://<LAN-IP>:8000/listen
```
They tap the large button to start receiving German audio.

---

## 8. Diagnostics

- **Test audio device**: `run_test_audio.bat` → saves `debug/test_capture.wav`
- **List devices**: `run_list_devices.bat`
- **Health check**: `http://localhost:8000/health`
- **Status JSON**: `http://localhost:8000/status`
- **Logs**: `logs/app.log`
- **Debug segments**: Set `debug.dump_segments: true` in config.yaml

---

## 9. MIDI (FreeShow / external controller)

1. Set `midi.enabled: true` in config.yaml.
2. Set `midi.device_name` to a partial name of your MIDI port.
3. Configure `midi.mapping` to match your controller's CC numbers.
4. Set `debug.log_midi_events: true` to see incoming events in the log,
   so you can identify the right CC numbers.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No audio detected | Run `run_list_devices.bat`, find your device index, set `audio.device` |
| CUDA out of memory | Switch to `stt.model_name: small` and `compute_type: int8` |
| Piper not found | Install `piper-tts` via pip, or place `piper.exe` on PATH |
| iPhone won't play audio | Safari requires a user tap before audio can play — the button handles this |
| High latency | Reduce `vad.min_silence_ms` to 500, use `stt.model_name: small` |
| Translation quality low | Switch to `large-v3-turbo` if GPU memory allows |
