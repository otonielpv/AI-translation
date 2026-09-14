"""AI-Translation - entry point.

Usage:
  python main.py                  # normal startup
  python main.py --test-audio     # record 10 s test WAV and exit
  python main.py --list-devices   # print audio devices and exit
  python main.py --config PATH    # use a custom config file
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import queue
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(description="AI Church Translation Server")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--test-audio", action="store_true", help="Record 10 s test WAV and exit")
    p.add_argument("--list-midi", action="store_true", help="List MIDI devices and exit")
    p.add_argument("--device", default=None, help="Audio device index or name (overrides config)")
    p.add_argument("--play-file", default=None, help="Path to a video/audio file to feed into the pipeline (test mode)")
    return p.parse_args()


def _resolve_audio_device(cfg: dict, cli_device=None) -> int:
    """
    Resolve the audio device index before the server starts, so we can
    prompt interactively on the terminal while stdin is still available.

    Priority: CLI --device flag > config.yaml audio.device > interactive prompt.
    """
    from src.audio.capture import find_device, select_device_interactive

    # CLI flag overrides config
    device_cfg = cli_device if cli_device is not None else cfg.get("audio", {}).get("device")

    if device_cfg is not None:
        # Try integer first
        try:
            return int(device_cfg)
        except (TypeError, ValueError):
            pass
        idx = find_device(device_cfg)
        if idx is None:
            from src.audio.capture import list_devices
            print(f"\n[ERROR] Audio device {device_cfg!r} not found.")
            print("Available devices:")
            for d in list_devices():
                print(f"  [{d['index']}] {d['name']}")
            idx = select_device_interactive()
        return idx
    else:
        return select_device_interactive()


def main():
    args = _parse_args()

    # ---------------------------------------------------------------- config
    from src.config import load_config

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)

    # ---------------------------------------------------------------- logging
    import src.logger as logger_setup

    debug_cfg = cfg.get("debug", {})
    logger_setup.setup(debug_cfg.get("log_level", "INFO"))

    import logging
    log = logging.getLogger("main")

    # --------------------------------------------------------- list devices
    if args.list_devices:
        from src.audio.capture import list_devices

        print("\nAvailable audio input devices:")
        for d in list_devices():
            print(f"  [{d['index']}] {d['name']}  ({d['channels']} ch)")
        sys.exit(0)

    # ---------------------------------------------------------- list MIDI
    if args.list_midi:
        from src.midi.controller import list_midi_inputs

        ports = list_midi_inputs()
        if ports:
            print("\nAvailable MIDI input ports:")
            for p in ports:
                print(f"  {p}")
        else:
            print("No MIDI input ports found.")
        sys.exit(0)

    # --------------------------------------------------------- test audio
    if args.test_audio:
        from src.audio.capture import AudioCapture

        dev_idx = _resolve_audio_device(cfg, args.device)
        sr = cfg.get("audio", {}).get("sample_rate", 16000)
        cap = AudioCapture(device_index=dev_idx, sample_rate=sr)
        cap.record_test_wav(duration_seconds=10.0, output_path=Path("debug/test_capture.wav"))
        sys.exit(0)

    # --------------------------------------------------------------- normal run
    log.info("=" * 60)
    log.info("AI-Translation starting up")
    log.info("=" * 60)

    # --- Resolve audio device NOW, before uvicorn takes over the terminal ---
    dev_idx = _resolve_audio_device(cfg, args.device)

    import sounddevice as sd
    dev_info = sd.query_devices(dev_idx)
    dev_name = dev_info["name"]
    log.info("Selected audio device: [%d] %s", dev_idx, dev_name)

    # Persist into config so pipeline doesn't prompt again
    if "audio" not in cfg:
        cfg["audio"] = {}
    cfg["audio"]["device"] = dev_idx

    # Network
    from src.network import get_lan_ip, make_qr_png, print_qr_terminal

    server_cfg = cfg.get("server", {})
    port = server_cfg.get("port", 8000)
    lan_ip = get_lan_ip()
    lan_url = f"http://{lan_ip}:{port}"
    listen_url = f"{lan_url}/listen"

    log.info("Server will be available at: %s", lan_url)
    print_qr_terminal(listen_url)

    try:
        qr_png = make_qr_png(listen_url)
    except Exception as exc:
        log.warning("Could not generate QR PNG: %s", exc)
        qr_png = b""

    # State
    from src.state import pipeline_state

    pipeline_state.audio_device_index = dev_idx
    pipeline_state.audio_device_name = dev_name

    # Broadcaster
    from src.realtime import LatestQueue
    audio_queue = LatestQueue(cfg.get("performance", {}).get("max_pending_segments", 3), "Broadcast")
    from src.streaming.broadcaster import AudioBroadcaster

    broadcaster = AudioBroadcaster(
        audio_queue=audio_queue,
        max_queue=cfg.get("streaming", {}).get("max_listener_queue", 3),
    )

    # Pipeline controller
    from src.pipeline import PipelineController

    controller = PipelineController(cfg=cfg, state=pipeline_state, broadcaster=broadcaster)

    # MIDI
    from src.midi.controller import setup_midi

    setup_midi(cfg, controller)

    # FastAPI app with lifespan
    play_file = args.play_file

    @contextlib.asynccontextmanager
    async def lifespan(app):
        # startup
        task = asyncio.create_task(broadcaster.run(), name="broadcaster")
        log.info("Broadcaster task started.")
        if play_file:
            log.info("Auto-starting pipeline with test file: %s", play_file)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, controller.start, play_file)
        yield
        # shutdown
        task.cancel()
        controller.stop()
        log.info("Server shutting down.")

    from src.web.server import create_app

    app = create_app(
        state=pipeline_state,
        broadcaster=broadcaster,
        pipeline_control=controller,
        qr_png_bytes=qr_png,
        lan_url=lan_url,
        lifespan=lifespan,
    )

    import uvicorn

    log.info("Operator page: %s/operator", lan_url)
    log.info("Listener page: %s/listen", lan_url)

    uvicorn.run(
        app,
        host=server_cfg.get("host", "0.0.0.0"),
        port=port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
