"""Optional MIDI input listener for pause/resume/stop control.

Uses mido + python-rtmidi. Runs in a daemon thread so it never blocks startup.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)


def list_midi_inputs() -> list[str]:
    try:
        import mido
        return mido.get_input_names()
    except Exception as exc:
        log.warning("Could not list MIDI devices: %s", exc)
        return []


def find_midi_port(name_or_none: Optional[str]) -> Optional[str]:
    ports = list_midi_inputs()
    if not ports:
        return None
    if name_or_none is None:
        return None
    needle = name_or_none.lower()
    for p in ports:
        if needle in p.lower():
            return p
    return None


class MidiController:
    """
    Listens for MIDI control messages and calls pipeline_control methods.

    Mapping config structure (from config.yaml):
      midi.mapping.pause:  {type, channel, cc, value}
      midi.mapping.resume: {type, channel, cc, value}
      midi.mapping.toggle: {type, channel, cc, value}
      midi.mapping.stop:   {type, channel, cc, value}
    """

    def __init__(self, port_name: str, mapping: dict, pipeline_control, log_events: bool = True):
        self.port_name = port_name
        self.mapping = mapping
        self.pipeline_control = pipeline_control
        self.log_events = log_events
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

    def _matches(self, msg, rule: dict) -> bool:
        if rule is None:
            return False
        msg_type = rule.get("type", "control_change")
        if msg.type != msg_type:
            return False
        channel = rule.get("channel")
        if channel is not None and msg.channel != channel:
            return False
        cc = rule.get("cc")
        if cc is not None and getattr(msg, "control", None) != cc:
            return False
        note = rule.get("note")
        if note is not None and getattr(msg, "note", None) != note:
            return False
        value = rule.get("value")
        if value is not None:
            actual = getattr(msg, "value", None) or getattr(msg, "velocity", None)
            if actual != value:
                return False
        return True

    def _handle(self, msg) -> None:
        if self.log_events:
            log.debug("[MIDI] %s", msg)

        for action, rule in self.mapping.items():
            if rule and self._matches(msg, rule):
                log.info("[MIDI] action=%s triggered by %s", action, msg)
                if action == "pause":
                    self.pipeline_control.pause()
                elif action == "resume":
                    self.pipeline_control.resume()
                elif action == "toggle":
                    self.pipeline_control.toggle_pause()
                elif action == "stop":
                    self.pipeline_control.stop()

    def _run(self) -> None:
        try:
            import mido
        except ImportError:
            log.warning("mido not installed — MIDI control disabled.")
            return

        log.info("MIDI listener starting on port: %s", self.port_name)
        try:
            with mido.open_input(self.port_name) as port:
                for msg in port:
                    if self._stopped:
                        break
                    if not msg.is_realtime:
                        self._handle(msg)
        except Exception as exc:
            log.error("MIDI error: %s", exc)
        log.info("MIDI listener stopped.")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="midi-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True


def setup_midi(cfg: dict, pipeline_control) -> Optional[MidiController]:
    if not cfg.get("midi", {}).get("enabled", False):
        return None

    ports = list_midi_inputs()
    if not ports:
        log.warning("MIDI enabled in config but no MIDI input ports found.")
        return None

    log.info("Available MIDI inputs: %s", ports)
    device_name = cfg.get("midi", {}).get("device_name")
    port = find_midi_port(device_name)
    if port is None:
        log.info("MIDI device %r not found. Available: %s", device_name, ports)
        return None

    mapping = cfg.get("midi", {}).get("mapping", {})
    log_events = cfg.get("debug", {}).get("log_midi_events", True)
    ctrl = MidiController(port, mapping, pipeline_control, log_events=log_events)
    ctrl.start()
    return ctrl
