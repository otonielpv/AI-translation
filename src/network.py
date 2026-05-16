"""Detect the LAN IP address and generate a QR code."""

from __future__ import annotations

import io
import logging
import socket

log = logging.getLogger(__name__)


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        log.warning("Could not determine LAN IP, falling back to 127.0.0.1")
        return "127.0.0.1"


def make_qr_png(url: str) -> bytes:
    """Return PNG bytes for a QR code pointing to `url`."""
    import qrcode

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=6, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def print_qr_terminal(url: str) -> None:
    """Print a QR code to the terminal using qrcode's ASCII output."""
    try:
        import qrcode

        qr = qrcode.QRCode()
        qr.add_data(url)
        qr.make(fit=True)
        print()
        qr.print_ascii(invert=True)
        print(f"\n  Listener URL: {url}\n")
    except Exception as exc:
        log.warning("Could not print QR to terminal: %s", exc)
        print(f"\n  Listener URL: {url}\n")
