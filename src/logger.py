"""Centralised logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE = "%H:%M:%S"


def setup(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(numeric)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(numeric)
    sh.setFormatter(logging.Formatter(_FMT, _DATE))
    root.addHandler(sh)

    fh = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(numeric)
    fh.setFormatter(logging.Formatter(_FMT, _DATE))
    root.addHandler(fh)

    # Quiet down noisy third-party loggers
    for noisy in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# Import handlers here so the module is self-contained
import logging.handlers  # noqa: E402
