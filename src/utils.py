"""
Utility helpers: logging setup, directory management, JSON I/O.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any

from config.config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    """Serialize *data* to a JSON file at *path*."""
    ensure_dirs(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load and return JSON from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def human_readable_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
