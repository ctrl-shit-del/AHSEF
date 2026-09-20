"""Per-iteration file logging for training runs.

Each iteration owns a private logger and a private log file.  Iteration 1's
log is never opened by iteration 2, and repeat runs of the same iteration
append behind a run banner rather than truncating history.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import ensure_dir


LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def create_iteration_logger(
    name: str,
    log_path: Path | str,
    console: bool = True,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger writing to ``log_path`` (append) and optionally stdout."""
    log_path = Path(log_path)
    ensure_dir(log_path.parent)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    close_logger(logger)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Detach and close every handler so the log file is released."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - handler already closed
            pass


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def log_mapping(logger: logging.Logger, title: str, mapping: dict) -> None:
    logger.info("%s:", title)
    for key in sorted(mapping):
        logger.info("  %-28s %s", key, mapping[key])
