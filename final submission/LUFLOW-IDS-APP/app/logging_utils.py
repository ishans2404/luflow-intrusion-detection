from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Tuple

DEFAULT_LOG_NAME = "application"
LOG_FILE_MAX_BYTES = 2 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 3


def configure_logging(log_dir: Path) -> Path:
    """Configure root logging to write into a rotating file and return log path."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{DEFAULT_LOG_NAME}.log"

    logger = logging.getLogger()
    # Avoid duplicate handlers when reloading in PyInstaller onefile mode.
    if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return log_path


def create_session_logger(log_dir: Path, session_name: str) -> Tuple[logging.Logger, Path]:
    """Create a dedicated logger for a prediction session."""

    session_dir = log_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "session.log"

    logger = logging.getLogger(f"session.{session_name}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger, log_path
