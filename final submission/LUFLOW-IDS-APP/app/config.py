from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final, Optional


def _resource_base() -> Path:
	if getattr(sys, "frozen", False):
		return Path(getattr(sys, "_MEIPASS", Path.cwd()))
	return Path(__file__).resolve().parent.parent


def _writable_base() -> Path:
	if getattr(sys, "frozen", False):
		root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LUFLOW-IDS"
		root.mkdir(parents=True, exist_ok=True)
		return root
	return _resource_base()


BASE_DIR: Final[Path] = _resource_base()
MODEL_DIR: Final[Path] = BASE_DIR / "xgboost_models"
WRITABLE_DIR: Final[Path] = _writable_base()
LOG_ROOT_DIR: Final[Path] = WRITABLE_DIR / "logs"
EXPORT_DIR: Final[Path] = WRITABLE_DIR / "exports"
DEFAULT_IDLE_TIMEOUT_SECONDS: Final[float] = 2.0
LIVE_CAPTURE_BATCH_SIZE: Final[int] = 1
MAX_TABLE_ROWS: Final[int] = 200
NOTEBOOK_PATH: Final[Path] = BASE_DIR / "network-intrusion-xgboost.ipynb"
ICON_PATH_CANDIDATE = BASE_DIR / "icon.ico"
ICON_PATH: Final[Optional[Path]] = ICON_PATH_CANDIDATE if ICON_PATH_CANDIDATE.exists() else None
APPLICATION_NAME: Final[str] = "LUFLOW Intrusion Detection"
VERSION: Final[str] = "1.0.0"
