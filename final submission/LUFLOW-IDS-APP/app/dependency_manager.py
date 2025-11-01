from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

DEFAULT_TSHARK_LOCATIONS = [
    Path("C:/Program Files/Wireshark/tshark.exe"),
    Path("C:/Program Files (x86)/Wireshark/tshark.exe"),
]

WINGET_ID = "WiresharkFoundation.Wireshark"


def find_tshark() -> Optional[Path]:
    candidate = shutil.which("tshark")
    if candidate:
        return Path(candidate).resolve()
    for path in DEFAULT_TSHARK_LOCATIONS:
        if path.exists():
            return path
    return None


def ensure_tshark(auto_install: bool = True) -> Optional[Path]:
    existing = find_tshark()
    if existing:
        LOGGER.info("Found tshark at %s", existing)
        _ensure_on_path(existing)
        return existing

    if not auto_install:
        return None

    if platform.system().lower() != "windows":
        LOGGER.info("Auto-install skipped: non-Windows platform detected")
        return None

    winget = shutil.which("winget")
    if not winget:
        LOGGER.warning("winget not found; cannot perform silent TShark installation")
        return None

    LOGGER.info("Attempting silent TShark installation via winget")
    command = [
        winget,
        "install",
        "--id",
        WINGET_ID,
        "--exact",
        "--silent",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        LOGGER.info("winget stdout: %s", completed.stdout.strip())
        if completed.stderr:
            LOGGER.debug("winget stderr: %s", completed.stderr.strip())
    except subprocess.CalledProcessError as exc:
        LOGGER.error("winget install failed (exit %s): %s", exc.returncode, exc.stderr)
        return None
    except Exception:
        LOGGER.exception("Unexpected error while running winget for TShark installation")
        return None

    installed = find_tshark()
    if installed:
        _ensure_on_path(installed)
    return installed


def _ensure_on_path(tshark_path: Path) -> None:
    directory = tshark_path.parent
    current_path = os.environ.get("PATH", "")
    if str(directory) not in current_path:
        os.environ["PATH"] = f"{directory}{os.pathsep}{current_path}" if current_path else str(directory)
