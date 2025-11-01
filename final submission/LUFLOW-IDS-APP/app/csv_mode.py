from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .model_manager import ModelManager

LOGGER = logging.getLogger(__name__)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    LOGGER.info("Loading CSV from %s", path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV file is empty")
    return df


def predict_from_csv(path: Path, model_manager: ModelManager) -> Tuple[pd.DataFrame, dict]:
    df = load_csv(path)
    result = model_manager.predict(df)
    return df, result
