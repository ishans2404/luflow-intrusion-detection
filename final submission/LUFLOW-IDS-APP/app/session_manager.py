from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from . import config
from .data_models import PredictionRecord, SessionStatistics
from .logging_utils import create_session_logger

LOGGER = logging.getLogger(__name__)


class SessionManager:
    """Track predictions and persist session level artefacts."""

    def __init__(self, log_root: Path | None = None) -> None:
        self.log_root = log_root or config.LOG_ROOT_DIR
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.logger, self.log_path = create_session_logger(self.log_root, self.session_id)
        self.stats = SessionStatistics()
        self.prediction_rows: List[dict] = []

        self.export_dir = config.EXPORT_DIR / self.session_id
        self.export_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Session %s initialised", self.session_id)
        self.logger.info("Session started")

    @property
    def session_folder(self) -> Path:
        return self.log_root / self.session_id

    def record_packet(self) -> None:
        self.stats.packets_captured += 1

    def record_capture_start(self) -> None:
        self.stats.capture_started_at = datetime.utcnow()
        self.logger.info("Capture started")

    def record_capture_stop(self) -> None:
        self.stats.capture_stopped_at = datetime.utcnow()
        self.logger.info("Capture stopped")

    def record_prediction(self, mode: str, source: str, label: str, probability: float, probability_by_class: dict, features: dict) -> None:
        record = PredictionRecord(
            timestamp=datetime.utcnow(),
            mode=mode,
            source=source,
            label=label,
            probability=probability,
            probability_by_class=probability_by_class,
            features=features,
        )
        self.stats.record_prediction(record)
        self.prediction_rows.append(
            {
                "timestamp": record.timestamp.isoformat(),
                "mode": mode,
                "source": source,
                "label": label,
                "probability": probability,
                "probability_by_class": json.dumps(probability_by_class),
                **features,
            }
        )
        self.logger.info(
            "Prediction | mode=%s | source=%s | label=%s | probability=%.4f",
            mode,
            source,
            label,
            probability,
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.prediction_rows:
            return pd.DataFrame()
        return pd.DataFrame(self.prediction_rows)

    def export_predictions_csv(self, destination: Path | None = None) -> Path:
        df = self.to_dataframe()
        if df.empty:
            raise ValueError("No predictions available to export.")

        destination = destination or (self.export_dir / "predictions.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)
        self.logger.info("Predictions exported to %s", destination)
        return destination

    def export_session_summary(self, destination: Path | None = None) -> Path:
        destination = destination or (self.export_dir / "session_summary.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.stats.as_dict(), handle, indent=2)
        self.logger.info("Session summary written to %s", destination)
        return destination

    def close(self) -> None:
        try:
            if self.prediction_rows:
                self.export_predictions_csv()
            self.export_session_summary()
        finally:
            self.logger.info("Session closed")
            for handler in list(self.logger.handlers):
                handler.close()
                self.logger.removeHandler(handler)


def export_multiple_sessions(session_ids: Iterable[str], log_root: Path | None = None) -> pd.DataFrame:
    """Utility to gather results across multiple sessions."""

    root = log_root or config.LOG_ROOT_DIR
    frames = []
    for session_id in session_ids:
        csv_path = root / session_id / "predictions.csv"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()
