from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config
from .converters import ip_to_int, protocol_to_numeric

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelAssets:
    model: Any
    encoder: Any
    feature_names: List[str]
    metadata: Dict[str, object]
    class_names: List[str]


class ModelManager:
    """Load model artefacts once and expose utility methods for inference."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or config.MODEL_DIR
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        assets = joblib.load(self.model_dir / "optimized_xgboost_luflow.pkl")
        encoder = joblib.load(self.model_dir / "label_encoder.pkl")
        feature_names = joblib.load(self.model_dir / "feature_names.pkl")
        metadata = joblib.load(self.model_dir / "model_metadata.pkl")

        self.assets = ModelAssets(
            model=assets,
            encoder=encoder,
            feature_names=feature_names,
            metadata=metadata,
            class_names=list(encoder.classes_),
        )

        LOGGER.info("Loaded model assets from %s", self.model_dir)

    @property
    def class_names(self) -> List[str]:
        return self.assets.class_names

    @property
    def metadata(self) -> Dict[str, object]:
        return self.assets.metadata

    @property
    def feature_names(self) -> List[str]:
        return self.assets.feature_names

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataframe columns and dtypes match what the model expects."""

        df = df.copy()
        expected = self.feature_names

        for col in expected:
            if col not in df.columns:
                df[col] = 0

        if "src_ip" in df.columns:
            df["src_ip"] = df["src_ip"].apply(ip_to_int)
        if "dest_ip" in df.columns:
            df["dest_ip"] = df["dest_ip"].apply(ip_to_int)
        if "proto" in df.columns:
            df["proto"] = df["proto"].apply(protocol_to_numeric)

        numeric_cols = [c for c in expected if c not in {"src_ip", "dest_ip", "proto"}]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df[expected]

    def predict(self, df: pd.DataFrame) -> Dict[str, object]:
        """Run inference and return structured results."""

        prepared = self.prepare_dataframe(df)
        model = self.assets.model
        encoder = self.assets.encoder

        predictions = model.predict(prepared)
        probabilities = model.predict_proba(prepared)
        predicted_labels = encoder.inverse_transform(predictions)

        return {
            "prepared_features": prepared,
            "predicted_codes": predictions,
            "predicted_labels": predicted_labels,
            "probabilities": probabilities,
            "class_names": self.class_names,
            "metadata": self.metadata,
        }

    def most_likely_probability(self, probabilities: np.ndarray) -> Tuple[float, str]:
        idx = int(np.argmax(probabilities))
        return float(probabilities[idx]), self.class_names[idx]
