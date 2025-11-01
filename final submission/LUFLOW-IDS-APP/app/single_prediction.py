from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import pandas as pd

from .model_manager import ModelManager


@dataclass
class ManualInput:
    values: Dict[str, object]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.values])


def predict_from_manual_input(pairs: Mapping[str, object], model_manager: ModelManager) -> dict:
    manual = ManualInput(values=dict(pairs))
    df = manual.to_dataframe()
    return model_manager.predict(df)
