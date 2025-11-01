from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class PredictionRecord:
    timestamp: datetime
    mode: str
    source: str
    label: str
    probability: float
    probability_by_class: Dict[str, float]
    features: Dict[str, float]


@dataclass
class SessionStatistics:
    packets_captured: int = 0
    flows_analyzed: int = 0
    alerts_triggered: int = 0
    capture_started_at: datetime | None = None
    capture_stopped_at: datetime | None = None
    predictions: List[PredictionRecord] = field(default_factory=list)

    def record_prediction(self, record: PredictionRecord) -> None:
        self.predictions.append(record)
        self.flows_analyzed += 1
        if record.label.lower() != "benign":
            self.alerts_triggered += 1

    def as_dict(self) -> Dict[str, object]:
        return {
            "packets_captured": self.packets_captured,
            "flows_analyzed": self.flows_analyzed,
            "alerts_triggered": self.alerts_triggered,
            "capture_started_at": self.capture_started_at.isoformat() if self.capture_started_at else None,
            "capture_stopped_at": self.capture_stopped_at.isoformat() if self.capture_stopped_at else None,
            "predictions": [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "mode": record.mode,
                    "source": record.source,
                    "label": record.label,
                    "probability": record.probability,
                    "probability_by_class": record.probability_by_class,
                    "features": record.features,
                }
                for record in self.predictions
            ],
        }
