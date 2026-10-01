"""Threat classifiers and their on-disk format.

* :class:`ThreatClassifier` wraps the Random Forest used by the adaptive
  protocol. It is saved with ``joblib`` together with its feature schema.
* :func:`build_lstm` creates the PyTorch sequence model trained on windowed
  QBER sequences. PyTorch is an optional dependency (the ``lstm`` extra).

Model files are serialised Python objects: only load files you produced
yourself, never ones obtained from an untrusted source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from adaptive_qkd.config import ATTACK_CLASSES, CLASS_NAMES, DEFAULT_SEED, FEATURE_COLUMNS

if TYPE_CHECKING:
    import torch

MODEL_FORMAT_VERSION = 2


def build_random_forest(seed: int = DEFAULT_SEED) -> RandomForestClassifier:
    # Trees are scale-invariant, so features are used without standardisation.
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


@dataclass
class ThreatClassifier:
    """A fitted session classifier together with the schema it was trained on."""

    model: RandomForestClassifier
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    class_names: tuple[str, ...] = CLASS_NAMES

    def predict_proba(self, features: dict[str, float]) -> dict[str, float]:
        """Class probabilities for one feature dictionary."""
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        row = np.array([[features[c] for c in self.feature_columns]], dtype=np.float64)
        probs = self.model.predict_proba(row)[0]
        by_class = dict.fromkeys(self.class_names, 0.0)
        for class_index, prob in zip(self.model.classes_, probs, strict=True):
            by_class[self.class_names[int(class_index)]] = float(prob)
        return by_class

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": MODEL_FORMAT_VERSION,
            "model": self.model,
            "feature_columns": list(self.feature_columns),
            "class_names": list(self.class_names),
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> ThreatClassifier:
        payload: dict[str, Any] = joblib.load(path)
        if payload.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError(f"{path} was written by an incompatible version; retrain with: adaptive-qkd train")
        # Single-row predictions are dominated by thread start-up cost when parallel.
        payload["model"].set_params(n_jobs=1)
        return cls(
            model=payload["model"],
            feature_columns=tuple(payload["feature_columns"]),
            class_names=tuple(payload["class_names"]),
        )


def attack_probability(class_probs: dict[str, float]) -> float:
    """Total probability mass on classes in which an eavesdropper is present."""
    return float(sum(p for name, p in class_probs.items() if name in ATTACK_CLASSES))


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def build_lstm(num_classes: int = len(CLASS_NAMES), hidden_dim: int = 64, num_layers: int = 2) -> torch.nn.Module:
    """LSTM classifier over a ``(batch, seq_len, 1)`` QBER sequence."""
    from torch import nn

    class QBERSequenceLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1])

    return QBERSequenceLSTM()
