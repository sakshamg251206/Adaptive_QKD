from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_qkd.config import Paths
from adaptive_qkd.dataset import generate_dataset, save_dataset
from adaptive_qkd.models import ThreatClassifier
from adaptive_qkd.training import train_models


@pytest.fixture(scope="session")
def trained_paths(tmp_path_factory: pytest.TempPathFactory) -> Paths:
    """A small dataset and Random Forest trained into a temporary directory."""
    root: Path = tmp_path_factory.mktemp("artifacts")
    paths = Paths(data_dir=root / "data", model_dir=root / "models", results_dir=root / "results")
    dataset = generate_dataset(sessions_per_class=80, seed=3)
    save_dataset(dataset, paths)
    train_models(paths, include_lstm=False, write_reports=True, seed=3)
    return paths


@pytest.fixture(scope="session")
def classifier(trained_paths: Paths) -> ThreatClassifier:
    return ThreatClassifier.load(trained_paths.rf_model)
