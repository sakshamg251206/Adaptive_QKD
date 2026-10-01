from __future__ import annotations

import json

import numpy as np
import pytest

from adaptive_qkd.benchmark import POLICIES, Scenario, render_benchmark_markdown, run_benchmark, summarize
from adaptive_qkd.config import CLASS_NAMES, Paths
from adaptive_qkd.dataset import CHECKPOINT_FRACTIONS, generate_dataset, load_dataset, sample_channel_params
from adaptive_qkd.models import ThreatClassifier
from adaptive_qkd.protocol import AdaptiveQKDProtocol
from adaptive_qkd.simulator import ChannelParams


def test_generate_dataset_shape() -> None:
    dataset = generate_dataset(sessions_per_class=3, seed=1)
    assert len(dataset.features) == 3 * len(CLASS_NAMES) * len(CHECKPOINT_FRACTIONS)
    assert dataset.sequences.shape == (3 * len(CLASS_NAMES), 20)
    assert dataset.full_sessions["label"].value_counts().to_dict() == dict.fromkeys(CLASS_NAMES, 3)


def test_sampled_parameters_match_their_label() -> None:
    rng = np.random.default_rng(0)
    for label in CLASS_NAMES:
        for _ in range(20):
            assert sample_channel_params(label, rng).label == label
    with pytest.raises(ValueError):
        sample_channel_params("unknown", rng)


def test_dataset_roundtrip(trained_paths: Paths) -> None:
    dataset = load_dataset(trained_paths)
    assert len(dataset.sequences) == dataset.features["session_id"].nunique()


def test_missing_dataset_has_helpful_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = Paths(tmp_path, tmp_path, tmp_path)
    with pytest.raises(FileNotFoundError, match="adaptive-qkd generate"):
        load_dataset(paths)


def test_training_writes_reports(trained_paths: Paths) -> None:
    metrics = json.loads((trained_paths.results_dir / "metrics.json").read_text())
    rf = metrics["models"][0]
    assert rf["binary_roc_auc"] > 0.9
    assert set(metrics["by_progress"]) == {f"{f:.2f}" for f in CHECKPOINT_FRACTIONS}
    for plot in ("confusion_matrix_rf.png", "roc_curve.png", "feature_importance.png"):
        assert (trained_paths.plots_dir / plot).exists()


def test_classifier_roundtrip(trained_paths: Paths, classifier: ThreatClassifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "copy.joblib"
    classifier.save(path)
    assert ThreatClassifier.load(path).class_names == CLASS_NAMES
    with pytest.raises(KeyError):
        classifier.predict_proba({"qber": 0.1})


def test_benchmark_summary(classifier: ThreatClassifier) -> None:
    scenarios = (Scenario("Clean", ChannelParams()), Scenario("PNS", ChannelParams(pns_prob=0.6)))
    results = run_benchmark(AdaptiveQKDProtocol(classifier), sessions_per_scenario=5, scenarios=scenarios)
    assert len(results) == 2 * 5 * len(POLICIES)
    summary = summarize(results).set_index(["scenario", "policy"])
    assert summary.loc[("PNS", "No defence"), "unsafe_rate"] == 1.0
    assert summary.loc[("PNS", "Adaptive (ML)"), "unsafe_rate"] == 0.0
    assert summary.loc[("Clean", "Adaptive (ML)"), "secure_key_rate"] > 0.3
    assert "| Clean |" in render_benchmark_markdown(summarize(results), 5)
