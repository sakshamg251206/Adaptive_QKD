"""Training and evaluation of the eavesdropping classifiers.

Two models are trained on a stratified 80/20 split of the generated dataset:

1. a Random Forest on the session-level features (used by the protocol), and
2. an optional LSTM on the windowed QBER sequence alone. The LSTM is an
   ablation that measures how much the *shape* of the QBER over time reveals
   without any of the other signals.

Evaluation writes ``metrics.md``/``metrics.json`` and plots to the results directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from adaptive_qkd.config import ATTACK_CLASSES, CLASS_NAMES, DEFAULT_SEED, FEATURE_COLUMNS, Paths
from adaptive_qkd.dataset import Dataset, load_dataset
from adaptive_qkd.models import ThreatClassifier, build_lstm, build_random_forest, torch_available

logger = logging.getLogger(__name__)

TEST_SIZE = 0.20


@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    macro_f1: float
    binary_roc_auc: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]


@dataclass
class TrainingReport:
    train_size: int
    test_size: int
    models: list[ModelMetrics] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    # Random Forest metrics on test sessions observed only up to a given progress.
    by_progress: dict[str, dict[str, float]] = field(default_factory=dict)


_ATTACK_INDICES = [i for i, c in enumerate(CLASS_NAMES) if c in ATTACK_CLASSES]


def _binary_attack_scores(
    y_true: NDArray[np.int64], probs: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Binary attacked/not-attacked targets and the summed attack-class probability."""
    return np.isin(y_true, _ATTACK_INDICES).astype(np.int64), probs[:, _ATTACK_INDICES].sum(axis=1)


def _evaluate(name: str, y_true: NDArray[np.int64], probs: NDArray[np.float64]) -> ModelMetrics:
    y_pred = probs.argmax(axis=1)
    is_attack, attack_score = _binary_attack_scores(y_true, probs)
    report: dict[str, Any] = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return ModelMetrics(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        binary_roc_auc=float(roc_auc_score(is_attack, attack_score)),
        per_class={
            c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")} for c in CLASS_NAMES
        },
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
    )


def _train_lstm(
    seq_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    seq_test: NDArray[np.float32],
    seed: int,
    epochs: int,
    model_path: Path,
) -> NDArray[np.float64]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    # Window QBERs are small numbers (mostly < 0.3); standardise for stable training.
    mean, std = float(seq_train.mean()), float(seq_train.std()) or 1.0

    def to_tensor(seq: NDArray[np.float32]) -> torch.Tensor:
        return torch.tensor((seq - mean) / std, dtype=torch.float32).unsqueeze(-1)

    loader = DataLoader(
        TensorDataset(to_tensor(seq_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = build_lstm()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch_x.size(0)
        if epoch == 1 or epoch % 5 == 0:
            logger.info("LSTM epoch %2d/%d  loss %.4f", epoch, epochs, total / len(seq_train))

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(to_tensor(seq_test)), dim=1).numpy().astype(np.float64)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "seq_mean": mean, "seq_std": std, "class_names": list(CLASS_NAMES)},
        model_path,
    )
    return probs


def train_models(
    paths: Paths | None = None,
    dataset: Dataset | None = None,
    include_lstm: bool = True,
    lstm_epochs: int = 25,
    write_reports: bool = True,
    seed: int = DEFAULT_SEED,
) -> TrainingReport:
    """Train the classifiers, save them under ``paths.model_dir`` and optionally write reports."""
    paths = paths or Paths.from_env()
    dataset = dataset if dataset is not None else load_dataset(paths)
    df = dataset.features

    unknown = set(df["label"]) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Dataset contains unknown labels: {sorted(unknown)}")

    # Split by session (not by row) so no checkpoint of a test session is seen in training.
    sessions = dataset.full_sessions
    session_labels = sessions["label"].map(CLASS_NAMES.index).to_numpy(dtype=np.int64)
    train_pos, test_pos = train_test_split(
        np.arange(len(sessions)), test_size=TEST_SIZE, random_state=seed, stratify=session_labels
    )
    train_ids = sessions["session_id"].to_numpy()[train_pos]
    test_ids = sessions["session_id"].to_numpy()[test_pos]
    y_train, y_test = session_labels[train_pos], session_labels[test_pos]
    seq_train, seq_test = dataset.sequences[train_pos], dataset.sequences[test_pos]

    def xy(frame: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        return (
            frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64),
            frame["label"].map(CLASS_NAMES.index).to_numpy(dtype=np.int64),
        )

    X_rows, y_rows = xy(df[df["session_id"].isin(train_ids)])
    # Full-session test rows in the same order as ``test_ids`` / ``y_test``.
    X_test, _ = xy(sessions.iloc[test_pos])
    report = TrainingReport(train_size=len(train_ids), test_size=len(test_ids))
    test_probs: dict[str, NDArray[np.float64]] = {}

    logger.info("Training Random Forest on %d sessions (%d checkpoint rows)", len(train_ids), len(y_rows))
    rf = build_random_forest(seed).fit(X_rows, y_rows)
    ThreatClassifier(model=rf).save(paths.rf_model)
    test_probs["Random Forest (session features)"] = rf.predict_proba(X_test)
    report.feature_importance = {
        c: float(v)
        for c, v in sorted(zip(FEATURE_COLUMNS, rf.feature_importances_, strict=True), key=lambda kv: -kv[1])
    }
    test_rows = df[df["session_id"].isin(test_ids)]
    for progress, frame in test_rows.groupby("progress"):
        X_p, y_p = xy(frame)
        m = _evaluate("rf", y_p, rf.predict_proba(X_p))
        report.by_progress[f"{float(progress):.2f}"] = {"accuracy": m.accuracy, "binary_roc_auc": m.binary_roc_auc}

    if include_lstm:
        if torch_available():
            logger.info("Training LSTM on QBER sequences for %d epochs", lstm_epochs)
            lstm_probs = _train_lstm(seq_train, y_train, seq_test, seed, lstm_epochs, paths.lstm_model)
            test_probs["LSTM (QBER sequence only)"] = lstm_probs
        else:
            logger.warning("PyTorch is not installed; skipping the LSTM. Install the 'lstm' extra to train it.")

    report.models = [_evaluate(name, y_test, probs) for name, probs in test_probs.items()]
    for m in report.models:
        logger.info("%-34s accuracy %.2f%%  binary ROC-AUC %.4f", m.name, 100 * m.accuracy, m.binary_roc_auc)

    if write_reports:
        write_training_reports(report, paths)
        _plot_roc_curves(y_test, test_probs, report, paths.plots_dir / "roc_curve.png")
    return report


def write_training_reports(report: TrainingReport, paths: Paths) -> None:
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "metrics.json").write_text(json.dumps(asdict(report), indent=2) + "\n")
    (paths.results_dir / "metrics.md").write_text(render_metrics_markdown(report))
    _plot_confusion_matrix(report.models[0], paths.plots_dir / "confusion_matrix_rf.png")
    _plot_feature_importance(report.feature_importance, paths.plots_dir / "feature_importance.png")


def render_metrics_markdown(report: TrainingReport) -> str:
    lines = [
        "# Classifier evaluation",
        "",
        f"Stratified hold-out evaluation on complete sessions: {report.train_size} training sessions,",
        f"{report.test_size} test sessions. Binary ROC-AUC scores *attacked* (intercept-resend, PNS,",
        "noise+attack) against *not attacked* (clean, noise only) using the summed probability of the",
        "attack classes.",
        "",
        "| Model | Accuracy (5 classes) | Macro F1 | Binary ROC-AUC |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {m.name} | {m.accuracy:.1%} | {m.macro_f1:.3f} | {m.binary_roc_auc:.4f} |" for m in report.models]
    for m in report.models:
        lines += [
            "",
            f"## {m.name}: per-class metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{c}` | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {int(v['support'])} |"
            for c, v in m.per_class.items()
        ]
    if report.by_progress:
        lines += [
            "",
            "## Random Forest on partial sessions (protocol checkpoints)",
            "",
            "| Session observed | Accuracy (5 classes) | Binary ROC-AUC |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| {float(p):.0%} | {v['accuracy']:.1%} | {v['binary_roc_auc']:.4f} |"
            for p, v in report.by_progress.items()
        ]
    lines += ["", "## Random Forest feature importance (mean decrease in impurity)", ""]
    lines += [f"{i}. `{c}`: {v:.3f}" for i, (c, v) in enumerate(report.feature_importance.items(), 1)]
    lines += ["", "_Generated by `adaptive-qkd train`._", ""]
    return "\n".join(lines)


def _plot_confusion_matrix(metrics: ModelMetrics, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(metrics.confusion)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{metrics.name}: confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(importance: dict[str, float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(importance)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, [importance[n] for n in names], color="#0f766e")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest feature importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc_curves(
    y_true: NDArray[np.int64], test_probs: dict[str, NDArray[np.float64]], report: TrainingReport, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for metrics, probs in zip(report.models, test_probs.values(), strict=True):
        fpr, tpr, _ = roc_curve(*_binary_attack_scores(y_true, probs))
        ax.plot(fpr, tpr, linewidth=2, label=f"{metrics.name} (AUC {metrics.binary_roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Attack detection ROC (attacked vs. not attacked)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
