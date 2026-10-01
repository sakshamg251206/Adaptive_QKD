"""Project-wide constants and filesystem locations.

Directory locations can be overridden with environment variables so the same
code works from a source checkout, a container, or a hosted dashboard where only
some paths are writable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Ground-truth session classes, in the fixed order used for model outputs.
CLASS_NAMES: tuple[str, ...] = (
    "clean",
    "noise_only",
    "intercept_resend",
    "pns",
    "noise+attack",
)

# Classes in which an eavesdropper (Eve) is present.
ATTACK_CLASSES: frozenset[str] = frozenset({"intercept_resend", "pns", "noise+attack"})

# Human-readable class names for reports and the dashboard.
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "clean": "Clean channel",
    "noise_only": "Channel noise only",
    "intercept_resend": "Intercept-resend attack",
    "pns": "Photon-number-splitting attack",
    "noise+attack": "Noise + attack",
}

# Session-level features consumed by the tabular classifier. Every feature is a
# ratio or rate, so it is independent of how many pulses a session contains and
# can be computed on a partial session at a protocol checkpoint.
FEATURE_COLUMNS: tuple[str, ...] = (
    "qber",
    "qber_std",
    "qber_min",
    "qber_max",
    "sifting_ratio",
    "basis_mismatch_rate",
    "timing_jitter_ns",
)

# Number of windows the sifted test bits are split into for the QBER sequence.
NUM_WINDOWS = 20

# Fraction of the sifted key disclosed publicly to estimate the QBER.
TEST_FRACTION = 0.25

DEFAULT_SEED = 42

_PROJECT_ROOT_MARKER = "pyproject.toml"


def _default_root() -> Path:
    """Return the repository root for a source checkout, else the working directory."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / _PROJECT_ROOT_MARKER).exists():
        return candidate
    return Path.cwd()


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Paths:
    """Locations of generated data, trained models and evaluation reports."""

    data_dir: Path
    model_dir: Path
    results_dir: Path

    @classmethod
    def from_env(cls) -> Paths:
        root = _default_root()
        return cls(
            data_dir=_env_path("ADAPTIVE_QKD_DATA_DIR", root / "data"),
            model_dir=_env_path("ADAPTIVE_QKD_MODEL_DIR", root / "models"),
            results_dir=_env_path("ADAPTIVE_QKD_RESULTS_DIR", root / "results"),
        )

    @property
    def dataset_csv(self) -> Path:
        return self.data_dir / "qkd_sessions.csv"

    @property
    def sequences_npz(self) -> Path:
        return self.data_dir / "qkd_sessions_sequential.npz"

    @property
    def rf_model(self) -> Path:
        return self.model_dir / "rf_classifier.joblib"

    @property
    def lstm_model(self) -> Path:
        return self.model_dir / "lstm_classifier.pt"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"
