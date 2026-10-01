"""Labelled dataset generation.

Simulates an equal number of BB84 sessions for every class with randomised
channel parameters and session lengths, and writes:

* ``qkd_sessions.csv`` - session-level features and metadata. Each session
  contributes one row per checkpoint in ``CHECKPOINT_FRACTIONS`` (column
  ``progress``), so the classifier also learns what partial sessions look like.
* ``qkd_sessions_sequential.npz`` - the windowed QBER sequence of each full session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from adaptive_qkd.config import CLASS_NAMES, DEFAULT_SEED, NUM_WINDOWS, Paths
from adaptive_qkd.features import extract_features, ordered_test_errors, session_record, window_error_rates
from adaptive_qkd.simulator import BB84Simulator, ChannelParams

logger = logging.getLogger(__name__)

MIN_PULSES = 800
MAX_PULSES = 1200

# Session progress at which features are recorded; matches the protocol's
# default of four evenly spaced checkpoints.
CHECKPOINT_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def sample_channel_params(label: str, rng: np.random.Generator) -> ChannelParams:
    """Draw random channel parameters representative of ``label``.

    The ranges overlap on purpose (e.g. strong noise vs. weak interception) so
    that the classification task is not trivially separable by QBER alone.
    """
    if label == "clean":
        return ChannelParams()
    if label == "noise_only":
        return ChannelParams(noise_prob=rng.uniform(0.01, 0.15))
    if label == "intercept_resend":
        return ChannelParams(intercept_prob=rng.uniform(0.10, 1.00))
    if label == "pns":
        return ChannelParams(pns_prob=rng.uniform(0.15, 0.90))
    if label == "noise+attack":
        noise = rng.uniform(0.01, 0.12)
        if rng.random() < 0.6:
            return ChannelParams(noise_prob=noise, intercept_prob=rng.uniform(0.10, 0.80))
        return ChannelParams(noise_prob=noise, pns_prob=rng.uniform(0.15, 0.80))
    raise ValueError(f"Unknown class label {label!r}; expected one of {CLASS_NAMES}")


@dataclass(frozen=True)
class Dataset:
    """Feature rows (one per session and checkpoint) and per-session QBER sequences.

    ``sequences[i]`` belongs to the ``i``-th session id in ascending order.
    """

    features: pd.DataFrame
    sequences: NDArray[np.float32]

    @property
    def full_sessions(self) -> pd.DataFrame:
        """One row per session: the features of the complete session."""
        return self.features[self.features["progress"] == 1.0].sort_values("session_id").reset_index(drop=True)


def generate_dataset(sessions_per_class: int = 2000, seed: int = DEFAULT_SEED) -> Dataset:
    """Simulate ``sessions_per_class`` sessions for each class."""
    if sessions_per_class < 1:
        raise ValueError(f"sessions_per_class must be positive, got {sessions_per_class}")
    rng = np.random.default_rng(seed)
    records = []
    sequences = []
    session_id = 0
    for label in CLASS_NAMES:
        logger.info("Simulating %d '%s' sessions", sessions_per_class, label)
        for _ in range(sessions_per_class):
            params = sample_channel_params(label, rng)
            num_pulses = int(rng.integers(MIN_PULSES, MAX_PULSES + 1))
            trace = BB84Simulator(num_qubits=num_pulses, seed=seed + session_id).simulate(params)
            record = session_record(trace)
            for fraction in CHECKPOINT_FRACTIONS:
                row = record if fraction == 1.0 else {**record, **extract_features(trace, fraction)}
                records.append({"session_id": session_id, "progress": fraction, **row})
            sequences.append(window_error_rates(ordered_test_errors(trace), NUM_WINDOWS))
            session_id += 1
    return Dataset(features=pd.DataFrame.from_records(records), sequences=np.stack(sequences))


def save_dataset(dataset: Dataset, paths: Paths) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    dataset.features.to_csv(paths.dataset_csv, index=False)
    np.savez_compressed(
        paths.sequences_npz,
        qber_sequences=dataset.sequences,
        session_ids=np.unique(dataset.features["session_id"].to_numpy()),
    )


def load_dataset(paths: Paths) -> Dataset:
    if not paths.dataset_csv.exists() or not paths.sequences_npz.exists():
        raise FileNotFoundError(f"Dataset not found in {paths.data_dir}. Generate it first with: adaptive-qkd generate")
    features = pd.read_csv(paths.dataset_csv)
    with np.load(paths.sequences_npz) as npz:
        sequences = npz["qber_sequences"].astype(np.float32)
        session_ids = npz["session_ids"]
    if "progress" not in features.columns:
        raise ValueError("Dataset was written by an older version; regenerate it with: adaptive-qkd generate")
    if not np.array_equal(session_ids, np.unique(features["session_id"].to_numpy())):
        raise ValueError("Tabular and sequential dataset files are out of sync; regenerate the dataset")
    return Dataset(features=features, sequences=sequences)
