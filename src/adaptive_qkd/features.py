"""Feature extraction from simulated BB84 sessions.

Features are computed on a *prefix* of the session so the same code serves both
offline training (the full session) and the adaptive protocol, which inspects
the session at intermediate checkpoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from adaptive_qkd.config import NUM_WINDOWS
from adaptive_qkd.simulator import SessionTrace

# QBER reported when no test bits are available yet: a maximally pessimistic
# estimate, since nothing rules out a fully compromised channel.
NO_EVIDENCE_QBER = 0.5


def window_error_rates(errors: NDArray[np.bool_], num_windows: int) -> NDArray[np.float32]:
    """Split an ordered error sequence into ``num_windows`` chunks and return each chunk's error rate.

    Windows that receive no bits (sequences shorter than ``num_windows``) take
    the overall error rate so the output always has a fixed length.
    """
    if num_windows < 1:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if errors.size == 0:
        return np.full(num_windows, NO_EVIDENCE_QBER, dtype=np.float32)
    overall = float(errors.mean())
    rates = [float(chunk.mean()) if chunk.size else overall for chunk in np.array_split(errors, num_windows)]
    return np.asarray(rates, dtype=np.float32)


def _prefix_length(trace: SessionTrace, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return max(1, round(trace.num_pulses * fraction))


def ordered_test_errors(trace: SessionTrace, fraction: float = 1.0) -> NDArray[np.bool_]:
    """Error flags of the disclosed test bits within the prefix, in transmission order."""
    m = _prefix_length(trace, fraction)
    return trace.error_mask[:m][trace.test_mask[:m]]


def extract_features(
    trace: SessionTrace,
    fraction: float = 1.0,
    num_windows: int = NUM_WINDOWS,
) -> dict[str, float]:
    """Compute the classifier feature vector for the first ``fraction`` of a session.

    For partial sessions the number of QBER windows shrinks proportionally, so
    each window covers about as many bits as in a full session and window-level
    statistics stay comparable with the training distribution.
    """
    m = _prefix_length(trace, fraction)
    detected = trace.detected[:m]
    sifted = trace.sifted_mask[:m]
    basis_mismatch = detected & (trace.alice_bases[:m] != trace.bob_bases[:m])

    errors = ordered_test_errors(trace, fraction)
    windows_used = num_windows if fraction >= 1.0 else max(2, round(num_windows * fraction))
    windows = window_error_rates(errors, windows_used)
    num_detected = int(detected.sum())

    return {
        "qber": float(errors.mean()) if errors.size else NO_EVIDENCE_QBER,
        "qber_std": float(windows.std()),
        "qber_min": float(windows.min()),
        "qber_max": float(windows.max()),
        "sifting_ratio": float(sifted.sum() / m),
        "basis_mismatch_rate": float(basis_mismatch.sum() / num_detected) if num_detected else 1.0,
        "timing_jitter_ns": trace.timing_jitter_ns,
    }


def session_record(trace: SessionTrace) -> dict[str, Any]:
    """Full-session feature row plus metadata, as stored in the dataset CSV."""
    return {
        **extract_features(trace),
        "num_pulses": trace.num_pulses,
        "sifted_key_length": trace.sifted_length,
        "test_bits": trace.test_length,
        "noise_prob": trace.params.noise_prob,
        "intercept_prob": trace.params.intercept_prob,
        "pns_prob": trace.params.pns_prob,
        "label": trace.label,
    }
