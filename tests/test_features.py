from __future__ import annotations

import numpy as np
import pytest

from adaptive_qkd.config import FEATURE_COLUMNS
from adaptive_qkd.features import NO_EVIDENCE_QBER, extract_features, session_record, window_error_rates
from adaptive_qkd.simulator import BB84Simulator, ChannelParams


def test_feature_names_match_schema() -> None:
    trace = BB84Simulator(num_qubits=800, seed=1).simulate()
    assert tuple(extract_features(trace)) == FEATURE_COLUMNS


def test_window_error_rates_fixed_length() -> None:
    errors = np.array([True, False, False, True, False, False], dtype=bool)
    rates = window_error_rates(errors, 3)
    assert rates.tolist() == pytest.approx([0.5, 0.5, 0.0])
    # Fewer bits than windows: empty windows take the overall rate.
    assert window_error_rates(errors[:2], 4).shape == (4,)
    assert window_error_rates(np.array([], dtype=bool), 5).tolist() == [NO_EVIDENCE_QBER] * 5
    with pytest.raises(ValueError):
        window_error_rates(errors, 0)


def test_prefix_features_only_use_the_prefix() -> None:
    trace = BB84Simulator(num_qubits=2000, seed=3).simulate(ChannelParams(intercept_prob=1.0))
    early = extract_features(trace, 0.25)
    m = 500
    errors = trace.error_mask[:m][trace.test_mask[:m]]
    assert early["qber"] == pytest.approx(errors.mean())
    assert early["sifting_ratio"] == pytest.approx(trace.sifted_mask[:m].mean())


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_invalid_fraction(fraction: float) -> None:
    trace = BB84Simulator(num_qubits=100, seed=0).simulate()
    with pytest.raises(ValueError):
        extract_features(trace, fraction)


def test_session_record_includes_metadata() -> None:
    record = session_record(BB84Simulator(num_qubits=900, seed=2).simulate(ChannelParams(pns_prob=0.3)))
    assert record["label"] == "pns"
    assert record["num_pulses"] == 900
    assert record["pns_prob"] == 0.3
