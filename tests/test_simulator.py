from __future__ import annotations

import numpy as np
import pytest

from adaptive_qkd.simulator import BB84Simulator, ChannelParams


def mean_qber(params: ChannelParams, sessions: int = 20, n: int = 2000) -> float:
    errors = tests = 0
    for seed in range(sessions):
        trace = BB84Simulator(num_qubits=n, seed=seed).simulate(params)
        sifted = trace.sifted_mask
        errors += int((trace.error_mask & sifted).sum())
        tests += int(sifted.sum())
    return errors / tests


def test_clean_channel_has_no_errors() -> None:
    trace = BB84Simulator(num_qubits=1000, seed=1).simulate(ChannelParams())
    assert not (trace.error_mask & trace.sifted_mask).any()


def test_sifting_keeps_about_half_the_pulses() -> None:
    trace = BB84Simulator(num_qubits=20_000, seed=2).simulate()
    assert trace.sifted_length / trace.num_pulses == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (ChannelParams(intercept_prob=1.0), 0.25),
        (ChannelParams(intercept_prob=0.4), 0.10),
        (ChannelParams(noise_prob=0.10), 0.05),
        (ChannelParams(noise_prob=0.10, intercept_prob=0.5), 0.05 + 0.125 - 2 * 0.05 * 0.125),
    ],
)
def test_qber_matches_theory(params: ChannelParams, expected: float) -> None:
    assert mean_qber(params) == pytest.approx(expected, abs=0.01)


def test_pns_leaves_qber_unchanged_but_lowers_detection() -> None:
    sim = BB84Simulator(num_qubits=20_000, seed=5)
    trace = sim.simulate(ChannelParams(pns_prob=0.8))
    assert not (trace.error_mask & trace.sifted_mask).any()
    assert trace.detected.mean() == pytest.approx(1 - 0.8 * 0.25, abs=0.01)


def test_same_seed_is_reproducible() -> None:
    params = ChannelParams(noise_prob=0.05, intercept_prob=0.3)
    a = BB84Simulator(num_qubits=500, seed=9).simulate(params)
    b = BB84Simulator(num_qubits=500, seed=9).simulate(params)
    assert np.array_equal(a.bob_bits, b.bob_bits)
    assert a.timing_jitter_ns == b.timing_jitter_ns


def test_test_bits_are_a_subset_of_sifted_bits() -> None:
    trace = BB84Simulator(num_qubits=1000, seed=4).simulate()
    assert not (trace.test_mask & ~trace.sifted_mask).any()
    assert trace.test_length == int(np.ceil(trace.sifted_length * 0.25))
    assert trace.key_length == trace.sifted_length - trace.test_length


@pytest.mark.parametrize(
    ("params", "label"),
    [
        (ChannelParams(), "clean"),
        (ChannelParams(noise_prob=0.1), "noise_only"),
        (ChannelParams(intercept_prob=0.1), "intercept_resend"),
        (ChannelParams(pns_prob=0.1), "pns"),
        (ChannelParams(noise_prob=0.1, pns_prob=0.1), "noise+attack"),
    ],
)
def test_label_is_derived_from_parameters(params: ChannelParams, label: str) -> None:
    assert params.label == label


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_invalid_probabilities_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        ChannelParams(noise_prob=bad)


def test_invalid_simulator_arguments() -> None:
    with pytest.raises(ValueError):
        BB84Simulator(num_qubits=0)
    with pytest.raises(ValueError):
        BB84Simulator(backend="gpu")  # type: ignore[arg-type]


def test_tiny_session_does_not_crash() -> None:
    trace = BB84Simulator(num_qubits=1, seed=0).simulate()
    assert trace.num_pulses == 1
