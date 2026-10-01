from __future__ import annotations

import pytest

from adaptive_qkd.quantum import qiskit_available
from adaptive_qkd.simulator import BB84Simulator, ChannelParams

pytestmark = pytest.mark.skipif(not qiskit_available(), reason="qiskit extra not installed")


def qiskit_qber(params: ChannelParams, n: int = 3000, seed: int = 11) -> float:
    trace = BB84Simulator(num_qubits=n, seed=seed, backend="qiskit").simulate(params)
    sifted = trace.sifted_mask
    return float((trace.error_mask & sifted).sum() / sifted.sum())


def test_qiskit_clean_channel_is_error_free() -> None:
    assert qiskit_qber(ChannelParams(), n=500) == 0.0


@pytest.mark.parametrize(
    ("params", "expected"),
    [(ChannelParams(intercept_prob=1.0), 0.25), (ChannelParams(noise_prob=0.2), 0.10)],
)
def test_qiskit_backend_agrees_with_theory(params: ChannelParams, expected: float) -> None:
    assert qiskit_qber(params) == pytest.approx(expected, abs=0.03)
