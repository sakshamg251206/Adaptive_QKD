from __future__ import annotations

import pytest

from adaptive_qkd.keyrate import binary_entropy, secret_fraction


def test_binary_entropy() -> None:
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.11) == pytest.approx(0.4999, abs=1e-3)


def test_secret_fraction_shor_preskill_limit() -> None:
    assert secret_fraction(0.0) == 1.0
    # With perfect error correction the rate vanishes near 11% QBER.
    assert secret_fraction(0.10, ec_efficiency=1.0) > 0
    assert secret_fraction(0.12, ec_efficiency=1.0) == 0.0


def test_tagged_pulses_reduce_the_secret_fraction() -> None:
    assert secret_fraction(0.0, tagged=0.4) == pytest.approx(0.6)
    assert secret_fraction(0.02, tagged=0.3) < secret_fraction(0.02)


def test_secret_fraction_is_monotone_in_qber() -> None:
    rates = [secret_fraction(q / 100) for q in range(0, 12)]
    assert rates == sorted(rates, reverse=True)


def test_secret_fraction_validates_inputs() -> None:
    with pytest.raises(ValueError):
        secret_fraction(0.01, tagged=1.0)
    with pytest.raises(ValueError):
        secret_fraction(0.01, ec_efficiency=0.9)
