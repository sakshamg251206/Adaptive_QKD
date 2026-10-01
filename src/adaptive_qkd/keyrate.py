"""Secret-key-rate formulas for BB84 post-processing.

The rates are asymptotic (infinite-key) bounds; finite-size corrections are not
modelled.

For a QBER ``e`` and a fraction ``tagged`` of pulses whose value may be known to
Eve (e.g. multi-photon pulses exposed to a photon-number-splitting attack), the
GLLP bound gives the extractable secret fraction of the remaining sifted key::

    r = (1 - tagged) * (1 - h(e / (1 - tagged))) - f_ec * h(e)

where ``h`` is the binary entropy and ``f_ec >= 1`` is the error-correction
inefficiency. With ``tagged = 0`` this reduces to the Shor-Preskill rate
``1 - h(e) - f_ec * h(e)``.
"""

from __future__ import annotations

import math

DEFAULT_EC_EFFICIENCY = 1.1


def binary_entropy(p: float) -> float:
    """Binary entropy ``h(p)`` in bits; ``h(0) = h(1) = 0``."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def secret_fraction(qber: float, tagged: float = 0.0, ec_efficiency: float = DEFAULT_EC_EFFICIENCY) -> float:
    """Fraction of the sifted key that survives error correction and privacy amplification."""
    if not 0.0 <= tagged < 1.0:
        raise ValueError(f"tagged fraction must be in [0, 1), got {tagged}")
    if ec_efficiency < 1.0:
        raise ValueError(f"ec_efficiency must be >= 1, got {ec_efficiency}")
    untagged = 1.0 - tagged
    phase_error = min(0.5, qber / untagged)
    rate = untagged * (1.0 - binary_entropy(phase_error)) - ec_efficiency * binary_entropy(qber)
    return max(0.0, rate)
