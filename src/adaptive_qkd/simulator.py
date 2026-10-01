"""BB84 quantum key distribution simulator with noise and eavesdropping models.

The simulator models a single BB84 session pulse by pulse:

1. Alice picks a random bit and a random basis (Z or X) for every pulse.
2. The pulse travels through a channel that may contain
   - an intercept-resend eavesdropper, who measures a fraction of pulses in a
     random basis and re-sends what she saw,
   - a photon-number-splitting (PNS) eavesdropper, who silently keeps a copy of
     multi-photon pulses and blocks some single-photon pulses,
   - depolarizing noise, which replaces the qubit by a maximally mixed state.
3. Bob measures every detected pulse in his own random basis.
4. Alice and Bob keep only pulses where their bases matched (sifting) and
   publicly compare a random sample of the sifted bits to estimate the quantum
   bit error rate (QBER).

Two interchangeable physics backends are available. The default ``"numpy"``
backend is vectorised and fast enough to generate large datasets. The
``"qiskit"`` backend builds the equivalent quantum circuit and executes it on
Qiskit Aer, which is useful for cross-checking the vectorised model.

The PNS attack and the detector timing jitter are classical, phenomenological
models: photon-number statistics and detector electronics cannot be expressed
as qubit gates, so both backends model them identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from adaptive_qkd.config import ATTACK_CLASSES, TEST_FRACTION

Backend = Literal["numpy", "qiskit"]

# Fraction of single-photon pulses blocked by a PNS eavesdropper per unit of
# attack strength. Blocking shows up as a lower detection (and sifting) rate.
PNS_BLOCK_FACTOR = 0.25

# Synthetic detector timing-jitter model (nanoseconds). The mean and spread grow
# with each physical disturbance; see ``_sample_timing_jitter``.
BASE_JITTER_NS = 0.5
BASE_JITTER_STD_NS = 0.05

IntArray = NDArray[np.int8]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ChannelParams:
    """Physical parameters of the quantum channel for one session.

    Attributes:
        noise_prob: Depolarizing probability ``p``. With probability ``p`` the
            qubit is replaced by the maximally mixed state, so the induced bit
            error rate is ``p / 2``.
        intercept_prob: Fraction of pulses Eve intercepts and re-sends. Full
            interception induces a 25% QBER.
        pns_prob: Strength of the photon-number-splitting attack, interpreted
            as the fraction of detected pulses whose value Eve learns.
    """

    noise_prob: float = 0.0
    intercept_prob: float = 0.0
    pns_prob: float = 0.0

    def __post_init__(self) -> None:
        for name in ("noise_prob", "intercept_prob", "pns_prob"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")

    @property
    def is_attacked(self) -> bool:
        return self.intercept_prob > 0.0 or self.pns_prob > 0.0

    @property
    def label(self) -> str:
        """Ground-truth class name derived from the physical parameters."""
        if not self.is_attacked:
            return "noise_only" if self.noise_prob > 0.0 else "clean"
        if self.noise_prob > 0.0:
            return "noise+attack"
        return "intercept_resend" if self.intercept_prob > 0.0 else "pns"


@dataclass(frozen=True)
class SessionTrace:
    """Pulse-level record of one simulated BB84 session.

    All arrays have one entry per pulse, in transmission order, so features can
    be computed on any prefix of the session (see :mod:`adaptive_qkd.features`).
    """

    params: ChannelParams
    alice_bits: IntArray
    alice_bases: IntArray
    bob_bases: IntArray
    bob_bits: IntArray
    detected: BoolArray
    test_mask: BoolArray
    timing_jitter_ns: float
    backend: Backend = "numpy"
    label: str = field(default="")

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.params.label)

    @property
    def num_pulses(self) -> int:
        return int(self.alice_bits.size)

    @property
    def sifted_mask(self) -> BoolArray:
        return self.detected & (self.alice_bases == self.bob_bases)

    @property
    def error_mask(self) -> BoolArray:
        return self.alice_bits != self.bob_bits

    @property
    def sifted_length(self) -> int:
        return int(self.sifted_mask.sum())

    @property
    def test_length(self) -> int:
        return int(self.test_mask.sum())

    @property
    def key_length(self) -> int:
        """Sifted bits left for the key after the test sample is disclosed."""
        return self.sifted_length - self.test_length

    @property
    def is_attacked(self) -> bool:
        return self.label in ATTACK_CLASSES


class BB84Simulator:
    """Simulate BB84 sessions of a fixed length with a reproducible random stream."""

    def __init__(
        self,
        num_qubits: int = 1000,
        seed: int | None = None,
        backend: Backend = "numpy",
        test_fraction: float = TEST_FRACTION,
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        if backend not in ("numpy", "qiskit"):
            raise ValueError(f"Unknown backend {backend!r}; expected 'numpy' or 'qiskit'")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        self.num_qubits = num_qubits
        self.seed = seed
        self.backend: Backend = backend
        self.test_fraction = test_fraction
        self.rng = np.random.default_rng(seed)

    def simulate(self, params: ChannelParams | None = None) -> SessionTrace:
        """Run one BB84 session through the configured channel."""
        params = params or ChannelParams()
        n, rng = self.num_qubits, self.rng

        alice_bits = rng.integers(0, 2, size=n, dtype=np.int8)
        alice_bases = rng.integers(0, 2, size=n, dtype=np.int8)
        bob_bases = rng.integers(0, 2, size=n, dtype=np.int8)

        # PNS: Eve blocks a share of single-photon pulses, lowering the detection rate.
        detected = rng.random(n) >= params.pns_prob * PNS_BLOCK_FACTOR

        if self.backend == "qiskit":
            from adaptive_qkd.quantum import transmit_qiskit

            bob_bits = transmit_qiskit(alice_bits, alice_bases, bob_bases, params, rng)
        else:
            bob_bits = _transmit_numpy(alice_bits, alice_bases, bob_bases, params, rng)

        sifted = detected & (alice_bases == bob_bases)
        test_mask = _sample_test_bits(sifted, self.test_fraction, rng)

        return SessionTrace(
            params=params,
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_bits=bob_bits,
            detected=detected,
            test_mask=test_mask,
            timing_jitter_ns=_sample_timing_jitter(params, rng),
            backend=self.backend,
        )


def _transmit_numpy(
    alice_bits: IntArray,
    alice_bases: IntArray,
    bob_bases: IntArray,
    params: ChannelParams,
    rng: np.random.Generator,
) -> IntArray:
    """Vectorised channel model; returns Bob's measurement outcomes."""
    n = alice_bits.size
    state_bits = alice_bits.copy()
    state_bases = alice_bases.copy()

    # Intercept-resend: a wrong-basis measurement randomises the bit and Eve
    # re-prepares the qubit in her own basis.
    intercepted = rng.random(n) < params.intercept_prob
    eve_bases = rng.integers(0, 2, size=n, dtype=np.int8)
    disturbed = intercepted & (eve_bases != state_bases)
    state_bits[disturbed] = rng.integers(0, 2, size=int(disturbed.sum()), dtype=np.int8)
    state_bases[disturbed] = eve_bases[disturbed]

    # Depolarizing noise: a depolarized qubit yields a uniformly random outcome,
    # i.e. its bit flips with probability 1/2 in either basis.
    depolarized = rng.random(n) < params.noise_prob
    flipped = depolarized & (rng.random(n) < 0.5)
    state_bits[flipped] ^= 1

    # Bob reads the bit faithfully only when he measures in the qubit's basis.
    random_outcomes = rng.integers(0, 2, size=n, dtype=np.int8)
    return np.where(bob_bases == state_bases, state_bits, random_outcomes).astype(np.int8)


def _sample_test_bits(sifted: BoolArray, test_fraction: float, rng: np.random.Generator) -> BoolArray:
    """Choose the random subset of sifted bits disclosed for QBER estimation."""
    sifted_idx = np.flatnonzero(sifted)
    test_mask = np.zeros(sifted.size, dtype=bool)
    num_test = math.ceil(sifted_idx.size * test_fraction)
    if num_test:
        test_mask[rng.choice(sifted_idx, size=num_test, replace=False)] = True
    return test_mask


def _sample_timing_jitter(params: ChannelParams, rng: np.random.Generator) -> float:
    """Synthetic session-level detector timing jitter in nanoseconds.

    This is a stand-in for a hardware side channel: each disturbance shifts the
    mean jitter and widens its spread, with the PNS attack (which re-routes
    multi-photon pulses) leaving the strongest signature. The coefficients are
    modelling assumptions, not measured values.
    """
    mean = BASE_JITTER_NS + 2.0 * params.noise_prob + 1.2 * params.intercept_prob + 2.5 * params.pns_prob
    std = BASE_JITTER_STD_NS + 0.5 * params.noise_prob + 0.35 * params.intercept_prob + 0.6 * params.pns_prob
    return float(abs(rng.normal(mean, std)))
