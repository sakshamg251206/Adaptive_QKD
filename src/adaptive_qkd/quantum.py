"""Qiskit Aer backend for the BB84 channel.

Each pulse is an independent qubit in one circuit:

* Alice prepares ``|0>``/``|1>`` (Z basis) or ``|+>``/``|->`` (X basis).
* An intercepting Eve measures in her basis (mid-circuit measurement) and the
  collapsed qubit is re-encoded in that basis, which is exactly a re-send.
* Depolarizing noise is inserted as an Aer ``QuantumError`` instruction.
* Bob rotates into his basis and measures.

The circuit only contains Clifford gates and Pauli noise, so it runs on Aer's
stabilizer method, which scales to thousands of qubits in a single shot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from adaptive_qkd.simulator import ChannelParams, IntArray

_INSTALL_HINT = "The Qiskit backend requires the optional 'quantum' extra: pip install -e '.[quantum]'"


def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
    except ImportError:
        return False
    return True


def transmit_qiskit(
    alice_bits: IntArray,
    alice_bases: IntArray,
    bob_bases: IntArray,
    params: ChannelParams,
    rng: np.random.Generator,
) -> IntArray:
    """Execute the BB84 channel as a quantum circuit and return Bob's bits."""
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import depolarizing_error
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_INSTALL_HINT) from exc

    n = alice_bits.size
    intercepted = rng.random(n) < params.intercept_prob
    eve_bases = rng.integers(0, 2, size=n, dtype=np.int8)

    qreg = QuantumRegister(n, "q")
    eve_reg = ClassicalRegister(n, "eve")
    bob_reg = ClassicalRegister(n, "bob")
    qc = QuantumCircuit(qreg, eve_reg, bob_reg)
    noise = depolarizing_error(params.noise_prob, 1) if params.noise_prob > 0 else None

    for i in range(n):
        if alice_bits[i]:
            qc.x(i)
        if alice_bases[i]:
            qc.h(i)
        if intercepted[i]:
            if eve_bases[i]:
                qc.h(i)
            qc.measure(i, eve_reg[i])
            if eve_bases[i]:
                qc.h(i)
        if noise is not None:
            qc.append(noise, [i])
        if bob_bases[i]:
            qc.h(i)
        qc.measure(i, bob_reg[i])

    simulator = AerSimulator(method="stabilizer", seed_simulator=int(rng.integers(0, 2**31 - 1)))
    memory = simulator.run(qc, shots=1, memory=True).result().get_memory()[0]
    # Registers are space separated, most recently added first; bit 0 is rightmost.
    bob_bitstring = memory.split(" ")[0]
    return np.array([int(b) for b in reversed(bob_bitstring)], dtype=np.int8)
