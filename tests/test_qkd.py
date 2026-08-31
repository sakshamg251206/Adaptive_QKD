"""
tests/test_qkd.py
Unit tests for BB84 Quantum Key Distribution simulator and adaptive protocol.
"""

import pytest
import numpy as np
from bb84_sim import BB84Simulator
from adaptive_protocol import AdaptiveQKDProtocol, binary_entropy


def test_bb84_clean_channel():
    """Verify clean channel yields QBER ~ 0%."""
    sim = BB84Simulator(num_qubits=1000, seed=42)
    res = sim.simulate_session_vectorized(noise_prob=0.0, attack_type='clean')
    assert res['qber'] == 0.0
    assert res['sifted_key_length'] > 400
    assert res['sifting_ratio'] > 0.40


def test_bb84_intercept_resend_attack():
    """Verify 100% Intercept-Resend attack yields QBER ~ 25%."""
    sim = BB84Simulator(num_qubits=1000, seed=42)
    res = sim.simulate_session_vectorized(attack_type='intercept_resend', intercept_prob=1.0)
    assert 0.20 <= res['qber'] <= 0.30


def test_bb84_noise_only():
    """Verify depolarizing noise yields expected error rate."""
    sim = BB84Simulator(num_qubits=1000, seed=42)
    res = sim.simulate_session_vectorized(noise_prob=0.10, attack_type='noise_only')
    assert 0.01 <= res['qber'] <= 0.08


def test_binary_entropy():
    """Verify binary entropy calculations."""
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert abs(binary_entropy(0.5) - 1.0) < 1e-6
    assert 0.45 < binary_entropy(0.1) < 0.50


def test_adaptive_protocol_session():
    """Verify adaptive protocol runs and produces valid audit log."""
    sim = BB84Simulator(num_qubits=1000, seed=42)
    protocol = AdaptiveQKDProtocol()
    res = protocol.run_adaptive_session(sim=sim, noise_prob=0.02, attack_type='clean', checkpoints=3)
    assert 'final_action' in res
    assert res['final_action'] in ('CONTINUE', 'HARDEN', 'ABORT')
    assert len(res['audit_log']) > 0
