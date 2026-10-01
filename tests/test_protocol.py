from __future__ import annotations

import pytest

from adaptive_qkd.models import ThreatClassifier
from adaptive_qkd.protocol import Action, AdaptiveQKDProtocol, PolicyConfig, finalize_key, heuristic_attack_probability
from adaptive_qkd.simulator import BB84Simulator, ChannelParams


def test_policy_thresholds() -> None:
    config = PolicyConfig(harden_threshold=0.3, abort_threshold=0.6)
    assert config.action_for(0.1) is Action.CONTINUE
    assert config.action_for(0.3) is Action.HARDEN
    assert config.action_for(0.6) is Action.ABORT


@pytest.mark.parametrize(
    "kwargs",
    [{"harden_threshold": 0.8, "abort_threshold": 0.5}, {"checkpoints": 0}, {"harden_tagged_fraction": 1.0}],
)
def test_invalid_policy_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        PolicyConfig(**kwargs)  # type: ignore[arg-type]


def test_heuristic_fallback_without_model() -> None:
    protocol = AdaptiveQKDProtocol(classifier=None)
    assert not protocol.uses_model
    result = protocol.run(BB84Simulator(num_qubits=1000, seed=1).simulate(ChannelParams(intercept_prob=1.0)))
    assert result.final_action is Action.ABORT
    assert heuristic_attack_probability(0.0) == 0.0
    assert heuristic_attack_probability(0.5) == 1.0


def test_abort_stops_at_the_first_checkpoint() -> None:
    config = PolicyConfig(checkpoints=4)
    protocol = AdaptiveQKDProtocol(classifier=None, config=config)
    result = protocol.run(BB84Simulator(num_qubits=2000, seed=2).simulate(ChannelParams(intercept_prob=1.0)))
    assert len(result.checkpoints) == 1
    assert result.outcome.released_bits == 0


def test_finalize_key_flags_unsafe_release_under_pns() -> None:
    trace = BB84Simulator(num_qubits=2000, seed=3).simulate(ChannelParams(pns_prob=0.5))
    standard = finalize_key(trace, Action.CONTINUE, PolicyConfig())
    hardened = finalize_key(trace, Action.HARDEN, PolicyConfig(harden_tagged_fraction=0.6))
    assert standard.unsafe and standard.released_bits > 0
    assert not hardened.unsafe and 0 < hardened.released_bits < standard.released_bits


def test_finalize_key_is_safe_on_clean_channel() -> None:
    trace = BB84Simulator(num_qubits=2000, seed=4).simulate()
    outcome = finalize_key(trace, Action.CONTINUE, PolicyConfig())
    assert not outcome.unsafe
    assert outcome.released_bits == trace.key_length
    assert outcome.secure_key_rate == outcome.key_rate > 0


def test_trained_protocol_separates_clean_and_pns(classifier: ThreatClassifier) -> None:
    protocol = AdaptiveQKDProtocol(classifier)
    clean = protocol.run(BB84Simulator(num_qubits=1000, seed=10).simulate())
    pns = protocol.run(BB84Simulator(num_qubits=1000, seed=11).simulate(ChannelParams(pns_prob=0.7)))
    assert clean.final_action is Action.CONTINUE
    assert pns.final_action is Action.ABORT
    assert sum(clean.final_checkpoint.class_probs.values()) == pytest.approx(1.0)
