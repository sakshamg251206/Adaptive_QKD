"""Closed-loop adaptive BB84 protocol.

The protocol inspects a session at evenly spaced checkpoints. At each
checkpoint it computes features on the part of the session received so far,
asks the threat classifier for the probability that an eavesdropper is present
(``P(attack)``) and picks an action:

* ``CONTINUE`` - ``P(attack) < harden_threshold``: standard post-processing.
* ``HARDEN``   - up to ``abort_threshold``: privacy amplification assumes that a
  share of the key (``harden_tagged_fraction``) is already known to Eve.
* ``ABORT``    - ``P(attack) >= abort_threshold``: discard the session.

An abort ends the session immediately. Otherwise the decision is re-evaluated
as evidence accumulates, and the last checkpoint (which sees the whole
session) decides how the key is post-processed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from adaptive_qkd.config import Paths
from adaptive_qkd.features import extract_features
from adaptive_qkd.keyrate import DEFAULT_EC_EFFICIENCY, secret_fraction
from adaptive_qkd.models import ThreatClassifier, attack_probability
from adaptive_qkd.simulator import SessionTrace

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Protocol actions, ordered by severity."""

    CONTINUE = 0
    HARDEN = 1
    ABORT = 2


@dataclass(frozen=True)
class PolicyConfig:
    harden_threshold: float = 0.25
    abort_threshold: float = 0.70
    checkpoints: int = 4
    harden_tagged_fraction: float = 0.5
    ec_efficiency: float = DEFAULT_EC_EFFICIENCY

    def __post_init__(self) -> None:
        if not 0.0 <= self.harden_threshold <= self.abort_threshold <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= harden_threshold <= abort_threshold <= 1")
        if self.checkpoints < 1:
            raise ValueError("checkpoints must be at least 1")
        if not 0.0 <= self.harden_tagged_fraction < 1.0:
            raise ValueError("harden_tagged_fraction must be in [0, 1)")

    def action_for(self, p_attack: float) -> Action:
        if p_attack >= self.abort_threshold:
            return Action.ABORT
        if p_attack >= self.harden_threshold:
            return Action.HARDEN
        return Action.CONTINUE

    def assumed_tagged_fraction(self, action: Action) -> float:
        return self.harden_tagged_fraction if action is Action.HARDEN else 0.0


@dataclass(frozen=True)
class Checkpoint:
    index: int
    progress: float
    features: dict[str, float]
    class_probs: dict[str, float]
    p_attack: float
    action: Action

    @property
    def predicted_class(self) -> str:
        if not self.class_probs:
            return "unknown"
        return max(self.class_probs, key=self.class_probs.__getitem__)


@dataclass(frozen=True)
class KeyOutcome:
    """Result of post-processing a session under one policy decision.

    ``secure_bits`` is the key length that is actually safe given the true
    channel (known only to the simulator). A release is *unsafe* when the
    protocol kept more bits than that, i.e. Eve may know part of the key.
    """

    action: Action
    released_bits: int
    secure_bits: int
    num_pulses: int

    @property
    def unsafe(self) -> bool:
        return self.released_bits > self.secure_bits

    @property
    def key_rate(self) -> float:
        """Released key bits per transmitted pulse."""
        return self.released_bits / self.num_pulses

    @property
    def secure_key_rate(self) -> float:
        """Released key bits per pulse, counting an unsafe release as zero."""
        return 0.0 if self.unsafe else self.key_rate


def finalize_key(trace: SessionTrace, action: Action, config: PolicyConfig) -> KeyOutcome:
    """Apply error correction and privacy amplification for ``action``.

    The estimated QBER comes from the disclosed test bits. The secure length is
    computed with the *true* tagged fraction (the PNS strength) and is used only
    to score whether the release was safe.
    """
    if action is Action.ABORT or trace.test_length == 0:
        return KeyOutcome(Action.ABORT, 0, 0, trace.num_pulses)
    qber = float(trace.error_mask[trace.test_mask].mean())
    n_key = trace.key_length
    assumed = config.assumed_tagged_fraction(action)
    released = int(n_key * secret_fraction(qber, assumed, config.ec_efficiency))
    secure = int(n_key * secret_fraction(qber, min(trace.params.pns_prob, 0.999), config.ec_efficiency))
    return KeyOutcome(action, released, secure, trace.num_pulses)


@dataclass(frozen=True)
class SessionResult:
    trace: SessionTrace
    checkpoints: list[Checkpoint]
    outcome: KeyOutcome

    @property
    def final_action(self) -> Action:
        return self.outcome.action

    @property
    def final_checkpoint(self) -> Checkpoint:
        return self.checkpoints[-1]


class AdaptiveQKDProtocol:
    """Runs the checkpointed decision loop for a simulated session."""

    def __init__(self, classifier: ThreatClassifier | None, config: PolicyConfig | None = None) -> None:
        self.classifier = classifier
        self.config = config or PolicyConfig()

    @classmethod
    def from_paths(cls, paths: Paths | None = None, config: PolicyConfig | None = None) -> AdaptiveQKDProtocol:
        """Load the trained classifier, falling back to a QBER heuristic if it is missing."""
        paths = paths or Paths.from_env()
        return cls(load_classifier(paths.rf_model), config)

    @property
    def uses_model(self) -> bool:
        return self.classifier is not None

    def assess(self, features: dict[str, float]) -> tuple[dict[str, float], float]:
        """Return class probabilities and ``P(attack)`` for a feature vector."""
        if self.classifier is None:
            return {}, heuristic_attack_probability(features["qber"])
        class_probs = self.classifier.predict_proba(features)
        return class_probs, attack_probability(class_probs)

    def run(self, trace: SessionTrace) -> SessionResult:
        checkpoints: list[Checkpoint] = []
        action = Action.CONTINUE
        n = self.config.checkpoints
        for index in range(1, n + 1):
            progress = index / n
            features = extract_features(trace, progress)
            class_probs, p_attack = self.assess(features)
            action = self.config.action_for(p_attack)
            checkpoints.append(Checkpoint(index, progress, features, class_probs, p_attack, action))
            if action is Action.ABORT:
                break
        return SessionResult(trace, checkpoints, finalize_key(trace, action, self.config))


def heuristic_attack_probability(qber: float) -> float:
    """QBER-only fallback used when no trained model is available."""
    return min(1.0, max(0.0, (qber - 0.04) / 0.20))


def load_classifier(path: Path) -> ThreatClassifier | None:
    if not path.exists():
        logger.warning("No trained classifier at %s; using the QBER heuristic", path)
        return None
    return ThreatClassifier.load(path)
