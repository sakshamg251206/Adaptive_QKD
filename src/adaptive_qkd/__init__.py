"""Adaptive QKD: machine-learning eavesdropping detection for BB84 quantum key distribution."""

from adaptive_qkd.simulator import BB84Simulator, ChannelParams, SessionTrace

__all__ = ["BB84Simulator", "ChannelParams", "SessionTrace", "__version__"]
__version__ = "2.0.0"
