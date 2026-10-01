from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adaptive_qkd.models import build_lstm  # noqa: E402


def test_lstm_output_shape() -> None:
    model = build_lstm(num_classes=5)
    logits = model(torch.zeros(8, 20, 1))
    assert logits.shape == (8, 5)
