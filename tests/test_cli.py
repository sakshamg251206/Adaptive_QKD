from __future__ import annotations

import pytest

from adaptive_qkd.cli import main


def test_simulate_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--intercept", "1.0", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "intercept_resend" in out
    assert "qber" in out


def test_train_without_dataset_fails_cleanly(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ADAPTIVE_QKD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADAPTIVE_QKD_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ADAPTIVE_QKD_RESULTS_DIR", str(tmp_path / "results"))
    assert main(["train", "--no-lstm"]) == 1


def test_invalid_arguments_exit_with_usage_error() -> None:
    with pytest.raises(SystemExit):
        main(["simulate", "--backend", "gpu"])
    assert main(["simulate", "--noise", "2"]) == 1
