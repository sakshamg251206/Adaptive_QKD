from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    # Point at empty directories: the app must work before anything is trained.
    for name in ("DATA", "MODEL", "RESULTS"):
        monkeypatch.setenv(f"ADAPTIVE_QKD_{name}_DIR", str(tmp_path / name.lower()))
    return AppTest.from_file(APP, default_timeout=60)


def test_dashboard_renders_without_trained_model(app: AppTest) -> None:
    app.run()
    assert not app.exception
    assert any("not been trained" in m.value for m in app.markdown)
    assert any('class="verdict' in m.value for m in app.markdown)  # an example session runs on first load


def test_dashboard_runs_each_preset(app: AppTest) -> None:
    app.run()
    expected = {"Clean channel": "clean", "Photon-number-splitting attack": "pns", "Noisy fibre": "noise_only"}
    for preset, label in expected.items():
        app.selectbox(key="preset").select(preset).run()
        next(b for b in app.button if b.label == "Run session").click().run()
        assert not app.exception
        assert app.session_state["result"][0].trace.label == label
