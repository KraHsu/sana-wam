"""RoboTwin step-limit overrides can be pinned per evaluation baseline."""

import importlib
from pathlib import Path

import yaml


def test_default_step_limits_leave_robotwin_limits_unchanged():
    path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "robotwin"
        / "step_limits.yml"
    )
    assert yaml.safe_load(path.read_text(encoding="utf-8")) is None


def test_step_limit_path_can_be_overridden(monkeypatch, tmp_path):
    limits = tmp_path / "limits.yaml"
    limits.write_text("adjust_bottle: 400\n", encoding="utf-8")
    monkeypatch.setenv("ROBOTWIN_STEP_LIMITS_PATH", str(limits))

    from benchmarks.robotwin import sana_wam2robotwin_interface as adapter

    adapter = importlib.reload(adapter)
    assert adapter._STEP_LIMITS_PATH == str(limits)
    assert adapter._STEP_LIM_OVERRIDES == {"adjust_bottle": 400}

    monkeypatch.delenv("ROBOTWIN_STEP_LIMITS_PATH")
    adapter = importlib.reload(adapter)
    assert adapter._STEP_LIM_OVERRIDES == {}
