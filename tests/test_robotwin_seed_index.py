from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SINGLE_EVAL = REPO_ROOT / "benchmarks" / "robotwin" / "single_eval.sh"


def _run_single_eval(
    tmp_path: Path,
    seed_index: str | None,
    seed_offset: str | None = None,
) -> subprocess.CompletedProcess[str]:
    robotwin = tmp_path / "RoboTwin"
    robotwin.mkdir()
    fake_python = tmp_path / "fake_robotwin_python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8"
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "ROBOTWIN_PATH": str(robotwin),
            "ROBOTWIN_PYTHON": str(fake_python),
        }
    )
    if seed_index is None:
        env.pop("ROBOTWIN_ENV_SEED_INDEX", None)
    else:
        env["ROBOTWIN_ENV_SEED_INDEX"] = seed_index
    if seed_offset is None:
        env.pop("ROBOTWIN_ENV_SEED_OFFSET", None)
    else:
        env["ROBOTWIN_ENV_SEED_OFFSET"] = seed_offset

    return subprocess.run(
        [
            "bash",
            str(SINGLE_EVAL),
            "adjust_bottle",
            "demo_clean",
            "seed-index-test",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("seed_index", "expected"),
    [(None, "0"), ("0", "0"), ("1", "1"), ("17", "17")],
)
def test_single_eval_forwards_environment_seed_index(tmp_path, seed_index, expected):
    result = _run_single_eval(tmp_path, seed_index)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    seed_flag = lines.index("--seed")
    assert lines[seed_flag + 1] == expected
    assert f"seed         : {expected}" in lines


@pytest.mark.parametrize("seed_index", ["-1", "+1", "1.0", "one", "1 2"])
def test_single_eval_rejects_invalid_environment_seed_index(tmp_path, seed_index):
    result = _run_single_eval(tmp_path, seed_index)

    assert result.returncode != 0
    assert "Invalid ROBOTWIN_ENV_SEED_INDEX" in result.stderr


@pytest.mark.parametrize("seed_offset", ["-1", "+1", "1.0", "one", "100000"])
def test_single_eval_rejects_invalid_environment_seed_offset(tmp_path, seed_offset):
    result = _run_single_eval(tmp_path, "1", seed_offset)

    assert result.returncode != 0
    assert "Invalid ROBOTWIN_ENV_SEED_OFFSET" in result.stderr


def test_single_eval_accepts_environment_seed_offset(tmp_path):
    result = _run_single_eval(tmp_path, "1", "7")

    assert result.returncode == 0, result.stderr
    assert "seed offset  : 7" in result.stdout
