"""Coverage for Stage-0 reserved-marker guards on auxiliary execution surfaces.

Submitted for review and intentionally NOT_RUN during Stage 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stats_config_rejects_before_dataset_import(tmp_path: Path) -> None:
    from sana_wam.dataloader.robotwin_stats_computation import _load_yaml_config

    config = tmp_path / "cach-stage0.yaml"
    config.write_text("cach_stage0:\n  status: draft_blocked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-executable"):
        _load_yaml_config(str(config))


def test_smoke_rejects_checkpoint_config_before_gpu_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import importlib.util

    path = ROOT / "scripts/smoke_ar_lownoise_gpu.py"
    spec = importlib.util.spec_from_file_location("cach_checkpoint_guard_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    deploy_config = tmp_path / "legacy-deploy.yaml"
    deploy_config.write_text("server:\n  host: 127.0.0.1\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "config.yaml").write_text(
        "cach_stage0:\n  status: draft_blocked\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--deploy-config",
            str(deploy_config),
            "--ckpt-dir",
            str(checkpoint_root),
        ],
    )
    with pytest.raises(RuntimeError, match="non-executable"):
        module.main()


@pytest.mark.parametrize(
    ("relative", "guard_token", "mutation_token"),
    [
        (
            "scripts/run_ar_lownoise_paired_arm.sh",
            "check_cach_stage0_reserved_config.py",
            'RUN_LOCK="${RUN_DIR}.run-lock"',
        ),
        (
            "scripts/eval_ar_lownoise_seedfixed.sh",
            "check_cach_stage0_reserved_config.py",
            "mkdir -p",
        ),
        (
            "benchmarks/robotwin/single_eval.sh",
            "check_cach_stage0_reserved_config.py",
            "runtime_config=\"$(mktemp",
        ),
        (
            "benchmarks/robotwin/eval_policy_wrapper.py",
            "_reject_reserved_stage0_cli_config()",
            "_prepare_runtime_root(robotwin_path)",
        ),
    ],
)
def test_outer_guard_precedes_first_state_or_gpu_boundary(
    relative: str, guard_token: str, mutation_token: str
) -> None:
    source = (ROOT / relative).read_text(encoding="utf-8")
    assert source.rindex(guard_token) < source.rindex(mutation_token)


def test_seedfixed_shell_guard_failure_is_terminal_without_errexit() -> None:
    source = (
        ROOT / "scripts/eval_ar_lownoise_seedfixed.sh"
    ).read_text(encoding="utf-8")
    guard = source.index("check_cach_stage0_reserved_config.py")
    first_mutation = source.index("mkdir -p")
    assert "|| exit 2" in source[guard:first_mutation]
