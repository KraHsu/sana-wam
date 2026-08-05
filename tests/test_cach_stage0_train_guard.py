"""Tests for the pre-torch CACH Stage-0 guard.

The tests are part of the Stage-0 review diff and intentionally remain
NOT_RUN until a later validation authorization.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRYPOINT = ROOT / "scripts/train.py"


def _load_train_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "cach_stage0_guard_train_entrypoint", TRAIN_ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("marker_value", [{"status": "draft_blocked"}, None, False])
def test_cach_stage0_marker_is_rejected(marker_value: object) -> None:
    module = _load_train_entrypoint()
    base = OmegaConf.create(
        {
            "cach_stage0": marker_value,
            "model": {"architecture": {"variant": "cach_sana_wam_v0"}},
        }
    )
    with pytest.raises(RuntimeError, match="non-executable"):
        module.reject_cach_stage0_base_config(
            base, entrypoint="tests/train-entrypoint"
        )


@pytest.mark.parametrize("base", [None, "legacy", ["cach_stage0"]])
def test_guard_rejects_non_mapping_base_configs(base: object) -> None:
    module = _load_train_entrypoint()
    with pytest.raises(RuntimeError, match="must be a mapping"):
        module.reject_cach_stage0_base_config(
            base, entrypoint="tests/train-entrypoint"
        )


def test_dotlist_cannot_remove_guard_before_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_train_entrypoint()
    config = tmp_path / "cach-stage0.yaml"
    config.write_text(
        "cach_stage0:\n"
        "  status: draft_blocked\n"
        "model:\n"
        "  architecture:\n"
        "    variant: cach_sana_wam_v0\n",
        encoding="utf-8",
    )
    merge_called = False

    def forbidden_merge(*_args, **_kwargs):
        nonlocal merge_called
        merge_called = True
        raise AssertionError("guard must run before OmegaConf.merge")

    monkeypatch.setattr(module.OmegaConf, "merge", forbidden_merge)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRAIN_ENTRYPOINT),
            "--config",
            str(config),
            "cach_stage0=null",
            "model.architecture.variant=gdn_autoregressive",
        ],
    )
    with pytest.raises(RuntimeError, match="non-executable"):
        module.main()
    assert merge_called is False
