"""Tests for the pre-model CACH Stage-0 deploy guard.

These tests are submitted for review and intentionally NOT_RUN in Stage 0.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ENTRYPOINT = ROOT / "scripts/deploy.py"


def _load_deploy_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "cach_stage0_guard_deploy_entrypoint", DEPLOY_ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dotlist_cannot_remove_deploy_guard_before_merge_or_model_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_deploy_entrypoint()
    config = tmp_path / "cach-stage0-deploy.yaml"
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
            str(DEPLOY_ENTRYPOINT),
            "--deploy-config",
            str(config),
            "--ckpt-dir",
            str(tmp_path / "not-used"),
            "cach_stage0=null",
            "model.architecture.variant=gdn_autoregressive",
        ],
    )
    policy_server_before = sys.modules.get("sana_wam.deploy.policy_server")
    with pytest.raises(RuntimeError, match="non-executable"):
        module.main()
    assert merge_called is False
    assert sys.modules.get("sana_wam.deploy.policy_server") is policy_server_before
