"""Checkpoint-config and direct policy-server Stage-0 denial tests.

Submitted for later execution; intentionally NOT_RUN during Stage 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def test_checkpoint_config_rejects_before_torch_or_checkpoint_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sana_wam.deploy import model_loader

    (tmp_path / "config.yaml").write_text(
        "cach_stage0:\n  status: draft_blocked\n", encoding="utf-8"
    )

    def forbidden_checkpoint_search(_path: str) -> str:
        raise AssertionError("checkpoint selection must follow the raw-config guard")

    monkeypatch.setattr(model_loader, "_find_latest_checkpoint", forbidden_checkpoint_search)
    with pytest.raises(RuntimeError, match="non-executable"):
        model_loader.load_from_checkpoint_dir(str(tmp_path))


def test_direct_policy_server_rejects_config_before_overrides(
    tmp_path: Path,
) -> None:
    from sana_wam.deploy import policy_server

    config = tmp_path / "cach-stage0.yaml"
    config.write_text("cach_stage0:\n  status: draft_blocked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-executable"):
        policy_server.main(
            [
                "--mock",
                "--config",
                str(config),
                "cach_stage0=null",
            ]
        )


def test_direct_server_builder_rejects_overlay_before_checkpoint_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sana_wam.deploy import model_loader, policy_server

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("checkpoint loader must follow caller-config guard")

    monkeypatch.setattr(model_loader, "load_from_checkpoint_dir", forbidden_loader)
    cfg = OmegaConf.create({"cach_stage0": None})
    with pytest.raises(RuntimeError, match="non-executable"):
        policy_server.build_server_from_config(cfg, "/not/used")
