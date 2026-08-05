"""Reserved-marker tests for generic GPU/cache entrypoints.

Submitted in Stage 0 for later execution; this file is intentionally NOT_RUN.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(
        "cach_stage0_guard_" + path.stem, path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def test_vae_cache_entrypoint_rejects_before_override_or_cache_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, path = _load("scripts/precompute_vae_latents.py")
    config = tmp_path / "cach-stage0.yaml"
    config.write_text("cach_stage0:\n  status: draft_blocked\n", encoding="utf-8")
    cache = tmp_path / "cache-must-not-exist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--config",
            str(config),
            "--cache-dir",
            str(cache),
            "cach_stage0=null",
        ],
    )
    with pytest.raises(RuntimeError, match="non-executable"):
        module.main()
    assert not cache.exists()


def test_gpu_smoke_entrypoint_rejects_before_checkpoint_or_gpu_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, path = _load("scripts/smoke_ar_lownoise_gpu.py")
    config = tmp_path / "cach-stage0.yaml"
    config.write_text("cach_stage0:\n  status: draft_blocked\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--deploy-config",
            str(config),
            "--ckpt-dir",
            str(tmp_path / "missing-checkpoint-root"),
        ],
    )
    with pytest.raises(RuntimeError, match="non-executable"):
        module.main()
