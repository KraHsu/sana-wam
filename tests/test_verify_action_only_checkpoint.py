from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from scripts.verify_action_only_checkpoint import compare_checkpoints


def _write(path, *, action=1.0, video=2.0):
    save_file(
        {
            "action_backbone.weight": torch.tensor([action]),
            "video_backbone.weight": torch.tensor([video]),
            "proprio_encoder.weight": torch.tensor([3.0]),
        },
        str(path),
    )


def test_identical_checkpoint_gate(tmp_path):
    base = tmp_path / "base.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write(base)
    _write(candidate)

    report = compare_checkpoints(str(base), str(candidate), expect="identical")

    assert report["changed_trainable_tensors"] == 0
    assert report["changed_frozen_tensors"] == 0


def test_action_only_change_gate_accepts_action_delta(tmp_path):
    base = tmp_path / "base.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write(base)
    _write(candidate, action=1.5)

    report = compare_checkpoints(str(base), str(candidate))

    assert report["changed_trainable_tensors"] == 1
    assert report["changed_frozen_tensors"] == 0


def test_action_only_change_gate_rejects_frozen_delta(tmp_path):
    base = tmp_path / "base.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write(base)
    _write(candidate, action=1.5, video=2.5)

    with pytest.raises(RuntimeError, match="frozen tensors changed"):
        compare_checkpoints(str(base), str(candidate))


def test_action_only_change_gate_requires_an_update(tmp_path):
    base = tmp_path / "base.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write(base)
    _write(candidate)

    with pytest.raises(RuntimeError, match="no trainable tensor changed"):
        compare_checkpoints(str(base), str(candidate))


def test_action_only_change_gate_can_bound_relative_drift(tmp_path):
    base = tmp_path / "base.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write(base)
    _write(candidate, action=1.5)

    with pytest.raises(RuntimeError, match="relative L2 drift"):
        compare_checkpoints(str(base), str(candidate), max_relative_l2=0.1)
