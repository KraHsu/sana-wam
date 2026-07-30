from __future__ import annotations

import pytest
import torch


class _FakeArchitecture:
    dtype = torch.float32
    device = torch.device("cpu")
    uses_proprioception = False
    _use_gradient_checkpointing = False
    _use_gradient_checkpointing_offload = False
    _max_timestep_boundary = 1.0
    _min_timestep_boundary = 0.0

    def preprocess(self, **kwargs):
        del kwargs
        return {}


def _sample(prompt: str, row=None) -> dict:
    sample = {
        "video": [object()],
        "prompt": prompt,
        "action": torch.zeros(4, 2),
    }
    if row is not None:
        sample["phase6_plan_row"] = row
    return sample


def test_prepare_inputs_preserves_exact_phase6_row_and_detaches_container():
    from sana_wam.model.base import BaseWAMArchitecture

    row = {
        "global_step": 1,
        "identity": {
            "task_name": "adjust_bottle",
            "episode_path": "/exact/episode0.hdf5",
            "start_frame": 7,
            "prompt": "exact prompt",
        },
        "action_sigma": 0.9,
        "domain_seeds": {"video-noise": 123},
        "plan_sha256": "a" * 64,
        "identity_sha256": "b" * 64,
    }
    inputs = BaseWAMArchitecture.prepare_inputs(
        _FakeArchitecture(), [_sample("exact prompt", row)]
    )

    assert inputs["phase6_plan_rows"] == (row,)
    assert inputs["phase6_plan_rows"][0] is not row
    assert inputs["phase6_plan_rows"][0]["identity"] is not row["identity"]
    row["identity"]["prompt"] = "mutated later"
    assert (
        inputs["phase6_plan_rows"][0]["identity"]["prompt"]
        == "exact prompt"
    )


def test_prepare_inputs_rejects_mixed_or_non_plain_phase6_rows():
    from sana_wam.model.base import BaseWAMArchitecture

    with pytest.raises(ValueError, match="Mixed Phase-6 plan metadata"):
        BaseWAMArchitecture.prepare_inputs(
            _FakeArchitecture(),
            [_sample("planned", {"global_step": 1}), _sample("legacy")],
        )

    with pytest.raises(TypeError, match="plain dict"):
        BaseWAMArchitecture.prepare_inputs(
            _FakeArchitecture(), [_sample("planned", ("not", "a", "dict"))]
        )
