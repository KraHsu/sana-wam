from __future__ import annotations

import numpy as np

from sana_wam.deploy.policy_server import _public_policy_step_info


def test_public_policy_step_info_keeps_boundary_coordinates_only():
    info = {
        "policy_step": 84,
        "generation_index": 3,
        "chunk_offset": 0,
        "buffer_remaining": 27,
        "generated": True,
        "chunk_len": 28,
        "chunk_index": 3,
        "action_frame_id": 9,
        "ar_step_after": 4,
        "predicted_actions": np.zeros((28, 20), dtype=np.float32),
        "predicted_actions_normalized": np.ones((28, 20), dtype=np.float32),
        "cache": {"layers": [1, 2, 3]},
        "cache_feedback": {
            "status": "predicted",
            "reason": "configured",
            "source_generation_index": 2,
            "private_array": np.zeros((28, 20), dtype=np.float32),
        },
    }

    assert _public_policy_step_info(info) == {
        "policy_step": 84,
        "generation_index": 3,
        "chunk_offset": 0,
        "buffer_remaining": 27,
        "generated": True,
        "chunk_len": 28,
        "chunk_index": 3,
        "action_frame_id": 9,
        "ar_step_after": 4,
        "cache_feedback": {
            "status": "predicted",
            "reason": "configured",
            "source_generation_index": 2,
        },
    }


def test_public_policy_step_info_drops_non_scalar_boundary_values():
    assert (
        _public_policy_step_info(
            {
                "generated": np.bool_(True),
                "chunk_offset": np.asarray(0),
                "generation_index": None,
            }
        )
        == {}
    )
