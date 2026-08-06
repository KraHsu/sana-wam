from __future__ import annotations

import ast
import random
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/smoke_libero_ar_t2_fixed_sample_20step_gpu.py"


def _call_names(tree: ast.AST) -> list[str]:
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        parts = []
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        if parts:
            result.append(".".join(reversed(parts)))
    return result


def _step_loop_source(source: str, tree: ast.AST) -> str:
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "step_index"
    ]
    assert len(loops) == 1
    segment = ast.get_source_segment(source, loops[0])
    assert segment is not None
    return segment


def test_t2_runner_has_bounded_update_and_checkpoint_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = _call_names(tree)

    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"T2_UPDATE_STEPS", "T2_LEARNABILITY_RATIO"}
    }
    assert constants == {
        "T2_LEARNABILITY_RATIO": 0.95,
        "T2_UPDATE_STEPS": 20,
    }
    assert calls.count("optimizer.step") == 1
    assert "architecture.prepare_inputs" in calls
    assert "architecture.compute_loss" in calls
    assert "loss.backward" in calls
    assert "torch.nn.utils.clip_grad_norm_" in calls
    assert "trainer._sync_master_gradients" in calls
    assert "trainer._copy_master_parameters_to_model" in calls
    assert "trainer.train" not in calls
    assert "torch.save" not in calls
    assert "torch.load" not in calls
    assert "architecture.save_checkpoint" not in calls
    assert "architecture.load_checkpoint" not in calls
    assert "manage_checkpoints" not in calls

    loop = _step_loop_source(source, tree)
    ordered_fragments = [
        "optimizer.zero_grad(set_to_none=True)",
        "parameter.grad = None",
        "fixed_forward(backward=True)",
        "clip_grad_norm_",
        "trainer._sync_master_gradients(master_pairs)",
        "optimizer.step()",
        "trainer._copy_master_parameters_to_model(master_pairs)",
    ]
    offsets = [loop.index(fragment) for fragment in ordered_fragments]
    assert offsets == sorted(offsets)

    assert "require_materialized_stats=True" in source
    assert "T2_UPDATE_STEPS + 2" in source
    assert 'not recipe_capture["action"] or not recipe_capture["video"]' in source
    assert "int(cfg.training.seed) != args.initialization_seed" in source
    assert "float(cfg.training.grad_clip) != 1.0" in source
    assert "SELECTED_STATS_SHA256" in source
    assert "SELECTED_POPULATION_SHA256" in source
    assert "SPATIAL_STATS_GR00T_SHA256" in source
    assert "EXPECTED_SPATIAL_SAMPLE_ASSET_SHA256" in source
    assert "T1_PREDECESSOR_RESULT_SHA256" in source
    assert '"sampled_master_update_l2"' in source
    assert '"sampled_bf16_update_l2"' in source
    assert '"sampled_optimizer_step_values"' in source
    assert "sentinel_adam_steps != {float(step_index + 1)}" in source
    assert "final_optimizer_master_ids != initial_master_ids" in source
    assert "allow_nan=False" in source
    assert '"execution_result": "PASS"' in source
    assert '"harness_result": "PASS"' in source
    assert '"result": verdict' in source
    assert '"scientific_verdict": verdict' in source
    assert '"formal_training_admission_granted": False' in source
    assert '"formal_training_executed": False' in source
    assert '"benchmark_evaluation_executed": False' in source
    assert '"simulator_executed": False' in source
    assert '"sana_wam_training_checkpoint_loaded": False' in source
    assert '"sana_wam_training_checkpoint_saved": False' in source


def test_t2_loss_gate_is_exact_and_fail_closed() -> None:
    from scripts.smoke_libero_ar_t2_fixed_sample_20step_gpu import (
        _learnability_summary,
    )

    decreasing = [10.0 - 0.25 * index for index in range(20)]
    summary = _learnability_summary(10.0, decreasing, 5.0)
    assert summary["loss_gate"] is True
    assert summary["post_to_initial_ratio"] == 0.5

    flat = _learnability_summary(10.0, [10.0] * 20, 10.0)
    assert flat["loss_gate"] is False

    endpoint_only = _learnability_summary(10.0, [10.0] * 20, 9.5)
    assert endpoint_only["loss_gate"] is False
    median_only = _learnability_summary(10.0, [10.0] * 5 + [9.0] * 15, 9.6)
    assert median_only["loss_gate"] is False
    boundary = _learnability_summary(10.0, [10.0] * 5 + [9.5] * 15, 9.5)
    assert boundary["loss_gate"] is True

    with pytest.raises(ValueError, match="exactly 20"):
        _learnability_summary(10.0, [10.0] * 19, 9.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _learnability_summary(10.0, [10.0] * 19 + [float("nan")], 9.0)
    with pytest.raises(ValueError, match="strictly positive"):
        _learnability_summary(0.0, [0.0] * 20, 0.0)


def test_t2_fixed_rng_recipe_is_repeatable_and_restores_cpu_rngs() -> None:
    from scripts.smoke_libero_ar_t2_fixed_sample_20step_gpu import (
        _fixed_rng_recipe,
    )

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    observations = []
    for _ in range(2):
        with _fixed_rng_recipe(torch, seed=777, cuda_device=None):
            observations.append(
                (random.random(), float(np.random.rand()), float(torch.rand(())))
            )
        assert random.getstate() == python_before
        numpy_after = np.random.get_state()
        assert numpy_after[0] == numpy_before[0]
        assert np.array_equal(numpy_after[1], numpy_before[1])
        assert numpy_after[2:] == numpy_before[2:]
        assert torch.equal(torch.random.get_rng_state(), torch_before)

    assert observations[0] == observations[1]
