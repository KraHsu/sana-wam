from __future__ import annotations

import ast
import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/smoke_libero_ar_t3_heldout_recipe_transfer_gpu.py"
T2_RUNNER = ROOT / "scripts/smoke_libero_ar_t2_fixed_sample_20step_gpu.py"


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


def test_t3_runner_preserves_t2_update_path_and_has_exact_probe_budget() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = _call_names(tree)

    assert calls.count("optimizer.step") == 1
    assert calls.count("architecture.compute_loss") == 1
    assert "loss.backward" in calls
    assert "trainer._sync_master_gradients" in calls
    assert "trainer._copy_master_parameters_to_model" in calls
    assert "trainer.train" not in calls
    assert "torch.save" not in calls
    assert "torch.load" not in calls
    assert "architecture.save_checkpoint" not in calls
    assert "architecture.load_checkpoint" not in calls
    assert "manage_checkpoints" not in calls

    loop = _step_loop_source(source, tree)
    fragments = [
        "optimizer.zero_grad(set_to_none=True)",
        "parameter.grad = None",
        "fixed_forward(",
        "clip_grad_norm_",
        "trainer._sync_master_gradients(master_pairs)",
        "optimizer.step()",
        "trainer._copy_master_parameters_to_model(master_pairs)",
    ]
    offsets = [loop.index(fragment) for fragment in fragments]
    assert offsets == sorted(offsets)

    pre_offset = source.index("heldout_before[heldout_seed] = scalars")
    initial_offset = source.index("initial_probe, initial_signature")
    loop_offset = source.index("for step_index in range(T3_UPDATE_STEPS)")
    final_offset = source.index("final_probe, final_signature")
    post_offset = source.index("heldout_after[heldout_seed] = scalars")
    assert pre_offset < initial_offset < loop_offset < final_offset < post_offset

    assert "architecture_forward_calls != T3_UPDATE_STEPS + 8" in source
    assert "len(set(recipe_identity_signatures)) != 4" in source
    assert '"architecture_forwards_total": T3_UPDATE_STEPS + 8' in source
    assert '"architecture_measurement_forwards": 8' in source
    assert '"heldout_measurement_forwards": 6' in source
    assert '"main_recipe_measurement_forwards": 2' in source
    assert '"unique_recipe_identity_count"' in source
    assert "recipe_seed=heldout_seed" in source
    assert "recipe_seed=args.loss_recipe_seed" in source
    assert '"T3_HELDOUT_RECIPE_TRANSFER_GO"' in source
    assert '"T3_HELDOUT_RECIPE_TRANSFER_INCONCLUSIVE"' in source
    assert '"role": "secondary_diagnostic_not_t3_primary_gate"' in source
    primary_verdict_block = source[
        source.index("verdict_reasons = []") : source.index("verdict = (")
    ]
    assert 'heldout_transfer["transfer_gate"]' in primary_verdict_block
    assert "projected_roots" not in primary_verdict_block
    assert "learnability" not in primary_verdict_block
    assert '"formal_training_executed": False' in source
    assert '"benchmark_evaluation_executed": False' in source
    assert '"simulator_executed": False' in source
    assert '"nonce": args.nonce' in source
    assert '"root": str(run_root)' in source
    assert "run-root basename does not bind the nonce" in source


def test_t3_heldout_calls_are_update_free_and_state_guarded() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    heldout_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "heldout_seed"
    ]
    assert len(heldout_loops) == 2
    for loop in heldout_loops:
        calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)]
        fixed_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "fixed_forward"
        ]
        assert len(fixed_calls) == 1
        keywords = {keyword.arg: keyword.value for keyword in fixed_calls[0].keywords}
        assert isinstance(keywords["backward"], ast.Constant)
        assert keywords["backward"].value is False
        loop_calls = _call_names(loop)
        assert "loss.backward" not in loop_calls
        assert "optimizer.step" not in loop_calls
        assert "trainer._sync_master_gradients" not in loop_calls
        assert "trainer._copy_master_parameters_to_model" not in loop_calls

    assert source.count("probe_mutation_snapshot() !=") == 2
    assert '"architecture_module_modes"' in source
    assert '"optimizer_state"' in source
    assert "fixed recipe did not restore caller RNG state" in source
    assert '"rng_state_restored_after_every_forward": True' in source


def test_t3_seed_derivation_is_frozen_and_predecessor_is_unchanged() -> None:
    from scripts.smoke_libero_ar_t3_heldout_recipe_transfer_gpu import (
        T2_PREDECESSOR_RUNNER_SHA256,
        T3_HELDOUT_RECIPE_SEEDS,
        T3_LOSS_RECIPE_SEED,
        _heldout_seed_manifest,
    )

    assert T3_HELDOUT_RECIPE_SEEDS == (897500337, 2142397805, 1238092489)
    assert len(set(T3_HELDOUT_RECIPE_SEEDS)) == 3
    assert T3_LOSS_RECIPE_SEED not in T3_HELDOUT_RECIPE_SEEDS
    assert _heldout_seed_manifest() == [
        {
            "derivation_index": 1,
            "payload_sha256": (
                "08c9698412591ea0f219f56d2f293482c06d2303fb70c2de85f41a314f64d856"
            ),
            "seed": 897500337,
        },
        {
            "derivation_index": 2,
            "payload_sha256": (
                "ef4652c542991a4ab1b7fc1d031e27ca0cda41d8cc509696458b83c8d1b8827c"
            ),
            "seed": 2142397805,
        },
        {
            "derivation_index": 3,
            "payload_sha256": (
                "f24a4b3b80a29dcc8b8dd1bf36cfcc5d747a69dd3ed50889112c6c3a76f055cd"
            ),
            "seed": 1238092489,
        },
    ]
    assert hashlib.sha256(T2_RUNNER.read_bytes()).hexdigest() == (
        T2_PREDECESSOR_RUNNER_SHA256
    )


def test_t3_transfer_gate_uses_all_three_paired_ratios() -> None:
    from scripts.smoke_libero_ar_t3_heldout_recipe_transfer_gpu import (
        _heldout_transfer_summary,
    )

    before = {897500337: 10.0, 2142397805: 20.0, 1238092489: 40.0}
    boundary = _heldout_transfer_summary(
        before,
        {897500337: 9.5, 2142397805: 18.0, 1238092489: 50.0},
    )
    assert boundary["median_post_to_pre_ratio"] == 0.95
    assert boundary["improved_count"] == 2
    assert boundary["transfer_gate"] is True

    unstable = _heldout_transfer_summary(
        before,
        {897500337: 9.6, 2142397805: 19.2, 1238092489: 20.0},
    )
    assert unstable["improved_count"] == 3
    assert unstable["median_post_to_pre_ratio"] == 0.96
    assert unstable["transfer_gate"] is False

    with pytest.raises(ValueError, match="frozen set"):
        _heldout_transfer_summary(
            {897500337: 10.0},
            {897500337: 9.0},
        )
    with pytest.raises(ValueError, match="positive pre-loss"):
        _heldout_transfer_summary(
            {897500337: 0.0, 2142397805: 20.0, 1238092489: 40.0},
            {897500337: 0.0, 2142397805: 18.0, 1238092489: 36.0},
        )


def test_t3_fixed_rng_recipe_repeats_and_restores_cpu_rngs() -> None:
    from scripts.smoke_libero_ar_t3_heldout_recipe_transfer_gpu import (
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
        with _fixed_rng_recipe(torch, seed=897500337, cuda_device=None):
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
