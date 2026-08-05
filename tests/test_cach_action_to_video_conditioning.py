"""Static Stage-1 checks for the CACH action-to-video implementation seam."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.model.action_chunk_layout import (
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "src/sana_wam/model/causal_action_hybrid.py"
ADAPTER = ROOT / "src/sana_wam/model/video_backbone/sana/adapter.py"
REDUCER = ROOT / "src/sana_wam/cach/action_conditioning.py"


def _function(tree: ast.Module, name: str, *, class_name: str | None = None):
    body = tree.body
    if class_name is not None:
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = owner.body
    return next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def test_stage1_constructor_denies_before_legacy_model_construction() -> None:
    tree = ast.parse(
        ARCHITECTURE.read_text(encoding="utf-8"),
        filename=str(ARCHITECTURE),
    )
    init = _function(
        tree,
        "__init__",
        class_name="CausalActionHybridArchitecture",
    )
    calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            or isinstance(node.func, ast.Attribute)
        )
    ]
    guard_line = next(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "reject_cach_model_build"
    )
    super_line = next(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "__init__"
    )
    assert guard_line < super_line


def test_reducer_uses_layout_span_end_and_never_fixed_atc_or_mean_pooling() -> None:
    source = REDUCER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REDUCER))
    reducer = _function(tree, "reduce_end_of_bin_action_condition")
    reducer_source = ast.get_source_segment(source, reducer)
    assert reducer_source is not None
    assert "span.action_end - chunk.action_start - 1" in reducer_source
    assert "chunk.valid_action_count" in reducer_source
    assert "no_action_slot" in reducer_source
    assert not any(
        isinstance(node, ast.Name) and node.id == "atc"
        for node in ast.walk(reducer)
    )
    assert "chunk.action_slot_capacity" in reducer_source
    assert "latent_valid_mask" in reducer_source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mean"
        for node in ast.walk(reducer)
    )


def test_vendor_mapping_is_explicit_and_does_not_use_final_layer_delta_actions() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ADAPTER))
    run_chunk = _function(tree, "run_chunk", class_name="SanaVideoBackbone")
    run_source = ast.get_source_segment(source, run_chunk)
    assert run_source is not None
    # Stage 2 may compact a fixed-K valid prefix before the pinned vendor call;
    # the explicit mapping must use that validated transaction-local tensor.
    assert '{"delta_actions": vendor_action_condition}' in run_source
    assert "prefix_plan.compact_condition" in run_source
    assert "use_delta_pose_additive" in run_source
    assert "_cach_delta_pose_output_zero_verified" in run_source
    assert "identity-by-None is forbidden" in run_source
    assert "frame_valid_mask" in run_source
    assert "padded action_condition slots must be exact zero" in run_source
    assert "use_delta_actions" not in run_source


def test_cach_training_cannot_inherit_legacy_fixed_atc_compute_loss() -> None:
    source = ARCHITECTURE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ARCHITECTURE))
    compute_loss = _function(
        tree,
        "compute_loss",
        class_name="CausalActionHybridArchitecture",
    )
    assert any(isinstance(node, ast.Raise) for node in ast.walk(compute_loss))
    method_source = ast.get_source_segment(source, compute_loss)
    assert method_source is not None
    assert "fixed-atc" in method_source


def test_cach_forward_accepts_only_typed_cache_and_keeps_codec_blocked() -> None:
    source = ARCHITECTURE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ARCHITECTURE))
    forward_chunk = _function(
        tree,
        "forward_chunk",
        class_name="CausalActionHybridArchitecture",
    )
    method_source = ast.get_source_segment(source, forward_chunk)
    assert method_source is not None
    assert "cache_read_view: DenoiseReadView" in method_source
    assert "cache_scratch: CacheScratch" in method_source
    argument_names = {
        argument.arg
        for argument in (
            list(forward_chunk.args.posonlyargs)
            + list(forward_chunk.args.args)
            + list(forward_chunk.args.kwonlyargs)
        )
    }
    assert "kv_cache" not in argument_names
    assert "save_kv_cache" in argument_names
    assert "typed-cache/list[10] codec" in method_source


def _layout(valid_raw_count: int):
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_raw_count,
        video_stride=1,
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=valid_raw_count,
        video_valid_mask=(True,) * valid_raw_count,
        action_valid_mask=(True,) * (valid_raw_count - 1),
    )


def test_cpu_reducer_bootstrap_uses_no_action_and_end_of_bin() -> None:
    chunk = _layout(33).chunks[0]
    actions = torch.arange(
        chunk.action_slot_capacity * 20,
        dtype=torch.float32,
    ).view(1, chunk.action_slot_capacity, 20)
    no_action = torch.full((20,), -7.0)

    reduced = reduce_end_of_bin_action_condition(
        actions,
        committed_actions=None,
        chunk=chunk,
        no_action_slot=no_action,
    )

    assert tuple(reduced.condition.shape) == (1, 3, 20)
    assert reduced.latent_valid_mask.tolist() == [[True, True, True]]
    assert torch.equal(reduced.condition[0, 0], no_action)
    assert torch.equal(reduced.condition[0, 1], actions[0, 7])
    assert torch.equal(reduced.condition[0, 2], actions[0, 15])


@pytest.mark.parametrize(("valid_raw_count", "chunk_id"), [(33, 1), (57, 2)])
def test_cpu_reducer_partial_tail_is_fixed_k_with_explicit_zero_pad(
    valid_raw_count: int,
    chunk_id: int,
) -> None:
    chunk = _layout(valid_raw_count).chunks[chunk_id]
    actions = torch.arange(
        chunk.action_slot_capacity * 20,
        dtype=torch.float32,
    ).view(1, chunk.action_slot_capacity, 20)
    committed = torch.zeros(1, chunk.action_start, 20)

    reduced = reduce_end_of_bin_action_condition(
        actions,
        committed_actions=committed,
        chunk=chunk,
        no_action_slot=torch.ones(20),
    )

    assert tuple(reduced.condition.shape) == (1, 3, 20)
    assert reduced.latent_valid_mask.tolist() == [[True, True, False]]
    assert torch.equal(reduced.condition[0, 0], actions[0, 7])
    assert torch.equal(reduced.condition[0, 1], actions[0, 15])
    assert torch.count_nonzero(reduced.condition[0, 2]).item() == 0

    poisoned_pad = actions.clone()
    poisoned_pad[:, chunk.valid_action_count :, :] = 1.0e6
    second = reduce_end_of_bin_action_condition(
        poisoned_pad,
        committed_actions=committed,
        chunk=chunk,
        no_action_slot=torch.ones(20),
    )
    assert torch.equal(reduced.condition, second.condition)


def test_cpu_reducer_rejects_valid_only_tail_and_wrong_history_shape() -> None:
    chunk = _layout(33).chunks[1]
    valid_only = torch.zeros(1, chunk.valid_action_count, 20)
    with pytest.raises(ValueError, match="fixed action-slot capacity"):
        reduce_end_of_bin_action_condition(
            valid_only,
            committed_actions=torch.zeros(1, chunk.action_start, 20),
            chunk=chunk,
            no_action_slot=torch.zeros(20),
        )

    padded = torch.zeros(1, chunk.action_slot_capacity, 20)
    with pytest.raises(ValueError, match="shape must be exactly"):
        reduce_end_of_bin_action_condition(
            padded,
            committed_actions=torch.zeros(1, chunk.action_start - 1, 20),
            chunk=chunk,
            no_action_slot=torch.zeros(20),
        )
