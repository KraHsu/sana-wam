from __future__ import annotations

import ast
import hashlib
import math
import statistics
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/smoke_libero_ar_t4_heldout_sample_transfer_gpu.py"

T3_RESULT_SHA256 = "c883608f2a47b6258f824d4d97a94f8a390d03bab671a592fb758eea61b3a01e"
SPATIAL_NAME = "libero_spatial_no_noops_1.0.0_lerobot"
TASK_INDEX = 0
START_FRAME = 0

# Episode 0 is the optimization sample.  These are the other 45 Spatial
# episodes carrying the same task, in the canonical eligible-manifest order.
ELIGIBLE_EPISODE_INDICES = (
    13,
    16,
    30,
    31,
    40,
    43,
    48,
    53,
    68,
    72,
    75,
    76,
    77,
    79,
    110,
    113,
    154,
    155,
    157,
    160,
    168,
    185,
    187,
    195,
    211,
    212,
    224,
    241,
    265,
    273,
    278,
    283,
    303,
    314,
    343,
    345,
    346,
    350,
    351,
    352,
    363,
    384,
    394,
    400,
    405,
)
EXPECTED_EPISODE_INDICES = (16, 405, 40)
EXPECTED_SELECTION_SHA256 = (
    "02938329ed205ce9032b9bc7c54d0e3f1f1716d86b8526bdc3e220bd784b6c6e",
    "09d59c3d6bec297af4f0d04d034d0f8ba65b2db2552f93cf8325946b1e6e3c72",
    "0d8d7c4a8f60985b3b0d2259bffae34b99c911bce2757b7559294e02e229fef8",
)
EXPECTED_ELIGIBLE_MANIFEST_SHA256 = (
    "e77462aa3380b1b14bee646b0965eb465baca76bff3d0805f9fc7d54d61419ef"
)
EXPECTED_SAMPLE_ASSET_SHA256 = {
    "data/chunk-000/episode_000016.parquet": (
        "6e440ab8884337435fae9b0868a0b3911fe8dca6be00ba1061f343e666d130db"
    ),
    "videos/chunk-000/observation.images.image/episode_000016.mp4": (
        "cef539255b3e7322e6b4e37ef4a3fc5c9d9946685460c3e5f7bde772f1e1e544"
    ),
    "videos/chunk-000/observation.images.wrist_image/episode_000016.mp4": (
        "9f6cf8979f6ab33a87084443218e86d18f27d29118d8880a338d4f4b8d60d6a1"
    ),
    "data/chunk-000/episode_000405.parquet": (
        "0887ba7d42fdfbe26bc2402055ab75073a3bc860408f3306dabc896f9d2a1611"
    ),
    "videos/chunk-000/observation.images.image/episode_000405.mp4": (
        "46be6fe35ecf534cb64a259ff33b1bc19756180628c6859cd1e9272d1e92daf6"
    ),
    "videos/chunk-000/observation.images.wrist_image/episode_000405.mp4": (
        "e6be28d3d32d836530643d92919340afcffd02747bcfb743511898cee7321f3e"
    ),
    "data/chunk-000/episode_000040.parquet": (
        "4b2f24a8818e18c489b041e4292553c0f76ca6f121de18bd695b0dd440806617"
    ),
    "videos/chunk-000/observation.images.image/episode_000040.mp4": (
        "9d4d5a9f2082020193d7576f2174314fd831169b8086de59d125b11f2ff70650"
    ),
    "videos/chunk-000/observation.images.wrist_image/episode_000040.mp4": (
        "b02619c14008af3dae2c39a8762d197cd19d3599d7a80d51553c5550713c7b9d"
    ),
}


def _source_and_tree() -> tuple[str, ast.Module]:
    source = RUNNER.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _top_level_literal(tree: ast.Module, *names: str) -> Any:
    for node in tree.body:
        target_name = None
        value = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target_name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        if target_name in names and value is not None:
            return ast.literal_eval(value)
    raise AssertionError(f"runner lacks a literal constant from {names!r}")


def _call_names(node: ast.AST) -> list[str]:
    result = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        function = candidate.func
        parts = []
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        if parts:
            result.append(".".join(reversed(parts)))
    return result


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected exactly one {name} function"
    return matches[0]


def _load_pure_function(tree: ast.Module, name: str):
    """Load one pure helper without importing the GPU runner module."""

    function = _function_node(tree, name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "T4_HELDOUT_EPISODE_INDICES": EXPECTED_EPISODE_INDICES,
        "T4_HELDOUT_MEDIAN_RATIO": 0.95,
        "T4_HELDOUT_MIN_IMPROVED": 2,
        "math": math,
        "statistics": statistics,
    }
    exec(compile(module, str(RUNNER), "exec"), namespace)  # noqa: S102
    return namespace[name]


def _selection_payload_sha256(episode_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T4_HELDOUT_SAMPLE_V1\n"
        f"T3_RESULT_SHA256={T3_RESULT_SHA256}\n"
        f"DATASET={SPATIAL_NAME}\n"
        f"TASK_INDEX={TASK_INDEX}\n"
        f"EPISODE_INDEX={episode_index}\n"
        f"START_FRAME={START_FRAME}\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _dict_value_sources(tree: ast.Module, key: str) -> list[str]:
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if isinstance(key_node, ast.Constant) and key_node.value == key:
                values.append(ast.unparse(value_node))
    return values


def test_t4_selection_is_domain_separated_and_manifest_pinned() -> None:
    source, tree = _source_and_tree()

    assert _top_level_literal(tree, "T4_HELDOUT_EPISODE_INDICES") == (
        EXPECTED_EPISODE_INDICES
    )
    selection_sha256 = _top_level_literal(
        tree,
        "T4_HELDOUT_SELECTION_PAYLOAD_SHA256",
        "T4_HELDOUT_SELECTION_SHA256",
    )
    if isinstance(selection_sha256, dict):
        selection_sha256 = tuple(
            selection_sha256[episode_index]
            for episode_index in EXPECTED_EPISODE_INDICES
        )
    assert selection_sha256 == EXPECTED_SELECTION_SHA256
    assert (
        _top_level_literal(
            tree,
            "T4_ELIGIBLE_MANIFEST_SHA256",
            "T4_ELIGIBLE_SAMPLE_MANIFEST_SHA256",
        )
        == EXPECTED_ELIGIBLE_MANIFEST_SHA256
    )
    sample_assets = _top_level_literal(tree, "T4_HELDOUT_SAMPLE_ASSET_SHA256")
    if sample_assets and all(
        isinstance(value, dict) for value in sample_assets.values()
    ):
        sample_assets = {
            path: digest
            for episode_assets in sample_assets.values()
            for path, digest in episode_assets.items()
        }
    assert sample_assets == EXPECTED_SAMPLE_ASSET_SHA256

    ranked = sorted(
        (_selection_payload_sha256(episode_index), episode_index)
        for episode_index in ELIGIBLE_EPISODE_INDICES
    )
    assert tuple(episode_index for _digest, episode_index in ranked[:3]) == (
        EXPECTED_EPISODE_INDICES
    )
    assert tuple(digest for digest, _episode_index in ranked[:3]) == (
        EXPECTED_SELECTION_SHA256
    )

    # These exact fields, including the final LF, are the frozen selection
    # domain.  The eligible manifest itself is canonical compact JSON with no LF.
    for fragment in (
        "SANA-WAM/LIBERO/T4_HELDOUT_SAMPLE_V1\\n",
        "T3_RESULT_SHA256=",
        "DATASET=",
        "TASK_INDEX=",
        "EPISODE_INDEX=",
        "START_FRAME=",
    ):
        assert fragment in source
    assert "T4_HELDOUT_RECIPE_SEEDS" not in source
    assert "_heldout_seed_manifest" not in source


def test_t4_heldout_sample_transfer_gate_uses_three_paired_episodes() -> None:
    _source, tree = _source_and_tree()
    summary = _load_pure_function(tree, "_heldout_sample_transfer_summary")

    before = {16: 10.0, 405: 20.0, 40: 40.0}
    boundary = summary(before, {16: 9.5, 405: 18.0, 40: 50.0})
    assert boundary["heldout_sample_count"] == 3
    assert boundary["median_post_to_pre_ratio"] == pytest.approx(0.95)
    assert boundary["improved_count"] == 2
    assert boundary["transfer_gate"] is True
    assert [row["episode_index"] for row in boundary["per_episode"]] == [
        16,
        405,
        40,
    ]
    assert all("seed" not in row for row in boundary["per_episode"])

    unstable = summary(before, {16: 9.6, 405: 19.2, 40: 20.0})
    assert unstable["improved_count"] == 3
    assert unstable["median_post_to_pre_ratio"] == pytest.approx(0.96)
    assert unstable["transfer_gate"] is False

    with pytest.raises(ValueError, match="frozen set"):
        summary({16: 10.0}, {16: 9.0})
    with pytest.raises(ValueError, match="positive pre-loss"):
        summary(
            {16: 0.0, 405: 20.0, 40: 40.0},
            {16: 0.0, 405: 18.0, 40: 36.0},
        )
    with pytest.raises(ValueError, match="finite"):
        summary(
            {16: 10.0, 405: 20.0, 40: 40.0},
            {16: float("nan"), 405: 18.0, 40: 36.0},
        )


def test_t4_runner_has_exact_28_20_8_4_schedule() -> None:
    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T4_UPDATE_STEPS") == 20

    calls = _call_names(tree)
    assert calls.count("optimizer.step") == 1
    assert "loss.backward" in calls
    assert "trainer._sync_master_gradients" in calls
    assert "trainer._copy_master_parameters_to_model" in calls
    assert "trainer.train" not in calls
    assert "torch.save" not in calls
    assert "torch.load" not in calls
    assert "architecture.save_checkpoint" not in calls
    assert "architecture.load_checkpoint" not in calls

    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "step_index"
    ]
    assert len(loops) == 1
    loop = loops[0]
    assert _call_names(loop).count("optimizer.step") == 1
    assert isinstance(loop.iter, ast.Call)
    assert isinstance(loop.iter.func, ast.Name)
    assert loop.iter.func.id == "range"
    assert len(loop.iter.args) == 1
    assert isinstance(loop.iter.args[0], ast.Name)
    assert loop.iter.args[0].id == "T4_UPDATE_STEPS"

    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    ordered_fragments = [
        "optimizer.zero_grad(set_to_none=True)",
        "parameter.grad = None",
        "fixed_forward(",
        "clip_grad_norm_",
        "trainer._sync_master_gradients(master_pairs)",
        "optimizer.step()",
        "trainer._copy_master_parameters_to_model(master_pairs)",
    ]
    offsets = [loop_source.index(fragment) for fragment in ordered_fragments]
    assert offsets == sorted(offsets)

    pre_offset = source.index("heldout_before[")
    initial_offset = source.index("initial_probe, initial_signature")
    loop_offset = source.index("for step_index in range(T4_UPDATE_STEPS)")
    final_offset = source.index("final_probe, final_signature")
    post_offset = source.index("heldout_after[")
    assert pre_offset < initial_offset < loop_offset < final_offset < post_offset

    assert "architecture_forward_calls != T4_UPDATE_STEPS + 8" in source
    assert "backward_calls != T4_UPDATE_STEPS" in source
    assert "optimizer_step_calls != T4_UPDATE_STEPS" in source
    assert "prepare_inputs_calls != 4" in source
    assert "heldout_sample_transfer" in source
    assert '"heldout_recipe_transfer":' not in source

    assert "T4_UPDATE_STEPS + 8" in _dict_value_sources(
        tree, "architecture_forwards_total"
    )
    assert "8" in _dict_value_sources(tree, "architecture_measurement_forwards")
    assert "T4_UPDATE_STEPS" in _dict_value_sources(
        tree, "architecture_training_forwards"
    )
    assert "backward_calls" in _dict_value_sources(tree, "backward_calls")
    assert "6" in _dict_value_sources(tree, "heldout_measurement_forwards")
    assert "2" in _dict_value_sources(tree, "main_recipe_measurement_forwards")
    assert "optimizer_step_calls" in _dict_value_sources(tree, "optimizer_steps")
    assert "prepare_inputs_calls" in _dict_value_sources(tree, "prepare_inputs_calls")

    assert "sana-wam-libero-t4-heldout-sample-transfer-v1" in source
    assert '"formal_training_executed": False' in source
    assert '"benchmark_evaluation_executed": False' in source
    assert '"simulator_executed": False' in source
    assert "/DATA/share/sana_wam_libero_nonformal_screens/t4" in source
    assert "args.run_root.expanduser() != expected_run_root" in source
    assert '"normalization_population_includes_probe_episodes": True' in source
    assert '"strict_dataset_holdout": False' in source
