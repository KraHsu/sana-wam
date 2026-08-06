from __future__ import annotations

import ast
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/smoke_libero_ar_t5_cross_task_transfer_gpu.py"
T4_RUNNER = ROOT / "scripts/smoke_libero_ar_t4_heldout_sample_transfer_gpu.py"
DOC = ROOT / "docs/libero/LIBERO_T5_CROSS_TASK_TRANSFER_20260806.md"

SPATIAL_NAME = "libero_spatial_no_noops_1.0.0_lerobot"
T4_RESULT_SHA256 = "4c33aaff6d202068d77efc0ac406c74198c56e72527cfabdde046fc9a3a4b6f4"
ELIGIBLE_MANIFEST_SHA256 = (
    "7b15d57b58ef398e7f0c3cd9371db03bd3312e4a4f3c3168f356df0f612bba45"
)
SELECTION_MANIFEST_SHA256 = (
    "e60eee38c8b4ba9f9e0ffe49eac3d102ced497981404ad3613ac30fa0f324886"
)
CORE_SHA256 = "e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352"
FIXED_RECIPE_SHA256 = "ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad"
T4_SOURCE_COMMIT = "7db8182ef46fce367f04f66bfbc34690fa86592b"
T4_RUNNER_SHA256 = "3d449761861b7fe75816d4d186b2e82402c160422eeb5f490fbb61ed274f088c"

EXPECTED_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000036.parquet": (
                "09d6f24419a6d70c766c42035c2cf993a59663070f1989a9ae54fa4cc83bc8f4"
            ),
            "videos/chunk-000/observation.images.image/episode_000036.mp4": (
                "a84cc4614ed6d7d08c9945567e7d3675e9ff7afa19113b9eaa5f68e3fe8af419"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000036.mp4": (
                "5aca422da74b6d059a8faac3c7e7ce1fa5e853c1c98486b106ba94c092a22114"
            ),
        },
        "episode_index": 36,
        "episode_length": 116,
        "episode_selection_payload_sha256": (
            "089747625a57d2d60e853d5d5c1c0c43ad6a01b477dbede629dda40933defcc9"
        ),
        "label": "H1",
        "start_frame": 0,
        "task": (
            "pick up the black bowl next to the ramekin and place it on the plate"
        ),
        "task_index": 7,
        "task_selection_payload_sha256": (
            "0a028e007220594e3d743dbd11db14944ca281ae37db8248ac7cf257cb742415"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000325.parquet": (
                "a7f51cc301d4e32a15717d6fea1384f45416d5fc5ab8e4183f7dee7c57b291d3"
            ),
            "videos/chunk-000/observation.images.image/episode_000325.mp4": (
                "edd51702dfcf53b60db9e12be38d18acad88678657e4cf063ec01da7517ac8d6"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000325.mp4": (
                "8511a776845f75aefeb1a2b904d00d668a6c5ddc25d03f7480474c6436ba2e58"
            ),
        },
        "episode_index": 325,
        "episode_length": 131,
        "episode_selection_payload_sha256": (
            "0a92dadc9f71161d6bc9e40476fbceed1d9ebc8bb297f111b402d17c8b9a50ca"
        ),
        "label": "H2",
        "start_frame": 0,
        "task": (
            "pick up the black bowl in the top drawer of the wooden cabinet and "
            "place it on the plate"
        ),
        "task_index": 1,
        "task_selection_payload_sha256": (
            "0ed2d254a2b9424e0c4a5dadc44532e66081c215edb6d651949c50c6e70d4432"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000011.parquet": (
                "5659083b099c426d2e9f4dde654fe3e556bde369c5c99f40eb0e521373b99c7f"
            ),
            "videos/chunk-000/observation.images.image/episode_000011.mp4": (
                "be57d261c1201b7258599f2b7f3b597e0464616daa8c92b8ffc4a5ef64c39a9a"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000011.mp4": (
                "a8ad057219920267213f88a801dc7aecf7f864c05b2beb327ea7820031167102"
            ),
        },
        "episode_index": 11,
        "episode_length": 84,
        "episode_selection_payload_sha256": (
            "02dbc353aaa077f46f0ee7cb4513417a6343c20146d2abfbf197279f35bbd13e"
        ),
        "label": "H3",
        "start_frame": 0,
        "task": (
            "pick up the black bowl between the plate and the ramekin and place it "
            "on the plate"
        ),
        "task_index": 4,
        "task_selection_payload_sha256": (
            "34e97146b609f1ff31d2b9cf44a2e86c3fc84205383535f7bdb0d2280b8b7ff2"
        ),
    },
)


def _source_and_tree(path: Path = RUNNER) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _top_level_literal(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"runner lacks literal constant {name}")


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


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


def _main_step_loop(tree: ast.Module) -> ast.For:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "step_index"
    ]
    assert len(matches) == 1
    return matches[0]


def _load_transfer_summary(tree: ast.Module):
    function = _function_node(tree, "_cross_task_transfer_summary")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "T5_CROSS_TASK_EPISODE_INDICES": (36, 325, 11),
        "T5_CROSS_TASK_MEDIAN_RATIO": 0.95,
        "T5_CROSS_TASK_MIN_IMPROVED": 2,
        "math": math,
        "statistics": statistics,
    }
    exec(compile(module, str(RUNNER), "exec"), namespace)  # noqa: S102
    return namespace["_cross_task_transfer_summary"]


def _task_payload_sha256(task_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T5_TASK_SELECTION_V1\n"
        f"T4_RESULT_SHA256={T4_RESULT_SHA256}\n"
        f"ELIGIBLE_MANIFEST_SHA256={ELIGIBLE_MANIFEST_SHA256}\n"
        f"DATASET={SPATIAL_NAME}\n"
        f"TASK_INDEX={task_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _episode_payload_sha256(task_index: int, episode_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T5_EPISODE_SELECTION_V1\n"
        f"T4_RESULT_SHA256={T4_RESULT_SHA256}\n"
        f"ELIGIBLE_MANIFEST_SHA256={ELIGIBLE_MANIFEST_SHA256}\n"
        f"DATASET={SPATIAL_NAME}\n"
        f"TASK_INDEX={task_index}\n"
        f"EPISODE_INDEX={episode_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_t5_two_stage_selection_and_manifests_are_frozen() -> None:
    source, tree = _source_and_tree()
    samples = _top_level_literal(tree, "T5_CROSS_TASK_SAMPLES")
    assert samples == EXPECTED_SAMPLES
    assert _top_level_literal(tree, "T5_CROSS_TASK_TASK_INDICES") == (7, 1, 4)
    assert _top_level_literal(tree, "T5_CROSS_TASK_EPISODE_INDICES") == (36, 325, 11)
    assert _top_level_literal(tree, "T5_ELIGIBLE_MANIFEST_SHA256") == (
        ELIGIBLE_MANIFEST_SHA256
    )
    assert _top_level_literal(tree, "T5_SELECTION_MANIFEST_SHA256") == (
        SELECTION_MANIFEST_SHA256
    )

    ranked_tasks = sorted(
        (_task_payload_sha256(index), index) for index in range(1, 10)
    )
    assert tuple(index for _digest, index in ranked_tasks[:3]) == (7, 1, 4)
    for sample in samples:
        assert (
            _task_payload_sha256(sample["task_index"])
            == (sample["task_selection_payload_sha256"])
        )
        assert (
            _episode_payload_sha256(sample["task_index"], sample["episode_index"])
            == sample["episode_selection_payload_sha256"]
        )

    selection_manifest = {
        "eligible_manifest_sha256": ELIGIBLE_MANIFEST_SHA256,
        "samples": list(samples),
        "schema_version": "sana-wam-libero-t5-cross-task-selection-v1",
        "selection_rule": (
            "rank tasks by task payload SHA256; take first 3; within each selected "
            "task rank episodes by episode payload SHA256 and take first"
        ),
        "t4_predecessor_result_sha256": T4_RESULT_SHA256,
    }
    payload = json.dumps(
        selection_manifest,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(payload) == 2697
    assert hashlib.sha256(payload).hexdigest() == SELECTION_MANIFEST_SHA256

    assert 'candidate_episode_count"] != 386' in source
    assert 'candidate_task_count"] != 9' in source
    assert "_build_eligible_sample_manifest" in source
    assert "live_manifest != eligible_manifest" in source
    assert "T4_RESULT_SHA256=" in source
    assert "ELIGIBLE_MANIFEST_SHA256=" in source
    for stale in ("episode_000016", "episode_000405", "episode_000040"):
        assert stale not in source


def test_t5_cross_task_gate_uses_three_paired_samples() -> None:
    _source, tree = _source_and_tree()
    summary = _load_transfer_summary(tree)
    before = {36: 10.0, 325: 20.0, 11: 40.0}
    boundary = summary(before, {36: 9.5, 325: 18.0, 11: 50.0})
    assert boundary["heldout_sample_count"] == 3
    assert boundary["heldout_task_count"] == 3
    assert boundary["tasks_distinct"] is True
    assert boundary["median_post_to_pre_ratio"] == pytest.approx(0.95)
    assert boundary["improved_count"] == 2
    assert boundary["transfer_gate"] is True
    assert [row["episode_index"] for row in boundary["per_sample"]] == [36, 325, 11]

    unstable = summary(before, {36: 9.6, 325: 19.2, 11: 20.0})
    assert unstable["improved_count"] == 3
    assert unstable["median_post_to_pre_ratio"] == pytest.approx(0.96)
    assert unstable["transfer_gate"] is False
    with pytest.raises(ValueError, match="frozen set"):
        summary({36: 10.0}, {36: 9.0})
    with pytest.raises(ValueError, match="positive pre-loss"):
        summary(before | {36: 0.0}, {36: 0.0, 325: 18.0, 11: 36.0})
    with pytest.raises(ValueError, match="finite"):
        summary(before, {36: float("nan"), 325: 18.0, 11: 36.0})


def test_t5_update_loop_is_t4_equivalent_and_exactly_bounded() -> None:
    source, tree = _source_and_tree()
    t4_source, t4_tree = _source_and_tree(T4_RUNNER)
    assert _top_level_literal(tree, "T5_UPDATE_STEPS") == 20

    calls = _call_names(tree)
    assert calls.count("optimizer.step") == 1
    assert "loss.backward" in calls
    assert "trainer._sync_master_gradients" in calls
    assert "trainer._copy_master_parameters_to_model" in calls
    for forbidden in (
        "trainer.train",
        "torch.save",
        "torch.load",
        "architecture.save_checkpoint",
        "architecture.load_checkpoint",
    ):
        assert forbidden not in calls

    loop = _main_step_loop(tree)
    t4_loop = _main_step_loop(t4_tree)
    assert ast.unparse(loop).replace("T5", "T4") == ast.unparse(t4_loop)
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    assert "forward_inputs=training_inputs" in loop_source
    assert "heldout_inputs" not in loop_source
    assert "optimizer.step()" in loop_source
    assert "for step_index in range(T5_UPDATE_STEPS)" in source

    pre_offset = source.index("heldout_before[")
    initial_offset = source.index("initial_probe, initial_signature")
    loop_offset = source.index("for step_index in range(T5_UPDATE_STEPS)")
    final_offset = source.index("final_probe, final_signature")
    post_offset = source.index("heldout_after[")
    assert pre_offset < initial_offset < loop_offset < final_offset < post_offset
    assert "architecture_forward_calls != T5_UPDATE_STEPS + 8" in source
    assert "backward_calls != T5_UPDATE_STEPS" in source
    assert "optimizer_step_calls != T5_UPDATE_STEPS" in source
    assert "prepare_inputs_calls != 4" in source
    assert "len(all_recipe_signatures) != T5_UPDATE_STEPS + 8" in source
    assert FIXED_RECIPE_SHA256 in source
    assert t4_source


def test_t5_preparation_keeps_prompts_and_contexts_independent() -> None:
    source, tree = _source_and_tree()
    calls = _call_names(tree)
    assert calls.count("architecture.prepare_inputs") == 1
    assert "torch.stack" not in calls
    assert "torch.cat" not in calls
    training_prepare = source.index('training_inputs = prepare_one("training"')
    cross_prepare = source.index(
        "for selected in heldout_selections:", training_prepare
    )
    assert training_prepare < cross_prepare
    assert "format_prompt_for_inference(T5_TRAIN_TASK_NAME)" in source
    assert 'format_prompt_for_inference(selected["task"])' in source
    assert "prepared tensors alias across samples" in source
    assert "prepared_context_report" in source
    assert "len(context_digests) != 4" in source
    assert '"context_sha256"' in source
    assert '"seq_lens"' in source
    assert '"cross_task_context": cross_task_context' in source
    assert '"inputs": training_inputs_report' in source


def test_t5_pins_t4_and_reports_only_cross_task_semantics() -> None:
    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T4_PREDECESSOR_RESULT_SHA256") == (
        T4_RESULT_SHA256
    )
    assert _top_level_literal(tree, "T4_PREDECESSOR_SOURCE_COMMIT") == (
        T4_SOURCE_COMMIT
    )
    assert _top_level_literal(tree, "T4_PREDECESSOR_RUNNER_SHA256") == (
        T4_RUNNER_SHA256
    )
    assert _top_level_literal(tree, "T5_TRAINING_CORE_SHA256") == CORE_SHA256
    assert "t3_training_core != t4_training_core" in source
    assert "current_training_core != t4_training_core" in source
    assert '"equal_to_frozen_t4": True' in source
    assert '"predecessor_result_sha256": T4_PREDECESSOR_RESULT_SHA256' in source
    assert '"cross_task_transfer": cross_task_transfer_report' in source
    assert '"schema_version": "sana-wam-libero-t5-cross-task-transfer-v1"' in source
    assert '"schema_version": "sana-wam-libero-t5-cross-task-failure-v1"' in source
    assert "T5_CROSS_TASK_TRANSFER_GO" in source
    assert "T5_CROSS_TASK_TRANSFER_INCONCLUSIVE" in source
    assert "T5_HELDOUT_SAMPLE_TRANSFER" not in source
    assert "/DATA/share/sana_wam_libero_nonformal_screens/t5" in source
    assert '"formal_training_executed": False' in source
    assert '"benchmark_evaluation_executed": False' in source
    assert '"simulator_executed": False' in source
    assert '"cross_suite_transfer_claimed": False' in source
    assert '"strict_dataset_holdout": False' in source

    t4_projection = _function_node(
        _source_and_tree(T4_RUNNER)[1], "_training_core_projection"
    )
    t5_projection = _function_node(tree, "_training_core_projection")
    assert ast.unparse(t5_projection).replace("T5", "T4") == ast.unparse(t4_projection)


def test_t5_document_records_the_frozen_result() -> None:
    text = DOC.read_text(encoding="utf-8")
    for fragment in (
        "executed / frozen GO",
        T4_RESULT_SHA256,
        ELIGIBLE_MANIFEST_SHA256,
        SELECTION_MANIFEST_SHA256,
        CORE_SHA256,
        "task 7 / episode 36",
        "task 1 / episode 325",
        "task 4 / episode 11",
        "28",
        "20",
        "T5_CROSS_TASK_TRANSFER_GO",
        "T5_CROSS_TASK_TRANSFER_INCONCLUSIVE",
        "5150693a0751200ef431968a863c69f8ca08a7df",
        "a656aaef1528537527fe830ad7d4107138b29e8e254b5606b43c46a47e323e83",
        "cf7dd8a1b1cef03511d2026a48e4a271",
    ):
        assert fragment in text
    assert "state_encoder" not in text
    assert all(
        root in text
        for root in (
            "action_backbone",
            "proprio_encoder",
            "proprio_video_embed",
            "proprio_action_embed",
        )
    )
