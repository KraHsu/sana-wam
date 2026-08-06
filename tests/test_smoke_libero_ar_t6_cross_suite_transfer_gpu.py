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
RUNNER = ROOT / "scripts/smoke_libero_ar_t6_cross_suite_transfer_gpu.py"
T5_RUNNER = ROOT / "scripts/smoke_libero_ar_t5_cross_task_transfer_gpu.py"
DOC = ROOT / "docs/libero/LIBERO_T6_CROSS_SUITE_TRANSFER_20260806.md"

OBJECT_NAME = "libero_object_no_noops_1.0.0_lerobot"
GOAL_NAME = "libero_goal_no_noops_1.0.0_lerobot"
LIBERO_10_NAME = "libero_10_no_noops_1.0.0_lerobot"
SUITE_ORDER = (OBJECT_NAME, GOAL_NAME, LIBERO_10_NAME)

T5_RESULT_SHA256 = "a656aaef1528537527fe830ad7d4107138b29e8e254b5606b43c46a47e323e83"
T5_SOURCE_COMMIT = "5150693a0751200ef431968a863c69f8ca08a7df"
T5_RUNNER_SHA256 = "a4bbdcf752aa0a34a43ea4f51e7875f7fd160985280a7274b29e36350d9605c1"
T5_NONCE = "cf7dd8a1b1cef03511d2026a48e4a271"
ELIGIBLE_MANIFEST_SHA256 = (
    "c4ec46f7f1ede73556646da52501b3192c8d951b82d51ae8f9b18b70d15545f2"
)
SELECTION_MANIFEST_SHA256 = (
    "e96c234997bbe05edae7509fa6f6007d20a4f5b098e0d6e07dedeafc38e6a9c5"
)
CORE_SHA256 = "e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352"
FIXED_RECIPE_SHA256 = "ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad"
SELECTION_RULE = (
    "in frozen config suite order object,goal,10; within each suite rank tasks "
    "by task payload SHA256 and take first; within selected task rank episodes "
    "by episode payload SHA256 and take first"
)

EXPECTED_METADATA_SHA256 = {
    OBJECT_NAME: {
        "episodes_jsonl_sha256": (
            "63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea"
        ),
        "tasks_jsonl_sha256": (
            "68ef5f9bc5a0bd74f46140f6721fa0ea74e997d74e37b8714a539f61337e7862"
        ),
    },
    GOAL_NAME: {
        "episodes_jsonl_sha256": (
            "548d91fe48b7d439248523dd3f7a5e4b15fc77d5eb1b7cfdd6da0033d422cb43"
        ),
        "tasks_jsonl_sha256": (
            "39f08f81b289ad3041f1c8ada88f679fe60774e9fde4083415881486edc23d55"
        ),
    },
    LIBERO_10_NAME: {
        "episodes_jsonl_sha256": (
            "5589f8f87cfddb34812782462160bf55b0d3082e404240682d1d0a89faba8265"
        ),
        "tasks_jsonl_sha256": (
            "45f9eb4d4b6b04999f64640c0aae380555372b7b273a904f5f459ad05d4a0a6a"
        ),
    },
}

EXPECTED_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000082.parquet": (
                "f25ac74111c22c9db0ff96d6021484f8a0e31d2e4413d47a160b83a0489cd1e9"
            ),
            "videos/chunk-000/observation.images.image/episode_000082.mp4": (
                "38294e54072c35ff4fc7edf963b688bd90d463e4a1ba623aca680ce2106b6ee7"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000082.mp4": (
                "f935b70c9925f1a57bc4df7aa8c6519c614df2ece97661fa18d12d0a3788a923"
            ),
        },
        "dataset": OBJECT_NAME,
        "episode_index": 82,
        "episode_length": 135,
        "episode_selection_payload_sha256": (
            "08a3351e6a26cb3e1690663c2d875185468bbf44b7c60bbafa64573377b4a6b7"
        ),
        "label": "S1",
        "start_frame": 0,
        "task": "pick up the bbq sauce and place it in the basket",
        "task_index": 3,
        "task_selection_payload_sha256": (
            "61071697d2905f3282f0be449981512ea13639547e41402b56df8100942f8856"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000070.parquet": (
                "6ce3cf58f533871b6413ea0e795f302060aca3a87b041f0110c77494a6d4b66b"
            ),
            "videos/chunk-000/observation.images.image/episode_000070.mp4": (
                "af716b54f26f8cd284e715ab2335ecf86c59b5cc89706a7c51165094a5899b2b"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000070.mp4": (
                "ded7a8d03de99df3c720c3e74f87403378818a92f1cbea6e37a4d37d11fe1027"
            ),
        },
        "dataset": GOAL_NAME,
        "episode_index": 70,
        "episode_length": 249,
        "episode_selection_payload_sha256": (
            "02d397d9f8a9168f3032f66e83710595c9424a7323929ea264cb8d5ff502bad2"
        ),
        "label": "S2",
        "start_frame": 0,
        "task": "open the top drawer and put the bowl inside",
        "task_index": 2,
        "task_selection_payload_sha256": (
            "0d5d43dedb9b602ee435ab3654a66eac53429afb9ead5a95ddc87a3941040b00"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000259.parquet": (
                "de9e09a471d8050a5624f016f765c61fba7514829990b59de37c4e3bae0eec37"
            ),
            "videos/chunk-000/observation.images.image/episode_000259.mp4": (
                "c11a5751444371331df5b24ab76b85212e2979636d803afd2e0f59a002bd0804"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000259.mp4": (
                "8800f488e8cc6f71cdfdf66d6fc531f92c54f54edf55da1403c60cb064f91786"
            ),
        },
        "dataset": LIBERO_10_NAME,
        "episode_index": 259,
        "episode_length": 228,
        "episode_selection_payload_sha256": (
            "058ae9b8c4575e89dbc41c059d994ca69b3527b2c53477336da94d96ace76a3b"
        ),
        "label": "S3",
        "start_frame": 0,
        "task": "turn on the stove and put the moka pot on it",
        "task_index": 3,
        "task_selection_payload_sha256": (
            "2cd3fff8b786308553138be1a90b8219122b1e308b784e2652ddb811e1ef6f20"
        ),
    },
)

EXPECTED_SAMPLE_KEYS = tuple(
    (sample["dataset"], sample["episode_index"]) for sample in EXPECTED_SAMPLES
)
EXPECTED_IDENTITIES = tuple(
    (
        sample["dataset"],
        sample["task_index"],
        sample["episode_index"],
        sample["start_frame"],
    )
    for sample in EXPECTED_SAMPLES
)
EXPECTED_LABELS = tuple(sample["label"] for sample in EXPECTED_SAMPLES)

# Complete episode-index populations for each mechanically selected task.  These
# let the test independently prove the second selection stage without reading
# mutable live metadata.
SELECTED_TASK_EPISODES = {
    OBJECT_NAME: (
        4,
        5,
        17,
        36,
        39,
        42,
        46,
        51,
        60,
        64,
        82,
        86,
        89,
        125,
        136,
        143,
        144,
        147,
        161,
        166,
        173,
        181,
        198,
        201,
        215,
        218,
        227,
        260,
        270,
        282,
        318,
        322,
        323,
        328,
        329,
        337,
        374,
        386,
        392,
        419,
        423,
        424,
        428,
        429,
        444,
        451,
    ),
    GOAL_NAME: (
        3,
        4,
        8,
        30,
        35,
        37,
        70,
        83,
        86,
        110,
        115,
        119,
        144,
        147,
        155,
        159,
        169,
        180,
        181,
        197,
        215,
        235,
        245,
        248,
        305,
        315,
        322,
        332,
        358,
        359,
        363,
        379,
        383,
        403,
        405,
        411,
    ),
    LIBERO_10_NAME: (
        6,
        38,
        40,
        45,
        48,
        49,
        50,
        66,
        72,
        87,
        93,
        95,
        134,
        150,
        153,
        162,
        182,
        184,
        185,
        202,
        203,
        218,
        225,
        239,
        240,
        243,
        253,
        259,
        263,
        272,
        277,
        278,
        287,
        292,
        303,
        321,
        345,
        351,
        354,
        358,
        361,
    ),
}


def _source_and_tree(path: Path = RUNNER) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _literal_value(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(value, names) for value in node.elts)
    if isinstance(node, ast.List):
        return [_literal_value(value, names) for value in node.elts]
    if isinstance(node, ast.Set):
        return {_literal_value(value, names) for value in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _literal_value(key, names): _literal_value(value, names)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _literal_value(node.operand, names)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_value(node.left, names) + _literal_value(node.right, names)
    raise ValueError(f"not a static literal expression: {ast.dump(node)}")


def _top_level_literal(tree: ast.Module, name: str) -> Any:
    names: dict[str, Any] = {}
    for node in tree.body:
        target = None
        value = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
            value = node.value
        if target is None or value is None:
            continue
        try:
            names[target] = _literal_value(value, names)
        except (KeyError, TypeError, ValueError):
            continue
        if target == name:
            return names[target]
    raise AssertionError(f"runner lacks literal constant {name}")


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _function_source(source: str, tree: ast.Module, name: str) -> str:
    result = ast.get_source_segment(source, _function_node(tree, name))
    assert result is not None
    return result


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
    function = _function_node(tree, "_cross_suite_transfer_summary")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "T6_CROSS_SUITE_DATASETS": SUITE_ORDER,
        "T6_CROSS_SUITE_MEDIAN_RATIO": 0.95,
        "T6_CROSS_SUITE_MIN_IMPROVED": 2,
        "T6_CROSS_SUITE_IDENTITIES": EXPECTED_IDENTITIES,
        "T6_CROSS_SUITE_LABELS": EXPECTED_LABELS,
        "T6_CROSS_SUITE_SAMPLE_KEYS": EXPECTED_SAMPLE_KEYS,
        "T6_CROSS_SUITE_SAMPLES": EXPECTED_SAMPLES,
        "T6_CROSS_SUITE_SUITE_ORDER": SUITE_ORDER,
        "math": math,
        "statistics": statistics,
    }
    exec(compile(module, str(RUNNER), "exec"), namespace)  # noqa: S102
    return namespace["_cross_suite_transfer_summary"]


def _task_payload_sha256(dataset: str, task_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T6_TASK_SELECTION_V1\n"
        f"T5_RESULT_SHA256={T5_RESULT_SHA256}\n"
        f"ELIGIBLE_MANIFEST_SHA256={ELIGIBLE_MANIFEST_SHA256}\n"
        f"DATASET={dataset}\n"
        f"TASK_INDEX={task_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _episode_payload_sha256(dataset: str, task_index: int, episode_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T6_EPISODE_SELECTION_V1\n"
        f"T5_RESULT_SHA256={T5_RESULT_SHA256}\n"
        f"ELIGIBLE_MANIFEST_SHA256={ELIGIBLE_MANIFEST_SHA256}\n"
        f"DATASET={dataset}\n"
        f"TASK_INDEX={task_index}\n"
        f"EPISODE_INDEX={episode_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_t6_one_per_suite_two_stage_selection_and_manifests_are_frozen() -> None:
    source, tree = _source_and_tree()
    samples = _top_level_literal(tree, "T6_CROSS_SUITE_SAMPLES")
    assert samples == EXPECTED_SAMPLES
    assert [sample["dataset"] for sample in samples] == list(SUITE_ORDER)
    assert [sample["label"] for sample in samples] == ["S1", "S2", "S3"]
    assert len({sample["dataset"] for sample in samples}) == 3
    assert len({sample["label"] for sample in samples}) == 3
    assert _top_level_literal(tree, "T6_ELIGIBLE_MANIFEST_SHA256") == (
        ELIGIBLE_MANIFEST_SHA256
    )
    assert _top_level_literal(tree, "T6_SELECTION_MANIFEST_SHA256") == (
        SELECTION_MANIFEST_SHA256
    )

    for sample in samples:
        dataset = sample["dataset"]
        ranked_tasks = sorted(
            (_task_payload_sha256(dataset, index), index) for index in range(10)
        )
        assert ranked_tasks[0][1] == sample["task_index"]
        assert ranked_tasks[0][0] == sample["task_selection_payload_sha256"]
        ranked_episodes = sorted(
            (
                _episode_payload_sha256(dataset, sample["task_index"], episode_index),
                episode_index,
            )
            for episode_index in SELECTED_TASK_EPISODES[dataset]
        )
        assert ranked_episodes[0][1] == sample["episode_index"]
        assert ranked_episodes[0][0] == sample["episode_selection_payload_sha256"]

    selection_manifest = {
        "eligible_manifest_sha256": ELIGIBLE_MANIFEST_SHA256,
        "samples": list(samples),
        "schema_version": "sana-wam-libero-t6-cross-suite-selection-v1",
        "selection_rule": SELECTION_RULE,
        "suite_order": list(SUITE_ORDER),
        "t5_predecessor_result_sha256": T5_RESULT_SHA256,
    }
    payload = json.dumps(
        selection_manifest,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(payload) == 2923
    assert hashlib.sha256(payload).hexdigest() == SELECTION_MANIFEST_SHA256

    task_selector = _function_source(source, tree, "_task_selection_payload_sha256")
    episode_selector = _function_source(
        source, tree, "_episode_selection_payload_sha256"
    )
    selector = _function_source(source, tree, "_selection_manifest")
    for payload_source in (task_selector, episode_selector):
        assert "dataset" in payload_source
        assert "T5_RESULT_SHA256=" in payload_source
        assert "ELIGIBLE_MANIFEST_SHA256=" in payload_source
        assert "T4_RESULT_SHA256=" not in payload_source
    assert "ranked_tasks[0]" in selector
    assert "ranked_episodes[0]" in selector
    assert "ranked_tasks[:3]" not in selector
    assert '"suite_order"' in selector
    assert '"dataset"' in selector
    assert 'f"S{rank}"' in selector or "f'S{rank}'" in selector


def test_t6_candidate_population_and_goal_exclusion_are_dataset_qualified() -> None:
    source, tree = _source_and_tree()
    builder = _function_source(source, tree, "_build_eligible_sample_manifest")
    for dataset, pins in EXPECTED_METADATA_SHA256.items():
        assert dataset in source
        assert pins["tasks_jsonl_sha256"] in source
        assert pins["episodes_jsonl_sha256"] in source
    for fragment in (
        '"candidate_episode_count"',
        '"candidate_suite_count"',
        '"candidate_task_count"',
        '"excluded_samples"',
        '"schema_version"',
        '"start_frame"',
        '"suites"',
        '"t5_predecessor_result_sha256"',
    ):
        assert fragment in builder
    assert 'manifest["candidate_episode_count"] != 1260' in source
    assert 'manifest["candidate_suite_count"] != 3' in source
    assert 'manifest["candidate_task_count"] != 30' in source
    assert "sana-wam-libero-t6-cross-suite-eligible-v1" in builder
    assert "T5_PREDECESSOR_RESULT_SHA256" in builder

    compact_builder = "".join(builder.split())
    assert (
        "(dataset,episode_index)" in compact_builder
        and (
            "inT6_EXCLUDED" in compact_builder or "notinT6_EXCLUDED" in compact_builder
        )
    ) or "dataset==GOAL_NAMEandepisode_index==82" in compact_builder
    assert "\n            if episode_index == 82:" not in builder
    assert '"dataset":GOAL_NAME,"episode_index":82' in compact_builder or (
        f'"dataset":"{GOAL_NAME}","episode_index":82' in compact_builder
    )

    # Episode index 82 is excluded only for Goal.  Object:82 is the frozen S1.
    assert EXPECTED_SAMPLES[0]["dataset"] == OBJECT_NAME
    assert EXPECTED_SAMPLES[0]["episode_index"] == 82
    assert (GOAL_NAME, 82) not in EXPECTED_SAMPLE_KEYS
    assert (OBJECT_NAME, 82) in EXPECTED_SAMPLE_KEYS
    assert "candidate_episode_count" in builder
    assert "live_manifest != eligible_manifest" in source


def test_t6_dataset_qualified_keys_and_labels_prevent_cross_suite_collisions() -> None:
    source, tree = _source_and_tree()
    selector = _function_source(source, tree, "_select_t6_samples")
    compact = "".join(selector.split())
    assert 'row["dataset"]' in selector
    assert "episode.dataset" in selector
    assert "episode.episode_index" in selector
    assert "start == 0" in selector or (
        "episode.dataset" in selector
        and "episode.task_index" in selector
        and "episode.episode_index" in selector
        and "start" in selector
        and 'row["start_frame"]' in selector
    )
    assert "(episode.task_index,episode.episode_index)" not in compact
    assert '(int(row["task_index"]),int(row["episode_index"]))' not in compact
    assert "SPATIAL_NAME and start == 0" not in selector

    assert _top_level_literal(tree, "T6_CROSS_SUITE_IDENTITIES") == (
        EXPECTED_IDENTITIES
    )
    assert _top_level_literal(tree, "T6_CROSS_SUITE_LABELS") == EXPECTED_LABELS

    summary = _load_transfer_summary(tree)
    before = dict(zip(EXPECTED_LABELS, (10.0, 20.0, 40.0), strict=True))
    after = dict(zip(EXPECTED_LABELS, (9.5, 18.0, 50.0), strict=True))
    result = summary(before, after)
    assert result["heldout_sample_count"] == 3
    assert result["heldout_suite_count"] == 3
    assert result["suites_distinct"] is True
    assert [row["dataset"] for row in result["per_sample"]] == list(SUITE_ORDER)
    assert [row["label"] for row in result["per_sample"]] == ["S1", "S2", "S3"]

    wrong_goal_collision = dict(zip(EXPECTED_SAMPLE_KEYS, (10.0, 20.0, 40.0)))
    with pytest.raises(ValueError, match="frozen set"):
        summary(wrong_goal_collision, after)
    with pytest.raises(ValueError, match="frozen set"):
        summary({82: 10.0, 70: 20.0, 259: 40.0}, after)
    with pytest.raises(ValueError, match="frozen set"):
        summary({"S1": 10.0, "S2": 20.0, "wrong": 40.0}, after)


def test_t6_cross_suite_gate_uses_three_paired_samples() -> None:
    _source, tree = _source_and_tree()
    summary = _load_transfer_summary(tree)
    before = dict(zip(EXPECTED_LABELS, (10.0, 20.0, 40.0), strict=True))
    boundary_after = dict(zip(EXPECTED_LABELS, (9.5, 18.0, 50.0), strict=True))
    boundary = summary(before, boundary_after)
    assert boundary["median_post_to_pre_ratio"] == pytest.approx(0.95)
    assert boundary["improved_count"] == 2
    assert boundary["transfer_gate"] is True

    unstable_after = dict(zip(EXPECTED_LABELS, (9.6, 19.2, 20.0), strict=True))
    unstable = summary(before, unstable_after)
    assert unstable["improved_count"] == 3
    assert unstable["median_post_to_pre_ratio"] == pytest.approx(0.96)
    assert unstable["transfer_gate"] is False
    with pytest.raises(ValueError, match="positive pre-loss"):
        summary(before | {EXPECTED_LABELS[0]: 0.0}, boundary_after)
    with pytest.raises(ValueError, match="finite"):
        summary(
            before,
            boundary_after | {EXPECTED_LABELS[0]: float("nan")},
        )


def test_t6_update_loop_is_t5_equivalent_and_exactly_bounded() -> None:
    source, tree = _source_and_tree()
    _t5_source, t5_tree = _source_and_tree(T5_RUNNER)
    assert _top_level_literal(tree, "T6_UPDATE_STEPS") == 20

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
    t5_loop = _main_step_loop(t5_tree)
    assert ast.unparse(loop).replace("T6", "T5") == ast.unparse(t5_loop)
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    assert "forward_inputs=training_inputs" in loop_source
    assert "heldout_inputs" not in loop_source
    assert "cross_suite_inputs" not in loop_source
    assert "optimizer.step()" in loop_source
    assert "for step_index in range(T6_UPDATE_STEPS)" in source

    assert "architecture_forward_calls != T6_UPDATE_STEPS + 8" in source
    assert "backward_calls != T6_UPDATE_STEPS" in source
    assert "optimizer_step_calls != T6_UPDATE_STEPS" in source
    assert "prepare_inputs_calls != 4" in source
    assert "len(all_recipe_signatures) != T6_UPDATE_STEPS + 8" in source
    assert FIXED_RECIPE_SHA256 in source
    for report_fragment in (
        '"architecture_forwards_total": T6_UPDATE_STEPS + 8',
        '"architecture_measurement_forwards": 8',
        '"architecture_training_forwards": T6_UPDATE_STEPS',
        '"cross_suite_measurement_forwards": 6',
        '"cross_suite_samples_in_backward_or_update": 0',
        '"prepare_inputs_calls": prepare_inputs_calls',
    ):
        assert report_fragment in source


def test_t6_preparation_keeps_suite_prompts_and_contexts_independent() -> None:
    source, tree = _source_and_tree()
    calls = _call_names(tree)
    assert calls.count("architecture.prepare_inputs") == 1
    assert "torch.stack" not in calls
    assert "torch.cat" not in calls
    training_prepare = source.index('training_inputs = prepare_one("training"')
    suite_prepare = source.index(
        "for selected in heldout_selections:", training_prepare
    )
    assert training_prepare < suite_prepare
    assert "format_prompt_for_inference(T6_TRAIN_TASK_NAME)" in source
    assert 'format_prompt_for_inference(selected["task"])' in source
    assert "prepared tensors alias across samples" in source
    assert "prepared_context_report" in source
    assert "len(context_digests) != 4" in source
    assert '"context_sha256"' in source
    assert '"seq_lens"' in source
    assert '"dataset": selected["dataset"]' in source
    assert '"cross_suite_context": cross_suite_context' in source
    assert '"inputs": training_inputs_report' in source


def test_t6_pins_t5_and_reproduces_the_frozen_training_core() -> None:
    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T5_PREDECESSOR_RESULT_SHA256") == (
        T5_RESULT_SHA256
    )
    assert _top_level_literal(tree, "T5_PREDECESSOR_SOURCE_COMMIT") == (
        T5_SOURCE_COMMIT
    )
    assert _top_level_literal(tree, "T5_PREDECESSOR_RUNNER_SHA256") == (
        T5_RUNNER_SHA256
    )
    assert _top_level_literal(tree, "T6_TRAINING_CORE_SHA256") == CORE_SHA256
    assert T5_NONCE in source
    assert "t5_training_core" in source
    assert "current_training_core != t5_training_core" in source
    assert '"equal_to_frozen_t5": True' in source
    assert '"predecessor_result_sha256": T5_PREDECESSOR_RESULT_SHA256' in source
    for identity_key in (
        "t5_predecessor_result_sha256",
        "t5_predecessor_runner_sha256",
        "t5_predecessor_source_commit",
    ):
        assert identity_key in source

    t5_projection = _function_node(
        _source_and_tree(T5_RUNNER)[1], "_training_core_projection"
    )
    t6_projection = _function_node(tree, "_training_core_projection")
    assert ast.unparse(t6_projection).replace("T6", "T5") == ast.unparse(t5_projection)


def test_t6_result_contract_has_only_the_narrow_cross_suite_claim() -> None:
    source, _tree = _source_and_tree()
    for fragment in (
        '"cross_suite_transfer": cross_suite_transfer_report',
        '"schema_version": "sana-wam-libero-t6-cross-suite-transfer-v1"',
        '"schema_version": "sana-wam-libero-t6-cross-suite-failure-v1"',
        "T6_CROSS_SUITE_TRANSFER_GO",
        "T6_CROSS_SUITE_TRANSFER_INCONCLUSIVE",
        "/DATA/share/sana_wam_libero_nonformal_screens/t6",
        '"formal_training_executed": False',
        '"benchmark_evaluation_executed": False',
        '"simulator_executed": False',
        '"cross_suite_samples_in_backward_or_update": 0',
        '"normalization_population_includes_probe_samples": True',
        '"strict_dataset_holdout": False',
        '"task_text_only_isolation": False',
        '"samples_never_entered_backward_or_update": True',
    ):
        assert fragment in source
    for forbidden in (
        '"benchmark_success_claimed": True',
        '"formal_training_executed": True',
        '"simulator_executed": True',
        '"strict_dataset_holdout": True',
        '"cross_suite_transfer_claimed": False',
        "T6_HELDOUT_SAMPLE_TRANSFER",
        "same-suite cross-suite",
    ):
        assert forbidden not in source
    assert "this is not part of the frozen T6 primary transfer gate" in source


def test_t6_document_records_the_frozen_result() -> None:
    text = DOC.read_text(encoding="utf-8")
    for fragment in (
        "executed / frozen / T6_CROSS_SUITE_TRANSFER_GO",
        "708b1d8569866608898031f9116566d52fdeb742",
        "004f444168f26162f012c408193e119a8ff428a64f90e7dbc9ab595df9082903",
        "67a02fcc85508e03f136e09221a6a9d4",
        "4855f3771b80349547c985d137426cce79e25597f910eff88e136328424b8b89",
        "0.1576670424",
        "0.1767697439",
        "0.1526860825",
        "not rollout success",
    ):
        assert fragment in text
    assert "Result placeholder" not in text
