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
RUNNER = ROOT / "scripts/smoke_libero_ar_t7_four_suite_cyclic_gpu.py"

SPATIAL_NAME = "libero_spatial_no_noops_1.0.0_lerobot"
OBJECT_NAME = "libero_object_no_noops_1.0.0_lerobot"
GOAL_NAME = "libero_goal_no_noops_1.0.0_lerobot"
LIBERO_10_NAME = "libero_10_no_noops_1.0.0_lerobot"
SUITE_ORDER = (SPATIAL_NAME, OBJECT_NAME, GOAL_NAME, LIBERO_10_NAME)

T6_RESULT_SHA256 = "4855f3771b80349547c985d137426cce79e25597f910eff88e136328424b8b89"
T6_SOURCE_COMMIT = "708b1d8569866608898031f9116566d52fdeb742"
T6_RUNNER_SHA256 = "004f444168f26162f012c408193e119a8ff428a64f90e7dbc9ab595df9082903"
T6_LINEAGE_CORE_SHA256 = (
    "e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352"
)
FIXED_RECIPE_SHA256 = "ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad"

ELIGIBLE_SCHEMA = "sana-wam-libero-t7-four-suite-cyclic-same-task-probe-eligible-v1"
ELIGIBLE_MANIFEST_BYTES = 8406
ELIGIBLE_MANIFEST_SHA256 = (
    "0b6bfe4c51e8b9e2ac3f538bac7853b5e1e871ea96dda5df6976ee27b5b25370"
)
SELECTION_SCHEMA = "sana-wam-libero-t7-four-suite-cyclic-same-task-probe-selection-v1"
SELECTION_RULE = (
    "in frozen config suite order spatial,object,goal,10; within each fixed "
    "update task rank eligible episodes by episode payload SHA256 and take first"
)
SELECTION_MANIFEST_BYTES = 3351
SELECTION_MANIFEST_SHA256 = (
    "e969e76e4bcb8a3c2b0b3fffff35e0ec10c478e45cb6535b7bd91061d3a6a7ee"
)

EXPECTED_METADATA_SHA256 = {
    SPATIAL_NAME: {
        "episodes.jsonl": (
            "690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7"
        ),
        "tasks.jsonl": (
            "399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1"
        ),
    },
    OBJECT_NAME: {
        "episodes.jsonl": (
            "63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea"
        ),
        "tasks.jsonl": (
            "68ef5f9bc5a0bd74f46140f6721fa0ea74e997d74e37b8714a539f61337e7862"
        ),
    },
    GOAL_NAME: {
        "episodes.jsonl": (
            "548d91fe48b7d439248523dd3f7a5e4b15fc77d5eb1b7cfdd6da0033d422cb43"
        ),
        "tasks.jsonl": (
            "39f08f81b289ad3041f1c8ada88f679fe60774e9fde4083415881486edc23d55"
        ),
    },
    LIBERO_10_NAME: {
        "episodes.jsonl": (
            "5589f8f87cfddb34812782462160bf55b0d3082e404240682d1d0a89faba8265"
        ),
        "tasks.jsonl": (
            "45f9eb4d4b6b04999f64640c0aae380555372b7b273a904f5f459ad05d4a0a6a"
        ),
    },
}

EXPECTED_TRAIN_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000000.parquet": (
                "3f875604fad478765549128759edfb33a64b69b7b82decebc9e4f38155f20a8c"
            ),
            "videos/chunk-000/observation.images.image/episode_000000.mp4": (
                "eba9a9b36611f7e1061f65233b3b0c53eaa3f9c347dfd073f7470a821695fdf7"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000000.mp4": (
                "cf4f97b9405e6e06d42f3816c902a7cec22ba0497dc4ea3fa201729302ba4168"
            ),
        },
        "dataset": SPATIAL_NAME,
        "episode_index": 0,
        "episode_length": 110,
        "label": "A0",
        "start_frame": 0,
        "task": (
            "pick up the black bowl next to the cookie box and place it on the plate"
        ),
        "task_index": 0,
    },
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
        "label": "A1",
        "start_frame": 0,
        "task": "pick up the bbq sauce and place it in the basket",
        "task_index": 3,
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
        "label": "A2",
        "start_frame": 0,
        "task": "open the top drawer and put the bowl inside",
        "task_index": 2,
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
        "label": "A3",
        "start_frame": 0,
        "task": "turn on the stove and put the moka pot on it",
        "task_index": 3,
    },
)

EXPECTED_STARTING_ACTION_LOSSES = {
    "A0": 13.679718971252441,
    "A1": 14.696584701538086,
    "A2": 12.06648063659668,
    "A3": 20.43115234375,
}

EXPECTED_HELDOUT_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000030.parquet": (
                "fa8fd7f972b1567af1e8cd290af80649dbf64dc4f14dacb2b12e80ebba62489d"
            ),
            "videos/chunk-000/observation.images.image/episode_000030.mp4": (
                "c93bb2ac9d565fff5e6939eebebf0ee34a73f6fa037510f8c27877943b9a09cc"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000030.mp4": (
                "2ae055381b99944dee3ef4d4e1b07d396e8723fa5616e5544d3a3dad8e64658c"
            ),
        },
        "dataset": SPATIAL_NAME,
        "episode_index": 30,
        "episode_length": 125,
        "episode_selection_payload_sha256": (
            "0a33e0f8df2afd9f256cba15af973cc4b5f7dde3ddbc0b5866f454a535e4e83d"
        ),
        "label": "H0",
        "start_frame": 0,
        "task": (
            "pick up the black bowl next to the cookie box and place it on the plate"
        ),
        "task_index": 0,
    },
    {
        "assets": {
            "data/chunk-000/episode_000166.parquet": (
                "ce386f7027088fb4d9648d7b8acaecc7fac8ed3150dceb5882659985a2d47b3a"
            ),
            "videos/chunk-000/observation.images.image/episode_000166.mp4": (
                "645069ae5c04c1cf2ce8ab3891fecafbb9704db0e5bbee5f507117e595f23a4d"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000166.mp4": (
                "c48ca15a905b46e76ab7b7afa15e26611bcad182fd94e74b5382a532a1e1f6ed"
            ),
        },
        "dataset": OBJECT_NAME,
        "episode_index": 166,
        "episode_length": 129,
        "episode_selection_payload_sha256": (
            "0378af218d388de30af08e0be5c417dfb5a7275939dca80da5b1a15d9c841df1"
        ),
        "label": "H1",
        "start_frame": 0,
        "task": "pick up the bbq sauce and place it in the basket",
        "task_index": 3,
    },
    {
        "assets": {
            "data/chunk-000/episode_000248.parquet": (
                "2511639d496676d6c670c7143bfa331e185a4e41b407c8ccd4b8e4d32713a880"
            ),
            "videos/chunk-000/observation.images.image/episode_000248.mp4": (
                "5f1803eb0dfe3a0e7f2b92cd8cfb77c6febfdf798812eb87fe9ca31405a2edad"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000248.mp4": (
                "855bfd92540b233e1352a540f6838295a1bb441aa81ebaa99e66313fb94775ba"
            ),
        },
        "dataset": GOAL_NAME,
        "episode_index": 248,
        "episode_length": 184,
        "episode_selection_payload_sha256": (
            "02c53f50cf758bc8185dbedbdfc5b7c05d297389b5fd55e04d738e0f65856a94"
        ),
        "label": "H2",
        "start_frame": 0,
        "task": "open the top drawer and put the bowl inside",
        "task_index": 2,
    },
    {
        "assets": {
            "data/chunk-000/episode_000278.parquet": (
                "369834fda56e67653a5273b6c07f46c70b87fdbc7f4361bf2a9e94e68bf05440"
            ),
            "videos/chunk-000/observation.images.image/episode_000278.mp4": (
                "3ecd041eccfe14e5ef87abb4533b671ac1f3577d52a9474dfbda65dba058926b"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000278.mp4": (
                "b55954d2eeb821a4d737fb4b57a809a4cf55f31bb9d757845af85e07b9052fe5"
            ),
        },
        "dataset": LIBERO_10_NAME,
        "episode_index": 278,
        "episode_length": 261,
        "episode_selection_payload_sha256": (
            "125e02f94d4e09f2216fbf649869d6bd3e1f0c43044992d031abc4dd8454b974"
        ),
        "label": "H3",
        "start_frame": 0,
        "task": "turn on the stove and put the moka pot on it",
        "task_index": 3,
    },
)

EXPECTED_EXCLUDED_SAMPLES = (
    {"dataset": SPATIAL_NAME, "task_index": 0, "episode_index": 0},
    {"dataset": SPATIAL_NAME, "task_index": 0, "episode_index": 16},
    {"dataset": SPATIAL_NAME, "task_index": 0, "episode_index": 405},
    {"dataset": SPATIAL_NAME, "task_index": 0, "episode_index": 40},
    {"dataset": SPATIAL_NAME, "task_index": 7, "episode_index": 36},
    {"dataset": SPATIAL_NAME, "task_index": 1, "episode_index": 325},
    {"dataset": SPATIAL_NAME, "task_index": 4, "episode_index": 11},
    {"dataset": OBJECT_NAME, "task_index": 3, "episode_index": 82},
    {"dataset": GOAL_NAME, "task_index": 2, "episode_index": 70},
    {"dataset": LIBERO_10_NAME, "task_index": 3, "episode_index": 259},
)

# Complete eligible populations inside the four fixed update tasks.  The test
# rebuilds both manifests without reading mutable live metadata.
EXPECTED_TASK_EPISODES = {
    SPATIAL_NAME: (
        (13, 113),
        (30, 125),
        (31, 157),
        (43, 113),
        (48, 115),
        (53, 117),
        (68, 124),
        (72, 141),
        (75, 135),
        (76, 133),
        (77, 125),
        (79, 122),
        (110, 122),
        (113, 124),
        (154, 133),
        (155, 133),
        (157, 152),
        (160, 118),
        (168, 123),
        (185, 106),
        (187, 119),
        (195, 129),
        (211, 122),
        (212, 124),
        (224, 123),
        (241, 146),
        (265, 130),
        (273, 127),
        (278, 122),
        (283, 121),
        (303, 114),
        (314, 133),
        (343, 120),
        (345, 113),
        (346, 137),
        (350, 123),
        (351, 124),
        (352, 108),
        (363, 108),
        (384, 155),
        (394, 124),
        (400, 130),
    ),
    OBJECT_NAME: (
        (4, 132),
        (5, 132),
        (17, 161),
        (36, 135),
        (39, 133),
        (42, 160),
        (46, 125),
        (51, 132),
        (60, 128),
        (64, 129),
        (86, 126),
        (89, 166),
        (125, 127),
        (136, 131),
        (143, 161),
        (144, 122),
        (147, 134),
        (161, 172),
        (166, 129),
        (173, 132),
        (181, 150),
        (198, 162),
        (201, 140),
        (215, 175),
        (218, 149),
        (227, 132),
        (260, 162),
        (270, 135),
        (282, 136),
        (318, 170),
        (322, 140),
        (323, 166),
        (328, 174),
        (329, 129),
        (337, 134),
        (374, 221),
        (386, 166),
        (392, 139),
        (419, 177),
        (423, 131),
        (424, 141),
        (428, 168),
        (429, 135),
        (444, 141),
        (451, 148),
    ),
    GOAL_NAME: (
        (3, 177),
        (4, 218),
        (8, 270),
        (30, 185),
        (35, 202),
        (37, 199),
        (83, 199),
        (86, 227),
        (110, 195),
        (115, 199),
        (119, 190),
        (144, 202),
        (147, 178),
        (155, 188),
        (159, 221),
        (169, 186),
        (180, 182),
        (181, 184),
        (197, 170),
        (215, 178),
        (235, 176),
        (245, 183),
        (248, 184),
        (305, 184),
        (315, 173),
        (322, 198),
        (332, 251),
        (358, 194),
        (359, 219),
        (363, 187),
        (379, 209),
        (383, 202),
        (403, 215),
        (405, 196),
        (411, 187),
    ),
    LIBERO_10_NAME: (
        (6, 279),
        (38, 261),
        (40, 228),
        (45, 267),
        (48, 278),
        (49, 308),
        (50, 290),
        (66, 248),
        (72, 271),
        (87, 263),
        (93, 235),
        (95, 242),
        (134, 262),
        (150, 298),
        (153, 255),
        (162, 272),
        (182, 285),
        (184, 239),
        (185, 269),
        (202, 232),
        (203, 243),
        (218, 245),
        (225, 257),
        (239, 323),
        (240, 289),
        (243, 299),
        (253, 275),
        (263, 307),
        (272, 272),
        (277, 238),
        (278, 261),
        (287, 218),
        (292, 312),
        (303, 244),
        (321, 240),
        (345, 295),
        (351, 265),
        (354, 222),
        (358, 250),
        (361, 301),
    ),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _episode_payload_sha256(dataset: str, task_index: int, episode_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T7_EPISODE_SELECTION_V1\n"
        f"T6_RESULT_SHA256={T6_RESULT_SHA256}\n"
        f"ELIGIBLE_MANIFEST_SHA256={ELIGIBLE_MANIFEST_SHA256}\n"
        f"DATASET={dataset}\n"
        f"TASK_INDEX={task_index}\n"
        f"EPISODE_INDEX={episode_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _expected_eligible_manifest() -> dict[str, Any]:
    train_by_dataset = {row["dataset"]: row for row in EXPECTED_TRAIN_SAMPLES}
    suites = []
    for dataset in SUITE_ORDER:
        training = train_by_dataset[dataset]
        episodes = [
            {"episode_index": episode_index, "length": length}
            for episode_index, length in EXPECTED_TASK_EPISODES[dataset]
        ]
        task = {
            "episode_count": len(episodes),
            "episodes": episodes,
            "task": training["task"],
            "task_index": training["task_index"],
        }
        suites.append(
            {
                "dataset": dataset,
                "episode_count": len(episodes),
                "episodes_jsonl_sha256": EXPECTED_METADATA_SHA256[dataset][
                    "episodes.jsonl"
                ],
                "task_count": 1,
                "tasks": [task],
                "tasks_jsonl_sha256": EXPECTED_METADATA_SHA256[dataset]["tasks.jsonl"],
                "update_task_index": training["task_index"],
            }
        )
    return {
        "candidate_episode_count": 162,
        "candidate_suite_count": 4,
        "candidate_task_count": 4,
        "excluded_samples": list(EXPECTED_EXCLUDED_SAMPLES),
        "schema_version": ELIGIBLE_SCHEMA,
        "start_frame": 0,
        "suites": suites,
        "t6_predecessor_result_sha256": T6_RESULT_SHA256,
    }


def _expected_selection_manifest() -> dict[str, Any]:
    return {
        "eligible_manifest_sha256": ELIGIBLE_MANIFEST_SHA256,
        "samples": list(EXPECTED_HELDOUT_SAMPLES),
        "schema_version": SELECTION_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "suite_order": list(SUITE_ORDER),
        "t6_predecessor_result_sha256": T6_RESULT_SHA256,
    }


def _source_and_tree() -> tuple[str, ast.Module]:
    source = RUNNER.read_text(encoding="utf-8")
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


def _top_level_path_argument(tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "Path"
            and len(node.value.args) == 1
        ):
            return ast.literal_eval(node.value.args[0])
    raise AssertionError(f"runner lacks Path constant {name}")


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"runner must define exactly one {name}"
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
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    ]
    assert len(matches) == 1
    return matches[0]


def _load_function(tree: ast.Module, name: str, namespace: dict[str, Any]):
    function = _function_node(tree, name)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(RUNNER), "exec"), namespace)  # noqa: S102
    return namespace[name]


def _load_functions(
    tree: ast.Module, names: tuple[str, ...], namespace: dict[str, Any]
) -> dict[str, Any]:
    module = ast.Module(
        body=[_function_node(tree, name) for name in names], type_ignores=[]
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(RUNNER), "exec"), namespace)  # noqa: S102
    return namespace


def test_t7_frozen_manifests_rebuild_to_exact_bytes_and_sha() -> None:
    eligible = _expected_eligible_manifest()
    eligible_payload = _canonical_bytes(eligible)
    assert len(eligible_payload) == ELIGIBLE_MANIFEST_BYTES
    assert hashlib.sha256(eligible_payload).hexdigest() == ELIGIBLE_MANIFEST_SHA256
    assert sum(len(rows) for rows in EXPECTED_TASK_EPISODES.values()) == 162

    selected = _expected_selection_manifest()
    selected_payload = _canonical_bytes(selected)
    assert len(selected_payload) == SELECTION_MANIFEST_BYTES
    assert hashlib.sha256(selected_payload).hexdigest() == SELECTION_MANIFEST_SHA256

    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T7_ELIGIBLE_MANIFEST_SHA256") == (
        ELIGIBLE_MANIFEST_SHA256
    )
    assert _top_level_literal(tree, "T7_SELECTION_MANIFEST_SHA256") == (
        SELECTION_MANIFEST_SHA256
    )
    assert _top_level_literal(tree, "T7_ELIGIBLE_MANIFEST_BYTES") == 8406
    assert _top_level_literal(tree, "T7_SELECTION_MANIFEST_BYTES") == 3351
    assert _top_level_literal(tree, "T7_SUITE_METADATA_SHA256") == (
        EXPECTED_METADATA_SHA256
    )
    assert _top_level_literal(tree, "T7_EXCLUDED_SAMPLES") == (
        EXPECTED_EXCLUDED_SAMPLES
    )

    builder = _function_source(source, tree, "_build_eligible_probe_manifest")
    compact_builder = "".join(builder.split())
    for fragment in (
        '"candidate_episode_count"',
        '"candidate_suite_count"',
        '"candidate_task_count"',
        '"excluded_samples"',
        '"update_task_index"',
        '"t6_predecessor_result_sha256"',
        ELIGIBLE_SCHEMA,
    ):
        assert fragment in builder
    assert "T7_EXCLUDED_SAMPLES" in builder
    assert "T7_ELIGIBLE_MANIFEST_BYTES" in builder
    assert "T7_ELIGIBLE_MANIFEST_SHA256" in builder
    assert 'manifest["candidate_episode_count"]!=162' in compact_builder
    assert 'manifest["candidate_suite_count"]!=4' in compact_builder
    assert 'manifest["candidate_task_count"]!=4' in compact_builder
    assert (
        "dataset" in builder and "task_index" in builder and "episode_index" in builder
    )
    assert "live_manifest != eligible_manifest" in source


def test_t7_training_and_fresh_heldout_samples_are_exact_and_same_task() -> None:
    source, tree = _source_and_tree()
    train = _top_level_literal(tree, "T7_CYCLIC_TRAIN_SAMPLES")
    heldout = _top_level_literal(tree, "T7_FRESH_HELDOUT_SAMPLES")
    assert train == EXPECTED_TRAIN_SAMPLES
    assert heldout == EXPECTED_HELDOUT_SAMPLES
    assert _top_level_literal(tree, "T7_TRAIN_LABELS") == ("A0", "A1", "A2", "A3")
    assert _top_level_literal(tree, "T7_HELDOUT_LABELS") == (
        "H0",
        "H1",
        "H2",
        "H3",
    )
    assert [row["dataset"] for row in train] == list(SUITE_ORDER)
    assert [row["dataset"] for row in heldout] == list(SUITE_ORDER)

    for update, probe in zip(train, heldout, strict=True):
        assert (probe["dataset"], probe["task_index"], probe["task"]) == (
            update["dataset"],
            update["task_index"],
            update["task"],
        )
        assert probe["episode_index"] != update["episode_index"]
        population = EXPECTED_TASK_EPISODES[probe["dataset"]]
        ranked = sorted(
            (
                _episode_payload_sha256(
                    probe["dataset"], probe["task_index"], episode_index
                ),
                episode_index,
                length,
            )
            for episode_index, length in population
        )
        assert ranked[0] == (
            probe["episode_selection_payload_sha256"],
            probe["episode_index"],
            probe["episode_length"],
        )

    selector = _function_source(source, tree, "_episode_selection_payload_sha256")
    for fragment in (
        "SANA-WAM/LIBERO/T7_EPISODE_SELECTION_V1",
        "T6_RESULT_SHA256=",
        "ELIGIBLE_MANIFEST_SHA256=",
        "DATASET=",
        "TASK_INDEX=",
        "EPISODE_INDEX=",
        "START_FRAME=0",
    ):
        assert fragment in selector
    assert "T5_RESULT_SHA256=" not in selector
    assert "TASK_SELECTION" not in source

    manifest_source = _function_source(source, tree, "_selection_manifest")
    assert "ranked_episodes[0]" in manifest_source
    assert "ranked_tasks" not in manifest_source
    assert _top_level_literal(tree, "T7_SELECTION_RULE") == SELECTION_RULE
    assert "T7_SELECTION_RULE" in manifest_source
    assert SELECTION_SCHEMA in manifest_source
    assert "T7_SELECTION_MANIFEST_BYTES" in manifest_source
    assert "T7_SELECTION_MANIFEST_SHA256" in manifest_source


def test_t7_full_identity_selection_prevents_cross_suite_collisions() -> None:
    source, tree = _source_and_tree()
    selector = _function_source(source, tree, "_select_t7_samples")
    compact = "".join(selector.split())
    for fragment in (
        'row["dataset"]',
        'row["task_index"]',
        'row["episode_index"]',
        'row["start_frame"]',
        "episode.dataset",
        "episode.task_index",
        "episode.episode_index",
    ):
        assert fragment in selector
    assert "(episode.dataset,episode.task_index,episode.episode_index,start)" in compact
    assert "(episode.task_index,episode.episode_index)" not in compact
    assert "tuple[str,int,int,int]" in compact

    train_identities = {
        (row["dataset"], row["task_index"], row["episode_index"], row["start_frame"])
        for row in EXPECTED_TRAIN_SAMPLES
    }
    heldout_identities = {
        (row["dataset"], row["task_index"], row["episode_index"], row["start_frame"])
        for row in EXPECTED_HELDOUT_SAMPLES
    }
    assert len(train_identities) == len(heldout_identities) == 4
    assert train_identities.isdisjoint(heldout_identities)
    assert "selection_by_episode" not in source


def test_t7_paired_gate_is_independent_for_train_and_fresh_heldout() -> None:
    source, tree = _source_and_tree()
    summary = _load_function(
        tree,
        "_paired_four_sample_summary",
        {
            "Any": Any,
            "T7_MIN_IMPROVED": 3,
            "T7_MEDIAN_RATIO": 0.95,
            "math": math,
            "statistics": statistics,
        },
    )
    before = {
        label: value
        for label, value in zip(
            ("A0", "A1", "A2", "A3"), (10.0, 20.0, 40.0, 80.0), strict=True
        )
    }
    boundary_after = {"A0": 9.5, "A1": 18.0, "A2": 38.0, "A3": 100.0}
    boundary = summary(before, boundary_after, samples=EXPECTED_TRAIN_SAMPLES)
    assert boundary["sample_count"] == 4
    assert boundary["suite_count"] == 4
    assert boundary["task_count"] == 4
    assert boundary["improved_count"] == 3
    assert boundary["median_post_to_pre_ratio"] == pytest.approx(0.95)
    assert boundary["paired_gate"] is True

    high_median = summary(
        before,
        {"A0": 9.6, "A1": 19.2, "A2": 38.4, "A3": 96.0},
        samples=EXPECTED_TRAIN_SAMPLES,
    )
    assert high_median["improved_count"] == 3
    assert high_median["median_post_to_pre_ratio"] == pytest.approx(0.96)
    assert high_median["paired_gate"] is False
    only_two = summary(
        before,
        {"A0": 9.0, "A1": 18.0, "A2": 40.0, "A3": 80.0},
        samples=EXPECTED_TRAIN_SAMPLES,
    )
    assert only_two["improved_count"] == 2
    assert only_two["paired_gate"] is False

    heldout_before = {
        row["label"]: float(index + 1)
        for index, row in enumerate(EXPECTED_HELDOUT_SAMPLES)
    }
    heldout_after = {label: value * 0.5 for label, value in heldout_before.items()}
    fresh = summary(
        heldout_before,
        heldout_after,
        samples=EXPECTED_HELDOUT_SAMPLES,
    )
    assert fresh["paired_gate"] is True
    assert [row["label"] for row in fresh["per_sample"]] == ["H0", "H1", "H2", "H3"]

    with pytest.raises(ValueError, match="frozen labels"):
        summary(before | {"wrong": 1.0}, boundary_after, samples=EXPECTED_TRAIN_SAMPLES)
    with pytest.raises(ValueError, match="positive pre-loss"):
        summary(before | {"A0": 0.0}, boundary_after, samples=EXPECTED_TRAIN_SAMPLES)
    with pytest.raises(ValueError, match="finite"):
        summary(
            before,
            boundary_after | {"A0": float("nan")},
            samples=EXPECTED_TRAIN_SAMPLES,
        )

    compact_source = "".join(source.split())
    assert "cyclic_training_summary=_paired_four_sample_summary(" in compact_source
    assert "fresh_heldout_summary=_paired_four_sample_summary(" in compact_source
    assert "T7_MIN_IMPROVED=3" in compact_source
    assert "T7_MEDIAN_RATIO=0.95" in compact_source


def test_t7_final_gate_requires_three_corresponding_suites() -> None:
    _source, tree = _source_and_tree()
    namespace = _load_functions(
        tree,
        ("_paired_four_sample_summary", "_four_suite_cyclic_summary"),
        {
            "Any": Any,
            "T7_CYCLIC_TRAIN_SAMPLES": EXPECTED_TRAIN_SAMPLES,
            "T7_FRESH_HELDOUT_SAMPLES": EXPECTED_HELDOUT_SAMPLES,
            "T7_MIN_IMPROVED": 3,
            "T7_MEDIAN_RATIO": 0.95,
            "T7_SUITE_ORDER": SUITE_ORDER,
            "math": math,
            "statistics": statistics,
        },
    )
    summary = namespace["_four_suite_cyclic_summary"]
    train_before = {f"A{index}": 10.0 for index in range(4)}
    heldout_before = {f"H{index}": 10.0 for index in range(4)}

    disjoint_edge = summary(
        train_before,
        {"A0": 5.0, "A1": 5.0, "A2": 5.0, "A3": 10.0},
        heldout_before,
        {"H0": 10.0, "H1": 5.0, "H2": 5.0, "H3": 5.0},
    )
    assert disjoint_edge["update_samples"]["paired_gate"] is True
    assert disjoint_edge["fresh_same_task_update_heldout"]["paired_gate"] is True
    assert disjoint_edge["paired_suite_improved_count"] == 2
    assert disjoint_edge["gate"] is False

    corresponding_three = summary(
        train_before,
        {"A0": 10.0, "A1": 5.0, "A2": 5.0, "A3": 5.0},
        heldout_before,
        {"H0": 10.0, "H1": 5.0, "H2": 5.0, "H3": 5.0},
    )
    assert corresponding_three["paired_suite_improved_count"] == 3
    assert corresponding_three["gate"] is True


def test_t7_cycle_is_a0_through_a3_five_times_and_exactly_bounded() -> None:
    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T7_UPDATE_STEPS") == 20
    assert _top_level_literal(tree, "T7_CYCLE_COUNT") == 5
    assert _top_level_literal(tree, "T7_UPDATES_PER_SAMPLE") == 5
    assert _top_level_literal(tree, "T7_STARTING_ACTION_LOSSES") == (
        EXPECTED_STARTING_ACTION_LOSSES
    )

    cyclic_label_for_step = _load_function(
        tree,
        "_cyclic_label_for_step",
        {"T7_TRAIN_LABELS": ("A0", "A1", "A2", "A3"), "T7_UPDATE_STEPS": 20},
    )
    expected_schedule = ("A0", "A1", "A2", "A3") * 5
    assert (
        tuple(cyclic_label_for_step(index) for index in range(20)) == expected_schedule
    )
    with pytest.raises(ValueError):
        cyclic_label_for_step(-1)
    with pytest.raises(ValueError):
        cyclic_label_for_step(20)

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
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    assert "_cyclic_label_for_step(step_index)" in loop_source
    assert "cyclic_training_inputs" in loop_source
    assert "fresh_heldout_inputs" not in loop_source
    assert "optimizer.step()" in loop_source
    assert "for step_index in range(T7_UPDATE_STEPS)" in loop_source

    for fragment in (
        "architecture_forward_calls != 36",
        "backward_calls != 20",
        "optimizer_step_calls != 20",
        "prepare_inputs_calls != 8",
        "len(all_recipe_signatures) != 36",
        '"architecture_forwards_total": 36',
        '"architecture_measurement_forwards": 16',
        '"architecture_training_forwards": 20',
        '"cyclic_training_measurement_forwards": 8',
        '"fresh_heldout_measurement_forwards": 8',
        '"fresh_heldout_samples_in_backward_or_update": 0',
        '"cyclic_training_updates_per_sample": 5',
    ):
        assert fragment in source
    assert FIXED_RECIPE_SHA256 in source


def test_t7_prepares_eight_independent_samples_and_probes_without_mutation() -> None:
    source, tree = _source_and_tree()
    calls = _call_names(tree)
    assert calls.count("architecture.prepare_inputs") == 1
    assert "torch.stack" not in calls
    assert "torch.cat" not in calls
    for fragment in (
        "cyclic_training_inputs",
        "fresh_heldout_inputs",
        "prepared tensors alias across samples",
        "prepared_input_snapshot",
        "probe_mutation_snapshot",
        "probe_state_unchanged",
        "rng_state_restored_after_every_forward",
        "optimizer state exists before pre probes",
        "fresh held-out pre probes mutated training state",
        "fresh held-out post probes mutated training state",
    ):
        assert fragment in source

    # A/H pairs intentionally share task text and therefore may share context
    # values.  The runner must not mistake equal text context for tensor aliasing.
    assert "len(context_digests) != 8" not in source
    assert '"same_task_probe": True' in source
    assert '"prepared_tensor_aliasing": False' in source
    assert '"prepared_inputs_immutable": True' in source
    assert '"fresh_heldout_never_entered_backward_or_update": True' in source
    assert "post_probe_reprepare_calls" in source


def test_t7_pins_t6_and_treats_the_old_core_as_lineage_only() -> None:
    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T6_PREDECESSOR_RESULT_SHA256") == (
        T6_RESULT_SHA256
    )
    assert _top_level_literal(tree, "T6_PREDECESSOR_SOURCE_COMMIT") == (
        T6_SOURCE_COMMIT
    )
    assert _top_level_literal(tree, "T6_PREDECESSOR_RUNNER_SHA256") == (
        T6_RUNNER_SHA256
    )
    assert _top_level_literal(tree, "T6_LINEAGE_TRAINING_CORE_SHA256") == (
        T6_LINEAGE_CORE_SHA256
    )
    for fragment in (
        "T6_CROSS_SUITE_TRANSFER_GO",
        '"t6_predecessor_result_sha256"',
        '"t6_predecessor_runner_sha256"',
        '"t6_predecessor_source_commit"',
        '"lineage_only": True',
        '"exact_current_core_reproduction_expected": False',
        '"starting_action_losses_equal_frozen_t6": True',
    ):
        assert fragment in source
    assert _top_level_path_argument(tree, "T6_PREDECESSOR_RESULT_PATH") == (
        "/DATA/share/sana_wam_libero_nonformal_screens/t6/708b1d856986/"
        "libero-t6-crosssuite3-fixed20-67a02fcc85508e03f136e09221a6a9d4/"
        "RESULT.json"
    )

    # T7 changes the optimization inputs and per-step values.  Requiring the
    # current report to equal T5/T6's single-Spatial core is a category error.
    for forbidden in (
        "current_training_core != t6_training_core",
        "current_training_core != t5_training_core",
        '"equal_to_frozen_t6": True',
        '"equal_to_frozen_t5": True',
        "T7_TRAINING_CORE_SHA256",
    ):
        assert forbidden not in source
    assert "T7_STARTING_ACTION_LOSSES" in source
    assert "starting action loss differs from frozen T6" in source


def test_t7_result_schema_keeps_train_and_fresh_gates_separate_and_narrow() -> None:
    source, _tree = _source_and_tree()
    for fragment in (
        '"schema_version": "sana-wam-libero-t7-four-suite-cyclic-v1"',
        '"schema_version": "sana-wam-libero-t7-four-suite-cyclic-failure-v1"',
        '"cyclic_training": cyclic_training_report',
        '"fresh_heldout_transfer": fresh_heldout_report',
        '"cyclic_schedule": cyclic_schedule_report',
        '"sample_roles": sample_roles_report',
        '"cyclic_training_gate"',
        '"fresh_heldout_gate"',
        "T7_FOUR_SUITE_CYCLIC_GO",
        "T7_FOUR_SUITE_CYCLIC_INCONCLUSIVE",
        "/DATA/share/sana_wam_libero_nonformal_screens/t7",
        '"formal_training_executed": False',
        '"benchmark_evaluation_executed": False',
        '"simulator_executed": False',
        '"sana_wam_training_checkpoint_loaded": False',
        '"sana_wam_training_checkpoint_saved": False',
        '"normalization_population_includes_probe_samples": True',
        '"strict_dataset_holdout": False',
        '"rollout_transfer_claimed": False',
        '"benchmark_success_claimed": False',
        '"same_task_probe_only": True',
    ):
        assert fragment in source

    compact = "".join(source.split())
    assert 'ifnotcyclic_training_summary["paired_gate"]' in compact
    assert 'ifnotfresh_heldout_summary["paired_gate"]' in compact
    for forbidden in (
        '"benchmark_success_claimed": True',
        '"formal_training_executed": True',
        '"simulator_executed": True',
        '"strict_dataset_holdout": True',
        '"rollout_transfer_claimed": True',
        "T7_UPDATE_STEPS = 40",
        "T7_CYCLE_COUNT = 10",
    ):
        assert forbidden not in source
