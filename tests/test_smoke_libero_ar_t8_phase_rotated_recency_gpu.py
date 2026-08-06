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
RUNNER = ROOT / "scripts/smoke_libero_ar_t8_phase_rotated_recency_gpu.py"

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

T7_RESULT_SHA256 = "9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a"
T7_SOURCE_COMMIT = "d19109a2314f8e7186571afed6b4acd816d7cfab"
T7_RUNNER_SHA256 = "b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b"
T7_RESULT_PATH = (
    "/DATA/share/sana_wam_libero_nonformal_screens/t7/d19109a2314f/"
    "libero-t7-foursuite-cyclic-fixed20-61e0a0ff817970e994b0875be4840ed6/"
    "RESULT.json"
)
CONFIG_RELATIVE_PATH = "configs/benchmarks/libero/train_libero_ar_baseline.yaml"
CONFIG_SHA256 = "5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45"

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

EXPECTED_STARTING_HELDOUT_ACTION_LOSSES = {
    "H0": 13.379679679870605,
    "H1": 14.698094367980957,
    "H2": 12.026171684265137,
    "H3": 20.885684967041016,
}

EXPECTED_PROMOTION_PROVENANCE = {
    "A1": {
        "episode_selection_payload_sha256": (
            "08a3351e6a26cb3e1690663c2d875185468bbf44b7c60bbafa64573377b4a6b7"
        ),
        "task_selection_payload_sha256": (
            "61071697d2905f3282f0be449981512ea13639547e41402b56df8100942f8856"
        ),
    },
    "A2": {
        "episode_selection_payload_sha256": (
            "02d397d9f8a9168f3032f66e83710595c9424a7323929ea264cb8d5ff502bad2"
        ),
        "task_selection_payload_sha256": (
            "0d5d43dedb9b602ee435ab3654a66eac53429afb9ead5a95ddc87a3941040b00"
        ),
    },
    "A3": {
        "episode_selection_payload_sha256": (
            "058ae9b8c4575e89dbc41c059d994ca69b3527b2c53477336da94d96ace76a3b"
        ),
        "task_selection_payload_sha256": (
            "2cd3fff8b786308553138be1a90b8219122b1e308b784e2652ddb811e1ef6f20"
        ),
    },
}

UPDATE_MANIFEST_SHA256 = (
    "5199c87af876c437ecec35b57da3e118a2924c44a9bb454c9846e7c0ca968aa6"
)
T7_TERMINAL_UPDATE_RATIOS = {
    "A0": 1.0947979413317903,
    "A1": 0.7655266031194603,
    "A2": 0.4848945538719168,
    "A3": 0.137526060026468,
}
T7_TERMINAL_HELDOUT_RATIOS = {
    "H0": 1.0763376119266599,
    "H1": 0.7577791145367958,
    "H2": 0.5153250642031886,
    "H3": 0.13820288765943442,
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


def _expected_update_manifest() -> dict[str, Any]:
    samples = []
    for frozen in EXPECTED_TRAIN_SAMPLES:
        sample = dict(frozen)
        sample.update(EXPECTED_PROMOTION_PROVENANCE.get(sample["label"], {}))
        samples.append(sample)
    return {
        "samples": samples,
        "schema_version": "sana-wam-libero-t7-four-suite-cyclic-update-samples-v1",
        "suite_order": list(SUITE_ORDER),
        "t6_predecessor_result_sha256": T6_RESULT_SHA256,
    }


def test_t8_reuses_the_exact_t7_eight_samples_and_three_manifests() -> None:
    eligible = _expected_eligible_manifest()
    eligible_payload = _canonical_bytes(eligible)
    assert len(eligible_payload) == ELIGIBLE_MANIFEST_BYTES
    assert hashlib.sha256(eligible_payload).hexdigest() == ELIGIBLE_MANIFEST_SHA256
    assert sum(len(rows) for rows in EXPECTED_TASK_EPISODES.values()) == 162

    selection = _expected_selection_manifest()
    selection_payload = _canonical_bytes(selection)
    assert len(selection_payload) == SELECTION_MANIFEST_BYTES
    assert hashlib.sha256(selection_payload).hexdigest() == SELECTION_MANIFEST_SHA256

    update = _expected_update_manifest()
    assert (
        hashlib.sha256(_canonical_bytes(update)).hexdigest() == UPDATE_MANIFEST_SHA256
    )

    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T8_CYCLIC_TRAIN_SAMPLES") == (
        EXPECTED_TRAIN_SAMPLES
    )
    assert _top_level_literal(tree, "T8_FRESH_HELDOUT_SAMPLES") == (
        EXPECTED_HELDOUT_SAMPLES
    )
    assert _top_level_literal(tree, "T8_EXCLUDED_SAMPLES") == (
        EXPECTED_EXCLUDED_SAMPLES
    )
    assert _top_level_literal(tree, "T8_SUITE_METADATA_SHA256") == (
        EXPECTED_METADATA_SHA256
    )
    assert _top_level_literal(tree, "T8_ELIGIBLE_MANIFEST_BYTES") == 8406
    assert _top_level_literal(tree, "T8_ELIGIBLE_MANIFEST_SHA256") == (
        ELIGIBLE_MANIFEST_SHA256
    )
    assert _top_level_literal(tree, "T8_SELECTION_MANIFEST_BYTES") == 3351
    assert _top_level_literal(tree, "T8_SELECTION_MANIFEST_SHA256") == (
        SELECTION_MANIFEST_SHA256
    )
    assert _top_level_literal(tree, "T8_UPDATE_SAMPLE_MANIFEST_SHA256") == (
        UPDATE_MANIFEST_SHA256
    )
    assert _top_level_literal(tree, "T6_PROMOTION_PROVENANCE") == (
        EXPECTED_PROMOTION_PROVENANCE
    )

    for update_sample, heldout_sample in zip(
        EXPECTED_TRAIN_SAMPLES, EXPECTED_HELDOUT_SAMPLES, strict=True
    ):
        assert (
            update_sample["dataset"],
            update_sample["task_index"],
            update_sample["task"],
        ) == (
            heldout_sample["dataset"],
            heldout_sample["task_index"],
            heldout_sample["task"],
        )
        assert update_sample["episode_index"] != heldout_sample["episode_index"]
        ranked = sorted(
            (
                _episode_payload_sha256(
                    heldout_sample["dataset"],
                    heldout_sample["task_index"],
                    episode_index,
                ),
                episode_index,
                length,
            )
            for episode_index, length in EXPECTED_TASK_EPISODES[
                heldout_sample["dataset"]
            ]
        )
        assert ranked[0] == (
            heldout_sample["episode_selection_payload_sha256"],
            heldout_sample["episode_index"],
            heldout_sample["episode_length"],
        )

    for literal in (
        ELIGIBLE_SCHEMA,
        SELECTION_SCHEMA,
        "SANA-WAM/LIBERO/T7_EPISODE_SELECTION_V1",
    ):
        assert literal in source
    assert "SANA-WAM/LIBERO/T8_EPISODE_SELECTION_V1" not in source


def test_t8_directly_pins_t7_result_source_runner_and_exact_config() -> None:
    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T7_PREDECESSOR_RESULT_SHA256") == (
        T7_RESULT_SHA256
    )
    assert _top_level_literal(tree, "T7_PREDECESSOR_SOURCE_COMMIT") == (
        T7_SOURCE_COMMIT
    )
    assert _top_level_literal(tree, "T7_PREDECESSOR_RUNNER_SHA256") == (
        T7_RUNNER_SHA256
    )
    assert _top_level_path_argument(tree, "T7_PREDECESSOR_RESULT_PATH") == (
        T7_RESULT_PATH
    )
    assert _top_level_literal(tree, "T8_CONFIG_RELATIVE_PATH") == (CONFIG_RELATIVE_PATH)
    assert _top_level_literal(tree, "T8_CONFIG_SHA256") == CONFIG_SHA256
    assert _top_level_literal(tree, "T8_STARTING_ACTION_LOSSES") == (
        EXPECTED_STARTING_ACTION_LOSSES
    )
    assert _top_level_literal(tree, "T8_STARTING_HELDOUT_ACTION_LOSSES") == (
        EXPECTED_STARTING_HELDOUT_ACTION_LOSSES
    )

    for fragment in (
        "T7_FOUR_SUITE_CYCLIC_GO",
        '"t7_predecessor_result_sha256"',
        '"t7_predecessor_runner_sha256"',
        '"t7_predecessor_source_commit"',
        "actual_config_sha256 != T8_CONFIG_SHA256",
        "expected_config_path = (ROOT / T8_CONFIG_RELATIVE_PATH).resolve()",
    ):
        assert fragment in source
    assert "T8 config path differs" in source


def test_t8_schedule_is_phase_rotated_and_phase_probes_are_post_update() -> None:
    source, tree = _source_and_tree()
    assert _top_level_literal(tree, "T8_UPDATE_STEPS") == 20
    assert _top_level_literal(tree, "T8_CYCLE_COUNT") == 5
    assert _top_level_literal(tree, "T8_UPDATES_PER_SAMPLE") == 5
    assert _top_level_literal(tree, "T8_PHASE_ROTATED_LABELS") == (
        "A1",
        "A2",
        "A3",
        "A0",
    )
    assert _top_level_literal(tree, "T8_PHASE_PROBE_STEPS") == {
        17: ("A1", "H1"),
        18: ("A2", "H2"),
        19: ("A3", "H3"),
        20: ("A0", "H0"),
    }
    assert _top_level_literal(tree, "T8_TRAILING_UPDATES") == {
        "A0": 0,
        "A1": 3,
        "A2": 2,
        "A3": 1,
    }

    cyclic_label_for_step = _load_function(
        tree,
        "_cyclic_label_for_step",
        {
            "T8_PHASE_ROTATED_LABELS": ("A1", "A2", "A3", "A0"),
            "T8_UPDATE_STEPS": 20,
        },
    )
    assert (
        tuple(cyclic_label_for_step(index) for index in range(20))
        == (
            "A1",
            "A2",
            "A3",
            "A0",
        )
        * 5
    )
    with pytest.raises(ValueError):
        cyclic_label_for_step(-1)
    with pytest.raises(ValueError):
        cyclic_label_for_step(20)

    loop = _main_step_loop(tree)
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    assert "_cyclic_label_for_step(step_index)" in loop_source
    assert "T8_PHASE_PROBE_STEPS.get(step_index + 1)" in loop_source
    assert loop_source.index("optimizer.step()") < loop_source.index(
        "T8_PHASE_PROBE_STEPS.get(step_index + 1)"
    )
    assert "fresh_heldout_inputs[update_label]" not in loop_source
    assert "phase_probe_state = probe_mutation_snapshot()" in loop_source
    assert "probe_mutation_snapshot() != phase_probe_state" in loop_source

    for fragment in (
        "architecture_forward_calls != 44",
        "backward_calls != 20",
        "optimizer_step_calls != 20",
        "prepare_inputs_calls != 8",
        "len(all_recipe_signatures) != 44",
        '"architecture_forwards_total": 44',
        '"architecture_measurement_forwards": 24',
        '"architecture_training_forwards": 20',
        '"phase_measurement_forwards": 8',
        '"fresh_heldout_samples_in_backward_or_update": 0',
    ):
        assert fragment in source
    assert FIXED_RECIPE_SHA256 in source


def test_t8_recency_summary_has_typed_boundary_outcomes() -> None:
    _source, tree = _source_and_tree()
    namespace = _load_functions(
        tree,
        (
            "_rankdata",
            "_spearman_rho",
            "_at_least_threshold",
            "_terminal_recency_summary",
        ),
        {
            "Any": Any,
            "T7_TERMINAL_HELDOUT_RATIOS": T7_TERMINAL_HELDOUT_RATIOS,
            "T7_TERMINAL_UPDATE_RATIOS": T7_TERMINAL_UPDATE_RATIOS,
            "T8_HELDOUT_LABELS": ("H0", "H1", "H2", "H3"),
            "T8_MIN_OVERWRITE_COUNT": 2,
            "T8_MIN_OVERWRITE_PENALTY": 1.02,
            "T8_MIN_SPEARMAN_RHO": 0.8,
            "T8_SUITE_ORDER": SUITE_ORDER,
            "T8_TRAILING_UPDATES": {"A0": 0, "A1": 3, "A2": 2, "A3": 1},
            "T8_TRAIN_LABELS": ("A0", "A1", "A2", "A3"),
            "math": math,
            "statistics": statistics,
        },
    )
    summary = namespace["_terminal_recency_summary"]
    exact_boundary_rho = namespace["_spearman_rho"](
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 3.0, 2.0, 4.0],
    )
    assert exact_boundary_rho == pytest.approx(0.8)
    assert namespace["_at_least_threshold"](exact_boundary_rho, 0.8) is True
    before = {label: 10.0 for label in ("A0", "A1", "A2", "A3", "H0", "H1", "H2", "H3")}

    # A0/H0 are rescued below their starting losses.  A1-A3 are best directly
    # after their own fifth update, then worsen monotonically with 3/2/1
    # trailing updates.  Exactly-two-at-threshold is a passing boundary.
    terminal_losses = {
        "A0": 8.0,
        "H0": 8.0,
        "A1": 9.5,
        "H1": 9.5,
        "A2": 9.0,
        "H2": 9.0,
        "A3": 8.5,
        "H3": 8.5,
    }
    own_fifth = {
        "A0": 8.0,
        "H0": 8.0,
        "A1": 9.5 / 1.02,
        "H1": 9.5 / 1.02,
        "A2": 9.0 / 1.02,
        "H2": 9.0 / 1.02,
        "A3": 8.5 / 1.019,
        "H3": 8.5 / 1.019,
    }
    recency = summary(before, own_fifth, terminal_losses, joint_viable=True)
    assert recency["spatial_terminal_rescued"] is True
    assert recency["per_suite"][0]["q_phase"] == pytest.approx(0.8)
    assert recency["per_suite"][0]["q_terminal"] == pytest.approx(0.8)
    assert recency["nonterminal_overwrite_count"] == 2
    assert recency["rho_recency"] == pytest.approx(1.0)
    assert recency["typed_verdict"] == "T8_TERMINAL_RECENCY_SUPPORTED"
    assert recency["joint_viable_diagnostic"] is True

    only_one_penalty = dict(own_fifth)
    for label in ("A2", "H2", "A3", "H3"):
        only_one_penalty[label] = terminal_losses[label] / 1.019
    mixed_penalty = summary(
        before, only_one_penalty, terminal_losses, joint_viable=False
    )
    assert mixed_penalty["nonterminal_overwrite_count"] == 1
    assert mixed_penalty["typed_verdict"] == "T8_PHASE_ROTATED_MIXED_INCONCLUSIVE"

    low_rho_terminal = {
        "A0": 8.0,
        "H0": 8.0,
        "A1": 6.0,
        "H1": 6.0,
        "A2": 9.0,
        "H2": 9.0,
        "A3": 8.5,
        "H3": 8.5,
    }
    low_rho_phase = {label: value / 1.1 for label, value in low_rho_terminal.items()}
    low_rho = summary(before, low_rho_phase, low_rho_terminal, joint_viable=True)
    assert low_rho["rho_recency"] < 0.8
    assert low_rho["typed_verdict"] == "T8_PHASE_ROTATED_MIXED_INCONCLUSIVE"

    suite_terminal = {
        **{label: 10.0 * ratio for label, ratio in T7_TERMINAL_UPDATE_RATIOS.items()},
        **{label: 10.0 * ratio for label, ratio in T7_TERMINAL_HELDOUT_RATIOS.items()},
    }
    suite_effect = summary(before, suite_terminal, suite_terminal, joint_viable=False)
    assert suite_effect["rho_suite"] == pytest.approx(1.0)
    spatial = suite_effect["per_suite"][0]
    assert spatial["terminal_update_ratio"] >= 1.0
    assert spatial["terminal_heldout_ratio"] >= 1.0
    assert suite_effect["typed_verdict"] == "T8_SUITE_EFFECT_SUPPORTED"

    with pytest.raises(ValueError, match="labels"):
        summary(before | {"wrong": 1.0}, own_fifth, terminal_losses, joint_viable=True)
    with pytest.raises(ValueError, match="finite"):
        summary(
            before,
            own_fifth | {"A1": float("nan")},
            terminal_losses,
            joint_viable=True,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        summary(before, own_fifth, terminal_losses | {"H3": -0.1}, joint_viable=True)


def test_t8_precision_and_global_state_guards_cover_every_step() -> None:
    source, tree = _source_and_tree()
    loop_source = ast.get_source_segment(source, _main_step_loop(tree))
    assert loop_source is not None

    # A fresh FP32 master must be the exact FP32 image of its BF16 model tensor,
    # before AdamW has any opportunity to hide a construction bug.
    assert "initial_master_projection_mismatches" in source
    assert "model.detach().float()" in source
    assert "T8 FP32 master identity/dtype differs" in source

    # Buffers are globally immutable, including across the twenty update
    # forwards (the local pre/post probe snapshots alone are insufficient).
    assert "initial_architecture_buffer_versions" in source
    assert "final_architecture_buffer_versions" in source
    assert "T8 architecture buffers changed during the update loop" in source

    # Projection equality is checked after every optimizer step, not merely at
    # steps 1 and 20, and the complete check-set is reported.
    assert "projection_exact_steps.append(step_index + 1)" in loop_source
    assert "if step_index in {0, T8_UPDATE_STEPS - 1}" not in loop_source
    assert '"projection_exact_after_steps": projection_exact_steps' in source

    assert "bool((loss < 0).item())" in source
    assert "bool((loss_action < 0).item())" in source
    assert "absolute_error = abs(observed_loss - expected_loss)" in source
    assert "allowed_error = 1.0e-6 + 1.0e-6 * abs(expected_loss)" in source
    assert '"components": starting_state_components' in source
    assert '"absolute_error": absolute_error' in source
    assert '"allowed_error": allowed_error' in source
    assert '"passed": passed' in source
    assert "--expected-config-sha256" not in source
    assert "spatial_material_shift" not in source


def test_t8_scope_is_nonformal_checkpoint_free_and_typed_only() -> None:
    source, tree = _source_and_tree()
    calls = _call_names(tree)
    assert calls.count("optimizer.step") == 1
    assert "loss.backward" in calls
    assert "trainer._sync_master_gradients" in calls
    assert "trainer._copy_master_parameters_to_model" in calls
    for forbidden_call in (
        "trainer.train",
        "torch.save",
        "torch.load",
        "architecture.save_checkpoint",
        "architecture.load_checkpoint",
    ):
        assert forbidden_call not in calls

    for fragment in (
        '"schema_version": "sana-wam-libero-t8-phase-rotated-recency-v1"',
        '"schema_version": "sana-wam-libero-t8-phase-rotated-recency-failure-v1"',
        "T8_TERMINAL_RECENCY_SUPPORTED",
        "T8_SUITE_EFFECT_SUPPORTED",
        "T8_PHASE_ROTATED_MIXED_INCONCLUSIVE",
        "/DATA/share/sana_wam_libero_nonformal_screens/t8",
        '"formal_training_executed": False',
        '"benchmark_evaluation_executed": False',
        '"simulator_executed": False',
        '"sana_wam_training_checkpoint_loaded": False',
        '"sana_wam_training_checkpoint_saved": False',
        '"rollout_transfer_claimed": False',
        '"benchmark_success_claimed": False',
    ):
        assert fragment in source
    for forbidden in (
        '"formal_training_executed": True',
        '"simulator_executed": True',
        '"rollout_transfer_claimed": True',
        '"benchmark_success_claimed": True',
        "T8_UPDATE_STEPS = 40",
        "T8_CYCLE_COUNT = 10",
    ):
        assert forbidden not in source
