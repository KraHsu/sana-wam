#!/usr/bin/env python
"""Run the bounded LIBERO T11 same-task/new-episode balanced-JOINT screen."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

from smoke_libero_ar_real_gpu import (  # noqa: E402
    DEFAULT_CONFIG,
    EXPECTED_BASE_PARTIAL_LOAD_PREFIX,
    EXPECTED_BASE_PARTIAL_LOAD_SAMPLES,
    _assert_idle_gpu,
    _query_gpu,
    _repo_commit,
    _sha256_file,
)
from smoke_libero_ar_real_update_gpu import (  # noqa: E402
    EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES,
    EXPECTED_SANA_COMMIT,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TRAINABLE_ROOTS,
    EXPECTED_TRAINABLE_TENSOR_COUNT,
    _assert_source_tree_clean,
    _capture_update_probes,
    _create_run_root,
    _freeze_run_root,
    _optimizer_state_is_finite,
    _parameter_root,
    _sha256_json,
    _summarize_updates,
    _tensor_scalar,
    _verify_external_assets,
    _verify_pinned_asset,
)


T11_MACRO_STEPS = 20
T11_UPDATE_STEPS = T11_MACRO_STEPS
T11_FORWARD_EXPOSURES_PER_SAMPLE = T11_MACRO_STEPS
T11_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO = 0.25
T11_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE = 5.0
T11_TRAINING_FORWARDS = T11_MACRO_STEPS * 4
T11_EXECUTION_VERDICT = "T11_NEW_EPISODE_JOINT_ARM_VALID"
T11_REPLICATED_VERDICT = "T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED"
T11_TRAINING_FIT_FAILURE_VERDICT = (
    "T11_SAME_TASK_NEW_EPISODE_JOINT_TRAINING_FIT_FAILURE"
)
T11_HELDOUT_GAP_VERDICT = "T11_SAME_TASK_NEW_EPISODE_JOINT_HELDOUT_GAP"
T11_ROOT_SLUG = "libero-t11-same-task-new-episode-balanced-joint-fixed20"
T11_INITIALIZATION_SEED = 20260806
T11_LOSS_RECIPE_SEED = 20260826
T11_CONFIG_SHA256 = "5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45"
T11_CONFIG_RELATIVE_PATH = "configs/benchmarks/libero/train_libero_ar_baseline.yaml"
SPATIAL_NAME = "libero_spatial_no_noops_1.0.0_lerobot"
OBJECT_NAME = "libero_object_no_noops_1.0.0_lerobot"
GOAL_NAME = "libero_goal_no_noops_1.0.0_lerobot"
LIBERO10_NAME = "libero_10_no_noops_1.0.0_lerobot"
T11_SUITE_ORDER = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)
T10_JOINT_Q_BY_SUITE = {
    SPATIAL_NAME: 0.27320731955314786,
    OBJECT_NAME: 0.26340073735799435,
    GOAL_NAME: 0.3232318040080743,
    LIBERO10_NAME: 0.18725771119867107,
}
T11_TRAIN_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000076.parquet": (
                "cfd720e6e95c6a8a0bde52fcf69db35bde57aab401639fe4429ca2252769013b"
            ),
            "videos/chunk-000/observation.images.image/episode_000076.mp4": (
                "ea8625462f38127e25e3a1ae055d2431ef12cde8bb158aa13aad0b1838dc4938"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000076.mp4": (
                "9428026582bf95cd48845c477be6280a55ef6b98c46bfea5de36fbcfbe88053d"
            ),
        },
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "episode_index": 76,
        "episode_length": 133,
        "episode_selection_payload_sha256": (
            "08a163e72aa9a987e873698475f81d2b456379e509d8efc4f445ad3067aeda5d"
        ),
        "label": "A4",
        "start_frame": 0,
        "task": (
            "pick up the black bowl next to the cookie box and place it on the plate"
        ),
        "task_index": 0,
    },
    {
        "assets": {
            "data/chunk-000/episode_000064.parquet": (
                "d3d04574cf1906bd0f7902ce787821abb43d96570076511b6c112911ab2cf4de"
            ),
            "videos/chunk-000/observation.images.image/episode_000064.mp4": (
                "21034dfa8b78a939e1f4e8e9f34ad5f3751b4d59c1520b74d170aca2159d7807"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000064.mp4": (
                "9e46aef2a66642b197dc6e359eb5fdc12ea8b2c1318a9bc7de3d750c8b5eb781"
            ),
        },
        "dataset": "libero_object_no_noops_1.0.0_lerobot",
        "episode_index": 64,
        "episode_length": 129,
        "episode_selection_payload_sha256": (
            "0731ad908d69f809d0a20f8bfbe4d6c6a4f26042764dd41339990dd3cbf732f4"
        ),
        "label": "A5",
        "start_frame": 0,
        "task": "pick up the bbq sauce and place it in the basket",
        "task_index": 3,
    },
    {
        "assets": {
            "data/chunk-000/episode_000030.parquet": (
                "7f01adce8356e1838a64053849f08bb53acc504835f73266643585bfbae36de4"
            ),
            "videos/chunk-000/observation.images.image/episode_000030.mp4": (
                "326f095656208f578772858753ef9272408b1d346e0669bb5adb14c3d706d04a"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000030.mp4": (
                "e3b300a329ad6257d02f800f1911ea68c11174ff5f6e36d663ed11ad08fbcc87"
            ),
        },
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "episode_index": 30,
        "episode_length": 185,
        "episode_selection_payload_sha256": (
            "0ef8699b3f006e9460a703e92d55ad6dd611e5f4b4a30b52ee4af18315b27765"
        ),
        "label": "A6",
        "start_frame": 0,
        "task": "open the top drawer and put the bowl inside",
        "task_index": 2,
    },
    {
        "assets": {
            "data/chunk-000/episode_000303.parquet": (
                "ecec19496a6e481ae861022d30cba46c834d1cf52925c1e152af78acc3a736af"
            ),
            "videos/chunk-000/observation.images.image/episode_000303.mp4": (
                "b8efac07dd2068e14491d372adbd5f9131e4860bf3f5f0c98bfb1c15effec387"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000303.mp4": (
                "002e37672e44b889bdbdaace057b98330d4748462bd801df1b497455b6e0295e"
            ),
        },
        "dataset": "libero_10_no_noops_1.0.0_lerobot",
        "episode_index": 303,
        "episode_length": 244,
        "episode_selection_payload_sha256": (
            "041f7d725846e2191e0c779bbda82323a319bc0a61530f3f0e268bca1390776d"
        ),
        "label": "A7",
        "start_frame": 0,
        "task": "turn on the stove and put the moka pot on it",
        "task_index": 3,
    },
)
T11_HELDOUT_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000265.parquet": (
                "d8f43909a98ccd8e13bbc50215ebcd607e0ee7f39895390edb4923086327a9e3"
            ),
            "videos/chunk-000/observation.images.image/episode_000265.mp4": (
                "c1e35e4387dc0b16ad77009bc34f8e232db896e9f09eb1c0237fdf0d9060d24f"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000265.mp4": (
                "55108bb251273886af1f1770e3b6ac5bf2601ad56eabfde36bafdbdbd42762f4"
            ),
        },
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "episode_index": 265,
        "episode_length": 130,
        "episode_selection_payload_sha256": (
            "08d54f12cf43862296897de4000ce1d75b00474417e6fecfd04c5ec805e2516e"
        ),
        "label": "H4",
        "start_frame": 0,
        "task": (
            "pick up the black bowl next to the cookie box and place it on the plate"
        ),
        "task_index": 0,
    },
    {
        "assets": {
            "data/chunk-000/episode_000337.parquet": (
                "2de8a3be69381e3ffd74ebcc377a991298441bfa9d0ad67d45dc5199addb00af"
            ),
            "videos/chunk-000/observation.images.image/episode_000337.mp4": (
                "2511a0e8c59d63db44217e0c1a79dc07c635a936a6ab03d545bef9c470ddd7af"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000337.mp4": (
                "3b2d3b3064ce400d94a01ae35b5004ec2c64b1c349e67347809fa9959849b3b0"
            ),
        },
        "dataset": "libero_object_no_noops_1.0.0_lerobot",
        "episode_index": 337,
        "episode_length": 134,
        "episode_selection_payload_sha256": (
            "0e837eb33e758153459105ce9adf417ef1505c808e445aa335d926592719a081"
        ),
        "label": "H5",
        "start_frame": 0,
        "task": "pick up the bbq sauce and place it in the basket",
        "task_index": 3,
    },
    {
        "assets": {
            "data/chunk-000/episode_000180.parquet": (
                "46e510231b659177dcea98031e21bd5dda7ef8a71279bdb54a5b8f0e179ff9ea"
            ),
            "videos/chunk-000/observation.images.image/episode_000180.mp4": (
                "d6bb499f59b5a3b185adffec4f0998656f4ff2f52d147930ab7fd5c74ad66095"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000180.mp4": (
                "3420e2b3b7bede74ba3f45773d973974ba7ac3d1ded4d6f829361db1360e2803"
            ),
        },
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "episode_index": 180,
        "episode_length": 182,
        "episode_selection_payload_sha256": (
            "11b1eca88d1a1a10c1903c061f5a6035bbf2bd45aa335975c83ce837641f508b"
        ),
        "label": "H6",
        "start_frame": 0,
        "task": "open the top drawer and put the bowl inside",
        "task_index": 2,
    },
    {
        "assets": {
            "data/chunk-000/episode_000272.parquet": (
                "ee3ce6f2a6a7388dd6266c4aec888dac5230645ce102d3d8609b8e6a85d45985"
            ),
            "videos/chunk-000/observation.images.image/episode_000272.mp4": (
                "41f685382ce65d8f145da5700fee2141ecf0783aacfc66a716a166d094ffa775"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000272.mp4": (
                "79e8972ff43d00618780369a893e34b7160dc2c396cd43701ce7bdc987034818"
            ),
        },
        "dataset": "libero_10_no_noops_1.0.0_lerobot",
        "episode_index": 272,
        "episode_length": 272,
        "episode_selection_payload_sha256": (
            "070192b8c136cf7d5c12d57f02ae10625839b2a9b88b871d79281025d70688ad"
        ),
        "label": "H7",
        "start_frame": 0,
        "task": "turn on the stove and put the moka pot on it",
        "task_index": 3,
    },
)
T11_TRAIN_LABELS = ("A4", "A5", "A6", "A7")
T11_HELDOUT_LABELS = ("H4", "H5", "H6", "H7")
T11_EXCLUDED_SAMPLES = (
    {
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "task_index": 0,
        "episode_index": 0,
    },
    {
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "task_index": 0,
        "episode_index": 16,
    },
    {
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "task_index": 0,
        "episode_index": 405,
    },
    {
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "task_index": 0,
        "episode_index": 40,
    },
    {
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "task_index": 7,
        "episode_index": 36,
    },
    {
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "task_index": 1,
        "episode_index": 325,
    },
    {
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "task_index": 4,
        "episode_index": 11,
    },
    {
        "dataset": "libero_object_no_noops_1.0.0_lerobot",
        "task_index": 3,
        "episode_index": 82,
    },
    {
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "task_index": 2,
        "episode_index": 70,
    },
    {
        "dataset": "libero_10_no_noops_1.0.0_lerobot",
        "task_index": 3,
        "episode_index": 259,
    },
    {
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "task_index": 0,
        "episode_index": 30,
    },
    {
        "dataset": "libero_object_no_noops_1.0.0_lerobot",
        "task_index": 3,
        "episode_index": 166,
    },
    {
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "task_index": 2,
        "episode_index": 248,
    },
    {
        "dataset": "libero_10_no_noops_1.0.0_lerobot",
        "task_index": 3,
        "episode_index": 278,
    },
    {
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "task_index": 5,
        "episode_index": 82,
    },
)
T11_ELIGIBLE_MANIFEST_BYTES = 8676
T11_ELIGIBLE_MANIFEST_SHA256 = (
    "25486093e047d38d95151b36277874b89851f934720e3418e1f817d0b736745b"
)
T11_SELECTION_MANIFEST_BYTES = 6198
T11_SELECTION_MANIFEST_SHA256 = (
    "29e9ce96b8d4d6b0d5e41c6cc2d0fb887c4061990d4c2669a498d806dbd82dcd"
)
T11_SELECTION_RULE = (
    "in frozen suite order spatial,object,goal,10; within each frozen T10 task "
    "exclude every T1-T10 measured-or-backpropagated episode and excluded Goal "
    "episode 82; rank remaining episodes by exact T11_NEW_SAMPLE_SELECTION_V1 "
    "payload SHA256; assign first to A4-A7 and second to H4-H7"
)
T11_SUITE_METADATA_SHA256 = {
    "libero_spatial_no_noops_1.0.0_lerobot": {
        "episodes.jsonl": "690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7",
        "tasks.jsonl": "399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1",
    },
    "libero_object_no_noops_1.0.0_lerobot": {
        "episodes.jsonl": "63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea",
        "tasks.jsonl": "68ef5f9bc5a0bd74f46140f6721fa0ea74e997d74e37b8714a539f61337e7862",
    },
    "libero_goal_no_noops_1.0.0_lerobot": {
        "episodes.jsonl": "548d91fe48b7d439248523dd3f7a5e4b15fc77d5eb1b7cfdd6da0033d422cb43",
        "tasks.jsonl": "39f08f81b289ad3041f1c8ada88f679fe60774e9fde4083415881486edc23d55",
    },
    "libero_10_no_noops_1.0.0_lerobot": {
        "episodes.jsonl": "5589f8f87cfddb34812782462160bf55b0d3082e404240682d1d0a89faba8265",
        "tasks.jsonl": "45f9eb4d4b6b04999f64640c0aae380555372b7b273a904f5f459ad05d4a0a6a",
    },
}
T11_FIXED_RECIPE_SIGNATURE_SHA256 = (
    "ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad"
)
T6_LINEAGE_TRAINING_CORE_SHA256 = (
    "e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352"
)
T7_PRE_ACTION_LOSSES = {
    "A0": 13.679718971252441,
    "A1": 14.696584701538086,
    "A2": 12.06648063659668,
    "A3": 20.43115234375,
}
T7_PRE_HELDOUT_ACTION_LOSSES = {
    "H0": 13.379679679870605,
    "H1": 14.698094367980957,
    "H2": 12.026171684265137,
    "H3": 20.885684967041016,
}
T7_TERMINAL_ACTION_LOSSES = {
    "A0": 14.97652816772461,
    "A1": 11.250626564025879,
    "A2": 5.85097074508667,
    "A3": 2.8098158836364746,
    "H0": 14.401052474975586,
    "H1": 11.137908935546875,
    "H2": 6.1973876953125,
    "H3": 2.8864619731903076,
}
SELECTED_STATS_SHA256 = (
    "e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7"
)
SELECTED_POPULATION_SHA256 = (
    "7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146"
)
T1_PREDECESSOR_RESULT_SHA256 = (
    "9d9c139fd67baa8c131ad9e6537862536b8262b732c2fda668a7c8b8b6a8621a"
)
T1_PREDECESSOR_RESULT_PATH = Path(
    "/tmp/sana-wam-libero-t1-one-update-fp32master-0337e28-20260806-a3/RESULT.json"
)
T2_PREDECESSOR_RESULT_SHA256 = (
    "67d250cdb13470bea9e9fa53531d3c76c65145e5040dc7a45079d84ac4960a57"
)
T2_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t2/2dc1ce730df3/"
    "libero-t2-fixed20-20260806-a1/RESULT.json"
)
T2_PREDECESSOR_SOURCE_COMMIT = "2dc1ce730df37bd9a2a71e4946f153380a8d4649"
T2_PREDECESSOR_RUNNER_SHA256 = (
    "f5e5e3e80d6b5afd586e54e83a7e63dcf43280192363c70580aa03145267e9a6"
)
T2_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t2_fixed_sample_20step_gpu.py"
)
T3_PREDECESSOR_RESULT_SHA256 = (
    "c883608f2a47b6258f824d4d97a94f8a390d03bab671a592fb758eea61b3a01e"
)
T3_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t3/ac431f8727ac/"
    "libero-t3-heldout3-fixed20-220afb60725d0cfd591bc4fe225cdd21/RESULT.json"
)
T3_PREDECESSOR_SOURCE_COMMIT = "ac431f8727ac345c3bd0ad4a442e53801310fd81"
T3_PREDECESSOR_RUNNER_SHA256 = (
    "0db0bfe086bc4b5ea92462e4832f0c448408656cca819796ec253f6226b4c017"
)
T3_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t3_heldout_recipe_transfer_gpu.py"
)
T4_PREDECESSOR_RESULT_SHA256 = (
    "4c33aaff6d202068d77efc0ac406c74198c56e72527cfabdde046fc9a3a4b6f4"
)
T4_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t4/7db8182ef46f/"
    "libero-t4-heldout3-fixed20-36120c596971575d286d378df42ac564/RESULT.json"
)
T4_PREDECESSOR_SOURCE_COMMIT = "7db8182ef46fce367f04f66bfbc34690fa86592b"
T4_PREDECESSOR_RUNNER_SHA256 = (
    "3d449761861b7fe75816d4d186b2e82402c160422eeb5f490fbb61ed274f088c"
)
T4_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t4_heldout_sample_transfer_gpu.py"
)
T5_PREDECESSOR_RESULT_SHA256 = (
    "a656aaef1528537527fe830ad7d4107138b29e8e254b5606b43c46a47e323e83"
)
T5_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t5/5150693a0751/"
    "libero-t5-crosstask3-fixed20-cf7dd8a1b1cef03511d2026a48e4a271/RESULT.json"
)
T5_PREDECESSOR_SOURCE_COMMIT = "5150693a0751200ef431968a863c69f8ca08a7df"
T5_PREDECESSOR_RUNNER_SHA256 = (
    "a4bbdcf752aa0a34a43ea4f51e7875f7fd160985280a7274b29e36350d9605c1"
)
T5_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t5_cross_task_transfer_gpu.py"
)
T6_PREDECESSOR_RESULT_SHA256 = (
    "4855f3771b80349547c985d137426cce79e25597f910eff88e136328424b8b89"
)
T6_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t6/708b1d856986/"
    "libero-t6-crosssuite3-fixed20-67a02fcc85508e03f136e09221a6a9d4/RESULT.json"
)
T6_PREDECESSOR_SOURCE_COMMIT = "708b1d8569866608898031f9116566d52fdeb742"
T6_PREDECESSOR_RUNNER_SHA256 = (
    "004f444168f26162f012c408193e119a8ff428a64f90e7dbc9ab595df9082903"
)
T6_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t6_cross_suite_transfer_gpu.py"
)
T7_PREDECESSOR_SOURCE_COMMIT = "d19109a2314f8e7186571afed6b4acd816d7cfab"
T7_PREDECESSOR_RUNNER_SHA256 = (
    "b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b"
)
T7_PREDECESSOR_RUNNER_PATH = (
    ROOT / "scripts/smoke_libero_ar_t7_four_suite_cyclic_gpu.py"
)
T7_PREDECESSOR_RESULT_SHA256 = (
    "9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a"
)
T7_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t7/d19109a2314f/"
    "libero-t7-foursuite-cyclic-fixed20-61e0a0ff817970e994b0875be4840ed6/"
    "RESULT.json"
)
T9_AGGREGATE_SOURCE_COMMIT = "b7cded5fd9cf83ffabeba18a7e352b1cb4438b66"
T9_AGGREGATE_RUNNER_SHA256 = (
    "b884ef028a48ee37afda70e730c431db48ed30f62be7f59cb2c665b6072ffc0d"
)
T9_AGGREGATE_RUNNER_PATH = ROOT / "scripts/summarize_libero_ar_t9_latin_square.py"
T9_AGGREGATE_RESULT_SHA256 = (
    "0cfc53b820939123de4bc2a626380495878d4d5c9a37a0be9be667173254149d"
)
T9_AGGREGATE_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate/b7cded5fd9cf/"
    "libero-t9-latin-square-combined-ab99dd8758ef03bb191d5fb48f3b9fbc/"
    "RESULT.json"
)
T9_AGGREGATE_ROOT = T9_AGGREGATE_RESULT_PATH.parent
T10_ARM_SOURCE_COMMIT = "6861e5a13fa8110986f1b46f3062de9f0b0e3954"
T10_ARM_RUNNER_SHA256 = (
    "d7e3e7da3cd0a13e5217f0a044b2f36a4a703be7227149a897268f82b74857dd"
)
T10_ARM_RUNNER_PATH = ROOT / "scripts/smoke_libero_ar_t10_exact_balanced_joint_gpu.py"
T10_SEQ_RESULT_SHA256 = (
    "d92fe05fe523c346e90ab6a392ddad9c3ec41764d5581211e6223895a42e8937"
)
T10_SEQ_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t10/6861e5a13fa8/"
    "libero-t10-seq-matched-core-fixed20-0d211cfc62324c0f4ab506cad2fd76f6/"
    "RESULT.json"
)
T10_JOINT_RESULT_SHA256 = (
    "423ce3e01bea7368786b1c470a790af666baf7504a2235894a59b82efef3ea9b"
)
T10_JOINT_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t10/6861e5a13fa8/"
    "libero-t10-joint-matched-core-fixed20-561f95c143f58bc635259bff41ef4366/"
    "RESULT.json"
)
T10_AGGREGATE_SOURCE_COMMIT = "128e1be8cd48f8cef1b7c5f24d1bdecfbe45054a"
T10_AGGREGATE_RUNNER_SHA256 = (
    "b45e9536b35790c06e599ba919ed1370c64e4e1dff477019fab9f0ccc40b61c5"
)
T10_AGGREGATE_RUNNER_PATH = (
    ROOT / "scripts/summarize_libero_ar_t10_exact_balanced_joint.py"
)
T10_AGGREGATE_RESULT_SHA256 = (
    "8fcd26ea3a59fe6e01cf8f279301c899ed5dab541ada9c57e2ce85888abd147b"
)
T10_AGGREGATE_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t10_aggregate/128e1be8cd48/"
    "libero-t10-exact-balanced-joint-combined-f77c1cd125343922634395db86812db9/"
    "RESULT.json"
)
T10_AGGREGATE_ROOT = T10_AGGREGATE_RESULT_PATH.parent
SPATIAL_STATS_GR00T_SHA256 = (
    "0a4b08f5afcdcbe186ec70ea6ff2569233bab706a4d99ed603b23a7688bc33bb"
)
T11_RUN_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t11")
_ACTIVE_RUN_ROOT: Path | None = None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"T11 metadata must be a regular non-symlink file: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise RuntimeError(
                f"T11 metadata contains a blank row: {path}:{line_number}"
            )
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(
                f"T11 metadata row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _build_eligible_probe_manifest(
    metadata_by_suite: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> dict[str, Any]:
    if set(metadata_by_suite) != set(T11_SUITE_ORDER):
        raise RuntimeError("T11 eligible suite set differs")
    expected_raw_counts = {
        SPATIAL_NAME: 432,
        OBJECT_NAME: 454,
        GOAL_NAME: 428,
        LIBERO10_NAME: 379,
    }
    expected_eligible_counts = {
        SPATIAL_NAME: 41,
        OBJECT_NAME: 44,
        GOAL_NAME: 34,
        LIBERO10_NAME: 39,
    }
    update_by_dataset = {row["dataset"]: row for row in T11_TRAIN_SAMPLES}
    consumed = {
        (row["dataset"], row["task_index"], row["episode_index"])
        for row in T11_EXCLUDED_SAMPLES
    }
    suites = []
    for dataset in T11_SUITE_ORDER:
        task_rows, episode_rows = metadata_by_suite[dataset]
        if any(set(row) != {"task_index", "task"} for row in task_rows):
            raise RuntimeError(f"T11 tasks metadata schema differs: {dataset}")
        task_by_index = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        if (
            sorted(task_by_index) != list(range(10))
            or len(task_by_index) != len(task_rows)
            or len(set(task_by_index.values())) != len(task_by_index)
        ):
            raise RuntimeError(f"T11 task registry differs: {dataset}")
        task_index_by_name = {name: index for index, name in task_by_index.items()}
        update_task_index = int(update_by_dataset[dataset]["task_index"])
        eligible_episodes: list[dict[str, int]] = []
        observed_episode_indices: set[int] = set()
        for row in episode_rows:
            if set(row) != {"episode_index", "tasks", "length"}:
                raise RuntimeError(f"T11 episodes metadata schema differs: {dataset}")
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            tasks = row["tasks"]
            if (
                episode_index in observed_episode_indices
                or not isinstance(tasks, list)
                or len(tasks) != 1
                or tasks[0] not in task_index_by_name
                or length <= 0
            ):
                raise RuntimeError(f"T11 episodes metadata identity differs: {dataset}")
            observed_episode_indices.add(episode_index)
            task_index = task_index_by_name[tasks[0]]
            if (
                task_index == update_task_index
                and (dataset, task_index, episode_index) not in consumed
            ):
                eligible_episodes.append(
                    {"episode_index": episode_index, "length": length}
                )
        if len(episode_rows) not in {
            expected_raw_counts[dataset],
            427 if dataset == GOAL_NAME else expected_raw_counts[dataset],
        }:
            raise RuntimeError(f"T11 source episode count differs: {dataset}")
        tasks = [
            {
                "episode_count": len(eligible_episodes),
                "episodes": sorted(
                    eligible_episodes,
                    key=lambda row: row["episode_index"],
                ),
                "task": task_by_index[update_task_index],
                "task_index": update_task_index,
            }
        ]
        if (
            len(eligible_episodes) != expected_eligible_counts[dataset]
            or update_by_dataset[dataset]["task"] != task_by_index[update_task_index]
        ):
            raise RuntimeError(f"T11 eligible episode population differs: {dataset}")
        suites.append(
            {
                "dataset": dataset,
                "episode_count": sum(row["episode_count"] for row in tasks),
                "episodes_jsonl_sha256": T11_SUITE_METADATA_SHA256[dataset][
                    "episodes.jsonl"
                ],
                "task_count": len(tasks),
                "tasks": tasks,
                "tasks_jsonl_sha256": T11_SUITE_METADATA_SHA256[dataset]["tasks.jsonl"],
                "update_task_index": update_task_index,
            }
        )
    manifest = {
        "candidate_episode_count": sum(row["episode_count"] for row in suites),
        "candidate_suite_count": len(suites),
        "candidate_task_count": sum(row["task_count"] for row in suites),
        "excluded_samples": list(T11_EXCLUDED_SAMPLES),
        "schema_version": ("sana-wam-libero-t11-same-task-new-episode-eligible-v1"),
        "start_frame": 0,
        "suites": suites,
        "t10_aggregate_result_sha256": T10_AGGREGATE_RESULT_SHA256,
    }
    if (
        manifest["candidate_suite_count"] != 4
        or manifest["candidate_task_count"] != 4
        or manifest["candidate_episode_count"] != 158
        or len(_canonical_json_bytes(manifest)) != T11_ELIGIBLE_MANIFEST_BYTES
        or _sha256_json(manifest) != T11_ELIGIBLE_MANIFEST_SHA256
    ):
        raise RuntimeError("T11 eligible same-task probe manifest differs")
    return manifest


def _eligible_sample_manifest(dataset_roots: dict[str, Path]) -> dict[str, Any]:
    metadata_by_suite = {}
    for dataset in T11_SUITE_ORDER:
        root = dataset_roots[dataset]
        tasks_path = root / "meta/tasks.jsonl"
        episodes_path = root / "meta/episodes.jsonl"
        if (
            _sha256_file(tasks_path)
            != T11_SUITE_METADATA_SHA256[dataset]["tasks.jsonl"]
        ):
            raise RuntimeError(f"T11 tasks metadata SHA differs: {dataset}")
        if (
            _sha256_file(episodes_path)
            != T11_SUITE_METADATA_SHA256[dataset]["episodes.jsonl"]
        ):
            raise RuntimeError(f"T11 episodes metadata SHA differs: {dataset}")
        metadata_by_suite[dataset] = (
            _read_jsonl(tasks_path),
            _read_jsonl(episodes_path),
        )
    return _build_eligible_probe_manifest(metadata_by_suite)


def _episode_selection_payload_sha256(
    dataset: str, task_index: int, episode_index: int
) -> str:
    payload = (
        "SANA-WAM/LIBERO/T11_NEW_SAMPLE_SELECTION_V1\n"
        f"T10_AGGREGATE_RESULT_SHA256={T10_AGGREGATE_RESULT_SHA256}\n"
        f"DATASET={dataset}\n"
        f"TASK_INDEX={task_index}\n"
        f"EPISODE_INDEX={episode_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _selection_manifest(eligible_manifest: dict[str, Any]) -> dict[str, Any]:
    samples = []
    suite_by_dataset = {row["dataset"]: row for row in eligible_manifest["suites"]}
    expected_by_label = {
        row["label"]: row for row in (*T11_TRAIN_SAMPLES, *T11_HELDOUT_SAMPLES)
    }
    for suite_rank, dataset in enumerate(T11_SUITE_ORDER):
        task_rows = suite_by_dataset[dataset]["tasks"]
        if len(task_rows) != 1:
            raise RuntimeError(f"T11 eligible probe task count differs: {dataset}")
        task_row = task_rows[0]
        task_index = int(task_row["task_index"])
        ranked_episodes = sorted(
            (
                _episode_selection_payload_sha256(
                    dataset, task_index, int(episode_row["episode_index"])
                ),
                episode_row,
            )
            for episode_row in task_row["episodes"]
        )
        for pair_rank, label_prefix in enumerate(("A", "H")):
            episode_sha256, episode_row = ranked_episodes[pair_rank]
            label = f"{label_prefix}{suite_rank + 4}"
            expected = expected_by_label[label]
            sample = {
                "assets": expected["assets"],
                "dataset": dataset,
                "episode_index": int(episode_row["episode_index"]),
                "episode_length": int(episode_row["length"]),
                "episode_selection_payload_sha256": episode_sha256,
                "label": label,
                "start_frame": 0,
                "task": task_row["task"],
                "task_index": task_index,
            }
            if sample != expected:
                raise RuntimeError(f"T11 mechanically selected sample {label} differs")
            samples.append(sample)
    manifest = {
        "eligible_manifest_sha256": T11_ELIGIBLE_MANIFEST_SHA256,
        "samples": samples,
        "schema_version": ("sana-wam-libero-t11-same-task-new-episode-selection-v1"),
        "selection_rule": T11_SELECTION_RULE,
        "suite_order": list(T11_SUITE_ORDER),
        "t10_aggregate_result_sha256": T10_AGGREGATE_RESULT_SHA256,
    }
    if (
        len(_canonical_json_bytes(manifest)) != T11_SELECTION_MANIFEST_BYTES
        or _sha256_json(manifest) != T11_SELECTION_MANIFEST_SHA256
    ):
        raise RuntimeError("T11 selected same-task probe manifest differs")
    return manifest


def _select_t11_samples(
    dataset: Any,
    eligible_manifest: dict[str, Any],
    selection_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    live_episode_rows: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in T11_SUITE_ORDER
    }
    live_task_rows: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in T11_SUITE_ORDER
    }
    for episode in dataset._episodes:
        if episode.dataset not in T11_SUITE_ORDER:
            continue
        if episode.dataset == GOAL_NAME and episode.episode_index == 82:
            raise RuntimeError("T11 excluded Goal episode 82 entered live registry")
        task_row = {"task_index": episode.task_index, "task": episode.task}
        suite_tasks = live_task_rows[episode.dataset]
        suite_episodes = live_episode_rows[episode.dataset]
        previous_task = suite_tasks.setdefault(episode.task_index, task_row)
        if previous_task != task_row or episode.episode_index in suite_episodes:
            raise RuntimeError("T11 live cross-suite registry identity is ambiguous")
        suite_episodes[episode.episode_index] = {
            "episode_index": episode.episode_index,
            "tasks": [episode.task],
            "length": episode.length,
        }
    live_manifest = _build_eligible_probe_manifest(
        {
            name: (
                [live_task_rows[name][index] for index in sorted(live_task_rows[name])],
                [
                    live_episode_rows[name][index]
                    for index in sorted(live_episode_rows[name])
                ],
            )
            for name in T11_SUITE_ORDER
        }
    )
    if live_manifest != eligible_manifest:
        raise RuntimeError("T11 live eligible cross-suite population differs")

    if selection_manifest["samples"] != [
        row
        for pair in zip(T11_TRAIN_SAMPLES, T11_HELDOUT_SAMPLES, strict=True)
        for row in pair
    ]:
        raise RuntimeError("T11 selected A/H sample interleave differs")
    role_rows = [*T11_TRAIN_SAMPLES, *T11_HELDOUT_SAMPLES]
    expected_keys = {
        (
            row["dataset"],
            int(row["task_index"]),
            int(row["episode_index"]),
            int(row["start_frame"]),
        )
        for row in role_rows
    }
    start_zero: dict[tuple[str, int, int, int], list[tuple[int, Any]]] = {
        key: [] for key in expected_keys
    }
    for dataset_index, (episode_position, start) in enumerate(dataset._windows):
        episode = dataset._episodes[episode_position]
        key = (episode.dataset, episode.task_index, episode.episode_index, start)
        if key in start_zero:
            start_zero[key].append((dataset_index, episode))

    selected = {}
    for row in role_rows:
        key = (
            row["dataset"],
            int(row["task_index"]),
            int(row["episode_index"]),
            int(row["start_frame"]),
        )
        matches = start_zero[key]
        if len(matches) != 1:
            raise RuntimeError(
                "T11 cross-suite episode/start must occur exactly once: "
                f"identity={key!r} matches={len(matches)}"
            )
        dataset_index, episode = matches[0]
        selected[row["label"]] = {
            **row,
            "dataset_index": dataset_index,
            "episode": episode,
        }
    if set(selected) != set(T11_TRAIN_LABELS) | set(T11_HELDOUT_LABELS):
        raise RuntimeError("T11 selected role label set differs")
    return selected


@contextlib.contextmanager
def _fixed_rng_recipe(torch_module: Any, *, seed: int, cuda_device: int | None):
    """Reset one loss recipe while restoring the caller's RNG states."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [] if cuda_device is None else [cuda_device]
    try:
        with torch_module.random.fork_rng(devices=devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch_module.manual_seed(seed)
            if cuda_device is not None:
                torch_module.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _paired_four_sample_summary(
    before_action_losses: dict[str, float],
    after_action_losses: dict[str, float],
    *,
    samples: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    labels = tuple(row["label"] for row in samples)
    expected_labels = set(labels)
    if (
        set(before_action_losses) != expected_labels
        or set(after_action_losses) != expected_labels
    ):
        raise ValueError("T11 frozen labels differ")
    sample_by_label = {row["label"]: row for row in samples}
    if set(sample_by_label) != expected_labels:
        raise ValueError("T11 sample labels differ from the frozen set")
    rows = []
    for label in labels:
        sample = sample_by_label[label]
        before = _require_strictly_positive_finite(
            before_action_losses[label],
            field=f"{label} before_action_loss",
        )
        after = _require_strictly_positive_finite(
            after_action_losses[label],
            field=f"{label} after_action_loss",
        )
        ratio = _require_strictly_positive_finite(
            after / before,
            field=f"{label} post_to_pre_ratio",
        )
        rows.append(
            {
                "absolute_delta": after - before,
                "after_action_loss": after,
                "before_action_loss": before,
                "dataset": sample["dataset"],
                "episode_index": sample["episode_index"],
                "improved": after < before,
                "label": label,
                "post_to_pre_ratio": ratio,
                "relative_drop": 1.0 - ratio,
                "start_frame": sample["start_frame"],
                "task_index": sample["task_index"],
            }
        )
    return {
        "improved_count": sum(row["improved"] for row in rows),
        "median_post_to_pre_ratio": statistics.median(
            row["post_to_pre_ratio"] for row in rows
        ),
        "all_ratios_strictly_below_one": all(
            row["post_to_pre_ratio"] < 1.0 for row in rows
        ),
        "per_sample": rows,
        "sample_count": len(rows),
        "sorted_post_to_pre_ratios": sorted(row["post_to_pre_ratio"] for row in rows),
        "suite_count": len({row["dataset"] for row in rows}),
        "suites_distinct": len({row["dataset"] for row in rows}) == len(rows),
        "task_count": len({(row["dataset"], row["task_index"]) for row in rows}),
    }


def _require_strictly_positive_finite(value: float, *, field: str) -> float:
    observed = float(value)
    if not math.isfinite(observed) or observed <= 0.0:
        raise ValueError(f"T11 {field} must be finite and strictly positive")
    return observed


def _no_intra_macro_mutation_evidence(
    before: dict[str, Any],
    after_microbacks: dict[str, Any],
    *,
    macro_step: int,
) -> dict[str, Any]:
    components = (
        "model_parameters",
        "fp32_master_parameters",
        "optimizer_state",
    )
    if macro_step < 1 or macro_step > T11_MACRO_STEPS:
        raise ValueError("T11 no-mutation evidence macro step is outside 1..20")
    if set(before) != set(components) or set(after_microbacks) != set(components):
        raise ValueError("T11 no-mutation evidence component schema differs")
    component_unchanged = {
        component: before[component] == after_microbacks[component]
        for component in components
    }
    before_sha256 = _sha256_json(before)
    after_sha256 = _sha256_json(after_microbacks)
    state_unchanged = all(component_unchanged.values()) and (
        before_sha256 == after_sha256
    )
    evidence = {
        "component_state_sha256": {
            component: _sha256_json(before[component]) for component in components
        },
        "components_unchanged": component_unchanged,
        "macro_step": macro_step,
        "post_microback_pre_optimizer_state_sha256": after_sha256,
        "pre_microback_state_sha256": before_sha256,
        "state_unchanged": state_unchanged,
    }
    if not state_unchanged:
        changed = [
            component
            for component, unchanged in component_unchanged.items()
            if not unchanged
        ]
        raise RuntimeError(
            "T11 training state mutated within macro before optimizer step "
            f"{macro_step}: {changed!r}"
        )
    return evidence


def _same_task_new_episode_summary(
    update_before: dict[str, float],
    update_after: dict[str, float],
    heldout_before: dict[str, float],
    heldout_after: dict[str, float],
) -> dict[str, Any]:
    cyclic_training_summary = _paired_four_sample_summary(
        update_before,
        update_after,
        samples=T11_TRAIN_SAMPLES,
    )
    fresh_heldout_summary = _paired_four_sample_summary(
        heldout_before,
        heldout_after,
        samples=T11_HELDOUT_SAMPLES,
    )
    update_by_dataset = {
        row["dataset"]: row for row in cyclic_training_summary["per_sample"]
    }
    heldout_by_dataset = {
        row["dataset"]: row for row in fresh_heldout_summary["per_sample"]
    }
    if set(update_by_dataset) != set(T11_SUITE_ORDER) or set(heldout_by_dataset) != set(
        T11_SUITE_ORDER
    ):
        raise ValueError("T11 suite coverage differs from the frozen order")
    per_suite = []
    for dataset in T11_SUITE_ORDER:
        update_row = update_by_dataset[dataset]
        heldout_row = heldout_by_dataset[dataset]
        r_a = float(update_row["post_to_pre_ratio"])
        r_h = float(heldout_row["post_to_pre_ratio"])
        q = 0.5 * (r_a + r_h)
        t10_joint_q = float(T10_JOINT_Q_BY_SUITE[dataset])
        per_suite.append(
            {
                "dataset": dataset,
                "q": q,
                "q_minus_t10_joint_q": q - t10_joint_q,
                "q_strictly_below_t10_joint_q": q < t10_joint_q,
                "rA": r_a,
                "rA_strictly_below_one": r_a < 1.0,
                "rH": r_h,
                "rH_strictly_below_one": r_h < 1.0,
                "t10_joint_q": t10_joint_q,
            }
        )
    all_update_improved = all(row["rA_strictly_below_one"] for row in per_suite)
    all_heldout_improved = all(row["rH_strictly_below_one"] for row in per_suite)
    if all_update_improved and all_heldout_improved:
        scientific_verdict = T11_REPLICATED_VERDICT
    elif not all_update_improved:
        scientific_verdict = T11_TRAINING_FIT_FAILURE_VERDICT
    else:
        scientific_verdict = T11_HELDOUT_GAP_VERDICT
    return {
        "all_four_update_ratios_strictly_below_one": all_update_improved,
        "all_four_heldout_ratios_strictly_below_one": all_heldout_improved,
        "all_eight_ratios_strictly_below_one": (
            all_update_improved and all_heldout_improved
        ),
        "any_update_ratio_at_least_one": not all_update_improved,
        "fresh_same_task_update_heldout": fresh_heldout_summary,
        "ordered_rule": [
            "all eight unrounded rA/rH ratios are strictly below 1",
            "otherwise any unrounded rA ratio is at least 1",
            "otherwise heldout gap",
        ],
        "per_suite": per_suite,
        "scientific_verdict": scientific_verdict,
        "t10_joint_q_comparison": {
            "excluded_from_scientific_classifier": True,
            "per_suite": [
                {
                    key: row[key]
                    for key in (
                        "dataset",
                        "q",
                        "q_minus_t10_joint_q",
                        "q_strictly_below_t10_joint_q",
                        "t10_joint_q",
                    )
                }
                for row in per_suite
            ],
            "q_strictly_below_t10_joint_count": sum(
                row["q_strictly_below_t10_joint_q"] for row in per_suite
            ),
        },
        "update_samples": cyclic_training_summary,
    }


def _loss_weights_for_macro(step_index: int) -> dict[str, float]:
    if step_index < 0 or step_index >= T11_UPDATE_STEPS:
        raise ValueError("T11 step index is outside the frozen 20-step schedule")
    return {label: T11_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO for label in T11_TRAIN_LABELS}


_CORE_OPTIMIZER_KEYS = (
    "adam_step_values",
    "betas",
    "constant_learning_rates",
    "cumulative_master_update",
    "cumulative_projected_bf16_update",
    "first_master_update",
    "first_projected_bf16_update",
    "master_object_identity_persistent",
    "optimizer_master_parameter_count",
    "optimizer_master_parameter_elements",
    "optimizer_state_finite",
    "production_accumulation_equivalent",
    "production_schedule_equivalent",
    "projection_exact_after_steps",
    "update_sentinel_count",
    "update_sentinel_names",
    "weight_decay",
)
_CORE_RANDOMNESS_KEYS = (
    "initialization_seed",
    "recipe_reset",
    "rng_state_restored_after_every_forward",
    "training_loss_recipe_seed",
    "training_recipe_signature_sha256",
    "training_recipe_signatures",
    "unique_training_recipe_signatures",
)
_CORE_SCOPE_KEYS = (
    "architecture_training_forwards",
    "backward_calls",
    "optimizer_steps",
    "sana_base_pretrained_checkpoint_loaded",
    "sana_wam_training_checkpoint_loaded",
    "sana_wam_training_checkpoint_saved",
)


def _training_core_projection(result: dict[str, Any]) -> dict[str, Any]:
    learnability = {
        key: value for key, value in result["learnability"].items() if key != "role"
    }
    per_step = [
        {
            key: value
            for key, value in row.items()
            if key not in {"peak_reserved_bytes", "seconds"}
        }
        for row in result["per_step"]
    ]
    if len(per_step) != T11_UPDATE_STEPS:
        raise ValueError("T11/T3 training-core view requires exactly 20 step rows")
    return {
        "architecture": result["architecture"],
        "assets": {
            key: result["assets"][key]
            for key in (
                "config_sha256",
                "spatial_files_post_sample_manifest_sha256",
                "stats_population_sha256",
                "stats_sha256",
                "stats_validation",
            )
        },
        "identity": {
            key: result["identity"][key]
            for key in (
                "config_sha256",
                "sana_commit",
                "spatial_assets_post_sample_sha256",
                "stats_population_sha256",
                "stats_sha256",
            )
        },
        "inputs": result["inputs"],
        "learnability": learnability,
        "optimizer": {key: result["optimizer"][key] for key in _CORE_OPTIMIZER_KEYS},
        "per_step": per_step,
        "randomness": {key: result["randomness"][key] for key in _CORE_RANDOMNESS_KEYS},
        "sample": result["sample"],
        "scope": {key: result["scope"][key] for key in _CORE_SCOPE_KEYS},
    }


def _write_terminal_report(path: Path, report: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            report,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def _tensor_observation(tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "values": [float(value) for value in values.tolist()],
    }


def _terminalize_failure(root: Path, error: BaseException) -> None:
    marker = root / "FAILED.json"
    result_marker = root / "RESULT.json"
    if (
        not result_marker.exists()
        and not result_marker.is_symlink()
        and not marker.exists()
        and not marker.is_symlink()
    ):
        _write_terminal_report(
            marker,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "error_type": type(error).__name__,
                "result": "FAIL",
                "schema_version": "sana-wam-libero-t11-matched-core-arm-failure-v1",
            },
        )
    _freeze_run_root(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--steps", type=int, default=T11_UPDATE_STEPS)
    parser.add_argument(
        "--initialization-seed", type=int, default=T11_INITIALIZATION_SEED
    )
    parser.add_argument("--loss-recipe-seed", type=int, default=T11_LOSS_RECIPE_SEED)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

    if args.steps != T11_UPDATE_STEPS:
        raise ValueError(f"T11 steps must be exactly {T11_UPDATE_STEPS}")
    if args.initialization_seed != T11_INITIALIZATION_SEED:
        raise ValueError("T11 initialization seed differs")
    if args.loss_recipe_seed != T11_LOSS_RECIPE_SEED:
        raise ValueError("T11 loss recipe seed differs")
    if len(args.nonce) != 32 or any(
        character not in "0123456789abcdef" for character in args.nonce
    ):
        raise ValueError("T11 nonce must be exactly 32 lowercase hex characters")
    expected_root_name = f"{T11_ROOT_SLUG}-{args.nonce}"
    if args.run_root.name != expected_root_name:
        raise ValueError("T11 run-root basename does not bind the nonce")
    if args.run_root.parent.name != args.expected_repo_commit[:12]:
        raise ValueError("T11 run-root parent does not bind the source commit")
    expected_run_root = (
        T11_RUN_NAMESPACE / args.expected_repo_commit[:12] / expected_root_name
    )
    if args.run_root.expanduser() != expected_run_root:
        raise ValueError(
            "T11 run root differs from the frozen non-formal namespace: "
            f"expected={expected_run_root} got={args.run_root.expanduser()}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {str(args.physical_gpu), args.expected_gpu_uuid}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the selected physical GPU "
            f"({args.physical_gpu} or {args.expected_gpu_uuid}), got {visible!r}"
        )

    config_candidate = args.config.expanduser()
    if config_candidate.is_symlink() or not config_candidate.is_file():
        raise ValueError(
            f"config must be a regular non-symlink file: {config_candidate}"
        )
    config_path = config_candidate.resolve()
    expected_config_path = (ROOT / T11_CONFIG_RELATIVE_PATH).resolve()
    if config_path != expected_config_path:
        raise ValueError(
            f"T11 config path differs: expected={expected_config_path} got={config_path}"
        )
    runner_path = Path(__file__).resolve()
    run_root = _create_run_root(args.run_root.expanduser())
    global _ACTIVE_RUN_ROOT
    _ACTIVE_RUN_ROOT = run_root

    actual_commit = _repo_commit()
    actual_config_sha256 = _sha256_file(config_path)
    if actual_config_sha256 != T11_CONFIG_SHA256:
        raise RuntimeError("T11 config content differs from the frozen pin")
    actual_runner_sha256 = _sha256_file(runner_path)
    expected_identity = {
        "config_sha256": T11_CONFIG_SHA256,
        "repo_commit": args.expected_repo_commit.lower(),
        "runner_sha256": args.expected_runner_sha256.lower(),
        "sana_commit": EXPECTED_SANA_COMMIT,
    }
    actual_identity = {
        "config_sha256": actual_config_sha256,
        "repo_commit": actual_commit,
        "runner_sha256": actual_runner_sha256,
        "sana_commit": EXPECTED_SANA_COMMIT,
    }
    if actual_identity != expected_identity:
        raise RuntimeError(
            "T11 source identity differs: "
            f"expected={expected_identity!r} actual={actual_identity!r}"
        )
    actual_identity["sana_commit"] = _assert_source_tree_clean(actual_commit)

    cfg = OmegaConf.load(config_path)
    if cfg.model.architecture.variant != "autoregressive":
        raise RuntimeError("T11 requires the AR architecture")
    if cfg.model.video_backbone.continuous_timestep_conditioning is not True:
        raise RuntimeError("T11 requires continuous FP32 video timesteps")
    if cfg.training.use_gradient_checkpointing is not True:
        raise RuntimeError("T11 requires gradient checkpointing")
    if int(cfg.training.seed) != args.initialization_seed:
        raise RuntimeError("T11 CLI initialization seed must equal training.seed")
    if tuple(cfg.training.trainable_modules) != EXPECTED_TRAINABLE_ROOTS:
        raise RuntimeError("T11 trainable module allowlist differs")
    if (
        tuple(cfg.training.preserve_frozen_input_grad_modules)
        != EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES
    ):
        raise RuntimeError("T11 frozen input-gradient preservation differs")
    if cfg.training.optimizer_master_weights is not True:
        raise RuntimeError("T11 requires persistent FP32 optimizer masters")
    if (
        float(cfg.training.lambda_video) != 0.0
        or float(cfg.training.lambda_action) != 1.0
    ):
        raise RuntimeError("T11 requires action-only loss weighting")
    if float(cfg.training.weight_decay) != 0.0:
        raise RuntimeError("T11 requires weight_decay=0")
    if (
        float(cfg.training.action_lr) != 1.0e-4
        or float(cfg.training.video_lr) != 1.0e-4
    ):
        raise RuntimeError("T11 requires fixed base learning rates of 1e-4")
    if float(cfg.training.grad_clip) != 1.0:
        raise RuntimeError("T11 requires the fixed gradient clip bound 1.0")
    forbidden_checkpoint_fields = (
        "training.init_checkpoint",
        "training.resume_checkpoint",
        "training.resume_from_checkpoint",
        "training.resume_manifest",
        "model.video_backbone.init_dit_from",
    )
    for path in forbidden_checkpoint_fields:
        if OmegaConf.select(cfg, path, default=None) is not None:
            raise RuntimeError(f"T11 forbids checkpoint field {path}")
    if str(cfg.training.action_stats_sha256) != SELECTED_STATS_SHA256:
        raise RuntimeError("T11 selected-row stats SHA pin differs")
    if str(cfg.training.action_stats_population_sha256) != SELECTED_POPULATION_SHA256:
        raise RuntimeError("T11 selected population SHA pin differs")
    stats_path = Path(str(cfg.dataloader.action_stats_path)).expanduser()
    if stats_path.is_symlink() or not stats_path.is_file():
        raise RuntimeError("T11 stats artifact must be a regular non-symlink file")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T11 selected-row stats artifact SHA differs")

    from sana_wam.train.libero_contract import validate_libero_training_config

    # This is the only full numeric live-source preflight. It runs before any
    # model construction or CUDA allocation.
    validate_libero_training_config(cfg, require_materialized_stats=True)
    external_assets = _verify_external_assets(cfg)
    configured_roots = [
        Path(str(path)).expanduser() for path in cfg.dataloader.dataset_roots
    ]
    dataset_roots = {path.name: path for path in configured_roots}
    expected_datasets = set(T11_SUITE_ORDER)
    if (
        len(dataset_roots) != len(configured_roots)
        or set(dataset_roots) != expected_datasets
    ):
        raise RuntimeError("T11 four-suite dataset root identity differs")
    spatial_root = dataset_roots[SPATIAL_NAME]
    eligible_sample_manifest = _eligible_sample_manifest(dataset_roots)
    selection_manifest = _selection_manifest(eligible_sample_manifest)
    external_assets["spatial_sample/meta/stats_gr00t.json"] = _verify_pinned_asset(
        spatial_root / "meta/stats_gr00t.json",
        SPATIAL_STATS_GR00T_SHA256,
    )
    external_assets["t1_predecessor/RESULT.json"] = _verify_pinned_asset(
        T1_PREDECESSOR_RESULT_PATH,
        T1_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t2_predecessor/RESULT.json"] = _verify_pinned_asset(
        T2_PREDECESSOR_RESULT_PATH,
        T2_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t2_predecessor/runner.py"] = _verify_pinned_asset(
        T2_PREDECESSOR_RUNNER_PATH,
        T2_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t3_predecessor/RESULT.json"] = _verify_pinned_asset(
        T3_PREDECESSOR_RESULT_PATH,
        T3_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t3_predecessor/runner.py"] = _verify_pinned_asset(
        T3_PREDECESSOR_RUNNER_PATH,
        T3_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t4_predecessor/RESULT.json"] = _verify_pinned_asset(
        T4_PREDECESSOR_RESULT_PATH,
        T4_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t4_predecessor/runner.py"] = _verify_pinned_asset(
        T4_PREDECESSOR_RUNNER_PATH,
        T4_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t5_predecessor/RESULT.json"] = _verify_pinned_asset(
        T5_PREDECESSOR_RESULT_PATH,
        T5_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t5_predecessor/runner.py"] = _verify_pinned_asset(
        T5_PREDECESSOR_RUNNER_PATH,
        T5_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t6_predecessor/RESULT.json"] = _verify_pinned_asset(
        T6_PREDECESSOR_RESULT_PATH,
        T6_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t6_predecessor/runner.py"] = _verify_pinned_asset(
        T6_PREDECESSOR_RUNNER_PATH,
        T6_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t7_predecessor/RESULT.json"] = _verify_pinned_asset(
        T7_PREDECESSOR_RESULT_PATH,
        T7_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t7_predecessor/runner.py"] = _verify_pinned_asset(
        T7_PREDECESSOR_RUNNER_PATH,
        T7_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t9_aggregate_predecessor/RESULT.json"] = _verify_pinned_asset(
        T9_AGGREGATE_RESULT_PATH,
        T9_AGGREGATE_RESULT_SHA256,
    )
    external_assets["t9_aggregate_predecessor/runner.py"] = _verify_pinned_asset(
        T9_AGGREGATE_RUNNER_PATH,
        T9_AGGREGATE_RUNNER_SHA256,
    )
    external_assets["t10_arm_predecessor/runner.py"] = _verify_pinned_asset(
        T10_ARM_RUNNER_PATH,
        T10_ARM_RUNNER_SHA256,
    )
    external_assets["t10_seq_predecessor/RESULT.json"] = _verify_pinned_asset(
        T10_SEQ_RESULT_PATH,
        T10_SEQ_RESULT_SHA256,
    )
    external_assets["t10_joint_predecessor/RESULT.json"] = _verify_pinned_asset(
        T10_JOINT_RESULT_PATH,
        T10_JOINT_RESULT_SHA256,
    )
    external_assets["t10_aggregate_predecessor/runner.py"] = _verify_pinned_asset(
        T10_AGGREGATE_RUNNER_PATH,
        T10_AGGREGATE_RUNNER_SHA256,
    )
    external_assets["t10_aggregate_predecessor/RESULT.json"] = _verify_pinned_asset(
        T10_AGGREGATE_RESULT_PATH,
        T10_AGGREGATE_RESULT_SHA256,
    )
    for dataset in T11_SUITE_ORDER:
        for filename, expected_sha256 in sorted(
            T11_SUITE_METADATA_SHA256[dataset].items()
        ):
            external_assets[f"cross_suite_metadata/{dataset}/meta/{filename}"] = (
                _verify_pinned_asset(
                    dataset_roots[dataset] / "meta" / filename,
                    expected_sha256,
                )
            )
    for role, samples in (
        ("update", T11_TRAIN_SAMPLES),
        ("heldout", T11_HELDOUT_SAMPLES),
    ):
        for selected in samples:
            dataset = selected["dataset"]
            label = selected["label"]
            for relative_path, expected_sha256 in sorted(selected["assets"].items()):
                external_assets[f"{role}_sample/{dataset}/{label}/{relative_path}"] = (
                    _verify_pinned_asset(
                        dataset_roots[dataset] / relative_path,
                        expected_sha256,
                    )
                )
    t2_predecessor = json.loads(T2_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t2_predecessor.get("valid_run") is not True
        or t2_predecessor.get("scientific_verdict") != "T2_FIXED_SAMPLE_LEARNABILITY_GO"
        or t2_predecessor.get("identity", {}).get("repo_commit")
        != T2_PREDECESSOR_SOURCE_COMMIT
        or t2_predecessor.get("identity", {}).get("runner_sha256")
        != T2_PREDECESSOR_RUNNER_SHA256
    ):
        raise RuntimeError("T11 predecessor T2 evidence semantics differ")
    t3_predecessor = json.loads(T3_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t3_predecessor.get("valid_run") is not True
        or t3_predecessor.get("scientific_verdict") != "T3_HELDOUT_RECIPE_TRANSFER_GO"
        or t3_predecessor.get("identity", {}).get("repo_commit")
        != T3_PREDECESSOR_SOURCE_COMMIT
        or t3_predecessor.get("identity", {}).get("runner_sha256")
        != T3_PREDECESSOR_RUNNER_SHA256
        or t3_predecessor.get("scope", {}).get("architecture_forwards_total") != 28
        or t3_predecessor.get("scope", {}).get("backward_calls") != T11_UPDATE_STEPS
        or t3_predecessor.get("scope", {}).get("optimizer_steps") != T11_UPDATE_STEPS
    ):
        raise RuntimeError("T11 predecessor T3 evidence semantics differ")
    t3_training_core = _training_core_projection(t3_predecessor)
    t4_predecessor = json.loads(T4_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t4_predecessor.get("valid_run") is not True
        or t4_predecessor.get("scientific_verdict") != "T4_HELDOUT_SAMPLE_TRANSFER_GO"
        or t4_predecessor.get("identity", {}).get("repo_commit")
        != T4_PREDECESSOR_SOURCE_COMMIT
        or t4_predecessor.get("identity", {}).get("runner_sha256")
        != T4_PREDECESSOR_RUNNER_SHA256
        or t4_predecessor.get("scope", {}).get("architecture_forwards_total") != 28
        or t4_predecessor.get("scope", {}).get("backward_calls") != T11_UPDATE_STEPS
        or t4_predecessor.get("scope", {}).get("optimizer_steps") != T11_UPDATE_STEPS
    ):
        raise RuntimeError("T11 predecessor T4 evidence semantics differ")
    t4_training_core = _training_core_projection(t4_predecessor)
    t5_predecessor = json.loads(T5_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t5_predecessor.get("valid_run") is not True
        or t5_predecessor.get("scientific_verdict") != "T5_CROSS_TASK_TRANSFER_GO"
        or t5_predecessor.get("identity", {}).get("repo_commit")
        != T5_PREDECESSOR_SOURCE_COMMIT
        or t5_predecessor.get("identity", {}).get("runner_sha256")
        != T5_PREDECESSOR_RUNNER_SHA256
        or t5_predecessor.get("scope", {}).get("architecture_forwards_total") != 28
        or t5_predecessor.get("scope", {}).get("backward_calls") != T11_UPDATE_STEPS
        or t5_predecessor.get("scope", {}).get("optimizer_steps") != T11_UPDATE_STEPS
    ):
        raise RuntimeError("T11 predecessor T5 evidence semantics differ")
    t5_training_core = _training_core_projection(t5_predecessor)
    t6_predecessor = json.loads(T6_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t6_predecessor.get("valid_run") is not True
        or t6_predecessor.get("scientific_verdict") != "T6_CROSS_SUITE_TRANSFER_GO"
        or t6_predecessor.get("identity", {}).get("repo_commit")
        != T6_PREDECESSOR_SOURCE_COMMIT
        or t6_predecessor.get("identity", {}).get("runner_sha256")
        != T6_PREDECESSOR_RUNNER_SHA256
        or t6_predecessor.get("scope", {}).get("architecture_forwards_total") != 28
        or t6_predecessor.get("scope", {}).get("backward_calls") != T11_UPDATE_STEPS
        or t6_predecessor.get("scope", {}).get("optimizer_steps") != T11_UPDATE_STEPS
    ):
        raise RuntimeError("T11 predecessor T6 evidence semantics differ")
    t6_training_core = _training_core_projection(t6_predecessor)
    if (
        t3_training_core != t4_training_core
        or t4_training_core != t5_training_core
        or t5_training_core != t6_training_core
        or _sha256_json(t3_training_core) != T6_LINEAGE_TRAINING_CORE_SHA256
        or _sha256_json(t4_training_core) != T6_LINEAGE_TRAINING_CORE_SHA256
        or _sha256_json(t5_training_core) != T6_LINEAGE_TRAINING_CORE_SHA256
        or _sha256_json(t6_training_core) != T6_LINEAGE_TRAINING_CORE_SHA256
    ):
        raise RuntimeError("T11 frozen T3/T4/T5/T6 training-core evidence differs")
    t7_predecessor = json.loads(T7_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    t7_pre_losses = {
        row["label"]: row["before_action_loss"]
        for section in ("cyclic_training", "fresh_heldout_transfer")
        for row in t7_predecessor[section]["per_sample"]
    }
    t7_terminal_losses = {
        row["label"]: row["after_action_loss"]
        for section in ("cyclic_training", "fresh_heldout_transfer")
        for row in t7_predecessor[section]["per_sample"]
    }
    if (
        t7_predecessor.get("valid_run") is not True
        or t7_predecessor.get("scientific_verdict") != "T7_FOUR_SUITE_CYCLIC_GO"
        or t7_predecessor.get("identity", {}).get("repo_commit")
        != T7_PREDECESSOR_SOURCE_COMMIT
        or t7_predecessor.get("identity", {}).get("runner_sha256")
        != T7_PREDECESSOR_RUNNER_SHA256
        or t7_predecessor.get("identity", {}).get("config_sha256") != T11_CONFIG_SHA256
        or t7_predecessor.get("scope", {}).get("architecture_forwards_total") != 36
        or t7_predecessor.get("scope", {}).get("backward_calls") != 20
        or t7_predecessor.get("scope", {}).get("optimizer_steps") != 20
        or t7_pre_losses != {**T7_PRE_ACTION_LOSSES, **T7_PRE_HELDOUT_ACTION_LOSSES}
        or t7_terminal_losses != T7_TERMINAL_ACTION_LOSSES
    ):
        raise RuntimeError("T11 predecessor T7 evidence semantics differ")
    t9_aggregate = json.loads(T9_AGGREGATE_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t9_aggregate.get("valid_run") is not True
        or t9_aggregate.get("scientific_verdict")
        != "T9_COMMON_POSITION_EFFECT_SUPPORTED"
        or t9_aggregate.get("identity", {}).get("repo_commit")
        != T9_AGGREGATE_SOURCE_COMMIT
        or t9_aggregate.get("identity", {}).get("runner_sha256")
        != T9_AGGREGATE_RUNNER_SHA256
        or t9_aggregate.get("identity", {}).get("config_sha256") != T11_CONFIG_SHA256
        or t9_aggregate.get("run", {}).get("root") != str(T9_AGGREGATE_ROOT)
        or t9_aggregate.get("scope", {}).get("input_result_count") != 4
        or t9_aggregate.get("scope", {}).get("gpu_or_model_executed") is not False
    ):
        raise RuntimeError("T11 direct predecessor T9 aggregate semantics differ")
    t10_arm_results = {
        "SEQ": json.loads(T10_SEQ_RESULT_PATH.read_text(encoding="utf-8")),
        "JOINT": json.loads(T10_JOINT_RESULT_PATH.read_text(encoding="utf-8")),
    }
    expected_t10_arm_roots = {
        "SEQ": str(T10_SEQ_RESULT_PATH.parent),
        "JOINT": str(T10_JOINT_RESULT_PATH.parent),
    }
    expected_t10_arm_verdicts = {
        "SEQ": "T10_SEQ_BRIDGE_VALID",
        "JOINT": "T10_JOINT_ARM_VALID",
    }
    for arm, predecessor in t10_arm_results.items():
        if (
            predecessor.get("valid_run") is not True
            or predecessor.get("verdict") != expected_t10_arm_verdicts[arm]
            or predecessor.get("identity", {}).get("repo_commit")
            != T10_ARM_SOURCE_COMMIT
            or predecessor.get("identity", {}).get("runner_sha256")
            != T10_ARM_RUNNER_SHA256
            or predecessor.get("identity", {}).get("config_sha256") != T11_CONFIG_SHA256
            or predecessor.get("run", {}).get("arm") != arm
            or predecessor.get("run", {}).get("root") != expected_t10_arm_roots[arm]
            or predecessor.get("scope", {}).get("architecture_forwards_total") != 96
            or predecessor.get("scope", {}).get("architecture_training_forwards") != 80
            or predecessor.get("scope", {}).get("backward_calls") != 80
            or predecessor.get("scope", {}).get("optimizer_steps") != 20
            or predecessor.get("scope", {}).get("sana_wam_training_checkpoint_loaded")
            is not False
            or predecessor.get("scope", {}).get("sana_wam_training_checkpoint_saved")
            is not False
        ):
            raise RuntimeError(f"T11 predecessor T10 {arm} arm semantics differ")
    t10_aggregate = json.loads(T10_AGGREGATE_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t10_aggregate.get("valid_run") is not True
        or t10_aggregate.get("scientific_verdict")
        != "T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE"
        or t10_aggregate.get("identity", {}).get("repo_commit")
        != T10_AGGREGATE_SOURCE_COMMIT
        or t10_aggregate.get("identity", {}).get("runner_sha256")
        != T10_AGGREGATE_RUNNER_SHA256
        or t10_aggregate.get("identity", {}).get("t10_arm_source_commit")
        != T10_ARM_SOURCE_COMMIT
        or t10_aggregate.get("identity", {}).get("t10_arm_runner_sha256")
        != T10_ARM_RUNNER_SHA256
        or t10_aggregate.get("identity", {}).get("config_sha256") != T11_CONFIG_SHA256
        or t10_aggregate.get("run", {}).get("root") != str(T10_AGGREGATE_ROOT)
        or t10_aggregate.get("scope", {}).get("input_result_count") != 2
        or t10_aggregate.get("scope", {}).get("gpu_or_model_executed") is not False
        or t10_aggregate.get("arm_identities", {}).get("SEQ", {}).get("result_sha256")
        != T10_SEQ_RESULT_SHA256
        or t10_aggregate.get("arm_identities", {}).get("JOINT", {}).get("result_sha256")
        != T10_JOINT_RESULT_SHA256
    ):
        raise RuntimeError("T11 direct predecessor T10 aggregate semantics differ")
    external_assets = dict(sorted(external_assets.items()))
    actual_identity.update(
        {
            "external_assets_sha256": _sha256_json(external_assets),
            "stats_population_sha256": SELECTED_POPULATION_SHA256,
            "stats_sha256": SELECTED_STATS_SHA256,
            "t1_predecessor_result_sha256": T1_PREDECESSOR_RESULT_SHA256,
            "t2_predecessor_result_sha256": T2_PREDECESSOR_RESULT_SHA256,
            "t2_predecessor_runner_sha256": T2_PREDECESSOR_RUNNER_SHA256,
            "t2_predecessor_source_commit": T2_PREDECESSOR_SOURCE_COMMIT,
            "t3_predecessor_result_sha256": T3_PREDECESSOR_RESULT_SHA256,
            "t3_predecessor_runner_sha256": T3_PREDECESSOR_RUNNER_SHA256,
            "t3_predecessor_source_commit": T3_PREDECESSOR_SOURCE_COMMIT,
            "t4_predecessor_result_sha256": T4_PREDECESSOR_RESULT_SHA256,
            "t4_predecessor_runner_sha256": T4_PREDECESSOR_RUNNER_SHA256,
            "t4_predecessor_source_commit": T4_PREDECESSOR_SOURCE_COMMIT,
            "t5_predecessor_result_sha256": T5_PREDECESSOR_RESULT_SHA256,
            "t5_predecessor_runner_sha256": T5_PREDECESSOR_RUNNER_SHA256,
            "t5_predecessor_source_commit": T5_PREDECESSOR_SOURCE_COMMIT,
            "t6_predecessor_result_sha256": T6_PREDECESSOR_RESULT_SHA256,
            "t6_predecessor_runner_sha256": T6_PREDECESSOR_RUNNER_SHA256,
            "t6_predecessor_source_commit": T6_PREDECESSOR_SOURCE_COMMIT,
            "t7_predecessor_result_sha256": T7_PREDECESSOR_RESULT_SHA256,
            "t7_predecessor_runner_sha256": T7_PREDECESSOR_RUNNER_SHA256,
            "t7_predecessor_source_commit": T7_PREDECESSOR_SOURCE_COMMIT,
            "t9_aggregate_predecessor_result_sha256": T9_AGGREGATE_RESULT_SHA256,
            "t9_aggregate_predecessor_runner_sha256": T9_AGGREGATE_RUNNER_SHA256,
            "t9_aggregate_predecessor_source_commit": T9_AGGREGATE_SOURCE_COMMIT,
            "t10_arm_predecessor_runner_sha256": T10_ARM_RUNNER_SHA256,
            "t10_arm_predecessor_source_commit": T10_ARM_SOURCE_COMMIT,
            "t10_seq_predecessor_result_sha256": T10_SEQ_RESULT_SHA256,
            "t10_joint_predecessor_result_sha256": T10_JOINT_RESULT_SHA256,
            "t10_aggregate_predecessor_result_sha256": (T10_AGGREGATE_RESULT_SHA256),
            "t10_aggregate_predecessor_runner_sha256": (T10_AGGREGATE_RUNNER_SHA256),
            "t10_aggregate_predecessor_source_commit": (T10_AGGREGATE_SOURCE_COMMIT),
            "eligible_manifest_sha256": T11_ELIGIBLE_MANIFEST_SHA256,
            "selection_manifest_sha256": T11_SELECTION_MANIFEST_SHA256,
        }
    )

    import fcntl

    lock_path = Path(f"/tmp/sana-wam-{args.expected_gpu_uuid}.lock")
    gpu_lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"selected GPU lock is already held: {lock_path}") from exc
    gpu_before = _assert_idle_gpu(args.physical_gpu, args.expected_gpu_uuid)
    if "H200" not in gpu_before["name"].upper():
        raise RuntimeError(f"T11 selected GPU is not an H200: {gpu_before['name']}")

    import torch

    from sana_wam.dataloader.transforms.multiview import format_prompt_for_inference
    from sana_wam.train.trainer import Trainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "T11 requires exactly one visible CUDA GPU; "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.initialization_seed)
    np.random.seed(args.initialization_seed % (2**32))
    torch.manual_seed(args.initialization_seed)
    torch.cuda.manual_seed_all(args.initialization_seed)

    warning_messages: list[str] = []

    class WarningCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warning_messages.append(record.getMessage())

    warning_capture = WarningCapture(level=logging.WARNING)
    builder_logger = logging.getLogger(
        "sana_wam.model.video_backbone.sana.pipeline_builder"
    )
    builder_logger.addHandler(warning_capture)
    gpu_preconstruction = _assert_idle_gpu(args.physical_gpu, args.expected_gpu_uuid)
    torch.cuda.reset_peak_memory_stats(device)
    build_started = time.perf_counter()
    try:
        trainer = Trainer(cfg)
    finally:
        builder_logger.removeHandler(warning_capture)
    torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - build_started
    build_memory = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }
    partial_load_warnings = [
        message
        for message in warning_messages
        if "ckpt load partial" in message.lower()
    ]
    if not (
        len(partial_load_warnings) == 1
        and partial_load_warnings[0].startswith(EXPECTED_BASE_PARTIAL_LOAD_PREFIX)
        and all(
            sample in partial_load_warnings[0]
            for sample in EXPECTED_BASE_PARTIAL_LOAD_SAMPLES
        )
    ):
        raise RuntimeError(
            "SANA base checkpoint partial-load identity differs from the known "
            f"fresh AR additions: {warning_messages!r}"
        )

    architecture = trainer.architecture
    video_backbone = architecture.video_backbone
    if any(parameter.requires_grad for parameter in video_backbone.parameters()):
        raise RuntimeError("T11 video-backbone parameters are not frozen")
    named_trainables = [
        (name, parameter)
        for name, parameter in architecture.named_parameters()
        if parameter.requires_grad
    ]
    named_frozen = [
        (name, parameter)
        for name, parameter in architecture.named_parameters()
        if not parameter.requires_grad
    ]
    observed_roots = tuple(
        sorted({_parameter_root(name) for name, _parameter in named_trainables})
    )
    if observed_roots != tuple(sorted(EXPECTED_TRAINABLE_ROOTS)):
        raise RuntimeError(f"T11 trainable parameter roots differ: {observed_roots!r}")
    if len(named_trainables) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        raise RuntimeError("T11 trainable tensor count differs")
    if sum(parameter.numel() for _name, parameter in named_trainables) != (
        EXPECTED_TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("T11 trainable parameter count differs")
    frozen_versions = {name: parameter._version for name, parameter in named_frozen}

    role_selections = _select_t11_samples(
        trainer.dataset,
        eligible_sample_manifest,
        selection_manifest,
    )
    raw_samples: dict[str, dict[str, Any]] = {}
    sample_seconds: dict[str, float] = {}
    role_assets_after_sample: dict[str, str] = {}
    role_order = (*T11_TRAIN_LABELS, *T11_HELDOUT_LABELS)
    for label in role_order:
        selected = role_selections[label]
        episode = selected["episode"]
        sample_started = time.perf_counter()
        raw_sample = trainer.dataset[selected["dataset_index"]]
        sample_seconds[label] = time.perf_counter() - sample_started
        if (
            episode.dataset != selected["dataset"]
            or episode.episode_index != selected["episode_index"]
            or episode.length != selected["episode_length"]
            or episode.task_index != selected["task_index"]
            or episode.task != selected["task"]
            or raw_sample["action_alignment"] != "observation_t_to_action_t"
            or raw_sample["dataset_name"] != selected["dataset"]
            or raw_sample["episode_index"] != selected["episode_index"]
            or raw_sample["episode_length"] != selected["episode_length"]
            or raw_sample["start_frame"] != selected["start_frame"]
            or raw_sample["task_index"] != selected["task_index"]
            or raw_sample["task_name"] != selected["task"]
            or raw_sample["prompt"] != format_prompt_for_inference(selected["task"])
        ):
            raise RuntimeError(f"T11 LIBERO sample metadata differs: {label}")
        raw_samples[label] = raw_sample
        role = "update" if label in T11_TRAIN_LABELS else "heldout"
        for relative_path, expected_sha256 in sorted(selected["assets"].items()):
            key = f"{role}_sample/{selected['dataset']}/{label}/{relative_path}"
            observed = _verify_pinned_asset(
                dataset_roots[selected["dataset"]] / relative_path,
                expected_sha256,
            )
            if external_assets[key] != observed:
                raise RuntimeError(f"T11 role asset changed while loading: {key}")
            role_assets_after_sample[key] = observed
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T11 selected-row stats changed while loading")
    actual_identity["role_assets_post_sample_sha256"] = _sha256_json(
        role_assets_after_sample
    )

    trainer._set_training_mode()
    if any(module.training for module in video_backbone.modules()):
        raise RuntimeError("T11 preserved video backbone left eval mode")
    prepare_inputs_calls = 0
    prepare_measurements: dict[str, dict[str, Any]] = {}

    def prepare_one(label: str, raw_sample: dict[str, Any]) -> dict[str, Any]:
        nonlocal prepare_inputs_calls
        torch.cuda.reset_peak_memory_stats(device)
        prepare_started = time.perf_counter()
        with torch.no_grad():
            prepared = architecture.prepare_inputs([raw_sample])
        prepare_inputs_calls += 1
        torch.cuda.synchronize(device)
        prepare_measurements[label] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "seconds": time.perf_counter() - prepare_started,
        }
        if prepared.get("use_gradient_checkpointing") is not True:
            raise RuntimeError(f"T11 {label} inputs dropped gradient checkpointing")
        if prepared.get("use_gradient_checkpointing_offload") is not False:
            raise RuntimeError("T11 requires checkpoint offload=false")
        invalid_tensors = [
            key
            for key, value in prepared.items()
            if isinstance(value, torch.Tensor)
            and (value.requires_grad or value.grad_fn is not None)
        ]
        if invalid_tensors:
            raise RuntimeError(
                f"T11 {label} prepared tensors retain autograd state: "
                f"{invalid_tensors!r}"
            )
        return prepared

    prepared_by_label = {
        label: prepare_one(label, raw_samples[label]) for label in role_order
    }
    cyclic_training_inputs = {
        label: prepared_by_label[label] for label in T11_TRAIN_LABELS
    }
    fresh_heldout_inputs = {
        label: prepared_by_label[label] for label in T11_HELDOUT_LABELS
    }
    metadata_after_sample = {}
    for dataset_name in T11_SUITE_ORDER:
        for filename, expected_sha256 in sorted(
            T11_SUITE_METADATA_SHA256[dataset_name].items()
        ):
            key = f"cross_suite_metadata/{dataset_name}/meta/{filename}"
            observed = _verify_pinned_asset(
                dataset_roots[dataset_name] / "meta" / filename,
                expected_sha256,
            )
            if external_assets[key] != observed:
                raise RuntimeError(f"T11 metadata changed while loading: {key}")
            metadata_after_sample[key] = observed
    actual_identity["cross_suite_metadata_post_sample_sha256"] = _sha256_json(
        metadata_after_sample
    )
    if prepare_inputs_calls != 8:
        raise RuntimeError(
            f"T11 prepare_inputs call count differs: {prepare_inputs_calls}"
        )
    eligible_manifest = eligible_sample_manifest
    live_manifest = _eligible_sample_manifest(dataset_roots)
    if live_manifest != eligible_manifest:
        raise RuntimeError(
            "T11 live eligible manifest changed during sample preparation"
        )

    observed_storage: dict[tuple[str, int], tuple[str, str]] = {}
    for label, prepared in prepared_by_label.items():
        for key, value in prepared.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue
            storage_key = (str(value.device), value.untyped_storage().data_ptr())
            previous = observed_storage.get(storage_key)
            if previous is not None and previous[0] != label:
                raise RuntimeError(
                    "T11 prepared tensors alias across samples: "
                    f"{previous!r} and {(label, key)!r}"
                )
            observed_storage[storage_key] = (label, key)

    def prepared_input_snapshot() -> list[tuple[Any, ...]]:
        return [
            (
                label,
                key,
                id(value),
                value.data_ptr(),
                value._version,
                str(value.dtype),
                tuple(value.shape),
            )
            for label, prepared in prepared_by_label.items()
            for key, value in sorted(prepared.items())
            if isinstance(value, torch.Tensor)
        ]

    initial_prepared_input_snapshot = prepared_input_snapshot()

    def prepared_context_report(
        prepared: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        context = prepared["context"]
        context_bytes = (
            context.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        )
        seq_lens = prepared.get("seq_lens")
        if isinstance(seq_lens, torch.Tensor):
            seq_lens_report: Any = [
                int(value) for value in seq_lens.detach().cpu().reshape(-1).tolist()
            ]
        elif isinstance(seq_lens, (list, tuple)):
            seq_lens_report = [int(value) for value in seq_lens]
        elif seq_lens is None:
            seq_lens_report = None
        else:
            seq_lens_report = int(seq_lens)
        return {
            "context_dtype": str(context.dtype),
            "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
            "context_shape": list(context.shape),
            "prompt": prompt,
            "seq_lens": seq_lens_report,
        }

    contexts_by_label = {}
    for label in role_order:
        selected = role_selections[label]
        contexts_by_label[label] = prepared_context_report(
            prepared_by_label[label], raw_samples[label]["prompt"]
        )
        contexts_by_label[label].update(
            {
                "dataset": selected["dataset"],
                "episode_index": selected["episode_index"],
                "task": selected["task"],
                "task_index": selected["task_index"],
            }
        )
    context_digests = {row["context_sha256"] for row in contexts_by_label.values()}
    paired_contexts_match = all(
        contexts_by_label[train_label]["context_sha256"]
        == contexts_by_label[heldout_label]["context_sha256"]
        for train_label, heldout_label in zip(
            T11_TRAIN_LABELS, T11_HELDOUT_LABELS, strict=True
        )
    )
    if (
        len(contexts_by_label) != 8
        or len(context_digests) != 4
        or not paired_contexts_match
    ):
        raise RuntimeError("T11 same-task context pairing differs")

    model_groups = trainer._param_groups()
    if len(model_groups) != 2 or any(
        float(group["lr"]) != 1.0e-4 for group in model_groups
    ):
        raise RuntimeError("T11 optimizer group/LR structure differs")
    params = [parameter for group in model_groups for parameter in group["params"]]
    parameter_ids = [id(parameter) for parameter in params]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {
        id(parameter) for _name, parameter in named_trainables
    }:
        raise RuntimeError("T11 optimizer parameters differ from trainables")
    optimizer_groups, master_pairs = trainer._optimizer_param_groups(model_groups)
    if len(master_pairs) != len(params):
        raise RuntimeError("T11 FP32 master coverage differs")
    if [id(model) for model, _master in master_pairs] != parameter_ids:
        raise RuntimeError("T11 FP32 master/model ordering differs")
    masters = [master for _model, master in master_pairs]
    initial_master_ids = [id(master) for master in masters]
    if (
        len(initial_master_ids) != len(set(initial_master_ids))
        or any(model.dtype != torch.bfloat16 for model, _master in master_pairs)
        or any(master.dtype != torch.float32 for _model, master in master_pairs)
        or any(not master.requires_grad for _model, master in master_pairs)
        or any(model.shape != master.shape for model, master in master_pairs)
        or any(model.device != master.device for model, master in master_pairs)
    ):
        raise RuntimeError("T11 FP32 master identity/dtype differs")
    if [
        id(parameter) for group in optimizer_groups for parameter in group["params"]
    ] != initial_master_ids:
        raise RuntimeError("T11 optimizer groups do not contain the FP32 masters")
    if [len(group["params"]) for group in optimizer_groups] != [
        len(group["params"]) for group in model_groups
    ] or [float(group["lr"]) for group in optimizer_groups] != [
        float(group["lr"]) for group in model_groups
    ]:
        raise RuntimeError("T11 FP32 master optimizer group structure differs")
    trainable_name_by_id = {id(parameter): name for name, parameter in named_trainables}
    named_masters = [
        (trainable_name_by_id[id(model)], master) for model, master in master_pairs
    ]
    model_by_name = dict(named_trainables)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(cfg.training.weight_decay),
        betas=(0.9, 0.95),
    )

    def macro_non_gradient_state_snapshot() -> dict[str, Any]:
        optimizer_state = []
        for name, master in named_masters:
            entries = []
            for key, value in sorted(optimizer.state.get(master, {}).items()):
                if isinstance(value, torch.Tensor):
                    observed = (
                        "tensor",
                        id(value),
                        value.data_ptr(),
                        value._version,
                        str(value.dtype),
                        tuple(value.shape),
                    )
                else:
                    observed = ("value", repr(value))
                entries.append((key, observed))
            optimizer_state.append((name, entries))
        return {
            "fp32_master_parameters": [
                (
                    name,
                    id(master),
                    master.data_ptr(),
                    master._version,
                    str(master.dtype),
                    tuple(master.shape),
                )
                for name, master in named_masters
            ],
            "model_parameters": [
                (
                    name,
                    id(parameter),
                    parameter.data_ptr(),
                    parameter._version,
                    str(parameter.dtype),
                    tuple(parameter.shape),
                )
                for name, parameter in architecture.named_parameters()
            ],
            "optimizer_state": optimizer_state,
        }

    def probe_mutation_snapshot() -> dict[str, Any]:
        if any(parameter.grad is not None for _name, parameter in named_trainables):
            raise RuntimeError("T11 update-free probe observed model gradients")
        if any(parameter.grad is not None for _name, parameter in named_frozen):
            raise RuntimeError("T11 update-free probe observed frozen gradients")
        if any(master.grad is not None for _name, master in named_masters):
            raise RuntimeError("T11 update-free probe observed master gradients")
        optimizer_state = []
        for name, master in named_masters:
            for key, value in sorted(optimizer.state.get(master, {}).items()):
                if isinstance(value, torch.Tensor):
                    observed = (
                        "tensor",
                        id(value),
                        value.data_ptr(),
                        value._version,
                    )
                else:
                    observed = ("value", repr(value))
                optimizer_state.append((name, key, observed))
        return {
            "architecture_buffers": [
                (name, id(buffer), buffer.data_ptr(), buffer._version)
                for name, buffer in architecture.named_buffers()
            ],
            "architecture_module_modes": [
                (name, module.training) for name, module in architecture.named_modules()
            ],
            "architecture_parameters": [
                (name, id(parameter), parameter.data_ptr(), parameter._version)
                for name, parameter in architecture.named_parameters()
            ],
            "master_parameters": [
                (name, id(master), master.data_ptr(), master._version)
                for name, master in named_masters
            ],
            "optimizer_state": optimizer_state,
            "prepared_inputs": prepared_input_snapshot(),
        }

    recipe_capture: dict[str, list[dict[str, Any]]] = {
        "action": [],
        "video": [],
    }
    architecture_forward_calls = 0
    backward_calls = 0
    optimizer_step_calls = 0
    video_embedder = video_backbone._dit.t_embedder
    original_video_embedder_forward = video_embedder.forward
    original_action_prepare_state = architecture.action_backbone.prepare_state
    original_architecture_forward = architecture.forward

    def capture_video_timestep(timestep):
        recipe_capture["video"].append(_tensor_observation(timestep))
        return original_video_embedder_forward(timestep)

    def capture_action_timestep(*positional, **keyword):
        if len(positional) >= 2 and isinstance(positional[1], torch.Tensor):
            recipe_capture["action"].append(_tensor_observation(positional[1]))
        token_timestep = keyword.get("token_timesteps")
        if isinstance(token_timestep, torch.Tensor):
            recipe_capture["action"].append(_tensor_observation(token_timestep))
        return original_action_prepare_state(*positional, **keyword)

    def capture_architecture_forward(*positional, **keyword):
        nonlocal architecture_forward_calls
        architecture_forward_calls += 1
        return original_architecture_forward(*positional, **keyword)

    video_embedder.forward = capture_video_timestep
    architecture.action_backbone.prepare_state = capture_action_timestep
    architecture.forward = capture_architecture_forward

    def fixed_forward(
        *,
        backward_scale: float | None,
        forward_inputs: dict[str, Any],
        recipe_seed: int,
    ) -> tuple[dict[str, float], str]:
        nonlocal backward_calls
        recipe_capture["action"].clear()
        recipe_capture["video"].clear()
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_cpu_rng_before = torch.random.get_rng_state().clone()
        torch_cuda_rng_before = torch.cuda.get_rng_state().clone()
        with _fixed_rng_recipe(
            torch,
            seed=recipe_seed,
            cuda_device=torch.cuda.current_device(),
        ):
            result = architecture.compute_loss(
                lambda_video=trainer.lambda_video,
                lambda_action=trainer.lambda_action,
                **forward_inputs,
            )
            loss = result["loss"]
            loss_action = result["loss_action"]
            if (
                loss.ndim != 0
                or loss_action.ndim != 0
                or not bool(torch.isfinite(loss).item())
                or not bool(torch.isfinite(loss_action).item())
                or bool((loss <= 0).item())
                or bool((loss_action <= 0).item())
            ):
                raise RuntimeError(
                    "T11 produced a scalar loss that is not finite and strictly "
                    "positive"
                )
            scalars = {
                "loss": _require_strictly_positive_finite(
                    _tensor_scalar(loss), field="forward total loss"
                ),
                "loss_action": _require_strictly_positive_finite(
                    _tensor_scalar(loss_action), field="forward action loss"
                ),
            }
            tolerance = 1.0e-6 + 1.0e-6 * abs(scalars["loss_action"])
            if abs(scalars["loss"] - scalars["loss_action"]) > tolerance:
                raise RuntimeError("T11 total/action loss identity differs")
            signature = _sha256_json(recipe_capture)
            if not recipe_capture["action"] or not recipe_capture["video"]:
                raise RuntimeError("T11 stochastic recipe capture is incomplete")
            if backward_scale is not None:
                if backward_scale not in {
                    0.0,
                    T11_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO,
                    1.0,
                }:
                    raise RuntimeError("T11 backward scale differs from matched core")
                (loss * backward_scale).backward()
                backward_calls += 1
        numpy_rng_after = np.random.get_state()
        if (
            random.getstate() != python_rng_before
            or numpy_rng_after[0] != numpy_rng_before[0]
            or not np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
            or numpy_rng_after[2:] != numpy_rng_before[2:]
            or not torch.equal(torch.random.get_rng_state(), torch_cpu_rng_before)
            or not torch.equal(torch.cuda.get_rng_state(), torch_cuda_rng_before)
        ):
            raise RuntimeError("T11 fixed recipe did not restore caller RNG state")
        del loss, loss_action, result
        return scalars, signature

    per_step: list[dict[str, Any]] = []
    update_recipe_signatures: list[str] = []
    per_macro_recipe_signatures: list[dict[str, str]] = []
    per_macro_loss_weights: list[dict[str, float]] = []
    per_macro_no_mutation_evidence: list[dict[str, Any]] = []
    measurement_before: dict[str, dict[str, float]] = {}
    measurement_after: dict[str, dict[str, float]] = {}
    measurement_signatures: dict[str, list[str]] = {label: [] for label in role_order}
    update_exposure_counts = {label: 0 for label in role_order}
    effective_weight_counts = {label: 0.0 for label in role_order}
    projection_exact_steps: list[int] = []
    master_probes = None
    model_probes = None
    first_master_update = None
    first_projected_update = None
    update_sentinels: list[dict[str, Any]] = []
    update_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for parameter in params:
            parameter.grad = None
        optimizer.zero_grad(set_to_none=True)
        if optimizer.state:
            raise RuntimeError("T11 optimizer state exists before pre probes")
        cyclic_pre_state = probe_mutation_snapshot()
        for label in T11_TRAIN_LABELS:
            scalars, signature = fixed_forward(
                backward_scale=None,
                forward_inputs=cyclic_training_inputs[label],
                recipe_seed=args.loss_recipe_seed,
            )
            measurement_before[label] = scalars
            measurement_signatures[label].append(signature)
        if probe_mutation_snapshot() != cyclic_pre_state or optimizer.state:
            raise RuntimeError("T11 cyclic training pre probes mutated training state")
        fresh_pre_state = probe_mutation_snapshot()
        for label in T11_HELDOUT_LABELS:
            scalars, signature = fixed_forward(
                backward_scale=None,
                forward_inputs=fresh_heldout_inputs[label],
                recipe_seed=args.loss_recipe_seed,
            )
            measurement_before[label] = scalars
            measurement_signatures[label].append(signature)
        if probe_mutation_snapshot() != fresh_pre_state or optimizer.state:
            raise RuntimeError("T11 fresh held-out pre probes mutated training state")
        for step_index in range(T11_UPDATE_STEPS):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for parameter in params:
                parameter.grad = None
            macro_state_before_microbacks = macro_non_gradient_state_snapshot()
            step_scalars_by_label: dict[str, dict[str, float]] = {}
            step_recipe_by_label: dict[str, str] = {}
            step_loss_weights = _loss_weights_for_macro(step_index)
            if set(step_loss_weights) != set(T11_TRAIN_LABELS) or not math.isclose(
                sum(step_loss_weights.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "T11 per-macro loss weights are not exactly unit-sum"
                )
            for update_label in T11_TRAIN_LABELS:
                step_scalars, recipe_signature = fixed_forward(
                    backward_scale=step_loss_weights[update_label],
                    forward_inputs=cyclic_training_inputs[update_label],
                    recipe_seed=args.loss_recipe_seed,
                )
                step_scalars_by_label[update_label] = step_scalars
                step_recipe_by_label[update_label] = recipe_signature
                update_recipe_signatures.append(recipe_signature)
                update_exposure_counts[update_label] += 1
                effective_weight_counts[update_label] += step_loss_weights[update_label]
                if step_index == 0:
                    expected = measurement_before[update_label]["loss_action"]
                    tolerance = 1.0e-6 + 1.0e-6 * abs(expected)
                    if abs(step_scalars["loss_action"] - expected) > tolerance:
                        raise RuntimeError(
                            "T11 fixed initial loss recipe is not reproducible "
                            f"for {update_label}"
                        )
            per_macro_no_mutation_evidence.append(
                _no_intra_macro_mutation_evidence(
                    macro_state_before_microbacks,
                    macro_non_gradient_state_snapshot(),
                    macro_step=step_index + 1,
                )
            )
            per_macro_recipe_signatures.append(step_recipe_by_label)
            per_macro_loss_weights.append(step_loss_weights)
            joint_loss = sum(
                step_scalars_by_label[label]["loss"] * step_loss_weights[label]
                for label in T11_TRAIN_LABELS
            )
            joint_action_loss = sum(
                step_scalars_by_label[label]["loss_action"] * step_loss_weights[label]
                for label in T11_TRAIN_LABELS
            )
            joint_tolerance = 1.0e-6 + 1.0e-6 * abs(joint_action_loss)
            if abs(joint_loss - joint_action_loss) > joint_tolerance:
                raise RuntimeError("T11 joint total/action loss identity differs")

            missing_gradients = []
            nonzero_gradient_tensors = 0
            nonzero_gradient_roots: set[str] = set()
            max_gradient_abs = 0.0
            for name, parameter in named_trainables:
                gradient = parameter.grad
                if gradient is None:
                    missing_gradients.append(name)
                    continue
                if not bool(torch.isfinite(gradient).all().item()):
                    raise RuntimeError(
                        f"T11 non-finite gradient at step {step_index + 1}: {name}"
                    )
                maximum = float(gradient.detach().abs().max().float().item())
                max_gradient_abs = max(max_gradient_abs, maximum)
                if maximum > 0.0:
                    nonzero_gradient_tensors += 1
                    nonzero_gradient_roots.add(_parameter_root(name))
            if missing_gradients:
                raise RuntimeError(
                    f"T11 missing gradients at step {step_index + 1}: "
                    f"{missing_gradients[:8]!r}"
                )
            if nonzero_gradient_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    f"T11 nonzero-gradient roots differ at step {step_index + 1}: "
                    f"{sorted(nonzero_gradient_roots)!r}"
                )
            if any(parameter.grad is not None for _name, parameter in named_frozen):
                raise RuntimeError("T11 frozen parameters received gradients")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, float(cfg.training.grad_clip)
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("T11 gradient norm is non-finite")
            trainer._sync_master_gradients(master_pairs)
            if step_index == 0:
                invalid_master_gradients = [
                    name
                    for name, master in named_masters
                    if master.grad is None
                    or master.grad.dtype != torch.float32
                    or not bool(torch.isfinite(master.grad).all().item())
                    or not bool(torch.count_nonzero(master.grad).item())
                ]
                if invalid_master_gradients:
                    raise RuntimeError(
                        "T11 first-step masters lack finite nonzero FP32 gradients: "
                        f"{invalid_master_gradients[:8]!r}"
                    )
                master_probes = _capture_update_probes(named_masters)
                model_probes = _capture_update_probes(named_trainables)
                if {row["name"] for row in master_probes} != {
                    name for name, _master in named_masters
                }:
                    raise RuntimeError("T11 first-step master probe coverage differs")

                for root in EXPECTED_TRAINABLE_ROOTS:
                    candidates = sorted(
                        (
                            probe
                            for probe in master_probes
                            if _parameter_root(probe["name"]) == root
                        ),
                        key=lambda probe: abs(probe["gradient"]),
                        reverse=True,
                    )[:4]
                    if len(candidates) != 4:
                        raise RuntimeError(
                            f"T11 lacks four update sentinels for {root}"
                        )
                    update_sentinels.extend(candidates)

            sentinel_before = [
                {
                    "before_master": _tensor_scalar(
                        probe["parameter"].detach().reshape(-1)[probe["index"]]
                    ),
                    "before_model": _tensor_scalar(
                        model_by_name[probe["name"]]
                        .detach()
                        .reshape(-1)[probe["index"]]
                    ),
                    "gradient": _tensor_scalar(
                        probe["parameter"].grad.detach().reshape(-1)[probe["index"]]
                    ),
                    "index": probe["index"],
                    "name": probe["name"],
                }
                for probe in update_sentinels
            ]

            optimizer.step()
            optimizer_step_calls += 1
            if step_index == 0:
                first_master_update, changed_master_ids = _summarize_updates(
                    master_probes
                )
                if len(changed_master_ids) != len(named_masters):
                    raise RuntimeError(
                        "T11 first step did not update every FP32 master"
                    )
            trainer._copy_master_parameters_to_model(master_pairs)
            sentinel_updates = []
            for before, probe in zip(sentinel_before, update_sentinels, strict=True):
                after_master = _tensor_scalar(
                    probe["parameter"].detach().reshape(-1)[probe["index"]]
                )
                after_model = _tensor_scalar(
                    model_by_name[probe["name"]].detach().reshape(-1)[probe["index"]]
                )
                row = {
                    **before,
                    "after_master": after_master,
                    "after_model": after_model,
                    "master_delta": after_master - before["before_master"],
                    "model_delta": after_model - before["before_model"],
                    "root": _parameter_root(probe["name"]),
                }
                if not all(
                    math.isfinite(row[key])
                    for key in (
                        "after_master",
                        "after_model",
                        "master_delta",
                        "model_delta",
                    )
                ):
                    raise RuntimeError("T11 update sentinel became non-finite")
                sentinel_updates.append(row)
            changed_master_sentinels = [
                row for row in sentinel_updates if row["master_delta"] != 0.0
            ]
            changed_master_sentinel_roots = {
                row["root"] for row in changed_master_sentinels
            }
            if changed_master_sentinel_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    "T11 sampled FP32 master update roots differ at step "
                    f"{step_index + 1}: "
                    f"{sorted(changed_master_sentinel_roots)!r}"
                )
            sentinel_adam_steps = {
                _tensor_scalar(optimizer.state[probe["parameter"]]["step"])
                for probe in update_sentinels
            }
            if sentinel_adam_steps != {float(step_index + 1)}:
                raise RuntimeError(
                    "T11 sampled AdamW step counters differ at step "
                    f"{step_index + 1}: {sorted(sentinel_adam_steps)!r}"
                )
            projection_mismatches = [
                trainable_name_by_id[id(model)]
                for model, master in master_pairs
                if not torch.equal(
                    model.detach(), master.detach().to(dtype=model.dtype)
                )
            ]
            if projection_mismatches:
                raise RuntimeError(
                    f"T11 BF16 projection differs: {projection_mismatches[:8]!r}"
                )
            projection_exact_steps.append(step_index + 1)
            if not _optimizer_state_is_finite(optimizer):
                raise RuntimeError("T11 AdamW state is non-finite")
            if step_index == 0:
                first_projected_update, _changed_model_ids = _summarize_updates(
                    model_probes
                )
            torch.cuda.synchronize(device)
            per_step.append(
                {
                    "component_datasets": {
                        label: role_selections[label]["dataset"]
                        for label in T11_TRAIN_LABELS
                    },
                    "component_episode_indices": {
                        label: role_selections[label]["episode_index"]
                        for label in T11_TRAIN_LABELS
                    },
                    "component_losses": step_scalars_by_label,
                    "component_recipe_signatures": step_recipe_by_label,
                    "component_task_indices": {
                        label: role_selections[label]["task_index"]
                        for label in T11_TRAIN_LABELS
                    },
                    "loss_weights": step_loss_weights,
                    "gradient_nonzero_roots": sorted(nonzero_gradient_roots),
                    "gradient_nonzero_tensor_count": nonzero_gradient_tensors,
                    "labels": list(T11_TRAIN_LABELS),
                    "loss": joint_loss,
                    "loss_action": joint_action_loss,
                    "macro_step": step_index + 1,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "pre_clip_global_norm": _tensor_scalar(grad_norm),
                    "pre_clip_max_abs": max_gradient_abs,
                    "recipe_signature_sha256": _sha256_json(step_recipe_by_label),
                    "sampled_bf16_update_changed_count": sum(
                        row["model_delta"] != 0.0 for row in sentinel_updates
                    ),
                    "sampled_bf16_update_changed_roots": sorted(
                        {
                            row["root"]
                            for row in sentinel_updates
                            if row["model_delta"] != 0.0
                        }
                    ),
                    "sampled_bf16_update_l2": math.sqrt(
                        sum(row["model_delta"] ** 2 for row in sentinel_updates)
                    ),
                    "sampled_bf16_update_max_abs": max(
                        abs(row["model_delta"]) for row in sentinel_updates
                    ),
                    "sampled_master_update_changed_count": len(
                        changed_master_sentinels
                    ),
                    "sampled_master_update_changed_roots": sorted(
                        changed_master_sentinel_roots
                    ),
                    "sampled_master_update_l2": math.sqrt(
                        sum(row["master_delta"] ** 2 for row in sentinel_updates)
                    ),
                    "sampled_master_update_max_abs": max(
                        abs(row["master_delta"]) for row in sentinel_updates
                    ),
                    "sampled_optimizer_step_values": sorted(sentinel_adam_steps),
                    "seconds": time.perf_counter() - step_started,
                    "step": step_index + 1,
                }
            )

        optimizer.zero_grad(set_to_none=True)
        for parameter in params:
            parameter.grad = None
        cyclic_post_state = probe_mutation_snapshot()
        for label in T11_TRAIN_LABELS:
            scalars, signature = fixed_forward(
                backward_scale=None,
                forward_inputs=cyclic_training_inputs[label],
                recipe_seed=args.loss_recipe_seed,
            )
            measurement_after[label] = scalars
            measurement_signatures[label].append(signature)
        if probe_mutation_snapshot() != cyclic_post_state:
            raise RuntimeError("T11 cyclic training post probes mutated training state")
        fresh_post_state = probe_mutation_snapshot()
        for label in T11_HELDOUT_LABELS:
            scalars, signature = fixed_forward(
                backward_scale=None,
                forward_inputs=fresh_heldout_inputs[label],
                recipe_seed=args.loss_recipe_seed,
            )
            measurement_after[label] = scalars
            measurement_signatures[label].append(signature)
        if probe_mutation_snapshot() != fresh_post_state:
            raise RuntimeError("T11 fresh held-out post probes mutated training state")
    finally:
        video_embedder.forward = original_video_embedder_forward
        architecture.action_backbone.prepare_state = original_action_prepare_state
        architecture.forward = original_architecture_forward

    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    if architecture_forward_calls != 96:
        raise RuntimeError(
            f"T11 architecture forward count differs: {architecture_forward_calls}"
        )
    if backward_calls != 80:
        raise RuntimeError(f"T11 backward call count differs: {backward_calls}")
    if optimizer_step_calls != 20:
        raise RuntimeError(
            f"T11 optimizer step call count differs: {optimizer_step_calls}"
        )
    all_macro_states_unchanged = (
        len(per_macro_no_mutation_evidence) == T11_MACRO_STEPS
        and [row["macro_step"] for row in per_macro_no_mutation_evidence]
        == list(range(1, T11_MACRO_STEPS + 1))
        and all(
            row["state_unchanged"]
            and all(row["components_unchanged"].values())
            and row["pre_microback_state_sha256"]
            == row["post_microback_pre_optimizer_state_sha256"]
            for row in per_macro_no_mutation_evidence
        )
    )
    if not all_macro_states_unchanged:
        raise RuntimeError("T11 no-intra-macro-mutation evidence coverage differs")
    if (
        len(update_recipe_signatures) != T11_TRAINING_FORWARDS
        or len(set(update_recipe_signatures)) != 1
    ):
        raise RuntimeError("T11 update RNG recipe signatures differ across forwards")
    invalid_measurement_signatures = {
        label: signatures
        for label, signatures in measurement_signatures.items()
        if len(signatures) != 2 or len(set(signatures)) != 1
    }
    if invalid_measurement_signatures:
        raise RuntimeError(
            "T11 measurement RNG recipe signatures differ: "
            f"{invalid_measurement_signatures!r}"
        )
    all_recipe_signatures = update_recipe_signatures + [
        signature for label in role_order for signature in measurement_signatures[label]
    ]
    if len(per_macro_recipe_signatures) != T11_MACRO_STEPS or any(
        set(signatures) != set(T11_TRAIN_LABELS)
        for signatures in per_macro_recipe_signatures
    ):
        raise RuntimeError("T11 per-macro recipe coverage differs")
    if per_macro_loss_weights != [
        _loss_weights_for_macro(step_index) for step_index in range(T11_MACRO_STEPS)
    ]:
        raise RuntimeError("T11 frozen matched-core loss-weight schedule differs")
    if len(all_recipe_signatures) != 96 or set(all_recipe_signatures) != {
        T11_FIXED_RECIPE_SIGNATURE_SHA256
    }:
        raise RuntimeError("T11 fixed loss recipe signature differs across forwards")
    expected_exposures = {
        **{label: T11_FORWARD_EXPOSURES_PER_SAMPLE for label in T11_TRAIN_LABELS},
        **{label: 0 for label in T11_HELDOUT_LABELS},
    }
    if update_exposure_counts != expected_exposures:
        raise RuntimeError(
            f"T11 joint update exposure counts differ: {update_exposure_counts!r}"
        )
    if projection_exact_steps != list(range(1, T11_MACRO_STEPS + 1)):
        raise RuntimeError("T11 per-macro full projection evidence differs")
    expected_effective_weights = {
        **{label: T11_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE for label in T11_TRAIN_LABELS},
        **{label: 0.0 for label in T11_HELDOUT_LABELS},
    }
    if effective_weight_counts != expected_effective_weights:
        raise RuntimeError("T11 effective cumulative loss weights differ")
    final_optimizer_master_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if final_optimizer_master_ids != initial_master_ids:
        raise RuntimeError("T11 FP32 master objects changed during the loop")
    if prepared_input_snapshot() != initial_prepared_input_snapshot:
        raise RuntimeError("T11 prepared input identities or versions changed")
    frozen_version_changes = [
        name
        for name, parameter in named_frozen
        if parameter._version != frozen_versions[name]
    ]
    if frozen_version_changes:
        raise RuntimeError(
            f"T11 frozen parameters changed: {frozen_version_changes[:8]!r}"
        )
    if any(not bool(torch.isfinite(master).all().item()) for master in masters):
        raise RuntimeError("T11 FP32 masters became non-finite")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in params):
        raise RuntimeError("T11 BF16 trainables became non-finite")
    if not _optimizer_state_is_finite(optimizer):
        raise RuntimeError("T11 final AdamW state is non-finite")
    adam_steps: list[float] = []
    for master in masters:
        state = optimizer.state.get(master)
        if not state or "step" not in state:
            raise RuntimeError("T11 AdamW state is missing for an FP32 master")
        step_value = _tensor_scalar(state["step"])
        if not math.isfinite(step_value):
            raise RuntimeError("T11 AdamW step counter is non-finite")
        adam_steps.append(step_value)
    if set(adam_steps) != {float(T11_UPDATE_STEPS)}:
        raise RuntimeError(
            f"T11 AdamW step counters differ: {sorted(set(adam_steps))!r}"
        )

    cumulative_master_update, cumulative_master_ids = _summarize_updates(master_probes)
    cumulative_projected_update, _cumulative_model_ids = _summarize_updates(
        model_probes
    )
    if len(cumulative_master_ids) != len(named_masters) or set(
        cumulative_master_update["changed_trainable_roots"]
    ) != set(EXPECTED_TRAINABLE_ROOTS):
        raise RuntimeError("T11 cumulative FP32 master update coverage differs")

    same_task_new_episode = _same_task_new_episode_summary(
        {label: measurement_before[label]["loss_action"] for label in T11_TRAIN_LABELS},
        {label: measurement_after[label]["loss_action"] for label in T11_TRAIN_LABELS},
        {
            label: measurement_before[label]["loss_action"]
            for label in T11_HELDOUT_LABELS
        },
        {
            label: measurement_after[label]["loss_action"]
            for label in T11_HELDOUT_LABELS
        },
    )
    sample_by_label = {
        row["label"]: row for row in (*T11_TRAIN_SAMPLES, *T11_HELDOUT_SAMPLES)
    }

    def enrich_loss_summary(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            **summary,
            "per_sample": [
                {
                    **row,
                    "after_total_loss": measurement_after[row["label"]]["loss"],
                    "assets": sample_by_label[row["label"]]["assets"],
                    "before_total_loss": measurement_before[row["label"]]["loss"],
                    "episode_length": sample_by_label[row["label"]]["episode_length"],
                    "post_recipe_signature_sha256": measurement_signatures[
                        row["label"]
                    ][1],
                    "pre_recipe_signature_sha256": measurement_signatures[row["label"]][
                        0
                    ],
                    "task": sample_by_label[row["label"]]["task"],
                    "effective_cumulative_loss_weight": effective_weight_counts[
                        row["label"]
                    ],
                    "update_exposure_count": update_exposure_counts[row["label"]],
                }
                for row in summary["per_sample"]
            ],
        }

    same_task_new_episode_report = {
        **same_task_new_episode,
        "fresh_same_task_update_heldout": enrich_loss_summary(
            same_task_new_episode["fresh_same_task_update_heldout"]
        ),
        "loss_recipe_seed": args.loss_recipe_seed,
        "measurement_state_unchanged": True,
        "probe_state_unchanged": True,
        "macro_objective": {
            "arm": "JOINT",
            "labels": list(T11_TRAIN_LABELS),
            "loss": "equal_quarter_weighted_mean",
        },
        "effective_cumulative_loss_weights": {
            label: effective_weight_counts[label] for label in T11_TRAIN_LABELS
        },
        "update_exposure_counts": update_exposure_counts,
        "update_samples": enrich_loss_summary(same_task_new_episode["update_samples"]),
    }
    cyclic_training_report = same_task_new_episode_report["update_samples"]
    fresh_heldout_report = same_task_new_episode_report[
        "fresh_same_task_update_heldout"
    ]
    matched_core_schedule_report = {
        "arm": "JOINT",
        "backward_calls_per_macro_step": 4,
        "effective_cumulative_loss_weights": {
            label: effective_weight_counts[label] for label in T11_TRAIN_LABELS
        },
        "forward_order_within_macro_step": list(T11_TRAIN_LABELS),
        "forward_exposures_per_sample": T11_FORWARD_EXPOSURES_PER_SAMPLE,
        "macro_steps": T11_MACRO_STEPS,
        "no_intra_macro_mutation": {
            "all_macro_steps_verified": all_macro_states_unchanged,
            "evidence": per_macro_no_mutation_evidence,
            "evidence_scope": [
                "all_architecture_model_parameter_identity_and_version",
                "all_fp32_master_parameter_identity_and_version",
                "all_optimizer_state_identity_and_version",
            ],
            "macro_steps_verified": len(per_macro_no_mutation_evidence),
            "verification_point": (
                "after_four_microbacks_before_gradient_clip_master_sync_and_"
                "optimizer_step"
            ),
        },
        "optimizer_steps_per_macro_step": 1,
        "parameters_constant_across_four_component_backward_calls": (
            all_macro_states_unchanged
        ),
        "per_macro_loss_weights": [
            {"macro_step": index + 1, "weights": weights}
            for index, weights in enumerate(per_macro_loss_weights)
        ],
    }
    projected_roots = set(cumulative_projected_update["changed_trainable_roots"])
    secondary_diagnostic_warnings = []
    if projected_roots != set(EXPECTED_TRAINABLE_ROOTS):
        secondary_diagnostic_warnings.append(
            "20-step BF16 projection did not visibly change every trainable root; "
            "the persistent FP32-master update coverage remained complete"
        )
    execution_verdict = T11_EXECUTION_VERDICT
    scientific_verdict = same_task_new_episode["scientific_verdict"]
    verdict_reasons = [
        "the frozen T11 same-task/new-episode JOINT arm completed validly",
        f"the ordered unrounded-ratio classifier returned {scientific_verdict}",
    ]
    gpu_after = _query_gpu(args.physical_gpu)
    architecture_report = {
        "action_dim": architecture.action_dim,
        "class": type(architecture).__name__,
        "continuous_timestep_conditioning": True,
        "parameter_count": sum(
            parameter.numel() for parameter in architecture.parameters()
        ),
        "proprio_dim": architecture.proprio_dim,
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
        "trainable_parameter_tensor_count": EXPECTED_TRAINABLE_TENSOR_COUNT,
        "trainable_roots": list(EXPECTED_TRAINABLE_ROOTS),
        "video_backbone_frozen": True,
    }
    assets_report = {
        "config_path": str(config_path),
        "config_sha256": actual_config_sha256,
        "external_files": external_assets,
        "external_files_manifest_sha256": actual_identity["external_assets_sha256"],
        "role_files_post_sample": role_assets_after_sample,
        "role_files_post_sample_manifest_sha256": actual_identity[
            "role_assets_post_sample_sha256"
        ],
        "suite_metadata_post_sample": metadata_after_sample,
        "suite_metadata_post_sample_manifest_sha256": actual_identity[
            "cross_suite_metadata_post_sample_sha256"
        ],
        "stats_path": str(stats_path.resolve()),
        "stats_population_sha256": SELECTED_POPULATION_SHA256,
        "stats_sha256": SELECTED_STATS_SHA256,
        "stats_validation": {
            "formal_training_admission_granted": False,
            "live_numeric_pretrainer_validation_executed": True,
            "materialized_selected_row_v2_required": True,
        },
    }
    identity_report = {
        **actual_identity,
        "identity_sha256": _sha256_json(actual_identity),
    }

    def prepared_shape_report(prepared: dict[str, Any]) -> dict[str, Any]:
        return {
            "action_is_pad_true": int(prepared["action_is_pad"].sum().item()),
            "actions": list(prepared["actions"].shape),
            "context": list(prepared["context"].shape),
            "input_latents": list(prepared["input_latents"].shape),
            "proprio_seq": list(prepared["proprio_seq"].shape),
            "proprio_state": list(prepared["proprio_state"].shape),
            "video_is_pad_true": int(prepared["video_is_pad"].sum().item()),
        }

    inputs_by_label = {
        label: prepared_shape_report(prepared_by_label[label]) for label in role_order
    }
    optimizer_report = {
        "adam_step_values": sorted(set(adam_steps)),
        "betas": [0.9, 0.95],
        "constant_learning_rates": [
            float(group["lr"]) for group in optimizer.param_groups
        ],
        "cumulative_master_update": cumulative_master_update,
        "cumulative_projected_bf16_update": cumulative_projected_update,
        "first_master_update": first_master_update,
        "first_projected_bf16_update": first_projected_update,
        "master_object_identity_persistent": True,
        "optimizer_master_parameter_count": len(master_pairs),
        "optimizer_master_parameter_elements": sum(
            master.numel() for master in masters
        ),
        "optimizer_state_finite": True,
        "production_accumulation_equivalent": False,
        "production_schedule_equivalent": False,
        "projection_exact_after_steps": projection_exact_steps,
        "update_sentinel_count": len(update_sentinels),
        "update_sentinel_names": [
            {
                "index": probe["index"],
                "name": probe["name"],
                "root": _parameter_root(probe["name"]),
            }
            for probe in update_sentinels
        ],
        "weight_decay": float(cfg.training.weight_decay),
    }
    randomness_report = {
        "eligible_same_task_probe_manifest": eligible_sample_manifest,
        "eligible_same_task_probe_manifest_sha256": T11_ELIGIBLE_MANIFEST_SHA256,
        "initialization_seed": args.initialization_seed,
        "measurement_recipe_signatures_per_label": measurement_signatures,
        "per_macro_recipe_signatures": per_macro_recipe_signatures,
        "training_loss_recipe_seed": args.loss_recipe_seed,
        "recipe_reset": "python+numpy+torch-cpu+torch-cuda before every forward",
        "rng_state_restored_after_every_forward": True,
        "selected_same_task_probe_manifest": selection_manifest,
        "selected_same_task_probe_manifest_sha256": T11_SELECTION_MANIFEST_SHA256,
        "total_recipe_signatures": len(all_recipe_signatures),
        "training_recipe_signature_sha256": update_recipe_signatures[0],
        "training_recipe_signatures": len(update_recipe_signatures),
        "unique_loss_recipe_signature_count": len(set(all_recipe_signatures)),
        "unique_training_recipe_signatures": len(set(update_recipe_signatures)),
    }
    sample_roles_report = {
        label: {
            "action_alignment": raw_samples[label]["action_alignment"],
            "assets": sample_by_label[label]["assets"],
            "dataset": sample_by_label[label]["dataset"],
            "episode_index": sample_by_label[label]["episode_index"],
            "episode_length": sample_by_label[label]["episode_length"],
            "role": "update" if label in T11_TRAIN_LABELS else "fresh_same_task_probe",
            "start_frame": sample_by_label[label]["start_frame"],
            "task": sample_by_label[label]["task"],
            "task_index": sample_by_label[label]["task_index"],
            "effective_cumulative_loss_weight": effective_weight_counts[label],
            "update_exposure_count": update_exposure_counts[label],
        }
        for label in role_order
    }
    scope_report = {
        "architecture_forwards_total": 96,
        "architecture_measurement_forwards": 16,
        "architecture_training_forwards": 80,
        "backward_calls": backward_calls,
        "benchmark_evaluation_executed": False,
        "effective_cumulative_loss_weight_per_update_sample": 5.0,
        "formal_training_executed": False,
        "fresh_heldout_measurement_forwards": 8,
        "fresh_heldout_samples_in_backward_or_update": 0,
        "joint_training_measurement_forwards": 8,
        "macro_optimizer_steps": optimizer_step_calls,
        "raw_training_forward_exposures_per_sample": 20,
        "optimizer_steps": optimizer_step_calls,
        "post_probe_reprepare_calls": 0,
        "prepare_inputs_calls": prepare_inputs_calls,
        "probe_order": [
            "pre_A4_A5_A6_A7_H4_H5_H6_H7",
            "twenty_joint_matched_core_macro_steps",
            "post_A4_A5_A6_A7_H4_H5_H6_H7",
        ],
        "sana_base_pretrained_checkpoint_loaded": True,
        "sana_wam_training_checkpoint_loaded": False,
        "sana_wam_training_checkpoint_saved": False,
        "simulator_executed": False,
    }
    predecessor_training_core_lineage = {
        "equal_across_t3_t4_t5_t6": True,
        "lineage_only": True,
        "projection_sha256": T6_LINEAGE_TRAINING_CORE_SHA256,
        "t3_result_sha256": T3_PREDECESSOR_RESULT_SHA256,
        "t4_result_sha256": T4_PREDECESSOR_RESULT_SHA256,
        "t5_result_sha256": T5_PREDECESSOR_RESULT_SHA256,
        "t6_result_sha256": T6_PREDECESSOR_RESULT_SHA256,
    }
    starting_state = {
        "fresh_initialization": True,
        "initialization_seed": args.initialization_seed,
        "observed_action_losses": {
            label: measurement_before[label]["loss_action"] for label in role_order
        },
        "sana_wam_training_checkpoint_loaded": False,
    }
    report = {
        "architecture": architecture_report,
        "assets": assets_report,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "after": gpu_after,
            "before": gpu_before,
            "cuda_visible_devices": visible,
            "lock_path": str(lock_path),
            "preconstruction": gpu_preconstruction,
            "torch_cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "visible_ordinal": 0,
        },
        "contexts_by_label": contexts_by_label,
        "scientific_classification": same_task_new_episode_report,
        "matched_core_schedule": matched_core_schedule_report,
        "update_samples": cyclic_training_report,
        "fresh_heldout_transfer": fresh_heldout_report,
        "identity": identity_report,
        "input_integrity": {
            "fresh_heldout_never_entered_backward_or_update": True,
            "prepared_inputs_immutable": True,
            "prepared_tensor_aliasing": False,
            "same_task_probe": True,
        },
        "inputs_by_label": inputs_by_label,
        "interpretation_limits": {
            "scientific_classification_executed": True,
            "benchmark_success_claimed": False,
            "heldout_axis": (
                "one deterministic fresh same-task episode per suite, held out from "
                "all optimizer updates"
            ),
            "normalization_population_includes_probe_samples": True,
            "rollout_transfer_claimed": False,
            "same_task_probe_only": True,
            "strict_dataset_holdout": False,
            "task_text_only_isolation": False,
            "single_seed_single_sample_pair_per_suite": True,
            "training_or_benchmark_readiness_claimed": False,
        },
        "memory": {
            "build": build_memory,
            "prepare": prepare_measurements,
            "update_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "update_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "optimizer": optimizer_report,
        "per_step": per_step,
        "randomness": randomness_report,
        "execution_result": "PASS",
        "harness_result": "PASS",
        "execution_verdict": execution_verdict,
        "result": execution_verdict,
        "run": {
            "arm": "JOINT",
            "nonce": args.nonce,
            "root": str(run_root),
        },
        "sample_roles": sample_roles_report,
        "schema_version": (
            "sana-wam-libero-t11-same-task-new-episode-balanced-joint-v1"
        ),
        "scope": scope_report,
        "starting_state": starting_state,
        "timings_seconds": {
            "model_and_dataset_build": build_seconds,
            "prepare_real_samples": {
                label: measurement["seconds"]
                for label, measurement in prepare_measurements.items()
            },
            "sample_load": sample_seconds,
            "twenty_matched_core_macro_updates_and_sixteen_measurements": update_seconds,
        },
        "predecessor_training_core_lineage": predecessor_training_core_lineage,
        "direct_predecessor": {
            "arm_result_sha256": {
                "JOINT": T10_JOINT_RESULT_SHA256,
                "SEQ": T10_SEQ_RESULT_SHA256,
            },
            "arm_runner_sha256": T10_ARM_RUNNER_SHA256,
            "arm_source_commit": T10_ARM_SOURCE_COMMIT,
            "result_path": str(T10_AGGREGATE_RESULT_PATH),
            "result_sha256": T10_AGGREGATE_RESULT_SHA256,
            "root": str(T10_AGGREGATE_ROOT),
            "runner_sha256": T10_AGGREGATE_RUNNER_SHA256,
            "source_commit": T10_AGGREGATE_SOURCE_COMMIT,
            "stage": "T10_AGGREGATE",
        },
        "valid_run": True,
        "scientific_verdict": scientific_verdict,
        "secondary_diagnostic_warnings": secondary_diagnostic_warnings,
        "verdict": execution_verdict,
        "verdict_reasons": verdict_reasons,
        "warnings": warning_messages,
    }
    if any(run_root.iterdir()):
        raise RuntimeError("T11 run root was not empty before terminal write")
    _write_terminal_report(run_root / "RESULT.json", report)
    _freeze_run_root(run_root)
    _ACTIVE_RUN_ROOT = None
    gpu_lock.close()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as error:
        if _ACTIVE_RUN_ROOT is not None:
            try:
                _terminalize_failure(_ACTIVE_RUN_ROOT, error)
            except BaseException as freeze_error:
                print(
                    "failed to terminalize T11 root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    raise SystemExit(exit_code)
