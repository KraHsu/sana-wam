#!/usr/bin/env python
"""Run T16 paired one-task LIBERO chunk-closed-loop architecture validation."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import random
import select
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

from smoke_libero_ar_real_gpu import (  # noqa: E402
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


T16_POLICY_STEPS_PER_ARM = 32
T16_AR_ACTION_TOKENS_PER_CHUNK = 28
T16_FULL_ARM_GENERATION_STEPS_ONE_BASED = (1, 29)
T16_RAW_NUM_FRAMES = 113
T16_VIDEO_NUM_FRAMES = 29
T16_VIDEO_STRIDE = 4
T16_TEMPORAL_COMPRESSION = 4
T16_LATENT_FRAMES = 8
T16_AR_CHUNKS = 4
T16_MODEL_NOISE_BASE_SEED = 20260807
T16_TASK = "pick up the black bowl on the cookie box and place it on the plate"
T16_TASK_SUITE = "libero_spatial"
T16_DATASET_TASK_INDEX = 5
T16_LIBERO_SUITE_TASK_INDEX = 3
T16_INIT_STATE_INDEX = 17
T16_ENVIRONMENT_SEED = 0
T16_EXECUTION_VERDICT = "T16_PAIRED_ONE_TASK_CHUNK_CLOSED_LOOP_INTERFACE_VALID"
T16_EARLY_TERMINAL_VERDICT = "INCONCLUSIVE_EARLY_TERMINAL_NO_SECOND_GENERATION"
T16_INITIAL_SUCCESS_VERDICT = (
    "INCONCLUSIVE_INITIAL_STATE_ALREADY_SUCCESS_NO_POLICY_STEP"
)
T16_ROOT_SLUG = "libero-t16-paired-one-task-chunk-closed-loop-fixed32"
T16_SIM_WORKER_RELATIVE_PATH = "scripts/libero_t16_sim_worker.py"
T16_SIM_CONFIG_RELATIVE_PATH = "configs/benchmarks/libero/t16_simulator/config.yaml"
T16_SIM_WORKER_SHA256 = (
    "653ccc274e6dcc4f08799564c06822b28c6080c9826bfaee7af3fa5744718584"
)
T16_SIM_CONFIG_SHA256 = (
    "29b385e0ec34a664c42635c3b94e817aaa711e948a83b980410fefd3be6f2ec3"
)
T16_LIBERO_CHECKOUT = Path(
    "/home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO"
)
T16_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
T16_SIM_PYTHON = Path("/home/zch/workspace/starVLA/.venv-libero/bin/python")
T16_BDDL_PATH = T16_LIBERO_CHECKOUT / (
    "libero/libero/bddl_files/libero_spatial/"
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl"
)
T16_INIT_FILE_PATH = T16_LIBERO_CHECKOUT / (
    "libero/libero/init_files/libero_spatial/"
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.pruned_init"
)
T16_SELECTION_MANIFEST_SHA256 = (
    "30bc48677e10f0674a5837833d4b77a3ffe48751767b28a93042e08af8e3a0d8"
)
T16_INIT_SELECTION_PAYLOAD_SHA256 = (
    "92b19e8daf3b6aef7b311d5ba66a48f7cee5a99abf99b1c23d8308a2f11166fd"
)
T16_T15_UPDATE_CORE_AST_SHA256 = (
    "357c111513e0b96e30297189f9d47bdef6a0b2486bd92b32d7e244b2f4799429"
)

T15_MACRO_STEPS = 20
T15_UPDATE_STEPS = T15_MACRO_STEPS
T15_FORWARD_EXPOSURES_PER_SAMPLE = T15_MACRO_STEPS
T15_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO = 0.125
T15_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE = 2.5
T15_CUMULATIVE_LOSS_WEIGHT_PER_SUITE = 5.0
T15_TRAINING_FORWARDS = T15_MACRO_STEPS * 8
T15_MEASUREMENT_FORWARDS = 32
T15_ARCHITECTURE_FORWARDS = T15_TRAINING_FORWARDS + T15_MEASUREMENT_FORWARDS
T15_EXECUTION_VERDICT = "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_ARM_VALID"
T15_REPLICATED_VERDICT = (
    "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED"
)
T15_TRAINING_FIT_FAILURE_VERDICT = (
    "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_TRAINING_FIT_FAILURE"
)
T15_HELDOUT_GAP_VERDICT = "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_HELDOUT_GAP"
T15_ROOT_SLUG = "libero-t15-loss-recipe-seed-replication-accum8-balanced-joint-fixed20"
T15_INITIALIZATION_SEED = 20260807
T15_DATALOADER_SEED = 20260806
T15_LOSS_RECIPE_SEED = 20260827
T15_CONFIG_SHA256 = "36daaada09409c22ef2e57e3382ca2ef28c6847f9d2e043cad354464c3dde929"
T15_CONFIG_RELATIVE_PATH = (
    "configs/experiments/libero_t14_initseed_replication_accum8.yaml"
)
BASELINE_CONFIG_SHA256 = (
    "5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45"
)
SPATIAL_NAME = "libero_spatial_no_noops_1.0.0_lerobot"
OBJECT_NAME = "libero_object_no_noops_1.0.0_lerobot"
GOAL_NAME = "libero_goal_no_noops_1.0.0_lerobot"
LIBERO10_NAME = "libero_10_no_noops_1.0.0_lerobot"
T13_SUITE_ORDER = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)
T12_Q_BY_SUITE = {
    SPATIAL_NAME: 0.08146295687673233,
    OBJECT_NAME: 0.11867336135926909,
    GOAL_NAME: 0.11409843880020352,
    LIBERO10_NAME: 0.07143344992977133,
}
T13_Q_BY_SUITE = {
    SPATIAL_NAME: 0.10359865738261917,
    OBJECT_NAME: 0.12481595725077935,
    GOAL_NAME: 0.1316228601408628,
    LIBERO10_NAME: 0.07901416496924872,
}
T14_Q_BY_SUITE = {
    SPATIAL_NAME: 0.06038898107589587,
    OBJECT_NAME: 0.04871424574422691,
    GOAL_NAME: 0.058966199161112065,
    LIBERO10_NAME: 0.05522630218227865,
}
T11_Q_BY_SUITE = {
    SPATIAL_NAME: 0.20404818402960948,
    OBJECT_NAME: 0.20277989493511397,
    GOAL_NAME: 0.24299099506480765,
    LIBERO10_NAME: 0.1329802905245845,
}
T13_TRAIN_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000191.parquet": "a3fa6fd32bbb8007d64a2d9c0a6228775290a1772a91786e5be200694d5e6ba6",
            "videos/chunk-000/observation.images.image/episode_000191.mp4": (
                "ce1c65516aa6b336cf052aeee06e8dd8ca388519bf56d49b0eea95203c763cea"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000191.mp4": (
                "f968092944d7e934140264f8469c8f726cac1afbce881dbf6b5735589e5c4806"
            ),
        },
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "episode_index": 191,
        "episode_length": 96,
        "episode_selection_payload_sha256": "005a0af619a3b4e69056d7e90ca72710eb004e6ea2d000df5f604d3f6ebbbcb0",
        "label": "A12",
        "start_frame": 0,
        "task": "pick up the black bowl on the cookie box and place it on the plate",
        "task_index": 5,
        "t12_task_selection_payload_sha256": (
            "390e72912baa953cddba592bf52ed7d95917295ec7fad58036fbc7cc7fdc9171"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000144.parquet": "d3eb4b79af6e4696f98006a82b5166dfb3c82fde0a39bbe148bdb465116cf99d",
            "videos/chunk-000/observation.images.image/episode_000144.mp4": (
                "84973a69956881a296f4780116d301d523afb2c6689b538c4a6bae51ca981ad1"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000144.mp4": (
                "1189d52537b0bfe363c8b1d5fcb371add0199a19d08d541d2ebd4ad9d8818f20"
            ),
        },
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "episode_index": 144,
        "episode_length": 92,
        "episode_selection_payload_sha256": "061ddc4c5aa65ec69c7e434e45e1e032e1147b4f65f112c3505f61788a467d7d",
        "label": "A13",
        "start_frame": 0,
        "task": "pick up the black bowl on the cookie box and place it on the plate",
        "task_index": 5,
        "t12_task_selection_payload_sha256": (
            "390e72912baa953cddba592bf52ed7d95917295ec7fad58036fbc7cc7fdc9171"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000019.parquet": "079c6a4b2f1969372d3d614b2df9fd21362ee07c71fdb5614765a1b66a9befe8",
            "videos/chunk-000/observation.images.image/episode_000019.mp4": (
                "8d456b1fd53b7fd23f20fc91797bb1cc79f8083e6e0d0fd967d78e7792307521"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000019.mp4": (
                "bf243844727abccd953ff95d819cd525a4a02db731786c17ea3c654ad731bf36"
            ),
        },
        "dataset": "libero_object_no_noops_1.0.0_lerobot",
        "episode_index": 19,
        "episode_length": 161,
        "episode_selection_payload_sha256": "02c99c747bb3b5ccb6125736cd4b1238efef04f402bcb59ea6a95e07cc7fa009",
        "label": "A14",
        "start_frame": 0,
        "task": "pick up the chocolate pudding and place it in the basket",
        "task_index": 9,
        "t12_task_selection_payload_sha256": (
            "2c4d697fbfebb3a91288e1b499eba53ba755692cd06e0b7a6308866e90ab36dd"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000170.parquet": "2b2b32edceebfbb490acf6d708c6d45d4162ccb793d4548771ae40f36a07e058",
            "videos/chunk-000/observation.images.image/episode_000170.mp4": (
                "f3ad510c632d7699f3efaab388ed4054546b91e3dc393b32f562a89fd626f5e9"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000170.mp4": (
                "04f3a92be26202da95456c3396a379979340909e55795592e492c3a05d30877c"
            ),
        },
        "dataset": "libero_object_no_noops_1.0.0_lerobot",
        "episode_index": 170,
        "episode_length": 160,
        "episode_selection_payload_sha256": "0883a03863e2225db35e0aeb06d163e83343d3613ed79391a31cf52365747e81",
        "label": "A15",
        "start_frame": 0,
        "task": "pick up the chocolate pudding and place it in the basket",
        "task_index": 9,
        "t12_task_selection_payload_sha256": (
            "2c4d697fbfebb3a91288e1b499eba53ba755692cd06e0b7a6308866e90ab36dd"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000075.parquet": "366d6da6b3cd6c663bcbdefdfef5030518d06db978396df99e3b30a5541c7dba",
            "videos/chunk-000/observation.images.image/episode_000075.mp4": (
                "d94f6b8c47fce66bf3229217ac494ed430015c57e7241e6380a737d9400233d9"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000075.mp4": (
                "dbbff0bf8a1b087cb381b7692b58437282a7eca6c671b32df8f447b6f395e705"
            ),
        },
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "episode_index": 75,
        "episode_length": 116,
        "episode_selection_payload_sha256": "05dc21148e96135b5b48dd662428a77515e8f09616de438ed993eea4b99efd8c",
        "label": "A16",
        "start_frame": 0,
        "task": "put the wine bottle on top of the cabinet",
        "task_index": 4,
        "t12_task_selection_payload_sha256": (
            "1b37e4af7c810362572f129a2295d6475c635d6b35ce0baf7a117106860bfe9f"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000038.parquet": "b9e6f893844432aed742d433a7fe988ca48a943f63970053adcd1056fec084ca",
            "videos/chunk-000/observation.images.image/episode_000038.mp4": (
                "3e866e4c0d15aba2f31149a124230b16ab1fb686a2d87800f0ac04aae42191e7"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000038.mp4": (
                "c606523c432ddb99ca4818b6f1196511366aebabaa8d8898a49a7f3f4e968e2a"
            ),
        },
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "episode_index": 38,
        "episode_length": 93,
        "episode_selection_payload_sha256": "0d734b094457fb2f7b36a89aa986972259cca2e05cb4fccd40e5e0e43e582c26",
        "label": "A17",
        "start_frame": 0,
        "task": "put the wine bottle on top of the cabinet",
        "task_index": 4,
        "t12_task_selection_payload_sha256": (
            "1b37e4af7c810362572f129a2295d6475c635d6b35ce0baf7a117106860bfe9f"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000070.parquet": "eebbafe42f78333ebf27b0f02b3d21da6c6377baec28249ebf7dd0921cba1914",
            "videos/chunk-000/observation.images.image/episode_000070.mp4": (
                "faf2aebf1b09091a9dca4692f2e8a9a634af39fe95f9c080d8fc3936990a2b3a"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000070.mp4": (
                "3f02e9b949d9afc253ae5a582c2bc362b98a1e4cfae02c22a0ede75a8d642841"
            ),
        },
        "dataset": "libero_10_no_noops_1.0.0_lerobot",
        "episode_index": 70,
        "episode_length": 416,
        "episode_selection_payload_sha256": "0435aa1ec5f6780ef6abd1d3225a9b852ba3dd400f5b307a89817945f6407e9c",
        "label": "A18",
        "start_frame": 0,
        "task": "put both moka pots on the stove",
        "task_index": 6,
        "t12_task_selection_payload_sha256": (
            "37f56c5d986865060b7e1b2fe1cedf1b18bb7bdcbb5ee4f929b8f09e4ec6d531"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000010.parquet": "690e333e374dc9ff9b363dc433b5e8fd12c73f00131334d7f8c9cffcd583ea95",
            "videos/chunk-000/observation.images.image/episode_000010.mp4": (
                "e1f1d1f702f1887980df476082c7fe8b1e62e9fb751fb915b519bf04f1b512f1"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000010.mp4": (
                "794928ec1fd9f2c31a36b5144a7cf373535d43ab4cd6cf58b06adc92276e33a1"
            ),
        },
        "dataset": "libero_10_no_noops_1.0.0_lerobot",
        "episode_index": 10,
        "episode_length": 383,
        "episode_selection_payload_sha256": "0754b924ac8d16ba0519d54874f214c7a4563117e53329ce3dc08611afca3c1c",
        "label": "A19",
        "start_frame": 0,
        "task": "put both moka pots on the stove",
        "task_index": 6,
        "t12_task_selection_payload_sha256": (
            "37f56c5d986865060b7e1b2fe1cedf1b18bb7bdcbb5ee4f929b8f09e4ec6d531"
        ),
    },
)
T13_HELDOUT_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000130.parquet": "3b4344624ee3c5dee4542c2f369198ed15c1a2847c4d86bda5edf88d5025f940",
            "videos/chunk-000/observation.images.image/episode_000130.mp4": (
                "be3623f656cd53e4b483af3a8d645466776dfd88c15d1b35e6a4bd56678c58f5"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000130.mp4": (
                "d4a3ecce265850030552cac1cfdb7d747c870cf1eff543e68b06da07e958047b"
            ),
        },
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "episode_index": 130,
        "episode_length": 96,
        "episode_selection_payload_sha256": "06a96ae8e8b2ae01f27826e22d0b4e0a31ae2e5907a44de3d95f55aff5f624aa",
        "label": "H12",
        "start_frame": 0,
        "task": "pick up the black bowl on the cookie box and place it on the plate",
        "task_index": 5,
        "t12_task_selection_payload_sha256": (
            "390e72912baa953cddba592bf52ed7d95917295ec7fad58036fbc7cc7fdc9171"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000397.parquet": "eb8977ecedb7a5714501d7e49fafc720253ed8e982c38a3990eb49cf859e3fae",
            "videos/chunk-000/observation.images.image/episode_000397.mp4": (
                "327fb7b06fdc5e30e8ece0fdc11fd4507adb0c4ad0289c08d1dea34587abd31b"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000397.mp4": (
                "45886d1d4466f0fd28065e8f16661411de3de0660d953d20a64d9676668cb626"
            ),
        },
        "dataset": "libero_spatial_no_noops_1.0.0_lerobot",
        "episode_index": 397,
        "episode_length": 101,
        "episode_selection_payload_sha256": "0a5081bb8c8177661234b984595bb9467184159929ce46658b89d37f70023981",
        "label": "H13",
        "start_frame": 0,
        "task": "pick up the black bowl on the cookie box and place it on the plate",
        "task_index": 5,
        "t12_task_selection_payload_sha256": (
            "390e72912baa953cddba592bf52ed7d95917295ec7fad58036fbc7cc7fdc9171"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000394.parquet": "e3479e2d87672880c6f15e2c33a3eb6676f0816ee962ac4b850dfc52afb61b13",
            "videos/chunk-000/observation.images.image/episode_000394.mp4": (
                "be9a81ad7897f2582105365d3fdfa20945ce2ec35abd47c61ce6786a57de42a7"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000394.mp4": (
                "697c858358b8f59d9bfa3c4aa10f052149670fa1928fe64a48f7d2dfb896fe7a"
            ),
        },
        "dataset": "libero_object_no_noops_1.0.0_lerobot",
        "episode_index": 394,
        "episode_length": 148,
        "episode_selection_payload_sha256": "09802da20ca799ced92f2fb1ce025bef7ed9c3e41b165e7998c195e058803dd7",
        "label": "H14",
        "start_frame": 0,
        "task": "pick up the chocolate pudding and place it in the basket",
        "task_index": 9,
        "t12_task_selection_payload_sha256": (
            "2c4d697fbfebb3a91288e1b499eba53ba755692cd06e0b7a6308866e90ab36dd"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000034.parquet": "01faa0d3e8c571a1f559c2cc655741910cbd8b8e9bcf6d7bb623de45c8c162ab",
            "videos/chunk-000/observation.images.image/episode_000034.mp4": (
                "14c8a9d349041156668e37c97281fd353de2f427da6a2b2896b229606e1a307d"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000034.mp4": (
                "cf8fb6facde8b7d278937c37234e8e46629fc9c2798b191e0c94f5f4c576aa44"
            ),
        },
        "dataset": "libero_object_no_noops_1.0.0_lerobot",
        "episode_index": 34,
        "episode_length": 224,
        "episode_selection_payload_sha256": "16adfd7a1fcf839888c8f56668423dfeb85d2738321b80b99717fcdc4ef7daa4",
        "label": "H15",
        "start_frame": 0,
        "task": "pick up the chocolate pudding and place it in the basket",
        "task_index": 9,
        "t12_task_selection_payload_sha256": (
            "2c4d697fbfebb3a91288e1b499eba53ba755692cd06e0b7a6308866e90ab36dd"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000154.parquet": "ae636d4ab5d6de45bc4d4d087583d705293986f211cf1a87dd6d37b9fc7d4f51",
            "videos/chunk-000/observation.images.image/episode_000154.mp4": (
                "c04c9181945a6f2dc00dae1e2e6ac91528af6559e9229032c45fc89a3949943d"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000154.mp4": (
                "af01f73a21066b8a5582bc77dbbd04793bec628f32d320100c2c9a0e5e9b837d"
            ),
        },
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "episode_index": 154,
        "episode_length": 113,
        "episode_selection_payload_sha256": "0f6ede85bc15ecf21276c08893983aee044b14f082645f98086cbc438bf63ad1",
        "label": "H16",
        "start_frame": 0,
        "task": "put the wine bottle on top of the cabinet",
        "task_index": 4,
        "t12_task_selection_payload_sha256": (
            "1b37e4af7c810362572f129a2295d6475c635d6b35ce0baf7a117106860bfe9f"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000195.parquet": "1d42a00dab9b0d57d8fb6aa89b0cae4baccdbb8fd12944ddb6e12c1bc81ae0f3",
            "videos/chunk-000/observation.images.image/episode_000195.mp4": (
                "7bafce43b500298ceef3910cdf3ad5f7169f719cabd713f66d54682a16958326"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000195.mp4": (
                "ced52ac7fb91846ce5fba9e8b6945ca8f2dab9f584b728934e54a0422b88d615"
            ),
        },
        "dataset": "libero_goal_no_noops_1.0.0_lerobot",
        "episode_index": 195,
        "episode_length": 104,
        "episode_selection_payload_sha256": "150fef44a91f392f8b71097081ae1c4d987b1347e848a1a68f633851b129fe82",
        "label": "H17",
        "start_frame": 0,
        "task": "put the wine bottle on top of the cabinet",
        "task_index": 4,
        "t12_task_selection_payload_sha256": (
            "1b37e4af7c810362572f129a2295d6475c635d6b35ce0baf7a117106860bfe9f"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000023.parquet": "8d6a2eb1cb4ab60028e660c6afe9868a9b27704e59dc5fa5071f5120bb67dd05",
            "videos/chunk-000/observation.images.image/episode_000023.mp4": (
                "0ad7fea9579fe9a37dc3748c7d9262ad0559e912c12838f57db06fc7884c15ce"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000023.mp4": (
                "f40dc4d606c51bb390ec5b2f04d68ed4d59c388d2561f35ad4069e10d93efdeb"
            ),
        },
        "dataset": "libero_10_no_noops_1.0.0_lerobot",
        "episode_index": 23,
        "episode_length": 455,
        "episode_selection_payload_sha256": "2054416b3ee082a6564d0e3c1d87499268688c826ac3747639465350b3082cd0",
        "label": "H18",
        "start_frame": 0,
        "task": "put both moka pots on the stove",
        "task_index": 6,
        "t12_task_selection_payload_sha256": (
            "37f56c5d986865060b7e1b2fe1cedf1b18bb7bdcbb5ee4f929b8f09e4ec6d531"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000306.parquet": "468b22170d30559b2afa4e8d1974cf5288e513e5081d4e667cf2d214cfaed674",
            "videos/chunk-000/observation.images.image/episode_000306.mp4": (
                "fb574ca96538aa1d34fee7f808940b7822d0cff020808b8a2a0e7fef7c449743"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000306.mp4": (
                "ec7bc702b9eef618446518e66d330db2da99756a676224d3bdfed5180580921b"
            ),
        },
        "dataset": "libero_10_no_noops_1.0.0_lerobot",
        "episode_index": 306,
        "episode_length": 505,
        "episode_selection_payload_sha256": "2616ed5d28ae57947d8159c49867cbcc84cc7dd856d961af52684454df4c7f5e",
        "label": "H19",
        "start_frame": 0,
        "task": "put both moka pots on the stove",
        "task_index": 6,
        "t12_task_selection_payload_sha256": (
            "37f56c5d986865060b7e1b2fe1cedf1b18bb7bdcbb5ee4f929b8f09e4ec6d531"
        ),
    },
)
T13_TRAIN_LABELS = ("A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19")
T13_HELDOUT_LABELS = ("H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19")
T13_CONTEXT_GROUPS = (
    ("A12", "A13", "H12", "H13"),
    ("A14", "A15", "H14", "H15"),
    ("A16", "A17", "H16", "H17"),
    ("A18", "A19", "H18", "H19"),
)
T13_PRIOR_MODEL_FACING_TASKS = (
    {"dataset": SPATIAL_NAME, "task_index": 0},
    {"dataset": SPATIAL_NAME, "task_index": 1},
    {"dataset": SPATIAL_NAME, "task_index": 4},
    {"dataset": SPATIAL_NAME, "task_index": 7},
    {"dataset": OBJECT_NAME, "task_index": 3},
    {"dataset": GOAL_NAME, "task_index": 2},
    {"dataset": LIBERO10_NAME, "task_index": 3},
)
T13_PRIOR_MODEL_FACING_SAMPLES = (
    {
        "dataset": SPATIAL_NAME,
        "task_index": 0,
        "episode_index": 0,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 0,
        "episode_index": 16,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 0,
        "episode_index": 405,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 0,
        "episode_index": 40,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 7,
        "episode_index": 36,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 1,
        "episode_index": 325,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 4,
        "episode_index": 11,
        "start_frame": 0,
    },
    {
        "dataset": OBJECT_NAME,
        "task_index": 3,
        "episode_index": 82,
        "start_frame": 0,
    },
    {
        "dataset": GOAL_NAME,
        "task_index": 2,
        "episode_index": 70,
        "start_frame": 0,
    },
    {
        "dataset": LIBERO10_NAME,
        "task_index": 3,
        "episode_index": 259,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 0,
        "episode_index": 30,
        "start_frame": 0,
    },
    {
        "dataset": OBJECT_NAME,
        "task_index": 3,
        "episode_index": 166,
        "start_frame": 0,
    },
    {
        "dataset": GOAL_NAME,
        "task_index": 2,
        "episode_index": 248,
        "start_frame": 0,
    },
    {
        "dataset": LIBERO10_NAME,
        "task_index": 3,
        "episode_index": 278,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 0,
        "episode_index": 76,
        "start_frame": 0,
    },
    {
        "dataset": SPATIAL_NAME,
        "task_index": 0,
        "episode_index": 265,
        "start_frame": 0,
    },
    {
        "dataset": OBJECT_NAME,
        "task_index": 3,
        "episode_index": 64,
        "start_frame": 0,
    },
    {
        "dataset": OBJECT_NAME,
        "task_index": 3,
        "episode_index": 337,
        "start_frame": 0,
    },
    {
        "dataset": GOAL_NAME,
        "task_index": 2,
        "episode_index": 30,
        "start_frame": 0,
    },
    {
        "dataset": GOAL_NAME,
        "task_index": 2,
        "episode_index": 180,
        "start_frame": 0,
    },
    {
        "dataset": LIBERO10_NAME,
        "task_index": 3,
        "episode_index": 303,
        "start_frame": 0,
    },
    {
        "dataset": LIBERO10_NAME,
        "task_index": 3,
        "episode_index": 272,
        "start_frame": 0,
    },
)
T13_CONFIG_EXCLUDED_SAMPLES = ({"dataset": GOAL_NAME, "episode_index": 82},)
T13_T12_MODEL_FACING_SAMPLES = (
    {
        "dataset": SPATIAL_NAME,
        "episode_index": 202,
        "label": "A8",
        "start_frame": 0,
        "task_index": 5,
    },
    {
        "dataset": SPATIAL_NAME,
        "episode_index": 226,
        "label": "H8",
        "start_frame": 0,
        "task_index": 5,
    },
    {
        "dataset": OBJECT_NAME,
        "episode_index": 407,
        "label": "A9",
        "start_frame": 0,
        "task_index": 9,
    },
    {
        "dataset": OBJECT_NAME,
        "episode_index": 27,
        "label": "H9",
        "start_frame": 0,
        "task_index": 9,
    },
    {
        "dataset": GOAL_NAME,
        "episode_index": 89,
        "label": "A10",
        "start_frame": 0,
        "task_index": 4,
    },
    {
        "dataset": GOAL_NAME,
        "episode_index": 387,
        "label": "H10",
        "start_frame": 0,
        "task_index": 4,
    },
    {
        "dataset": LIBERO10_NAME,
        "episode_index": 316,
        "label": "A11",
        "start_frame": 0,
        "task_index": 6,
    },
    {
        "dataset": LIBERO10_NAME,
        "episode_index": 314,
        "label": "H11",
        "start_frame": 0,
        "task_index": 6,
    },
)
T13_FIXED_TASKS = (
    {
        "dataset": SPATIAL_NAME,
        "task": "pick up the black bowl on the cookie box and place it on the plate",
        "task_index": 5,
    },
    {
        "dataset": OBJECT_NAME,
        "task": "pick up the chocolate pudding and place it in the basket",
        "task_index": 9,
    },
    {
        "dataset": GOAL_NAME,
        "task": "put the wine bottle on top of the cabinet",
        "task_index": 4,
    },
    {
        "dataset": LIBERO10_NAME,
        "task": "put both moka pots on the stove",
        "task_index": 6,
    },
)
T13_SAMPLE_ORDER = (
    "A12",
    "A13",
    "H12",
    "H13",
    "A14",
    "A15",
    "H14",
    "H15",
    "A16",
    "A17",
    "H16",
    "H17",
    "A18",
    "A19",
    "H18",
    "H19",
)
T13_ELIGIBLE_MANIFEST_BYTES = 53728
T13_ELIGIBLE_MANIFEST_SHA256 = (
    "50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47"
)
T13_SELECTION_MANIFEST_BYTES = 16763
T13_SELECTION_MANIFEST_SHA256 = (
    "fa00b477558eb26ec5657b87e11e4d6023181f88ea023889e2e86d94b3fd5305"
)
T13_SELECTION_RULE = (
    "in frozen suite order spatial,object,goal,10; inherit exactly the T12-selected "
    "task identity in each suite; fix start_frame=0 and exclude every candidate "
    "whose (dataset,task_index,episode_index,start_frame) occurs in the complete "
    "frozen T1-T12 model-facing measured-or-backpropagated sample set; apply no "
    "episode-length filter; rank all remaining eligible episodes by SHA256 of the "
    "exact T13_MULTI_EPISODE_ACCUM8_SELECTION_V1 ASCII payload with final LF; "
    "take the first four; assign ranks 1 and 2 to update and ranks 3 and 4 to "
    "heldout"
)
T13_SUITE_METADATA_SHA256 = {
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
T11_PREDECESSOR_SOURCE_COMMIT = "7d53d618234e3e37367c5b1e39e169742c2cf514"
T11_PREDECESSOR_RUNNER_SHA256 = (
    "ab66fcf6052582631424c97f149c9187d942b1da253e201fc698142e5ad41b54"
)
T11_PREDECESSOR_RUNNER_PATH = (
    ROOT / "scripts/smoke_libero_ar_t11_same_task_new_episode_balanced_joint_gpu.py"
)
T11_PREDECESSOR_RESULT_SHA256 = (
    "6da12f2e3622d4e6427070bfd10b3dab9550c338a834b7662309cefea98d7417"
)
T11_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t11/7d53d618234e/"
    "libero-t11-same-task-new-episode-balanced-joint-fixed20-"
    "05e0d719fbc5c25e66ddf435cb47eba2/RESULT.json"
)
T11_PREDECESSOR_ROOT = T11_PREDECESSOR_RESULT_PATH.parent
T12_PREDECESSOR_SOURCE_COMMIT = "bf4e6f43395e0ba2c177d81856ad37f87c6e91fe"
T12_PREDECESSOR_RUNNER_SHA256 = (
    "baf19d25d39412a6be02ce35d5c06cae1e8a3089d1c005c35768ca0e0f3581d0"
)
T12_PREDECESSOR_RUNNER_PATH = (
    ROOT / "scripts/smoke_libero_ar_t12_new_task_balanced_joint_gpu.py"
)
T12_PREDECESSOR_RESULT_SHA256 = (
    "c1f045e62c854ab897305fbed504e765bd229d02479d2d11861f210d26abfe08"
)
T12_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t12/bf4e6f43395e/"
    "libero-t12-new-task-balanced-joint-fixed20-"
    "09032d945cdf4dac558ab13b9dabafdd/RESULT.json"
)
T12_PREDECESSOR_ROOT = T12_PREDECESSOR_RESULT_PATH.parent
T12_ELIGIBLE_MANIFEST_SHA256 = (
    "50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47"
)
T12_SELECTION_MANIFEST_SHA256 = (
    "7f9ba75f17c7bdeaf3e68a70eaa9d752b26cf057250ff7f9c3fa37e2a895fe89"
)
T13_PREDECESSOR_SOURCE_COMMIT = "68fa88157973f59383a87be1cb3107f5824e64cf"
T13_PREDECESSOR_RUNNER_SHA256 = (
    "ddbd141f3c32f9f89c4e936b1432d379cf56f0b886f84f38406fa4481654efb8"
)
T13_PREDECESSOR_RUNNER_PATH = (
    ROOT / "scripts/smoke_libero_ar_t13_multi_episode_accum8_balanced_joint_gpu.py"
)
T13_PREDECESSOR_RESULT_SHA256 = (
    "a75739991561965d212ad98cc2504cabf8696b2dcaf2e0ad5b46aa774ca00763"
)
T13_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t13/68fa88157973/"
    "libero-t13-multi-episode-accum8-balanced-joint-fixed20-"
    "ce58b18994fa066b49b5e52bbd98b81e/RESULT.json"
)
T13_PREDECESSOR_ROOT = T13_PREDECESSOR_RESULT_PATH.parent
T14_PREDECESSOR_SOURCE_COMMIT = "d230798ec66bd05fb5320bf8862b367fb0dfecbb"
T14_PREDECESSOR_RUNNER_SHA256 = (
    "2e44ed88267bdc8b96deb79533a35f3327ea2e731ac1556c5716caebb4fca73c"
)
T14_PREDECESSOR_RUNNER_PATH = (
    ROOT / "scripts/smoke_libero_ar_t14_initseed_replication_accum8_gpu.py"
)
T14_PREDECESSOR_RESULT_SHA256 = (
    "9c0e6b5f83e989c76570363ea182a0d547c61ebabc997f760139906bee7a930d"
)
T14_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t14/d230798ec66b/"
    "libero-t14-initseed-replication-accum8-balanced-joint-fixed20-"
    "a9d1dd749b4dd62b454324872401cdcb/RESULT.json"
)
T14_PREDECESSOR_ROOT = T14_PREDECESSOR_RESULT_PATH.parent
T14_PREDECESSOR_INITIALIZATION_SEED = 20260807
T14_PREDECESSOR_LOSS_RECIPE_SEED = 20260826
T14_PREDECESSOR_RECIPE_SIGNATURE_SHA256 = (
    "ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad"
)
SPATIAL_STATS_GR00T_SHA256 = (
    "0a4b08f5afcdcbe186ec70ea6ff2569233bab706a4d99ed603b23a7688bc33bb"
)
T15_PREDECESSOR_SOURCE_COMMIT = "d669a2e0bc201941fd911f484df36d7afec44cd6"
T15_PREDECESSOR_RUNNER_SHA256 = (
    "25ebbe148287a7ee9795f56e9253b1db238d0cf5dbea52778c7b61932ee9023a"
)
T15_PREDECESSOR_RUNNER_PATH = (
    ROOT / "scripts/smoke_libero_ar_t15_lossrecipe_replication_accum8_gpu.py"
)
T15_PREDECESSOR_RESULT_SHA256 = (
    "9e4ee9cb515e5cfafc5598eff33041232ad702b32e641b26435bcb1c643820b7"
)
T15_PREDECESSOR_ROOT = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t15/d669a2e0bc20/libero-t15-loss-recipe-seed-replication-accum8-balanced-joint-fixed20-9732531de18094840229d7ca0598a648"
)
T15_PREDECESSOR_RESULT_PATH = T15_PREDECESSOR_ROOT / "RESULT.json"
T15_PREDECESSOR_RECIPE_SIGNATURE_SHA256 = (
    "1ed9a3c26a9035f9b58a38df471dc09fb62a5b9110d08c99488a1eb7e96b69c5"
)
T16_RUN_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t16")
_ACTIVE_RUN_ROOT: Path | None = None
_ACTIVE_SIM_WORKER: Any | None = None


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
        raise RuntimeError(
            f"T15 inherited T13 metadata must be a regular non-symlink file: {path}"
        )
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise RuntimeError(
                f"T15 inherited T13 metadata contains a blank row: {path}:{line_number}"
            )
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(
                f"T15 inherited T13 metadata row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _build_t13_eligible_manifest(
    metadata_by_suite: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> dict[str, Any]:
    if set(metadata_by_suite) != set(T13_SUITE_ORDER):
        raise RuntimeError("T13 candidate suite set differs")
    expected_raw_counts = {
        SPATIAL_NAME: 432,
        OBJECT_NAME: 454,
        GOAL_NAME: 428,
        LIBERO10_NAME: 379,
    }
    expected_eligible_task_counts = {
        SPATIAL_NAME: 6,
        OBJECT_NAME: 9,
        GOAL_NAME: 9,
        LIBERO10_NAME: 9,
    }
    expected_eligible_episode_counts = {
        SPATIAL_NAME: 254,
        OBJECT_NAME: 408,
        GOAL_NAME: 391,
        LIBERO10_NAME: 338,
    }
    excluded_tasks = {
        (str(row["dataset"]), int(row["task_index"]))
        for row in T13_PRIOR_MODEL_FACING_TASKS
    }
    suites = []
    for dataset in T13_SUITE_ORDER:
        task_rows, episode_rows = metadata_by_suite[dataset]
        if any(set(row) != {"task_index", "task"} for row in task_rows):
            raise RuntimeError(f"T13 tasks metadata schema differs: {dataset}")
        task_by_index = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        if (
            sorted(task_by_index) != list(range(10))
            or len(task_by_index) != len(task_rows)
            or len(set(task_by_index.values())) != len(task_by_index)
        ):
            raise RuntimeError(f"T13 task registry differs: {dataset}")
        task_index_by_name = {name: index for index, name in task_by_index.items()}
        episodes_by_task: dict[int, list[dict[str, int]]] = {
            task_index: [] for task_index in task_by_index
        }
        observed_episode_indices: set[int] = set()
        for row in episode_rows:
            if set(row) != {"episode_index", "tasks", "length"}:
                raise RuntimeError(f"T13 episodes metadata schema differs: {dataset}")
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
                raise RuntimeError(f"T13 episodes metadata identity differs: {dataset}")
            observed_episode_indices.add(episode_index)
            task_index = task_index_by_name[tasks[0]]
            if dataset == GOAL_NAME and episode_index == 82:
                continue
            episodes_by_task[task_index].append(
                {"episode_index": episode_index, "length": length}
            )
        if len(episode_rows) not in {
            expected_raw_counts[dataset],
            427 if dataset == GOAL_NAME else expected_raw_counts[dataset],
        }:
            raise RuntimeError(f"T13 source episode count differs: {dataset}")
        tasks = []
        for task_index in sorted(task_by_index):
            if (dataset, task_index) in excluded_tasks:
                continue
            eligible_episodes = sorted(
                episodes_by_task[task_index],
                key=lambda row: row["episode_index"],
            )
            tasks.append(
                {
                    "episode_count": len(eligible_episodes),
                    "episodes": eligible_episodes,
                    "task": task_by_index[task_index],
                    "task_index": task_index,
                }
            )
        if (
            len(tasks) != expected_eligible_task_counts[dataset]
            or sum(row["episode_count"] for row in tasks)
            != expected_eligible_episode_counts[dataset]
        ):
            raise RuntimeError(
                f"T13 inherited T12 eligible population differs: {dataset}"
            )
        suites.append(
            {
                "dataset": dataset,
                "episode_count": sum(row["episode_count"] for row in tasks),
                "episodes_jsonl_sha256": T13_SUITE_METADATA_SHA256[dataset][
                    "episodes.jsonl"
                ],
                "task_count": len(tasks),
                "tasks": tasks,
                "tasks_jsonl_sha256": T13_SUITE_METADATA_SHA256[dataset]["tasks.jsonl"],
            }
        )
    manifest = {
        "candidate_episode_count": sum(row["episode_count"] for row in suites),
        "candidate_suite_count": len(suites),
        "candidate_task_count": sum(row["task_count"] for row in suites),
        "excluded_samples": list(T13_CONFIG_EXCLUDED_SAMPLES),
        "excluded_tasks": list(T13_PRIOR_MODEL_FACING_TASKS),
        "schema_version": "sana-wam-libero-t12-new-task-balanced-joint-eligible-v1",
        "start_frame": 0,
        "suites": suites,
        "t11_predecessor_result_sha256": T11_PREDECESSOR_RESULT_SHA256,
    }
    if (
        manifest["candidate_suite_count"] != 4
        or manifest["candidate_task_count"] != 33
        or manifest["candidate_episode_count"] != 1391
        or len(_canonical_json_bytes(manifest)) != T13_ELIGIBLE_MANIFEST_BYTES
        or _sha256_json(manifest) != T13_ELIGIBLE_MANIFEST_SHA256
    ):
        raise RuntimeError("T13 inherited T12 eligible manifest differs")
    return manifest


def _eligible_sample_manifest(dataset_roots: dict[str, Path]) -> dict[str, Any]:
    metadata_by_suite = {}
    for dataset in T13_SUITE_ORDER:
        root = dataset_roots[dataset]
        tasks_path = root / "meta/tasks.jsonl"
        episodes_path = root / "meta/episodes.jsonl"
        if (
            _sha256_file(tasks_path)
            != T13_SUITE_METADATA_SHA256[dataset]["tasks.jsonl"]
        ):
            raise RuntimeError(f"T13 tasks metadata SHA differs: {dataset}")
        if (
            _sha256_file(episodes_path)
            != T13_SUITE_METADATA_SHA256[dataset]["episodes.jsonl"]
        ):
            raise RuntimeError(f"T13 episodes metadata SHA differs: {dataset}")
        metadata_by_suite[dataset] = (
            _read_jsonl(tasks_path),
            _read_jsonl(episodes_path),
        )
    return _build_t13_eligible_manifest(metadata_by_suite)


def _episode_selection_payload_sha256(
    dataset: str, task_index: int, episode_index: int
) -> str:
    payload = (
        "SANA-WAM/LIBERO/T13_MULTI_EPISODE_ACCUM8_SELECTION_V1\n"
        f"T12_RESULT_SHA256={T12_PREDECESSOR_RESULT_SHA256}\n"
        f"T12_SELECTION_MANIFEST_SHA256={T12_SELECTION_MANIFEST_SHA256}\n"
        f"DATASET={dataset}\n"
        f"TASK_INDEX={task_index}\n"
        f"EPISODE_INDEX={episode_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _selection_manifest(eligible_manifest: dict[str, Any]) -> dict[str, Any]:
    samples = []
    suite_by_dataset = {row["dataset"]: row for row in eligible_manifest["suites"]}
    fixed_task_by_dataset = {row["dataset"]: row for row in T13_FIXED_TASKS}
    expected_by_label = {
        row["label"]: row for row in (*T13_TRAIN_SAMPLES, *T13_HELDOUT_SAMPLES)
    }
    excluded_sample_keys = {
        (
            str(row["dataset"]),
            int(row["task_index"]),
            int(row["episode_index"]),
            int(row["start_frame"]),
        )
        for row in (*T13_PRIOR_MODEL_FACING_SAMPLES, *T13_T12_MODEL_FACING_SAMPLES)
    }
    candidate_episode_counts_by_suite = {}
    for suite_rank, dataset in enumerate(T13_SUITE_ORDER):
        task_rows = suite_by_dataset[dataset]["tasks"]
        fixed_task = fixed_task_by_dataset[dataset]
        matching_tasks = [
            row
            for row in task_rows
            if int(row["task_index"]) == int(fixed_task["task_index"])
            and row["task"] == fixed_task["task"]
        ]
        if len(matching_tasks) != 1:
            raise RuntimeError(f"T13 inherited T12 task identity differs: {dataset}")
        task_row = matching_tasks[0]
        task_index = int(task_row["task_index"])
        ranked_episodes = sorted(
            (
                _episode_selection_payload_sha256(
                    dataset, task_index, int(episode_row["episode_index"])
                ),
                episode_row,
            )
            for episode_row in task_row["episodes"]
            if (dataset, task_index, int(episode_row["episode_index"]), 0)
            not in excluded_sample_keys
        )
        candidate_episode_counts_by_suite[dataset] = len(ranked_episodes)
        for zero_rank, label in enumerate(T13_CONTEXT_GROUPS[suite_rank]):
            episode_sha256, episode_row = ranked_episodes[zero_rank]
            expected = expected_by_label[label]
            sample = {
                "assets": expected["assets"],
                "dataset": dataset,
                "episode_index": int(episode_row["episode_index"]),
                "episode_length": int(episode_row["length"]),
                "episode_selection_payload_sha256": episode_sha256,
                "label": label,
                "role": "update" if zero_rank < 2 else "heldout",
                "selection_rank": zero_rank + 1,
                "start_frame": 0,
                "suite_index": suite_rank,
                "t12_task_selection_payload_sha256": expected[
                    "t12_task_selection_payload_sha256"
                ],
                "task": task_row["task"],
                "task_index": task_index,
                "within_suite_role_index": zero_rank % 2,
            }
            expected_projection = {
                key: value
                for key, value in sample.items()
                if key
                not in {
                    "role",
                    "selection_rank",
                    "suite_index",
                    "within_suite_role_index",
                }
            }
            if expected_projection != expected:
                raise RuntimeError(f"T13 mechanically selected sample {label} differs")
            samples.append(sample)
    manifest = {
        "candidate_episode_count": sum(candidate_episode_counts_by_suite.values()),
        "candidate_episode_counts_by_suite": candidate_episode_counts_by_suite,
        "candidate_filter": {
            "episode_length_filter": "none",
            "model_facing_exclusion_through": "T12",
            "start_frame": 0,
            "task_identity_source": "T12 selected task per suite",
        },
        "excluded_model_facing_samples_within_selected_tasks": list(
            T13_T12_MODEL_FACING_SAMPLES
        ),
        "heldout_probe_order": list(T13_HELDOUT_LABELS),
        "samples": samples,
        "schema_version": "sana-wam-libero-t13-multi-episode-accum8-selection-v1",
        "selection_payload_template": (
            "SANA-WAM/LIBERO/T13_MULTI_EPISODE_ACCUM8_SELECTION_V1\n"
            f"T12_RESULT_SHA256={T12_PREDECESSOR_RESULT_SHA256}\n"
            f"T12_SELECTION_MANIFEST_SHA256={T12_SELECTION_MANIFEST_SHA256}\n"
            "DATASET={dataset}\n"
            "TASK_INDEX={task_index}\n"
            "EPISODE_INDEX={episode_index}\n"
            "START_FRAME=0\n"
        ),
        "selection_rule": T13_SELECTION_RULE,
        "suite_order": list(T13_SUITE_ORDER),
        "t12_eligible_manifest_sha256": T12_ELIGIBLE_MANIFEST_SHA256,
        "t12_predecessor_result_sha256": T12_PREDECESSOR_RESULT_SHA256,
        "t12_selection_manifest_sha256": T12_SELECTION_MANIFEST_SHA256,
        "update_accumulation_order": list(T13_TRAIN_LABELS),
    }
    if (
        candidate_episode_counts_by_suite
        != {
            SPATIAL_NAME: 41,
            OBJECT_NAME: 48,
            GOAL_NAME: 45,
            LIBERO10_NAME: 27,
        }
        or manifest["candidate_episode_count"] != 161
        or len(_canonical_json_bytes(manifest)) != T13_SELECTION_MANIFEST_BYTES
        or _sha256_json(manifest) != T13_SELECTION_MANIFEST_SHA256
    ):
        raise RuntimeError("T13 multi-episode selection manifest differs")
    return manifest


def _select_t13_samples(
    dataset: Any,
    eligible_manifest: dict[str, Any],
    selection_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    live_episode_rows: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in T13_SUITE_ORDER
    }
    live_task_rows: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in T13_SUITE_ORDER
    }
    for episode in dataset._episodes:
        if episode.dataset not in T13_SUITE_ORDER:
            continue
        if episode.dataset == GOAL_NAME and episode.episode_index == 82:
            raise RuntimeError("T13 excluded Goal episode 82 entered live registry")
        task_row = {"task_index": episode.task_index, "task": episode.task}
        suite_tasks = live_task_rows[episode.dataset]
        suite_episodes = live_episode_rows[episode.dataset]
        previous_task = suite_tasks.setdefault(episode.task_index, task_row)
        if previous_task != task_row or episode.episode_index in suite_episodes:
            raise RuntimeError("T13 live cross-suite registry identity is ambiguous")
        suite_episodes[episode.episode_index] = {
            "episode_index": episode.episode_index,
            "tasks": [episode.task],
            "length": episode.length,
        }
    live_manifest = _build_t13_eligible_manifest(
        {
            name: (
                [live_task_rows[name][index] for index in sorted(live_task_rows[name])],
                [
                    live_episode_rows[name][index]
                    for index in sorted(live_episode_rows[name])
                ],
            )
            for name in T13_SUITE_ORDER
        }
    )
    if live_manifest != eligible_manifest:
        raise RuntimeError("T13 live eligible cross-suite population differs")

    expected_by_label = {
        row["label"]: row for row in (*T13_TRAIN_SAMPLES, *T13_HELDOUT_SAMPLES)
    }
    selected_sample_projection = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "role",
                "selection_rank",
                "suite_index",
                "within_suite_role_index",
            }
        }
        for row in selection_manifest["samples"]
    ]
    if selected_sample_projection != [
        expected_by_label[label] for label in T13_SAMPLE_ORDER
    ]:
        raise RuntimeError("T13 selected 4-way sample grouping differs")
    role_rows = [expected_by_label[label] for label in T13_SAMPLE_ORDER]
    prior_tasks = {
        (str(row["dataset"]), int(row["task_index"]))
        for row in T13_PRIOR_MODEL_FACING_TASKS
    }
    selected_tasks = {
        (str(row["dataset"]), int(row["task_index"])) for row in role_rows
    }
    fixed_tasks = {
        (str(row["dataset"]), int(row["task_index"])) for row in T13_FIXED_TASKS
    }
    if selected_tasks & prior_tasks or selected_tasks != fixed_tasks:
        raise RuntimeError("T13 selected tasks differ from the four fixed T12 tasks")
    prior_samples = {
        (
            str(row["dataset"]),
            int(row["task_index"]),
            int(row["episode_index"]),
            int(row["start_frame"]),
        )
        for row in (*T13_PRIOR_MODEL_FACING_SAMPLES, *T13_T12_MODEL_FACING_SAMPLES)
    }
    expected_keys = {
        (
            row["dataset"],
            int(row["task_index"]),
            int(row["episode_index"]),
            int(row["start_frame"]),
        )
        for row in role_rows
    }
    if expected_keys & prior_samples or len(expected_keys) != 16:
        raise RuntimeError("T13 selected samples overlap prior model-facing samples")
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
                "T13 cross-suite episode/start must occur exactly once: "
                f"identity={key!r} matches={len(matches)}"
            )
        dataset_index, episode = matches[0]
        selected[row["label"]] = {
            **row,
            "dataset_index": dataset_index,
            "episode": episode,
        }
    if set(selected) != set(T13_TRAIN_LABELS) | set(T13_HELDOUT_LABELS):
        raise RuntimeError("T13 selected role label set differs")
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


def _multi_episode_sample_summary(
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
        raise ValueError("T15 frozen labels differ")
    sample_by_label = {row["label"]: row for row in samples}
    if set(sample_by_label) != expected_labels:
        raise ValueError("T15 sample labels differ from the frozen set")
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
        "samples_per_suite": {
            dataset: sum(row["dataset"] == dataset for row in rows)
            for dataset in T13_SUITE_ORDER
        },
        "task_count": len({(row["dataset"], row["task_index"]) for row in rows}),
    }


def _require_strictly_positive_finite(value: float, *, field: str) -> float:
    observed = float(value)
    if not math.isfinite(observed) or observed <= 0.0:
        raise ValueError(f"T15 {field} must be finite and strictly positive")
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
    if macro_step < 1 or macro_step > T15_MACRO_STEPS:
        raise ValueError("T15 no-mutation evidence macro step is outside 1..20")
    if set(before) != set(components) or set(after_microbacks) != set(components):
        raise ValueError("T15 no-mutation evidence component schema differs")
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
            "T15 training state mutated within macro before optimizer step "
            f"{macro_step}: {changed!r}"
        )
    return evidence


def _multi_episode_summary(
    update_before: dict[str, float],
    update_after: dict[str, float],
    heldout_before: dict[str, float],
    heldout_after: dict[str, float],
) -> dict[str, Any]:
    joint_update_summary = _multi_episode_sample_summary(
        update_before,
        update_after,
        samples=T13_TRAIN_SAMPLES,
    )
    fresh_heldout_summary = _multi_episode_sample_summary(
        heldout_before,
        heldout_after,
        samples=T13_HELDOUT_SAMPLES,
    )
    update_by_dataset = {
        dataset: [
            row
            for row in joint_update_summary["per_sample"]
            if row["dataset"] == dataset
        ]
        for dataset in T13_SUITE_ORDER
    }
    heldout_by_dataset = {
        dataset: [
            row
            for row in fresh_heldout_summary["per_sample"]
            if row["dataset"] == dataset
        ]
        for dataset in T13_SUITE_ORDER
    }
    if set(update_by_dataset) != set(T13_SUITE_ORDER) or set(heldout_by_dataset) != set(
        T13_SUITE_ORDER
    ):
        raise ValueError("T15 suite coverage differs from the frozen order")
    per_suite = []
    for dataset in T13_SUITE_ORDER:
        update_rows = update_by_dataset[dataset]
        heldout_rows = heldout_by_dataset[dataset]
        if len(update_rows) != 2 or len(heldout_rows) != 2:
            raise ValueError(
                f"T15 suite must retain two update and two heldout rows: {dataset}"
            )
        update_ratios = [float(row["post_to_pre_ratio"]) for row in update_rows]
        heldout_ratios = [float(row["post_to_pre_ratio"]) for row in heldout_rows]
        q = statistics.mean([*update_ratios, *heldout_ratios])
        t14_q = float(T14_Q_BY_SUITE[dataset])
        per_suite.append(
            {
                "all_four_ratios_strictly_below_one": all(
                    ratio < 1.0 for ratio in [*update_ratios, *heldout_ratios]
                ),
                "dataset": dataset,
                "heldout_ratios": heldout_ratios,
                "q": q,
                "q_minus_t14_q": q - t14_q,
                "q_strictly_below_t14_q": q < t14_q,
                "t14_q": t14_q,
                "update_ratios": update_ratios,
            }
        )
    all_update_improved = joint_update_summary["all_ratios_strictly_below_one"]
    all_heldout_improved = fresh_heldout_summary["all_ratios_strictly_below_one"]
    if all_update_improved and all_heldout_improved:
        scientific_verdict = T15_REPLICATED_VERDICT
    elif not all_update_improved:
        scientific_verdict = T15_TRAINING_FIT_FAILURE_VERDICT
    else:
        scientific_verdict = T15_HELDOUT_GAP_VERDICT
    return {
        "all_eight_update_ratios_strictly_below_one": all_update_improved,
        "all_eight_heldout_ratios_strictly_below_one": all_heldout_improved,
        "all_sixteen_ratios_strictly_below_one": (
            all_update_improved and all_heldout_improved
        ),
        "any_update_ratio_at_least_one": not all_update_improved,
        "same_task_heldout_samples": fresh_heldout_summary,
        "ordered_rule": [
            "all sixteen unrounded update/heldout ratios are strictly below 1",
            "otherwise any unrounded update ratio is at least 1",
            "otherwise heldout gap",
        ],
        "per_suite": per_suite,
        "scientific_verdict": scientific_verdict,
        "t14_q_comparison": {
            "excluded_from_scientific_classifier": True,
            "per_suite": [
                {
                    key: row[key]
                    for key in (
                        "dataset",
                        "q",
                        "q_minus_t14_q",
                        "q_strictly_below_t14_q",
                        "t14_q",
                    )
                }
                for row in per_suite
            ],
            "q_strictly_below_t14_count": sum(
                row["q_strictly_below_t14_q"] for row in per_suite
            ),
        },
        "update_samples": joint_update_summary,
    }


def _loss_weights_for_macro(step_index: int) -> dict[str, float]:
    if step_index < 0 or step_index >= T15_UPDATE_STEPS:
        raise ValueError("T15 step index is outside the frozen 20-step schedule")
    return {label: T15_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO for label in T13_TRAIN_LABELS}


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
    if len(per_step) != T15_UPDATE_STEPS:
        raise ValueError("T15/T3 training-core view requires exactly 20 step rows")
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


def _json_safe_observation_identity(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep simulator evidence while deliberately excluding base64 image bodies."""

    return {
        "jpeg_sha256": observation["jpeg_sha256"],
        "objects": observation["objects"],
        "observation_sha256": observation["observation_sha256"],
        "sim_state": observation["sim_state"],
        "sim_state_sha256": observation["sim_state_sha256"],
        "state": observation["state"],
    }


class T16SimulatorWorkerClient:
    """One persistent, pinned Python-3.10 LIBERO worker for both T16 arms."""

    def __init__(
        self,
        *,
        worker_path: Path,
        config_path: Path,
        render_gpu_device_id: int,
    ) -> None:
        worker_path = worker_path.resolve()
        config_path = config_path.resolve()
        environment = os.environ.copy()
        environment.update(
            {
                "LIBERO_CONFIG_PATH": str(config_path.parent),
                "CUDA_VISIBLE_DEVICES": str(render_gpu_device_id),
                "MUJOCO_EGL_DEVICE_ID": str(render_gpu_device_id),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": ":".join((str(T16_LIBERO_CHECKOUT), str(ROOT))),
            }
        )
        command = [
            str(T16_SIM_PYTHON),
            str(worker_path),
            "--libero-path",
            str(T16_LIBERO_CHECKOUT),
            "--config",
            str(config_path),
            "--bddl",
            str(T16_BDDL_PATH),
            "--init-file",
            str(T16_INIT_FILE_PATH),
            "--render-gpu-device-id",
            str(render_gpu_device_id),
            "--expected-config-sha256",
            T16_SIM_CONFIG_SHA256,
            "--expected-worker-sha256",
            T16_SIM_WORKER_SHA256,
        ]
        self._process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._next_seq = 1
        self._closed = False
        try:
            ready = self._read_response(timeout_seconds=240.0)
            if (
                ready.get("schema_version") != "sana-wam-libero-t16-sim-worker-v1"
                or ready.get("event") != "ready"
                or ready.get("status") != "ok"
                or ready.get("seq") != 0
            ):
                raise RuntimeError(
                    f"T16 simulator worker ready record differs: {ready!r}"
                )
            identity = ready.get("identity", {})
            expected_identity = {
                "bddl_language": (
                    "pick the akita black bowl on the cookies box and place it on the plate"
                ),
                "bddl_sha256": (
                    "3d4ccf070c3d9883ae0676f2d888588f98c696ddad71b7694f47c379fc99ef36"
                ),
                "config_sha256": T16_SIM_CONFIG_SHA256,
                "init_file_sha256": (
                    "0627f5f5ce3ef23be546571012be8ef603d93bcb4032bc80feb34937ba580140"
                ),
                "init_index": T16_INIT_STATE_INDEX,
                "libero_commit": T16_LIBERO_COMMIT,
                "render_gpu": {
                    "cuda_visible_devices": str(render_gpu_device_id),
                    "mujoco_egl_device_id": str(render_gpu_device_id),
                    "render_gpu_device_id": render_gpu_device_id,
                },
                "task": T16_TASK,
                "worker_sha256": T16_SIM_WORKER_SHA256,
            }
            observed_identity = {key: identity.get(key) for key in expected_identity}
            if observed_identity != expected_identity:
                raise RuntimeError(
                    "T16 simulator worker identity differs: "
                    f"expected={expected_identity!r} observed={observed_identity!r}"
                )
            self.ready = ready
        except BaseException:
            self.abort()
            raise

    def _read_response(self, *, timeout_seconds: float) -> dict[str, Any]:
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError("T16 simulator worker stdout is unavailable")
        readable, _, _ = select.select([stdout], [], [], timeout_seconds)
        if not readable:
            raise TimeoutError("T16 simulator worker response timed out")
        line = stdout.readline()
        if not line:
            return_code = self._process.poll()
            raise RuntimeError(
                f"T16 simulator worker exited before a response: {return_code}"
            )
        if len(line.encode("utf-8")) > 32 * 1024 * 1024:
            raise RuntimeError("T16 simulator worker response exceeds line-size bound")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("T16 simulator worker response is not an object")
        return response

    def request(self, command: str, **payload: Any) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("T16 simulator worker is already closed")
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("T16 simulator worker stdin is unavailable")
        request = {"command": command, "seq": self._next_seq, **payload}
        stdin.write(
            json.dumps(
                request,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        stdin.flush()
        response = self._read_response(timeout_seconds=240.0)
        if (
            response.get("schema_version") != "sana-wam-libero-t16-sim-worker-v1"
            or response.get("seq") != self._next_seq
            or response.get("status") != "ok"
        ):
            raise RuntimeError(f"T16 simulator worker response differs: {response!r}")
        self._next_seq += 1
        return response

    def close(self) -> dict[str, Any]:
        response = self.request("close")
        if response.get("event") != "closed":
            raise RuntimeError("T16 simulator worker close response differs")
        stdin = self._process.stdin
        if stdin is not None:
            stdin.close()
        return_code = self._process.wait(timeout=30.0)
        if return_code != 0:
            raise RuntimeError(
                f"T16 simulator worker close exit differs: {return_code}"
            )
        self._closed = True
        return response

    def abort(self) -> None:
        if self._closed:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10.0)
        self._closed = True


def _tensor_content_sha256(value: Any) -> str | None:
    if value is None:
        return None
    import torch

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _scheduler_snapshot(architecture: Any) -> dict[str, Any]:
    result = {}
    for label, scheduler in (
        ("video_scheduler", architecture.video_backbone.scheduler),
        ("action_scheduler", architecture.action_backbone.scheduler),
    ):
        result[label] = {
            "linear_timesteps_weights_sha256": _tensor_content_sha256(
                getattr(scheduler, "linear_timesteps_weights", None)
            ),
            "num_train_timesteps": getattr(scheduler, "num_train_timesteps", None),
            "sigmas_sha256": _tensor_content_sha256(getattr(scheduler, "sigmas", None)),
            "timesteps_sha256": _tensor_content_sha256(
                getattr(scheduler, "timesteps", None)
            ),
            "training": bool(getattr(scheduler, "training", False)),
        }
    return result


def _caller_rng_sha256(torch_module: Any) -> str:
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(repr(numpy_state[2:]).encode("ascii"))
    digest.update(torch_module.get_rng_state().cpu().numpy().tobytes())
    for state in torch_module.cuda.get_rng_state_all():
        digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def _build_t16_runtime_config(training_cfg: Any) -> Any:
    runtime_cfg = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(training_cfg, resolve=False)),
        OmegaConf.create(
            {
                "inference": {
                    "action_steps": 4,
                    "ar_action_tokens_per_chunk": T16_AR_ACTION_TOKENS_PER_CHUNK,
                    "ar_obs_chunk_mode": "rolling_buffer",
                    "ar_obs_latent_band": "auto",
                    "ar_proprio_mode": "per_step",
                    "cache_feedback_fallback": "predicted",
                    "cache_feedback_mode": "predicted",
                    "episode_noise_base_seed": T16_MODEL_NOISE_BASE_SEED,
                    "episode_noise_mode": "paired",
                    "generation_noise_schedule": {
                        "base_seed": T16_MODEL_NOISE_BASE_SEED,
                        "mode": "monotonic",
                    },
                    "generation_zero_rerank": {"enabled": False},
                    "height": 384,
                    "num_frames": T16_RAW_NUM_FRAMES,
                    "video_num_frames": T16_VIDEO_NUM_FRAMES,
                    "video_steps": 4,
                    "width": 320,
                },
                "optimization": {
                    "async_inference": {"mode": "none"},
                    "compile": {"mode": "none"},
                },
                "policy": {
                    "execute_horizon": None,
                    "history_len": T16_RAW_NUM_FRAMES,
                    "temporal_ensemble": False,
                },
                "telemetry": {"enabled": False},
            }
        ),
    )
    frozen = {
        "action_tokens": int(runtime_cfg.inference.ar_action_tokens_per_chunk),
        "action_steps": int(runtime_cfg.inference.action_steps),
        "async_mode": str(runtime_cfg.optimization.async_inference.mode),
        "cache_feedback": str(runtime_cfg.inference.cache_feedback_mode),
        "compile_mode": str(runtime_cfg.optimization.compile.mode),
        "execute_horizon": runtime_cfg.policy.execute_horizon,
        "history_len": int(runtime_cfg.policy.history_len),
        "num_frames": int(runtime_cfg.inference.num_frames),
        "temporal_ensemble": bool(runtime_cfg.policy.temporal_ensemble),
        "telemetry": bool(runtime_cfg.telemetry.enabled),
        "video_num_frames": int(runtime_cfg.inference.video_num_frames),
        "video_steps": int(runtime_cfg.inference.video_steps),
    }
    expected = {
        "action_tokens": 28,
        "action_steps": 4,
        "async_mode": "none",
        "cache_feedback": "predicted",
        "compile_mode": "none",
        "execute_horizon": None,
        "history_len": 113,
        "num_frames": 113,
        "temporal_ensemble": False,
        "telemetry": False,
        "video_num_frames": 29,
        "video_steps": 4,
    }
    if frozen != expected:
        raise RuntimeError(f"T16 runtime config differs: {frozen!r}")
    return runtime_cfg


def _run_closed_loop_rollout(
    *,
    arm: str,
    architecture: Any,
    cfg: Any,
    model_noise_seed: int,
    noise_pair_key: str,
    probe_mutation_snapshot: Any,
    sim_worker: T16SimulatorWorkerClient,
    trainer: Any,
) -> dict[str, Any]:
    """Execute one update-free 32-step greedy arm against the same simulator."""

    import torch

    from sana_wam.dataloader.transforms.normalize import (
        ActionNormalizer,
        Normalizer,
        YAML_TO_NORM_MODE,
    )
    from sana_wam.deploy.ar_engine import ARInferenceEngine
    from sana_wam.deploy.libero_model_loader import LiberoActionStateNormalizer
    from sana_wam.deploy.libero_policy_server import LiberoPolicyServer

    if arm not in {"pre", "post"}:
        raise ValueError(f"unsupported T16 rollout arm {arm!r}")
    runtime_cfg = _build_t16_runtime_config(cfg)
    prior_normalizer = architecture.action_normalizer
    if prior_normalizer is not None:
        raise RuntimeError(
            "T16 fresh training architecture unexpectedly has a normalizer"
        )
    dual_normalizer = LiberoActionStateNormalizer(
        action_normalizer=ActionNormalizer(
            mode=YAML_TO_NORM_MODE[str(cfg.dataloader.normalize_mode)],
            stats=trainer.dataset.action_stats,
        ),
        state_normalizer=Normalizer(
            mode=YAML_TO_NORM_MODE[str(cfg.dataloader.state_normalize_mode)],
            stats=trainer.dataset.state_stats,
        ),
    )
    mutation_before = probe_mutation_snapshot()
    scheduler_before = _scheduler_snapshot(architecture)
    rng_before = _caller_rng_sha256(torch)
    server = None
    engine = None
    rollout_error = None
    started = time.perf_counter()
    try:
        architecture.attach_action_normalizer(dual_normalizer)
        architecture.eval()
        engine = ARInferenceEngine(runtime_cfg, architecture=architecture)
        if (
            engine._raw_num_frames != T16_RAW_NUM_FRAMES
            or engine._video_num_frames != T16_VIDEO_NUM_FRAMES
            or engine._video_stride != T16_VIDEO_STRIDE
            or engine._temporal_compression != T16_TEMPORAL_COMPRESSION
            or engine._fcs != 2
            or engine._action_tokens_per_chunk != T16_AR_ACTION_TOKENS_PER_CHUNK
        ):
            raise RuntimeError("T16 AR engine geometry differs")
        latent_frames = (
            1 + (engine._video_num_frames - 1) // engine._temporal_compression
        )
        ar_chunks = latent_frames // engine._fcs
        if latent_frames != T16_LATENT_FRAMES or ar_chunks != T16_AR_CHUNKS:
            raise RuntimeError("T16 latent chunk geometry differs")
        server = LiberoPolicyServer(
            engine=engine,
            cfg=runtime_cfg,
            identity_cfg=cfg,
        )
        reset_ack = server.reset(
            {
                "type": "reset",
                "episode_key": f"t16/{arm}",
                "model_noise_seed": model_noise_seed,
                "noise_pair_key": noise_pair_key,
            }
        )
        if (
            reset_ack.get("status") != "ok"
            or reset_ack.get("duplicate") is not False
            or reset_ack.get("model_noise_seed") != model_noise_seed
            or reset_ack.get("noise_pair_key") != noise_pair_key
        ):
            raise RuntimeError(f"T16 policy reset ack differs: {reset_ack!r}")
        simulator_reset = sim_worker.request("reset", arm=arm)
        if simulator_reset.get("event") != "reset":
            raise RuntimeError("T16 simulator reset response differs")
        observation = simulator_reset["observation"]
        initial_identity = _json_safe_observation_identity(observation)
        policy_requests = 0
        environment_steps = 0
        generation_steps: list[int] = []
        generation_chunks = []
        trace = []
        success = bool(simulator_reset.get("success"))
        done = False
        for environment_step in range(T16_POLICY_STEPS_PER_ARM):
            if success or done:
                break
            input_identity = _json_safe_observation_identity(observation)
            prediction = server.predict(
                {
                    "images": dict(observation["images"]),
                    "prompt": T16_TASK,
                    "state": list(observation["state"]),
                }
            )
            policy_requests += 1
            policy = prediction.get("policy", {})
            expected_generation_index = (
                environment_step // T16_AR_ACTION_TOKENS_PER_CHUNK
            )
            expected_chunk_offset = environment_step % T16_AR_ACTION_TOKENS_PER_CHUNK
            expected_generated = expected_chunk_offset == 0
            expected_buffer_remaining = (
                T16_AR_ACTION_TOKENS_PER_CHUNK - expected_chunk_offset - 1
            )
            expected_public = {
                "buffer_remaining": expected_buffer_remaining,
                "chunk_offset": expected_chunk_offset,
                "generated": expected_generated,
                "generation_index": expected_generation_index,
                "policy_step": environment_step,
            }
            observed_public = {key: policy.get(key) for key in expected_public}
            if observed_public != expected_public:
                raise RuntimeError(
                    "T16 policy coordinates differ: "
                    f"expected={expected_public!r} observed={observed_public!r}"
                )
            if prediction.get("step") != environment_step + 1:
                raise RuntimeError("T16 policy request counter differs")
            if expected_generated:
                expected_frame_id = 1 if environment_step == 0 else 5
                if expected_generation_index == 1 and (
                    not trace or input_identity != trace[-1]["output_observation"]
                ):
                    raise RuntimeError(
                        "T16 step-29 generation did not consume the realized step-28 observation"
                    )
                generation_expected = {
                    "action_frame_id": expected_frame_id,
                    "ar_step_after": expected_generation_index + 1,
                    "chunk_index": expected_generation_index,
                    "chunk_len": T16_AR_ACTION_TOKENS_PER_CHUNK,
                }
                if {
                    key: policy.get(key) for key in generation_expected
                } != generation_expected:
                    raise RuntimeError("T16 generated-chunk telemetry differs")
                if policy.get("cache_feedback") != {
                    "reason": "configured",
                    "status": "predicted",
                }:
                    raise RuntimeError("T16 predicted cache-feedback telemetry differs")
                buffered = [
                    np.asarray(prediction["action"], dtype=np.float32),
                    *[
                        np.asarray(action, dtype=np.float32)
                        for action in server._policy._action_buffer
                    ],
                ]
                chunk = np.stack(buffered, axis=0)
                if (
                    chunk.shape != (T16_AR_ACTION_TOKENS_PER_CHUNK, 7)
                    or not np.isfinite(chunk).all()
                ):
                    raise RuntimeError(
                        f"T16 generated chunk shape differs: {chunk.shape}"
                    )
                generation_steps.append(environment_step + 1)
                generation_chunks.append(
                    {
                        "actions": chunk.tolist(),
                        "generation_index": expected_generation_index,
                        "generation_noise_seed": policy.get("generation_noise_seed"),
                        "generator_state_sha256": _tensor_content_sha256(
                            engine._gen.get_state()
                        ),
                        "input_observation_sha256": input_identity[
                            "observation_sha256"
                        ],
                        "sha256": hashlib.sha256(chunk.tobytes()).hexdigest(),
                        "step_one_based": environment_step + 1,
                    }
                )
            action = np.asarray(prediction.get("action"), dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError("T16 policy action must be finite shape (7,)")
            transition = sim_worker.request(
                "step",
                action=action.tolist(),
                arm=arm,
                step=environment_step,
            )
            environment_steps += 1
            if (
                transition.get("event") != "step"
                or transition.get("step") != environment_step
            ):
                raise RuntimeError("T16 simulator transition coordinates differ")
            observation = transition["observation"]
            done = bool(transition["done"])
            success = bool(transition["success"])
            trace.append(
                {
                    "action": action.tolist(),
                    "done": done,
                    "env_action": transition["env_action"],
                    "input_observation": input_identity,
                    "latency_ms": prediction["latency_ms"],
                    "output_observation": _json_safe_observation_identity(observation),
                    "policy": policy,
                    "reward": transition["reward"],
                    "step_one_based": environment_step + 1,
                    "success": success,
                }
            )
        if policy_requests != environment_steps:
            raise RuntimeError("T16 policy/simulator step counts differ")
        if environment_steps == T16_POLICY_STEPS_PER_ARM:
            if tuple(generation_steps) != T16_FULL_ARM_GENERATION_STEPS_ONE_BASED:
                raise RuntimeError(
                    f"T16 full-arm generation steps differ: {generation_steps!r}"
                )
            if len(generation_chunks) != 2:
                raise RuntimeError("T16 full arm did not materialize two chunks")
        elif environment_steps >= 29:
            if generation_steps != [1, 29]:
                raise RuntimeError("T16 late-terminal arm lacks step-29 generation")
        elif generation_steps != ([1] if environment_steps else []):
            raise RuntimeError("T16 early-terminal generation coverage differs")
        server_info = server.get_info()
        inference_runtime = server_info.get("inference_runtime", {})
        if (
            inference_runtime.get("ar_step") != len(generation_chunks)
            or inference_runtime.get("cache_feedback_commits") != 0
            or inference_runtime.get("cache_feedback_fallbacks") != 0
            or inference_runtime.get("cache_feedback_mode") != "predicted"
        ):
            raise RuntimeError("T16 terminal AR/cache runtime telemetry differs")
        result = {
            "arm": arm,
            "elapsed_seconds": time.perf_counter() - started,
            "environment_steps": environment_steps,
            "generation_chunks": generation_chunks,
            "generation_steps": generation_steps,
            "initial_observation": initial_identity,
            "model_noise_seed": model_noise_seed,
            "noise_pair_key": noise_pair_key,
            "policy_requests": policy_requests,
            "reset_ack": reset_ack,
            "server_info": server_info,
            "simulator_reset": {
                key: simulator_reset[key] for key in ("arm", "settle_steps", "success")
            },
            "success": success,
            "terminal_done": done,
            "trace": trace,
        }
    except BaseException as error:
        rollout_error = error
        raise
    finally:
        if server is not None:
            server.shutdown()
        architecture.attach_action_normalizer(prior_normalizer)
        del server
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        architecture.init_training_schedulers(1000)
        trainer._set_training_mode()
        scheduler_after = _scheduler_snapshot(architecture)
        rng_after = _caller_rng_sha256(torch)
        mutation_after = (
            None if rollout_error is not None else probe_mutation_snapshot()
        )
        if rollout_error is None:
            if scheduler_after != scheduler_before:
                raise RuntimeError(
                    "T16 training scheduler restoration differs: "
                    f"before={scheduler_before!r} after={scheduler_after!r}"
                )
            if mutation_after != mutation_before:
                raise RuntimeError("T16 update-free rollout mutated training state")
            if rng_after != rng_before:
                raise RuntimeError("T16 rollout changed caller RNG state")
    result["caller_rng_unchanged"] = True
    result["scheduler_restoration"] = {
        "after": scheduler_after,
        "before": scheduler_before,
        "exact": True,
    }
    result["training_state_unchanged"] = True
    return result


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
                "schema_version": (
                    "sana-wam-libero-t16-paired-one-task-closed-loop-failure-v1"
                ),
            },
        )
    _freeze_run_root(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / T15_CONFIG_RELATIVE_PATH)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument(
        "--sim-worker", type=Path, default=ROOT / T16_SIM_WORKER_RELATIVE_PATH
    )
    parser.add_argument(
        "--sim-config", type=Path, default=ROOT / T16_SIM_CONFIG_RELATIVE_PATH
    )
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    parser.add_argument("--steps", type=int, default=T15_UPDATE_STEPS)
    parser.add_argument(
        "--initialization-seed", type=int, default=T15_INITIALIZATION_SEED
    )
    parser.add_argument("--loss-recipe-seed", type=int, default=T15_LOSS_RECIPE_SEED)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("GDN_DISABLE_COMPILE", "1")
    if os.environ.get("SANA_WAM_CHECKPOINT_SHA256"):
        raise RuntimeError("T16 forbids inherited SANA-WAM checkpoint identity")

    if args.steps != T15_UPDATE_STEPS:
        raise ValueError(f"T15 steps must be exactly {T15_UPDATE_STEPS}")
    if args.initialization_seed != T15_INITIALIZATION_SEED:
        raise ValueError("T15 initialization seed differs")
    if args.loss_recipe_seed != T15_LOSS_RECIPE_SEED:
        raise ValueError("T15 loss recipe seed differs")
    if args.render_gpu_device_id is None:
        args.render_gpu_device_id = args.physical_gpu
    if args.render_gpu_device_id != args.physical_gpu:
        raise ValueError("T16 simulator EGL device id must equal the physical GPU")
    if len(args.nonce) != 32 or any(
        character not in "0123456789abcdef" for character in args.nonce
    ):
        raise ValueError("T16 nonce must be exactly 32 lowercase hex characters")
    expected_root_name = f"{T16_ROOT_SLUG}-{args.nonce}"
    if args.run_root.name != expected_root_name:
        raise ValueError("T16 run-root basename does not bind the nonce")
    if args.run_root.parent.name != args.expected_repo_commit[:12]:
        raise ValueError("T16 run-root parent does not bind the source commit")
    expected_run_root = (
        T16_RUN_NAMESPACE / args.expected_repo_commit[:12] / expected_root_name
    )
    if args.run_root.expanduser() != expected_run_root:
        raise ValueError(
            "T16 run root differs from the frozen non-formal namespace: "
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
    expected_config_path = (ROOT / T15_CONFIG_RELATIVE_PATH).resolve()
    if config_path != expected_config_path:
        raise ValueError(
            f"T15 config path differs: expected={expected_config_path} got={config_path}"
        )
    sim_worker_candidate = args.sim_worker.expanduser()
    sim_config_candidate = args.sim_config.expanduser()
    if sim_worker_candidate.is_symlink() or not sim_worker_candidate.is_file():
        raise ValueError("T16 simulator worker must be a regular non-symlink file")
    if sim_config_candidate.is_symlink() or not sim_config_candidate.is_file():
        raise ValueError("T16 simulator config must be a regular non-symlink file")
    sim_worker_path = sim_worker_candidate.resolve()
    sim_config_path = sim_config_candidate.resolve()
    if sim_worker_path != (ROOT / T16_SIM_WORKER_RELATIVE_PATH).resolve():
        raise ValueError("T16 simulator worker path differs")
    if sim_config_path != (ROOT / T16_SIM_CONFIG_RELATIVE_PATH).resolve():
        raise ValueError("T16 simulator config path differs")
    runner_path = Path(__file__).resolve()
    run_root = _create_run_root(args.run_root.expanduser())
    global _ACTIVE_RUN_ROOT
    _ACTIVE_RUN_ROOT = run_root

    actual_commit = _repo_commit()
    actual_config_sha256 = _sha256_file(config_path)
    if actual_config_sha256 != T15_CONFIG_SHA256:
        raise RuntimeError("T15 config content differs from the frozen pin")
    actual_runner_sha256 = _sha256_file(runner_path)
    actual_sim_worker_sha256 = _sha256_file(sim_worker_path)
    actual_sim_config_sha256 = _sha256_file(sim_config_path)
    expected_identity = {
        "config_sha256": T15_CONFIG_SHA256,
        "repo_commit": args.expected_repo_commit.lower(),
        "runner_sha256": args.expected_runner_sha256.lower(),
        "sana_commit": EXPECTED_SANA_COMMIT,
        "sim_config_sha256": T16_SIM_CONFIG_SHA256,
        "sim_worker_sha256": T16_SIM_WORKER_SHA256,
    }
    actual_identity = {
        "config_sha256": actual_config_sha256,
        "repo_commit": actual_commit,
        "runner_sha256": actual_runner_sha256,
        "sana_commit": EXPECTED_SANA_COMMIT,
        "sim_config_sha256": actual_sim_config_sha256,
        "sim_worker_sha256": actual_sim_worker_sha256,
    }
    if actual_identity != expected_identity:
        raise RuntimeError(
            "T16 source identity differs: "
            f"expected={expected_identity!r} actual={actual_identity!r}"
        )
    actual_identity["sana_commit"] = _assert_source_tree_clean(actual_commit)

    cfg = OmegaConf.load(config_path)
    if cfg.model.architecture.variant != "autoregressive":
        raise RuntimeError("T15 requires the AR architecture")
    if cfg.model.video_backbone.continuous_timestep_conditioning is not True:
        raise RuntimeError("T15 requires continuous FP32 video timesteps")
    if cfg.training.use_gradient_checkpointing is not True:
        raise RuntimeError("T15 requires gradient checkpointing")
    if int(cfg.training.batch_size) != 1:
        raise RuntimeError("T15 requires production singleton batch_size=1")
    if int(cfg.training.gradient_accumulation_steps) != 8:
        raise RuntimeError("T15 requires production gradient_accumulation_steps=8")
    if int(cfg.training.seed) != args.initialization_seed:
        raise RuntimeError("T15 CLI initialization seed must equal training.seed")
    if int(cfg.dataloader.seed) != T15_DATALOADER_SEED:
        raise RuntimeError("T15 dataloader seed must remain the frozen T15 value")
    if tuple(cfg.training.trainable_modules) != EXPECTED_TRAINABLE_ROOTS:
        raise RuntimeError("T15 trainable module allowlist differs")
    if (
        tuple(cfg.training.preserve_frozen_input_grad_modules)
        != EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES
    ):
        raise RuntimeError("T15 frozen input-gradient preservation differs")
    if cfg.training.optimizer_master_weights is not True:
        raise RuntimeError("T15 requires persistent FP32 optimizer masters")
    if (
        float(cfg.training.lambda_video) != 0.0
        or float(cfg.training.lambda_action) != 1.0
    ):
        raise RuntimeError("T15 requires action-only loss weighting")
    if float(cfg.training.weight_decay) != 0.0:
        raise RuntimeError("T15 requires weight_decay=0")
    if (
        float(cfg.training.action_lr) != 1.0e-4
        or float(cfg.training.video_lr) != 1.0e-4
    ):
        raise RuntimeError("T15 requires fixed base learning rates of 1e-4")
    if float(cfg.training.grad_clip) != 1.0:
        raise RuntimeError("T15 requires the fixed gradient clip bound 1.0")
    forbidden_checkpoint_fields = (
        "training.init_checkpoint",
        "training.resume_checkpoint",
        "training.resume_from_checkpoint",
        "training.resume_manifest",
        "model.video_backbone.init_dit_from",
    )
    for path in forbidden_checkpoint_fields:
        if OmegaConf.select(cfg, path, default=None) is not None:
            raise RuntimeError(f"T15 forbids checkpoint field {path}")
    if str(cfg.training.action_stats_sha256) != SELECTED_STATS_SHA256:
        raise RuntimeError("T15 selected-row stats SHA pin differs")
    if str(cfg.training.action_stats_population_sha256) != SELECTED_POPULATION_SHA256:
        raise RuntimeError("T15 selected population SHA pin differs")
    stats_path = Path(str(cfg.dataloader.action_stats_path)).expanduser()
    if stats_path.is_symlink() or not stats_path.is_file():
        raise RuntimeError("T15 stats artifact must be a regular non-symlink file")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T15 selected-row stats artifact SHA differs")

    from sana_wam.train.libero_contract import validate_libero_training_config

    # This is the only full numeric live-source preflight. It runs before any
    # model construction or CUDA allocation.
    validate_libero_training_config(cfg, require_materialized_stats=True)
    external_assets = _verify_external_assets(cfg)
    configured_roots = [
        Path(str(path)).expanduser() for path in cfg.dataloader.dataset_roots
    ]
    dataset_roots = {path.name: path for path in configured_roots}
    expected_datasets = set(T13_SUITE_ORDER)
    if (
        len(dataset_roots) != len(configured_roots)
        or set(dataset_roots) != expected_datasets
    ):
        raise RuntimeError("T15 four-suite dataset root identity differs")
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
    external_assets["t11_predecessor/runner.py"] = _verify_pinned_asset(
        T11_PREDECESSOR_RUNNER_PATH,
        T11_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t11_predecessor/RESULT.json"] = _verify_pinned_asset(
        T11_PREDECESSOR_RESULT_PATH,
        T11_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t12_predecessor/runner.py"] = _verify_pinned_asset(
        T12_PREDECESSOR_RUNNER_PATH,
        T12_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t12_predecessor/RESULT.json"] = _verify_pinned_asset(
        T12_PREDECESSOR_RESULT_PATH,
        T12_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t13_predecessor/runner.py"] = _verify_pinned_asset(
        T13_PREDECESSOR_RUNNER_PATH,
        T13_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t13_predecessor/RESULT.json"] = _verify_pinned_asset(
        T13_PREDECESSOR_RESULT_PATH,
        T13_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t14_predecessor/runner.py"] = _verify_pinned_asset(
        T14_PREDECESSOR_RUNNER_PATH,
        T14_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t14_predecessor/RESULT.json"] = _verify_pinned_asset(
        T14_PREDECESSOR_RESULT_PATH,
        T14_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t15_predecessor/runner.py"] = _verify_pinned_asset(
        T15_PREDECESSOR_RUNNER_PATH,
        T15_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t15_predecessor/RESULT.json"] = _verify_pinned_asset(
        T15_PREDECESSOR_RESULT_PATH,
        T15_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t16/simulator_worker.py"] = _verify_pinned_asset(
        sim_worker_path,
        T16_SIM_WORKER_SHA256,
    )
    external_assets["t16/simulator_config.yaml"] = _verify_pinned_asset(
        sim_config_path,
        T16_SIM_CONFIG_SHA256,
    )
    for dataset in T13_SUITE_ORDER:
        for filename, expected_sha256 in sorted(
            T13_SUITE_METADATA_SHA256[dataset].items()
        ):
            external_assets[f"cross_suite_metadata/{dataset}/meta/{filename}"] = (
                _verify_pinned_asset(
                    dataset_roots[dataset] / "meta" / filename,
                    expected_sha256,
                )
            )
    for role, samples in (
        ("update", T13_TRAIN_SAMPLES),
        ("heldout", T13_HELDOUT_SAMPLES),
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
        raise RuntimeError("T15 predecessor T2 evidence semantics differ")
    t3_predecessor = json.loads(T3_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t3_predecessor.get("valid_run") is not True
        or t3_predecessor.get("scientific_verdict") != "T3_HELDOUT_RECIPE_TRANSFER_GO"
        or t3_predecessor.get("identity", {}).get("repo_commit")
        != T3_PREDECESSOR_SOURCE_COMMIT
        or t3_predecessor.get("identity", {}).get("runner_sha256")
        != T3_PREDECESSOR_RUNNER_SHA256
        or t3_predecessor.get("scope", {}).get("architecture_forwards_total") != 28
        or t3_predecessor.get("scope", {}).get("backward_calls") != T15_UPDATE_STEPS
        or t3_predecessor.get("scope", {}).get("optimizer_steps") != T15_UPDATE_STEPS
    ):
        raise RuntimeError("T15 predecessor T3 evidence semantics differ")
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
        or t4_predecessor.get("scope", {}).get("backward_calls") != T15_UPDATE_STEPS
        or t4_predecessor.get("scope", {}).get("optimizer_steps") != T15_UPDATE_STEPS
    ):
        raise RuntimeError("T15 predecessor T4 evidence semantics differ")
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
        or t5_predecessor.get("scope", {}).get("backward_calls") != T15_UPDATE_STEPS
        or t5_predecessor.get("scope", {}).get("optimizer_steps") != T15_UPDATE_STEPS
    ):
        raise RuntimeError("T15 predecessor T5 evidence semantics differ")
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
        or t6_predecessor.get("scope", {}).get("backward_calls") != T15_UPDATE_STEPS
        or t6_predecessor.get("scope", {}).get("optimizer_steps") != T15_UPDATE_STEPS
    ):
        raise RuntimeError("T15 predecessor T6 evidence semantics differ")
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
        raise RuntimeError("T15 frozen T3/T4/T5/T6 training-core evidence differs")
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
        or t7_predecessor.get("identity", {}).get("config_sha256")
        != BASELINE_CONFIG_SHA256
        or t7_predecessor.get("scope", {}).get("architecture_forwards_total") != 36
        or t7_predecessor.get("scope", {}).get("backward_calls") != 20
        or t7_predecessor.get("scope", {}).get("optimizer_steps") != 20
        or t7_pre_losses != {**T7_PRE_ACTION_LOSSES, **T7_PRE_HELDOUT_ACTION_LOSSES}
        or t7_terminal_losses != T7_TERMINAL_ACTION_LOSSES
    ):
        raise RuntimeError("T15 predecessor T7 evidence semantics differ")
    t9_aggregate = json.loads(T9_AGGREGATE_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t9_aggregate.get("valid_run") is not True
        or t9_aggregate.get("scientific_verdict")
        != "T9_COMMON_POSITION_EFFECT_SUPPORTED"
        or t9_aggregate.get("identity", {}).get("repo_commit")
        != T9_AGGREGATE_SOURCE_COMMIT
        or t9_aggregate.get("identity", {}).get("runner_sha256")
        != T9_AGGREGATE_RUNNER_SHA256
        or t9_aggregate.get("identity", {}).get("config_sha256")
        != BASELINE_CONFIG_SHA256
        or t9_aggregate.get("run", {}).get("root") != str(T9_AGGREGATE_ROOT)
        or t9_aggregate.get("scope", {}).get("input_result_count") != 4
        or t9_aggregate.get("scope", {}).get("gpu_or_model_executed") is not False
    ):
        raise RuntimeError("T15 direct predecessor T9 aggregate semantics differ")
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
            or predecessor.get("identity", {}).get("config_sha256")
            != BASELINE_CONFIG_SHA256
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
            raise RuntimeError(f"T15 predecessor T10 {arm} arm semantics differ")
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
        or t10_aggregate.get("identity", {}).get("config_sha256")
        != BASELINE_CONFIG_SHA256
        or t10_aggregate.get("run", {}).get("root") != str(T10_AGGREGATE_ROOT)
        or t10_aggregate.get("scope", {}).get("input_result_count") != 2
        or t10_aggregate.get("scope", {}).get("gpu_or_model_executed") is not False
        or t10_aggregate.get("arm_identities", {}).get("SEQ", {}).get("result_sha256")
        != T10_SEQ_RESULT_SHA256
        or t10_aggregate.get("arm_identities", {}).get("JOINT", {}).get("result_sha256")
        != T10_JOINT_RESULT_SHA256
    ):
        raise RuntimeError("T15 underlying predecessor T10 aggregate semantics differ")
    if (
        T11_PREDECESSOR_ROOT.is_symlink()
        or not T11_PREDECESSOR_ROOT.is_dir()
        or (T11_PREDECESSOR_ROOT.stat().st_mode & 0o777) != 0o500
        or (T11_PREDECESSOR_RESULT_PATH.stat().st_mode & 0o777) != 0o400
        or sorted(path.name for path in T11_PREDECESSOR_ROOT.iterdir())
        != ["RESULT.json"]
    ):
        raise RuntimeError("T15 underlying predecessor T11 terminal root differs")
    t11_predecessor = json.loads(
        T11_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8")
    )
    t11_q_by_suite = {
        row["dataset"]: row["q"]
        for row in t11_predecessor.get("scientific_classification", {}).get(
            "per_suite", []
        )
    }
    if (
        t11_predecessor.get("valid_run") is not True
        or t11_predecessor.get("verdict") != "T11_NEW_EPISODE_JOINT_ARM_VALID"
        or t11_predecessor.get("execution_verdict") != "T11_NEW_EPISODE_JOINT_ARM_VALID"
        or t11_predecessor.get("scientific_verdict")
        != "T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED"
        or t11_predecessor.get("identity", {}).get("repo_commit")
        != T11_PREDECESSOR_SOURCE_COMMIT
        or t11_predecessor.get("identity", {}).get("runner_sha256")
        != T11_PREDECESSOR_RUNNER_SHA256
        or t11_predecessor.get("identity", {}).get("config_sha256")
        != BASELINE_CONFIG_SHA256
        or t11_predecessor.get("run", {}).get("root") != str(T11_PREDECESSOR_ROOT)
        or t11_predecessor.get("scope", {}).get("architecture_forwards_total") != 96
        or t11_predecessor.get("scope", {}).get("architecture_training_forwards") != 80
        or t11_predecessor.get("scope", {}).get("backward_calls") != 80
        or t11_predecessor.get("scope", {}).get("optimizer_steps") != 20
        or t11_predecessor.get("scope", {}).get("sana_wam_training_checkpoint_loaded")
        is not False
        or t11_predecessor.get("scope", {}).get("sana_wam_training_checkpoint_saved")
        is not False
        or t11_q_by_suite != T11_Q_BY_SUITE
    ):
        raise RuntimeError("T15 underlying predecessor T11 semantics differ")
    if (
        T12_PREDECESSOR_ROOT.is_symlink()
        or not T12_PREDECESSOR_ROOT.is_dir()
        or (T12_PREDECESSOR_ROOT.stat().st_mode & 0o777) != 0o500
        or (T12_PREDECESSOR_RESULT_PATH.stat().st_mode & 0o777) != 0o400
        or sorted(path.name for path in T12_PREDECESSOR_ROOT.iterdir())
        != ["RESULT.json"]
    ):
        raise RuntimeError("T15 underlying predecessor T12 terminal root differs")
    t12_predecessor = json.loads(
        T12_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8")
    )
    t12_q_by_suite = {
        row["dataset"]: row["q"]
        for row in t12_predecessor.get("scientific_classification", {}).get(
            "per_suite", []
        )
    }
    if (
        t12_predecessor.get("valid_run") is not True
        or t12_predecessor.get("verdict") != "T12_NEW_TASK_JOINT_ARM_VALID"
        or t12_predecessor.get("execution_verdict") != "T12_NEW_TASK_JOINT_ARM_VALID"
        or t12_predecessor.get("scientific_verdict")
        != "T12_NEW_TASK_BALANCED_JOINT_REPLICATED"
        or t12_predecessor.get("identity", {}).get("repo_commit")
        != T12_PREDECESSOR_SOURCE_COMMIT
        or t12_predecessor.get("identity", {}).get("runner_sha256")
        != T12_PREDECESSOR_RUNNER_SHA256
        or t12_predecessor.get("identity", {}).get("config_sha256")
        != BASELINE_CONFIG_SHA256
        or t12_predecessor.get("run", {}).get("root") != str(T12_PREDECESSOR_ROOT)
        or t12_predecessor.get("scope", {}).get("architecture_forwards_total") != 96
        or t12_predecessor.get("scope", {}).get("architecture_training_forwards") != 80
        or t12_predecessor.get("scope", {}).get("backward_calls") != 80
        or t12_predecessor.get("scope", {}).get("optimizer_steps") != 20
        or t12_predecessor.get("scope", {}).get("prepare_inputs_calls") != 8
        or t12_predecessor.get("scope", {}).get("sana_wam_training_checkpoint_loaded")
        is not False
        or t12_predecessor.get("scope", {}).get("sana_wam_training_checkpoint_saved")
        is not False
        or t12_predecessor.get("randomness", {}).get(
            "selected_new_task_manifest_sha256"
        )
        != T12_SELECTION_MANIFEST_SHA256
        or t12_q_by_suite != T12_Q_BY_SUITE
    ):
        raise RuntimeError("T15 underlying predecessor T12 semantics differ")
    if (
        T13_PREDECESSOR_ROOT.is_symlink()
        or not T13_PREDECESSOR_ROOT.is_dir()
        or (T13_PREDECESSOR_ROOT.stat().st_mode & 0o777) != 0o500
        or (T13_PREDECESSOR_RESULT_PATH.stat().st_mode & 0o777) != 0o400
        or sorted(path.name for path in T13_PREDECESSOR_ROOT.iterdir())
        != ["RESULT.json"]
    ):
        raise RuntimeError("T15 underlying predecessor T13 terminal root differs")
    t13_predecessor = json.loads(
        T13_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8")
    )
    t13_q_by_suite = {
        row["dataset"]: row["q"]
        for row in t13_predecessor.get("scientific_classification", {}).get(
            "per_suite", []
        )
    }
    if (
        t13_predecessor.get("valid_run") is not True
        or t13_predecessor.get("verdict") != "T13_MULTI_EPISODE_ACCUM8_JOINT_ARM_VALID"
        or t13_predecessor.get("execution_verdict")
        != "T13_MULTI_EPISODE_ACCUM8_JOINT_ARM_VALID"
        or t13_predecessor.get("scientific_verdict")
        != "T13_MULTI_EPISODE_ACCUM8_BALANCED_JOINT_REPLICATED"
        or t13_predecessor.get("identity", {}).get("repo_commit")
        != T13_PREDECESSOR_SOURCE_COMMIT
        or t13_predecessor.get("identity", {}).get("runner_sha256")
        != T13_PREDECESSOR_RUNNER_SHA256
        or t13_predecessor.get("identity", {}).get("config_sha256")
        != BASELINE_CONFIG_SHA256
        or t13_predecessor.get("run", {}).get("root") != str(T13_PREDECESSOR_ROOT)
        or t13_predecessor.get("scope", {}).get("architecture_forwards_total")
        != T15_ARCHITECTURE_FORWARDS
        or t13_predecessor.get("scope", {}).get("architecture_training_forwards")
        != T15_TRAINING_FORWARDS
        or t13_predecessor.get("scope", {}).get("architecture_measurement_forwards")
        != T15_MEASUREMENT_FORWARDS
        or t13_predecessor.get("scope", {}).get("backward_calls") != 160
        or t13_predecessor.get("scope", {}).get("optimizer_steps") != 20
        or t13_predecessor.get("scope", {}).get("prepare_inputs_calls") != 16
        or t13_predecessor.get("scope", {}).get("sana_wam_training_checkpoint_loaded")
        is not False
        or t13_predecessor.get("scope", {}).get("sana_wam_training_checkpoint_saved")
        is not False
        or t13_predecessor.get("randomness", {}).get("initialization_seed") != 20260806
        or t13_predecessor.get("randomness", {}).get("training_loss_recipe_seed")
        != T14_PREDECESSOR_LOSS_RECIPE_SEED
        or t13_predecessor.get("randomness", {}).get(
            "selected_multi_episode_manifest_sha256"
        )
        != T13_SELECTION_MANIFEST_SHA256
        or t13_q_by_suite != T13_Q_BY_SUITE
    ):
        raise RuntimeError("T15 underlying predecessor T13 semantics differ")
    if (
        T14_PREDECESSOR_ROOT.is_symlink()
        or not T14_PREDECESSOR_ROOT.is_dir()
        or (T14_PREDECESSOR_ROOT.stat().st_mode & 0o777) != 0o500
        or (T14_PREDECESSOR_RESULT_PATH.stat().st_mode & 0o777) != 0o400
        or sorted(path.name for path in T14_PREDECESSOR_ROOT.iterdir())
        != ["RESULT.json"]
    ):
        raise RuntimeError("T15 direct predecessor T14 terminal root differs")
    t14_predecessor = json.loads(
        T14_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8")
    )
    t14_classification = t14_predecessor.get("scientific_classification", {})
    t14_q_by_suite = {
        row["dataset"]: row["q"] for row in t14_classification.get("per_suite", [])
    }
    t14_randomness = t14_predecessor.get("randomness", {})
    t14_scope = t14_predecessor.get("scope", {})
    if (
        t14_predecessor.get("valid_run") is not True
        or t14_predecessor.get("verdict")
        != "T14_INITSEED_REPLICATION_ACCUM8_JOINT_ARM_VALID"
        or t14_predecessor.get("execution_verdict")
        != "T14_INITSEED_REPLICATION_ACCUM8_JOINT_ARM_VALID"
        or t14_predecessor.get("scientific_verdict")
        != "T14_INITSEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED"
        or t14_predecessor.get("identity", {}).get("repo_commit")
        != T14_PREDECESSOR_SOURCE_COMMIT
        or t14_predecessor.get("identity", {}).get("runner_sha256")
        != T14_PREDECESSOR_RUNNER_SHA256
        or t14_predecessor.get("identity", {}).get("config_sha256") != T15_CONFIG_SHA256
        or t14_predecessor.get("run", {}).get("root") != str(T14_PREDECESSOR_ROOT)
        or t14_scope.get("architecture_forwards_total") != T15_ARCHITECTURE_FORWARDS
        or t14_scope.get("architecture_training_forwards") != T15_TRAINING_FORWARDS
        or t14_scope.get("architecture_measurement_forwards")
        != T15_MEASUREMENT_FORWARDS
        or t14_scope.get("backward_calls") != 160
        or t14_scope.get("optimizer_steps") != 20
        or t14_scope.get("prepare_inputs_calls") != 16
        or t14_scope.get("formal_training_executed") is not False
        or t14_scope.get("benchmark_evaluation_executed") is not False
        or t14_scope.get("simulator_executed") is not False
        or t14_scope.get("sana_wam_training_checkpoint_loaded") is not False
        or t14_scope.get("sana_wam_training_checkpoint_saved") is not False
        or t14_randomness.get("initialization_seed")
        != T14_PREDECESSOR_INITIALIZATION_SEED
        or t14_randomness.get("training_loss_recipe_seed")
        != T14_PREDECESSOR_LOSS_RECIPE_SEED
        or t14_randomness.get("training_recipe_signature_sha256")
        != T14_PREDECESSOR_RECIPE_SIGNATURE_SHA256
        or t14_randomness.get("total_recipe_signatures") != T15_ARCHITECTURE_FORWARDS
        or t14_randomness.get("unique_loss_recipe_signature_count") != 1
        or t14_randomness.get("selected_multi_episode_manifest_sha256")
        != T13_SELECTION_MANIFEST_SHA256
        or t14_randomness.get("selected_multi_episode_manifest") != selection_manifest
        or t14_classification.get("all_sixteen_ratios_strictly_below_one") is not True
        or t14_q_by_suite != T14_Q_BY_SUITE
    ):
        raise RuntimeError("T15 direct predecessor T14 semantics differ")
    if (
        T15_PREDECESSOR_ROOT.is_symlink()
        or not T15_PREDECESSOR_ROOT.is_dir()
        or (T15_PREDECESSOR_ROOT.stat().st_mode & 0o777) != 0o500
        or (T15_PREDECESSOR_RESULT_PATH.stat().st_mode & 0o777) != 0o400
        or sorted(path.name for path in T15_PREDECESSOR_ROOT.iterdir())
        != ["RESULT.json"]
    ):
        raise RuntimeError("T16 direct predecessor T15 terminal root differs")
    t15_predecessor = json.loads(
        T15_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8")
    )
    t15_scope = t15_predecessor.get("scope", {})
    t15_randomness = t15_predecessor.get("randomness", {})
    if (
        t15_predecessor.get("valid_run") is not True
        or t15_predecessor.get("verdict")
        != "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_ARM_VALID"
        or t15_predecessor.get("execution_verdict")
        != "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_ARM_VALID"
        or t15_predecessor.get("scientific_verdict")
        != "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED"
        or t15_predecessor.get("identity", {}).get("repo_commit")
        != T15_PREDECESSOR_SOURCE_COMMIT
        or t15_predecessor.get("identity", {}).get("runner_sha256")
        != T15_PREDECESSOR_RUNNER_SHA256
        or t15_predecessor.get("identity", {}).get("config_sha256") != T15_CONFIG_SHA256
        or t15_predecessor.get("run", {}).get("root") != str(T15_PREDECESSOR_ROOT)
        or t15_scope.get("architecture_forwards_total") != T15_ARCHITECTURE_FORWARDS
        or t15_scope.get("architecture_training_forwards") != T15_TRAINING_FORWARDS
        or t15_scope.get("architecture_measurement_forwards")
        != T15_MEASUREMENT_FORWARDS
        or t15_scope.get("backward_calls") != T15_TRAINING_FORWARDS
        or t15_scope.get("optimizer_steps") != T15_UPDATE_STEPS
        or t15_scope.get("prepare_inputs_calls") != 16
        or t15_scope.get("simulator_executed") is not False
        or t15_scope.get("formal_training_executed") is not False
        or t15_scope.get("benchmark_evaluation_executed") is not False
        or t15_scope.get("sana_wam_training_checkpoint_loaded") is not False
        or t15_scope.get("sana_wam_training_checkpoint_saved") is not False
        or t15_randomness.get("initialization_seed") != T15_INITIALIZATION_SEED
        or t15_randomness.get("training_loss_recipe_seed") != T15_LOSS_RECIPE_SEED
        or t15_randomness.get("training_recipe_signature_sha256")
        != T15_PREDECESSOR_RECIPE_SIGNATURE_SHA256
        or t15_randomness.get("total_recipe_signatures") != T15_ARCHITECTURE_FORWARDS
        or t15_randomness.get("unique_loss_recipe_signature_count") != 1
    ):
        raise RuntimeError("T16 direct predecessor T15 semantics differ")
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
            "t11_predecessor_result_sha256": T11_PREDECESSOR_RESULT_SHA256,
            "t11_predecessor_runner_sha256": T11_PREDECESSOR_RUNNER_SHA256,
            "t11_predecessor_source_commit": T11_PREDECESSOR_SOURCE_COMMIT,
            "t12_predecessor_result_sha256": T12_PREDECESSOR_RESULT_SHA256,
            "t12_predecessor_runner_sha256": T12_PREDECESSOR_RUNNER_SHA256,
            "t12_predecessor_source_commit": T12_PREDECESSOR_SOURCE_COMMIT,
            "t13_predecessor_result_sha256": T13_PREDECESSOR_RESULT_SHA256,
            "t13_predecessor_runner_sha256": T13_PREDECESSOR_RUNNER_SHA256,
            "t13_predecessor_source_commit": T13_PREDECESSOR_SOURCE_COMMIT,
            "t14_predecessor_result_sha256": T14_PREDECESSOR_RESULT_SHA256,
            "t14_predecessor_runner_sha256": T14_PREDECESSOR_RUNNER_SHA256,
            "t14_predecessor_source_commit": T14_PREDECESSOR_SOURCE_COMMIT,
            "t15_predecessor_result_sha256": T15_PREDECESSOR_RESULT_SHA256,
            "t15_predecessor_runner_sha256": T15_PREDECESSOR_RUNNER_SHA256,
            "t15_predecessor_source_commit": T15_PREDECESSOR_SOURCE_COMMIT,
            "t16_selection_manifest_sha256": T16_SELECTION_MANIFEST_SHA256,
            "t16_init_selection_payload_sha256": (T16_INIT_SELECTION_PAYLOAD_SHA256),
            "t16_t15_update_core_ast_sha256": T16_T15_UPDATE_CORE_AST_SHA256,
            "eligible_manifest_sha256": T13_ELIGIBLE_MANIFEST_SHA256,
            "selection_manifest_sha256": T13_SELECTION_MANIFEST_SHA256,
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
        raise RuntimeError(f"T15 selected GPU is not an H200: {gpu_before['name']}")

    import torch

    from sana_wam.dataloader.transforms.multiview import format_prompt_for_inference
    from sana_wam.train.trainer import Trainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "T15 requires exactly one visible CUDA GPU; "
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
        raise RuntimeError("T15 video-backbone parameters are not frozen")
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
        raise RuntimeError(f"T15 trainable parameter roots differ: {observed_roots!r}")
    if len(named_trainables) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        raise RuntimeError("T15 trainable tensor count differs")
    if sum(parameter.numel() for _name, parameter in named_trainables) != (
        EXPECTED_TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("T15 trainable parameter count differs")
    frozen_versions = {name: parameter._version for name, parameter in named_frozen}

    role_selections = _select_t13_samples(
        trainer.dataset,
        eligible_sample_manifest,
        selection_manifest,
    )
    raw_samples: dict[str, dict[str, Any]] = {}
    sample_seconds: dict[str, float] = {}
    role_assets_after_sample: dict[str, str] = {}
    role_order = (*T13_TRAIN_LABELS, *T13_HELDOUT_LABELS)
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
            raise RuntimeError(f"T15 LIBERO sample metadata differs: {label}")
        raw_samples[label] = raw_sample
        role = "update" if label in T13_TRAIN_LABELS else "heldout"
        for relative_path, expected_sha256 in sorted(selected["assets"].items()):
            key = f"{role}_sample/{selected['dataset']}/{label}/{relative_path}"
            observed = _verify_pinned_asset(
                dataset_roots[selected["dataset"]] / relative_path,
                expected_sha256,
            )
            if external_assets[key] != observed:
                raise RuntimeError(f"T15 role asset changed while loading: {key}")
            role_assets_after_sample[key] = observed
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T15 selected-row stats changed while loading")
    actual_identity["role_assets_post_sample_sha256"] = _sha256_json(
        role_assets_after_sample
    )

    if architecture.training is not True:
        raise RuntimeError("T15 Trainer did not leave the architecture in train mode")
    if any(module.training for module in video_backbone.modules()):
        raise RuntimeError("T15 preserved video backbone left eval mode")
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
            raise RuntimeError(f"T15 {label} inputs dropped gradient checkpointing")
        if prepared.get("use_gradient_checkpointing_offload") is not False:
            raise RuntimeError("T15 requires checkpoint offload=false")
        invalid_tensors = [
            key
            for key, value in prepared.items()
            if isinstance(value, torch.Tensor)
            and (value.requires_grad or value.grad_fn is not None)
        ]
        if invalid_tensors:
            raise RuntimeError(
                f"T15 {label} prepared tensors retain autograd state: "
                f"{invalid_tensors!r}"
            )
        return prepared

    prepared_by_label = {
        label: prepare_one(label, raw_samples[label]) for label in role_order
    }
    cyclic_training_inputs = {
        label: prepared_by_label[label] for label in T13_TRAIN_LABELS
    }
    fresh_heldout_inputs = {
        label: prepared_by_label[label] for label in T13_HELDOUT_LABELS
    }
    metadata_after_sample = {}
    for dataset_name in T13_SUITE_ORDER:
        for filename, expected_sha256 in sorted(
            T13_SUITE_METADATA_SHA256[dataset_name].items()
        ):
            key = f"cross_suite_metadata/{dataset_name}/meta/{filename}"
            observed = _verify_pinned_asset(
                dataset_roots[dataset_name] / "meta" / filename,
                expected_sha256,
            )
            if external_assets[key] != observed:
                raise RuntimeError(f"T15 metadata changed while loading: {key}")
            metadata_after_sample[key] = observed
    actual_identity["cross_suite_metadata_post_sample_sha256"] = _sha256_json(
        metadata_after_sample
    )
    if prepare_inputs_calls != 16:
        raise RuntimeError(
            f"T15 prepare_inputs call count differs: {prepare_inputs_calls}"
        )
    eligible_manifest = eligible_sample_manifest
    live_manifest = _eligible_sample_manifest(dataset_roots)
    if live_manifest != eligible_manifest:
        raise RuntimeError(
            "T15 live eligible manifest changed during sample preparation"
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
                    "T15 prepared tensors alias across samples: "
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
    grouped_contexts_match = all(
        len({contexts_by_label[label]["context_sha256"] for label in group}) == 1
        and len({contexts_by_label[label]["dataset"] for label in group}) == 1
        and len({contexts_by_label[label]["task_index"] for label in group}) == 1
        and len({contexts_by_label[label]["task"] for label in group}) == 1
        for group in T13_CONTEXT_GROUPS
    )
    if (
        len(contexts_by_label) != 16
        or len(context_digests) != 4
        or not grouped_contexts_match
    ):
        raise RuntimeError("T15 per-task four-way context grouping differs")

    model_groups = trainer._param_groups()
    if len(model_groups) != 2 or any(
        float(group["lr"]) != 1.0e-4 for group in model_groups
    ):
        raise RuntimeError("T15 optimizer group/LR structure differs")
    params = [parameter for group in model_groups for parameter in group["params"]]
    parameter_ids = [id(parameter) for parameter in params]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {
        id(parameter) for _name, parameter in named_trainables
    }:
        raise RuntimeError("T15 optimizer parameters differ from trainables")
    optimizer_groups, master_pairs = trainer._optimizer_param_groups(model_groups)
    if len(master_pairs) != len(params):
        raise RuntimeError("T15 FP32 master coverage differs")
    if [id(model) for model, _master in master_pairs] != parameter_ids:
        raise RuntimeError("T15 FP32 master/model ordering differs")
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
        raise RuntimeError("T15 FP32 master identity/dtype differs")
    if [
        id(parameter) for group in optimizer_groups for parameter in group["params"]
    ] != initial_master_ids:
        raise RuntimeError("T15 optimizer groups do not contain the FP32 masters")
    if [len(group["params"]) for group in optimizer_groups] != [
        len(group["params"]) for group in model_groups
    ] or [float(group["lr"]) for group in optimizer_groups] != [
        float(group["lr"]) for group in model_groups
    ]:
        raise RuntimeError("T15 FP32 master optimizer group structure differs")
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
            raise RuntimeError("T15 update-free probe observed model gradients")
        if any(parameter.grad is not None for _name, parameter in named_frozen):
            raise RuntimeError("T15 update-free probe observed frozen gradients")
        if any(master.grad is not None for _name, master in named_masters):
            raise RuntimeError("T15 update-free probe observed master gradients")
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

    from benchmarks.libero.sana_wam2libero_interface import derive_model_noise_seed

    model_noise_seed = derive_model_noise_seed(
        base_seed=T16_MODEL_NOISE_BASE_SEED,
        suite=T16_TASK_SUITE,
        task_id=T16_LIBERO_SUITE_TASK_INDEX,
        trial_index=0,
        init_state_index=T16_INIT_STATE_INDEX,
    )
    noise_pair_key = (
        "t16/libero_spatial/suite-task3/dataset-task5/init17/paired-pre-post/"
        "fresh-t15-update"
    )
    sim_worker = T16SimulatorWorkerClient(
        worker_path=sim_worker_path,
        config_path=sim_config_path,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    global _ACTIVE_SIM_WORKER
    _ACTIVE_SIM_WORKER = sim_worker
    pre_closed_loop = _run_closed_loop_rollout(
        arm="pre",
        architecture=architecture,
        cfg=cfg,
        model_noise_seed=model_noise_seed,
        noise_pair_key=noise_pair_key,
        probe_mutation_snapshot=probe_mutation_snapshot,
        sim_worker=sim_worker,
        trainer=trainer,
    )
    # Keep the PRE→training boundary explicit in the top-level harness.  The
    # rollout helper already verifies a first exact restoration; this second
    # deterministic rebuild is the final gate immediately before the frozen
    # T15 update core.
    architecture.init_training_schedulers(1000)
    trainer._set_training_mode()
    explicit_training_scheduler_state = _scheduler_snapshot(architecture)
    if not all(
        row["training"]
        and row["timesteps_sha256"] is not None
        and row["sigmas_sha256"] is not None
        and row["linear_timesteps_weights_sha256"] is not None
        for row in explicit_training_scheduler_state.values()
    ):
        raise RuntimeError("T16 explicit pre-update training scheduler gate differs")
    if architecture.training is not True or any(
        module.training for module in video_backbone.modules()
    ):
        raise RuntimeError("T16 explicit pre-update module-mode gate differs")

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
                    "T15 produced a scalar loss that is not finite and strictly "
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
                raise RuntimeError("T15 total/action loss identity differs")
            signature = _sha256_json(recipe_capture)
            if not recipe_capture["action"] or not recipe_capture["video"]:
                raise RuntimeError("T15 stochastic recipe capture is incomplete")
            if backward_scale is not None:
                if backward_scale not in {
                    0.0,
                    T15_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO,
                    1.0,
                }:
                    raise RuntimeError("T15 backward scale differs from matched core")
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
            raise RuntimeError("T15 fixed recipe did not restore caller RNG state")
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
            raise RuntimeError("T15 optimizer state exists before pre probes")
        cyclic_pre_state = probe_mutation_snapshot()
        for label in T13_TRAIN_LABELS:
            scalars, signature = fixed_forward(
                backward_scale=None,
                forward_inputs=cyclic_training_inputs[label],
                recipe_seed=args.loss_recipe_seed,
            )
            measurement_before[label] = scalars
            measurement_signatures[label].append(signature)
        if probe_mutation_snapshot() != cyclic_pre_state or optimizer.state:
            raise RuntimeError("T15 cyclic training pre probes mutated training state")
        fresh_pre_state = probe_mutation_snapshot()
        for label in T13_HELDOUT_LABELS:
            scalars, signature = fixed_forward(
                backward_scale=None,
                forward_inputs=fresh_heldout_inputs[label],
                recipe_seed=args.loss_recipe_seed,
            )
            measurement_before[label] = scalars
            measurement_signatures[label].append(signature)
        if probe_mutation_snapshot() != fresh_pre_state or optimizer.state:
            raise RuntimeError("T15 fresh held-out pre probes mutated training state")
        for step_index in range(T15_UPDATE_STEPS):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for parameter in params:
                parameter.grad = None
            macro_state_before_microbacks = macro_non_gradient_state_snapshot()
            step_scalars_by_label: dict[str, dict[str, float]] = {}
            step_recipe_by_label: dict[str, str] = {}
            step_loss_weights = _loss_weights_for_macro(step_index)
            if set(step_loss_weights) != set(T13_TRAIN_LABELS) or not math.isclose(
                sum(step_loss_weights.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "T15 per-macro loss weights are not exactly unit-sum"
                )
            for update_label in T13_TRAIN_LABELS:
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
                            "T15 fixed initial loss recipe is not reproducible "
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
                for label in T13_TRAIN_LABELS
            )
            joint_action_loss = sum(
                step_scalars_by_label[label]["loss_action"] * step_loss_weights[label]
                for label in T13_TRAIN_LABELS
            )
            joint_tolerance = 1.0e-6 + 1.0e-6 * abs(joint_action_loss)
            if abs(joint_loss - joint_action_loss) > joint_tolerance:
                raise RuntimeError("T15 joint total/action loss identity differs")

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
                        f"T15 non-finite gradient at step {step_index + 1}: {name}"
                    )
                maximum = float(gradient.detach().abs().max().float().item())
                max_gradient_abs = max(max_gradient_abs, maximum)
                if maximum > 0.0:
                    nonzero_gradient_tensors += 1
                    nonzero_gradient_roots.add(_parameter_root(name))
            if missing_gradients:
                raise RuntimeError(
                    f"T15 missing gradients at step {step_index + 1}: "
                    f"{missing_gradients[:8]!r}"
                )
            if nonzero_gradient_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    f"T15 nonzero-gradient roots differ at step {step_index + 1}: "
                    f"{sorted(nonzero_gradient_roots)!r}"
                )
            if any(parameter.grad is not None for _name, parameter in named_frozen):
                raise RuntimeError("T15 frozen parameters received gradients")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, float(cfg.training.grad_clip)
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("T15 gradient norm is non-finite")
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
                        "T15 first-step masters lack finite nonzero FP32 gradients: "
                        f"{invalid_master_gradients[:8]!r}"
                    )
                master_probes = _capture_update_probes(named_masters)
                model_probes = _capture_update_probes(named_trainables)
                if {row["name"] for row in master_probes} != {
                    name for name, _master in named_masters
                }:
                    raise RuntimeError("T15 first-step master probe coverage differs")

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
                            f"T15 lacks four update sentinels for {root}"
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
                        "T15 first step did not update every FP32 master"
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
                    raise RuntimeError("T15 update sentinel became non-finite")
                sentinel_updates.append(row)
            changed_master_sentinels = [
                row for row in sentinel_updates if row["master_delta"] != 0.0
            ]
            changed_master_sentinel_roots = {
                row["root"] for row in changed_master_sentinels
            }
            if changed_master_sentinel_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    "T15 sampled FP32 master update roots differ at step "
                    f"{step_index + 1}: "
                    f"{sorted(changed_master_sentinel_roots)!r}"
                )
            sentinel_adam_steps = {
                _tensor_scalar(optimizer.state[probe["parameter"]]["step"])
                for probe in update_sentinels
            }
            if sentinel_adam_steps != {float(step_index + 1)}:
                raise RuntimeError(
                    "T15 sampled AdamW step counters differ at step "
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
                    f"T15 BF16 projection differs: {projection_mismatches[:8]!r}"
                )
            projection_exact_steps.append(step_index + 1)
            if not _optimizer_state_is_finite(optimizer):
                raise RuntimeError("T15 AdamW state is non-finite")
            if step_index == 0:
                first_projected_update, _changed_model_ids = _summarize_updates(
                    model_probes
                )
            torch.cuda.synchronize(device)
            per_step.append(
                {
                    "component_datasets": {
                        label: role_selections[label]["dataset"]
                        for label in T13_TRAIN_LABELS
                    },
                    "component_episode_indices": {
                        label: role_selections[label]["episode_index"]
                        for label in T13_TRAIN_LABELS
                    },
                    "component_losses": step_scalars_by_label,
                    "component_recipe_signatures": step_recipe_by_label,
                    "component_task_indices": {
                        label: role_selections[label]["task_index"]
                        for label in T13_TRAIN_LABELS
                    },
                    "loss_weights": step_loss_weights,
                    "gradient_nonzero_roots": sorted(nonzero_gradient_roots),
                    "gradient_nonzero_tensor_count": nonzero_gradient_tensors,
                    "labels": list(T13_TRAIN_LABELS),
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
        for label in T13_TRAIN_LABELS:
            scalars, signature = fixed_forward(
                backward_scale=None,
                forward_inputs=cyclic_training_inputs[label],
                recipe_seed=args.loss_recipe_seed,
            )
            measurement_after[label] = scalars
            measurement_signatures[label].append(signature)
        if probe_mutation_snapshot() != cyclic_post_state:
            raise RuntimeError("T15 cyclic training post probes mutated training state")
        fresh_post_state = probe_mutation_snapshot()
        for label in T13_HELDOUT_LABELS:
            scalars, signature = fixed_forward(
                backward_scale=None,
                forward_inputs=fresh_heldout_inputs[label],
                recipe_seed=args.loss_recipe_seed,
            )
            measurement_after[label] = scalars
            measurement_signatures[label].append(signature)
        if probe_mutation_snapshot() != fresh_post_state:
            raise RuntimeError("T15 fresh held-out post probes mutated training state")
    finally:
        video_embedder.forward = original_video_embedder_forward
        architecture.action_backbone.prepare_state = original_action_prepare_state
        architecture.forward = original_architecture_forward

    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    post_closed_loop = _run_closed_loop_rollout(
        arm="post",
        architecture=architecture,
        cfg=cfg,
        model_noise_seed=model_noise_seed,
        noise_pair_key=noise_pair_key,
        probe_mutation_snapshot=probe_mutation_snapshot,
        sim_worker=sim_worker,
        trainer=trainer,
    )
    simulator_close = sim_worker.close()
    _ACTIVE_SIM_WORKER = None
    if (
        pre_closed_loop["initial_observation"]
        != post_closed_loop["initial_observation"]
    ):
        raise RuntimeError("T16 PRE/POST simulator reset fingerprints differ")
    pre_noise_signatures = [
        row["generator_state_sha256"] for row in pre_closed_loop["generation_chunks"]
    ]
    post_noise_signatures = [
        row["generator_state_sha256"] for row in post_closed_loop["generation_chunks"]
    ]
    shared_generation_count = min(len(pre_noise_signatures), len(post_noise_signatures))
    if (
        pre_noise_signatures[:shared_generation_count]
        != post_noise_signatures[:shared_generation_count]
    ):
        raise RuntimeError("T16 PRE/POST paired model-noise signatures differ")
    generation_noise_signatures_match = True if shared_generation_count > 0 else None
    both_reached_second_generation = all(
        tuple(arm_report["generation_steps"]) == T16_FULL_ARM_GENERATION_STEPS_ONE_BASED
        for arm_report in (pre_closed_loop, post_closed_loop)
    )
    both_full_32_steps = all(
        arm_report["environment_steps"] == T16_POLICY_STEPS_PER_ARM
        for arm_report in (pre_closed_loop, post_closed_loop)
    )
    if architecture_forward_calls != T15_ARCHITECTURE_FORWARDS:
        raise RuntimeError(
            f"T15 architecture forward count differs: {architecture_forward_calls}"
        )
    if backward_calls != T15_TRAINING_FORWARDS:
        raise RuntimeError(f"T15 backward call count differs: {backward_calls}")
    if optimizer_step_calls != 20:
        raise RuntimeError(
            f"T15 optimizer step call count differs: {optimizer_step_calls}"
        )
    all_macro_states_unchanged = (
        len(per_macro_no_mutation_evidence) == T15_MACRO_STEPS
        and [row["macro_step"] for row in per_macro_no_mutation_evidence]
        == list(range(1, T15_MACRO_STEPS + 1))
        and all(
            row["state_unchanged"]
            and all(row["components_unchanged"].values())
            and row["pre_microback_state_sha256"]
            == row["post_microback_pre_optimizer_state_sha256"]
            for row in per_macro_no_mutation_evidence
        )
    )
    if not all_macro_states_unchanged:
        raise RuntimeError("T15 no-intra-macro-mutation evidence coverage differs")
    if (
        len(update_recipe_signatures) != T15_TRAINING_FORWARDS
        or len(set(update_recipe_signatures)) != 1
    ):
        raise RuntimeError("T15 update RNG recipe signatures differ across forwards")
    invalid_measurement_signatures = {
        label: signatures
        for label, signatures in measurement_signatures.items()
        if len(signatures) != 2 or len(set(signatures)) != 1
    }
    if invalid_measurement_signatures:
        raise RuntimeError(
            "T15 measurement RNG recipe signatures differ: "
            f"{invalid_measurement_signatures!r}"
        )
    all_recipe_signatures = update_recipe_signatures + [
        signature for label in role_order for signature in measurement_signatures[label]
    ]
    if len(per_macro_recipe_signatures) != T15_MACRO_STEPS or any(
        set(signatures) != set(T13_TRAIN_LABELS)
        for signatures in per_macro_recipe_signatures
    ):
        raise RuntimeError("T15 per-macro recipe coverage differs")
    if per_macro_loss_weights != [
        _loss_weights_for_macro(step_index) for step_index in range(T15_MACRO_STEPS)
    ]:
        raise RuntimeError("T15 frozen matched-core loss-weight schedule differs")
    unique_recipe_signatures = set(all_recipe_signatures)
    if (
        len(all_recipe_signatures) != T15_ARCHITECTURE_FORWARDS
        or len(unique_recipe_signatures) != 1
    ):
        raise RuntimeError("T15 fixed loss recipe signature differs across forwards")
    t15_recipe_signature_sha256 = next(iter(unique_recipe_signatures))
    if t15_recipe_signature_sha256 == T14_PREDECESSOR_RECIPE_SIGNATURE_SHA256:
        raise RuntimeError("T15 loss recipe seed did not change the captured recipe")
    expected_exposures = {
        **{label: T15_FORWARD_EXPOSURES_PER_SAMPLE for label in T13_TRAIN_LABELS},
        **{label: 0 for label in T13_HELDOUT_LABELS},
    }
    if update_exposure_counts != expected_exposures:
        raise RuntimeError(
            f"T15 joint update exposure counts differ: {update_exposure_counts!r}"
        )
    if projection_exact_steps != list(range(1, T15_MACRO_STEPS + 1)):
        raise RuntimeError("T15 per-macro full projection evidence differs")
    expected_effective_weights = {
        **{label: T15_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE for label in T13_TRAIN_LABELS},
        **{label: 0.0 for label in T13_HELDOUT_LABELS},
    }
    if effective_weight_counts != expected_effective_weights:
        raise RuntimeError("T15 effective cumulative loss weights differ")
    final_optimizer_master_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if final_optimizer_master_ids != initial_master_ids:
        raise RuntimeError("T15 FP32 master objects changed during the loop")
    if prepared_input_snapshot() != initial_prepared_input_snapshot:
        raise RuntimeError("T15 prepared input identities or versions changed")
    frozen_version_changes = [
        name
        for name, parameter in named_frozen
        if parameter._version != frozen_versions[name]
    ]
    if frozen_version_changes:
        raise RuntimeError(
            f"T15 frozen parameters changed: {frozen_version_changes[:8]!r}"
        )
    if any(not bool(torch.isfinite(master).all().item()) for master in masters):
        raise RuntimeError("T15 FP32 masters became non-finite")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in params):
        raise RuntimeError("T15 BF16 trainables became non-finite")
    if not _optimizer_state_is_finite(optimizer):
        raise RuntimeError("T15 final AdamW state is non-finite")
    adam_steps: list[float] = []
    for master in masters:
        state = optimizer.state.get(master)
        if not state or "step" not in state:
            raise RuntimeError("T15 AdamW state is missing for an FP32 master")
        step_value = _tensor_scalar(state["step"])
        if not math.isfinite(step_value):
            raise RuntimeError("T15 AdamW step counter is non-finite")
        adam_steps.append(step_value)
    if set(adam_steps) != {float(T15_UPDATE_STEPS)}:
        raise RuntimeError(
            f"T15 AdamW step counters differ: {sorted(set(adam_steps))!r}"
        )

    cumulative_master_update, cumulative_master_ids = _summarize_updates(master_probes)
    cumulative_projected_update, _cumulative_model_ids = _summarize_updates(
        model_probes
    )
    if len(cumulative_master_ids) != len(named_masters) or set(
        cumulative_master_update["changed_trainable_roots"]
    ) != set(EXPECTED_TRAINABLE_ROOTS):
        raise RuntimeError("T15 cumulative FP32 master update coverage differs")

    classification_summary = _multi_episode_summary(
        {label: measurement_before[label]["loss_action"] for label in T13_TRAIN_LABELS},
        {label: measurement_after[label]["loss_action"] for label in T13_TRAIN_LABELS},
        {
            label: measurement_before[label]["loss_action"]
            for label in T13_HELDOUT_LABELS
        },
        {
            label: measurement_after[label]["loss_action"]
            for label in T13_HELDOUT_LABELS
        },
    )
    sample_by_label = {
        row["label"]: row for row in (*T13_TRAIN_SAMPLES, *T13_HELDOUT_SAMPLES)
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

    multi_episode_report = {
        **classification_summary,
        "same_task_heldout_samples": enrich_loss_summary(
            classification_summary["same_task_heldout_samples"]
        ),
        "loss_recipe_seed": args.loss_recipe_seed,
        "measurement_state_unchanged": True,
        "probe_state_unchanged": True,
        "macro_objective": {
            "arm": "JOINT",
            "labels": list(T13_TRAIN_LABELS),
            "loss": "equal_eighth_weighted_mean",
        },
        "effective_cumulative_loss_weights": {
            label: effective_weight_counts[label] for label in T13_TRAIN_LABELS
        },
        "update_exposure_counts": update_exposure_counts,
        "update_samples": enrich_loss_summary(classification_summary["update_samples"]),
    }
    joint_update_report = multi_episode_report["update_samples"]
    fresh_heldout_report = multi_episode_report["same_task_heldout_samples"]
    matched_core_schedule_report = {
        "accumulation_boundary_matches_production": True,
        "arm": "JOINT",
        "backward_calls_per_macro_step": 8,
        "configured_batch_size": int(cfg.training.batch_size),
        "configured_gradient_accumulation_steps": int(
            cfg.training.gradient_accumulation_steps
        ),
        "effective_cumulative_loss_weights": {
            label: effective_weight_counts[label] for label in T13_TRAIN_LABELS
        },
        "forward_order_within_macro_step": list(T13_TRAIN_LABELS),
        "forward_exposures_per_sample": T15_FORWARD_EXPOSURES_PER_SAMPLE,
        "macro_steps": T15_MACRO_STEPS,
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
                "after_eight_microbacks_before_gradient_clip_master_sync_and_"
                "optimizer_step"
            ),
        },
        "optimizer_steps_per_macro_step": 1,
        "parameters_constant_across_eight_component_backward_calls": (
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
    first_action_comparison = None
    if pre_closed_loop["trace"] and post_closed_loop["trace"]:
        pre_first_action = np.asarray(
            pre_closed_loop["trace"][0]["action"], dtype=np.float64
        )
        post_first_action = np.asarray(
            post_closed_loop["trace"][0]["action"], dtype=np.float64
        )
        first_action_delta = post_first_action - pre_first_action
        first_action_comparison = {
            "changed": bool(np.any(first_action_delta != 0.0)),
            "delta": first_action_delta.tolist(),
            "l2": float(np.linalg.norm(first_action_delta)),
            "max_abs": float(np.max(np.abs(first_action_delta))),
            "post": post_first_action.tolist(),
            "pre": pre_first_action.tolist(),
            "same_initial_observation": True,
        }
    overlap_steps = min(len(pre_closed_loop["trace"]), len(post_closed_loop["trace"]))
    paired_action_l2 = []
    paired_project_state_l2 = []
    paired_sim_state_l2 = []
    for step_index in range(overlap_steps):
        pre_action = np.asarray(
            pre_closed_loop["trace"][step_index]["action"], dtype=np.float64
        )
        post_action = np.asarray(
            post_closed_loop["trace"][step_index]["action"], dtype=np.float64
        )
        paired_action_l2.append(float(np.linalg.norm(post_action - pre_action)))
        pre_output = pre_closed_loop["trace"][step_index]["output_observation"]
        post_output = post_closed_loop["trace"][step_index]["output_observation"]
        paired_project_state_l2.append(
            float(
                np.linalg.norm(
                    np.asarray(post_output["state"], dtype=np.float64)
                    - np.asarray(pre_output["state"], dtype=np.float64)
                )
            )
        )
        paired_sim_state_l2.append(
            float(
                np.linalg.norm(
                    np.asarray(post_output["sim_state"], dtype=np.float64)
                    - np.asarray(pre_output["sim_state"], dtype=np.float64)
                )
            )
        )
    trajectories_identical = (
        len(pre_closed_loop["trace"]) == len(post_closed_loop["trace"])
        and all(value == 0.0 for value in paired_action_l2)
        and all(
            pre_row["output_observation"]["sim_state_sha256"]
            == post_row["output_observation"]["sim_state_sha256"]
            for pre_row, post_row in zip(
                pre_closed_loop["trace"],
                post_closed_loop["trace"],
                strict=True,
            )
        )
    )
    success_classification = {
        (False, False): "NEITHER_SUCCESS",
        (False, True): "POST_ONLY_SUCCESS",
        (True, False): "PRE_ONLY_SUCCESS",
        (True, True): "BOTH_SUCCESS",
    }[(pre_closed_loop["success"], post_closed_loop["success"])]

    def arm_behavior_diagnostics(arm_report: dict[str, Any]) -> dict[str, Any]:
        initial = arm_report["initial_observation"]
        final = (
            arm_report["trace"][-1]["output_observation"]
            if arm_report["trace"]
            else initial
        )
        initial_state = np.asarray(initial["state"], dtype=np.float64)
        final_state = np.asarray(final["state"], dtype=np.float64)
        initial_sim = np.asarray(initial["sim_state"], dtype=np.float64)
        final_sim = np.asarray(final["sim_state"], dtype=np.float64)
        object_displacements = {}
        for name in sorted(initial["objects"]):
            initial_position = np.asarray(
                initial["objects"][name]["position"], dtype=np.float64
            )
            final_position = np.asarray(
                final["objects"][name]["position"], dtype=np.float64
            )
            displacement = final_position - initial_position
            object_displacements[name] = {
                "delta_xyz": displacement.tolist(),
                "l2": float(np.linalg.norm(displacement)),
            }
        trace_projection = [
            {
                "action": row["action"],
                "done": row["done"],
                "output_sim_state_sha256": row["output_observation"][
                    "sim_state_sha256"
                ],
                "reward": row["reward"],
                "success": row["success"],
            }
            for row in arm_report["trace"]
        ]
        return {
            "cumulative_reward": float(
                sum(row["reward"] for row in arm_report["trace"])
            ),
            "eef_delta_xyz": (final_state[:3] - initial_state[:3]).tolist(),
            "eef_displacement_l2": float(
                np.linalg.norm(final_state[:3] - initial_state[:3])
            ),
            "environment_steps": arm_report["environment_steps"],
            "object_displacements": object_displacements,
            "sim_state_displacement_l2": float(np.linalg.norm(final_sim - initial_sim)),
            "success": arm_report["success"],
            "terminal_done": arm_report["terminal_done"],
            "termination_step": (
                arm_report["environment_steps"]
                if arm_report["success"] or arm_report["terminal_done"]
                else None
            ),
            "trajectory_projection_sha256": _sha256_json(trace_projection),
        }

    trajectory_diagnostics = {
        "action_l2_by_overlapping_step": paired_action_l2,
        "action_l2_max": max(paired_action_l2, default=None),
        "action_l2_mean": (
            statistics.fmean(paired_action_l2) if paired_action_l2 else None
        ),
        "classification": (
            "TRAJECTORY_IDENTICAL" if trajectories_identical else "TRAJECTORY_DIFFERENT"
        ),
        "overlap_steps": overlap_steps,
        "post": arm_behavior_diagnostics(post_closed_loop),
        "pre": arm_behavior_diagnostics(pre_closed_loop),
        "project_state_l2_by_overlapping_step": paired_project_state_l2,
        "sim_state_l2_by_overlapping_step": paired_sim_state_l2,
        "success_classification": success_classification,
    }
    both_executed_policy = all(
        arm_report["policy_requests"] >= 1
        for arm_report in (pre_closed_loop, post_closed_loop)
    )
    if not both_executed_policy:
        execution_verdict = T16_INITIAL_SUCCESS_VERDICT
        scientific_verdict = T16_INITIAL_SUCCESS_VERDICT
    elif both_reached_second_generation:
        execution_verdict = T16_EXECUTION_VERDICT
        if first_action_comparison is not None and first_action_comparison["changed"]:
            scientific_verdict = "T16_PAIRED_ONE_TASK_UPDATE_EFFECT_OBSERVED"
        else:
            scientific_verdict = "T16_PAIRED_ONE_TASK_NO_FIRST_ACTION_EFFECT"
    else:
        execution_verdict = T16_EXECUTION_VERDICT
        scientific_verdict = T16_EARLY_TERMINAL_VERDICT
    verdict_reasons = [
        "the exact T15 twenty-macro-step update core completed between paired simulator arms",
        "PRE and POST used one persistent LIBERO environment with identical reset fingerprints",
        (
            "an initial-state success prevented at least one policy step"
            if not both_executed_policy
            else (
                "both arms reached the mandatory step-29 second AR generation"
                if both_reached_second_generation
                else "at least one arm terminated before closing the mandatory two-generation geometry"
            )
        ),
        (
            "the same-initial-observation first action changed after the update"
            if first_action_comparison is not None
            and first_action_comparison["changed"]
            else "no changed first action was observed"
        ),
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
        "simulator_config_path": str(sim_config_path),
        "simulator_config_sha256": actual_sim_config_sha256,
        "simulator_worker_path": str(sim_worker_path),
        "simulator_worker_sha256": actual_sim_worker_sha256,
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
        "production_accumulation_boundary_matched": True,
        "production_accumulation_equivalent": True,
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
        "inherited_t12_eligible_manifest": eligible_sample_manifest,
        "inherited_t12_eligible_manifest_sha256": T13_ELIGIBLE_MANIFEST_SHA256,
        "inherited_t13_selection_manifest": selection_manifest,
        "inherited_t13_selection_manifest_sha256": T13_SELECTION_MANIFEST_SHA256,
        "initialization_seed": args.initialization_seed,
        "measurement_recipe_signatures_per_label": measurement_signatures,
        "per_macro_recipe_signatures": per_macro_recipe_signatures,
        "training_loss_recipe_seed": args.loss_recipe_seed,
        "recipe_reset": "python+numpy+torch-cpu+torch-cuda before every forward",
        "rng_state_restored_after_every_forward": True,
        "selected_multi_episode_manifest": selection_manifest,
        "selected_multi_episode_manifest_sha256": T13_SELECTION_MANIFEST_SHA256,
        "t14_predecessor_recipe_signature_sha256": (
            T14_PREDECESSOR_RECIPE_SIGNATURE_SHA256
        ),
        "t15_recipe_signature_differs_from_t14": True,
        "total_recipe_signatures": len(all_recipe_signatures),
        "training_recipe_signature_sha256": t15_recipe_signature_sha256,
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
            "role": (
                "update" if label in T13_TRAIN_LABELS else "same_task_heldout_probe"
            ),
            "start_frame": sample_by_label[label]["start_frame"],
            "task": sample_by_label[label]["task"],
            "task_index": sample_by_label[label]["task_index"],
            "effective_cumulative_loss_weight": effective_weight_counts[label],
            "update_exposure_count": update_exposure_counts[label],
        }
        for label in role_order
    }
    scope_report = {
        "architecture_forwards_total": T15_ARCHITECTURE_FORWARDS,
        "architecture_measurement_forwards": T15_MEASUREMENT_FORWARDS,
        "architecture_training_forwards": T15_TRAINING_FORWARDS,
        "backward_calls": backward_calls,
        "benchmark_evaluation_executed": False,
        "closed_loop_environment_steps": (
            pre_closed_loop["environment_steps"] + post_closed_loop["environment_steps"]
        ),
        "closed_loop_model_generations": (
            len(pre_closed_loop["generation_chunks"])
            + len(post_closed_loop["generation_chunks"])
        ),
        "closed_loop_policy_requests": (
            pre_closed_loop["policy_requests"] + post_closed_loop["policy_requests"]
        ),
        "effective_cumulative_loss_weight_per_suite": (
            T15_CUMULATIVE_LOSS_WEIGHT_PER_SUITE
        ),
        "effective_cumulative_loss_weight_per_update_sample": (
            T15_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE
        ),
        "formal_training_executed": False,
        "fresh_heldout_measurement_forwards": 16,
        "fresh_heldout_samples_in_backward_or_update": 0,
        "joint_training_measurement_forwards": 16,
        "macro_optimizer_steps": optimizer_step_calls,
        "raw_training_forward_exposures_per_sample": 20,
        "optimizer_steps": optimizer_step_calls,
        "post_probe_reprepare_calls": 0,
        "prepare_inputs_calls": prepare_inputs_calls,
        "probe_order": [
            "pre_A12_A13_A14_A15_A16_A17_A18_A19_H12_H13_H14_H15_H16_H17_H18_H19",
            "twenty_accumulation8_joint_matched_core_macro_steps",
            "post_A12_A13_A14_A15_A16_A17_A18_A19_H12_H13_H14_H15_H16_H17_H18_H19",
        ],
        "sana_base_pretrained_checkpoint_loaded": True,
        "sana_wam_training_checkpoint_loaded": False,
        "sana_wam_training_checkpoint_saved": False,
        "simulator_executed": True,
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
    closed_loop_report = {
        "action_tokens_per_chunk": T16_AR_ACTION_TOKENS_PER_CHUNK,
        "first_action_comparison": first_action_comparison,
        "both_full_32_steps": both_full_32_steps,
        "both_executed_at_least_one_policy_step": both_executed_policy,
        "both_reached_second_generation": both_reached_second_generation,
        "generation_noise_shared_generation_count": shared_generation_count,
        "generation_noise_signatures_match": generation_noise_signatures_match,
        "max_policy_steps_per_arm": T16_POLICY_STEPS_PER_ARM,
        "model_noise_seed": model_noise_seed,
        "noise_pair_key": noise_pair_key,
        "paired_reset_identity_exact": True,
        "post": post_closed_loop,
        "pre": pre_closed_loop,
        "required_generation_steps_one_based": list(
            T16_FULL_ARM_GENERATION_STEPS_ONE_BASED
        ),
        "simulator_close": simulator_close,
        "simulator_ready": sim_worker.ready,
        "task_selection": {
            "environment_seed": T16_ENVIRONMENT_SEED,
            "init_selection_payload_sha256": T16_INIT_SELECTION_PAYLOAD_SHA256,
            "init_state_index": T16_INIT_STATE_INDEX,
            "selection_manifest_sha256": T16_SELECTION_MANIFEST_SHA256,
            "suite": T16_TASK_SUITE,
            "task": T16_TASK,
            "dataset_task_index": T16_DATASET_TASK_INDEX,
            "libero_suite_task_index": T16_LIBERO_SUITE_TASK_INDEX,
        },
        "trajectory_diagnostics": trajectory_diagnostics,
    }
    report = {
        "architecture": architecture_report,
        "assets": assets_report,
        "closed_loop": closed_loop_report,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experimental_axis": {
            "changed_field": "model_parameters_after_exact_t15_update_core",
            "predecessor_value": "fresh_initialization_before_updates",
            "replication_value": "same_in_memory_model_after_twenty_macro_updates",
            "held_fixed": {
                "config_sha256": T15_CONFIG_SHA256,
                "dataloader.seed": T15_DATALOADER_SEED,
                "initialization_seed": T15_INITIALIZATION_SEED,
                "model_noise_seed": model_noise_seed,
                "macro_steps": T15_MACRO_STEPS,
                "sample_count": 16,
                "selection_manifest_sha256": T13_SELECTION_MANIFEST_SHA256,
                "simulator_init_state_index": T16_INIT_STATE_INDEX,
                "dataset_task_index": T16_DATASET_TASK_INDEX,
                "libero_suite_task_index": T16_LIBERO_SUITE_TASK_INDEX,
            },
            "same_simulator_object": True,
            "same_reset_fingerprint": True,
            "only_scientific_axis": "the exact in-memory T15 update core",
        },
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
        "scientific_classification": multi_episode_report,
        "underlying_t15_update_scientific_verdict": classification_summary[
            "scientific_verdict"
        ],
        "matched_core_schedule": matched_core_schedule_report,
        "update_samples": joint_update_report,
        "fresh_heldout_transfer": fresh_heldout_report,
        "identity": identity_report,
        "input_integrity": {
            "fresh_heldout_never_entered_backward_or_update": True,
            "prepared_inputs_immutable": True,
            "prepared_tensor_aliasing": False,
            "selected_samples_disjoint_from_t1_t12_model_facing_samples": True,
            "selected_samples_exactly_reused_from_t13": True,
            "selected_samples_exactly_reused_from_t14": True,
            "prior_model_facing_sample_count": len(T13_PRIOR_MODEL_FACING_SAMPLES)
            + len(T13_T12_MODEL_FACING_SAMPLES),
            "prior_model_facing_task_count": len(T13_PRIOR_MODEL_FACING_TASKS),
            "same_task_four_way_grouping": True,
            "selected_sample_count": 16,
            "selected_task_count": 4,
        },
        "inputs_by_label": inputs_by_label,
        "interpretation_limits": {
            "scientific_classification_executed": True,
            "benchmark_success_claimed": False,
            "closed_loop_architecture_validation_only": True,
            "heldout_axis": "one fixed LIBERO simulator task and init state",
            "model_facing_task_definition": (
                "T15 deliberately reuses the exact T13 tasks and sixteen samples; "
                "those samples remain disjoint from the complete T1-T12 model-facing "
                "sample set"
            ),
            "normalization_population_includes_probe_samples": True,
            "rollout_transfer_claimed": False,
            "multi_episode_retention_claim_only": True,
            "same_task_heldout_within_each_fixed_task": True,
            "strict_dataset_holdout": False,
            "task_text_only_isolation": False,
            "global_recipe_seed_controls_measurement_and_training": True,
            "initialization_seed_held_fixed": T15_INITIALIZATION_SEED,
            "loss_recipe_seed_axis_only": True,
            "pure_training_recipe_effect_claimed": False,
            "q_difference_attributed_to_training_recipe_only": False,
            "t14_loss_recipe_seed": T14_PREDECESSOR_LOSS_RECIPE_SEED,
            "t15_loss_recipe_seed": T15_LOSS_RECIPE_SEED,
            "training_or_benchmark_readiness_claimed": False,
            "single_task_single_init_generalization_claimed": False,
            "success_rate_claimed": False,
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
            "arm": "PAIRED_PRE_UPDATE_POST",
            "nonce": args.nonce,
            "root": str(run_root),
        },
        "sample_roles": sample_roles_report,
        "schema_version": "sana-wam-libero-t16-paired-one-task-closed-loop-v1",
        "scope": scope_report,
        "starting_state": starting_state,
        "timings_seconds": {
            "model_and_dataset_build": build_seconds,
            "prepare_real_samples": {
                label: measurement["seconds"]
                for label, measurement in prepare_measurements.items()
            },
            "sample_load": sample_seconds,
            "twenty_accumulation8_macro_updates_and_thirty_two_measurements": (
                update_seconds
            ),
            "pre_closed_loop": pre_closed_loop["elapsed_seconds"],
            "post_closed_loop": post_closed_loop["elapsed_seconds"],
        },
        "predecessor_training_core_lineage": predecessor_training_core_lineage,
        "direct_predecessor": {
            "result_path": str(T15_PREDECESSOR_RESULT_PATH),
            "result_sha256": T15_PREDECESSOR_RESULT_SHA256,
            "root": str(T15_PREDECESSOR_ROOT),
            "runner_sha256": T15_PREDECESSOR_RUNNER_SHA256,
            "source_commit": T15_PREDECESSOR_SOURCE_COMMIT,
            "stage": "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_BALANCED_JOINT",
        },
        "underlying_t14_predecessor": {
            "result_path": str(T14_PREDECESSOR_RESULT_PATH),
            "result_sha256": T14_PREDECESSOR_RESULT_SHA256,
            "root": str(T14_PREDECESSOR_ROOT),
            "runner_sha256": T14_PREDECESSOR_RUNNER_SHA256,
            "source_commit": T14_PREDECESSOR_SOURCE_COMMIT,
            "stage": "T14_INITSEED_REPLICATION_ACCUM8_BALANCED_JOINT",
        },
        "underlying_t13_predecessor": {
            "result_path": str(T13_PREDECESSOR_RESULT_PATH),
            "result_sha256": T13_PREDECESSOR_RESULT_SHA256,
            "root": str(T13_PREDECESSOR_ROOT),
            "runner_sha256": T13_PREDECESSOR_RUNNER_SHA256,
            "source_commit": T13_PREDECESSOR_SOURCE_COMMIT,
            "stage": "T13_MULTI_EPISODE_ACCUM8_BALANCED_JOINT",
        },
        "underlying_t12_predecessor": {
            "result_path": str(T12_PREDECESSOR_RESULT_PATH),
            "result_sha256": T12_PREDECESSOR_RESULT_SHA256,
            "root": str(T12_PREDECESSOR_ROOT),
            "runner_sha256": T12_PREDECESSOR_RUNNER_SHA256,
            "source_commit": T12_PREDECESSOR_SOURCE_COMMIT,
            "stage": "T12_NEW_TASK_BALANCED_JOINT",
        },
        "underlying_t11_predecessor": {
            "result_path": str(T11_PREDECESSOR_RESULT_PATH),
            "result_sha256": T11_PREDECESSOR_RESULT_SHA256,
            "root": str(T11_PREDECESSOR_ROOT),
            "runner_sha256": T11_PREDECESSOR_RUNNER_SHA256,
            "source_commit": T11_PREDECESSOR_SOURCE_COMMIT,
            "stage": "T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT",
        },
        "underlying_t10_predecessor": {
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
            "stage": "T10_EXACT_BALANCED_JOINT_AGGREGATE",
        },
        "valid_run": True,
        "scientific_verdict": scientific_verdict,
        "secondary_diagnostic_warnings": secondary_diagnostic_warnings,
        "verdict": execution_verdict,
        "verdict_reasons": verdict_reasons,
        "warnings": warning_messages,
    }
    if any(run_root.iterdir()):
        raise RuntimeError("T16 run root was not empty before terminal write")
    _write_terminal_report(run_root / "RESULT.json", report)
    _freeze_run_root(run_root)
    _ACTIVE_RUN_ROOT = None
    gpu_lock.close()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as error:
        if _ACTIVE_SIM_WORKER is not None:
            try:
                _ACTIVE_SIM_WORKER.abort()
            except BaseException as worker_error:
                print(
                    "failed to terminate T16 simulator worker: "
                    f"{type(worker_error).__name__}: {worker_error}",
                    file=sys.stderr,
                    flush=True,
                )
        if _ACTIVE_RUN_ROOT is not None:
            try:
                _terminalize_failure(_ACTIVE_RUN_ROOT, error)
            except BaseException as freeze_error:
                print(
                    "failed to terminalize T16 root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    raise SystemExit(exit_code)
