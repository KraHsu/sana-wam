from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "configs/baselines/openwam_ar_lownoise_reference.yaml"
CLEAN_PATH = ROOT / "configs/train_ar_lownoise_action_only_clean.yaml"
DAGGER_PATH = ROOT / "configs/train_ar_lownoise_action_only_historical_dagger.yaml"
FRESH_PATH = ROOT / "configs/train_ar_lownoise_action_only_fresh_history113.yaml"

CHECKPOINT_SHA256 = "aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d"
ACTION_STATS_SHA256 = "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
FRESH_MANIFEST_SHA256 = (
    "30e8996370924f1f5c4b519e5a2b0ace892fdf68289131f53a76f1a3cf7ab1f6"
)
DEPLOY_DATA_FIELDS = (
    "task_name",
    "robot",
    "action_mode",
    "num_frames",
    "video_stride",
    "height",
    "width",
    "multiview",
    "camera_layout",
    "target_camera",
    "normalize_mode",
    "temporal_compression",
    "causal_temporal",
    "delta_action",
)


def _load(path: Path) -> dict:
    # The archived OpenWAM reference contains Hydra's `${now:...}` resolver in
    # an unrelated project field; compare the literal model/data contract.
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_weight(source: dict) -> dict:
    normalized = deepcopy(source)
    normalized.pop("enabled")
    normalized.pop("weight")
    return normalized


def test_action_only_models_exactly_match_recorded_baseline():
    reference_model = _load(REFERENCE_PATH)["model"]
    assert _load(CLEAN_PATH)["model"] == reference_model
    assert _load(DAGGER_PATH)["model"] == reference_model


def test_action_only_training_contract_is_identical_except_output():
    clean = _load(CLEAN_PATH)
    dagger = _load(DAGGER_PATH)

    clean_training = deepcopy(clean["training"])
    dagger_training = deepcopy(dagger["training"])
    clean_output = clean_training.pop("output_dir")
    dagger_output = dagger_training.pop("output_dir")

    assert clean_output != dagger_output
    assert clean_training == dagger_training
    assert clean_training["init_checkpoint_sha256"] == CHECKPOINT_SHA256
    assert clean_training["action_stats_sha256"] == ACTION_STATS_SHA256
    assert clean_training["trainable_modules"] == ["action_backbone"]
    assert "freeze" not in clean_training
    assert clean_training["lambda_video"] == 0.0
    assert clean_training["lambda_action"] == 1.0
    assert clean_training["video_lr"] == 0.0
    assert clean_training["action_lr"] == 1.0e-5
    assert clean_training["batch_size"] == 1
    assert clean_training["gradient_accumulation_steps"] == 4
    assert clean_training["num_workers"] == 4
    assert clean_training["warmup_steps"] == 20
    assert clean_training["lr_schedule_steps"] == 1000
    assert clean_training["weight_decay"] == 0.0
    assert clean_training["seed"] == 42

    assert set(clean) == set(dagger) == {"model", "dataloader", "training"}


def test_checkpoint_ladder_keeps_only_the_four_required_artifacts():
    for path in (CLEAN_PATH, DAGGER_PATH):
        training = _load(path)["training"]
        assert training["save_initial_checkpoint"] is True
        assert training["save_steps"] == 0
        assert training["save_at_steps"] == [100, 500, 1000]
        assert training["max_steps"] == 1000
        assert training["keep_last_k"] == 4
        assert [0, *training["save_at_steps"]] == [0, 100, 500, 1000]


def test_clean_data_matches_reference_and_mixture_changes_only_sources():
    reference = _load(REFERENCE_PATH)["dataloader"]
    clean = _load(CLEAN_PATH)["dataloader"]
    mixture = _load(DAGGER_PATH)["dataloader"]

    for key, value in reference.items():
        assert clean[key] == value
    assert clean["delta_action"] is False

    assert mixture["type"] == "mixture"
    assert mixture["weight_strategy"] == "manual"
    assert len(mixture["datasets"]) == 2
    for key in DEPLOY_DATA_FIELDS:
        assert mixture[key] == clean[key]

    clean_source, dagger_source = mixture["datasets"]
    assert clean_source["weight"] == 1.0
    assert dagger_source["weight"] == 0.5
    assert _without_weight(clean_source) == clean

    normalized_dagger = _without_weight(dagger_source)
    normalized_dagger["dataset_dir"] = clean["dataset_dir"]
    normalized_dagger["variant"] = clean["variant"]
    assert normalized_dagger == clean


def test_recorded_artifact_hashes_match_when_files_are_available():
    clean = _load(CLEAN_PATH)
    dagger = _load(DAGGER_PATH)
    configs = (clean, dagger)

    checkpoint_paths = {Path(cfg["training"]["init_checkpoint"]) for cfg in configs}
    assert len(checkpoint_paths) == 1
    assert len(CHECKPOINT_SHA256) == 64
    int(CHECKPOINT_SHA256, 16)
    for path in checkpoint_paths:
        if os.environ.get("SANA_WAM_VERIFY_LARGE_ARTIFACTS") == "1":
            assert path.is_file()
            assert _digest(path) == CHECKPOINT_SHA256

    stats_paths = {Path(clean["dataloader"]["action_stats_path"])}
    stats_paths.update(
        Path(source["action_stats_path"]) for source in dagger["dataloader"]["datasets"]
    )
    assert len(stats_paths) == 1
    assert len(ACTION_STATS_SHA256) == 64
    int(ACTION_STATS_SHA256, 16)
    for path in stats_paths:
        if path.is_file():
            assert _digest(path) == ACTION_STATS_SHA256


def test_fresh_history113_treatment_preserves_model_and_clean_source_contract():
    reference = _load(REFERENCE_PATH)
    clean = _load(CLEAN_PATH)
    fresh = _load(FRESH_PATH)

    assert fresh["model"] == reference["model"]
    dataloader = fresh["dataloader"]
    assert dataloader["type"] == "mixture"
    assert dataloader["weight_strategy"] == "manual"
    assert len(dataloader["datasets"]) == 2
    for key in DEPLOY_DATA_FIELDS:
        assert dataloader[key] == clean["dataloader"][key]

    clean_source, history_source = dataloader["datasets"]
    assert clean_source["weight"] == 1.0
    assert _without_weight(clean_source) == clean["dataloader"]
    assert history_source["type"] == "robotwin_history_dagger"
    assert history_source["weight"] == 0.5
    assert history_source["source_manifest_sha256"] == FRESH_MANIFEST_SHA256
    assert history_source["action_stats_sha256"] == ACTION_STATS_SHA256
    assert history_source["expected_policy_history_len"] == 113
    assert history_source["num_frames"] == 113
    assert history_source["video_stride"] == 4
    assert history_source["temporal_compression"] == 4
    assert history_source["height"] == 384
    assert history_source["width"] == 320
    assert history_source["filter_static_segments"] is False
    assert history_source["val_ratio"] == 0.0


def test_fresh_history113_treatment_uses_exact_s100_action_only_contract():
    training = _load(FRESH_PATH)["training"]

    assert training["init_checkpoint_sha256"] == CHECKPOINT_SHA256
    assert training["action_stats_sha256"] == ACTION_STATS_SHA256
    assert training["trainable_modules"] == ["action_backbone"]
    assert training["expected_world_size"] == 4
    assert training["max_steps"] == 100
    assert training["save_initial_checkpoint"] is True
    assert training["save_steps"] == 0
    assert training["save_at_steps"] == [100]
    assert training["batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 8
    assert training["video_lr"] == 0.0
    assert training["action_lr"] == 1.0e-5
    assert training["lambda_video"] == 0.0
    assert training["lambda_action"] == 1.0
    assert training["warmup_steps"] == 20
    assert training["lr_schedule_steps"] == 1000
    assert training["seed"] == 42


def test_fresh_history113_manifest_hash_and_counts_match_when_available():
    history_source = _load(FRESH_PATH)["dataloader"]["datasets"][1]
    manifest_path = Path(history_source["source_manifest_path"])
    if not manifest_path.is_file():
        return

    assert _digest(manifest_path) == FRESH_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["anchor_count"] == 5
    assert manifest["valid_sample_count"] == 2
    assert len(manifest["samples"]) == 2
