from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.verify_paired_model_telemetry import verify_model_pair
from tests.test_closed_loop_telemetry import _write_complete_paired_run


CONTROL_SHA256 = "a" * 64
TREATMENT_SHA256 = "b" * 64


def _rewrite_jsonl(path: Path, transform) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        transform(record)
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records))


def _set_checkpoint_identity(root: Path, *, path: str, sha256: str, size: int) -> None:
    def update(record):
        identity = record.get("runtime", {}).get("deployment_identity")
        if isinstance(identity, dict):
            identity.update(
                {
                    "checkpoint_path": path,
                    "checkpoint_size": size,
                    "checkpoint_sha256": sha256,
                }
            )

    for producer_file in (
        root / "telemetry/server/episodes.jsonl",
        root / "telemetry/server/steps.jsonl",
    ):
        _rewrite_jsonl(producer_file, update)
    (root / "checkpoint.sha256").write_text(f"{sha256}  {path}\n")


def _rewrite_manifest(root: Path, transform) -> str:
    path = root / "prompt_manifest.json"
    manifest = json.loads(path.read_text())
    transform(manifest)
    encoded = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()

    def update_context(record):
        if record.get("event") != "episode_start":
            return
        context = record.get("context", {})
        context["prompt_manifest_path"] = str(path.resolve())
        context["prompt_manifest_sha256"] = digest

    for producer in ("server", "client"):
        _rewrite_jsonl(root / f"telemetry/{producer}/episodes.jsonl", update_context)
    return digest


def _bind_treatment_manifest_to_control(control: Path, treatment: Path) -> str:
    control_starts = [
        record
        for record in (
            json.loads(line)
            for line in (control / "telemetry/server/episodes.jsonl")
            .read_text()
            .splitlines()
        )
        if record.get("event") == "episode_start"
    ]

    def bind(manifest):
        manifest["source"].update(
            {
                "run_path": str(control.resolve()),
                "run_id": control_starts[0]["run_id"],
                "checkpoint_sha256": CONTROL_SHA256,
            }
        )

    return _rewrite_manifest(treatment, bind)


def _make_treatment_generation_zero_different(treatment: Path) -> None:
    for path in (treatment / "telemetry/server/chunks").glob("*.npz"):
        with np.load(path, allow_pickle=False) as chunk:
            arrays = {name: chunk[name] for name in chunk.files}
        arrays["predicted_actions"] = arrays["predicted_actions"] + np.float32(7)
        arrays["predicted_actions_normalized"] = arrays[
            "predicted_actions_normalized"
        ] + np.float32(7)
        np.savez_compressed(path, **arrays)


def _build_model_pair(tmp_path: Path) -> tuple[Path, Path, str]:
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(
        control,
        mode="predicted",
        environment_seeds=(100001,),
        replay_manifest=False,
    )
    _write_complete_paired_run(
        treatment,
        mode="predicted",
        environment_seeds=(100001,),
        replay_manifest=True,
    )
    _set_checkpoint_identity(
        control,
        path="/artifacts/control.safetensors",
        sha256=CONTROL_SHA256,
        size=101,
    )
    _set_checkpoint_identity(
        treatment,
        path="/artifacts/treatment.safetensors",
        sha256=TREATMENT_SHA256,
        size=202,
    )
    manifest_sha256 = _bind_treatment_manifest_to_control(control, treatment)
    _make_treatment_generation_zero_different(treatment)
    return control, treatment, manifest_sha256


def _verify(control: Path, treatment: Path, manifest_sha256: str | None = None):
    return verify_model_pair(
        control,
        treatment,
        expected_episodes=1,
        control_client_path=control,
        treatment_client_path=treatment,
        expected_prompt_manifest_sha256=manifest_sha256,
    )


def test_model_pair_accepts_only_checkpoint_identity_and_output_differences(tmp_path):
    control, treatment, manifest_sha256 = _build_model_pair(tmp_path)

    report = _verify(control, treatment, manifest_sha256)

    assert report["episodes"] == 1
    assert report["control_checkpoint_sha256"] == CONTROL_SHA256
    assert report["treatment_checkpoint_sha256"] == TREATMENT_SHA256
    assert report["prompt_manifest_sha256"] == manifest_sha256


def test_model_pair_rejects_non_checkpoint_identity_difference(tmp_path):
    control, treatment, _ = _build_model_pair(tmp_path)

    def change_history(record):
        identity = record.get("runtime", {}).get("deployment_identity")
        if isinstance(identity, dict):
            identity["history_len"] = 114

    _rewrite_jsonl(treatment / "telemetry/server/episodes.jsonl", change_history)
    _rewrite_jsonl(treatment / "telemetry/server/steps.jsonl", change_history)

    with pytest.raises(SystemExit, match="non-checkpoint deployment identities"):
        _verify(control, treatment)


def test_model_pair_rejects_ordered_scene_pair_difference(tmp_path):
    control, treatment, _ = _build_model_pair(tmp_path)
    different_seed = 100999
    different_pair = f"robotwin/adjust_bottle/demo_clean/scene-{different_seed}"

    def change_server(record):
        if record.get("event") != "episode_start":
            return
        record["context"]["environment_seed"] = different_seed
        record["runtime"]["current_noise_pair_key"] = different_pair

    _rewrite_jsonl(treatment / "telemetry/server/episodes.jsonl", change_server)

    def change_client_episode(record):
        if record.get("event") == "episode_start":
            record["context"]["environment_seed"] = different_seed
            record["context"]["noise_pair_key"] = different_pair
        else:
            record["environment_seed"] = different_seed

    _rewrite_jsonl(treatment / "telemetry/client/episodes.jsonl", change_client_episode)

    def change_client_step(record):
        record["environment_seed"] = different_seed

    _rewrite_jsonl(treatment / "telemetry/client/steps.jsonl", change_client_step)

    with pytest.raises(SystemExit, match="ordered scene/model-noise pairs"):
        _verify(control, treatment)


def test_model_pair_rejects_prompt_difference(tmp_path):
    control, treatment, _ = _build_model_pair(tmp_path)
    prompt = "a different treatment prompt"
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()

    def change_context(record):
        if record.get("event") == "episode_start":
            record["context"]["prompt"] = prompt
            record["context"]["prompt_sha256"] = prompt_sha256

    for producer in ("server", "client"):
        _rewrite_jsonl(
            treatment / f"telemetry/{producer}/episodes.jsonl", change_context
        )

    def change_step(record):
        record["prompt"] = prompt

    _rewrite_jsonl(treatment / "telemetry/client/steps.jsonl", change_step)

    with pytest.raises(SystemExit, match="control/treatment prompt mismatch"):
        _verify(control, treatment)


def test_model_pair_requires_different_checkpoint_sha(tmp_path):
    control, treatment, _ = _build_model_pair(tmp_path)
    _set_checkpoint_identity(
        treatment,
        path="/artifacts/treatment.safetensors",
        sha256=CONTROL_SHA256,
        size=202,
    )

    with pytest.raises(SystemExit, match="checkpoint SHA-256 values must differ"):
        _verify(control, treatment)


def test_model_pair_rejects_manifest_not_bound_to_control(tmp_path):
    control, treatment, _ = _build_model_pair(tmp_path)

    def bind_wrong_checkpoint(manifest):
        manifest["source"]["checkpoint_sha256"] = TREATMENT_SHA256

    _rewrite_manifest(treatment, bind_wrong_checkpoint)

    with pytest.raises(SystemExit, match="manifest checkpoint is not the control"):
        _verify(control, treatment)


def test_runner_model_pairing_is_explicit_opt_in():
    runner = (
        Path(__file__).resolve().parent.parent / "scripts/run_ar_lownoise_paired_arm.sh"
    ).read_text()
    assert 'PAIR_COMPARISON_KIND="${PAIR_COMPARISON_KIND:-noise}"' in runner
    assert "scripts/verify_paired_model_telemetry.py" in runner
    assert "scripts/verify_paired_noise_telemetry.py" in runner
