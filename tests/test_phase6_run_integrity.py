from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT / "src"))

from sana_wam.train import phase6_run_integrity as integrity  # noqa: E402
from sana_wam.dataloader.robotwin_plan_binding import (  # noqa: E402
    phase6_plan_row_dict,
)
from sana_wam.dataloader.task_sample_plan import (  # noqa: E402
    PlannedSample,
)


def _digest(value: bytes | str) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return sha256(data).hexdigest()


def _task_sha(tasks: list[str]) -> str:
    return _digest("".join(f"{task}\n" for task in sorted(tasks)))


def _build_plan_artifact(root: Path) -> tuple[bytes, str, str, str]:
    tasks = [f"task_{index:02d}" for index in range(42)]
    holdouts = [f"holdout_{index:02d}" for index in range(8)]
    domains = [
        "video-noise",
        "action-noise",
        "expansion-noise",
        "expansion-direction",
        "reference-query",
        "prompt-choice",
    ]
    rows = []
    for cycle in range(12):
        sigma = (1.0, 0.9, 0.5)[cycle // 4]
        for position, task in enumerate(tasks):
            step = cycle * 42 + position + 1
            rows.append(
                {
                    "action_sigma": sigma,
                    "cycle": cycle,
                    "domain_seeds": {
                        domain: domain_index * 10_000 + step
                        for domain_index, domain in enumerate(domains, start=1)
                    },
                    "global_step": step,
                    "identity": {
                        "dataset_index": 100_000 + step,
                        "episode_index": cycle,
                        "episode_path": str(
                            root / "episodes" / task / f"episode{cycle}.hdf5"
                        ),
                        "prompt": f"Synthetic prompt for {task}, cycle {cycle}",
                        "source_dataset": "RoboTwin",
                        "source_kind": "ordinary_expert",
                        "source_variant": "clean_50",
                        "start_frame": position + cycle,
                        "task_name": task,
                    },
                    "position_in_cycle": position,
                }
            )
    plan = {
        "action_sigmas": [1.0, 0.9, 0.5],
        "holdout_task_sha256": _task_sha(holdouts),
        "holdout_tasks": holdouts,
        "optimizer_steps": 504,
        "protocol_seed": 20260724,
        "rows": rows,
        "schema_version": integrity.PLAN_SCHEMA_VERSION,
        "seed_domains": [
            "task-order",
            "sample-choice",
            "action-sigma",
            *domains,
        ],
        "sigma_exposures_per_task": 4,
        "source_contract": {
            "dataset": "RoboTwin",
            "kind": "ordinary_expert",
            "variant": "clean_50",
        },
        "task_cycles": 12,
        "train_task_sha256": _task_sha(tasks),
        "train_tasks": tasks,
    }
    plan_sha = _digest(integrity.canonical_json_bytes(plan))
    identity_sha = integrity._identity_sha256(plan)
    artifact = {
        "artifact_schema_version": integrity.PLAN_ARTIFACT_SCHEMA_VERSION,
        "expansion_eligibility_amendment_sha256": (
            integrity.EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "expansion_support_contract": integrity.expansion_support_contract(),
        "expansion_support_contract_sha256": (
            integrity.EXPANSION_SUPPORT_CONTRACT_SHA256
        ),
        "identity_sha256": identity_sha,
        "plan": plan,
        "plan_sha256": plan_sha,
    }
    data = integrity.canonical_json_bytes(artifact)
    return data, _digest(data), plan_sha, identity_sha


def _repin_plan_artifact(
    artifact: dict[str, object],
) -> tuple[bytes, str, str, str]:
    plan = artifact["plan"]
    assert isinstance(plan, dict)
    plan_sha = _digest(integrity.canonical_json_bytes(plan))
    identity_sha = integrity._identity_sha256(plan)
    artifact["plan_sha256"] = plan_sha
    artifact["identity_sha256"] = identity_sha
    data = integrity.canonical_json_bytes(artifact)
    return data, _digest(data), plan_sha, identity_sha


def _measurements(arm: str, step: int = 1) -> dict[str, object]:
    factors = integrity.ARM_FACTORS[arm]
    video = 0.5 + step / 10_000.0
    if factors["E"]:
        rate = 8.5 if step % 2 else 7.5
        expansion_loss = max((rate - 8.0) / 8.0, 0.0) ** 2
        expansion = {
            "active": True,
            "loss": float(expansion_loss),
            "rate": float(rate),
            "violation": rate > 8.0,
        }
    else:
        expansion_loss = 0.0
        expansion = {
            "active": False,
            "loss": 0.0,
            "rate": None,
            "violation": None,
        }
    if factors["A"]:
        reference = 0.2
        student = 0.21 + (step % 3) * 0.001
        relative = (student - reference) / (reference + 1.0e-4)
        nr_loss = max(relative, 0.0) ** 2
        action = {
            "active": True,
            "non_regression_loss": float(nr_loss),
            "reference_error": float(reference),
            "relative_excess": float(relative),
            "student_error": float(student),
        }
        adapter_lr: float | None = 1.0e-4
    else:
        nr_loss = 0.0
        action = {
            "active": False,
            "non_regression_loss": 0.0,
            "reference_error": None,
            "relative_excess": None,
            "student_error": float(0.19 + (step % 5) * 0.001),
        }
        adapter_lr = None
    if factors["A"]:
        gradients = {
            "action_trainable_param_count": 0,
            "adapter_gradient_connected": True,
            "adapter_grad_norm": 0.75,
            "finite": True,
            "global_norm": 1.25,
            "model_parameters_finite": True,
            "nr_gradient_target_adapter_only": True,
            "optimizer_master_parameters_finite": True,
            "optimizer_state_finite": True,
            "proprio_trainable_param_count": 0,
            "video_grad_norm": 1.0,
        }
    else:
        gradients = {
            "action_trainable_param_count": 0,
            "adapter_gradient_connected": None,
            "adapter_grad_norm": None,
            "finite": True,
            "global_norm": 1.25,
            "model_parameters_finite": True,
            "nr_gradient_target_adapter_only": None,
            "optimizer_master_parameters_finite": True,
            "optimizer_state_finite": True,
            "proprio_trainable_param_count": 0,
            "video_grad_norm": 1.25,
        }
    return {
        "action": action,
        "common_input_trace_sha256": _digest(f"common-input-step-{step}"),
        "expansion": expansion,
        "gradients": gradients,
        "learning_rates": {
            "action_adapter": adapter_lr,
            "video": 5.0e-6,
        },
        "memory": {"peak_reserved_bytes": 16 * 1024**3 + step},
        "total_loss": float(video + expansion_loss + nr_loss),
        "video_on_path_loss": float(video),
    }


def _config_projection(arm: str) -> dict[str, object]:
    factors = integrity.ARM_FACTORS[arm]
    return {
        "arm": arm,
        "dataset": {
            "action_mode": "eef",
            "num_frames": 113,
            "variant": "clean_50",
        },
        "factors": dict(factors),
        "optimizer": {
            "action_memory_lr": 1.0e-4 if factors["A"] else None,
            "grad_clip": 1.0,
            "lambda_action": 0.0,
            "lambda_video": 1.0,
            "master_weights": True,
            "video_lr": 5.0e-6,
            "weight_decay": 0.0,
        },
        "plan_sha256": _digest("frozen plan projection pin"),
        "preflight_purpose": "training",
        "schedule": {"kind": "cosine", "max_steps": 504, "warmup_steps": 50},
        "seed": 20260724,
        "trainable_parameter_patterns": ["video_backbone.*"],
    }


class Phase6RunIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve()
        (
            cls.plan_bytes,
            cls.plan_artifact_sha,
            cls.plan_sha,
            cls.identity_sha,
        ) = _build_plan_artifact(cls.root)
        cls.binding = integrity.Phase6PlanBinding.from_artifact_bytes(
            cls.plan_bytes,
            expected_artifact_sha256=cls.plan_artifact_sha,
            expected_plan_sha256=cls.plan_sha,
            expected_identity_sha256=cls.identity_sha,
        )
        raw_rows = json.loads(cls.plan_bytes)["plan"]["rows"]
        cls.runtime_rows = tuple(PlannedSample.from_dict(row) for row in raw_rows)
        cls.provenance = {
            "action_reference_sha256": _digest("action reference"),
            "action_stats_sha256": _digest("action stats"),
            "arm_config_sha256": _digest("arm config"),
            "arm_config_projection_sha256": _digest(
                integrity.canonical_json_bytes(_config_projection("T1_E1A1"))
            ),
            "dataset_contract_sha256": _digest("dataset contract"),
            "initial_checkpoint_sha256": _digest("student checkpoint"),
            "launch_manifest_sha256": _digest("launch manifest"),
            "phase1_checkpoint_sha256": _digest("phase1 checkpoint"),
            "preflight_request_sha256": _digest("preflight request"),
            "runtime_support_manifest_sha256": _digest("runtime support"),
            "smoke_artifact_sha256": _digest("real 2b smoke"),
            "source_manifest_sha256": _digest("source manifest"),
            "spot_verification_sha256": _digest("spot verification"),
        }
        cls.preflight_report_value = {
            **integrity.recovery_lifecycle(),
            "purpose": "training",
            "reference_precompute_completed": True,
            "registered_before_recovery_cohort_started": True,
            "request_sha256": cls.provenance["preflight_request_sha256"],
            "schema_version": "synthetic-training-preflight-report-v1",
            "status": "pass",
            "smoke_completed": True,
        }
        cls.preflight_report_bytes = integrity.canonical_json_bytes(
            cls.preflight_report_value
        )
        cls.provenance["preflight_report_sha256"] = _digest(cls.preflight_report_bytes)
        cls.preflight_report_path = cls.root / "training-preflight-report.json"
        cls.preflight_report_path.write_bytes(cls.preflight_report_bytes)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.case_root = Path(tempfile.mkdtemp(dir=self.root)).resolve()

    def tearDown(self) -> None:
        shutil.rmtree(self.case_root)

    def writer(
        self,
        arm: str = "T1_E1A1",
        *,
        name: str = "run",
        report_path: Path | None = None,
        provenance: Mapping[str, str] | None = None,
    ) -> integrity.Phase6MetricsWriter:
        run_id = f"synthetic_{name}"
        selected_provenance = dict(provenance or self.provenance)
        if provenance is None and arm != "T1_E1A1":
            selected_provenance["arm_config_sha256"] = _digest(f"arm config:{arm}")
            selected_provenance["arm_config_projection_sha256"] = _digest(
                integrity.canonical_json_bytes(_config_projection(arm))
            )
        return integrity.Phase6MetricsWriter(
            run_directory=self.case_root / run_id,
            preflight_report_path=report_path or self.preflight_report_path,
            binding=self.binding,
            arm=arm,
            run_id=run_id,
            provenance=selected_provenance,
        )

    def runtime_row(self, step: int) -> dict:
        return phase6_plan_row_dict(
            self.runtime_rows[step - 1],
            plan_sha256=self.plan_sha,
            identity_sha256=self.identity_sha,
        )

    def append(self, writer: integrity.Phase6MetricsWriter, step: int) -> None:
        writer.append_step(
            global_step=step,
            observed_plan_row=self.runtime_row(step),
            measurements=_measurements(writer._arm, step),
        )

    def primary_outputs(
        self,
        writer: integrity.Phase6MetricsWriter,
        *,
        frozen_equal: bool = True,
    ) -> dict[str, dict[str, str]]:
        run_directory = Path(writer.run_directory)
        checkpoint = run_directory / "checkpoint_step_504.safetensors"
        config = run_directory / "resolved_config.yaml"
        deploy_config = run_directory / "config.yaml"
        action_stats = run_directory / "action_stats.npy"
        checkpoint.write_bytes(b"synthetic step-504 checkpoint")
        config.write_bytes(b"resolved arm config with num_frames=113")
        deploy_config.write_bytes(config.read_bytes())
        action_stats.write_bytes(b"action stats")
        checkpoint_sha = _digest(checkpoint.read_bytes())
        resolved_config_sha = _digest(config.read_bytes())
        frozen_projection = _config_projection(writer._arm)
        resolved_config_manifest = {
            "arm": writer._arm,
            "deploy_config_equivalent": True,
            "deploy_config_sha256": resolved_config_sha,
            "factors": dict(integrity.ARM_FACTORS[writer._arm]),
            "frozen_projection": frozen_projection,
            "frozen_projection_sha256": _digest(
                integrity.canonical_json_bytes(frozen_projection)
            ),
            "input_arm_config_sha256": writer._provenance["arm_config_sha256"],
            "resolved_saved_config_sha256": resolved_config_sha,
            "schema_version": integrity.RESOLVED_CONFIG_VERIFICATION_SCHEMA_VERSION,
        }
        resolved_config_manifest_path = (
            run_directory / "resolved_config_verification.json"
        )
        resolved_config_manifest_path.write_bytes(
            integrity.canonical_json_bytes(resolved_config_manifest)
        )
        action_initial = _digest("frozen action tensors")
        proprio_initial = _digest("frozen proprio tensors")
        adapter_expected = integrity.ARM_FACTORS[writer._arm]["A"]
        action_keys = [f"action_backbone.tensor_{index:03d}" for index in range(556)]
        proprio_keys = list(integrity._PROPRIO_STATE_KEYS)
        frozen_manifest = {
            "action_backbone": {
                "bitwise_equal": frozen_equal,
                "final_sha256": (
                    action_initial if frozen_equal else _digest("changed action")
                ),
                "initial_sha256": action_initial,
                "key_names_sha256": _digest(
                    integrity.canonical_json_bytes(action_keys)
                ),
                "selected_keys": action_keys,
                "tensor_count": 556,
            },
            "action_video_memory_adapter": {
                "expected": adapter_expected,
                "key_count": 5 if adapter_expected else 0,
                "key_names_sha256": (
                    _digest(
                        integrity.canonical_json_bytes(integrity._ADAPTER_STATE_KEYS)
                    )
                    if adapter_expected
                    else None
                ),
                "present": adapter_expected,
                "selected_keys": (
                    list(integrity._ADAPTER_STATE_KEYS) if adapter_expected else []
                ),
                "state_sha256": (
                    _digest("trained adapter state") if adapter_expected else None
                ),
            },
            "initial_checkpoint_sha256": writer._provenance[
                "initial_checkpoint_sha256"
            ],
            "proprio": {
                "bitwise_equal": True,
                "final_sha256": proprio_initial,
                "initial_sha256": proprio_initial,
                "key_names_sha256": _digest(
                    integrity.canonical_json_bytes(proprio_keys)
                ),
                "selected_keys": proprio_keys,
                "tensor_count": 6,
            },
            "schema_version": integrity.FROZEN_TENSOR_VERIFICATION_SCHEMA_VERSION,
            "step504_checkpoint_sha256": checkpoint_sha,
        }
        frozen_path = run_directory / "frozen_action_proprio_verification.json"
        frozen_path.write_bytes(integrity.canonical_json_bytes(frozen_manifest))
        paths = {
            "action_stats": action_stats,
            "frozen_tensor_verification_manifest": frozen_path,
            "resolved_config_verification_manifest": resolved_config_manifest_path,
            "saved_arm_config": config,
            "step504_checkpoint": checkpoint,
        }
        return {
            role: {"path": str(path), "sha256": _digest(path.read_bytes())}
            for role, path in paths.items()
        }

    def test_plan_artifact_and_identity_are_strictly_bound(self) -> None:
        binding = integrity.Phase6PlanBinding.from_artifact_bytes(
            self.plan_bytes,
            expected_artifact_sha256=self.plan_artifact_sha,
            expected_plan_sha256=self.plan_sha,
            expected_identity_sha256=self.identity_sha,
        )
        self.assertEqual(binding.expected_row(504)["global_step"], 504)
        raw_row = binding.expected_row(1)
        runtime_row = binding.expected_runtime_row(1)
        self.assertEqual(
            set(runtime_row), set(raw_row) | {"identity_sha256", "plan_sha256"}
        )
        self.assertEqual(runtime_row["identity_sha256"], self.identity_sha)
        self.assertEqual(runtime_row["plan_sha256"], self.plan_sha)
        with self.assertRaisesRegex(integrity.RunIntegrityError, "artifact SHA"):
            integrity.Phase6PlanBinding.from_artifact_bytes(
                self.plan_bytes,
                expected_artifact_sha256=_digest("wrong"),
                expected_plan_sha256=self.plan_sha,
                expected_identity_sha256=self.identity_sha,
            )

    def test_all_runtime_rows_match_the_dataloader_envelope_contract(self) -> None:
        self.assertEqual(len(self.runtime_rows), 504)
        for step, row in enumerate(self.runtime_rows, start=1):
            raw_row = self.binding.expected_row(step)
            runtime_row = phase6_plan_row_dict(
                row,
                plan_sha256=self.plan_sha,
                identity_sha256=self.identity_sha,
            )
            self.assertEqual(runtime_row, self.binding.expected_runtime_row(step))
            self.assertEqual(
                integrity.canonical_json_bytes(raw_row),
                self.binding._row_bytes[step - 1],
            )
            self.assertNotEqual(
                integrity.canonical_json_bytes(runtime_row),
                self.binding._row_bytes[step - 1],
            )

    def test_plan_rejects_old_outer_v1_even_when_repinned(self) -> None:
        artifact = json.loads(self.plan_bytes)
        artifact["artifact_schema_version"] = "sana-phase6-task-plan-artifact-v1"
        data, artifact_sha, plan_sha, identity_sha = _repin_plan_artifact(artifact)
        with self.assertRaisesRegex(integrity.RunIntegrityError, "schema differs"):
            integrity.Phase6PlanBinding.from_artifact_bytes(
                data,
                expected_artifact_sha256=artifact_sha,
                expected_plan_sha256=plan_sha,
                expected_identity_sha256=identity_sha,
            )

    def test_plan_rejects_expansion_provenance_mutations_when_repinned(self) -> None:
        mutations = (
            (
                "eligibility amendment",
                lambda artifact: artifact.__setitem__(
                    "expansion_eligibility_amendment_sha256", "0" * 64
                ),
            ),
            (
                "support contract hash",
                lambda artifact: artifact.__setitem__(
                    "expansion_support_contract_sha256", "0" * 64
                ),
            ),
            (
                "nested support contract",
                lambda artifact: artifact["expansion_support_contract"].__setitem__(
                    "frame_chunk_size", 4
                ),
            ),
        )
        for name, mutate in mutations:
            artifact = json.loads(self.plan_bytes)
            mutate(artifact)
            data, artifact_sha, plan_sha, identity_sha = _repin_plan_artifact(artifact)
            with (
                self.subTest(name=name),
                self.assertRaises(integrity.RunIntegrityError),
            ):
                integrity.Phase6PlanBinding.from_artifact_bytes(
                    data,
                    expected_artifact_sha256=artifact_sha,
                    expected_plan_sha256=plan_sha,
                    expected_identity_sha256=identity_sha,
                )

    def test_plan_rejects_equal_numeric_aliases_when_repinned(self) -> None:
        mutations = (
            (
                "optimizer_steps float",
                lambda plan: plan.__setitem__("optimizer_steps", 504.0),
            ),
            (
                "task_cycles float",
                lambda plan: plan.__setitem__("task_cycles", 12.0),
            ),
            (
                "sigma_exposures float",
                lambda plan: plan.__setitem__("sigma_exposures_per_task", 4.0),
            ),
            (
                "protocol_seed float",
                lambda plan: plan.__setitem__("protocol_seed", 20260724.0),
            ),
            (
                "action_sigmas integer",
                lambda plan: plan["action_sigmas"].__setitem__(0, 1),
            ),
            (
                "global_step float",
                lambda plan: plan["rows"][0].__setitem__("global_step", 1.0),
            ),
            (
                "cycle boolean",
                lambda plan: plan["rows"][0].__setitem__("cycle", False),
            ),
            (
                "position float",
                lambda plan: plan["rows"][0].__setitem__("position_in_cycle", 0.0),
            ),
            (
                "action_sigma integer",
                lambda plan: plan["rows"][0].__setitem__("action_sigma", 1),
            ),
            (
                "domain seed float",
                lambda plan: plan["rows"][0]["domain_seeds"].__setitem__(
                    "video-noise", 10001.0
                ),
            ),
            (
                "dataset index float",
                lambda plan: plan["rows"][0]["identity"].__setitem__(
                    "dataset_index", 100001.0
                ),
            ),
            (
                "episode index float",
                lambda plan: plan["rows"][0]["identity"].__setitem__(
                    "episode_index", 0.0
                ),
            ),
            (
                "start frame float",
                lambda plan: plan["rows"][0]["identity"].__setitem__(
                    "start_frame", 0.0
                ),
            ),
        )
        for name, mutate in mutations:
            artifact = json.loads(self.plan_bytes)
            mutate(artifact["plan"])
            data, artifact_sha, plan_sha, identity_sha = _repin_plan_artifact(artifact)
            with (
                self.subTest(name=name),
                self.assertRaises(integrity.RunIntegrityError),
            ):
                integrity.Phase6PlanBinding.from_artifact_bytes(
                    data,
                    expected_artifact_sha256=artifact_sha,
                    expected_plan_sha256=plan_sha,
                    expected_identity_sha256=identity_sha,
                )

    def test_plan_rejects_broken_task_cycle_even_when_repinned(self) -> None:
        artifact = json.loads(self.plan_bytes)
        artifact["plan"]["rows"][1]["identity"]["task_name"] = "task_00"
        artifact["plan_sha256"] = _digest(
            integrity.canonical_json_bytes(artifact["plan"])
        )
        artifact["identity_sha256"] = integrity._identity_sha256(artifact["plan"])
        data = integrity.canonical_json_bytes(artifact)
        with self.assertRaisesRegex(integrity.RunIntegrityError, "repeats a task"):
            integrity.Phase6PlanBinding.from_artifact_bytes(
                data,
                expected_artifact_sha256=_digest(data),
                expected_plan_sha256=artifact["plan_sha256"],
                expected_identity_sha256=artifact["identity_sha256"],
            )

    def test_run_directory_is_exclusive_and_nonresumable(self) -> None:
        writer = self.writer(name="exclusive")
        self.append(writer, 1)
        writer.close()
        with self.assertRaisesRegex(integrity.RunIntegrityError, "resume/append"):
            self.writer(name="exclusive")
        self.assertFalse(Path(writer.summary_path).exists())

    def test_invalid_preflight_never_creates_run_directory(self) -> None:
        value = dict(self.preflight_report_value)
        value["purpose"] = "reference_precompute"
        path = self.case_root / "precompute-report.json"
        data = integrity.canonical_json_bytes(value)
        path.write_bytes(data)
        provenance = dict(self.provenance)
        provenance["preflight_report_sha256"] = _digest(data)
        with self.assertRaisesRegex(integrity.RunIntegrityError, "training-purpose"):
            self.writer(
                name="must-not-exist",
                report_path=path,
                provenance=provenance,
            )
        self.assertFalse((self.case_root / "synthetic_must-not-exist").exists())

    def test_training_preflight_requires_completed_smoke_and_recovery_lifecycle(
        self,
    ) -> None:
        for mutation, message in (
            ({"smoke_completed": False}, "real-2B smoke"),
            (
                {"historical_formal_training_started": False},
                "historical_formal_training_started",
            ),
            ({"recovery_cohort_started": True}, "recovery_cohort_started"),
            ({"historical_checkpoint_count": False}, "historical_checkpoint_count"),
            (
                {"historical_optimizer_step_calls_per_arm": True},
                "historical_optimizer_step_calls_per_arm",
            ),
            (
                {"historical_scheduled_lr_scale_at_step0": 0},
                "historical_scheduled_lr_scale_at_step0",
            ),
            ({"registered_before_recovery_cohort_started": False}, "registration"),
        ):
            value = {**self.preflight_report_value, **mutation}
            data = integrity.canonical_json_bytes(value)
            path = self.case_root / f"bad-lifecycle-{len(message)}.json"
            path.write_bytes(data)
            provenance = {
                **self.provenance,
                "preflight_report_sha256": _digest(data),
            }
            with self.assertRaisesRegex(integrity.RunIntegrityError, message):
                self.writer(
                    name=f"bad-lifecycle-{len(message)}",
                    report_path=path,
                    provenance=provenance,
                )

    def test_preflight_snapshot_and_full_finalize(self) -> None:
        writer = self.writer(name="complete")
        self.assertEqual(
            Path(writer.preflight_report_snapshot_path).read_bytes(),
            self.preflight_report_bytes,
        )
        for step in range(1, 505):
            self.append(writer, step)
        primary_outputs = self.primary_outputs(writer)
        finalized = writer.finalize(primary_outputs=primary_outputs)
        metrics_path = Path(writer.metrics_path)
        summary_path = Path(writer.summary_path)
        self.assertEqual(finalized.summary["metrics"]["row_count"], 504)
        self.assertEqual(
            finalized.summary["metrics"]["sha256"], _digest(metrics_path.read_bytes())
        )
        self.assertEqual(finalized.summary["provenance"], self.provenance)
        self.assertEqual(
            set(finalized.summary["primary_outputs"]),
            {
                "action_stats",
                "frozen_tensor_verification_manifest",
                "resolved_config_verification_manifest",
                "saved_arm_config",
                "step504_checkpoint",
            },
        )
        self.assertEqual(
            finalized.summary["action_student_error_distribution"]["count"],
            504,
        )
        expected_trace_sequence = _digest(
            integrity.canonical_json_bytes(
                [_digest(f"common-input-step-{step}") for step in range(1, 505)]
            )
        )
        self.assertEqual(
            finalized.summary["common_input_trace"]["sequence_sha256"],
            expected_trace_sequence,
        )
        self.assertEqual(
            finalized.summary["common_input_trace"]["contract"],
            integrity.COMMON_INPUT_TRACE_CONTRACT,
        )
        self.assertEqual(
            finalized.summary["coverage"]["steps_by_sigma"],
            {"0.5": 168, "0.9": 168, "1.0": 168},
        )
        self.assertTrue(
            all(
                count == 12
                for count in finalized.summary["coverage"]["steps_by_task"].values()
            )
        )
        summary_bytes = summary_path.read_bytes()
        self.assertEqual(
            summary_bytes, integrity.canonical_json_bytes(finalized.summary)
        )
        self.assertEqual(finalized.summary_sha256, _digest(summary_bytes))
        completion_bytes = Path(writer.completion_path).read_bytes()
        self.assertEqual(
            completion_bytes,
            integrity.canonical_json_bytes(finalized.completion),
        )
        self.assertEqual(finalized.completion_sha256, _digest(completion_bytes))
        self.assertEqual(finalized.completion["status"], "complete")
        self.assertEqual(
            finalized.completion["metrics_summary"]["sha256"],
            finalized.summary_sha256,
        )

        original = metrics_path.read_bytes()
        truncated = self.case_root / "truncated.jsonl"
        truncated.write_bytes(original[:-1])
        with self.assertRaises(integrity.RunIntegrityError):
            integrity.validate_metrics_jsonl(
                truncated,
                binding=self.binding,
                arm="T1_E1A1",
                run_id="synthetic_complete",
                provenance=self.provenance,
            )
        lines = original.splitlines(keepends=True)
        first = json.loads(lines[0])
        first["domain_seeds"]["video-noise"] += 1
        tampered = self.case_root / "tampered.jsonl"
        tampered.write_bytes(
            integrity.canonical_json_bytes(first) + b"".join(lines[1:])
        )
        with self.assertRaisesRegex(integrity.RunIntegrityError, "domain_seeds"):
            integrity.validate_metrics_jsonl(
                tampered,
                binding=self.binding,
                arm="T1_E1A1",
                run_id="synthetic_complete",
                provenance=self.provenance,
            )

    def test_504_rows_without_primary_output_closure_cannot_complete(self) -> None:
        writer = self.writer(name="no_primary_closure")
        for step in range(1, 505):
            self.append(writer, step)
        with self.assertRaisesRegex(
            integrity.RunIntegrityError, "primary output closure"
        ):
            writer.finalize()
        self.assertFalse(Path(writer.summary_path).exists())
        self.assertFalse(Path(writer.completion_path).exists())

    def test_frozen_tensor_and_resolved_config_manifests_are_hard_gates(self) -> None:
        writer = self.writer(name="bad_frozen_tensors")
        outputs = self.primary_outputs(writer, frozen_equal=False)
        with self.assertRaisesRegex(integrity.RunIntegrityError, "bitwise identical"):
            integrity._validate_primary_outputs(
                outputs,
                run_directory=writer.run_directory,
                provenance=self.provenance,
                arm="T1_E1A1",
            )
        writer.close()

        writer = self.writer(name="bad_resolved_config")
        outputs = self.primary_outputs(writer)
        manifest_path = Path(outputs["resolved_config_verification_manifest"]["path"])
        manifest = json.loads(manifest_path.read_bytes())
        manifest["frozen_projection"]["optimizer"]["lambda_action"] = 1.0
        manifest_path.write_bytes(integrity.canonical_json_bytes(manifest))
        outputs["resolved_config_verification_manifest"]["sha256"] = _digest(
            manifest_path.read_bytes()
        )
        with self.assertRaisesRegex(
            integrity.RunIntegrityError, "projection SHA differs"
        ):
            integrity._validate_primary_outputs(
                outputs,
                run_directory=writer.run_directory,
                provenance=self.provenance,
                arm="T1_E1A1",
            )
        writer.close()

        writer = self.writer(name="extra_checkpoint")
        outputs = self.primary_outputs(writer)
        (Path(writer.run_directory) / "checkpoint_step_0.safetensors").write_bytes(
            b"forbidden intermediate checkpoint"
        )
        with self.assertRaisesRegex(integrity.RunIntegrityError, "exactly one primary"):
            integrity._validate_primary_outputs(
                outputs,
                run_directory=writer.run_directory,
                provenance=self.provenance,
                arm="T1_E1A1",
            )
        writer.close()

        writer = self.writer(name="missing_adapter")
        outputs = self.primary_outputs(writer)
        manifest_path = Path(outputs["frozen_tensor_verification_manifest"]["path"])
        manifest = json.loads(manifest_path.read_bytes())
        manifest["action_video_memory_adapter"]["present"] = False
        manifest_path.write_bytes(integrity.canonical_json_bytes(manifest))
        outputs["frozen_tensor_verification_manifest"]["sha256"] = _digest(
            manifest_path.read_bytes()
        )
        with self.assertRaisesRegex(integrity.RunIntegrityError, "five adapter keys"):
            integrity._validate_primary_outputs(
                outputs,
                run_directory=writer.run_directory,
                provenance=self.provenance,
                arm="T1_E1A1",
            )
        writer.close()

        writer = self.writer("T0_E0A0", name="a0_no_adapter")
        outputs = self.primary_outputs(writer)
        observed = integrity._validate_primary_outputs(
            outputs,
            run_directory=writer.run_directory,
            provenance=writer._provenance,
            arm="T0_E0A0",
        )
        self.assertIn("step504_checkpoint", observed)
        writer.close()

    def test_e_and_a_off_use_null_not_fake_zero_measurements(self) -> None:
        writer = self.writer("T0_E0A0", name="off")
        self.append(writer, 1)
        writer.close()
        record = json.loads(Path(writer.metrics_path).read_bytes())
        self.assertFalse(record["expansion"]["active"])
        self.assertIsNone(record["expansion"]["rate"])
        self.assertIsNone(record["expansion"]["violation"])
        self.assertEqual(record["expansion"]["loss"], 0.0)
        self.assertFalse(record["action"]["active"])
        self.assertIsInstance(record["action"]["student_error"], float)
        self.assertGreaterEqual(record["action"]["student_error"], 0.0)
        self.assertIsNone(record["action"]["reference_error"])
        self.assertIsNone(record["learning_rates"]["action_adapter"])

    def test_off_factor_numeric_zero_measurements_are_rejected(self) -> None:
        mutations = (
            (
                "expansion rate",
                lambda value: value["expansion"].__setitem__("rate", 0.0),
            ),
            (
                "reference error",
                lambda value: value["action"].__setitem__("reference_error", 0.0),
            ),
            (
                "adapter LR",
                lambda value: value["learning_rates"].__setitem__(
                    "action_adapter", 0.0
                ),
            ),
            (
                "adapter grad",
                lambda value: value["gradients"].__setitem__("adapter_grad_norm", 0.0),
            ),
        )
        for index, (name, mutate) in enumerate(mutations):
            writer = self.writer("T0_E0A0", name=f"off_zero_{index}")
            value = _measurements("T0_E0A0")
            mutate(value)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(integrity.RunIntegrityError, "null"),
            ):
                writer.append_step(
                    global_step=1,
                    observed_plan_row=self.runtime_row(1),
                    measurements=value,
                )
            writer.close()
            self.assertEqual(Path(writer.metrics_path).stat().st_size, 0)

    def test_gradient_groups_and_frozen_parameter_counts_are_enforced(self) -> None:
        mutations = (
            lambda value: value["gradients"].__setitem__("global_norm", 9.0),
            lambda value: value["gradients"].__setitem__(
                "nr_gradient_target_adapter_only", False
            ),
            lambda value: value["gradients"].__setitem__(
                "adapter_gradient_connected", False
            ),
            lambda value: value["gradients"].__setitem__(
                "action_trainable_param_count", 1
            ),
            lambda value: value["gradients"].__setitem__(
                "proprio_trainable_param_count", 1
            ),
        )
        for index, mutate in enumerate(mutations):
            writer = self.writer(name=f"grad_contract_{index}")
            value = _measurements("T1_E1A1")
            mutate(value)
            with self.assertRaises(integrity.RunIntegrityError):
                writer.append_step(
                    global_step=1,
                    observed_plan_row=self.runtime_row(1),
                    measurements=value,
                )

    def test_active_factors_require_real_measurements(self) -> None:
        writer = self.writer(name="active_null")
        value = _measurements("T1_E1A1")
        value["expansion"]["rate"] = None
        with self.assertRaisesRegex(integrity.RunIntegrityError, "finite JSON float"):
            writer.append_step(
                global_step=1,
                observed_plan_row=self.runtime_row(1),
                measurements=value,
            )
        writer.close()

    def test_loss_and_violation_formulas_are_enforced(self) -> None:
        mutations = (
            (
                "total",
                lambda value: value.__setitem__(
                    "total_loss", value["total_loss"] + 1.0
                ),
            ),
            ("expansion", lambda value: value["expansion"].__setitem__("loss", 0.0)),
            (
                "violation",
                lambda value: value["expansion"].__setitem__("violation", False),
            ),
            (
                "relative",
                lambda value: value["action"].__setitem__("relative_excess", 0.0),
            ),
            (
                "NR",
                lambda value: value["action"].__setitem__("non_regression_loss", 0.0),
            ),
        )
        for index, (name, mutate) in enumerate(mutations):
            writer = self.writer(name=f"formula_{index}")
            value = _measurements("T1_E1A1")
            mutate(value)
            with (
                self.subTest(name=name),
                self.assertRaises(integrity.RunIntegrityError),
            ):
                writer.append_step(
                    global_step=1,
                    observed_plan_row=self.runtime_row(1),
                    measurements=value,
                )
            writer.close()

    def test_observed_plan_task_sigma_seeds_and_prompt_cannot_drift(self) -> None:
        mutations = (
            lambda row: row["identity"].__setitem__("task_name", "task_01"),
            lambda row: row.__setitem__("action_sigma", 0.5),
            lambda row: row["domain_seeds"].__setitem__("video-noise", 999),
            lambda row: row["identity"].__setitem__("prompt", "changed prompt"),
            lambda row: row.__setitem__("identity_sha256", "0" * 64),
            lambda row: row.__setitem__("plan_sha256", "0" * 64),
            lambda row: row.pop("plan_sha256"),
            lambda row: row.pop("identity_sha256"),
            lambda row: row.__setitem__("unexpected_envelope", True),
        )
        for index, mutate in enumerate(mutations):
            writer = self.writer(name=f"row_drift_{index}")
            row = self.runtime_row(1)
            mutate(row)
            with self.assertRaisesRegex(
                integrity.RunIntegrityError, "observed plan row differs"
            ):
                writer.append_step(
                    global_step=1,
                    observed_plan_row=row,
                    measurements=_measurements("T1_E1A1"),
                )
            writer.close()
            self.assertEqual(Path(writer.metrics_path).stat().st_size, 0)

    def test_raw_or_wrapped_plan_row_is_not_a_runtime_audit_payload(self) -> None:
        invalid_rows = (
            self.binding.expected_row(1),
            {"phase6_plan_row": self.runtime_row(1)},
        )
        for index, row in enumerate(invalid_rows):
            writer = self.writer(name=f"invalid_envelope_{index}")
            with self.assertRaisesRegex(
                integrity.RunIntegrityError, "observed plan row differs"
            ):
                writer.append_step(
                    global_step=1,
                    observed_plan_row=row,
                    measurements=_measurements("T1_E1A1"),
                )
            writer.close()
            self.assertEqual(Path(writer.metrics_path).stat().st_size, 0)

    def test_record_keeps_the_raw_plan_row_sha256_semantics(self) -> None:
        writer = self.writer(name="raw_row_sha")
        self.append(writer, 1)
        writer.close()
        record = json.loads(Path(writer.metrics_path).read_bytes())
        raw_bytes = self.binding._row_bytes[0]
        runtime_bytes = integrity.canonical_json_bytes(self.runtime_row(1))
        self.assertEqual(record["plan_row_sha256"], _digest(raw_bytes))
        self.assertNotEqual(record["plan_row_sha256"], _digest(runtime_bytes))

    def test_global_steps_are_contiguous_and_single_append(self) -> None:
        writer = self.writer(name="step_skip")
        with self.assertRaisesRegex(
            integrity.RunIntegrityError, "expected global_step=1"
        ):
            writer.append_step(
                global_step=2,
                observed_plan_row=self.runtime_row(2),
                measurements=_measurements("T1_E1A1", 2),
            )
        with self.assertRaisesRegex(integrity.RunIntegrityError, "failed state"):
            self.append(writer, 1)

        writer = self.writer(name="step_duplicate")
        self.append(writer, 1)
        with self.assertRaisesRegex(
            integrity.RunIntegrityError, "expected global_step=2"
        ):
            self.append(writer, 1)
        writer.close()

    def test_nan_is_rejected_before_append(self) -> None:
        writer = self.writer(name="nan")
        value = _measurements("T1_E1A1")
        value["total_loss"] = math.nan
        with self.assertRaisesRegex(integrity.RunIntegrityError, "finite JSON float"):
            writer.append_step(
                global_step=1,
                observed_plan_row=self.runtime_row(1),
                measurements=value,
            )
        writer.close()
        self.assertEqual(Path(writer.metrics_path).stat().st_size, 0)

    def test_nonfinite_gradient_stop_row_is_fsynced_then_writer_fails(self) -> None:
        writer = self.writer(name="nonfinite")
        value = _measurements("T1_E1A1")
        value["gradients"] = {
            "action_trainable_param_count": 0,
            "adapter_gradient_connected": True,
            "adapter_grad_norm": None,
            "finite": False,
            "global_norm": None,
            "model_parameters_finite": True,
            "nr_gradient_target_adapter_only": True,
            "optimizer_master_parameters_finite": True,
            "optimizer_state_finite": True,
            "proprio_trainable_param_count": 0,
            "video_grad_norm": None,
        }
        with self.assertRaisesRegex(integrity.NonFiniteGradientError, "global_step=1"):
            writer.append_step(
                global_step=1,
                observed_plan_row=self.runtime_row(1),
                measurements=value,
            )
        record = json.loads(Path(writer.metrics_path).read_bytes())
        self.assertFalse(record["gradients"]["finite"])
        self.assertIsNone(record["gradients"]["global_norm"])
        with self.assertRaisesRegex(integrity.RunIntegrityError, "failed"):
            writer.finalize()

    def test_nonfinite_post_step_states_are_durably_distinguished(self) -> None:
        cases = (
            (
                "model_parameters_finite",
                integrity.NonFiniteModelParametersError,
            ),
            (
                "optimizer_master_parameters_finite",
                integrity.NonFiniteOptimizerMasterParametersError,
            ),
            (
                "optimizer_state_finite",
                integrity.NonFiniteOptimizerStateError,
            ),
        )
        for index, (field, error_type) in enumerate(cases):
            writer = self.writer(name=f"nonfinite_post_step_{index}")
            value = _measurements("T1_E1A1")
            value["gradients"][field] = False
            with self.assertRaisesRegex(error_type, "global_step=1"):
                writer.append_step(
                    global_step=1,
                    observed_plan_row=self.runtime_row(1),
                    measurements=value,
                )
            record = json.loads(Path(writer.metrics_path).read_bytes())
            self.assertFalse(record["gradients"][field])
            with self.assertRaisesRegex(integrity.RunIntegrityError, "failed"):
                writer.finalize()

    def test_each_append_calls_fsync(self) -> None:
        original_fsync = os.fsync
        with mock.patch.object(integrity.os, "fsync", wraps=original_fsync) as fsync:
            writer = self.writer(name="fsync")
            before = fsync.call_count
            self.append(writer, 1)
            self.assertGreater(fsync.call_count, before)
            writer.close()

    def test_external_append_is_detected(self) -> None:
        writer = self.writer(name="external_append")
        self.append(writer, 1)
        with open(writer.metrics_path, "ab") as stream:
            stream.write(b"{}\n")
            stream.flush()
            os.fsync(stream.fileno())
        with self.assertRaisesRegex(integrity.RunIntegrityError, "externally appended"):
            self.append(writer, 2)
        writer.close()

    def test_metrics_path_replacement_is_detected(self) -> None:
        writer = self.writer(name="replacement")
        self.append(writer, 1)
        path = Path(writer.metrics_path)
        backup = path.with_suffix(".original")
        try:
            os.replace(path, backup)
            path.write_bytes(b"")
        except OSError as exc:
            writer.close()
            self.skipTest(f"open-file replacement unavailable: {exc}")
        with self.assertRaisesRegex(integrity.RunIntegrityError, "replaced"):
            self.append(writer, 2)
        writer.close()

    def test_summary_publication_refuses_existing_path(self) -> None:
        destination = self.case_root / "summary.json"
        destination.write_bytes(b"sentinel\n")
        with self.assertRaisesRegex(integrity.RunIntegrityError, "already exists"):
            integrity._exclusive_atomic_write(
                str(destination), integrity.canonical_json_bytes({"ok": True})
            )
        self.assertEqual(destination.read_bytes(), b"sentinel\n")

    def test_cross_arm_common_input_sequence_must_match(self) -> None:
        paths = {}
        trace_sha = _digest("common trace sequence")
        for arm in integrity.ARM_FACTORS:
            provenance = dict(self.provenance)
            provenance["arm_config_sha256"] = _digest(f"config:{arm}")
            provenance["arm_config_projection_sha256"] = _digest(
                integrity.canonical_json_bytes(_config_projection(arm))
            )
            provenance["preflight_request_sha256"] = _digest(f"request:{arm}")
            provenance["preflight_report_sha256"] = _digest(f"report:{arm}")
            summary = {
                "action_student_error_distribution": {},
                "arm": arm,
                "common_input_trace": {
                    "contract": integrity.COMMON_INPUT_TRACE_CONTRACT,
                    "sequence_sha256": trace_sha,
                },
                "coverage": {},
                "factors": dict(integrity.ARM_FACTORS[arm]),
                "identity_sha256": self.identity_sha,
                "integrity": {},
                "metrics": {"row_count": 504},
                "plan_artifact_sha256": self.plan_artifact_sha,
                "plan_sha256": self.plan_sha,
                "preflight_purpose": "training",
                "preflight_report_snapshot": {},
                "primary_outputs": {},
                "provenance": provenance,
                "record_type": "summary",
                "run_id": f"cross_{arm}",
                "schema_version": integrity.SUMMARY_SCHEMA_VERSION,
            }
            path = self.case_root / f"{arm}.summary.json"
            path.write_bytes(integrity.canonical_json_bytes(summary))
            paths[arm] = path
        result = integrity.validate_cross_arm_common_input_traces(paths)
        self.assertEqual(result["common_input_trace_sequence_sha256"], trace_sha)

        value = json.loads(paths["T1_E1A1"].read_bytes())
        value["common_input_trace"]["sequence_sha256"] = _digest("drift")
        paths["T1_E1A1"].write_bytes(integrity.canonical_json_bytes(value))
        with self.assertRaisesRegex(integrity.RunIntegrityError, "differs across"):
            integrity.validate_cross_arm_common_input_traces(paths)

    def test_import_does_not_import_torch(self) -> None:
        source_root = CANDIDATE_ROOT / "src"
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(source_root)!r}); "
            "import sana_wam.train.phase6_run_integrity; "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
