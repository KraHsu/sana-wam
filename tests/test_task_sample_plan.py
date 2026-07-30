from __future__ import annotations

from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

CANDIDATE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(CANDIDATE_SRC))

from sana_wam.dataloader.task_sample_plan import (
    EXPECTED_HOLDOUT_TASK_SHA256,
    EXPECTED_TRAIN_TASK_SHA256,
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    FACTORIAL_ARMS,
    ROW_SEED_DOMAINS,
    PlanSampler,
    PlanValidationError,
    SampleIdentity,
    SamplerContract,
    TaskRoundRobinPlan,
    derive_seed,
    verify_shared_factorial_plans,
)


REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "logs/sana_principles_audit_20260722/phase6_principled_architecture_20260724"
    / "phase6_registry_seed20260724.json"
)
FIXTURE_PLAN_SHA256 = "e2e64a3f75e32797e14b28fb3ed8b2fd79eba7f2ed30dd05094e8642efaca7ee"
FIXTURE_IDENTITY_SHA256 = (
    "48374c4c45f59965b219fb7637e32ec13276adbc161eedf18b8ef32b39d604ab"
)
FIXTURE_JSONL_SHA256 = (
    "3f139dab0a51f88451fa2b76412fa07e20f546475a7480a353f0efba8a4e4961"
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


class TaskSamplePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = SamplerContract.from_registry(REGISTRY_PATH)
        cls.pools = cls.make_pools(cls.contract)
        cls.plan = TaskRoundRobinPlan.build(cls.contract, cls.pools)

    @staticmethod
    def make_pools(
        contract: SamplerContract,
        *,
        reverse_tasks: bool = False,
        reverse_samples: bool = False,
        candidates_per_task: int = 15,
    ) -> dict[str, list[SampleIdentity]]:
        tasks = list(contract.train_tasks)
        if reverse_tasks:
            tasks.reverse()
        pools: dict[str, list[SampleIdentity]] = {}
        canonical_task_number = {
            task: number for number, task in enumerate(contract.train_tasks)
        }
        for task in tasks:
            task_number = canonical_task_number[task]
            identities = [
                SampleIdentity(
                    task_name=task,
                    episode_index=sample_number,
                    episode_path=(
                        f"/DATA/robotwin/clean_50/{task}/episode_{sample_number:03d}.hdf5"
                    ),
                    start_frame=sample_number * 3,
                    prompt=f"Exact_prompt for {task} sample_{sample_number:03d}",
                    dataset_index=task_number * 1000 + sample_number,
                )
                for sample_number in range(candidates_per_task)
            ]
            if reverse_samples:
                identities.reverse()
            pools[task] = identities
        return pools

    def test_frozen_contract_and_balanced_counts(self) -> None:
        self.assertEqual(len(self.plan.rows), 504)
        self.assertEqual(len(self.plan.train_tasks), 42)
        self.assertEqual(self.plan.train_task_sha256, EXPECTED_TRAIN_TASK_SHA256)
        self.assertEqual(self.plan.holdout_task_sha256, EXPECTED_HOLDOUT_TASK_SHA256)

        task_counts = Counter(row.identity.task_name for row in self.plan.rows)
        self.assertEqual(set(task_counts), set(self.contract.train_tasks))
        self.assertTrue(all(count == 12 for count in task_counts.values()))
        self.assertFalse(set(task_counts) & set(self.contract.holdout_tasks))

        for cycle in range(12):
            rows = self.plan.rows[cycle * 42 : (cycle + 1) * 42]
            self.assertEqual(len(rows), 42)
            self.assertEqual(
                {row.identity.task_name for row in rows},
                set(self.contract.train_tasks),
            )

    def test_each_task_has_four_occurrences_of_each_sigma(self) -> None:
        expected = {1.0: 4, 0.9: 4, 0.5: 4}
        for task in self.contract.train_tasks:
            observed = Counter(
                row.action_sigma
                for row in self.plan.rows
                if row.identity.task_name == task
            )
            self.assertEqual(dict(observed), expected)

    def test_only_plain_clean_50_sources_are_selected(self) -> None:
        for row in self.plan.rows:
            self.assertEqual(row.identity.source_dataset, "RoboTwin")
            self.assertEqual(row.identity.source_variant, "clean_50")
            self.assertEqual(row.identity.source_kind, "ordinary_expert")

        rejected_changes = (
            {"source_variant": "DAgger"},
            {"source_variant": "recovery"},
            {"source_variant": "history"},
            {"source_kind": "dagger"},
            {"source_kind": "recovery_annotation"},
            {"source_kind": "history_window"},
            {"source_dataset": "RoboTwin-recovery"},
        )
        first_task = self.contract.train_tasks[0]
        for changes in rejected_changes:
            with self.subTest(changes=changes):
                pools = self.make_pools(self.contract)
                pools[first_task][0] = replace(pools[first_task][0], **changes)
                with self.assertRaises(PlanValidationError):
                    TaskRoundRobinPlan.build(self.contract, pools)

    def test_build_is_byte_identical_under_input_reordering(self) -> None:
        reordered = self.make_pools(
            self.contract, reverse_tasks=True, reverse_samples=True
        )
        repeated_plan = TaskRoundRobinPlan.build(self.contract, reordered)
        self.assertEqual(
            self.plan.canonical_json_bytes(), repeated_plan.canonical_json_bytes()
        )
        self.assertEqual(self.plan.plan_sha256, repeated_plan.plan_sha256)
        self.assertEqual(self.plan.identity_sha256, repeated_plan.identity_sha256)
        self.assertEqual(self.plan.to_artifact_bytes(), repeated_plan.to_artifact_bytes())
        self.assertEqual(self.plan.plan_sha256, FIXTURE_PLAN_SHA256)
        self.assertEqual(self.plan.identity_sha256, FIXTURE_IDENTITY_SHA256)
        self.assertEqual(self.plan.jsonl_sha256, FIXTURE_JSONL_SHA256)
        self.assertIsInstance(hash(self.plan), int)

    def test_registry_task_input_order_is_explicitly_canonicalized(self) -> None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        registry["data"]["train_tasks"].reverse()
        registry["data"]["holdout_tasks"].reverse()
        reordered_contract = SamplerContract.from_registry(registry)
        self.assertEqual(reordered_contract, self.contract)

        registry["data"]["train_tasks"][0] = "not_a_frozen_task"
        with self.assertRaises(PlanValidationError):
            SamplerContract.from_registry(registry)

    def test_stateless_domain_seeds_repeat_and_do_not_alias(self) -> None:
        fields = ("adjust_bottle", 3, "episode_003", 9, "exact_prompt")
        observed = {
            domain: derive_seed(self.contract.protocol_seed, domain, *fields)
            for domain in ROW_SEED_DOMAINS
        }
        repeated = {
            domain: derive_seed(self.contract.protocol_seed, domain, *fields)
            for domain in ROW_SEED_DOMAINS
        }
        self.assertEqual(observed, repeated)
        self.assertEqual(len(set(observed.values())), len(ROW_SEED_DOMAINS))
        for row in self.plan.rows:
            self.assertEqual(tuple(dict(row.domain_seeds)), ROW_SEED_DOMAINS)
            self.assertEqual(len(set(dict(row.domain_seeds).values())), len(ROW_SEED_DOMAINS))

    def test_metadata_is_preserved_without_normalization(self) -> None:
        metadata = {
            "task_name": "place_object_stand",
            "episode_index": 17,
            "episode_path": r"D:\raw_data\under_score\episode_017.hdf5",
            "start_frame": 23,
            "prompt": "Do_NOT rewrite_this prompt: \u4e25\u683c\u4fdd\u7559",
            "unrelated_tensor": object(),
        }
        identity = SampleIdentity.from_metadata(metadata, dataset_index=9917)
        self.assertEqual(identity.task_name, metadata["task_name"])
        self.assertEqual(identity.episode_index, metadata["episode_index"])
        self.assertEqual(identity.episode_path, metadata["episode_path"])
        self.assertEqual(identity.start_frame, metadata["start_frame"])
        self.assertEqual(identity.prompt, metadata["prompt"])
        self.assertEqual(
            SampleIdentity.from_dict(identity.to_dict()).audit_tuple,
            identity.audit_tuple,
        )

    def test_missing_duplicate_and_short_pools_fail_closed(self) -> None:
        first_task = self.contract.train_tasks[0]

        missing = self.make_pools(self.contract)
        del missing[first_task]
        with self.assertRaisesRegex(PlanValidationError, "candidate task pools differ"):
            TaskRoundRobinPlan.build(self.contract, missing)

        short = self.make_pools(self.contract)
        short[first_task] = short[first_task][:11]
        with self.assertRaisesRegex(PlanValidationError, "12 unique windows required"):
            TaskRoundRobinPlan.build(self.contract, short)

        duplicate_identity = self.make_pools(self.contract)
        original = duplicate_identity[first_task][0]
        duplicate_identity[first_task][1] = replace(
            original, dataset_index=duplicate_identity[first_task][1].dataset_index
        )
        with self.assertRaisesRegex(PlanValidationError, "duplicate audit identity"):
            TaskRoundRobinPlan.build(self.contract, duplicate_identity)

        duplicate_index = self.make_pools(self.contract)
        duplicate_index[first_task][1] = replace(
            duplicate_index[first_task][1],
            dataset_index=duplicate_index[first_task][0].dataset_index,
        )
        with self.assertRaisesRegex(PlanValidationError, "dataset_index"):
            TaskRoundRobinPlan.build(self.contract, duplicate_index)

    def test_wrong_task_or_heldout_identity_fails_closed(self) -> None:
        first_task = self.contract.train_tasks[0]
        pools = self.make_pools(self.contract)
        pools[first_task][0] = replace(
            pools[first_task][0], task_name=self.contract.holdout_tasks[0]
        )
        with self.assertRaises(PlanValidationError):
            TaskRoundRobinPlan.build(self.contract, pools)

    def test_artifact_round_trip_checks_embedded_and_pinned_hashes(self) -> None:
        artifact = self.plan.to_artifact_bytes()
        loaded = TaskRoundRobinPlan.from_artifact_bytes(
            artifact,
            expected_plan_sha256=self.plan.plan_sha256,
            expected_identity_sha256=self.plan.identity_sha256,
        )
        self.assertEqual(loaded, self.plan)
        self.assertEqual(loaded.canonical_json_bytes(), self.plan.canonical_json_bytes())

        with self.assertRaisesRegex(PlanValidationError, "pinned value"):
            TaskRoundRobinPlan.from_artifact_bytes(
                artifact,
                expected_plan_sha256="0" * 64,
                expected_identity_sha256=self.plan.identity_sha256,
            )

    def test_outer_artifact_v2_binds_support_without_bumping_embedded_plan(self) -> None:
        artifact_bytes = self.plan.to_artifact_bytes()
        artifact = json.loads(artifact_bytes)
        self.assertEqual(
            artifact["artifact_schema_version"],
            "sana-phase6-task-plan-artifact-v2",
        )
        self.assertEqual(
            artifact["plan"]["schema_version"],
            "sana-phase6-task-plan-v1",
        )
        self.assertEqual(
            artifact["expansion_support_contract_sha256"],
            EXPANSION_SUPPORT_CONTRACT_SHA256,
        )
        self.assertEqual(
            artifact["expansion_eligibility_amendment_sha256"],
            EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
        )
        for key in (
            "expansion_support_contract_sha256",
            "expansion_eligibility_amendment_sha256",
        ):
            tampered = json.loads(artifact_bytes)
            tampered[key] = "0" * 64
            with self.assertRaises(PlanValidationError):
                TaskRoundRobinPlan.from_artifact_bytes(
                    canonical_json_bytes(tampered)
                )

    def test_outer_artifact_rehashes_embedded_support_contract_types(self) -> None:
        artifact_bytes = self.plan.to_artifact_bytes()
        mutations = (
            ("raw_window_frames", 113.0),
            ("bootstrap_clean_prefix", 1),
            (
                "video_sample_indices",
                [float(index) for index in range(0, 113, 4)],
            ),
        )
        for key, replacement in mutations:
            tampered = json.loads(artifact_bytes)
            tampered["expansion_support_contract"][key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(
                PlanValidationError,
                "expansion-support contract",
            ):
                TaskRoundRobinPlan.from_artifact_bytes(
                    canonical_json_bytes(tampered),
                    expected_plan_sha256=self.plan.plan_sha256,
                    expected_identity_sha256=self.plan.identity_sha256,
                )

    def test_serialized_prompt_and_count_tampering_fail_closed(self) -> None:
        artifact_value = json.loads(self.plan.to_artifact_bytes())
        artifact_value["plan"]["rows"][0]["identity"]["prompt"] += " tampered"
        tampered_bytes = canonical_json_bytes(artifact_value)
        with self.assertRaisesRegex(PlanValidationError, "embedded plan SHA256"):
            TaskRoundRobinPlan.from_artifact_bytes(tampered_bytes)

        truncated = replace(self.plan, rows=self.plan.rows[:-1])
        truncated_artifact = json.loads(self.plan.to_artifact_bytes())
        truncated_artifact["plan"] = truncated.to_payload()
        truncated_artifact["plan_sha256"] = truncated.plan_sha256
        truncated_artifact["identity_sha256"] = truncated.identity_sha256
        with self.assertRaisesRegex(PlanValidationError, "expected 504"):
            TaskRoundRobinPlan.from_artifact_bytes(
                canonical_json_bytes(truncated_artifact)
            )

    def test_noncanonical_or_duplicate_key_artifacts_fail_closed(self) -> None:
        artifact = self.plan.to_artifact_bytes()
        noncanonical = json.dumps(json.loads(artifact), indent=2).encode("utf-8")
        with self.assertRaisesRegex(PlanValidationError, "not canonical"):
            TaskRoundRobinPlan.from_artifact_bytes(noncanonical)

        duplicate = b'{"artifact_schema_version":"x","artifact_schema_version":"y"}\n'
        with self.assertRaisesRegex(PlanValidationError, "duplicate JSON key"):
            TaskRoundRobinPlan.from_artifact_bytes(duplicate)

    def test_prompt_start_task_seed_sigma_and_component_seed_are_hashed(self) -> None:
        first = self.plan.rows[0]
        alternate_sigma = next(
            sigma for sigma in self.plan.action_sigmas if sigma != first.action_sigma
        )
        mutations = (
            replace(
                self.plan,
                rows=(
                    replace(
                        first,
                        identity=replace(
                            first.identity,
                            episode_path=first.identity.episode_path + ".changed",
                        ),
                    ),
                )
                + self.plan.rows[1:],
            ),
            replace(
                self.plan,
                rows=(
                    replace(
                        first,
                        identity=replace(first.identity, prompt=first.identity.prompt + "!"),
                    ),
                )
                + self.plan.rows[1:],
            ),
            replace(
                self.plan,
                rows=(
                    replace(
                        first,
                        identity=replace(
                            first.identity, start_frame=first.identity.start_frame + 1
                        ),
                    ),
                )
                + self.plan.rows[1:],
            ),
            replace(
                self.plan,
                rows=(
                    replace(
                        first,
                        identity=replace(
                            first.identity, task_name=self.contract.holdout_tasks[0]
                        ),
                    ),
                )
                + self.plan.rows[1:],
            ),
            replace(self.plan, protocol_seed=self.plan.protocol_seed + 1),
            replace(
                self.plan,
                rows=(replace(first, action_sigma=alternate_sigma),) + self.plan.rows[1:],
            ),
            replace(
                self.plan,
                rows=(
                    replace(
                        first,
                        domain_seeds=(
                            (
                                first.domain_seeds[0][0],
                                first.domain_seeds[0][1] ^ 1,
                            ),
                        )
                        + first.domain_seeds[1:],
                    ),
                )
                + self.plan.rows[1:],
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation_sha=mutation.plan_sha256):
                self.assertNotEqual(mutation.plan_sha256, self.plan.plan_sha256)
                with self.assertRaises(PlanValidationError):
                    mutation.validate()

    def test_changing_protocol_seed_builds_a_different_valid_plan(self) -> None:
        alternate_contract = replace(
            self.contract, protocol_seed=self.contract.protocol_seed + 1
        )
        alternate = TaskRoundRobinPlan.build(alternate_contract, self.pools)
        alternate.validate()
        self.assertNotEqual(alternate.plan_sha256, self.plan.plan_sha256)
        self.assertNotEqual(alternate.identity_sha256, self.plan.identity_sha256)

    def test_sampler_is_fixed_and_yields_global_dataset_indices(self) -> None:
        sampler = PlanSampler(
            self.plan,
            expected_plan_sha256=self.plan.plan_sha256,
            expected_identity_sha256=self.plan.identity_sha256,
        )
        expected = [row.identity.dataset_index for row in self.plan.rows]
        self.assertEqual(list(sampler), expected)
        self.assertEqual(list(sampler), expected)
        self.assertEqual(len(sampler), 504)
        sampler.set_epoch(0)
        with self.assertRaises(PlanValidationError):
            sampler.set_epoch(1)
        with self.assertRaises(PlanValidationError):
            sampler.set_epoch(False)
        with self.assertRaisesRegex(PlanValidationError, "plan SHA256"):
            PlanSampler(
                self.plan,
                expected_plan_sha256="f" * 64,
                expected_identity_sha256=self.plan.identity_sha256,
            )

    def test_four_factorial_arms_share_exact_plan_and_identity_hashes(self) -> None:
        arms = {arm: self.plan for arm in FACTORIAL_ARMS[:4]}
        verify_shared_factorial_plans(
            arms,
            expected_plan_sha256=self.plan.plan_sha256,
            expected_identity_sha256=self.plan.identity_sha256,
        )

        changed_row = replace(
            self.plan.rows[0],
            identity=replace(self.plan.rows[0].identity, prompt="different prompt"),
        )
        different = replace(self.plan, rows=(changed_row,) + self.plan.rows[1:])
        arms[FACTORIAL_ARMS[3]] = different
        with self.assertRaises(PlanValidationError):
            verify_shared_factorial_plans(
                arms,
                expected_plan_sha256=self.plan.plan_sha256,
                expected_identity_sha256=self.plan.identity_sha256,
            )

    def test_json_and_jsonl_exports_are_canonical_and_complete(self) -> None:
        canonical = self.plan.canonical_json_bytes()
        self.assertTrue(canonical.endswith(b"\n"))
        self.assertEqual(sha256(canonical).hexdigest(), self.plan.plan_sha256)
        jsonl = self.plan.canonical_jsonl_bytes()
        self.assertTrue(jsonl.endswith(b"\n"))
        records = [json.loads(line) for line in jsonl.splitlines()]
        self.assertEqual(len(records), 505)
        self.assertEqual(records[0]["record_type"], "header")
        self.assertTrue(all(record["record_type"] == "sample" for record in records[1:]))
        self.assertEqual(sha256(jsonl).hexdigest(), self.plan.jsonl_sha256)


if __name__ == "__main__":
    unittest.main()
