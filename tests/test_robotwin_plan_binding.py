from __future__ import annotations

from pathlib import Path
import hashlib
import json
import random
from types import MethodType
import unittest
from unittest import mock


from sana_wam.dataloader.robotwin_dataset import (
    MultiTaskRoboTwinDataset,
    RoboTwinDataset,
)
from sana_wam.dataloader.robotwin_plan_binding import (
    PHASE6_PLAN_ROW_KEYS,
    PlanBoundRoboTwinDataset,
    enumerate_phase6_candidate_pools,
    phase6_expansion_support_at,
    validate_phase6_robotwin_dataset,
)
from sana_wam.dataloader.phase6_dataset_contract import (
    Phase6DatasetContract,
)
from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    expansion_support_contract,
)
from sana_wam.dataloader.phase6_padding_semantics import (
    PADDING_SEMANTICS_AMENDMENT_SHA256,
    PADDING_SEMANTICS_CONTRACT_SHA256,
    padding_semantics_contract,
)
from sana_wam.dataloader.task_sample_plan import (
    PlanValidationError,
    SamplerContract,
    TaskRoundRobinPlan,
    derive_prompt_choice_seed,
)


REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "logs/sana_principles_audit_20260722/phase6_principled_architecture_20260724"
    / "phase6_registry_seed20260724.json"
)
if not REGISTRY_PATH.is_file():
    REGISTRY_PATH = Path(__file__).resolve().parents[2] / "phase6_registry_seed20260724.json"


class RecordingTransform:
    def __init__(self, events: list[tuple[str, str]], name: str, *, dropout_p=0.0):
        self.events = events
        self.name = name
        self.dropout_p = dropout_p

    def apply(self, sample: dict) -> dict:
        self.events.append((self.name, sample["prompt"]))
        sample[f"{self.name}_applied"] = True
        return sample


def _fake_build_sample(
    self,
    ep_idx: int,
    start: int,
    logical_len=None,
    *,
    prompt_override=None,
):
    self.decode_calls += 1
    self.prompt_overrides.append(prompt_override)
    return {
        "video": [f"decoded:{self.task_name}:{start}"],
        "action": f"action:{start}",
        "prompt": prompt_override,
        "episode_index": ep_idx,
        "episode_path": self._episode_files[ep_idx],
        "start_frame": start,
        "task_name": self.task_name,
        "_is_static": True,
    }


def make_child(task: str, *, candidates: int = 13) -> RoboTwinDataset:
    child = RoboTwinDataset.__new__(RoboTwinDataset)
    child.variant = "clean_50"
    child.repeat = 1
    child.split = "train"
    child.task_name = task
    child.prompt_task_name = task.replace("_", " ")
    child.growing_history = False
    child._filter_static_segments = False
    child._text_embedding_transform = None
    child._vae_latent_transform = None
    child._val_samples = None
    child._window_index = [(0, index * 3, 113) for index in range(candidates)]
    child._episode_files = [f"/DATA/robotwin/clean_50/{task}/episode000.hdf5"]
    child._episode_lengths = [200]
    child._raw_window_len = 113
    child.num_frames = 113
    child.video_stride = 4
    child._video_sample_indices = list(range(0, 113, 4))
    child.num_video_frames = 29
    child.causal_temporal = True
    child.temporal_compression = 4
    child._instructions = {
        "episode000.json": {
            "seen": [
                f"First exact instruction for {task}",
                f"Second_exact instruction for {task}",
            ]
        }
    }
    child.decode_calls = 0
    child.prompt_overrides = []
    child._build_sample = MethodType(_fake_build_sample, child)
    return child


def make_multitask(
    contract: SamplerContract,
    *,
    candidates: int = 13,
) -> MultiTaskRoboTwinDataset:
    dataset = MultiTaskRoboTwinDataset.__new__(MultiTaskRoboTwinDataset)
    dataset.dataset_type = "robotwin"
    dataset.variant = "clean_50"
    dataset.action_mode = "eef"
    dataset.task_name = None
    dataset._task_names = list(contract.train_tasks)
    dataset._sub_datasets = [
        make_child(task, candidates=candidates) for task in contract.train_tasks
    ]
    dataset._cumulative_lengths = []
    cumulative = 0
    for child in dataset._sub_datasets:
        cumulative += len(child)
        dataset._cumulative_lengths.append(cumulative)
    dataset._total_length = cumulative
    dataset._action_dim_value = 20
    dataset._action_stats_shared = {"mean": [0.0], "std": [1.0]}
    dataset.action_stats_path = "/DATA/action_stats.npy"
    return dataset


class RoboTwinPlanBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = SamplerContract.from_registry(REGISTRY_PATH)

    def build_fixture(self):
        dataset = make_multitask(self.contract)
        with mock.patch.object(
            random,
            "choice",
            side_effect=AssertionError("global prompt RNG was touched"),
        ):
            pools = dataset.phase6_candidate_pools(
                self.contract,
                dataset_type="robotwin",
            )
        plan = TaskRoundRobinPlan.build(self.contract, pools)
        return dataset, pools, plan

    def bind(self, dataset, plan):
        dataset_contract = _make_dataset_contract(dataset, plan, self.contract)
        return PlanBoundRoboTwinDataset(
            dataset,
            plan,
            contract=self.contract,
            dataset_type="robotwin",
            world_size=1,
            expected_plan_sha256=plan.plan_sha256,
            expected_identity_sha256=plan.identity_sha256,
            dataset_contract=dataset_contract,
        )

    def test_identity_pool_enumeration_never_decodes_or_uses_global_prompt_rng(self):
        dataset, pools, _plan = self.build_fixture()
        self.assertEqual(set(pools), set(self.contract.train_tasks))
        self.assertTrue(all(len(pool) == 13 for pool in pools.values()))
        self.assertTrue(
            all(child.decode_calls == 0 for child in dataset._sub_datasets)
        )
        for task, identities in pools.items():
            self.assertTrue(all(identity.task_name == task for identity in identities))
            self.assertTrue(all("_" in identity.task_name for identity in identities))
            self.assertTrue(all(identity.source_variant == "clean_50" for identity in identities))

    def test_prompt_choice_is_stateless_and_exact(self):
        dataset = make_multitask(self.contract)
        index = 0
        random.seed(1)
        first = dataset.phase6_identity_at(
            index, protocol_seed=self.contract.protocol_seed
        )
        random.seed(999999)
        second = dataset.phase6_identity_at(
            index, protocol_seed=self.contract.protocol_seed
        )
        self.assertEqual(first, second)
        self.assertIn("executing the following instruction:", first.prompt)
        self.assertIn(first.task_name, first.episode_path)

    def test_recorded_prompt_seed_is_the_seed_that_selected_the_prompt(self):
        _dataset, _pools, plan = self.build_fixture()
        for row in plan.rows:
            identity = row.identity
            expected = derive_prompt_choice_seed(
                plan.protocol_seed,
                identity.task_name,
                identity.episode_index,
                identity.episode_path,
                identity.start_frame,
                identity.dataset_index,
            )
            self.assertEqual(row.seed_for("prompt-choice"), expected)

    def test_global_index_resolution_is_stable_and_bounds_checked(self):
        dataset = make_multitask(self.contract, candidates=13)
        self.assertEqual(dataset.resolve_global_index(0), (0, 0))
        self.assertEqual(dataset.resolve_global_index(12), (0, 12))
        self.assertEqual(dataset.resolve_global_index(13), (1, 0))
        self.assertEqual(
            dataset.resolve_global_index(len(dataset) - 1),
            (41, 12),
        )
        for bad in (-1, len(dataset), True, 1.5):
            with self.subTest(index=bad):
                with self.assertRaises((IndexError, TypeError)):
                    dataset.resolve_global_index(bad)

    def test_source_type_variant_repeat_static_history_and_cohort_fail_closed(self):
        mutations = (
            ("dataset_type", lambda dataset: None, "mixture"),
            ("variant", lambda dataset: setattr(dataset, "variant", "both"), "robotwin"),
            (
                "repeat",
                lambda dataset: setattr(dataset._sub_datasets[0], "repeat", 2),
                "robotwin",
            ),
            (
                "static",
                lambda dataset: setattr(
                    dataset._sub_datasets[0], "_filter_static_segments", True
                ),
                "robotwin",
            ),
            (
                "history",
                lambda dataset: setattr(
                    dataset._sub_datasets[0], "growing_history", True
                ),
                "robotwin",
            ),
            (
                "child_variant",
                lambda dataset: setattr(
                    dataset._sub_datasets[0], "variant", "recovery"
                ),
                "robotwin",
            ),
            (
                "cohort",
                lambda dataset: dataset._task_names.__setitem__(0, "wrong_task"),
                "robotwin",
            ),
        )
        for label, mutate, dataset_type in mutations:
            with self.subTest(label=label):
                dataset = make_multitask(self.contract)
                mutate(dataset)
                with self.assertRaises(PlanValidationError):
                    validate_phase6_robotwin_dataset(
                        dataset,
                        self.contract,
                        dataset_type=dataset_type,
                    )

    def test_text_embedding_dropout_fails_closed(self):
        dataset = make_multitask(self.contract)
        dataset._sub_datasets[0]._text_embedding_transform = RecordingTransform(
            [], "text", dropout_p=0.1
        )
        with self.assertRaisesRegex(PlanValidationError, "dropout"):
            enumerate_phase6_candidate_pools(
                dataset,
                self.contract,
                dataset_type="robotwin",
            )

    def test_plan_binding_validates_all_504_rows_without_decoding(self):
        dataset, _pools, plan = self.build_fixture()
        bound = self.bind(dataset, plan)
        self.assertEqual(len(plan.rows), 504)
        self.assertEqual(len(bound._rows_by_dataset_index), 504)
        self.assertTrue(
            all(child.decode_calls == 0 for child in dataset._sub_datasets)
        )

    def test_runtime_injects_exact_prompt_before_text_transform_and_skips_retry(self):
        dataset, _pools, plan = self.build_fixture()
        row = plan.rows[0]
        child_index, _local_index = dataset.resolve_global_index(
            row.identity.dataset_index
        )
        child = dataset._sub_datasets[child_index]
        events: list[tuple[str, str]] = []
        child._text_embedding_transform = RecordingTransform(events, "text")
        child._vae_latent_transform = RecordingTransform(events, "vae")
        bound = self.bind(dataset, plan)

        with mock.patch.object(
            random,
            "choice",
            side_effect=AssertionError("legacy random prompt path was touched"),
        ), mock.patch.object(
            random,
            "randint",
            side_effect=AssertionError("static retry path was touched"),
        ):
            sample = bound[row.identity.dataset_index]

        self.assertEqual(child.decode_calls, 1)
        self.assertEqual(child.prompt_overrides, [row.identity.prompt])
        self.assertEqual(
            events,
            [("text", row.identity.prompt), ("vae", row.identity.prompt)],
        )
        self.assertEqual(sample["prompt"], row.identity.prompt)
        self.assertEqual(sample["task_name"], row.identity.task_name)
        self.assertNotIn("_is_static", sample)

    def test_phase6_plan_row_is_one_complete_plain_dict(self):
        dataset, _pools, plan = self.build_fixture()
        row = plan.rows[0]
        sample = self.bind(dataset, plan)[row.identity.dataset_index]
        payload = sample["phase6_plan_row"]
        self.assertIs(type(payload), dict)
        self.assertEqual(set(payload), PHASE6_PLAN_ROW_KEYS)
        self.assertEqual(payload["global_step"], row.global_step)
        self.assertEqual(payload["action_sigma"], row.action_sigma)
        self.assertEqual(payload["identity"], row.identity.to_dict())
        self.assertEqual(payload["domain_seeds"], dict(row.domain_seeds))
        self.assertEqual(payload["plan_sha256"], plan.plan_sha256)
        self.assertEqual(payload["identity_sha256"], plan.identity_sha256)
        self.assertNotIn("plan_sha256", sample)
        self.assertNotIn("identity_sha256", sample)

    def test_unplanned_index_fails_closed(self):
        dataset, _pools, plan = self.build_fixture()
        bound = self.bind(dataset, plan)
        planned = {row.identity.dataset_index for row in plan.rows}
        unplanned = next(index for index in range(len(dataset)) if index not in planned)
        with self.assertRaisesRegex(PlanValidationError, "not present"):
            bound[unplanned]

    def test_dataset_drift_after_binding_fails_before_decode(self):
        dataset, _pools, plan = self.build_fixture()
        bound = self.bind(dataset, plan)
        row = plan.rows[0]
        child_index, _ = dataset.resolve_global_index(row.identity.dataset_index)
        child = dataset._sub_datasets[child_index]
        child._episode_files[0] += ".changed"
        with self.assertRaisesRegex(PlanValidationError, "identity drift"):
            bound[row.identity.dataset_index]
        self.assertEqual(child.decode_calls, 0)

    def test_logical_length_drift_after_binding_fails_before_decode(self):
        dataset, _pools, plan = self.build_fixture()
        bound = self.bind(dataset, plan)
        row = plan.rows[0]
        child, local_index = _selected_child(dataset, row)
        episode_index, start_frame, logical_length = child._window_index[local_index]
        child._window_index[local_index] = (
            episode_index,
            start_frame,
            logical_length - 1,
        )
        with self.assertRaisesRegex(PlanValidationError, "logical-window drift"):
            bound[row.identity.dataset_index]
        self.assertEqual(child.decode_calls, 0)

    def test_path_start_prompt_and_task_drift_fail_plan_binding(self):
        drift_mutators = (
            (
                "path",
                lambda dataset, row: dataset._sub_datasets[
                    dataset.resolve_global_index(row.identity.dataset_index)[0]
                ]._episode_files.__setitem__(0, row.identity.episode_path + ".moved"),
            ),
            (
                "start",
                lambda dataset, row: _mutate_selected_start(dataset, row),
            ),
            (
                "prompt",
                lambda dataset, row: _mutate_selected_prompt(dataset, row),
            ),
            (
                "task",
                lambda dataset, row: _mutate_selected_task(dataset, row),
            ),
        )
        for label, mutate in drift_mutators:
            with self.subTest(label=label):
                dataset, _pools, plan = self.build_fixture()
                mutate(dataset, plan.rows[0])
                with self.assertRaises(PlanValidationError):
                    self.bind(dataset, plan)

    def test_materializer_rechecks_exact_fields_after_transforms(self):
        dataset, _pools, plan = self.build_fixture()
        row = plan.rows[0]
        child_index, _ = dataset.resolve_global_index(row.identity.dataset_index)

        class RewritingTransform(RecordingTransform):
            def apply(self, sample):
                sample = super().apply(sample)
                sample["prompt"] += " rewritten"
                return sample

        dataset._sub_datasets[child_index]._text_embedding_transform = (
            RewritingTransform([], "text")
        )
        bound = self.bind(dataset, plan)
        with self.assertRaisesRegex(PlanValidationError, "prompt"):
            bound[row.identity.dataset_index]

    def test_bound_dataset_rejects_later_epochs(self):
        dataset, _pools, plan = self.build_fixture()
        bound = self.bind(dataset, plan)
        bound.set_epoch(0)
        for bad in (1, False, -1):
            with self.subTest(epoch=bad):
                with self.assertRaises(PlanValidationError):
                    bound.set_epoch(bad)

    def test_bound_dataset_requires_world_size_one(self):
        dataset, _pools, plan = self.build_fixture()
        for world_size in (2, 4, 0, True):
            with self.subTest(world_size=world_size):
                with self.assertRaisesRegex(PlanValidationError, "world_size=1"):
                    PlanBoundRoboTwinDataset(
                        dataset,
                        plan,
                        contract=self.contract,
                        dataset_type="robotwin",
                        world_size=world_size,
                        expected_plan_sha256=plan.plan_sha256,
                        expected_identity_sha256=plan.identity_sha256,
                        dataset_contract=None,
                    )


def _canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _make_dataset_contract(dataset, plan, sampler_contract):
    source_ids = tuple(
        sorted(
            (
                "sana_wam.dataloader.robotwin_dataset",
                "sana_wam.dataloader.transforms.multiview",
                "sana_wam.dataloader.transforms.normalize",
                "sana_wam.dataloader.transforms.rotation",
            )
        )
    )
    source_manifest = [
        {
            "source_id": source_id,
            "path": f"/source/{source_id}.py",
            "resolved_path": f"/source/{source_id}.py",
            "size_bytes": 1,
            "sha256": hashlib.sha256(source_id.encode()).hexdigest(),
        }
        for source_id in source_ids
    ]
    tasks = list(sampler_contract.train_tasks)
    preprocessing = {
        "action_mode": "eef",
        "causal_temporal": True,
        "delta_action": False,
        "filter_static_segments": False,
        "growing_history": False,
        "height": 384,
        "multiview": True,
        "normalize_mode": "min-max",
        "num_frames": 113,
        "repeat": 1,
        "robot": "aloha-agilex",
        "split": "train",
        "temporal_compression": 4,
        "text_embedding_dropout": 0.0,
        "val_ratio": 0.0,
        "variant": "clean_50",
        "video_stride": 4,
        "width": 320,
        "window_stride": 1,
        "camera_layout": ["head_camera", "left_camera", "right_camera"],
        "target_camera": "head_camera",
        "action_stats_path": "/DATA/action_stats.npy",
        "action_stats_sha256": "3" * 64,
        "vae_type": "wan",
        "raw_window_len": 113,
        "num_video_frames": 29,
        "video_sample_indices": list(range(0, 113, 4)),
        "history_min_frames": 2,
        "history_stride": 1,
        "gdn_chunk_size": 1,
        "static_segment_threshold": 1.0e-5,
        "max_static_retry": 3,
        "text_embedding_cache_dir": None,
        "vae_cache_dir": None,
        "dataset_dir": "/DATA/robotwin",
        "seed": 42,
        "num_val_samples": 4,
        "backbone": None,
        "robotwin_dataset_source_sha256": source_manifest[0]["sha256"],
        "preprocessing_source_manifest": source_manifest,
        "preprocessing_source_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(source_manifest)
        ).hexdigest(),
        "canonical_train_tasks": tasks,
        "canonical_train_tasks_sha256": hashlib.sha256(
            "".join(f"{task}\n" for task in sorted(tasks)).encode()
        ).hexdigest(),
        "canonical_train_task_order_sha256": hashlib.sha256(
            _canonical_json_bytes(tasks)
        ).hexdigest(),
    }
    episodes = []
    episode_map = {}
    for row in plan.rows:
        identity = row.identity
        if identity.episode_path not in episode_map:
            episode = {
                "task_name": identity.task_name,
                "episode_index": identity.episode_index,
                "episode_path": identity.episode_path,
                "resolved_path": identity.episode_path,
                "size_bytes": 1,
                "sha256": "4" * 64,
                "instruction_source": {
                    "kind": "json_file",
                    "path": identity.episode_path + ".json",
                    "resolved_path": identity.episode_path + ".json",
                    "size_bytes": 1,
                    "sha256": "5" * 64,
                    "fallback_descriptor": None,
                },
            }
            episode_map[identity.episode_path] = episode
            episodes.append(episode)
    episodes.sort(key=lambda episode: episode["episode_path"])
    rows = [
        {
            "global_step": row.global_step,
            "dataset_index": row.identity.dataset_index,
            "task_name": row.identity.task_name,
            "episode_index": row.identity.episode_index,
            "episode_path": row.identity.episode_path,
            "start_frame": row.identity.start_frame,
            "window_logical_length": 113,
            "episode_sha256": "4" * 64,
            "instruction_source_sha256": "5" * 64,
            **phase6_expansion_support_at(dataset, row.identity.dataset_index),
        }
        for row in plan.rows
    ]
    return Phase6DatasetContract(
        {
            "schema_version": "sana-phase6-dataset-contract-v2",
            "plan_sha256": plan.plan_sha256,
            "identity_sha256": plan.identity_sha256,
            "action_stats_sha256": "3" * 64,
            "expansion_eligibility_amendment_sha256": (
                EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
            ),
            "expansion_support_contract": expansion_support_contract(),
            "expansion_support_contract_sha256": (
                EXPANSION_SUPPORT_CONTRACT_SHA256
            ),
            "padding_semantics_amendment_sha256": (
                PADDING_SEMANTICS_AMENDMENT_SHA256
            ),
            "padding_semantics_contract": padding_semantics_contract(),
            "padding_semantics_contract_sha256": (
                PADDING_SEMANTICS_CONTRACT_SHA256
            ),
            "preprocessing_contract": preprocessing,
            "rows": rows,
            "episodes": episodes,
        }
    )


def _selected_child(dataset, row):
    child_index, local_index = dataset.resolve_global_index(row.identity.dataset_index)
    return dataset._sub_datasets[child_index], local_index


def _mutate_selected_start(dataset, row) -> None:
    child, local_index = _selected_child(dataset, row)
    ep_idx, start, logical_len = child._window_index[local_index]
    child._window_index[local_index] = (ep_idx, start + 1, logical_len)


def _mutate_selected_prompt(dataset, row) -> None:
    child, _ = _selected_child(dataset, row)
    child._instructions["episode000.json"]["seen"] = ["changed prompt"]


def _mutate_selected_task(dataset, row) -> None:
    child, _ = _selected_child(dataset, row)
    child.task_name = "different_task"


if __name__ == "__main__":
    unittest.main()
