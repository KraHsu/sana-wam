from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import pytest
import torch

from sana_wam.model import cach_a4_av3_paired_reduced_scale as av3


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale.yaml"
RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale.py"
MANIFEST_PATH = (
    REPO_ROOT
    / "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_DATA_SELECTION.json"
)
MANIFEST_SHA256 = "fecd1a9fb2f4371011d9c3c86469fed8f24f7f6aa5b3f1be9811fd447fb2ab95"
SELECTION_SHA256 = "49a56dc087d693a4f25005095f390996dab8639eb4e387bf8c70b872dfeb50d9"
GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def test_config_is_canonical_and_exact_materialization_only() -> None:
    payload = CONFIG_PATH.read_bytes()
    config = _json(CONFIG_PATH)
    assert payload == _canonical_bytes(config)
    assert frozenset(config) == {
        "architecture_id",
        "authority",
        "data_contract",
        "forbidden",
        "hardware_contract",
        "integration_path",
        "metric_contract",
        "operator_class",
        "operator_level",
        "resource_contract",
        "rollout_contract",
        "run_contract",
        "schema",
        "state",
        "thresholds",
        "training_contract",
        "verdict_contract",
    }
    assert config["schema"] == av3.AV3_CONFIG_SCHEMA
    assert config["architecture_id"] == av3.AV3_ARCHITECTURE_ID
    assert config["operator_level"] == "VENDOR_KERNEL"
    assert config["integration_path"] == "EXPERIMENTAL_PATH"
    assert config["state"] == "MATERIALIZED_NOT_EXECUTION_AUTHORIZED"

    authority = config["authority"]
    assert authority == {
        "execution_authorized": False,
        "materialization_authorized": True,
        "root_creation_authorized": False,
        "unlisted_capability_default": "DENIED",
    }
    forbidden = config["forbidden"]
    assert isinstance(forbidden, list)
    for denied in (
        "RUN_NAMESPACE_OR_ROOT_CREATE",
        "GPU_CUDA_TRITON_OR_JIT",
        "GPU_OR_VENDOR_MODEL_EXECUTION",
        "OPTIMIZER_OR_PARAMETER_UPDATE",
        "TRAINING",
        "CHECKPOINT_LOAD_OR_SAVE",
        "AUTOMATIC_EXECUTION_OR_RERUN",
    ):
        assert denied in forbidden

    data = config["data_contract"]
    assert isinstance(data, dict)
    assert data["dataset_root"] == "/DATA/share/RoboTwin2.0/dataset"
    assert data["manifest_path"] == av3.AV3_DATA_MANIFEST_PATH
    assert data["manifest_sha256"] == MANIFEST_SHA256
    assert data["train_microbatches"] == [
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    assert data["heldout_microbatches"] == [
        list(range(32, 40)),
        list(range(40, 48)),
    ]
    assert data["excluded_av2_episode_ids"] == list(range(16))
    assert data["raw_rows"] == [0, 33]
    assert data["action_rows"] == [1, 33]
    assert data["frame_indices"] == [0, 8, 16, 24, 32]
    assert data["target_forward_access"] is False

    training = config["training_contract"]
    assert training == {
        "betas": [0.9, 0.99],
        "candidate_only_initialization_seed": 2026080522,
        "checkpoint_load": False,
        "checkpoint_save": False,
        "final_model_state_save": False,
        "gradient_clip_global_l2": 1.0,
        "loss": "POSITIVE_MASKED_FUTURE_MSE_NO_SUBTRACTION",
        "lr_end": 3e-5,
        "lr_schedule": "COSINE",
        "lr_start": 3e-3,
        "optimizer": "AdamW",
        "optimizer_epsilon": 1e-8,
        "optimizer_steps_per_arm": 1000,
        "screen_seed": 202608030,
        "serial_arm_order": ["reference", "candidate"],
        "shared_initialization_seed": 2026080331,
        "train_action_mode": "CORRECT_ONLY",
        "train_batch_schedule": "STEP_INDEX_MODULO_2",
        "weight_decay": 0.0,
    }
    assert config["resource_contract"] == {
        "max_gpu_allocated_memory_bytes": 8 * 1024**3,
        "max_process_rss_bytes": 8 * 1024**3,
        "max_root_bytes": 64 * 1024**2,
        "max_total_wall_minutes": 60,
        "pre_root_observation_seconds": 120,
        "total_deadline_includes_pre_root_observation": True,
    }
    hardware = config["hardware_contract"]
    assert isinstance(hardware, dict)
    assert hardware["gpu_uuid"] == GPU_UUID
    assert hardware["cuda_visible_devices"] == GPU_UUID
    av3.validate_av3_config(config)


def test_frozen_manifest_is_statically_compatible_with_source_loader() -> None:
    payload = MANIFEST_PATH.read_bytes()
    manifest = _json(MANIFEST_PATH)
    assert payload == _canonical_bytes(manifest)
    assert hashlib.sha256(payload).hexdigest() == MANIFEST_SHA256
    assert manifest["schema"] == av3.AV3_DATA_MANIFEST_SCHEMA
    assert manifest["selection_digest"] == {
        "canonical_json": av3._canonical_compact_no_lf(
            av3._selection_digest_material()
        ),
        "payload_has_trailing_lf": False,
        "serialization": "UTF8_JSON_SORTED_KEYS_COMPACT_SEPARATORS_NO_TRAILING_LF",
        "sha256": SELECTION_SHA256,
    }
    assert (
        av3._verify_manifest_selection(
            manifest, dataset_root="/DATA/share/RoboTwin2.0/dataset"
        )
        == SELECTION_SHA256
    )

    expected_rows = av3._expected_file_rows(
        Path("/DATA/share/RoboTwin2.0/dataset")
    )
    selected = manifest["selected_files"]
    assert isinstance(selected, list)
    assert len(selected) == len(expected_rows) == 32
    for row, expected in zip(selected, expected_rows, strict=True):
        assert isinstance(row, dict)
        assert frozenset(row) == {
            "episode_id",
            "episode_length",
            "microbatch_index",
            "path",
            "sha256",
            "size_bytes",
            "split",
        }
        assert {name: row[name] for name in expected} == expected
        assert type(row["episode_length"]) is int and row["episode_length"] >= 33
        assert type(row["size_bytes"]) is int and row["size_bytes"] > 0
        assert av3._is_sha256(row["sha256"])

    derived = av3._derived_manifest_table(manifest["derived_microbatches"])
    assert frozenset(derived) == {
        ("train", 0),
        ("train", 1),
        ("heldout", 0),
        ("heldout", 1),
    }
    for row in derived.values():
        hashes = row["tensor_sha256"]
        assert isinstance(hashes, dict)
        assert frozenset(hashes) == {
            "action_valid_mask",
            "actions",
            "frame_valid_mask",
            "noisy_video",
            "proprio",
            "target",
        }
        assert all(av3._is_sha256(value) for value in hashes.values())
    assert manifest["adequacy"]["all_adequate"] is True
    assert len(manifest["derived_aggregates"]) == 2


def _target_microbatch() -> torch.Tensor:
    value = torch.zeros((8, 5, 3, 1, 1), dtype=torch.float32)
    for horizon in range(1, 5):
        value[:, horizon] = float(horizon)
    return value


def _prediction(
    target: torch.Tensor, *, anchor_offset: float, future_error: float
) -> torch.Tensor:
    value = target.clone()
    value[:, 0] = anchor_offset
    value[:, 1:] += future_error
    return value


def test_two_microbatch_metric_has_registered_denominator_and_primary() -> None:
    targets = (_target_microbatch(), _target_microbatch())
    masks = (
        torch.ones((8, 5), dtype=torch.bool),
        torch.ones((8, 5), dtype=torch.bool),
    )
    reference = tuple(
        _prediction(target, anchor_offset=1.0, future_error=3.0)
        for target in targets
    )
    candidate_correct = tuple(
        _prediction(target, anchor_offset=2.0, future_error=3.0)
        for target in targets
    )
    candidate_shuffle = tuple(
        _prediction(target, anchor_offset=0.0, future_error=4.0)
        for target in targets
    )
    metrics = av3.compute_av3_metrics(
        {
            "reference": {
                "correct": reference,
                "shuffle": reference,
                "no_action": reference,
            },
            "candidate": {
                "correct": candidate_correct,
                "shuffle": candidate_shuffle,
                "no_action": reference,
            },
        },
        targets,
        masks,
    )

    assert metrics["Q"] == pytest.approx({"1": 1.0, "2": 4.0, "4": 16.0})
    q124 = 0.2 * 1.0 + 0.3 * 4.0 + 0.5 * 16.0
    assert q124 == pytest.approx(9.4)
    for horizon in ("1", "2", "4"):
        assert metrics["V"]["reference"]["correct"][horizon] == pytest.approx(9.0)
        assert metrics["D"]["reference"]["correct"][horizon] == pytest.approx(4.0)
        assert metrics["V"]["candidate"]["correct"][horizon] == pytest.approx(9.0)
        assert metrics["D"]["candidate"]["correct"][horizon] == pytest.approx(1.0)
        assert metrics["V"]["candidate"]["shuffle"][horizon] == pytest.approx(16.0)
        assert metrics["D"]["candidate"]["shuffle"][horizon] == pytest.approx(16.0)

    primary = metrics["primary"]
    assert primary["reference"]["correct"] == pytest.approx(6.5 / q124)
    assert primary["candidate"]["correct"] == pytest.approx(5.0 / q124)
    assert primary["candidate"]["shuffle"] == pytest.approx(16.0 / q124)
    decision = metrics["decision_metrics"]
    assert decision["candidate_gain_vs_reference"] == pytest.approx(1.0 - 5.0 / 6.5)
    assert decision["candidate_to_reference_primary_ratio"] == pytest.approx(5.0 / 6.5)
    assert decision["correct_vs_shuffle_improvement"] == pytest.approx(1.0 - 5.0 / 16.0)
    assert decision["correct_vs_no_action_improvement"] == pytest.approx(1.0 - 5.0 / 6.5)
    assert decision["h1_candidate_vs_reference_ratio"] == pytest.approx(1.0)
    assert metrics["reference_modes_bitwise_equal"] is True


def _decision_metrics(
    *, gain: float, ratio: float, shuffle: float, h1_ratio: float
) -> dict[str, float]:
    return {
        "candidate_gain_vs_reference": gain,
        "candidate_to_reference_primary_ratio": ratio,
        "correct_vs_shuffle_improvement": shuffle,
        "h1_candidate_vs_reference_ratio": h1_ratio,
    }


def test_classifier_covers_go_stop_inconclusive_and_nonfinite() -> None:
    state, reason, checks = av3.classify_av3(
        all_validity=True,
        metrics=_decision_metrics(gain=0.10, ratio=0.90, shuffle=0.10, h1_ratio=1.0),
    )
    assert (state, reason) == ("GO", None)
    assert all(checks.values())

    state, reason, _ = av3.classify_av3(
        all_validity=True,
        metrics=_decision_metrics(gain=0.0, ratio=1.0, shuffle=0.0, h1_ratio=1.06),
    )
    assert (state, reason) == ("REDUCED_ARCH_STOP", "AV3_STRONG_STOP_LINE")
    state, reason, _ = av3.classify_av3(
        all_validity=True,
        metrics=_decision_metrics(gain=-0.25, ratio=1.25, shuffle=0.10, h1_ratio=1.0),
    )
    assert (state, reason) == ("REDUCED_ARCH_STOP", "AV3_STRONG_STOP_LINE")

    state, reason, _ = av3.classify_av3(
        all_validity=True,
        metrics=_decision_metrics(gain=0.04, ratio=0.96, shuffle=0.10, h1_ratio=1.0),
    )
    assert (state, reason) == (
        "REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED",
        "AV3_REVIEW_BAND_TOKEN_CONSUMED",
    )

    state, reason, checks = av3.classify_av3(
        all_validity=True,
        metrics=_decision_metrics(
            gain=float("nan"), ratio=1.0, shuffle=0.10, h1_ratio=1.0
        ),
    )
    assert (state, reason) == ("NUMERICAL_INVALID", "NONFINITE_DECISION_METRIC")
    assert dict(checks) == {}


def test_microbatch_schedule_starts_zero_one_and_rejects_out_of_range() -> None:
    assert av3._microbatch_schedule(0) == 0
    assert av3._microbatch_schedule(1) == 1
    assert av3._microbatch_schedule(2) == 0
    assert av3._microbatch_schedule(999) == 1
    for invalid in (-1, 1000, True):
        with pytest.raises(av3.AV3ExecutionError, match="outside the AV-3 schedule"):
            av3._microbatch_schedule(invalid)


def _call_records(function: object) -> list[tuple[int, str, str | None]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    records: list[tuple[int, str, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        cursor = node.func
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            parts.append(cursor.id)
        name = ".".join(reversed(parts))
        stage = None
        for keyword in node.keywords:
            if (
                keyword.arg == "stage"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                stage = keyword.value.value
        records.append((node.lineno, name, stage))
    return sorted(records)


def test_deadline_and_theta0_callback_ast_ordering() -> None:
    records = _call_records(av3.run_av3_paired_reduced_scale)

    def line(name: str, *, stage: str | None = None) -> int:
        return min(
            lineno
            for lineno, observed_name, observed_stage in records
            if observed_name == name and (stage is None or observed_stage == stage)
        )

    load_line = line("load_av3_real_data")
    adequacy_line = line("assess_av3_data_adequacy")
    cuda_probe_line = line("torch.cuda.is_available")
    materialize_line = line("av2.materialize_a4_task")
    build_line = line("a4.build_cach_a4_pair")
    theta0_line = line("theta0_callback")
    first_forward_boundary = min(
        line("_evaluate_outputs"),
        line("_mean_correct_loss"),
        line("_train_serial_arm"),
    )

    assert line("_require_before_deadline", stage="orchestrator_entry") < load_line
    assert load_line < adequacy_line
    assert (
        adequacy_line
        < line("_require_before_deadline", stage="post_data_adequacy")
        < cuda_probe_line
    )
    assert (
        cuda_probe_line
        < line("_require_before_deadline", stage="post_cuda_binding")
        < materialize_line
    )
    assert materialize_line < build_line < theta0_line < first_forward_boundary
    assert (
        theta0_line
        < line("_require_before_deadline", stage="post_theta0_manifest_publication")
        < first_forward_boundary
    )


def _ast_name(node: ast.expr) -> str:
    parts: list[str] = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
    return ".".join(reversed(parts))


def _ast_call_names(node: ast.AST) -> list[tuple[int, str]]:
    return [
        (call.lineno, _ast_name(call.func))
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    ]


def test_runner_pin_mode_contract_matches_immutable_byte_contract() -> None:
    tree = ast.parse(RUNNER_PATH.read_bytes(), filename=str(RUNNER_PATH))
    functions = {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    verify_pin = functions["_verify_pin"]
    frozen_argument = next(
        argument
        for argument in verify_pin.args.kwonlyargs
        if argument.arg == "frozen"
    )
    frozen_default = verify_pin.args.kw_defaults[
        verify_pin.args.kwonlyargs.index(frozen_argument)
    ]
    assert isinstance(frozen_default, ast.Constant)
    assert frozen_default.value is True

    def frozen_keywords(function_name: str) -> list[bool | None]:
        calls = sorted(
            (
                call
                for call in ast.walk(functions[function_name])
                if isinstance(call, ast.Call) and _ast_name(call.func) == "_verify_pin"
            ),
            key=lambda call: call.lineno,
        )
        values: list[bool | None] = []
        for call in calls:
            keyword = next(
                (item.value for item in call.keywords if item.arg == "frozen"),
                None,
            )
            if keyword is None:
                values.append(None)
            else:
                assert isinstance(keyword, ast.Constant)
                assert isinstance(keyword.value, bool)
                values.append(keyword.value)
        return values

    # Planning/direct/transitive predecessors are immutable by byte SHA.  Their
    # historical POSIX modes are not part of the frozen planning contract.
    assert frozen_keywords("_verify_planning_predecessors") == [False]
    # Materialized AV-3 files retain the default 0444 requirement; direct and
    # transitive predecessor pins use SHA-only byte immutability.
    assert frozen_keywords("_verify_card") == [None, False, False]


def test_runner_static_preflight_is_ast_isolated_from_execution_capabilities() -> None:
    # Parse bytes only: this test must not import or execute the runner, read its
    # cards/data, inspect a GPU, or create a namespace/root.
    tree = ast.parse(RUNNER_PATH.read_bytes(), filename=str(RUNNER_PATH))

    imported_roots: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module is not None:
            imported_roots.add(statement.module.split(".", 1)[0])
    assert "torch" not in imported_roots
    assert "sana_wam" not in imported_roots

    main = next(
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "main"
    )
    execution_try_index, execution_try = next(
        (index, statement)
        for index, statement in enumerate(main.body)
        if isinstance(statement, ast.Try)
        and any(isinstance(child, ast.If) for child in statement.body)
    )
    static_branch_index, static_branch = next(
        (index, statement)
        for index, statement in enumerate(execution_try.body)
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and node.attr == "static_preflight"
            for node in ast.walk(statement.test)
        )
    )
    assert isinstance(static_branch.body[-1], ast.Return)
    assert isinstance(static_branch.body[-1].value, ast.Constant)
    assert static_branch.body[-1].value.value == 0

    execution_capabilities = {
        "RootLifecycle",
        "_check_exact_env",
        "_load_source",
        "_observe_resources",
        "_verify_selected_data_files",
    }
    assert not (
        {name for _, name in _ast_call_names(static_branch)} & execution_capabilities
    )
    for statement in main.body[:execution_try_index]:
        assert not (
            {name for _, name in _ast_call_names(statement)} & execution_capabilities
        )
    for statement in execution_try.body[: static_branch_index + 1]:
        assert not (
            {name for _, name in _ast_call_names(statement)} & execution_capabilities
        )

    return_line = static_branch.body[-1].lineno
    capability_calls = [
        lineno
        for lineno, name in _ast_call_names(main)
        if name in execution_capabilities
    ]
    assert {name for _, name in _ast_call_names(main)} >= execution_capabilities
    assert return_line < min(capability_calls)
