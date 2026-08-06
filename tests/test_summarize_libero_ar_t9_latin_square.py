from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import stat
import statistics
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/summarize_libero_ar_t9_latin_square.py"

SPATIAL = "libero_spatial_no_noops_1.0.0_lerobot"
OBJECT = "libero_object_no_noops_1.0.0_lerobot"
GOAL = "libero_goal_no_noops_1.0.0_lerobot"
LIBERO10 = "libero_10_no_noops_1.0.0_lerobot"
SUITES = (SPATIAL, OBJECT, GOAL, LIBERO10)

T7_RESULT_SHA256 = "9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a"
T7_SOURCE_COMMIT = "d19109a2314f8e7186571afed6b4acd816d7cfab"
T7_RUNNER_SHA256 = "b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b"
T8_RESULT_SHA256 = "bfdb852e14a5bd9b1c8e776be9f4ff108899eae65d557cebe42b06f1991b0a18"
T8_SOURCE_COMMIT = "9cd1c490d14b3c2437e82225ee8cfdf58646837e"
T8_RUNNER_SHA256 = "9ec74d83c8443d78d2ba765dd766f82a96cccef894ae63d08cf9e2a8dbf97669"
CONFIG_SHA256 = "5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45"
T9_ARM_SOURCE_COMMIT = "f73a7950eded2787ad57b26523da686dc5722512"
T9_ARM_RUNNER_SHA256 = (
    "d0d4cacf35d4116db3a8c166ffcdc92578a28f5037614a960e22b4ccbcc4a84c"
)

CELL_MAP = {
    "A0": {0: "B", 1: "C", 2: "D", 3: "A"},
    "A1": {0: "C", 1: "D", 2: "A", 3: "B"},
    "A2": {0: "D", 1: "A", 2: "B", 3: "C"},
    "A3": {0: "A", 1: "B", 2: "C", 3: "D"},
}
TRAILING = {
    "A": {"A0": 3, "A1": 2, "A2": 1, "A3": 0},
    "B": {"A0": 0, "A1": 3, "A2": 2, "A3": 1},
    "C": {"A0": 1, "A1": 0, "A2": 3, "A3": 2},
    "D": {"A0": 2, "A1": 1, "A2": 0, "A3": 3},
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _source_tree() -> tuple[str, ast.Module]:
    source = SCRIPT.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(SCRIPT))


def _literal(node: ast.AST, names: dict[str, Any] | None = None) -> Any:
    names = {} if names is None else names
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(child, names) for child in node.elts)
    if isinstance(node, ast.List):
        return [_literal(child, names) for child in node.elts]
    if isinstance(node, ast.Set):
        return {_literal(child, names) for child in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _literal(key, names): _literal(value, names)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand, names)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal(node.left, names) + _literal(node.right, names)
    raise ValueError(f"not a static literal: {ast.dump(node)}")


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            matches.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            matches.append(node.value)
    assert len(matches) == 1, name
    return matches[0]


def _top_literal(tree: ast.Module, name: str) -> Any:
    return _literal(_assignment(tree, name))


def _path_argument(tree: ast.Module, name: str) -> str:
    node = _assignment(tree, name)
    assert isinstance(node, ast.Call)
    assert isinstance(node.func, ast.Name) and node.func.id == "Path"
    assert len(node.args) == 1
    return _literal(node.args[0])


def _call_names(tree: ast.AST) -> list[str]:
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        parts = []
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        if parts:
            result.append(".".join(reversed(parts)))
    return result


def _load_module():
    spec = importlib.util.spec_from_file_location("t9_aggregate_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_frozen_result(root: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    root.mkdir(parents=True)
    result = root / "RESULT.json"
    data = _canonical_bytes(payload)
    result.write_bytes(data)
    result.chmod(0o400)
    root.chmod(0o500)
    return result, hashlib.sha256(data).hexdigest()


def _arm_payload(
    arm: str,
    root: Path,
    *,
    nonce: str,
    source_commit: str,
    runner_sha256: str,
    q_by_label: dict[str, float],
) -> dict[str, Any]:
    schemas = {
        "A": "sana-wam-libero-t7-four-suite-cyclic-v1",
        "B": "sana-wam-libero-t8-phase-rotated-recency-v1",
        "C": "sana-wam-libero-t9-latin-square-arm-v1",
        "D": "sana-wam-libero-t9-latin-square-arm-v1",
    }
    verdicts = {
        "A": "T7_FOUR_SUITE_CYCLIC_GO",
        "B": "T8_PHASE_ROTATED_MIXED_INCONCLUSIVE",
        "C": "T9_LATIN_ARM_C_VALID",
        "D": "T9_LATIN_ARM_D_VALID",
    }
    schedules = {
        "A": ("A0", "A1", "A2", "A3"),
        "B": ("A1", "A2", "A3", "A0"),
        "C": ("A2", "A3", "A0", "A1"),
        "D": ("A3", "A0", "A1", "A2"),
    }
    update_rows = []
    heldout_rows = []
    observed = {}
    per_suite = []
    terminal_cells = []
    for index, suite in enumerate(SUITES):
        update_label = f"A{index}"
        heldout_label = f"H{index}"
        pre_a = {
            "A0": 13.679718971252441,
            "A1": 14.696584701538086,
            "A2": 12.06648063659668,
            "A3": 20.43115234375,
        }[update_label]
        pre_h = {
            "H0": 13.379679679870605,
            "H1": 14.698094367980957,
            "H2": 12.026171684265137,
            "H3": 20.885684967041016,
        }[heldout_label]
        q = q_by_label[update_label]
        # Deliberately use unequal rA/rH so tests detect accidental use of only
        # one member of the pair. Their arithmetic mean remains exactly q.
        r_a = q - 0.1
        r_h = q + 0.1
        terminal_a = pre_a * r_a
        terminal_h = pre_h * r_h
        r_a = terminal_a / pre_a
        r_h = terminal_h / pre_h
        q_terminal = statistics.mean((r_a, r_h))
        r_a_phase = r_a / 1.1
        r_h_phase = r_h / 1.1
        q_phase = statistics.mean((r_a_phase, r_h_phase))
        overwrite_penalty = q_terminal / q_phase
        update_identity = {
            "action_alignment": "absolute frame-0 aligned",
            "assets": {"asset": f"sha-update-{index}"},
            "dataset": suite,
            "episode_index": 10 + index,
            "episode_length": 100 + index,
            "role": "update",
            "start_frame": 0,
            "task": f"frozen task {index}",
            "task_index": index,
            "update_exposure_count": 5,
        }
        heldout_identity = {
            "action_alignment": "absolute frame-0 aligned",
            "assets": {"asset": f"sha-heldout-{index}"},
            "dataset": suite,
            "episode_index": 20 + index,
            "episode_length": 120 + index,
            "role": "fresh_same_task_probe",
            "start_frame": 0,
            "task": f"frozen task {index}",
            "task_index": index,
            "update_exposure_count": 0,
        }
        observed[update_label] = pre_a
        observed[heldout_label] = pre_h
        update_rows.append(
            {
                "after_action_loss": terminal_a,
                "after_total_loss": terminal_a,
                "assets": update_identity["assets"],
                "before_action_loss": pre_a,
                "before_total_loss": pre_a,
                "dataset": suite,
                "episode_index": update_identity["episode_index"],
                "episode_length": update_identity["episode_length"],
                "label": update_label,
                "post_to_pre_ratio": r_a,
                "start_frame": 0,
                "task": update_identity["task"],
                "task_index": index,
                "update_exposure_count": 5,
            }
        )
        heldout_rows.append(
            {
                "after_action_loss": terminal_h,
                "after_total_loss": terminal_h,
                "assets": heldout_identity["assets"],
                "before_action_loss": pre_h,
                "before_total_loss": pre_h,
                "dataset": suite,
                "episode_index": heldout_identity["episode_index"],
                "episode_length": heldout_identity["episode_length"],
                "label": heldout_label,
                "post_to_pre_ratio": r_h,
                "start_frame": 0,
                "task": heldout_identity["task"],
                "task_index": index,
                "update_exposure_count": 0,
            }
        )
        per_suite.append(
            {
                "dataset": suite,
                "heldout_label": heldout_label,
                "overwrite_penalty": overwrite_penalty,
                "q_phase": q_phase,
                "q_terminal": q_terminal,
                "rA_phase": r_a_phase,
                "rH_phase": r_h_phase,
                "terminal_heldout_ratio": r_h,
                "terminal_update_ratio": r_a,
                "trailing_updates": TRAILING[arm][update_label],
                "update_label": update_label,
            }
        )
        terminal_cells.append(
            {
                "dataset": suite,
                "heldout_label": heldout_label,
                "q_terminal": q,
                "rA_terminal": r_a,
                "rH_terminal": r_h,
                "trailing_updates": TRAILING[arm][update_label],
                "update_label": update_label,
            }
        )

    payload: dict[str, Any] = {
        "cyclic_schedule": {
            "arm": arm if arm in {"C", "D"} else None,
            "cycle_count": 5,
            "labels": list(schedules[arm]) * 5,
            "updates_per_sample": 5,
        },
        "cyclic_training": {"per_sample": update_rows},
        "execution_result": "PASS",
        "fresh_heldout_transfer": {"per_sample": heldout_rows},
        "harness_result": "PASS",
        "identity": {
            **({"arm": arm} if arm in {"C", "D"} else {}),
            "config_sha256": CONFIG_SHA256,
            "repo_commit": source_commit,
            "runner_sha256": runner_sha256,
            **(
                {
                    "t8_predecessor_result_sha256": T8_RESULT_SHA256,
                    "t8_predecessor_runner_sha256": T8_RUNNER_SHA256,
                    "t8_predecessor_source_commit": T8_SOURCE_COMMIT,
                }
                if arm in {"C", "D"}
                else {}
            ),
        },
        "result": verdicts[arm],
        "run": {
            "nonce": nonce,
            "root": str(root),
            **({"arm": arm} if arm in {"C", "D"} else {}),
        },
        "schema_version": schemas[arm],
        "scope": {
            "architecture_forwards_total": 36 if arm == "A" else 44,
            "architecture_measurement_forwards": 16 if arm == "A" else 24,
            "architecture_training_forwards": 20,
            "backward_calls": 20,
            "benchmark_evaluation_executed": False,
            "formal_training_executed": False,
            "fresh_heldout_samples_in_backward_or_update": 0,
            "optimizer_steps": 20,
            "prepare_inputs_calls": 8,
            "sana_wam_training_checkpoint_loaded": False,
            "sana_wam_training_checkpoint_saved": False,
            "simulator_executed": False,
        },
        "scientific_verdict": verdicts[arm],
        "sample_roles": {
            **{
                f"A{index}": {
                    "action_alignment": "absolute frame-0 aligned",
                    "assets": {"asset": f"sha-update-{index}"},
                    "dataset": suite,
                    "episode_index": 10 + index,
                    "episode_length": 100 + index,
                    "role": "update",
                    "start_frame": 0,
                    "task": f"frozen task {index}",
                    "task_index": index,
                    "update_exposure_count": 5,
                }
                for index, suite in enumerate(SUITES)
            },
            **{
                f"H{index}": {
                    "action_alignment": "absolute frame-0 aligned",
                    "assets": {"asset": f"sha-heldout-{index}"},
                    "dataset": suite,
                    "episode_index": 20 + index,
                    "episode_length": 120 + index,
                    "role": "fresh_same_task_probe",
                    "start_frame": 0,
                    "task": f"frozen task {index}",
                    "task_index": index,
                    "update_exposure_count": 0,
                }
                for index, suite in enumerate(SUITES)
            },
        },
        "starting_state_reproduction": {
            "observed_action_losses": observed,
            "reproduced": True,
        },
        "valid_run": True,
        "verdict": verdicts[arm],
    }
    if arm == "B":
        payload["phase_rotated_recency"] = {
            "per_suite": per_suite,
            "typed_verdict": verdicts[arm],
        }
    if arm in {"C", "D"}:
        payload["latin_square_arm"] = {
            "arm": arm,
            "arm_id": arm,
            "cycle_labels": list(schedules[arm]),
            "per_suite": per_suite,
            "terminal_cells": terminal_cells,
            "terminal_phase_reproduction": {
                label: True
                for label in (f"A{int(arm == 'D') + 1}", f"H{int(arm == 'D') + 1}")
            },
            "phase_overwrite_diagnostic": per_suite,
            "trailing_updates_by_label": TRAILING[arm],
            "typed_verdict": verdicts[arm],
        }
    return payload


def _frozen_arm_roots(tmp_path: Path) -> dict[str, dict[str, Any]]:
    q_by_suite_by_trailing = {
        "A0": (1.0, 3.0, 2.0, 4.0),  # rho == 0.8 boundary
        "A1": (1.0, 2.0, 3.0, 4.0),
        "A2": (1.0, 2.0, 4.0, 3.0),  # rho == 0.8 boundary
        "A3": (1.0, 3.0, 2.0, 4.0),
    }
    sources = {
        "A": T7_SOURCE_COMMIT,
        "B": T8_SOURCE_COMMIT,
        "C": T9_ARM_SOURCE_COMMIT,
        "D": T9_ARM_SOURCE_COMMIT,
    }
    runners = {
        "A": T7_RUNNER_SHA256,
        "B": T8_RUNNER_SHA256,
        "C": T9_ARM_RUNNER_SHA256,
        "D": T9_ARM_RUNNER_SHA256,
    }
    nonces = {
        "A": "61e0a0ff817970e994b0875be4840ed6",
        "B": "2e1efcc69bc552affb5c85b7feeb5175",
        "C": "1" * 32,
        "D": "2" * 32,
    }
    slugs = {
        "A": "libero-t7-foursuite-cyclic-fixed20",
        "B": "libero-t8-phase-rotated-fixed20",
        "C": "libero-t9-arm-c-latin-fixed20",
        "D": "libero-t9-arm-d-latin-fixed20",
    }
    result = {}
    for arm in ("A", "B", "C", "D"):
        q_by_label = {
            label: q_by_suite_by_trailing[label][TRAILING[arm][label]]
            for label in CELL_MAP
        }
        if arm in {"C", "D"}:
            root = (
                tmp_path
                / "inputs"
                / "t9"
                / sources[arm][:12]
                / f"{slugs[arm]}-{nonces[arm]}"
            )
        else:
            root = (
                tmp_path
                / "inputs"
                / arm.lower()
                / sources[arm][:12]
                / f"{slugs[arm]}-{nonces[arm]}"
            )
        payload = _arm_payload(
            arm,
            root,
            nonce=nonces[arm],
            source_commit=sources[arm],
            runner_sha256=runners[arm],
            q_by_label=q_by_label,
        )
        path, digest = _write_frozen_result(root, payload)
        result[arm] = {
            "nonce": nonces[arm],
            "path": path,
            "payload": payload,
            "root": root,
            "sha256": digest,
            "source_commit": sources[arm],
            "runner_sha256": runners[arm],
        }
    result["q_by_suite_by_trailing"] = q_by_suite_by_trailing
    return result


def test_static_contract_pins_cli_cell_map_and_cpu_only_scope() -> None:
    source, tree = _source_tree()
    for pin in (
        T7_RESULT_SHA256,
        T7_SOURCE_COMMIT,
        T7_RUNNER_SHA256,
        T8_RESULT_SHA256,
        T8_SOURCE_COMMIT,
        T8_RUNNER_SHA256,
        T9_ARM_SOURCE_COMMIT,
        T9_ARM_RUNNER_SHA256,
        CONFIG_SHA256,
    ):
        assert pin in source
    for literal in (
        "sana-wam-libero-t7-four-suite-cyclic-v1",
        "sana-wam-libero-t8-phase-rotated-recency-v1",
        "sana-wam-libero-t9-latin-square-arm-v1",
        "sana-wam-libero-t9-latin-square-combined-v1",
        "sana-wam-libero-t9-latin-square-combined-failure-v1",
        "T7_FOUR_SUITE_CYCLIC_GO",
        "T8_PHASE_ROTATED_MIXED_INCONCLUSIVE",
        "T9_LATIN_ARM_C_VALID",
        "T9_LATIN_ARM_D_VALID",
        "T9_COMMON_POSITION_EFFECT_SUPPORTED",
        "T9_POSITION_WITH_IDENTITY_INTERACTION",
        "T9_NO_COMMON_POSITION_EFFECT",
        "/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate",
        "libero-t9-latin-square-combined-",
    ):
        assert literal in source

    argument_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    argument_names = {node.args[0].value for node in argument_calls}
    assert argument_names == {
        "--arm-c-result",
        "--arm-c-result-sha256",
        "--arm-d-result",
        "--arm-d-result-sha256",
        "--run-root",
        "--nonce",
        "--expected-repo-commit",
        "--expected-runner-sha256",
    }

    imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"torch", "numpy", "sana_wam", "transformers"})
    calls = _call_names(tree)
    for forbidden in (
        "backward",
        "optimizer.step",
        "torch.load",
        "torch.save",
        "trainer.train",
    ):
        assert forbidden not in calls
    for false_scope in (
        '"checkpoint_loaded_or_saved": False',
        '"formal_training_or_evaluation_executed": False',
        '"gpu_or_model_executed": False',
        '"read_only_evidence_assembly": True',
    ):
        assert false_scope in source
    assert 'arms["C"]["nonce"] == arms["D"]["nonce"]' in source
    assert 'arms["C"]["root"] == arms["D"]["root"]' in source
    assert 'arms["C"]["result_sha256"] == arms["D"]["result_sha256"]' in source
    assert "if _ACTIVE_ROOT is not None:" in source
    assert "_terminalize_failure(_ACTIVE_ROOT, error)" in source

    # A literal or mechanically evaluable mapping must bind each suite/position
    # cell to exactly one of the four arms.
    module = _load_module()
    normalized = {
        label: {
            int(module.ARM_SPECS[arm]["trailing"][label]): arm
            for arm in ("A", "B", "C", "D")
        }
        for label in ("A0", "A1", "A2", "A3")
    }
    assert normalized == CELL_MAP


def test_classifier_freezes_rho_boundary_ties_and_ordered_verdicts() -> None:
    module = _load_module()
    classify = module._classify_latin_square_results
    common = {
        suite: dict(enumerate(values))
        for suite, values in zip(
            SUITES,
            (
                (1.0, 3.0, 2.0, 4.0),  # rho exactly 0.8
                (1.0, 2.0, 3.0, 4.0),
                (1.0, 2.0, 4.0, 3.0),  # rho exactly 0.8
                (1.0, 3.0, 2.0, 4.0),
            ),
            strict=True,
        )
    }
    result = classify(common)
    assert result["endpoint_count"] == 4
    assert result["strong_count"] == 4
    assert result["typed_verdict"] == "T9_COMMON_POSITION_EFFECT_SUPPORTED"
    assert result["per_suite"][0]["rho"] == pytest.approx(0.8)
    assert result["per_suite"][0]["strong_rho_at_least_0_8"] is True

    interaction = {suite: dict(values) for suite, values in common.items()}
    interaction[GOAL] = dict(enumerate((1.0, 4.0, 3.0, 2.0)))
    interaction[LIBERO10] = dict(enumerate((1.0, 3.0, 4.0, 2.0)))
    result = classify(interaction)
    assert result["endpoint_count"] == 4
    assert result["strong_count"] == 2
    assert result["typed_verdict"] == "T9_POSITION_WITH_IDENTITY_INTERACTION"

    no_common = {suite: dict(values) for suite, values in common.items()}
    no_common[LIBERO10][3] = no_common[LIBERO10][0]  # equality is not worse
    result = classify(no_common)
    assert result["endpoint_count"] == 3
    assert result["typed_verdict"] == "T9_NO_COMMON_POSITION_EFFECT"

    tied = {suite: dict(values) for suite, values in common.items()}
    tied[SPATIAL] = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    row = classify(tied)["per_suite"][0]
    assert row["rho"] is None
    assert row["strong_rho_at_least_0_8"] is False

    with pytest.raises(ValueError, match="suite"):
        classify(common | {"wrong": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}})
    with pytest.raises(ValueError, match="trailing"):
        classify(common | {SPATIAL: {0: 1.0, 1: 2.0, 3: 4.0}})
    with pytest.raises(ValueError, match=r"finite and .*positive"):
        classify(common | {SPATIAL: {0: 1.0, 1: 2.0, 2: math.nan, 3: 4.0}})
    with pytest.raises(ValueError, match=r"finite and .*positive"):
        classify(common | {SPATIAL: {0: 0.0, 1: 2.0, 2: 3.0, 3: 4.0}})


def test_frozen_result_loader_accepts_only_canonical_single_file_0400_0500(
    tmp_path: Path,
) -> None:
    module = _load_module()
    frozen = _frozen_arm_roots(tmp_path / "happy")
    result, evidence = module._load_frozen_result(
        frozen["A"]["path"], frozen["A"]["sha256"], arm="A"
    )
    assert result == frozen["A"]["payload"]
    assert evidence["result_sha256"] == frozen["A"]["sha256"]
    assert evidence["file_mode"] == "0400"
    assert evidence["root_mode"] == "0500"
    assert evidence["single_result_file"] is True

    bad_sha = _frozen_arm_roots(tmp_path / "sha")["A"]
    with pytest.raises(ValueError, match="SHA256"):
        module._load_frozen_result(bad_sha["path"], "0" * 64, arm="A")

    noncanonical = _frozen_arm_roots(tmp_path / "canonical")["A"]
    noncanonical["root"].chmod(0o700)
    noncanonical["path"].chmod(0o600)
    payload = (json.dumps(noncanonical["payload"], indent=2) + "\n").encode()
    noncanonical["path"].write_bytes(payload)
    noncanonical["path"].chmod(0o400)
    noncanonical["root"].chmod(0o500)
    with pytest.raises(ValueError, match="canonical"):
        module._load_frozen_result(
            noncanonical["path"], hashlib.sha256(payload).hexdigest(), arm="A"
        )

    wrong_file_mode = _frozen_arm_roots(tmp_path / "file-mode")["A"]
    wrong_file_mode["path"].chmod(0o444)
    with pytest.raises(PermissionError, match="0400"):
        module._load_frozen_result(
            wrong_file_mode["path"], wrong_file_mode["sha256"], arm="A"
        )

    wrong_root_mode = _frozen_arm_roots(tmp_path / "root-mode")["A"]
    wrong_root_mode["root"].chmod(0o555)
    with pytest.raises(PermissionError, match="0500"):
        module._load_frozen_result(
            wrong_root_mode["path"], wrong_root_mode["sha256"], arm="A"
        )

    extra = _frozen_arm_roots(tmp_path / "extra")["A"]
    extra["root"].chmod(0o700)
    (extra["root"] / "EXTRA.json").write_text("{}\n", encoding="utf-8")
    (extra["root"] / "EXTRA.json").chmod(0o400)
    extra["root"].chmod(0o500)
    with pytest.raises(ValueError, match="only RESULT"):
        module._load_frozen_result(extra["path"], extra["sha256"], arm="A")

    linked = _frozen_arm_roots(tmp_path / "symlink")["A"]
    target = tmp_path / "symlink-target.json"
    target.write_bytes(_canonical_bytes(linked["payload"]))
    target.chmod(0o400)
    linked["root"].chmod(0o700)
    linked["path"].unlink()
    linked["path"].symlink_to(target)
    linked["root"].chmod(0o500)
    with pytest.raises(ValueError, match="symlink"):
        module._load_frozen_result(linked["path"], linked["sha256"], arm="A")


def _validated_synthetic_arms(
    module: Any, tmp_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = _frozen_arm_roots(tmp_path)
    module.T9_ARM_NAMESPACE = tmp_path / "inputs" / "t9"
    arms = {}
    for arm in ("A", "B", "C", "D"):
        result, evidence = module._load_frozen_result(
            frozen[arm]["path"], frozen[arm]["sha256"], arm=arm
        )
        arms[arm] = module._validate_arm_result(arm, result, evidence)
    return frozen, arms


def test_arm_validation_binds_schema_verdict_source_runner_config_and_scope(
    tmp_path: Path,
) -> None:
    module = _load_module()
    frozen, arms = _validated_synthetic_arms(module, tmp_path / "valid")
    assert set(arms) == {"A", "B", "C", "D"}
    assert arms["C"]["nonce"] != arms["D"]["nonce"]
    assert arms["C"]["source_commit"] == arms["D"]["source_commit"]
    assert arms["C"]["runner_sha256"] == arms["D"]["runner_sha256"]

    original = frozen["C"]["payload"]
    _loaded, evidence = module._load_frozen_result(
        frozen["C"]["path"], frozen["C"]["sha256"], arm="C"
    )

    def rejects(mutator: Any, match: str) -> None:
        candidate = json.loads(json.dumps(original))
        mutator(candidate)
        with pytest.raises(ValueError, match=match):
            module._validate_arm_result("C", candidate, evidence)

    rejects(lambda row: row.__setitem__("schema_version", "wrong"), "schema")
    rejects(lambda row: row.__setitem__("scientific_verdict", "wrong"), "verdict")
    rejects(lambda row: row["identity"].__setitem__("repo_commit", "0" * 40), "source")
    rejects(
        lambda row: row["identity"].__setitem__("runner_sha256", "0" * 64), "runner"
    )
    rejects(
        lambda row: row["identity"].__setitem__("config_sha256", "0" * 64), "config"
    )
    rejects(lambda row: row["scope"].__setitem__("optimizer_steps", 19), "scope")
    rejects(lambda row: row["run"].__setitem__("nonce", "1" * 31), "nonce")


def test_arm_validator_rejects_inconsistent_terminal_status_fields(
    tmp_path: Path,
) -> None:
    module = _load_module()
    frozen = _frozen_arm_roots(tmp_path)
    result, evidence = module._load_frozen_result(
        frozen["C"]["path"], frozen["C"]["sha256"], arm="C"
    )
    module.T9_ARM_NAMESPACE = tmp_path / "inputs" / "t9"

    for field, bad_value, match in (
        ("result", "T9_NO_COMMON_POSITION_EFFECT", "verdict"),
        ("execution_result", "FAIL", "execution or harness"),
        ("harness_result", "FAIL", "execution or harness"),
    ):
        candidate = json.loads(json.dumps(result))
        candidate[field] = bad_value
        with pytest.raises(ValueError, match=match):
            module._validate_arm_result("C", candidate, evidence)


def test_arm_validator_rejects_incomplete_measurement_and_nonformal_scope(
    tmp_path: Path,
) -> None:
    module = _load_module()
    frozen = _frozen_arm_roots(tmp_path)
    result, evidence = module._load_frozen_result(
        frozen["C"]["path"], frozen["C"]["sha256"], arm="C"
    )
    module.T9_ARM_NAMESPACE = tmp_path / "inputs" / "t9"
    for key, bad_value in (
        ("architecture_measurement_forwards", 23),
        ("benchmark_evaluation_executed", True),
    ):
        candidate = json.loads(json.dumps(result))
        candidate["scope"][key] = bad_value
        with pytest.raises(ValueError, match="scope"):
            module._validate_arm_result("C", candidate, evidence)


def test_arm_validator_requires_exact_phase_replay_and_overwrite_diagnostics(
    tmp_path: Path,
) -> None:
    module = _load_module()
    frozen = _frozen_arm_roots(tmp_path)
    result, evidence = module._load_frozen_result(
        frozen["C"]["path"], frozen["C"]["sha256"], arm="C"
    )
    module.T9_ARM_NAMESPACE = tmp_path / "inputs" / "t9"

    empty_replay = json.loads(json.dumps(result))
    empty_replay["latin_square_arm"]["terminal_phase_reproduction"] = {}
    with pytest.raises(ValueError, match="phase replay"):
        module._validate_arm_result("C", empty_replay, evidence)

    missing_diagnostic = json.loads(json.dumps(result))
    missing_diagnostic["latin_square_arm"].pop("phase_overwrite_diagnostic")
    with pytest.raises(ValueError, match="overwrite"):
        module._validate_arm_result("C", missing_diagnostic, evidence)


def test_assembly_recomputes_all_sixteen_raw_loss_cells_and_cell_map(
    tmp_path: Path,
) -> None:
    module = _load_module()
    frozen, arms = _validated_synthetic_arms(module, tmp_path)
    assembly = module._assemble_latin_square(arms)
    assert len(assembly["cells"]) == 16
    assert assembly["typed_verdict"] == "T9_COMMON_POSITION_EFFECT_SUPPORTED"
    assert assembly["classifier"]["endpoint_count"] == 4
    assert assembly["classifier"]["strong_count"] == 4
    assert assembly["arm_to_cell"] == {
        arm: {label: trailing for label, trailing in TRAILING[arm].items()}
        for arm in ("A", "B", "C", "D")
    }
    for index, suite in enumerate(SUITES):
        label = f"A{index}"
        for trailing in range(4):
            expected_q = frozen["q_by_suite_by_trailing"][label][trailing]
            assert assembly["matrices"]["q"][suite][str(trailing)] == pytest.approx(
                expected_q
            )
            r_a = assembly["matrices"]["rA"][suite][str(trailing)]
            r_h = assembly["matrices"]["rH"][suite][str(trailing)]
            assert assembly["matrices"]["q"][suite][str(trailing)] == pytest.approx(
                (r_a + r_h) / 2.0
            )
            expected_arm = CELL_MAP[label][trailing]
            cell = next(
                row
                for row in assembly["cells"]
                if row["suite"] == suite and row["trailing_updates"] == trailing
            )
            assert cell["arm"] == expected_arm
            assert cell["evidence"]["result_sha256"] == frozen[expected_arm]["sha256"]

    corrupted = json.loads(json.dumps(arms))
    corrupted["D"]["rows"]["A0"]["sample_identity"]["episode_index"] += 1
    with pytest.raises(ValueError, match="sample identity"):
        module._assemble_latin_square(corrupted)


def test_success_and_failure_roots_are_canonical_immutable_and_never_reused(
    tmp_path: Path,
) -> None:
    module = _load_module()
    parent = tmp_path / "aggregate" / ("c" * 12)
    parent.mkdir(parents=True)

    success = parent / f"libero-t9-latin-square-combined-{'3' * 32}"
    assert module._create_run_root(success) == success
    assert stat.S_IMODE(success.stat().st_mode) == 0o700
    report = {
        "result": "T9_NO_COMMON_POSITION_EFFECT",
        "schema_version": module.T9_RESULT_SCHEMA,
        "valid_run": True,
    }
    module._write_terminal_report(success / "RESULT.json", report)
    module._freeze_run_root(success)
    success_bytes = (success / "RESULT.json").read_bytes()
    assert success_bytes == _canonical_bytes(report)
    assert stat.S_IMODE((success / "RESULT.json").stat().st_mode) == 0o444
    assert stat.S_IMODE(success.stat().st_mode) == 0o555
    with pytest.raises(FileExistsError, match="reuse"):
        module._create_run_root(success)
    assert (success / "RESULT.json").read_bytes() == success_bytes

    failed = parent / f"libero-t9-latin-square-combined-{'4' * 32}"
    module._create_run_root(failed)
    module._terminalize_failure(failed, RuntimeError("synthetic aggregate failure"))
    assert [path.name for path in failed.iterdir()] == ["FAILED.json"]
    failed_bytes = (failed / "FAILED.json").read_bytes()
    failed_report = json.loads(failed_bytes)
    assert failed_bytes == _canonical_bytes(failed_report)
    assert failed_report["schema_version"] == module.T9_FAILURE_SCHEMA
    assert failed_report["result"] == "FAIL"
    assert failed_report["error_type"] == "RuntimeError"
    assert stat.S_IMODE((failed / "FAILED.json").stat().st_mode) == 0o444
    assert stat.S_IMODE(failed.stat().st_mode) == 0o555

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "linked-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._create_run_root(symlink_parent / "new-root")


def test_aggregate_source_identity_binds_cli_to_self_not_arm_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    aggregate_commit = "e" * 40
    aggregate_runner_sha256 = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "_repo_commit", lambda: aggregate_commit)
    assert module._aggregate_source_identity(
        aggregate_commit, aggregate_runner_sha256
    ) == {
        "repo_commit": aggregate_commit,
        "runner_path": str(SCRIPT.resolve()),
        "runner_sha256": aggregate_runner_sha256,
    }
    with pytest.raises(RuntimeError, match="identity differs"):
        module._aggregate_source_identity("f" * 40, aggregate_runner_sha256)
    with pytest.raises(RuntimeError, match="identity differs"):
        module._aggregate_source_identity(aggregate_commit, "0" * 64)


def test_main_builds_one_fresh_canonical_aggregate_from_four_frozen_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    frozen = _frozen_arm_roots(tmp_path)
    # In production C/D bind the real frozen T8 SHA. This synthetic end-to-end
    # case replaces T8 itself, so refresh the same direct pin in both arm files.
    for arm in ("C", "D"):
        artifact = frozen[arm]
        artifact["payload"]["identity"]["t8_predecessor_result_sha256"] = frozen["B"][
            "sha256"
        ]
        artifact["root"].chmod(0o700)
        artifact["path"].chmod(0o600)
        data = _canonical_bytes(artifact["payload"])
        artifact["path"].write_bytes(data)
        artifact["path"].chmod(0o400)
        artifact["root"].chmod(0o500)
        artifact["sha256"] = hashlib.sha256(data).hexdigest()
    module.T7_RESULT_PATH = frozen["A"]["path"]
    module.T7_RESULT_SHA256 = frozen["A"]["sha256"]
    module.T8_RESULT_PATH = frozen["B"]["path"]
    module.T8_RESULT_SHA256 = frozen["B"]["sha256"]
    module.T9_ARM_NAMESPACE = tmp_path / "inputs" / "t9"
    namespace = tmp_path / "aggregate"
    aggregate_commit = "e" * 40
    aggregate_runner_sha256 = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "_repo_commit", lambda: aggregate_commit)
    (namespace / aggregate_commit[:12]).mkdir(parents=True)
    module.T9_AGGREGATE_NAMESPACE = namespace
    nonce = "3" * 32
    run_root = (
        namespace / aggregate_commit[:12] / f"libero-t9-latin-square-combined-{nonce}"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--arm-c-result",
            str(frozen["C"]["path"]),
            "--arm-c-result-sha256",
            frozen["C"]["sha256"],
            "--arm-d-result",
            str(frozen["D"]["path"]),
            "--arm-d-result-sha256",
            frozen["D"]["sha256"],
            "--run-root",
            str(run_root),
            "--nonce",
            nonce,
            "--expected-repo-commit",
            aggregate_commit,
            "--expected-runner-sha256",
            aggregate_runner_sha256,
        ],
    )
    assert module.main() == 0
    result_path = run_root / "RESULT.json"
    payload = result_path.read_bytes()
    report = json.loads(payload)
    assert payload == _canonical_bytes(report)
    assert report["schema_version"] == module.T9_RESULT_SCHEMA
    assert report["valid_run"] is True
    assert report["scientific_verdict"] == "T9_COMMON_POSITION_EFFECT_SUPPORTED"
    assert report["verdict"] == report["result"] == report["scientific_verdict"]
    assert report["assembly"]["classifier"]["endpoint_count"] == 4
    assert report["assembly"]["classifier"]["strong_count"] == 4
    assert report["identity"] == {
        "config_sha256": CONFIG_SHA256,
        "repo_commit": aggregate_commit,
        "runner_path": str(SCRIPT.resolve()),
        "runner_sha256": aggregate_runner_sha256,
        "t9_arm_runner_sha256": T9_ARM_RUNNER_SHA256,
        "t9_arm_source_commit": T9_ARM_SOURCE_COMMIT,
    }
    assert run_root.parent.name == aggregate_commit[:12]
    assert report["arm_identities"]["C"]["source_commit"] == T9_ARM_SOURCE_COMMIT
    assert report["arm_identities"]["D"]["runner_sha256"] == T9_ARM_RUNNER_SHA256
    assert report["scope"] == {
        "checkpoint_loaded_or_saved": False,
        "formal_training_or_evaluation_executed": False,
        "gpu_or_model_executed": False,
        "input_result_count": 4,
        "read_only_evidence_assembly": True,
    }
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(run_root.stat().st_mode) == 0o555
    before = hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError, match="reuse"):
        module.main()
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == before


def test_main_rejects_same_cd_nonce_and_entrypoint_failure_can_freeze_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    commit = "c" * 40
    runner_sha = "d" * 64
    nonce = "3" * 32
    namespace = tmp_path / "aggregate"
    (namespace / commit[:12]).mkdir(parents=True)
    module.T9_AGGREGATE_NAMESPACE = namespace
    monkeypatch.setattr(
        module,
        "_aggregate_source_identity",
        lambda expected_commit, expected_runner_sha256: {
            "repo_commit": expected_commit,
            "runner_path": str(SCRIPT.resolve()),
            "runner_sha256": expected_runner_sha256,
        },
    )
    module.T7_RESULT_PATH = tmp_path / "A" / "RESULT.json"
    module.T8_RESULT_PATH = tmp_path / "B" / "RESULT.json"
    module.T7_RESULT_SHA256 = "a" * 64
    module.T8_RESULT_SHA256 = "b" * 64
    c_path = tmp_path / "C" / "RESULT.json"
    d_path = tmp_path / "D" / "RESULT.json"
    run_root = namespace / commit[:12] / f"libero-t9-latin-square-combined-{nonce}"

    def fake_load(path: Path, expected_sha256: str, *, arm: str):
        return {}, {
            "gid": 1000,
            "result_path": str(path),
            "result_sha256": expected_sha256,
            "root": str(path.parent),
            "uid": 1000,
        }

    def fake_validate(
        arm: str,
        _result: dict[str, Any],
        evidence: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "nonce": "1" * 32 if arm in {"C", "D"} else arm.lower() * 32,
            "result_sha256": evidence["result_sha256"],
            "root": evidence["root"],
        }

    monkeypatch.setattr(module, "_load_frozen_result", fake_load)
    monkeypatch.setattr(module, "_validate_arm_result", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--arm-c-result",
            str(c_path),
            "--arm-c-result-sha256",
            "1" * 64,
            "--arm-d-result",
            str(d_path),
            "--arm-d-result-sha256",
            "2" * 64,
            "--run-root",
            str(run_root),
            "--nonce",
            nonce,
            "--expected-repo-commit",
            commit,
            "--expected-runner-sha256",
            runner_sha,
        ],
    )
    with pytest.raises(ValueError, match="different nonces") as captured:
        module.main()
    assert module._ACTIVE_ROOT == run_root
    module._terminalize_failure(run_root, captured.value)
    assert stat.S_IMODE(run_root.stat().st_mode) == 0o555
    assert stat.S_IMODE((run_root / "FAILED.json").stat().st_mode) == 0o444
    failure = json.loads((run_root / "FAILED.json").read_bytes())
    assert failure["schema_version"] == module.T9_FAILURE_SCHEMA
