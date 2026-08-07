from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/smoke_libero_ar_t16_paired_one_task_closed_loop_gpu.py"
WORKER = ROOT / "scripts/libero_t16_sim_worker.py"
SIM_CONFIG = ROOT / "configs/benchmarks/libero/t16_simulator/config.yaml"
T15_RUNNER = ROOT / "scripts/smoke_libero_ar_t15_lossrecipe_replication_accum8_gpu.py"

EXPECTED_SIM_CONFIG = (
    "assets: /home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO/"
    "libero/libero/assets\n"
    "bddl_files: /home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO/"
    "libero/libero/bddl_files\n"
    "benchmark_root: /home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO/"
    "libero/libero\n"
    "datasets: /DATA/share/LIBERO/libero\n"
    "init_states: /home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO/"
    "libero/libero/init_files\n"
)

T15_SOURCE_COMMIT = "d669a2e0bc201941fd911f484df36d7afec44cd6"
T15_RUNNER_SHA256 = "25ebbe148287a7ee9795f56e9253b1db238d0cf5dbea52778c7b61932ee9023a"
T15_RESULT_SHA256 = "9e4ee9cb515e5cfafc5598eff33041232ad702b32e641b26435bcb1c643820b7"
T15_ROOT = (
    "/DATA/share/sana_wam_libero_nonformal_screens/t15/d669a2e0bc20/"
    "libero-t15-loss-recipe-seed-replication-accum8-balanced-joint-fixed20-"
    "9732531de18094840229d7ca0598a648"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_and_tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _literal(node: ast.AST, known: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in known:
        return known[node.id]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_literal(item, known) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.Set):
            return set(values)
        return values
    if isinstance(node, ast.Dict):
        return {
            _literal(key, known): _literal(value, known)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand, known)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _literal(node.left, known)
        right = _literal(node.right, known)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    raise ValueError(ast.dump(node))


def _top_level_literals(tree: ast.Module) -> dict[str, Any]:
    known: dict[str, Any] = {}
    for statement in tree.body:
        name = None
        value = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            name = statement.target.id
            value = statement.value
        if name is None or value is None:
            continue
        try:
            known[name] = _literal(value, known)
        except (KeyError, TypeError, ValueError):
            pass
    return known


def _one_of(literals: dict[str, Any], *names: str) -> Any:
    matches = [(name, literals[name]) for name in names if name in literals]
    assert len(matches) == 1, f"expected exactly one of {names}, observed {matches}"
    return matches[0][1]


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.AST = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _assignment_name(statement: ast.stmt) -> str | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id
    return None


def _update_core_ast_sha256(tree: ast.Module) -> str:
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    names = [_assignment_name(statement) for statement in main.body]
    start = names.index("recipe_capture")
    end = names.index("update_seconds")
    projection = ast.Module(body=main.body[start : end + 1], type_ignores=[])
    return hashlib.sha256(ast.dump(projection).encode("utf-8")).hexdigest()


def _rollout_arm(call: ast.Call) -> str | None:
    name = _call_name(call).lower()
    if "rollout" not in name and "closed_loop" not in name:
        return None
    for keyword in call.keywords:
        if keyword.arg == "arm" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value if isinstance(keyword.value.value, str) else None
    for argument in call.args:
        if isinstance(argument, ast.Constant) and argument.value in {"pre", "post"}:
            return argument.value
    return None


def test_t16_simulator_config_is_exactly_frozen() -> None:
    assert SIM_CONFIG.read_text(encoding="utf-8") == EXPECTED_SIM_CONFIG
    assert _sha256(SIM_CONFIG) == (
        "29b385e0ec34a664c42635c3b94e817aaa711e948a83b980410fefd3be6f2ec3"
    )


def test_t16_worker_pins_identity_and_a_two_arm_32_step_protocol() -> None:
    source, tree = _source_and_tree(WORKER)
    literals = _top_level_literals(tree)
    assert literals["PROTOCOL_SCHEMA"] == "sana-wam-libero-t16-sim-worker-v1"
    assert literals["PINNED_LIBERO_COMMIT"] == (
        "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    )
    assert literals["PINNED_INTERPRETER_SHA256"] == (
        "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
    )
    assert literals["PINNED_BDDL_SHA256"] == (
        "3d4ccf070c3d9883ae0676f2d888588f98c696ddad71b7694f47c379fc99ef36"
    )
    assert literals["PINNED_INIT_SHA256"] == (
        "0627f5f5ce3ef23be546571012be8ef603d93bcb4032bc80feb34937ba580140"
    )
    assert literals["PINNED_INIT_INDEX"] == 17
    assert literals["PINNED_INIT_COUNT"] == 50
    assert literals["PINNED_ENV_SEED"] == 0
    assert literals["PINNED_SETTLE_STEPS"] == 5
    assert literals["PINNED_MAX_POLICY_STEPS"] == 32
    assert literals["_ARM_ORDER"] == ("pre", "post")
    assert literals["PINNED_MODEL_PROMPT"] == (
        "pick up the black bowl on the cookie box and place it on the plate"
    )
    assert literals["PINNED_BDDL_LANGUAGE"] == (
        "pick the akita black bowl on the cookies box and place it on the plate"
    )

    # One persistent stdin loop must carry both resets.  The worker itself
    # enforces byte-identical reset observations and simulator state.
    for fragment in (
        "for raw_line in sys.stdin:",
        'if command == "reset":',
        'elif command == "step":',
        'elif command == "close":',
        "if next_arm_index >= len(_ARM_ORDER):",
        "elif identity != reset_identity:",
        "if not 0 <= expected_step < PINNED_MAX_POLICY_STEPS:",
        'parser.add_argument("--expected-config-sha256", required=True)',
        'parser.add_argument("--expected-worker-sha256", required=True)',
        "success = bool(env.check_success())",
        "terminal = bool(done) or success",
    ):
        assert fragment in source
    assert source.index("np.random.seed(PINNED_ENV_SEED)") < source.index(
        "env = OffScreenRenderEnv("
    )
    assert literals["PINNED_VERSIONS"]["Pillow"] == "12.2.0"

    calls = [_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(name.endswith(".clip") or name == "clip" for name in calls)
    assert "policy_action_to_libero" in calls
    assert "torch.save" not in calls


def test_t16_runner_pins_worker_config_and_direct_t15_predecessor() -> None:
    source, tree = _source_and_tree(RUNNER)
    literals = _top_level_literals(tree)
    project_root_path_insert = "sys.path.insert(0, str(ROOT))"
    assert project_root_path_insert in source
    assert source.index(project_root_path_insert) < source.index(
        "from smoke_libero_ar_real_gpu import"
    )
    assert literals["T15_PREDECESSOR_SOURCE_COMMIT"] == T15_SOURCE_COMMIT
    assert literals["T15_PREDECESSOR_RUNNER_SHA256"] == T15_RUNNER_SHA256
    assert literals["T15_PREDECESSOR_RESULT_SHA256"] == T15_RESULT_SHA256
    assert T15_ROOT in source
    assert "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_ARM_VALID" in source
    assert "T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED" in source
    assert _one_of(
        literals, "T16_SIM_WORKER_SHA256", "T16_SIMULATOR_WORKER_SHA256"
    ) == _sha256(WORKER)
    assert _one_of(
        literals, "T16_SIM_CONFIG_SHA256", "T16_SIMULATOR_CONFIG_SHA256"
    ) == _sha256(SIM_CONFIG)
    assert "/DATA/share/sana_wam_libero_nonformal_screens/t16" in source
    assert "libero-t16-paired-one-task-chunk-closed-loop-fixed32" in source
    assert 'external_assets["t15_predecessor/RESULT.json"]' in source
    assert 'external_assets["t15_predecessor/runner.py"]' in source


def test_t16_pins_physical_egl_identity_and_fail_closed_worker_startup() -> None:
    runner_source, _runner_tree = _source_and_tree(RUNNER)
    worker_source, _worker_tree = _source_and_tree(WORKER)
    for fragment in (
        '"CUDA_VISIBLE_DEVICES": str(render_gpu_device_id)',
        '"MUJOCO_EGL_DEVICE_ID": str(render_gpu_device_id)',
        "args.render_gpu_device_id = args.physical_gpu",
        "if args.render_gpu_device_id != args.physical_gpu:",
        "except BaseException:\n            self.abort()",
    ):
        assert fragment in runner_source
    for fragment in (
        'os.environ.get("CUDA_VISIBLE_DEVICES") != expected_render_device',
        'os.environ.get("MUJOCO_EGL_DEVICE_ID") != expected_render_device',
        '"render_gpu_device_id": args.render_gpu_device_id',
    ):
        assert fragment in worker_source


def test_t16_runner_freezes_training_aligned_step29_generation_contract() -> None:
    source, tree = _source_and_tree(RUNNER)
    literals = _top_level_literals(tree)
    max_steps = _one_of(literals, "T16_MAX_POLICY_STEPS", "T16_POLICY_STEPS_PER_ARM")
    action_tokens = _one_of(
        literals,
        "T16_ACTION_TOKENS_PER_CHUNK",
        "T16_AR_ACTION_TOKENS_PER_CHUNK",
    )
    generation_steps = _one_of(
        literals,
        "T16_EXPECTED_FULL_ARM_GENERATION_STEPS_ONE_BASED",
        "T16_FULL_ARM_GENERATION_STEPS_ONE_BASED",
    )
    assert max_steps == 32
    assert action_tokens == 28
    assert generation_steps == (1, 29)
    assert (max_steps + action_tokens - 1) // action_tokens == 2

    # The main environment loop still sends every new observation to policy,
    # while the recorded `generated` markers prove model calls only at 1/29.
    for fragment in (
        "ar_action_tokens_per_chunk",
        "rolling_buffer",
        "cache_feedback",
        "predicted",
        "generation_steps",
        "policy_requests",
        "environment_steps",
    ):
        assert fragment in source
    assert "execute_horizon=1" not in source
    assert "ar_action_tokens_per_chunk=1" not in source


def test_t16_exactly_preserves_the_frozen_t15_update_core_ast() -> None:
    _source, runner_tree = _source_and_tree(RUNNER)
    _t15_source, t15_tree = _source_and_tree(T15_RUNNER)
    expected = "357c111513e0b96e30297189f9d47bdef6a0b2486bd92b32d7e244b2f4799429"
    assert _update_core_ast_sha256(t15_tree) == expected
    assert _update_core_ast_sha256(runner_tree) == expected


def test_t16_pre_rollout_scheduler_restore_update_and_post_rollout_are_ordered() -> (
    None
):
    source, tree = _source_and_tree(RUNNER)
    main_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(main_nodes) == 1
    main = main_nodes[0]
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    pre_calls = [node for node in calls if _rollout_arm(node) == "pre"]
    post_calls = [node for node in calls if _rollout_arm(node) == "post"]
    scheduler_calls = [
        node
        for node in calls
        if _call_name(node).endswith("architecture.init_training_schedulers")
    ]
    training_mode_calls = [
        node
        for node in calls
        if _call_name(node).endswith("trainer._set_training_mode")
    ]
    assert len(pre_calls) == len(post_calls) == 1
    assert len(scheduler_calls) == len(training_mode_calls) == 1
    scheduler = scheduler_calls[0]
    assert len(scheduler.args) == 1
    assert isinstance(scheduler.args[0], ast.Constant)
    assert scheduler.args[0].value == 1000

    macro_loops = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "step_index"
        and isinstance(node.iter, ast.Call)
        and _call_name(node.iter) == "range"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "T15_UPDATE_STEPS"
    ]
    assert len(macro_loops) == 1
    assert (
        pre_calls[0].lineno
        < scheduler.lineno
        < training_mode_calls[0].lineno
        < macro_loops[0].lineno
        < post_calls[0].lineno
    )
    assert source.count("subprocess.Popen(") == 1
    assert "ARInferenceEngine(" in source
    assert "LiberoPolicyServer(" in source
    assert "video_scheduler" in source and "action_scheduler" in source


def test_t16_never_loads_or_saves_a_sana_wam_checkpoint() -> None:
    source, tree = _source_and_tree(RUNNER)
    calls = [_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden_call_suffixes = (
        "torch.load",
        "torch.save",
        "load_checkpoint",
        "save_checkpoint",
        "load_from_checkpoint_dir",
        "load_libero_from_checkpoint_dir",
        "build_libero_server_from_config",
    )
    assert not any(
        name == suffix or name.endswith(f".{suffix}")
        for name in calls
        for suffix in forbidden_call_suffixes
    )
    for field in (
        "training.init_checkpoint",
        "training.resume_checkpoint",
        "training.resume_from_checkpoint",
        "training.resume_manifest",
        "model.video_backbone.init_dit_from",
    ):
        assert f'"{field}"' in source
    assert '"sana_wam_training_checkpoint_loaded": False' in source
    assert '"sana_wam_training_checkpoint_saved": False' in source
    assert '"formal_training_executed": False' in source
    assert '"benchmark_evaluation_executed": False' in source
    assert 'os.environ.get("SANA_WAM_CHECKPOINT_SHA256")' in source


def test_t16_result_keeps_early_terminal_and_diagnostics_semantics_explicit() -> None:
    source, literals_tree = _source_and_tree(RUNNER)
    literals = _top_level_literals(literals_tree)
    assert literals["T16_EXECUTION_VERDICT"] == (
        "T16_PAIRED_ONE_TASK_CHUNK_CLOSED_LOOP_INTERFACE_VALID"
    )
    for fragment in (
        "execution_verdict = T16_EXECUTION_VERDICT",
        "scientific_verdict = T16_EARLY_TERMINAL_VERDICT",
        '"generation_noise_shared_generation_count": shared_generation_count',
        '"generation_noise_signatures_match": generation_noise_signatures_match',
        '"success_classification": success_classification',
        '"trajectory_diagnostics": trajectory_diagnostics',
        'json.dumps(\n            report,\n            allow_nan=False,\n            sort_keys=True,\n            separators=(",", ":"),',
        "_freeze_run_root(run_root)",
    ):
        assert fragment in source
