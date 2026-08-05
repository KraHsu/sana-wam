"""Focused contracts for the AV-1B vendor-GDN bridge.

The unmarked tests are static or CPU construction-only: they do not execute a
vendor forward, initialize CUDA, create a result root, read real data, or load
a checkpoint.  The single GPU smoke is opt-in via
``CACH_AV1B_AUTHORIZED_GPU_TEST=1`` and is valid only under the exact frozen
card environment and reserved UUID.  It runs no optimizer loop and writes no
checkpoint.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = REPO_ROOT / "docs/cach_sana_wam/architecture_validation/av1b/AV1B_RUN_CARD.json"
CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_av1b_vendor_gdn_bridge.yaml"
BRIDGE_PATH = REPO_ROOT / "src/sana_wam/model/cach_av1b_vendor_gdn_bridge.py"
CARD_SHA256 = "c0e15bcb8c20441d7d989ec6d375284e45f24962c98e15b984dbbbbd198248fb"
ROOT = (
    "/DATA/share/sana_cach_wam_nonformal_screens/av1b/06f5d09127f8/"
    "av1b-1a9a3a6ca4b18fac904fdb806c2a8fb3"
)
NONCE = "1a9a3a6ca4b18fac904fdb806c2a8fb3"
GPU_UUID = "GPU-41c95a43-ce96-fff3-33e0-739a3931d603"
TASK_RECIPE_SHA256 = "9d2a749d722775324ee233862772def530fbffe0ea9e55aeea87c160b87a51e4"
SHUFFLE_PERMUTATION = (1, 0, 3, 2, 5, 4, 7, 6)

EXPECTED_TOP_LEVEL_CONFIG_KEYS = {
    "schema",
    "identity",
    "authority",
    "classification",
    "runtime",
    "topology",
    "recipe",
    "initialization",
    "optimizer",
    "budget",
    "diagnostics",
    "metrics",
    "thresholds",
    "artifacts",
}
EXPECTED_CAPABILITIES = {
    "code_write": True,
    "root_create": True,
    "single_gpu": True,
    "cuda": True,
    "triton_jit": True,
    "vendor_autograd_kernel": True,
    "model_execute": True,
    "forward_backward": True,
    "jvp": True,
    "optimizer": True,
    "parameter_update": True,
    "synthetic_nonformal_training": True,
    "lightweight_tests": True,
    "diagnostic_metrics": True,
    "real_data": False,
    "checkpoint_load": False,
    "checkpoint_save": False,
    "review_token_consume": False,
    "full_2b": False,
    "formal_evaluation": False,
    "admission": False,
    "deploy_capture": False,
    "av2": False,
    "global_stage3": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_card() -> dict[str, Any]:
    return json.loads(CARD_PATH.read_text(encoding="utf-8"), object_pairs_hook=_strict_object_pairs)


def _load_config() -> dict[str, Any]:
    yaml = importlib.import_module("yaml")

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return yaml.load(CONFIG_PATH.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)


def _load_bridge():
    return importlib.import_module("sana_wam.model.cach_av1b_vendor_gdn_bridge")


def _tensor_leaves(value: Any, torch) -> Iterator[Any]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if is_dataclass(value):
        for field in fields(value):
            yield from _tensor_leaves(getattr(value, field.name), torch)
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _tensor_leaves(value[key], torch)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item, torch)


def _vendor_modules(arm) -> tuple[Any, ...]:
    return tuple(
        module
        for module in arm.modules()
        if type(module).__module__ == "diffusion.model.nets.sana_gdn_blocks_triton"
        and type(module).__name__ == "ChunkCausalGDNTriton"
    )


def _prediction(output):
    value = getattr(output, "video_prediction", None)
    assert value is not None, "AV1BVendorSequenceOutput.video_prediction is required"
    return value


def _diagnostic_rms(value: Any) -> float:
    for name in ("rms", "jvp_rms"):
        if hasattr(value, name):
            result = float(getattr(value, name))
            assert math.isfinite(result)
            return result
    raise AssertionError("AV1BLocalJVPDiagnostic must expose rms or jvp_rms")


def _forbidden(*_args, **_kwargs):
    raise AssertionError("forbidden proxy/data/checkpoint/CUDA fallback was used")


def test_frozen_card_config_authority_source_and_scope_contract() -> None:
    assert _sha256(CARD_PATH) == CARD_SHA256
    card = _load_card()
    config = _load_config()

    assert card["schema"] == "cach.architecture_validation.av1b_run_card.v1"
    assert card["card_id"] == "cach-av1b-vendor-gdn-single-call-v1"
    assert card["classification"]["operator_level"] == "VENDOR_KERNEL"
    assert card["classification"]["integration_path"] == "EXPERIMENTAL_PATH"
    assert card["classification"]["proxy_or_reference_operator_fallback"] == "FORBIDDEN"
    assert card["classification"]["explicitly_not_assessed"] == [
        "CROSS_CALL_INIT_STATE_OR_FINAL_STATE_IO",
        "TWO_CALL_DIFFERENTIABLE_SCRATCH_STATE_CONTINUATION",
        "LIVE_CACHE_COMMIT_OR_CONTENT_TIME",
        "PUBLIC_WRAPPER_DISPATCHER_OR_OWNER_PARITY",
        "STATEFUL_CHUNKWISE_FORWARD_BACKWARD_QUALIFICATION",
        "REAL_DATA_OR_CLOSED_LOOP_CAPABILITY",
        "FULL_2B",
    ]

    assert set(config) == EXPECTED_TOP_LEVEL_CONFIG_KEYS
    assert config["schema"] == "cach.av1b.vendor_gdn_bridge.config.v1"
    identity = config["identity"]
    assert identity["run_card_path"] == card["identity"]["self_path"]
    assert identity["run_card_sha256"] == CARD_SHA256
    assert identity["resolved_root"] == ROOT
    assert identity["run_nonce"] == NONCE
    assert identity["gpu_physical_index"] == 7
    assert identity["gpu_uuid"] == GPU_UUID

    authority = config["authority"]
    assert (
        hashlib.sha256(
            authority["execution_statement"].encode("utf-8")
        ).hexdigest()
        == authority["execution_statement_sha256"]
        == "d96e7681321ef58d1450632d6816be95733605fe392dc9781205ac7e4d565dd4"
    )
    assert (
        hashlib.sha256(
            authority["confirmation_statement"].encode("utf-8")
        ).hexdigest()
        == authority["confirmation_statement_sha256"]
        == "6ba9eaf13617f87b87c09b0f3ca64b72403c84188011e0e9185f304c3365a654"
    )
    assert authority["effective_interpretation"] == (
        "CLARIFICATION_CONTROLS_AND_PERMITS_CARD_LOCAL_JVP_AND_TWO_ARM_200_STEP_SYNTHETIC_TRAINING"
    )
    assert authority["review_token_state"] == "UNCONSUMED"
    assert authority["capabilities"] == EXPECTED_CAPABILITIES

    runtime = config["runtime"]
    expected_env = runtime["environment_before_any_torch_or_vendor_import"]
    assert expected_env == card["launcher_contract"]["environment_must_be_set_before_any_torch_or_vendor_import"]
    assert expected_env["CUDA_VISIBLE_DEVICES"] == GPU_UUID
    assert expected_env["GDN_DISABLE_COMPILE"] == "1"
    assert expected_env["TORCHDYNAMO_DISABLE"] == "1"
    assert expected_env["FUSED_GDN_PRECISION"] == "0"
    assert expected_env["TRITON_CACHE_DIR"] == f"{ROOT}/triton_cache"
    assert runtime["allow_real_data"] is False
    assert runtime["allow_checkpoint"] is False
    assert runtime["allow_proxy_or_reference_fallback"] is False

    assert config["topology"]["HW"] == [5, 1, 1]
    assert config["topology"]["chunk_boundaries"] == [0, 3, 5]
    assert config["topology"]["vendor_use_autograd_kernel"] is True
    assert config["topology"]["persistent_state_input_or_output"] is False
    assert config["recipe"]["task_recipe_sha256"] == TASK_RECIPE_SHA256
    assert tuple(config["recipe"]["shuffle_permutation"]) == SHUFFLE_PERMUTATION
    assert config["optimizer"]["optimizer_steps_per_arm"] == 200
    assert config["budget"]["max_optimizer_steps_total"] == 400
    assert config["artifacts"]["write_model_checkpoint"] is False
    assert config["artifacts"]["write_optimizer_state"] is False

    additive = card["additive_policy"]["exact_additive_paths"]
    generated = [entry["path"] for entry in identity["generated_paths"].values()]
    assert additive == [card["identity"]["self_path"], *generated]
    for pin in identity["source_pins"].values():
        path = pin.get("path")
        digest = pin.get("sha256")
        if path is not None:
            assert _sha256(REPO_ROOT / path) == digest, path


def test_bridge_source_is_vendor_only_and_exposes_card_api() -> None:
    source = BRIDGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BRIDGE_PATH))
    imported_from: dict[str, set[str]] = {}
    imported_modules: set[str] = set()
    function_names: set[str] = set()
    class_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_modules.add(module)
            imported_from.setdefault(module, set()).update(alias.name for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            class_names.add(node.name)

    assert "diffusion.model.nets.sana_gdn_blocks_triton" in imported_modules
    assert "ChunkCausalGDNTriton" in imported_from[
        "diffusion.model.nets.sana_gdn_blocks_triton"
    ]
    assert {
        "build_av1b_vendor_task",
        "build_av1b_vendor_gdn_pair",
        "pure_torch_last_block_tail",
        "last_block_local_seam_parameter_jvp",
        "last_block_local_action_condition_jvp",
    } <= function_names
    assert {
        "AV1BVendorGDNBridgeSpec",
        "AV1BVendorTask",
        "AV1BVendorTaskChunk",
        "AV1BVendorGDNBridgeArm",
        "AV1BVendorGDNBridgePair",
        "AV1BVendorSequenceOutput",
        "AV1BLocalJVPDiagnostic",
    } <= class_names

    forbidden_modules = {
        "sana_wam.model.cach_minimal_cpu",
        "sana_wam.model.cach_production_path_mini",
        "diffusion.model.ops.fused_gdn_chunkwise",
        "diffusion.model.nets.sana_v2v_attn_blocks",
    }
    assert imported_modules.isdisjoint(forbidden_modules)
    proxy_imports = imported_from.get(
        "sana_wam.model.cach_av1_transition_proxy",
        set(),
    )
    assert proxy_imports.isdisjoint(
        {"AV1TransitionProxyArm", "AV1TransitionProxyPair", "build_av1_pair", "run_av1_screen"}
    )
    assert "AV1B_CARD_SHA256" in source
    assert "AV1B_TASK_RECIPE_SHA256" in source
    assert "AV1B_CHUNK_BOUNDARIES" in source


def test_cpu_task_and_model_construction_are_fresh_vendor_only_and_cuda_free(
    monkeypatch,
) -> None:
    assert os.environ.get("GDN_DISABLE_COMPILE") == "1"
    assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    assert os.environ.get("FUSED_GDN_PRECISION") == "0"
    torch = importlib.import_module("torch")
    assert not torch.cuda.is_initialized()
    bridge = _load_bridge()
    av1 = importlib.import_module("sana_wam.model.cach_av1_transition_proxy")
    monkeypatch.setattr(av1, "build_av1_pair", _forbidden)
    monkeypatch.setattr(av1, "run_av1_screen", _forbidden)
    monkeypatch.setattr(torch, "load", _forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbidden)

    spec = bridge.AV1BVendorGDNBridgeSpec()
    assert spec.batch_size == 8
    assert spec.hidden_dim == 64
    assert spec.heads == 2
    assert spec.head_dim == 32
    assert spec.depth == 20
    assert tuple(spec.chunk_boundaries) == (0, 3, 5)
    assert spec.use_autograd_kernel is True
    assert spec.conv_kernel_size == 0

    # The AV-1B public task/pair builders are deliberately frozen to cuda:0.
    # CPU coverage therefore validates the pinned source task bytes and the
    # exact CPU common-template construction used internally by the pair
    # builder, without executing a vendor forward.
    source_spec = av1.AV1TransitionProxySpec()
    task = av1.build_av1_synthetic_task(spec=source_spec)
    assert task.recipe_digest == TASK_RECIPE_SHA256
    tensors = tuple(_tensor_leaves(task, torch))
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(bool(torch.isfinite(tensor).all()) for tensor in tensors if tensor.is_floating_point())

    with pytest.raises(ValueError, match="cuda:0"):
        bridge.build_av1b_vendor_task(spec, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="cuda:0"):
        bridge.build_av1b_vendor_gdn_pair(spec, device=torch.device("cpu"))

    reference = bridge.AV1BVendorGDNBridgeArm(
        spec=spec,
        staging_variant=bridge.CACHStagingVariant.REF_GDN_CORRECTED,
    )
    candidate = copy.deepcopy(reference)
    candidate.enable_candidate_action_seam()
    reference.eval()
    candidate.eval()
    assert reference is not candidate
    reference_vendor = _vendor_modules(reference)
    candidate_vendor = _vendor_modules(candidate)
    assert len(reference_vendor) == len(candidate_vendor) == 20
    assert all(module.use_autograd_kernel is True for module in (*reference_vendor, *candidate_vendor))

    for arm in (reference, candidate):
        for tensor in tuple(arm.parameters()) + tuple(arm.buffers()):
            assert tensor.device.type == "cpu"
        assert not arm.training

    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    reference_action_names = {
        name
        for name in reference_parameters
        if "action_conditioner" in name or "action_output_projection" in name
    }
    assert not reference_action_names
    seam_names = sorted(
        name
        for name in candidate_parameters
        if name.endswith("action_output_projection.weight")
        or name.endswith("action_output_projection.bias")
    )
    assert len(seam_names) == 40
    assert all(not bool(candidate_parameters[name].detach().count_nonzero()) for name in seam_names)

    common_names = set(reference_parameters) & set(candidate_parameters)
    assert common_names
    for name in sorted(common_names):
        left = reference_parameters[name].detach()
        right = candidate_parameters[name].detach()
        assert left.dtype == right.dtype
        assert tuple(left.shape) == tuple(right.shape)
        assert torch.equal(left, right), name
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()

    candidate_tensors = {
        **candidate_parameters,
        **dict(candidate.named_buffers()),
    }
    no_action_names = [name for name in candidate_tensors if name.endswith("no_action_slot")]
    assert len(no_action_names) == 1
    no_action = candidate_tensors[no_action_names[0]]
    assert not no_action.requires_grad
    assert not bool(no_action.detach().count_nonzero())
    assert not torch.cuda.is_initialized()


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("CACH_AV1B_AUTHORIZED_GPU_TEST") != "1",
    reason="requires the exact authorized AV-1B reserved-GPU environment",
)
def test_authorized_vendor_forward_backward_and_local_jvp_smoke(monkeypatch) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    assert os.environ.get("GDN_DISABLE_COMPILE") == "1"
    assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    assert os.environ.get("FUSED_GDN_PRECISION") == "0"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    assert os.environ.get("TRITON_CACHE_DIR") == f"{ROOT}/triton_cache"

    torch = importlib.import_module("torch")
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert "H200" in torch.cuda.get_device_name(0)
    bridge = _load_bridge()
    av1 = importlib.import_module("sana_wam.model.cach_av1_transition_proxy")
    monkeypatch.setattr(av1, "build_av1_pair", _forbidden)
    monkeypatch.setattr(av1, "run_av1_screen", _forbidden)
    monkeypatch.setattr(torch, "load", _forbidden)

    device = torch.device("cuda:0")
    spec = bridge.AV1BVendorGDNBridgeSpec()
    task = bridge.build_av1b_vendor_task(spec, device=device)
    pair = bridge.build_av1b_vendor_gdn_pair(spec, device=device)
    reference_vendor = _vendor_modules(pair.reference)
    candidate_vendor = _vendor_modules(pair.candidate)
    assert len(reference_vendor) == len(candidate_vendor) == 20
    assert all(module.use_autograd_kernel is True for module in (*reference_vendor, *candidate_vendor))

    vendor_calls: list[int] = []
    hooks = [
        module.register_forward_hook(
            lambda module, _inputs, _output: vendor_calls.append(id(module))
        )
        for module in candidate_vendor
    ]
    try:
        candidate_output = pair.candidate(
            task,
            mode="correct",
            target_override=None,
            capture_diagnostics=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert len(vendor_calls) == 20
    assert len(set(vendor_calls)) == 20

    reference_output = pair.reference(
        task,
        mode="correct",
        target_override=None,
        capture_diagnostics=True,
    )
    candidate_prediction = _prediction(candidate_output)
    reference_prediction = _prediction(reference_output)
    assert candidate_prediction.device.type == "cuda"
    assert candidate_prediction.dtype is torch.float32
    assert tuple(candidate_prediction.shape) == (8, 5, 3, 1, 1)
    assert bool(torch.isfinite(candidate_prediction).all())
    assert torch.equal(candidate_prediction, reference_prediction)

    candidate_shuffle = pair.candidate(
        task,
        mode="shuffle",
        target_override=None,
        capture_diagnostics=False,
    )
    candidate_no_action = pair.candidate(
        task,
        mode="no_action",
        target_override=None,
        capture_diagnostics=False,
    )
    assert torch.equal(_prediction(candidate_shuffle), candidate_prediction)
    assert torch.equal(_prediction(candidate_no_action), candidate_prediction)

    target = task.video_target.detach().clone().requires_grad_(True)
    target_output = pair.candidate(
        task,
        mode="correct",
        target_override=target,
        capture_diagnostics=False,
    )
    assert torch.equal(_prediction(target_output), candidate_prediction)
    target_gradient = torch.autograd.grad(
        _prediction(target_output).square().sum(),
        target,
        allow_unused=True,
    )[0]
    assert target_gradient is None or not bool(target_gradient.count_nonzero())

    parameters = dict(pair.candidate.named_parameters())
    seam_names = tuple(
        sorted(
            name
            for name, value in parameters.items()
            if value.requires_grad and name.endswith("action_output_projection.weight")
        )
    )
    assert len(seam_names) == 20
    vendor_parameters: list[Any] = []
    seen: set[int] = set()
    for module in candidate_vendor:
        for value in module.parameters():
            if value.requires_grad and id(value) not in seen:
                seen.add(id(value))
                vendor_parameters.append(value)
    assert vendor_parameters

    loss = (candidate_prediction[:, 1:] - task.video_target[:, 1:]).square().mean()
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters[name] for name in seam_names) + tuple(vendor_parameters),
        retain_graph=True,
        allow_unused=True,
    )
    seam_gradients = gradients[: len(seam_names)]
    vendor_gradients = gradients[len(seam_names) :]
    assert all(gradient is not None for gradient in seam_gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in seam_gradients)
    seam_rms = math.sqrt(
        sum(gradient.double().square().sum().item() for gradient in seam_gradients)
        / sum(gradient.numel() for gradient in seam_gradients)
    )
    assert seam_rms > 1.0e-8
    finite_vendor = [
        gradient
        for gradient in vendor_gradients
        if gradient is not None and bool(torch.isfinite(gradient).all())
    ]
    assert finite_vendor
    assert any(bool(gradient.count_nonzero()) for gradient in finite_vendor)

    jvp = bridge.last_block_local_seam_parameter_jvp(
        pair.candidate,
        candidate_output,
    )
    assert _diagnostic_rms(jvp) > 1.0e-8
    finite_flag = getattr(jvp, "finite", True)
    assert bool(finite_flag)
    assert getattr(jvp, "primal_digest", None)

    assert not hasattr(candidate_output, "final_state")
    assert not hasattr(candidate_output, "committed_state")
    assert not hasattr(candidate_output, "cache_commit")
