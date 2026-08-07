"""Minimal torchrun DDP + bf16 trainer for the SANA block-AR model.

The real train step is two calls — ``architecture.prepare_inputs(batch)`` (VAE-encode +
collate) then ``architecture.compute_loss(**inputs)`` (the AR per-chunk loss);
everything else here is freeze setup, optimizer groups, the loop, and writing
the deploy checkpoint contract.

Distributed: plain ``torchrun``. Because the entry point is ``compute_loss``
(not ``forward``), we cannot rely on DDP's forward hook — instead each rank runs
the full step and we **manually all-reduce trainable gradients** (no-op when
world_size==1). Frozen modules (VAE, text encoder) stay on the no-grad path.
"""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import stat
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from sana_wam.config import flatten_model_cfg
from sana_wam.dataloader.mixture import build_training_dataset
from sana_wam.model import build_architecture
from sana_wam.train.checkpointing import (
    manage_checkpoints,
    save_action_stats,
    save_config,
)
from sana_wam.train.phase6_recovery import recovery_lifecycle

logger = logging.getLogger(__name__)

_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX = "action_video_memory_adapter."
_ACTION_VIDEO_MEMORY_IDENTITY_ZERO_PARAMETERS = (
    "k_right",
    "v_right",
    "z_logits",
)


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


class Trainer:
    def __init__(
        self,
        cfg,
        *,
        launch_context=None,
        post_primary_afcc_authorized: bool = False,
        exact_output_dir: str | os.PathLike[str] | None = None,
    ):
        # This is the direct-construction guard.  It must remain before CUDA,
        # datasets, model construction, checkpoint selection, and root writes.
        from sana_wam.cach.authority import reject_cach_training_before_runtime

        reject_cach_training_before_runtime(cfg, launch_context)
        self.cfg = cfg
        self.t = cfg.training
        if type(post_primary_afcc_authorized) is not bool:
            raise TypeError("post_primary_afcc_authorized must be boolean")
        self._post_primary_afcc_authorized = post_primary_afcc_authorized
        self._phase6_launch_context = self._coerce_phase6_launch_context(launch_context)
        self._exact_output_dir = self._validate_exact_output_dir(exact_output_dir)
        self._phase6_preflight_report = None
        self._phase6_preflight_report_bytes = None
        self._phase6_preflight_report_sha256 = None
        self._run_phase6_preflight_early()
        self._validate_phase6_launch_context_early()
        self._validate_world_size_contract()
        self.lambda_video = float(self.t.get("lambda_video", 1.0))
        self.lambda_action = float(self.t.get("lambda_action", 1.0))

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        if (
            self._phase6_launch_context is not None
            or self._post_primary_afcc_authorized
            or self._exact_output_dir is not None
        ) and self.device.type != "cuda":
            raise RuntimeError("governed training requires one CUDA device")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        self.base_seed = int(self.t.get("seed", 0))
        self.process_seed = self._seed_process(self.base_seed, _rank())
        self._phase6_plan = None
        self._phase6_plan_sampler = None
        self._phase6_dataset_contract = None
        self._phase1_action_reference_table = None
        self._phase1_action_reference_enabled_config = False
        self._action_facing_cache_reference_store = None
        self._action_facing_cache_consistency_enabled_config = False
        self._verified_phase1_reference_checkpoint_path = None
        self._verified_action_stats_path = None
        self._verified_action_stats_sha256 = None

        # --- dataset ---
        self.dataset = build_training_dataset(cfg.dataloader, split="train")
        self._verify_action_stats()
        self._configure_phase6_plan()
        self._configure_phase1_action_reference()
        self._configure_action_facing_cache_reference()

        # Persist the dataset's RESOLVED num_frames back into the config so the saved
        # checkpoint is self-contained. growing_history auto-sets num_frames =
        # raw_window_len at dataset-build time (spanning the longest episode); the
        # deploy engine needs that exact value to reconstruct the chunk geometry
        # (action_tokens_per_chunk / clip window). Without this write-back the deploy
        # config has no num_frames → falls back to a default (81) → wrong atc →
        # action/video misalignment at rollout. Value-agnostic: persists whatever the
        # dataset resolved, for any task/stride.
        try:
            _subs = getattr(self.dataset, "_sub_datasets", None)
            _resolved_nf = (
                getattr(_subs[0], "num_frames", None)
                if _subs
                else getattr(self.dataset, "num_frames", None)
            )
            if _resolved_nf and getattr(cfg, "dataloader", None) is not None:
                cfg.dataloader.num_frames = int(_resolved_nf)
        except Exception as _e:  # never block training on a config-sync hiccup
            logging.getLogger(__name__).warning("num_frames write-back skipped: %s", _e)

        # --- architecture (dispatched by architecture.variant) ---
        # Every distributed rank must start from byte-identical parameters.
        # Dataset sampling/noise is rank-specific, but model construction is not.
        if _world_size() > 1:
            self._seed_process(int(self.t.get("initialization_seed", self.base_seed)), 0)
        self.architecture = build_architecture(flatten_model_cfg(cfg.model))
        self.architecture.set_dtype_device(torch.bfloat16, self.device)
        self._load_initial_checkpoint()
        self.architecture.init_training_schedulers(1000)
        self.architecture.set_training_runtime(
            use_gradient_checkpointing=bool(
                self.t.get("use_gradient_checkpointing", False)
            ),
        )

        # --- freeze / exact trainable-module allowlist ---
        self._trainable_module_allowlist: tuple[str, ...] | None = None
        self._trainable_parameter_patterns: tuple[str, ...] | None = None
        self._frozen_module_paths: tuple[str, ...] = ()
        self._frozen_input_grad_module_paths: tuple[str, ...] = ()
        self._eval_module_paths: tuple[str, ...] = self._resolve_eval_modules()
        frozen = self._configure_trainable_modules()
        promoted = self._configure_trainable_parameter_dtype()
        if _rank() == 0:
            logger.info("Frozen module roots: %s", frozen)
            if promoted:
                logger.info(
                    "Promoted %d trainable tensors to float32 optimizer precision",
                    promoted,
                )

        # --- action stats into buffers (deploy normalization parity) ---
        self._load_action_stats()
        if _world_size() > 1:
            self._broadcast_initial_model_state()
            self.process_seed = self._seed_process(self.base_seed, _rank())
        self._data_generator = torch.Generator().manual_seed(self.process_seed)

    @staticmethod
    def _coerce_phase6_launch_context(value):
        if value is None:
            return None
        from sana_wam.train.phase6_run_integrity import (
            Phase6ValidatedLaunchContext,
        )

        if isinstance(value, Phase6ValidatedLaunchContext):
            value = {
                "arm": value.arm,
                "authorization": value.authorization,
                "config_path": value.input_arm_config_path,
                "config_sha256": value.input_arm_config_sha256,
                "launch_manifest_path": value.launch_manifest_path,
                "launch_manifest_sha256": value.launch_manifest_sha256,
                "output_directory": value.run_directory,
                "run_id": value.run_id,
                "ticket_path": value.ticket_path,
                "ticket_sha256": value.ticket_sha256,
            }
        if not isinstance(value, Mapping):
            raise TypeError("launch_context must be a validated mapping or dataclass")
        return Phase6ValidatedLaunchContext.from_mapping(value)

    def _run_phase6_preflight_early(self) -> None:
        """Run the formal integrity gate before touching CUDA or expensive state."""

        field_names = (
            "phase6_preflight_request",
            "phase6_preflight_request_sha256",
            "phase6_preflight_report",
            "phase6_preflight_report_sha256",
        )
        configured = {
            name: self.t.get(name, None) not in (None, "") for name in field_names
        }
        if self._post_primary_afcc_authorized:
            if any(configured.values()):
                raise ValueError(
                    "post-primary AFCC must not reuse primary Phase-6 preflight fields"
                )
            if self.t.get("post_primary_afcc_condition", None) != "T1_E1A0_F1":
                raise ValueError("post-primary AFCC condition marker differs")
            return
        phase6_markers = (
            "phase6_arm",
            "phase6_registry",
            "phase6_plan_artifact",
            "phase6_dataset_contract_artifact",
        )
        is_phase6 = any(
            self.t.get(name, None) not in (None, "") for name in phase6_markers
        )
        if not is_phase6:
            if any(configured.values()):
                raise ValueError(
                    "Phase-6 preflight fields are forbidden for legacy training"
                )
            return
        if not all(configured.values()):
            missing = [name for name, present in configured.items() if not present]
            raise ValueError(
                "formal Phase-6 requires request/SHA/report preflight fields; "
                f"missing={missing}"
            )

        from sana_wam.train.phase6_preflight import (
            canonical_json_bytes,
            load_and_revalidate_phase6_preflight_report,
        )

        report = load_and_revalidate_phase6_preflight_report(
            self.t.get("phase6_preflight_request"),
            self.t.get("phase6_preflight_report"),
            expected_request_sha256=self._normalize_expected_sha256(
                self.t.get("phase6_preflight_request_sha256"),
                "phase6_preflight_request_sha256",
            ),
            expected_report_sha256=self._normalize_expected_sha256(
                self.t.get("phase6_preflight_report_sha256"),
                "phase6_preflight_report_sha256",
            ),
            expected_purpose="training",
        )
        if not isinstance(report, dict) or report.get("purpose") != "training":
            raise RuntimeError(
                "Phase-6 Trainer received a non-training preflight report"
            )
        report_bytes = canonical_json_bytes(report)
        report_path = Path(os.fspath(self.t.get("phase6_preflight_report"))).resolve()
        if report_path.read_bytes() != report_bytes:
            raise RuntimeError("persisted Phase-6 preflight report bytes differ")
        self._phase6_preflight_report = report
        self._phase6_preflight_report_bytes = report_bytes
        self._phase6_preflight_report_sha256 = hashlib.sha256(report_bytes).hexdigest()

    def _validate_phase6_launch_context_early(self) -> None:
        """Cross-check authorizer facts against preflight, raw YAML, and cfg."""

        context = self._phase6_launch_context
        is_phase6 = self._phase6_preflight_report is not None
        if self._post_primary_afcc_authorized:
            if context is not None or is_phase6:
                raise ValueError(
                    "post-primary AFCC cannot reuse a primary launch context"
                )
            return
        if not is_phase6:
            if context is not None:
                raise ValueError("launch_context is forbidden for legacy training")
            return
        if context is None:
            raise ValueError("formal Phase-6 requires a validated launch_context")
        context.revalidate_pinned_files()
        report = self._phase6_preflight_report
        if report.get("authorization") != context.authorization:
            raise RuntimeError(
                "launch authorization differs from the training preflight report"
            )
        for key, expected in (
            *recovery_lifecycle().items(),
            ("reference_precompute_completed", True),
            ("smoke_completed", True),
            ("registered_before_recovery_cohort_started", True),
        ):
            if (
                type(report.get(key)) is not type(expected)
                or report.get(key) != expected
            ):
                raise RuntimeError(f"training preflight lifecycle {key} differs")
        if report.get("request_sha256") != self._normalize_expected_sha256(
            self.t.get("phase6_preflight_request_sha256"),
            "phase6_preflight_request_sha256",
        ):
            raise RuntimeError("training preflight request SHA differs from config")
        input_config = report.get("input_config")
        if (
            not isinstance(input_config, Mapping)
            or input_config.get("sha256") != context.arm_config_projection_sha256
            or report.get("input_config_role") != "arm_scientific_projection_sidecar"
        ):
            raise RuntimeError(
                "training preflight scientific projection differs from launch"
            )

        from omegaconf import OmegaConf
        from sana_wam.train.phase6_arm_config import (
            build_phase6_arm_projection,
            canonical_json_bytes as arm_canonical_json_bytes,
            load_arm_config_bytes,
            validate_arm_config,
        )
        from sana_wam.train.phase6_run_integrity import read_stable_pinned_file

        raw_config = load_arm_config_bytes(
            context.read_input_arm_config_bytes(),
            source=context.input_arm_config_path,
        )
        if validate_arm_config(raw_config, verify_files=False) != context.arm:
            raise RuntimeError("authorized raw YAML resolves to a different arm")
        actual_config = OmegaConf.to_container(self.cfg, resolve=True)
        if type(actual_config) is not dict or actual_config != raw_config:
            raise RuntimeError(
                "loaded scientific config differs from the authorized raw YAML; "
                "CLI overrides are forbidden"
            )
        projection = build_phase6_arm_projection(context.arm)
        projection_sha = hashlib.sha256(
            arm_canonical_json_bytes(projection)
        ).hexdigest()
        if projection_sha != context.arm_config_projection_sha256:
            raise RuntimeError("actual config projection differs from authorization")
        training = actual_config["training"]
        expected_scalars = {
            "output_dir": context.run_directory,
            "phase6_arm": context.arm,
            "phase6_run_id": context.run_id,
            "phase6_scientific_projection_sha256": projection_sha,
        }
        for key, expected in expected_scalars.items():
            if training.get(key) != expected:
                raise RuntimeError(f"training.{key} differs from launch authorization")
        projection_path = training.get("phase6_scientific_projection")
        if not isinstance(projection_path, str):
            raise RuntimeError("training.phase6_scientific_projection is absent")
        projection_bytes = read_stable_pinned_file(
            projection_path,
            projection_sha,
            "scientific projection sidecar",
        )
        if projection_bytes != arm_canonical_json_bytes(projection):
            raise RuntimeError("scientific projection sidecar changed or differs")

    def _validate_world_size_contract(self) -> None:
        expected = self.t.get("expected_world_size", None)
        if expected is None:
            return
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
            raise ValueError(
                "training.expected_world_size must be a positive integer, "
                f"got {expected!r}"
            )
        actual = _world_size()
        if actual != expected:
            raise RuntimeError(
                "distributed world-size contract mismatch: "
                f"expected {expected}, got {actual}"
            )
        # Formal configs are immutable scientific inputs.  Legacy training keeps
        # the historical convenience field in its saved config.
        if (
            self._phase6_launch_context is None
            and not self._post_primary_afcc_authorized
        ):
            self.t["actual_world_size"] = actual
        if _rank() == 0:
            logger.info("Verified distributed world size: %d", actual)

    def _validate_exact_output_dir(
        self, exact_output_dir: str | os.PathLike[str] | None
    ) -> str | None:
        """Bind an explicitly pre-created, non-symlink LIBERO output root."""

        enabled = self.t.get("exact_output_dir", False)
        if type(enabled) is not bool:
            raise ValueError("training.exact_output_dir must be boolean")
        if not enabled:
            if exact_output_dir is not None:
                raise RuntimeError(
                    "exact_output_dir argument requires training.exact_output_dir=true"
                )
            return None
        if exact_output_dir is None:
            raise RuntimeError(
                "training.exact_output_dir=true requires the governed launcher argument"
            )
        if (
            self._phase6_launch_context is not None
            or self._post_primary_afcc_authorized
        ):
            raise RuntimeError(
                "formal LIBERO exact output cannot be combined with another authority"
            )
        if self.cfg.dataloader.get("type", None) != "libero":
            raise RuntimeError("exact output directory is restricted to LIBERO training")
        if self.t.get("formal_libero_training", None) is not True:
            raise RuntimeError("exact output directory requires formal LIBERO authority")
        if self.t.get("formal_non_resumable", None) is not True:
            raise RuntimeError("formal LIBERO training must be explicitly non-resumable")

        configured = os.fspath(self.t.get("output_dir", ""))
        supplied = os.fspath(exact_output_dir)
        if not configured or not supplied:
            raise RuntimeError("formal LIBERO output directory cannot be empty")
        if not os.path.isabs(configured) or not os.path.isabs(supplied):
            raise RuntimeError("formal LIBERO output directory must be absolute")
        if os.path.normpath(configured) != configured or supplied != configured:
            raise RuntimeError(
                "formal LIBERO output argument differs from training.output_dir"
            )
        metadata = os.lstat(supplied)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("formal LIBERO output root must be a real directory")
        if os.path.realpath(supplied) != supplied:
            raise RuntimeError("formal LIBERO output root ancestry must not use symlinks")
        expected_world_size = self.t.get("expected_world_size", None)
        if (
            isinstance(expected_world_size, bool)
            or not isinstance(expected_world_size, int)
            or expected_world_size < 1
        ):
            raise RuntimeError(
                "formal LIBERO exact-root training requires expected_world_size >= 1"
            )
        if _world_size() != expected_world_size:
            raise RuntimeError(
                "formal LIBERO exact-root world size differs from expected_world_size"
            )
        return supplied

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _seed_process(base_seed: int, rank: int) -> int:
        """Seed every RNG used by data/model setup, offset once per DDP rank."""
        process_seed = int(base_seed) + int(rank)
        random.seed(process_seed)
        np.random.seed(process_seed % (2**32))
        torch.manual_seed(process_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(process_seed)
        return process_seed

    def _broadcast_initial_model_state(self) -> None:
        """Make the complete CUDA model state identical before rank RNG diverges."""

        if not _is_dist() or _world_size() == 1:
            self.distributed_initialization_summary = {
                "broadcast": False,
                "world_size": 1,
            }
            return
        entries, manifest = self._ordered_model_state_manifest()
        manifest_bytes = self._canonical_json_bytes(manifest)
        local_structure = {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "parameter_tensor_count": sum(
                item["kind"] == "parameter" for item in manifest
            ),
            "buffer_tensor_count": sum(item["kind"] == "buffer" for item in manifest),
            "tensor_count": len(manifest),
            "numel": sum(item["numel"] for item in manifest),
            "device_types": sorted({item["device_type"] for item in manifest}),
        }
        gathered_structures: list[Any] = [None] * _world_size()
        dist.all_gather_object(gathered_structures, local_structure)
        if any(item != local_structure for item in gathered_structures):
            raise RuntimeError(
                "distributed formal initialization model structure differs across ranks"
            )
        if local_structure["device_types"] != ["cuda"]:
            raise RuntimeError(
                "distributed formal initialization found non-CUDA model state"
            )

        broadcast_tensors = 0
        broadcast_numel = 0
        for _kind, _name, tensor in entries:
            if tensor.numel() == 0:
                continue
            dist.broadcast(tensor.detach(), src=0)
            broadcast_tensors += 1
            broadcast_numel += tensor.numel()
        self.distributed_initialization_summary = {
            "broadcast": True,
            "source_rank": 0,
            "tensor_count": broadcast_tensors,
            "numel": broadcast_numel,
            "world_size": _world_size(),
            "structure_consensus": True,
            "structure_manifest_sha256": local_structure["manifest_sha256"],
            "parameter_tensor_count": local_structure["parameter_tensor_count"],
            "buffer_tensor_count": local_structure["buffer_tensor_count"],
        }
        dist.barrier()

    @staticmethod
    def _canonical_json_bytes(value: Any) -> bytes:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def _ordered_model_state_manifest(
        self,
    ) -> tuple[list[tuple[str, str, torch.Tensor]], list[dict[str, Any]]]:
        """Describe every unique parameter and buffer in collective order."""

        entries: list[tuple[str, str, torch.Tensor]] = []
        manifest: list[dict[str, Any]] = []
        named_groups = (
            ("parameter", self.architecture.named_parameters()),
            ("buffer", self.architecture.named_buffers()),
        )
        for kind, named_tensors in named_groups:
            for name, tensor in named_tensors:
                item = {
                    "kind": kind,
                    "name": name,
                    "shape": list(tensor.shape),
                    "stride": (
                        list(tensor.stride())
                        if tensor.layout == torch.strided
                        else None
                    ),
                    "dtype": str(tensor.dtype),
                    "layout": str(tensor.layout),
                    "device_type": tensor.device.type,
                    "numel": tensor.numel(),
                    "requires_grad": bool(tensor.requires_grad),
                }
                entries.append((kind, name, tensor))
                manifest.append(item)
        return entries, manifest

    def _complete_model_state_digest_consensus(self) -> dict[str, Any]:
        """Hash every parameter/buffer byte and require one digest on all ranks."""

        entries, manifest = self._ordered_model_state_manifest()
        manifest_bytes = self._canonical_json_bytes(manifest)
        digest = hashlib.sha256()
        digest.update(len(manifest_bytes).to_bytes(8, "big"))
        digest.update(manifest_bytes)
        chunk_bytes = 64 * 1024 * 1024
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with torch.no_grad():
            for kind, name, tensor in entries:
                if tensor.layout != torch.strided:
                    raise RuntimeError(
                        "formal LIBERO state digest requires strided tensors, got "
                        f"{kind} {name}: {tensor.layout}"
                    )
                raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
                for offset in range(0, raw.numel(), chunk_bytes):
                    block = raw.narrow(
                        0, offset, min(chunk_bytes, raw.numel() - offset)
                    ).cpu()
                    digest.update(block.numpy().tobytes(order="C"))
        local_digest = digest.hexdigest()
        gathered_digests = [local_digest]
        if _is_dist():
            gathered_digests = [None] * _world_size()
            dist.all_gather_object(gathered_digests, local_digest)
        if any(item != local_digest for item in gathered_digests):
            raise RuntimeError(
                "formal LIBERO final parameter/buffer state differs across ranks"
            )
        return {
            "algorithm": "sha256",
            "sha256": local_digest,
            "consensus": True,
            "consensus_world_size": _world_size(),
            "tensor_count": len(entries),
            "parameter_tensor_count": sum(
                item["kind"] == "parameter" for item in manifest
            ),
            "buffer_tensor_count": sum(item["kind"] == "buffer" for item in manifest),
            "numel": sum(item["numel"] for item in manifest),
            "includes": "all_named_parameters_and_buffers",
        }

    @staticmethod
    def _normalize_expected_sha256(value, field_name: str) -> str:
        digest = str(value or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"training.{field_name} must be exactly 64 hexadecimal characters"
            )
        return digest

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _verify_file_sha256(self, path, expected, *, field_name: str) -> str:
        """Resolve and hash an artifact once on rank 0, then share the result."""
        resolved = str(Path(os.fspath(path)).expanduser().resolve())
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"training.{field_name} artifact is not a file: {resolved}"
            )
        expected_digest = self._normalize_expected_sha256(
            expected, f"{field_name}_sha256"
        )

        if _is_dist():
            payload = [None]
            if _rank() == 0:
                try:
                    payload[0] = (self._sha256_file(resolved), None)
                except Exception as exc:  # broadcast failure so peers do not hang
                    payload[0] = (None, f"{type(exc).__name__}: {exc}")
            dist.broadcast_object_list(payload, src=0)
            actual_digest, error = payload[0]
            if error is not None:
                raise RuntimeError(f"failed to hash training.{field_name}: {error}")
        else:
            actual_digest = self._sha256_file(resolved)

        if actual_digest != expected_digest:
            raise ValueError(
                f"training.{field_name} SHA-256 mismatch: expected {expected_digest}, "
                f"got {actual_digest} ({resolved})"
            )
        return resolved

    def _verify_action_stats(self) -> None:
        expected = self.t.get("action_stats_sha256", None)
        if expected in (None, ""):
            return
        path = getattr(self.dataset, "action_stats_path", None)
        if not path:
            raise ValueError(
                "training.action_stats_sha256 requires dataset.action_stats_path"
            )
        self._verified_action_stats_path = self._verify_file_sha256(
            path,
            expected,
            field_name="action_stats",
        )
        self._verified_action_stats_sha256 = self._normalize_expected_sha256(
            expected, "action_stats_sha256"
        )
        if _rank() == 0:
            logger.info(
                "Verified action-stats SHA-256: %s", self._verified_action_stats_path
            )

    @staticmethod
    def _config_value(config, key: str, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    def _phase6_action_reference_enabled(self) -> bool:
        model_cfg = getattr(self.cfg, "model", None)
        architecture_cfg = self._config_value(model_cfg, "architecture", None)
        raw_weight = self._config_value(
            architecture_cfg, "action_non_regression_weight", 0.0
        )
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(
                "model.architecture.action_non_regression_weight must be exactly 0 or 1"
            )
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight not in (0.0, 1.0):
            raise ValueError(
                "model.architecture.action_non_regression_weight must be exactly 0 or 1"
            )
        adapter_cfg = self._config_value(
            architecture_cfg, "action_video_memory_adapter", None
        )
        adapter_enabled = self._config_value(adapter_cfg, "enabled", False)
        if type(adapter_enabled) is not bool:
            raise ValueError(
                "model.architecture.action_video_memory_adapter.enabled must be boolean"
            )
        if adapter_enabled != (weight == 1.0):
            raise ValueError(
                "action reference factor and action video-memory adapter enabled flag differ"
            )
        return weight == 1.0

    def _configure_phase1_action_reference(self) -> None:
        """Verify common Phase1 provenance and load a v2 table only for A1."""

        common_fields = (
            "phase1_reference_checkpoint",
            "phase1_reference_checkpoint_sha256",
        )
        table_fields = (
            "phase1_action_reference_artifact",
            "phase1_action_reference_artifact_sha256",
            "phase1_action_reference_checkpoint_sha256",
            "phase1_action_reference_live_spot_check_required",
            "phase1_action_reference_live_spot_check_artifact",
            "phase1_action_reference_live_spot_check_artifact_sha256",
        )
        common_present = {
            name: self.t.get(name, None) not in (None, "") for name in common_fields
        }
        table_present = {
            name: self.t.get(name, None) not in (None, "") for name in table_fields
        }
        if self._phase6_plan is None:
            if any(common_present.values()) or any(table_present.values()):
                raise ValueError(
                    "Phase1 Phase-6 reference fields are forbidden for legacy training"
                )
            return
        if not all(common_present.values()):
            missing = [name for name, present in common_present.items() if not present]
            raise ValueError(
                "every Phase-6 arm must pin the common Phase1 checkpoint; "
                f"missing={missing}"
            )

        expected_checkpoint_sha256 = self._normalize_expected_sha256(
            self.t.get("phase1_reference_checkpoint_sha256"),
            "phase1_reference_checkpoint_sha256",
        )
        self._verified_phase1_reference_checkpoint_path = self._verify_file_sha256(
            self.t.get("phase1_reference_checkpoint"),
            expected_checkpoint_sha256,
            field_name="phase1_reference_checkpoint",
        )
        action_reference_enabled = self._phase6_action_reference_enabled()
        self._phase1_action_reference_enabled_config = action_reference_enabled
        if not action_reference_enabled:
            if any(table_present.values()):
                forbidden = [name for name, present in table_present.items() if present]
                raise ValueError(
                    "A0 must not configure or load an action-reference table; "
                    f"forbidden={forbidden}"
                )
            return
        if not all(table_present.values()):
            missing = [name for name, present in table_present.items() if not present]
            raise ValueError(
                "A1 requires every action-reference artifact/checkpoint/spot field; "
                f"missing={missing}"
            )
        if self.t.get("phase1_action_reference_live_spot_check_required") is not True:
            raise ValueError(
                "A1 requires phase1_action_reference_live_spot_check_required=true"
            )
        table_checkpoint_sha256 = self._normalize_expected_sha256(
            self.t.get("phase1_action_reference_checkpoint_sha256"),
            "phase1_action_reference_checkpoint_sha256",
        )
        if table_checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError(
                "common and action-table Phase1 checkpoint SHA256 values differ"
            )
        table_path = self._verify_file_sha256(
            self.t.get("phase1_action_reference_artifact"),
            self.t.get("phase1_action_reference_artifact_sha256"),
            field_name="phase1_action_reference_artifact",
        )
        expected_table_sha256 = self._normalize_expected_sha256(
            self.t.get("phase1_action_reference_artifact_sha256"),
            "phase1_action_reference_artifact_sha256",
        )
        expected_dataset_contract_sha256 = self._normalize_expected_sha256(
            self.t.get("phase6_dataset_contract_artifact_sha256"),
            "phase6_dataset_contract_artifact_sha256",
        )
        expected_source_manifest_sha256 = self._normalize_expected_sha256(
            self.t.get("phase6_code_source_manifest_sha256"),
            "phase6_code_source_manifest_sha256",
        )
        if (
            self._phase6_dataset_contract.artifact_sha256
            != expected_dataset_contract_sha256
        ):
            raise RuntimeError("loaded dataset-contract artifact SHA changed")

        from sana_wam.train.action_reference_table import (
            ActionReferenceTable,
            validate_live_spot_artifact_bytes,
        )

        self._phase1_action_reference_table = ActionReferenceTable.from_artifact_bytes(
            Path(table_path).read_bytes(),
            self._phase6_plan,
            self._phase6_dataset_contract.rows,
            expected_artifact_sha256=expected_table_sha256,
            expected_dataset_contract_artifact_sha256=(
                expected_dataset_contract_sha256
            ),
            expected_reference_checkpoint_sha256=expected_checkpoint_sha256,
            expected_action_stats_sha256=self._verified_action_stats_sha256,
            expected_precompute_source_manifest_sha256=(
                expected_source_manifest_sha256
            ),
        )
        spot_path = self._verify_file_sha256(
            self.t.get("phase1_action_reference_live_spot_check_artifact"),
            self.t.get("phase1_action_reference_live_spot_check_artifact_sha256"),
            field_name="phase1_action_reference_live_spot_check_artifact",
        )
        validate_live_spot_artifact_bytes(
            Path(spot_path).read_bytes(),
            self._phase1_action_reference_table,
            self._phase6_plan,
            self._phase6_dataset_contract.rows,
            expected_artifact_sha256=self._normalize_expected_sha256(
                self.t.get("phase1_action_reference_live_spot_check_artifact_sha256"),
                "phase1_action_reference_live_spot_check_artifact_sha256",
            ),
            expected_reference_artifact_sha256=expected_table_sha256,
        )
        if _rank() == 0:
            logger.info(
                "Loaded Phase1 action-reference v2: rows=%d artifact_sha256=%s",
                len(self._phase1_action_reference_table.error_bits),
                expected_table_sha256,
            )

    def _configure_action_facing_cache_reference(self) -> None:
        """Load the independent AFCC tensor store only for the F1 treatment."""

        model_cfg = getattr(self.cfg, "model", None)
        architecture_cfg = self._config_value(model_cfg, "architecture", None)
        from sana_wam.model.action_facing_cache_consistency import (
            validate_afcc_weight,
        )

        weight = validate_afcc_weight(
            self._config_value(
                architecture_cfg,
                "action_facing_cache_consistency_weight",
                0.0,
            )
        )
        enabled = weight == 1.0
        self._action_facing_cache_consistency_enabled_config = enabled
        fields = (
            "phase1_action_facing_cache_reference_manifest",
            "phase1_action_facing_cache_reference_manifest_sha256",
        )
        present = {
            field: self.t.get(field, None) not in (None, "") for field in fields
        }
        if not enabled:
            if any(present.values()):
                forbidden = [field for field, value in present.items() if value]
                raise ValueError(
                    "AFCC reference fields are forbidden when F=0; "
                    f"forbidden={forbidden}"
                )
            return
        if self._phase6_plan is None or self._phase6_dataset_contract is None:
            raise ValueError("AFCC requires the frozen 504-row Phase-6 plan")
        if not all(present.values()):
            missing = [field for field, value in present.items() if not value]
            raise ValueError(f"AFCC requires manifest path and SHA256; missing={missing}")
        if self._phase1_action_reference_table is not None:
            raise ValueError("AFCC and the legacy A1 scalar reference are exclusive")
        if self.lambda_action != 0.0:
            raise ValueError("AFCC requires lambda_action exactly zero")
        if int(self.t.get("batch_size", 0)) != 1:
            raise ValueError("AFCC requires batch_size exactly one")
        expansion_weight = float(
            self._config_value(
                architecture_cfg, "video_local_expansion_weight", 0.0
            )
        )
        action_nr_weight = float(
            self._config_value(
                architecture_cfg, "action_non_regression_weight", 0.0
            )
        )
        adapter_cfg = self._config_value(
            architecture_cfg, "action_video_memory_adapter", None
        )
        adapter_enabled = self._config_value(adapter_cfg, "enabled", False)
        video_cfg = self._config_value(model_cfg, "video_backbone", None)
        continuous_timestep = self._config_value(
            video_cfg, "continuous_timestep_conditioning", False
        )
        if (
            expansion_weight != 1.0
            or action_nr_weight != 0.0
            or adapter_enabled is not False
            or continuous_timestep is not True
        ):
            raise ValueError("AFCC requires the exact T1/E1/A0 scientific condition")
        from sana_wam.train.phase6_arm_config import VIDEO_PATTERNS

        patterns = self._as_config_list(
            self.t.get("trainable_parameter_patterns", None),
            "trainable_parameter_patterns",
        )
        if tuple(patterns) != VIDEO_PATTERNS:
            raise ValueError("AFCC trainable patterns differ from T1_E1A0 control")

        manifest_path = self._verify_file_sha256(
            self.t.get("phase1_action_facing_cache_reference_manifest"),
            self.t.get("phase1_action_facing_cache_reference_manifest_sha256"),
            field_name="phase1_action_facing_cache_reference_manifest",
        )
        expected_reference_checkpoint_sha256 = self._normalize_expected_sha256(
            self.t.get("phase1_reference_checkpoint_sha256"),
            "phase1_reference_checkpoint_sha256",
        )
        if self._verified_phase1_reference_checkpoint_path is None:
            raise RuntimeError("AFCC common Phase1 checkpoint was not verified")
        from sana_wam.train.action_facing_cache_reference import (
            ActionFacingCacheReferenceStore,
        )

        self._action_facing_cache_reference_store = (
            ActionFacingCacheReferenceStore(
                manifest_path=manifest_path,
                manifest_sha256=self._normalize_expected_sha256(
                    self.t.get(
                        "phase1_action_facing_cache_reference_manifest_sha256"
                    ),
                    "phase1_action_facing_cache_reference_manifest_sha256",
                ),
                expected_plan_sha256=self._phase6_plan.plan_sha256,
                expected_identity_sha256=self._phase6_plan.identity_sha256,
                expected_dataset_contract_sha256=(
                    self._phase6_dataset_contract.artifact_sha256
                ),
                expected_reference_checkpoint_sha256=(
                    expected_reference_checkpoint_sha256
                ),
                expected_action_stats_sha256=self._verified_action_stats_sha256,
                expected_source_manifest_sha256=self._normalize_expected_sha256(
                    self.t.get("phase6_code_source_manifest_sha256"),
                    "phase6_code_source_manifest_sha256",
                ),
            )
        )
        if _rank() == 0:
            logger.info(
                "Loaded AFCC Phase1 references: rows=%d manifest_sha256=%s",
                len(self._action_facing_cache_reference_store.rows),
                self._action_facing_cache_reference_store.manifest_sha256,
            )

    def _configure_phase6_plan(self) -> None:
        """Bind the exact 504-row Phase-6 artifact or preserve the legacy loader."""
        field_names = (
            "phase6_registry",
            "phase6_registry_sha256",
            "phase6_plan_artifact",
            "phase6_plan_artifact_sha256",
            "phase6_plan_sha256",
            "phase6_identity_sha256",
            "phase6_dataset_contract_artifact",
            "phase6_dataset_contract_artifact_sha256",
        )
        configured = {
            name: self.t.get(name, None) not in (None, "") for name in field_names
        }
        if not any(configured.values()):
            return
        if not all(configured.values()):
            missing = [name for name, present in configured.items() if not present]
            raise ValueError(
                "Phase-6 plan configuration must set every registry/artifact/hash "
                f"field together; missing={missing}"
            )

        contract_values = {
            "expected_world_size": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 504,
            "lr_schedule_steps": 504,
        }
        for name, expected in contract_values.items():
            value = self.t.get(name, None)
            if isinstance(value, bool) or value != expected:
                raise ValueError(
                    f"Phase-6 requires training.{name}={expected}, got {value!r}"
                )
        scalar_contract = {
            "video_lr": 5.0e-6,
            "warmup_steps": 50.0,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "lambda_video": 1.0,
            "lambda_action": 0.0,
        }
        for name, expected in scalar_contract.items():
            value = self.t.get(name, None)
            if isinstance(value, bool):
                raise ValueError(
                    f"Phase-6 requires training.{name}={expected}, got {value!r}"
                )
            try:
                observed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Phase-6 requires training.{name}={expected}, got {value!r}"
                ) from exc
            if not math.isfinite(observed) or observed != expected:
                raise ValueError(
                    f"Phase-6 requires training.{name}={expected}, got {value!r}"
                )
        if self.t.get("optimizer_master_weights", None) is not True:
            raise ValueError("Phase-6 requires training.optimizer_master_weights=true")
        if _world_size() != 1:
            raise RuntimeError("one Phase-6 arm must run on exactly one process/GPU")

        registry_path = self._verify_file_sha256(
            self.t.get("phase6_registry"),
            self.t.get("phase6_registry_sha256"),
            field_name="phase6_registry",
        )
        artifact_path = self._verify_file_sha256(
            self.t.get("phase6_plan_artifact"),
            self.t.get("phase6_plan_artifact_sha256"),
            field_name="phase6_plan_artifact",
        )
        dataset_contract_path = self._verify_file_sha256(
            self.t.get("phase6_dataset_contract_artifact"),
            self.t.get("phase6_dataset_contract_artifact_sha256"),
            field_name="phase6_dataset_contract_artifact",
        )

        from sana_wam.dataloader.robotwin_plan_binding import (
            PlanBoundRoboTwinDataset,
        )
        from sana_wam.dataloader.phase6_dataset_contract import (
            Phase6DatasetContract,
        )
        from sana_wam.dataloader import robotwin_dataset as robotwin_dataset_module
        from sana_wam.dataloader.transforms import multiview as multiview_module
        from sana_wam.dataloader.transforms import normalize as normalize_module
        from sana_wam.dataloader.transforms import rotation as rotation_module
        from sana_wam.dataloader.task_sample_plan import (
            PlanSampler,
            SamplerContract,
            TaskRoundRobinPlan,
        )

        contract = SamplerContract.from_registry(registry_path)
        expected_plan_sha256 = self._normalize_expected_sha256(
            self.t.get("phase6_plan_sha256"), "phase6_plan_sha256"
        )
        expected_identity_sha256 = self._normalize_expected_sha256(
            self.t.get("phase6_identity_sha256"), "phase6_identity_sha256"
        )
        artifact_bytes = Path(artifact_path).read_bytes()
        plan = TaskRoundRobinPlan.from_artifact_bytes(
            artifact_bytes,
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
        )
        expected_action_stats_sha256 = self._normalize_expected_sha256(
            self.t.get("action_stats_sha256", None), "action_stats_sha256"
        )
        if self._verified_action_stats_sha256 != expected_action_stats_sha256:
            raise RuntimeError(
                "Phase-6 action stats were not byte-verified before plan binding"
            )
        expected_dataset_contract_sha256 = self._normalize_expected_sha256(
            self.t.get("phase6_dataset_contract_artifact_sha256"),
            "phase6_dataset_contract_artifact_sha256",
        )
        dataset_contract = Phase6DatasetContract.from_artifact_bytes(
            Path(dataset_contract_path).read_bytes(),
            expected_artifact_sha256=expected_dataset_contract_sha256,
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
            expected_action_stats_sha256=expected_action_stats_sha256,
        )
        dataloader_cfg = self.cfg.dataloader
        dataset_type = str(dataloader_cfg.get("type", ""))
        preprocessing_modules = {
            "sana_wam.dataloader.robotwin_dataset": robotwin_dataset_module,
            "sana_wam.dataloader.transforms.multiview": multiview_module,
            "sana_wam.dataloader.transforms.normalize": normalize_module,
            "sana_wam.dataloader.transforms.rotation": rotation_module,
        }
        preprocessing_source_paths = {
            source_id: inspect.getsourcefile(module)
            for source_id, module in preprocessing_modules.items()
        }
        missing_sources = [
            source_id
            for source_id, source_path in preprocessing_source_paths.items()
            if not source_path
        ]
        if missing_sources:
            raise RuntimeError(
                f"cannot locate Phase-6 preprocessing source files: {missing_sources}"
            )
        dataset_contract.validate_runtime(
            self.dataset,
            plan,
            dataloader_config=dataloader_cfg,
            action_stats_sha256=expected_action_stats_sha256,
            preprocessing_source_paths=preprocessing_source_paths,
            train_tasks=contract.train_tasks,
        )
        self.dataset = PlanBoundRoboTwinDataset(
            self.dataset,
            plan,
            contract=contract,
            dataset_type=dataset_type,
            world_size=_world_size(),
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
            dataset_contract=dataset_contract,
        )
        self._phase6_plan = plan
        self._phase6_dataset_contract = dataset_contract
        self._phase6_plan_sampler = PlanSampler(
            plan,
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
        )
        if _rank() == 0:
            logger.info(
                "Bound Phase-6 plan and dataset contract: rows=%d plan_sha256=%s "
                "identity_sha256=%s dataset_contract_sha256=%s",
                len(plan.rows),
                plan.plan_sha256,
                plan.identity_sha256,
                dataset_contract.artifact_sha256,
            )

    def _load_initial_checkpoint(self) -> None:
        checkpoint = self.t.get("init_checkpoint", None)
        expected = self.t.get("init_checkpoint_sha256", None)
        has_checkpoint = checkpoint not in (None, "")
        has_digest = expected not in (None, "")
        if has_checkpoint != has_digest:
            raise ValueError(
                "training.init_checkpoint and training.init_checkpoint_sha256 must be set together"
            )
        if not has_checkpoint:
            return

        checkpoint_path = self._verify_file_sha256(
            checkpoint,
            expected,
            field_name="init_checkpoint",
        )
        allow_missing = self._action_adapter_warm_start_contract(checkpoint_path)
        if allow_missing:
            self.architecture.load_checkpoint(
                checkpoint_path,
                strict=True,
                allow_missing_patterns=allow_missing,
            )
        else:
            self.architecture.load_checkpoint(checkpoint_path, strict=True)
        if _rank() == 0:
            logger.info(
                "Initialized weights from %s (strict load%s; optimizer and LR "
                "schedule start fresh)",
                checkpoint_path,
                ", identity-init action video-memory adapter" if allow_missing else "",
            )

    def _action_adapter_warm_start_contract(
        self, checkpoint_path: str
    ) -> tuple[str, ...]:
        """Allow only a complete A-off -> identity-init A-on warm start."""
        from safetensors import safe_open

        adapter = getattr(self.architecture, "action_video_memory_adapter", None)
        if adapter is None:
            # Preserve the legacy A-off strict-load path exactly. Any adapter
            # keys in a real checkpoint remain unexpected under strict loading.
            return ()
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            present = {
                key
                for key in checkpoint.keys()
                if key.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
            }

        expected = {
            f"{_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX}{key}"
            for key in adapter.state_dict()
        }
        if not expected:
            raise RuntimeError(
                "A-on architecture constructed an adapter with no checkpoint keys"
            )
        if present:
            missing = sorted(expected - present)
            unexpected = sorted(present - expected)
            state = "complete" if present == expected else "partial"
            raise RuntimeError(
                "A-on student initialization requires an A-off checkpoint with "
                "all action video-memory adapter keys absent; "
                f"found {state} adapter state, missing={missing}, "
                f"unexpected={unexpected}"
            )

        nonfinite = [
            name
            for name, value in adapter.state_dict().items()
            if not bool(torch.isfinite(value).all().item())
        ]
        if nonfinite:
            raise RuntimeError(
                "A-on adapter identity initialization contains non-finite "
                f"parameters: {nonfinite}"
            )
        nonzero = []
        for name in _ACTION_VIDEO_MEMORY_IDENTITY_ZERO_PARAMETERS:
            parameter = getattr(adapter, name, None)
            if parameter is None or bool(torch.count_nonzero(parameter).item()):
                nonzero.append(name)
        if nonzero:
            raise RuntimeError(
                "A-on adapter is not exact identity initialization; nonzero or "
                f"missing identity parameters: {nonzero}"
            )
        return (_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX,)

    @staticmethod
    def _as_config_list(value, field_name: str) -> list:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"training.{field_name} must be a list")
        return list(value)

    def _resolve_eval_modules(self) -> tuple[str, ...]:
        raw = self.t.get("eval_modules", None)
        if raw is None:
            return ()
        paths = self._as_config_list(raw, "eval_modules")
        normalized = []
        for path in paths:
            if not isinstance(path, str) or not path or path.strip() != path:
                raise ValueError(
                    "training.eval_modules entries must be non-empty dotted paths"
                )
            try:
                self.architecture.get_submodule(path)
            except (AttributeError, KeyError) as exc:
                raise ValueError(
                    f"unknown training.eval_modules path: {path!r}"
                ) from exc
            if path in normalized:
                raise ValueError(f"duplicate training.eval_modules entry: {path}")
            normalized.append(path)
        return tuple(normalized)

    def _configure_trainable_modules(self) -> list[str]:
        """Apply legacy freezes or an exact, fail-closed top-level allowlist."""
        raw_allowlist = self.t.get("trainable_modules", None)
        raw_patterns = self.t.get("trainable_parameter_patterns", None)
        raw_preserve_input_grad = self.t.get(
            "preserve_frozen_input_grad_modules", None
        )
        if raw_preserve_input_grad is not None and raw_allowlist is None:
            raise ValueError(
                "training.preserve_frozen_input_grad_modules requires "
                "training.trainable_modules"
            )
        if raw_patterns is not None:
            if raw_allowlist is not None or self.t.get("freeze", None):
                raise ValueError(
                    "training.trainable_parameter_patterns cannot be combined with "
                    "training.trainable_modules or training.freeze"
                )
            patterns = self._as_config_list(
                raw_patterns, "trainable_parameter_patterns"
            )
            if not patterns:
                raise ValueError(
                    "training.trainable_parameter_patterns must not be empty"
                )
            normalized = []
            for pattern in patterns:
                if (
                    not isinstance(pattern, str)
                    or not pattern
                    or pattern.strip() != pattern
                ):
                    raise ValueError(
                        "training.trainable_parameter_patterns entries must be "
                        "non-empty glob patterns"
                    )
                if pattern in normalized:
                    raise ValueError(
                        f"duplicate training.trainable_parameter_patterns entry: {pattern}"
                    )
                normalized.append(pattern)

            named_parameters = dict(self.architecture.named_parameters())
            match_counts = {pattern: 0 for pattern in normalized}
            trainable_names = []
            for name, parameter in named_parameters.items():
                matched = False
                for pattern in normalized:
                    if fnmatch.fnmatchcase(name, pattern):
                        match_counts[pattern] += 1
                        matched = True
                parameter.requires_grad_(matched)
                if matched:
                    trainable_names.append(name)
            unmatched = [
                pattern for pattern, count in match_counts.items() if count == 0
            ]
            if unmatched:
                raise ValueError(
                    "training.trainable_parameter_patterns matched no parameters: "
                    f"{unmatched}"
                )
            if not trainable_names:
                raise RuntimeError("parameter-pattern contract selected no parameters")
            roots = sorted({name.split(".", 1)[0] for name in trainable_names})
            self._trainable_parameter_patterns = tuple(normalized)
            self._trainable_module_allowlist = tuple(roots)
            self._frozen_module_paths = ()
            if _rank() == 0:
                logger.info(
                    "Trainable parameter patterns selected %d tensors under roots %s",
                    len(trainable_names),
                    roots,
                )
            return []
        if raw_allowlist is None:
            freeze_list = list(self.t.get("freeze", []) or [])
            frozen = self.architecture.freeze_modules(freeze_list)
            self._frozen_module_paths = tuple(frozen)
            return frozen

        allowlist = self._as_config_list(raw_allowlist, "trainable_modules")
        if not allowlist:
            raise ValueError("training.trainable_modules must not be empty")
        if self.t.get("freeze", None):
            raise ValueError(
                "training.freeze cannot be combined with training.trainable_modules; "
                "the allowlist is the sole freeze authority"
            )

        normalized = []
        for name in allowlist:
            if (
                not isinstance(name, str)
                or not name
                or name.strip() != name
                or "." in name
            ):
                raise ValueError(
                    "training.trainable_modules entries must be exact top-level module names"
                )
            if name in normalized:
                raise ValueError(f"duplicate training.trainable_modules entry: {name}")
            normalized.append(name)

        top_level = dict(self.architecture.named_children())
        unknown = sorted(set(normalized) - set(top_level))
        if unknown:
            raise ValueError(
                f"unknown training.trainable_modules entries: {unknown}; "
                f"available top-level modules: {sorted(top_level)}"
            )

        preserve_input_grad = []
        if raw_preserve_input_grad is not None:
            requested = self._as_config_list(
                raw_preserve_input_grad,
                "preserve_frozen_input_grad_modules",
            )
            for name in requested:
                if (
                    not isinstance(name, str)
                    or not name
                    or name.strip() != name
                    or "." in name
                ):
                    raise ValueError(
                        "training.preserve_frozen_input_grad_modules entries "
                        "must be exact top-level module names"
                    )
                if name in preserve_input_grad:
                    raise ValueError(
                        "duplicate training.preserve_frozen_input_grad_modules "
                        f"entry: {name}"
                    )
                if name not in top_level:
                    raise ValueError(
                        "unknown training.preserve_frozen_input_grad_modules "
                        f"entry: {name}"
                    )
                if name in normalized:
                    raise ValueError(
                        "training.preserve_frozen_input_grad_modules must name "
                        f"a frozen module, got trainable module: {name}"
                    )
                preserve_input_grad.append(name)

        # Direct architecture parameters have no top-level module owner and are
        # therefore outside the allowlist by definition.
        for parameter in self.architecture.parameters(recurse=False):
            parameter.requires_grad_(False)

        frozen = []
        for name in top_level:
            if name in normalized:
                continue
            if name in preserve_input_grad:
                actually_frozen = self.architecture.freeze_modules(
                    [name], preserve_input_grad=True
                )
            else:
                actually_frozen = self.architecture.freeze_modules([name])
            if actually_frozen != [name]:
                raise RuntimeError(
                    f"failed to freeze non-allowlisted top-level module: {name}"
                )
            frozen.append(name)

        trainable_names = [
            name
            for name, parameter in self.architecture.named_parameters()
            if parameter.requires_grad
        ]
        trainable_roots = {name.split(".", 1)[0] for name in trainable_names}
        unexpected = sorted(trainable_roots - set(normalized))
        missing = sorted(set(normalized) - trainable_roots)
        if not trainable_names or unexpected or missing:
            raise RuntimeError(
                "trainable-module allowlist contract failed: "
                f"missing_roots={missing}, unexpected_roots={unexpected}, "
                f"trainable_params={trainable_names[:20]}"
            )

        self._trainable_module_allowlist = tuple(normalized)
        self._frozen_module_paths = tuple(frozen)
        self._frozen_input_grad_module_paths = tuple(preserve_input_grad)
        return frozen

    def _configure_trainable_parameter_dtype(self) -> int:
        """Optionally retain small parameter-pattern warm starts in FP32.

        AdamW updates model parameters in their storage dtype. BF16's relative
        spacing can therefore erase low-LR updates entirely; promoting only the
        explicit parameter allowlist preserves accumulated updates without
        changing the frozen backbone's memory footprint.
        """
        raw = self.t.get("trainable_parameter_dtype", None)
        if raw is None:
            return 0
        if self._trainable_parameter_patterns is None:
            raise ValueError(
                "training.trainable_parameter_dtype requires "
                "training.trainable_parameter_patterns"
            )
        if str(raw).strip().lower() not in {"float32", "fp32"}:
            raise ValueError(
                "training.trainable_parameter_dtype currently supports only float32"
            )

        promoted = 0
        for parameter in self.architecture.parameters():
            if not parameter.requires_grad:
                continue
            parameter.data = parameter.data.to(dtype=torch.float32)
            promoted += 1
        if promoted == 0:
            raise RuntimeError("no trainable parameters were available for promotion")
        return promoted

    def _set_training_mode(self) -> None:
        """Enter train mode, then restore every frozen subtree to eval mode."""
        self.architecture.train()
        for path in self._frozen_module_paths:
            module = self.architecture.get_submodule(path)
            for submodule in module.modules():
                submodule.training = False
        for path in self._eval_module_paths:
            module = self.architecture.get_submodule(path)
            for submodule in module.modules():
                submodule.training = False

    def _load_action_stats(self):
        stats = getattr(self.dataset, "action_stats", None)
        if callable(stats):
            stats = stats()
        if stats is None:
            return
        mean = torch.from_numpy(stats["mean"].astype(np.float32))
        std = torch.from_numpy(np.maximum(stats["std"].astype(np.float32), 1e-3))
        self.architecture.action_mean.copy_(mean)
        self.architecture.action_std.copy_(std)
        if _rank() == 0:
            logger.info("Loaded action stats into architecture buffers")

    def _param_groups(self):
        """Separate action, action-memory, and video optimizer LR groups."""
        mods = self.architecture.get_trainable_modules(freeze_list=())
        action_params, action_memory_params, video_params = [], [], []
        for name, mod in mods.items():
            if name == "action_backbone":
                tgt = action_params
            elif name == "action_video_memory_adapter":
                tgt = action_memory_params
            else:
                tgt = video_params
            tgt += [p for p in mod.parameters() if p.requires_grad]
        groups = []
        if action_params:
            groups.append({"params": action_params, "lr": float(self.t.action_lr)})
        if action_memory_params:
            raw_lr = self.t.get("action_memory_lr", 1.0e-4)
            if isinstance(raw_lr, bool):
                raise ValueError(
                    "training.action_memory_lr must be a positive finite number"
                )
            try:
                action_memory_lr = float(raw_lr)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "training.action_memory_lr must be a positive finite number"
                ) from exc
            if not math.isfinite(action_memory_lr) or action_memory_lr <= 0.0:
                raise ValueError(
                    "training.action_memory_lr must be a positive finite number"
                )
            groups.append({"params": action_memory_params, "lr": action_memory_lr})
        if video_params:
            groups.append({"params": video_params, "lr": float(self.t.video_lr)})
        return groups

    def _trainable_params(self):
        return [p for g in self._param_groups() for p in g["params"]]

    def _optimizer_param_groups(self, model_groups):
        """Optionally back BF16 model tensors with FP32 AdamW master weights."""
        raw = self.t.get("optimizer_master_weights", False)
        if not isinstance(raw, bool):
            raise ValueError("training.optimizer_master_weights must be a boolean")
        if not raw:
            return model_groups, []

        optimizer_groups = []
        pairs = []
        for group in model_groups:
            master_params = []
            for model_parameter in group["params"]:
                master = torch.nn.Parameter(
                    model_parameter.detach().to(dtype=torch.float32),
                    requires_grad=True,
                )
                master_params.append(master)
                pairs.append((model_parameter, master))
            optimizer_groups.append(
                {
                    **{key: value for key, value in group.items() if key != "params"},
                    "params": master_params,
                }
            )
        return optimizer_groups, pairs

    def _optimizer_foreach_setting(self) -> bool | None:
        """Return the explicitly configured AdamW foreach policy, if any."""
        raw = self.t.get("optimizer_foreach", None)
        if raw is not None and type(raw) is not bool:
            raise ValueError("training.optimizer_foreach must be a boolean or null")
        return raw

    @staticmethod
    def _sync_master_gradients(pairs) -> None:
        for model_parameter, master in pairs:
            master.grad = (
                None
                if model_parameter.grad is None
                else model_parameter.grad.detach().to(dtype=torch.float32)
            )

    @staticmethod
    @torch.no_grad()
    def _copy_master_parameters_to_model(pairs) -> None:
        for model_parameter, master in pairs:
            model_parameter.copy_(master.to(dtype=model_parameter.dtype))

    def _set_data_epoch(self, epoch: int, sampler=None) -> None:
        set_epoch = getattr(self.dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)

    def _save_initial_checkpoint(self, output_path: str) -> bool:
        if not bool(self.t.get("save_initial_checkpoint", False)):
            return False
        self._save(output_path, 0)
        if _is_dist():
            dist.barrier()
        return True

    def _explicit_save_steps(self) -> set[int]:
        raw = self.t.get("save_at_steps", None)
        if raw is None:
            return set()
        values = self._as_config_list(raw, "save_at_steps")
        steps: set[int] = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"training.save_at_steps entries must be positive integers, got {value!r}"
                )
            if value in steps:
                raise ValueError(f"duplicate training.save_at_steps entry: {value}")
            steps.add(value)
        return steps

    @staticmethod
    def _checkpoint_due(step: int, save_steps: int, save_at_steps: set[int]) -> bool:
        return bool(
            (save_steps > 0 and step % save_steps == 0) or step in save_at_steps
        )

    def _resolve_lr_schedule_steps(self, max_steps: int) -> int:
        raw = self.t.get("lr_schedule_steps", max_steps)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(
                f"training.lr_schedule_steps must be a positive integer, got {raw!r}"
            )
        if raw < max_steps:
            raise ValueError(
                "training.lr_schedule_steps must be >= training.max_steps: "
                f"{raw} < {max_steps}"
            )
        return raw

    def _lr_lambda(self, step, total):
        warmup = int(self.t.get("warmup_steps", 0))
        if warmup and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    @staticmethod
    def _exclusive_write_bytes(
        path: str | os.PathLike[str], data: bytes, *, mode: int = 0o444
    ) -> None:
        destination = os.path.abspath(os.fspath(path))
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if hasattr(os, "O_DIRECTORY"):
                parent_descriptor = os.open(
                    os.path.dirname(destination), os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)
        except BaseException:
            try:
                os.unlink(destination)
            except FileNotFoundError:
                pass
            raise

    # ------------------------------------------------------------------ step
    def _validate_phase6_batch(self, batch, expected_global_step: int) -> dict:
        if getattr(self, "_phase6_plan", None) is None:
            raise RuntimeError("Phase-6 batch validation called without a plan")
        if (
            isinstance(expected_global_step, bool)
            or not isinstance(expected_global_step, int)
            or not 1 <= expected_global_step <= len(self._phase6_plan.rows)
        ):
            raise ValueError(
                f"invalid Phase-6 expected global step: {expected_global_step!r}"
            )
        if not isinstance(batch, list) or len(batch) != 1:
            raise RuntimeError("Phase-6 requires one materialized sample per step")
        sample = batch[0]
        if not isinstance(sample, dict):
            raise TypeError("Phase-6 collate output must contain one plain dict")
        row = sample.get("phase6_plan_row")
        if not isinstance(row, dict):
            raise RuntimeError("Phase-6 sample is missing its plain plan-row payload")
        if row.get("global_step") != expected_global_step:
            raise RuntimeError(
                "Phase-6 sample order drift: "
                f"expected global_step={expected_global_step}, "
                f"observed={row.get('global_step')!r}"
            )
        expected = self._phase6_plan.rows[expected_global_step - 1]
        from sana_wam.dataloader.robotwin_plan_binding import phase6_plan_row_dict

        expected_payload = phase6_plan_row_dict(
            expected,
            plan_sha256=self._phase6_plan.plan_sha256,
            identity_sha256=self._phase6_plan.identity_sha256,
        )
        if row != expected_payload:
            raise RuntimeError(
                f"Phase-6 plan-row payload drift at global_step={expected_global_step}"
            )
        return row

    def _compute_loss(self, batch, *, expected_phase6_step: int | None = None):
        raw_phase6_row = None
        if getattr(self, "_phase6_plan", None) is not None:
            if expected_phase6_step is None:
                raise RuntimeError(
                    "Phase-6 loss call requires the expected global step"
                )
            raw_phase6_row = self._validate_phase6_batch(batch, expected_phase6_step)
        elif expected_phase6_step is not None:
            raise RuntimeError("legacy loss call received a Phase-6 global step")
        inputs = self.architecture.prepare_inputs(batch)
        reference_table = None
        afcc_reference_store = None
        if raw_phase6_row is not None:
            prepared_rows = inputs.get("phase6_plan_rows")
            if prepared_rows != (raw_phase6_row,):
                raise RuntimeError(
                    "prepare_inputs changed or dropped the Phase-6 plan-row payload"
                )
            if "phase1_action_reference_error" in inputs:
                raise RuntimeError(
                    "prepare_inputs must not synthesize a Phase1 reference error"
                )
            if "phase1_action_facing_cache_reference" in inputs:
                raise RuntimeError(
                    "prepare_inputs must not synthesize an AFCC reference tensor"
                )
            # Every formal arm records the same plan-bound input trace. A1 also
            # checks it against the independently precomputed Phase1 row below.
            inputs["phase6_common_input_trace_required"] = True
            reference_table = getattr(self, "_phase1_action_reference_table", None)
            if reference_table is not None:
                reference_error = reference_table.error_for_global_step(
                    expected_phase6_step
                )
                inputs["phase1_action_reference_error"] = torch.tensor(
                    reference_error,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False,
                ).detach()
            elif getattr(self, "_phase1_action_reference_enabled_config", False):
                raise RuntimeError("A1 loss call has no loaded Phase1 reference table")
            afcc_reference_store = getattr(
                self, "_action_facing_cache_reference_store", None
            )
            if afcc_reference_store is not None:
                reference = afcc_reference_store.tensor_for_global_step(
                    expected_phase6_step
                )
                inputs["phase1_action_facing_cache_reference"] = (
                    reference.unsqueeze(0)
                    .to(device=self.device, dtype=torch.bfloat16)
                    .detach()
                )
            elif getattr(
                self,
                "_action_facing_cache_consistency_enabled_config",
                False,
            ):
                raise RuntimeError("AFCC loss call has no loaded tensor store")
        result = self.architecture.compute_loss(
            **inputs,
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
        )
        if raw_phase6_row is not None:
            observed_trace = result.get("phase6_common_input_trace_sha256")
            if (
                not isinstance(observed_trace, str)
                or re.fullmatch(r"[0-9a-f]{64}", observed_trace) is None
            ):
                raise RuntimeError(
                    "formal Phase-6 loss did not produce one common input trace"
                )
            if result.get("phase6_timestep_forward_trace_sha256") is not None:
                raise RuntimeError(
                    "formal training must not compute the T0-only forward trace"
                )
        if raw_phase6_row is not None and reference_table is not None:
            expected_trace = reference_table.common_input_trace_sha256[
                expected_phase6_step - 1
            ]
            if observed_trace != expected_trace:
                raise RuntimeError(
                    "A1 common input/noise trace differs from the paired Phase1 row "
                    f"at global_step={expected_phase6_step}"
                )
        if raw_phase6_row is not None and afcc_reference_store is not None:
            expected_trace = afcc_reference_store.common_trace_for_global_step(
                expected_phase6_step
            )
            if observed_trace != expected_trace:
                raise RuntimeError(
                    "AFCC common input/noise trace differs from the paired Phase1 "
                    f"row at global_step={expected_phase6_step}"
                )
            active = result.get("action_facing_cache_consistency_active")
            if (
                not isinstance(active, torch.Tensor)
                or active.numel() != 1
                or int(active.detach().item()) != 1
            ):
                raise RuntimeError("AFCC treatment did not activate its source loss")
        return result

    def _allreduce_grads(self, params):
        if _world_size() == 1:
            return
        ws = _world_size()
        presence = torch.tensor(
            [parameter.grad is not None for parameter in params],
            dtype=torch.int32,
            device=self.device,
        )
        minimum = presence.clone()
        maximum = presence.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if not torch.equal(minimum, maximum):
            raise RuntimeError("gradient presence differs across distributed ranks")
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= ws

    def _phase6_named_trainables(self):
        return [
            (name, parameter)
            for name, parameter in self.architecture.named_parameters()
            if parameter.requires_grad
        ]

    @staticmethod
    def _phase6_is_proprio_name(name: str) -> bool:
        return name.startswith(
            (
                "proprio_action_embed.",
                "proprio_encoder.",
                "proprio_video_embed.",
            )
        )

    def _validate_phase6_trainable_partition(self, named_trainables) -> None:
        context = self._phase6_launch_context
        if context is None:
            return
        action = [
            name for name, _ in named_trainables if name.startswith("action_backbone.")
        ]
        proprio = [
            name for name, _ in named_trainables if self._phase6_is_proprio_name(name)
        ]
        if action or proprio:
            raise RuntimeError(
                "formal Phase-6 action/proprio trainable count must be zero: "
                f"action={action[:10]}, proprio={proprio[:10]}"
            )
        adapter = sorted(
            name
            for name, _ in named_trainables
            if name.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
        )
        expected_adapter = (
            [
                f"{_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX}{name}"
                for name in (
                    "k_left",
                    "k_right",
                    "v_left",
                    "v_right",
                    "z_logits",
                )
            ]
            if context.factor_mapping["A"]
            else []
        )
        if adapter != sorted(expected_adapter):
            raise RuntimeError(
                "formal Phase-6 trainable adapter key set differs from A factor"
            )
        video = [
            name
            for name, _ in named_trainables
            if not name.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
        ]
        if not video:
            raise RuntimeError("formal Phase-6 selected no trainable video parameters")

    def _validate_afcc_trainable_partition(self, named_trainables) -> None:
        """Require the exact existing T1_E1A0 video-only allowlist."""

        from sana_wam.train.phase6_arm_config import VIDEO_PATTERNS

        names = [name for name, _parameter in named_trainables]
        invalid = [
            name
            for name in names
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in VIDEO_PATTERNS)
        ]
        if len(names) != 380 or invalid:
            raise RuntimeError(
                "formal AFCC trainable partition differs from control: "
                f"count={len(names)}, invalid={invalid[:20]}"
            )
        if any(
            name.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
            or name.startswith("action_backbone.")
            or self._phase6_is_proprio_name(name)
            for name in names
        ):
            raise RuntimeError("formal AFCC selected action/adapter/proprio parameters")

    def _phase6_probe_nr_autograd(self, result, named_trainables):
        """Probe the actual NR surrogate graph without changing accumulated grads."""

        context = self._phase6_launch_context
        if context is None or not context.factor_mapping["A"]:
            return None, None
        surrogate = result.get("phase6_action_non_regression_surrogate")
        if not isinstance(surrogate, torch.Tensor) or surrogate.ndim != 0:
            raise RuntimeError("A1 result lacks the scalar NR autograd surrogate")
        parameters = [parameter for _name, parameter in named_trainables]
        gradients = torch.autograd.grad(
            surrogate,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
            materialize_grads=False,
        )
        adapter_connected = True
        target_adapter_only = True
        for (name, _parameter), gradient in zip(
            named_trainables, gradients, strict=True
        ):
            is_adapter = name.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
            if is_adapter:
                if gradient is None or not bool(torch.isfinite(gradient).all().item()):
                    adapter_connected = False
            elif gradient is not None:
                target_adapter_only = False
        if not adapter_connected or not target_adapter_only:
            raise RuntimeError(
                "actual action NR surrogate is not connected adapter-only"
            )
        return adapter_connected, target_adapter_only

    @staticmethod
    def _phase6_tensors_finite(parameters) -> bool:
        return all(
            bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in parameters
        )

    @classmethod
    def _phase6_optimizer_state_finite(cls, optimizer) -> bool:
        def values(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, Mapping):
                for nested in value.values():
                    yield from values(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    yield from values(nested)

        return all(
            bool(torch.isfinite(tensor.detach()).all().item())
            for state in optimizer.state.values()
            for tensor in values(state)
        )

    def _phase6_gradient_measurements(
        self,
        named_trainables,
        *,
        adapter_gradient_connected,
        nr_gradient_target_adapter_only,
    ) -> dict:
        finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad.detach()).all().item())
            for _name, parameter in named_trainables
        )

        def group_norm(prefix_adapter: bool) -> float:
            norms = []
            for name, parameter in named_trainables:
                if (
                    name.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
                    != prefix_adapter
                ):
                    continue
                if parameter.grad is not None:
                    norms.append(
                        float(
                            torch.linalg.vector_norm(
                                parameter.grad.detach().float()
                            ).item()
                        )
                    )
            return math.sqrt(math.fsum(value * value for value in norms))

        factors = self._phase6_launch_context.factor_mapping
        if finite:
            video_norm = group_norm(False)
            adapter_norm = group_norm(True) if factors["A"] else None
            global_norm = math.sqrt(video_norm**2 + (adapter_norm or 0.0) ** 2)
            if not all(
                math.isfinite(value)
                for value in (video_norm, global_norm, adapter_norm or 0.0)
            ):
                finite = False
                video_norm = adapter_norm = global_norm = None
        else:
            video_norm = adapter_norm = global_norm = None
        return {
            "action_trainable_param_count": sum(
                parameter.numel()
                for name, parameter in named_trainables
                if name.startswith("action_backbone.")
            ),
            "adapter_gradient_connected": adapter_gradient_connected,
            "adapter_grad_norm": adapter_norm,
            "finite": finite,
            "global_norm": global_norm,
            "model_parameters_finite": True,
            "nr_gradient_target_adapter_only": nr_gradient_target_adapter_only,
            "optimizer_master_parameters_finite": True,
            "optimizer_state_finite": True,
            "proprio_trainable_param_count": sum(
                parameter.numel()
                for name, parameter in named_trainables
                if self._phase6_is_proprio_name(name)
            ),
            "video_grad_norm": video_norm,
        }

    @staticmethod
    def _phase6_scalar(result, key: str, *, minimum: float | None = None) -> float:
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise RuntimeError(f"formal Phase-6 result {key} is not scalar")
            observed = float(value.detach().float().item())
        else:
            observed = float(value)
        if not math.isfinite(observed) or (minimum is not None and observed < minimum):
            raise RuntimeError(f"formal Phase-6 result {key} is invalid: {observed}")
        return observed

    def _phase6_learning_rates(self, model_groups, optimizer) -> dict:
        adapter_lr = None
        video_lr = None
        name_by_id = {
            id(parameter): name
            for name, parameter in self.architecture.named_parameters()
        }
        for model_group, optimizer_group in zip(
            model_groups, optimizer.param_groups, strict=True
        ):
            names = [name_by_id[id(parameter)] for parameter in model_group["params"]]
            is_adapter = all(
                name.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX) for name in names
            )
            if not is_adapter and any(
                name.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX) for name in names
            ):
                raise RuntimeError("optimizer group mixes video and adapter parameters")
            observed = float(optimizer_group["lr"])
            if not math.isfinite(observed) or observed < 0.0:
                raise RuntimeError("optimizer learning rate is non-finite or negative")
            if is_adapter:
                if adapter_lr is not None:
                    raise RuntimeError("multiple action-adapter optimizer groups")
                adapter_lr = observed
            else:
                if video_lr is not None:
                    raise RuntimeError("multiple video optimizer groups")
                video_lr = observed
        factors = self._phase6_launch_context.factor_mapping
        if video_lr is None or (adapter_lr is not None) != factors["A"]:
            raise RuntimeError("optimizer LR groups differ from Phase-6 factors")
        return {"action_adapter": adapter_lr, "video": video_lr}

    def _phase6_measurements(
        self,
        result,
        gradients,
        learning_rates,
        *,
        peak_reserved_bytes: int,
    ) -> dict:
        factors = self._phase6_launch_context.factor_mapping
        student_error = self._phase6_scalar(
            result, "phase6_action_unweighted_mse", minimum=0.0
        )
        if factors["A"]:
            reference_error = self._phase6_scalar(
                result, "action_non_regression_reference_error", minimum=0.0
            )
            relative_excess = self._phase6_scalar(
                result, "action_non_regression_relative_excess"
            )
            nr_loss = self._phase6_scalar(
                result, "loss_action_non_regression", minimum=0.0
            )
        else:
            reference_error = relative_excess = None
            nr_loss = 0.0
        if factors["E"]:
            expansion_rate = self._phase6_scalar(result, "video_local_expansion_rate")
            expansion_loss = self._phase6_scalar(
                result, "loss_video_local_expansion", minimum=0.0
            )
            expansion_violation = expansion_rate > 8.0
        else:
            expansion_rate = expansion_violation = None
            expansion_loss = 0.0
        return {
            "action": {
                "active": factors["A"],
                "non_regression_loss": nr_loss,
                "reference_error": reference_error,
                "relative_excess": relative_excess,
                "student_error": student_error,
            },
            "common_input_trace_sha256": result["phase6_common_input_trace_sha256"],
            "expansion": {
                "active": factors["E"],
                "loss": expansion_loss,
                "rate": expansion_rate,
                "violation": expansion_violation,
            },
            "gradients": gradients,
            "learning_rates": learning_rates,
            "memory": {"peak_reserved_bytes": int(peak_reserved_bytes)},
            "total_loss": self._phase6_scalar(result, "loss", minimum=0.0),
            "video_on_path_loss": self._phase6_scalar(
                result, "loss_video_on_path", minimum=0.0
            ),
        }

    def _phase6_provenance(self) -> dict[str, str]:
        context = self._phase6_launch_context
        fields = {
            "action_reference_sha256": (
                "phase6_action_reference_provenance_artifact_sha256"
            ),
            "action_stats_sha256": "action_stats_sha256",
            "dataset_contract_sha256": ("phase6_dataset_contract_artifact_sha256"),
            "initial_checkpoint_sha256": "init_checkpoint_sha256",
            "phase1_checkpoint_sha256": "phase1_reference_checkpoint_sha256",
            "preflight_report_sha256": "phase6_preflight_report_sha256",
            "preflight_request_sha256": "phase6_preflight_request_sha256",
            "runtime_support_manifest_sha256": (
                "phase6_runtime_support_manifest_sha256"
            ),
            "smoke_artifact_sha256": ("phase6_real_2b_smoke_gate_artifact_sha256"),
            "source_manifest_sha256": "phase6_code_source_manifest_sha256",
            "spot_verification_sha256": (
                "phase6_action_reference_spot_provenance_artifact_sha256"
            ),
        }
        provenance = {
            role: self._normalize_expected_sha256(self.t.get(config_key), config_key)
            for role, config_key in fields.items()
        }
        provenance.update(
            {
                "arm_config_sha256": context.input_arm_config_sha256,
                "arm_config_projection_sha256": (context.arm_config_projection_sha256),
                "launch_manifest_sha256": context.launch_manifest_sha256,
            }
        )
        return provenance

    def _create_phase6_metrics_writer(self):
        from sana_wam.train.phase6_run_integrity import (
            Phase6MetricsWriter,
            Phase6PlanBinding,
        )

        context = self._phase6_launch_context
        binding = Phase6PlanBinding.from_artifact_path(
            self.t.get("phase6_plan_artifact"),
            expected_artifact_sha256=self._normalize_expected_sha256(
                self.t.get("phase6_plan_artifact_sha256"),
                "phase6_plan_artifact_sha256",
            ),
            expected_plan_sha256=self._normalize_expected_sha256(
                self.t.get("phase6_plan_sha256"), "phase6_plan_sha256"
            ),
            expected_identity_sha256=self._normalize_expected_sha256(
                self.t.get("phase6_identity_sha256"), "phase6_identity_sha256"
            ),
        )
        if (
            binding.plan_sha256 != self._phase6_plan.plan_sha256
            or binding.identity_sha256 != self._phase6_plan.identity_sha256
        ):
            raise RuntimeError("metrics plan binding differs from Trainer plan")
        return Phase6MetricsWriter(
            run_directory=context.run_directory,
            preflight_report_path=self.t.get("phase6_preflight_report"),
            binding=binding,
            arm=context.arm,
            run_id=context.run_id,
            provenance=self._phase6_provenance(),
        )

    @staticmethod
    def _phase6_link_or_copy_config(deploy_path: str, resolved_path: str) -> None:
        if os.path.lexists(resolved_path):
            raise FileExistsError(f"resolved config already exists: {resolved_path}")
        try:
            os.link(deploy_path, resolved_path)
        except OSError:
            with open(deploy_path, "rb") as source:
                payload = source.read()
            Trainer._exclusive_write_bytes(resolved_path, payload)
        else:
            if hasattr(os, "O_DIRECTORY"):
                descriptor = os.open(
                    os.path.dirname(resolved_path), os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        if Path(deploy_path).read_bytes() != Path(resolved_path).read_bytes():
            raise RuntimeError("config.yaml and resolved_config.yaml differ")

    def _finalize_phase6_integrity(self, writer, output_path: str) -> None:
        from sana_wam.train.phase6_run_integrity import (
            build_frozen_tensor_verification_manifest,
            build_resolved_config_verification_manifest,
            write_integrity_manifest,
        )

        context = self._phase6_launch_context
        checkpoint = os.path.join(output_path, "checkpoint_step_504.safetensors")
        deploy_config = os.path.join(output_path, "config.yaml")
        resolved_config = os.path.join(output_path, "resolved_config.yaml")
        action_stats = os.path.join(output_path, "action_stats.npy")
        self._phase6_link_or_copy_config(deploy_config, resolved_config)
        checkpoint_sha = self._sha256_file(checkpoint)
        action_stats_sha = self._sha256_file(action_stats)
        expected_action_stats_sha = self._normalize_expected_sha256(
            self.t.get("action_stats_sha256"), "action_stats_sha256"
        )
        if action_stats_sha != expected_action_stats_sha:
            raise RuntimeError("saved Phase-6 action_stats.npy is not byte-exact")
        frozen_manifest = build_frozen_tensor_verification_manifest(
            initial_checkpoint_path=self.t.get("init_checkpoint"),
            step504_checkpoint_path=checkpoint,
            initial_checkpoint_sha256=self._normalize_expected_sha256(
                self.t.get("init_checkpoint_sha256"), "init_checkpoint_sha256"
            ),
            step504_checkpoint_sha256=checkpoint_sha,
            arm=context.arm,
        )
        frozen_path = os.path.join(
            output_path, "frozen_action_proprio_verification.json"
        )
        frozen_sha = write_integrity_manifest(frozen_path, frozen_manifest)
        config_manifest = build_resolved_config_verification_manifest(
            arm=context.arm,
            input_arm_config_path=context.input_arm_config_path,
            input_arm_config_sha256=context.input_arm_config_sha256,
            resolved_config_path=resolved_config,
            deploy_config_path=deploy_config,
            expected_projection_sha256=context.arm_config_projection_sha256,
        )
        config_manifest_path = os.path.join(
            output_path, "resolved_config_verification.json"
        )
        config_manifest_sha = write_integrity_manifest(
            config_manifest_path, config_manifest
        )
        primary_outputs = {
            "action_stats": {
                "path": os.path.abspath(action_stats),
                "sha256": action_stats_sha,
            },
            "frozen_tensor_verification_manifest": {
                "path": os.path.abspath(frozen_path),
                "sha256": frozen_sha,
            },
            "resolved_config_verification_manifest": {
                "path": os.path.abspath(config_manifest_path),
                "sha256": config_manifest_sha,
            },
            "saved_arm_config": {
                "path": os.path.abspath(resolved_config),
                "sha256": self._sha256_file(resolved_config),
            },
            "step504_checkpoint": {
                "path": os.path.abspath(checkpoint),
                "sha256": checkpoint_sha,
            },
        }
        writer.finalize(primary_outputs=primary_outputs)

    # ------------------------------------------------------------------ train
    def train(self):
        t = self.t
        formal_phase6 = self._phase6_launch_context is not None
        formal_afcc = self._post_primary_afcc_authorized
        formal_libero = self._exact_output_dir is not None
        governed_training = formal_phase6 or formal_afcc
        max_steps = int(t.max_steps)
        lr_schedule_steps = self._resolve_lr_schedule_steps(max_steps)
        save_steps = int(t.get("save_steps", 0) or 0)
        save_at_steps = self._explicit_save_steps()
        grad_accum = int(t.get("gradient_accumulation_steps", 1))
        batch_size = int(t.get("batch_size", 1))
        grad_clip = float(t.get("grad_clip", 0.0) or 0.0)
        debug = bool(t.get("debug", False))
        formal_final_step = None
        if formal_libero:
            raw_final_step = t.get("formal_final_step", max_steps)
            if (
                isinstance(raw_final_step, bool)
                or not isinstance(raw_final_step, int)
                or raw_final_step < 1
                or raw_final_step != max_steps
            ):
                raise RuntimeError(
                    "formal LIBERO requires formal_final_step=max_steps as a "
                    "positive integer"
                )
            formal_final_step = raw_final_step
            global_batch = batch_size * grad_accum * _world_size()
            expected_global_batch = t.get("expected_global_batch_size", global_batch)
            if (
                isinstance(expected_global_batch, bool)
                or not isinstance(expected_global_batch, int)
                or expected_global_batch != global_batch
            ):
                raise RuntimeError(
                    "formal LIBERO global batch differs from "
                    "training.expected_global_batch_size"
                )

        if governed_training:
            expected_save_contract = {
                "keep_last_k": 1,
                "max_steps": 504,
                "save_initial_checkpoint": False,
                "save_steps": 0,
            }
            for key, expected in expected_save_contract.items():
                if t.get(key, None) != expected:
                    raise RuntimeError(
                        f"governed training requires training.{key}={expected!r}"
                    )
            if save_at_steps != {504}:
                raise RuntimeError(
                    "governed training permits only the step-504 checkpoint"
                )
        if formal_libero:
            expected_save_contract = {
                "keep_last_k": 1,
                "max_steps": formal_final_step,
                "save_initial_checkpoint": False,
                "save_steps": 0,
            }
            for key, expected in expected_save_contract.items():
                if t.get(key, None) != expected:
                    raise RuntimeError(
                        f"formal LIBERO training requires training.{key}={expected!r}"
                    )
            if save_at_steps:
                raise RuntimeError(
                    "formal LIBERO training permits only its final checkpoint"
                )
        if formal_phase6:
            output_path = self._phase6_launch_context.run_directory
            metrics_writer = self._create_phase6_metrics_writer()
        elif formal_afcc:
            output_path = os.path.abspath(os.fspath(t.get("output_dir")))
            if _rank() == 0:
                os.mkdir(output_path, 0o700)
                parent_descriptor = os.open(
                    os.path.dirname(output_path), os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)
                metrics_descriptor = os.open(
                    os.path.join(output_path, "afcc_step_metrics_v1.jsonl"),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o400,
                )
                os.fchmod(metrics_descriptor, 0o400)
                afcc_metrics_stream = os.fdopen(
                    metrics_descriptor, "w", encoding="utf-8"
                )
            else:
                afcc_metrics_stream = None
            metrics_writer = None
        elif formal_libero:
            output_path = self._validate_exact_output_dir(self._exact_output_dir)
            metrics_writer = None
            afcc_metrics_stream = None
        else:
            ts = time.strftime("%Y%m%d_%H%M%S") if not debug else "debug"
            output_path = os.path.join(str(t.get("output_dir", "outputs/ar_sana")), ts)
            if _rank() == 0:
                os.makedirs(output_path, exist_ok=True)
            metrics_writer = None
            afcc_metrics_stream = None
        if not formal_afcc:
            afcc_metrics_stream = None

        saved_steps: set[int] = set()
        if not governed_training and self._save_initial_checkpoint(output_path):
            saved_steps.add(0)

        if self._phase6_plan_sampler is not None:
            sampler = self._phase6_plan_sampler
        else:
            sampler = (
                DistributedSampler(self.dataset, shuffle=True, seed=self.base_seed)
                if _is_dist()
                else None
            )
        num_workers = int(t.get("num_workers", 0))
        # MixtureDataset rebuilds its virtual index at each epoch. Persistent
        # worker copies would not observe that main-process mutation.
        epoch_aware_dataset = callable(getattr(self.dataset, "set_epoch", None))
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and self._phase6_plan_sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=list,
            drop_last=True,
            persistent_workers=bool(num_workers > 0 and not epoch_aware_dataset),
            generator=self._data_generator,
        )

        model_groups = self._param_groups()
        params = [parameter for group in model_groups for parameter in group["params"]]
        named_trainables = self._phase6_named_trainables()
        if formal_phase6:
            self._validate_phase6_trainable_partition(named_trainables)
        elif formal_afcc:
            self._validate_afcc_trainable_partition(named_trainables)
        optimizer_groups, master_pairs = self._optimizer_param_groups(model_groups)
        if (governed_training or formal_libero) and (
            len(master_pairs) != len(params)
            or any(master.dtype != torch.float32 for _model, master in master_pairs)
        ):
            raise RuntimeError(
                "governed training requires one FP32 optimizer master per model parameter"
            )
        optimizer_foreach = self._optimizer_foreach_setting()
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=float(t.get("weight_decay", 0.0)),
            betas=(0.9, 0.95),
            foreach=optimizer_foreach,
        )
        base_lrs = [g["lr"] for g in optimizer.param_groups]

        if _rank() == 0:
            n_train = sum(p.numel() for p in params)
            logger.info(
                "Trainable params: %.1fM | output: %s | train_steps=%d | lr_schedule_steps=%d",
                n_train / 1e6,
                output_path,
                max_steps,
                lr_schedule_steps,
            )

        self._set_training_mode()
        if formal_libero:
            torch.cuda.reset_peak_memory_stats(self.device)
        step, epoch = 0, 0
        data_iter = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        for parameter in params:
            parameter.grad = None
        trajectory_state_counts: dict[int, int] = {}
        formal_libero_metrics: dict[str, Any] | None = None
        while step < max_steps:
            if governed_training:
                torch.cuda.reset_peak_memory_stats(self.device)
            if formal_phase6:
                observed_phase6_row = None
                adapter_gradient_connected = None
                nr_gradient_target_adapter_only = None
            for micro in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if self._phase6_plan_sampler is not None:
                        raise RuntimeError(
                            "Phase-6 504-row loader exhausted before max_steps"
                        )
                    epoch += 1
                    self._set_data_epoch(epoch, sampler)
                    data_iter = iter(loader)
                    batch = next(data_iter)
                result = self._compute_loss(
                    batch,
                    expected_phase6_step=(
                        step + 1 if self._phase6_plan is not None else None
                    ),
                )
                if formal_phase6:
                    observed_phase6_row = batch[0]["phase6_plan_row"]
                    (
                        adapter_gradient_connected,
                        nr_gradient_target_adapter_only,
                    ) = self._phase6_probe_nr_autograd(result, named_trainables)
                elif formal_afcc:
                    observed_afcc_row = batch[0]["phase6_plan_row"]
                    afcc_reference_metadata = (
                        self._action_facing_cache_reference_store
                        .audit_metadata_for_global_step(step + 1)
                    )
                    if (
                        observed_afcc_row["action_sigma"]
                        != afcc_reference_metadata["action_sigma"]
                        or observed_afcc_row["identity"]["task_name"]
                        != afcc_reference_metadata["task_name"]
                    ):
                        raise RuntimeError(
                            "AFCC runtime plan row differs from reference metadata"
                        )
                loss = result["loss"]
                lv = result.get("loss_video")
                la = result.get("loss_action")
                state_index = result.get("video_trajectory_supervised_state_index")
                if state_index is not None:
                    index = int(state_index)
                    if index >= 0:
                        trajectory_state_counts[index] = (
                            trajectory_state_counts.get(index, 0) + 1
                        )
                (loss / grad_accum).backward()
                if formal_libero:
                    # Preserve scalar diagnostics only; release the completed
                    # autograd graph before the FP32-master AdamW update.
                    formal_libero_metrics = {
                        key: (
                            value.detach() if isinstance(value, torch.Tensor) else value
                        )
                        for key, value in {
                            "loss": loss,
                            "loss_video": lv,
                            "loss_action": la,
                            "loss_video_on_path": result.get("loss_video_on_path"),
                            "loss_video_trajectory_endpoint": result.get(
                                "loss_video_trajectory_endpoint"
                            ),
                            "loss_video_trajectory_velocity": result.get(
                                "loss_video_trajectory_velocity"
                            ),
                            "loss_video_trajectory_consistency": result.get(
                                "loss_video_trajectory_consistency"
                            ),
                            "video_trajectory_supervised_state_index": result.get(
                                "video_trajectory_supervised_state_index"
                            ),
                        }.items()
                    }
                    del batch, loss, lv, la, result

            self._allreduce_grads(params)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if formal_phase6:
                gradient_measurements = self._phase6_gradient_measurements(
                    named_trainables,
                    adapter_gradient_connected=adapter_gradient_connected,
                    nr_gradient_target_adapter_only=(nr_gradient_target_adapter_only),
                )
            if master_pairs:
                self._sync_master_gradients(master_pairs)
            if formal_afcc:
                afcc_gradient_norms = []
                for parameter in params:
                    if parameter.grad is None or not bool(
                        torch.isfinite(parameter.grad).all().item()
                    ):
                        raise RuntimeError(
                            "formal AFCC requires finite gradients for every "
                            "trainable tensor"
                        )
                    afcc_gradient_norms.append(
                        float(
                            torch.linalg.vector_norm(
                                parameter.grad.detach().float()
                            ).item()
                        )
                    )
                afcc_gradient_global_norm = math.sqrt(
                    math.fsum(value * value for value in afcc_gradient_norms)
                )
            # cosine-with-warmup LR
            scale = self._lr_lambda(step, lr_schedule_steps)
            for g, blr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = blr * scale
            if formal_phase6:
                learning_rates = self._phase6_learning_rates(model_groups, optimizer)
                if not gradient_measurements["finite"]:
                    gradient_measurements["model_parameters_finite"] = (
                        self._phase6_tensors_finite(params)
                    )
                    gradient_measurements["optimizer_master_parameters_finite"] = (
                        self._phase6_tensors_finite(
                            [master for _model, master in master_pairs]
                        )
                    )
                    gradient_measurements["optimizer_state_finite"] = (
                        self._phase6_optimizer_state_finite(optimizer)
                    )
                    measurements = self._phase6_measurements(
                        result,
                        gradient_measurements,
                        learning_rates,
                        peak_reserved_bytes=max(
                            1, int(torch.cuda.max_memory_reserved(self.device))
                        ),
                    )
                    # append_step fsyncs the failure row before raising.
                    metrics_writer.append_step(
                        global_step=step + 1,
                        observed_plan_row=observed_phase6_row,
                        measurements=measurements,
                    )
                    raise AssertionError("non-finite Phase-6 gradients did not stop")
            optimizer.step()
            if master_pairs:
                self._copy_master_parameters_to_model(master_pairs)
            step += 1

            if formal_phase6:
                gradient_measurements["model_parameters_finite"] = (
                    self._phase6_tensors_finite(params)
                )
                gradient_measurements["optimizer_master_parameters_finite"] = (
                    self._phase6_tensors_finite(
                        [master for _model, master in master_pairs]
                    )
                )
                gradient_measurements["optimizer_state_finite"] = (
                    self._phase6_optimizer_state_finite(optimizer)
                )
                measurements = self._phase6_measurements(
                    result,
                    gradient_measurements,
                    learning_rates,
                    peak_reserved_bytes=max(
                        1, int(torch.cuda.max_memory_reserved(self.device))
                    ),
                )
                # Finiteness failures are durably appended before writer raises.
                metrics_writer.append_step(
                    global_step=step,
                    observed_plan_row=observed_phase6_row,
                    measurements=measurements,
                )
            elif formal_afcc:
                if not (
                    self._phase6_tensors_finite(params)
                    and self._phase6_tensors_finite(
                        [master for _model, master in master_pairs]
                    )
                    and self._phase6_optimizer_state_finite(optimizer)
                ):
                    raise RuntimeError(
                        "formal AFCC model/master/optimizer state became non-finite"
                    )
                afcc_record = {
                    "action_sigma": float(observed_afcc_row["action_sigma"]),
                    "afcc_loss": self._phase6_scalar(
                        result,
                        "loss_action_facing_cache_consistency",
                        minimum=0.0,
                    ),
                    "common_input_trace_sha256": result[
                        "phase6_common_input_trace_sha256"
                    ],
                    "global_step": step,
                    "gradient_global_norm": afcc_gradient_global_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "peak_reserved_bytes": max(
                        1, int(torch.cuda.max_memory_reserved(self.device))
                    ),
                    "plan_row_sha256": afcc_reference_metadata[
                        "plan_row_sha256"
                    ],
                    "schema_version": "sana-phase6-afcc-step-metrics-v1",
                    "task_name": observed_afcc_row["identity"]["task_name"],
                    "total_loss": self._phase6_scalar(
                        result, "loss", minimum=0.0
                    ),
                    "video_local_expansion_loss": self._phase6_scalar(
                        result, "loss_video_local_expansion", minimum=0.0
                    ),
                    "video_on_path_loss": self._phase6_scalar(
                        result, "loss_video_on_path", minimum=0.0
                    ),
                }
                afcc_metrics_stream.write(
                    json.dumps(
                        afcc_record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                afcc_metrics_stream.flush()
                os.fsync(afcc_metrics_stream.fileno())

            optimizer.zero_grad(set_to_none=True)
            for parameter in params:
                parameter.grad = None

            if _rank() == 0 and (step % 10 == 0 or step == 1):
                if formal_libero:
                    if formal_libero_metrics is None:
                        raise RuntimeError("formal LIBERO metrics were not captured")
                    loss = formal_libero_metrics["loss"]
                    lv = formal_libero_metrics["loss_video"]
                    la = formal_libero_metrics["loss_action"]
                    diagnostic_result = formal_libero_metrics
                else:
                    diagnostic_result = result
                logger.info(
                    "step %d/%d  loss=%.4f  video=%.4f  action=%.4f  lr=%.2e",
                    step,
                    max_steps,
                    float(loss),
                    float(lv) if lv is not None else 0.0,
                    float(la) if la is not None else 0.0,
                    optimizer.param_groups[0]["lr"],
                )
                endpoint = diagnostic_result.get("loss_video_trajectory_endpoint")
                if endpoint is not None:
                    logger.info(
                        "trajectory components  on_path=%.4f  endpoint=%.4f  "
                        "velocity=%.4f  consistency=%.4f  state=%d",
                        float(diagnostic_result.get("loss_video_on_path", 0.0)),
                        float(endpoint),
                        float(
                            diagnostic_result.get(
                                "loss_video_trajectory_velocity", 0.0
                            )
                        ),
                        float(
                            diagnostic_result.get(
                                "loss_video_trajectory_consistency", 0.0
                            )
                        ),
                        int(
                            diagnostic_result.get(
                                "video_trajectory_supervised_state_index", -1
                            )
                        ),
                    )

            if not formal_phase6 and not formal_libero and self._checkpoint_due(
                step, save_steps, save_at_steps
            ):
                self._save(output_path, step)
                saved_steps.add(step)

        if formal_phase6:
            if metrics_writer.rows_written != 504 or step != 504:
                raise RuntimeError("formal Phase-6 did not complete exactly 504 rows")
            self._save(output_path, 504)
            self._finalize_phase6_integrity(metrics_writer, output_path)
        elif formal_afcc:
            if step != 504 or saved_steps != {504}:
                raise RuntimeError(
                    "formal AFCC did not complete exactly 504 rows and one save"
                )
            afcc_metrics_stream.flush()
            os.fsync(afcc_metrics_stream.fileno())
            afcc_metrics_stream.close()
        elif formal_libero:
            if step != formal_final_step or saved_steps:
                if formal_final_step == 2000:
                    raise RuntimeError(
                        "formal LIBERO training did not complete exactly 2000 steps "
                        "without an intermediate save"
                    )
                raise RuntimeError(
                    f"formal LIBERO training did not complete exactly "
                    f"{formal_final_step} steps without an intermediate save"
                )
            if formal_libero_metrics is None:
                raise RuntimeError("formal LIBERO final metrics are missing")
            final_loss_tensor = formal_libero_metrics["loss"].detach().float().reshape(1)
            finite_values = [
                math.isfinite(float(final_loss_tensor.item())),
                self._phase6_tensors_finite(params),
                self._phase6_tensors_finite(
                    [master for _model, master in master_pairs]
                ),
                self._phase6_optimizer_state_finite(optimizer),
            ]
            peak_tensor = torch.tensor(
                [int(torch.cuda.max_memory_reserved(self.device))],
                dtype=torch.int64,
                device=self.device,
            )
            if _is_dist():
                dist.all_reduce(final_loss_tensor, op=dist.ReduceOp.SUM)
                final_loss_tensor /= _world_size()
                finite_tensor = torch.tensor(
                    finite_values, dtype=torch.int32, device=self.device
                )
                dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
                finite_values = [bool(value) for value in finite_tensor.tolist()]
                dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
            final_loss = float(final_loss_tensor.item())
            finite_state = {
                "final_loss_finite": finite_values[0],
                "model_parameters_finite": finite_values[1],
                "optimizer_master_parameters_finite": finite_values[2],
                "optimizer_state_finite": finite_values[3],
            }
            if not all(finite_state.values()):
                raise RuntimeError(
                    "formal LIBERO final model/master/optimizer state is non-finite"
                )
            final_model_state = self._complete_model_state_digest_consensus()
            self.formal_libero_run_summary = {
                **finite_state,
                "final_loss": final_loss,
                "final_learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
                "optimizer_steps": step,
                "optimizer_foreach": optimizer_foreach,
                "autograd_graph_released_before_optimizer": True,
                "peak_memory_reserved_bytes": int(peak_tensor.item()),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in params
                ),
                "trainable_parameter_tensor_count": len(params),
                "world_size": _world_size(),
                "per_device_batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "global_batch_size": batch_size * grad_accum * _world_size(),
                "dataset_windows": len(self.dataset),
                "window_draws": step * batch_size * grad_accum * _world_size(),
                "rank_process_seeds": [
                    self.base_seed + rank for rank in range(_world_size())
                ],
                "distributed_initialization": getattr(
                    self,
                    "distributed_initialization_summary",
                    {"broadcast": False, "world_size": 1},
                ),
                "final_model_state": final_model_state,
            }
            self._save(output_path, formal_final_step)
        # Legacy final state is always present without a duplicate large file.
        elif step not in saved_steps:
            self._save(output_path, step)
        if _rank() == 0 and trajectory_state_counts:
            logger.info(
                "Trajectory supervised state counts: %s",
                dict(sorted(trajectory_state_counts.items())),
            )
        if _is_dist():
            dist.barrier()
        return output_path

    # ------------------------------------------------------------------ save
    def _save(self, output_path: str, step: int):
        if _rank() != 0:
            return
        if self._post_primary_afcc_authorized:
            self._save_post_primary_afcc(output_path, step)
            return
        ckpt = os.path.join(output_path, f"checkpoint_step_{step}.safetensors")
        if self._exact_output_dir is not None and os.path.lexists(ckpt):
            raise FileExistsError(
                f"formal LIBERO checkpoint already exists: {ckpt}"
            )
        save_config(output_path, self.cfg)
        self.architecture.save_checkpoint(ckpt)
        if self._phase6_launch_context is None:
            save_action_stats(output_path, self.dataset)
        else:
            from sana_wam.train.phase6_run_integrity import read_stable_pinned_file

            action_stats = read_stable_pinned_file(
                self._verified_action_stats_path,
                self._verified_action_stats_sha256,
                "Phase-6 action stats",
            )
            self._exclusive_write_bytes(
                os.path.join(output_path, "action_stats.npy"), action_stats
            )
        self.architecture.copy_deploy_artifacts(output_path, self.cfg)
        manage_checkpoints(output_path, int(self.t.get("keep_last_k", 2)))
        logger.info("Saved checkpoint: %s", ckpt)

    def _save_post_primary_afcc(self, output_path: str, step: int) -> None:
        """Publish the single AFCC checkpoint without an overwrite-capable path."""

        from omegaconf import OmegaConf

        output = Path(output_path)
        checkpoint = output / f"checkpoint_step_{step}.safetensors"
        config_path = output / "config.yaml"
        action_stats_path = output / "action_stats.npy"
        self._exclusive_write_bytes(
            config_path,
            OmegaConf.to_yaml(self.cfg).encode("utf-8"),
            mode=0o400,
        )
        self._exclusive_write_bytes(
            action_stats_path,
            Path(self._verified_action_stats_path).read_bytes(),
            mode=0o400,
        )

        temporary = output / (
            f".{checkpoint.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        if temporary.exists() or temporary.is_symlink() or checkpoint.exists():
            raise FileExistsError("AFCC checkpoint publication path already exists")
        self.architecture.save_checkpoint(str(temporary))
        os.chmod(temporary, 0o400)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, checkpoint)
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        temporary.unlink()
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.architecture.copy_deploy_artifacts(output_path, self.cfg)
        logger.info("Saved immutable AFCC checkpoint: %s", checkpoint)
