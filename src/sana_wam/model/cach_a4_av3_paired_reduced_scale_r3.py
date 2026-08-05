"""AV3-R3 adapter for allocator initialization before peak-stat reset.

The frozen AV3 execution reaches ``torch.cuda.reset_peak_memory_stats``
before PyTorch's CUDA caching allocator has been initialized.  On the pinned
H200 PyTorch build that call fails with ``Invalid device argument``.  This
additive adapter preserves the frozen R2 loader and frozen base execution,
and scopes one shim to the existing reset call: initialize CUDA, then invoke
the original reset function with the original device argument.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, NoReturn


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
R2_SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale_r2.py"
R2_SOURCE_SHA256 = "d069c3af8a324842cdb8b9f3bdc1cc6d052c3e16e7398de5cd5290fcb8852d75"


class R3SourceContractError(RuntimeError):
    """Raised when the frozen R2 source or the scoped reset seam differs."""


def _fail(message: str) -> NoReturn:
    raise R3SourceContractError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_r2_source() -> ModuleType:
    metadata = R2_SOURCE_PATH.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("frozen AV3-R2 source is not a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        _fail("frozen AV3-R2 source mode differs from 0444")
    if _sha(R2_SOURCE_PATH) != R2_SOURCE_SHA256:
        _fail("frozen AV3-R2 source SHA differs")
    name = "_cach_a4_av3_paired_reduced_scale_frozen_r2_source_for_r3"
    spec = importlib.util.spec_from_file_location(name, R2_SOURCE_PATH)
    if spec is None or spec.loader is None:
        _fail("cannot construct frozen AV3-R2 source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_r2 = _load_frozen_r2_source()

AV3ExecutionError = _r2.AV3ExecutionError
AV3RealDataBundle = _r2.AV3RealDataBundle
assert_av3_train_holdout_disjoint = _r2.assert_av3_train_holdout_disjoint
load_av3_real_data = _r2.load_av3_real_data
assess_av3_data_adequacy = _r2.assess_av3_data_adequacy
validate_av3_config = _r2.validate_av3_config
compute_av3_metrics = _r2.compute_av3_metrics
classify_av3 = _r2.classify_av3

_REAL_TORCH = _r2._base.torch


class _CudaProxy:
    """Delegate CUDA except for the one registered frozen reset call."""

    def __init__(self, real_cuda: object, expected_caller_code: object) -> None:
        self._real_cuda = real_cuda
        self._expected_caller_code = expected_caller_code
        self.reset_calls = 0

    def reset_peak_memory_stats(self, device: object = None) -> None:
        caller_code = sys._getframe(1).f_code
        if caller_code is not self._expected_caller_code:
            _fail("AV3-R3 reset seam was reached by an unregistered caller")
        self.reset_calls += 1
        if self.reset_calls != 1:
            _fail("frozen AV3 execution called peak-memory reset more than once")
        self._real_cuda.init()
        self._real_cuda.reset_peak_memory_stats(device)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_cuda, name)


class _TorchProxy:
    """Module-local proxy; no process-global torch attribute is mutated."""

    def __init__(self, real_torch: object, cuda_proxy: _CudaProxy) -> None:
        self._real_torch = real_torch
        self.cuda = cuda_proxy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_torch, name)


def run_av3_paired_reduced_scale(
    config: Mapping[str, object],
    *,
    expected_gpu_uuid: str,
    total_deadline_monotonic: float,
    source_data_and_predecessor_pins_match: bool,
    manifest: Mapping[str, object],
    manifest_path: str | os.PathLike[str],
    theta0_callback: Callable[[Mapping[str, object]], None],
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    preloaded_bundle: AV3RealDataBundle | None = None,
) -> Mapping[str, object]:
    """Execute frozen R2 with one scoped init-before-reset shim."""

    base = _r2._base
    if base.torch is not _REAL_TORCH:
        _fail("frozen AV3 torch module seam differs before AV3-R3 execution")
    cuda_proxy = _CudaProxy(
        _REAL_TORCH.cuda,
        base.run_av3_paired_reduced_scale.__code__,
    )
    torch_proxy = _TorchProxy(_REAL_TORCH, cuda_proxy)
    try:
        base.torch = torch_proxy
        result = _r2.run_av3_paired_reduced_scale(
            config,
            expected_gpu_uuid=expected_gpu_uuid,
            total_deadline_monotonic=total_deadline_monotonic,
            source_data_and_predecessor_pins_match=(
                source_data_and_predecessor_pins_match
            ),
            manifest=manifest,
            manifest_path=manifest_path,
            theta0_callback=theta0_callback,
            progress_callback=progress_callback,
            preloaded_bundle=preloaded_bundle,
        )
        if cuda_proxy.reset_calls != 1:
            _fail("frozen AV3 execution did not cross the registered reset seam")
        if base.torch is not torch_proxy:
            _fail("frozen AV3 torch module seam changed during AV3-R3 execution")
        return result
    finally:
        base.torch = _REAL_TORCH


def __getattr__(name: str) -> Any:
    return getattr(_r2, name)


__all__ = [
    "AV3ExecutionError",
    "AV3RealDataBundle",
    "assert_av3_train_holdout_disjoint",
    "assess_av3_data_adequacy",
    "classify_av3",
    "compute_av3_metrics",
    "load_av3_real_data",
    "run_av3_paired_reduced_scale",
    "validate_av3_config",
]
