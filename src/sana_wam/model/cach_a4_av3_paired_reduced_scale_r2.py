"""AV3-R2 source adapter with one corrected 16+16 disjointness contract.

The frozen AV3 source correctly materializes two train and two held-out
microbatches of eight windows each, but then passes the flattened 16+16
cohort to an AV2-only helper that intentionally rejects any cardinality other
than 8+8.  This additive adapter reuses every frozen loader/model/training/
metric implementation and replaces only that final disjointness assertion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, NoReturn


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
BASE_SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale.py"
BASE_SOURCE_SHA256 = "bafdb5475e1d7dddfba6d93e68ab2b67bf3960dba613da09ac26c48c8627bb6c"


class R2SourceContractError(RuntimeError):
    """Raised when the frozen source predecessor differs."""


def _fail(message: str) -> NoReturn:
    raise R2SourceContractError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_base_source() -> ModuleType:
    metadata = BASE_SOURCE_PATH.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("frozen AV3 base source is not a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        _fail("frozen AV3 base source mode differs from 0444")
    if _sha(BASE_SOURCE_PATH) != BASE_SOURCE_SHA256:
        _fail("frozen AV3 base source SHA differs")
    name = "_cach_a4_av3_paired_reduced_scale_frozen_base_for_r2"
    spec = importlib.util.spec_from_file_location(name, BASE_SOURCE_PATH)
    if spec is None or spec.loader is None:
        _fail("cannot construct frozen AV3 base source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_frozen_base_source()

AV3ExecutionError = _base.AV3ExecutionError
AV3RealDataBundle = _base.AV3RealDataBundle
AV3_DATA_MANIFEST_SCHEMA = _base.AV3_DATA_MANIFEST_SCHEMA
AV3_TRAIN_MICROBATCH_EPISODES = _base.AV3_TRAIN_MICROBATCH_EPISODES
AV3_HELDOUT_MICROBATCH_EPISODES = _base.AV3_HELDOUT_MICROBATCH_EPISODES
AV3_EXCLUDED_AV2_EPISODES = _base.AV3_EXCLUDED_AV2_EPISODES


def assert_av3_train_holdout_disjoint(
    train: Sequence[object], heldout: Sequence[object]
) -> None:
    """Validate the registered flattened AV3 cohort at its native 16+16 size."""

    if len(train) != 16 or len(heldout) != 16:
        raise AV3ExecutionError(
            "AV-3 requires exactly 16 train and 16 held-out windows"
        )
    train_set = set(train)
    heldout_set = set(heldout)
    if len(train_set) != 16 or len(heldout_set) != 16:
        raise AV3ExecutionError("AV-3 split contains duplicate window identity")
    for train_group in (train[:8], train[8:]):
        for heldout_group in (heldout[:8], heldout[8:]):
            _base.av2.av2_scaffold.assert_train_holdout_disjoint(
                train_group,
                heldout_group,
            )


def load_av3_real_data(
    dataset_root: str | os.PathLike[str],
    manifest: Mapping[str, object],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    expected_manifest_sha256: str | None = None,
) -> AV3RealDataBundle:
    """Run the frozen AV3 loader with only the corrected 16+16 assertion."""

    root = Path(dataset_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise AV3ExecutionError("dataset_root must resolve to a directory")
    value = _base._require_mapping(manifest, label="manifest")
    if value.get("schema") != AV3_DATA_MANIFEST_SCHEMA:
        raise AV3ExecutionError("AV-3 data manifest schema differs")
    expected_selection_sha = _base._verify_manifest_selection(
        value,
        dataset_root=str(root),
    )

    manifest_sha = _base._canonical_sha256(value)
    if expected_manifest_sha256 is not None:
        if not _base._is_sha256(expected_manifest_sha256) or manifest_path is None:
            raise AV3ExecutionError("manifest file pin is incomplete")
        resolved_manifest_path = Path(manifest_path).resolve(strict=True)
        if resolved_manifest_path.read_bytes() != _base._canonical_bytes(value):
            raise AV3ExecutionError("AV-3 manifest file is not canonical JSON+LF")
        observed_manifest_sha = _base._file_sha256(resolved_manifest_path)
        if observed_manifest_sha != expected_manifest_sha256:
            raise AV3ExecutionError("AV-3 manifest file SHA differs")
        manifest_sha = observed_manifest_sha

    verified_files = _base._verify_selected_files(
        root,
        value.get("selected_files"),
    )
    derived = _base._derived_manifest_table(value.get("derived_microbatches"))
    train = tuple(
        _base.av2._load_split(root, name="train", episode_ids=episode_ids)
        for episode_ids in AV3_TRAIN_MICROBATCH_EPISODES
    )
    heldout = tuple(
        _base.av2._load_split(root, name="heldout", episode_ids=episode_ids)
        for episode_ids in AV3_HELDOUT_MICROBATCH_EPISODES
    )
    if len(train) != 2 or len(heldout) != 2:
        raise AV3ExecutionError("AV-3 microbatch count differs")
    for split_name, microbatches in (("train", train), ("heldout", heldout)):
        for microbatch_index, split in enumerate(microbatches):
            _base.av2._validate_real_split(split)
            expected_row = derived[(split_name, microbatch_index)]
            observed_tensor_sha = {
                **dict(split.tensor_sha256),
                "action_valid_mask": _base.av2._tensor_sha256(
                    split.action_valid_mask
                ),
                "frame_valid_mask": _base.av2._tensor_sha256(
                    split.frame_valid_mask
                ),
            }
            if observed_tensor_sha != dict(expected_row["tensor_sha256"]):
                raise AV3ExecutionError("derived tensor SHA differs from manifest")
            if dict(split.episode_file_sha256) != {
                path: verified_files[path] for path in split.episode_file_sha256
            }:
                raise AV3ExecutionError("episode file SHA differs from manifest")

    train_windows = tuple(window for split in train for window in split.windows)
    heldout_windows = tuple(
        window for split in heldout for window in split.windows
    )
    assert_av3_train_holdout_disjoint(train_windows, heldout_windows)
    observed_ids = {
        int(window.episode_id) for window in (*train_windows, *heldout_windows)
    }
    expected_ids = set(range(16, 48))
    if observed_ids != expected_ids or observed_ids & set(AV3_EXCLUDED_AV2_EPISODES):
        raise AV3ExecutionError("AV-3/AV-2 episode identity contract differs")
    return AV3RealDataBundle(
        dataset_root=str(root),
        train_microbatches=(train[0], train[1]),
        heldout_microbatches=(heldout[0], heldout[1]),
        selection_sha256=expected_selection_sha,
        manifest_sha256=manifest_sha,
    )


assess_av3_data_adequacy = _base.assess_av3_data_adequacy
validate_av3_config = _base.validate_av3_config
compute_av3_metrics = _base.compute_av3_metrics
classify_av3 = _base.classify_av3


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
    """Delegate the frozen run, forcing the corrected loader if not preloaded."""

    bundle = preloaded_bundle
    if bundle is None:
        # Preserve the frozen entry point's deny-before-data ordering.  The
        # delegated base entry point validates again, deliberately keeping
        # its own contract intact.
        _base.validate_av3_config(config)
        data = _base._config_mapping(config, "data_contract")
        bundle = load_av3_real_data(
            str(data["dataset_root"]),
            manifest,
            manifest_path=manifest_path,
            expected_manifest_sha256=str(data["manifest_sha256"]),
        )
    return _base.run_av3_paired_reduced_scale(
        config,
        expected_gpu_uuid=expected_gpu_uuid,
        total_deadline_monotonic=total_deadline_monotonic,
        source_data_and_predecessor_pins_match=source_data_and_predecessor_pins_match,
        manifest=manifest,
        manifest_path=manifest_path,
        theta0_callback=theta0_callback,
        progress_callback=progress_callback,
        preloaded_bundle=bundle,
    )


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


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
