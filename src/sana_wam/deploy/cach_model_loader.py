"""Dedicated CACH checkpoint-loader boundary.

Stage 1 defines the descriptor schema but intentionally has no torch/model
loading backend.  This keeps CACH out of the legacy latest-glob and permissive
missing-key loader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from sana_wam.cach.authority import CACHAuthorityError

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_BASENAME = re.compile(
    r"checkpoint_step_(0|[1-9][0-9]*)\.safetensors\Z"
)


@dataclass(frozen=True, slots=True)
class CACHDeployCheckpointDescriptor:
    candidate_revision: str
    endpoint_step: int
    checkpoint_absolute_path: str
    checkpoint_raw_sha256: str
    checkpoint_schema_sha256: str
    resolved_config_sha256: str
    source_manifest_sha256: str
    engine_schema_sha256: str

    def __post_init__(self) -> None:
        if type(self.candidate_revision) is not str or not self.candidate_revision:
            raise ValueError("candidate_revision must be non-empty")
        if type(self.endpoint_step) is not int or self.endpoint_step < 0:
            raise ValueError("endpoint_step must be a non-negative integer")
        if (
            type(self.checkpoint_absolute_path) is not str
            or not self.checkpoint_absolute_path
        ):
            raise ValueError("checkpoint path must be canonical and absolute")
        path = PurePosixPath(self.checkpoint_absolute_path)
        if (
            not path.is_absolute()
            or self.checkpoint_absolute_path == "/"
            or "//" in self.checkpoint_absolute_path
            or path.as_posix() != self.checkpoint_absolute_path
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("checkpoint path must be canonical and absolute")
        if any(char in self.checkpoint_absolute_path for char in "*?[]{}"):
            raise ValueError("checkpoint path cannot contain a glob")
        if any(
            "latest"
            in "".join(character for character in part.lower() if character.isalnum())
            for part in path.parts
        ):
            raise ValueError("checkpoint path cannot select latest")
        basename = _CHECKPOINT_BASENAME.fullmatch(path.name)
        if basename is None:
            raise ValueError(
                "checkpoint basename must be "
                "checkpoint_step_<endpoint_step>.safetensors"
            )
        if int(basename.group(1)) != self.endpoint_step:
            raise ValueError("checkpoint path endpoint step differs")
        for name in (
            "checkpoint_raw_sha256",
            "checkpoint_schema_sha256",
            "resolved_config_sha256",
            "source_manifest_sha256",
            "engine_schema_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be one lowercase SHA256")


def load_cach_from_descriptor(
    descriptor: CACHDeployCheckpointDescriptor,
    *,
    authority: object,
):
    """Hard-deny until Stage 2+ commissions the exact external-source loader."""

    del descriptor, authority
    raise CACHAuthorityError(
        "CACH model loading is not implemented or authorized in Stage 1"
    )


__all__ = [
    "CACHDeployCheckpointDescriptor",
    "load_cach_from_descriptor",
]
