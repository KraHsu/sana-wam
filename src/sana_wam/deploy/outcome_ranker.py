"""Fail-closed loader and scorer for generation-zero action outcome rankers."""

from __future__ import annotations

import hashlib
import hmac
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RANKER_SCHEMA_VERSION = 1
ACTION_TOKENS = 28
ACTION_DIM = 20
FEATURE_DIM = 180


def summary_action_features(actions: np.ndarray) -> np.ndarray:
    """Build the nine temporal summaries used to train the outcome ranker."""
    values = np.asarray(actions)
    squeezed = values.ndim == 2
    if squeezed:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (ACTION_TOKENS, ACTION_DIM):
        raise ValueError(
            "expected actions shaped [candidate,28,20] or [28,20], got "
            f"{values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("actions contain non-finite values")
    delta = np.diff(values, axis=1)
    summaries = (
        values[:, 0],
        values[:, -1],
        values.mean(axis=1),
        values.std(axis=1),
        values.min(axis=1),
        values.max(axis=1),
        values[:, -1] - values[:, 0],
        np.abs(delta).sum(axis=1),
        np.abs(delta).max(axis=1),
    )
    features = np.concatenate(summaries, axis=1).astype(np.float64)
    return features[0] if squeezed else features


def _scalar_int(arrays, name: str) -> int:
    value = np.asarray(arrays[name])
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"ranker {name} must be an integer scalar")
    return int(value)


@dataclass(frozen=True)
class OutcomeActionRanker:
    """Immutable linear ranker over normalized generation-zero action chunks."""

    path: str
    sha256: str
    schema_version: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight: np.ndarray

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str) -> "OutcomeActionRanker":
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("expected_ranker_sha256 must be 64 lowercase hex characters")

        resolved = Path(path).expanduser().resolve(strict=True)
        payload = resolved.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(actual, expected):
            raise ValueError(
                f"ranker SHA256 mismatch: expected {expected}, got {actual}"
            )

        required = {
            "schema_version",
            "feature_mean",
            "feature_scale",
            "weight",
            "action_tokens",
            "action_dim",
            "feature_dim",
        }
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(f"ranker is missing arrays: {sorted(missing)}")
            schema_version = _scalar_int(arrays, "schema_version")
            action_tokens = _scalar_int(arrays, "action_tokens")
            action_dim = _scalar_int(arrays, "action_dim")
            feature_dim = _scalar_int(arrays, "feature_dim")
            feature_mean = np.asarray(arrays["feature_mean"], dtype=np.float64).copy()
            feature_scale = np.asarray(arrays["feature_scale"], dtype=np.float64).copy()
            weight = np.asarray(arrays["weight"], dtype=np.float64).copy()

        if schema_version != RANKER_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ranker schema_version={schema_version}; "
                f"expected {RANKER_SCHEMA_VERSION}"
            )
        geometry = (action_tokens, action_dim, feature_dim)
        expected_geometry = (ACTION_TOKENS, ACTION_DIM, FEATURE_DIM)
        if geometry != expected_geometry:
            raise ValueError(
                f"ranker geometry {geometry} does not match {expected_geometry}"
            )
        for name, value in (
            ("feature_mean", feature_mean),
            ("feature_scale", feature_scale),
            ("weight", weight),
        ):
            if value.shape != (FEATURE_DIM,):
                raise ValueError(
                    f"ranker {name} has shape {value.shape}, expected {(FEATURE_DIM,)}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"ranker {name} contains non-finite values")
            value.setflags(write=False)
        if np.any(feature_scale <= 0):
            raise ValueError("ranker feature_scale must be strictly positive")
        if np.linalg.norm(weight) <= 0:
            raise ValueError("ranker weight must be non-zero")

        return cls(
            path=str(resolved),
            sha256=actual,
            schema_version=schema_version,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            weight=weight,
        )

    def score(self, actions: np.ndarray) -> float | np.ndarray:
        features = summary_action_features(actions)
        scores = ((features - self.feature_mean) / self.feature_scale) @ self.weight
        if not np.isfinite(scores).all():
            raise ValueError("ranker produced a non-finite score")
        return float(scores) if np.ndim(scores) == 0 else scores

    @property
    def identity(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "sha256": self.sha256,
            "action_tokens": ACTION_TOKENS,
            "action_dim": ACTION_DIM,
            "feature_dim": FEATURE_DIM,
        }
