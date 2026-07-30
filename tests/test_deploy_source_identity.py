from __future__ import annotations

import hashlib
import json

from scripts.deploy import _deployment_source_identity


def test_deployment_source_identity_hashes_exact_config_and_ordered_overrides(
    tmp_path,
):
    config = tmp_path / "deploy.yaml"
    raw = b"policy:\n  history_len: 113\n"
    config.write_bytes(raw)
    overrides = ["inference.seed=7", "policy.execute_horizon=null"]

    identity = _deployment_source_identity(str(config), overrides)

    encoded_overrides = json.dumps(
        overrides, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    assert identity == {
        "deploy_config_path": str(config.resolve()),
        "deploy_config_size": len(raw),
        "deploy_config_sha256": hashlib.sha256(raw).hexdigest(),
        "deploy_overrides": overrides,
        "deploy_overrides_sha256": hashlib.sha256(encoded_overrides).hexdigest(),
    }
