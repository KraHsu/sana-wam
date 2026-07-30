#!/usr/bin/env python
"""Serve a trained SANA block-AR checkpoint as a policy server.

    python scripts/deploy.py --ckpt-dir outputs/ar_2task/<ts> [--device cuda:0]
        [--deploy-config configs/deploy_ar_sana.yaml] [key=value ...]

Loads the checkpoint (``load_from_checkpoint_dir``), builds the closed-loop
``ARInferenceEngine`` via ``build_engine``, wraps it in a greedy ``WAMPolicy``,
and serves HTTP (/predict, /reset, /health, /info) + WebSocket. The RoboTwin
eval client connects over HTTP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

# GDN scan helpers use ``@torch.compile``; torch 2.7.1 inductor ↔ triton 3.5.1
# ABI break (triton_key removed). Disable the compile wrapper; bare-triton GDN
# kernels are unaffected. No-op for the linear_relu path. Before diffusion.*.
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party" / "Sana"))

from sana_wam.deploy.policy_server import build_server_from_config  # noqa: E402


def _deployment_source_identity(config_path: str, overrides: list[str]) -> dict:
    resolved = Path(config_path).expanduser().resolve()
    raw = resolved.read_bytes()
    override_bytes = json.dumps(
        list(overrides), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "deploy_config_path": str(resolved),
        "deploy_config_size": len(raw),
        "deploy_config_sha256": hashlib.sha256(raw).hexdigest(),
        "deploy_overrides": list(overrides),
        "deploy_overrides_sha256": hashlib.sha256(override_bytes).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-name", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deploy-config", default=str(_ROOT / "configs" / "deploy_ar_sana.yaml"))
    ap.add_argument("--host", default=None)
    ap.add_argument("--ws-port", type=int, default=None)
    ap.add_argument("--http-port", type=int, default=None)
    args, overrides = ap.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = OmegaConf.merge(OmegaConf.load(args.deploy_config), OmegaConf.from_dotlist(overrides))

    ckpt_name = args.ckpt_name or OmegaConf.select(cfg, "checkpoint_name", default=None)
    server = build_server_from_config(
        cfg,
        args.ckpt_dir,
        device=args.device,
        ckpt_name=ckpt_name,
    )
    server._deployment_identity.update(
        _deployment_source_identity(args.deploy_config, overrides)
    )

    host = args.host or OmegaConf.select(cfg, "server.host", default="0.0.0.0")
    ws_port = args.ws_port or OmegaConf.select(cfg, "server.ws_port", default=8850)
    http_port = args.http_port or OmegaConf.select(cfg, "server.http_port", default=8848)
    logging.getLogger(__name__).info("Serving on %s (ws=%s http=%s)", host, ws_port, http_port)
    server.run(host=host, port=int(ws_port), http_port=int(http_port))


if __name__ == "__main__":
    main()
