#!/usr/bin/env python
"""Start the strict LIBERO SANA-WAM policy server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party" / "Sana"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Deploy-side YAML")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host")
    parser.add_argument("--ws-port", type=int)
    parser.add_argument("--http-port", type=int)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Deploy-only OmegaConf dotlist overrides",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if "defaults" in cfg:
        OmegaConf.update(cfg, "defaults", OmegaConf.create([]), merge=False)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    from sana_wam.deploy.libero_policy_server import build_libero_server_from_config

    server = build_libero_server_from_config(
        cfg,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        ckpt_name=args.ckpt_name,
    )
    server_cfg = OmegaConf.select(cfg, "server", default=None)
    if server_cfg is None:
        server_cfg = OmegaConf.select(cfg, "deploy.server", default={})
    host = args.host or OmegaConf.select(server_cfg, "host", default="0.0.0.0")
    ws_port = args.ws_port or OmegaConf.select(server_cfg, "ws_port", default=8850)
    http_port = args.http_port or OmegaConf.select(
        server_cfg, "http_port", default=8848
    )
    server.run(host=host, port=ws_port, http_port=http_port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
