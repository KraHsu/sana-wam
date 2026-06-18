# sana-wam

Standalone, simplified extraction of the **SANA block-autoregressive
(diffusion-forcing) world-action model** (`dual_system_autoregressive`) from
OpenWAM. Keeps only the AR path; drops the multi-backbone / registry / mega-base
abstractions. Modern uv + `pyproject.toml` layout.

A SANA-Video 2B linear-attention DiT denoiser + a dual-stream ActionDiT, coupled
by joint attention, trained as a block-causal diffusion-forcing model (LingBot-VA
topology) over a Wan2.1 causal-VAE latent space. Per-chunk proprioception (F4) +
chunk-0 bootstrap (F3).

## Layout
```
src/sana_wam/
  model/        architecture.py (DualSystemAR + parent), base.py (slim),
                ar/ (linear-attn kernel + drivers + inference cache),
                action_backbone/, video_backbone/{adapter, sana/, wan/vae}
  dataloader/   robotwin_dataset + transforms + stats
  train/        trainer.py (minimal torchrun DDP), checkpointing.py
  deploy/       ar_engine, model_loader, policy, policy_server
  config.py     flatten_model_cfg (replaces the registry)
configs/        train_ar_sana.yaml, deploy_ar_sana.yaml
scripts/        train.py, deploy.py, smoke_ar.sh, train_ar.sh
benchmarks/robotwin/   RoboTwin online-eval client
third_party/Sana       git submodule (upstream SANA `diffusion` package)
tests/          AR unit tests (CPU mini-backbone; a few real-weight gated)
```

## Setup
```bash
git submodule update --init third_party/Sana   # NVlabs/Sana @ 6554c8d
uv sync --extra dev
```
Weights (H200 box): SANA-Video 2B at `/DATA/share/SANA-Video_2B_480p`, Gemma text
encoder at `/DATA/share/gemma-2-2b-it`, RoboTwin data at `/DATA/share/RoboTwin2.0/dataset`.

## Test (CPU; needs the submodule, not the 2B weights)
```bash
uv run pytest -m "not gpu and not data" tests/    # 98 passed
```

## Train
```bash
bash scripts/smoke_ar.sh 0                          # single-GPU smoke (20 steps)
NPROC_PER_NODE=8 bash scripts/train_ar.sh           # 8-GPU 2-task run
```
Writes `outputs/<run>/{config.yaml, checkpoint_step_N.safetensors, action_stats.npy}`.
`NCCL_NVLS_ENABLE=0` is set in the scripts (required for >2-GPU NCCL init on this box).

AR data constraint: `num_frames=49` → causal-VAE latent T=4, divisible by
`ar_frame_chunk_size=2`. The default 33 (latent T=3) breaks AR chunking.

## Deploy + eval
```bash
uv run python scripts/deploy.py --ckpt-dir outputs/<run> --device cuda:0
# then drive it from the RoboTwin simulator (conda RoboTwin env):
bash benchmarks/robotwin/single_eval.sh adjust_bottle demo_clean openwam <gpu> 8848 127.0.0.1
```
Greedy policy is mandatory (the AR KV cache is stateful closed-loop; async /
receding-horizon / temporal-ensemble break cache alignment).

## What was simplified vs OpenWAM
- **Registry dropped** — one architecture / backbone / dataset, direct construction
  (`config.flatten_model_cfg` replaces `resolve_architecture_config`).
- **base.py** — only the AR-needed slice (proprio, prepare_inputs collation,
  freeze, checkpoint I/O); the Wan/Cosmos/encoder/lerobot factories are gone.
- **Trainer** — a ~250-line torchrun DDP + bf16 loop (manual grad all-reduce)
  replaces the 1143-line accelerate/deepspeed `OpenWAMTrainer`. The on-disk
  checkpoint contract is identical, so deploy round-trips.
- **Deploy** — async/optimization paths removed (AR is sync-only).
