#!/usr/bin/env bash
# Single-GPU smoke: loads real SANA-2B + Wan VAE + Gemma, trains a few steps on
# one task, writes a checkpoint dir. Verifies the full train path end to end.
#   bash scripts/smoke_ar.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export NCCL_NVLS_ENABLE=0

uv run python scripts/train.py --config configs/train_ar_sana.yaml \
    dataloader.dataset_dir=/DATA/share/RoboTwin2.0/dataset \
    dataloader.task_name=adjust_bottle \
    dataloader.variant=clean_50 \
    training.max_steps=20 \
    training.save_steps=20 \
    training.batch_size=1 \
    training.debug=true \
    training.output_dir=outputs/smoke
