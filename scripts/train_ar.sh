#!/usr/bin/env bash
# Multi-GPU AR training (2-task example). NCCL_NVLS_ENABLE=0 is required on the
# H200/NVSwitch box for >2-GPU NCCL init.
#   NPROC_PER_NODE=8 bash scripts/train_ar.sh
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC="${NPROC_PER_NODE:-8}"
export NCCL_NVLS_ENABLE=0

uv run torchrun --nproc_per_node="$NPROC" scripts/train.py \
    --config configs/train_ar_sana.yaml \
    dataloader.dataset_dir=/DATA/share/RoboTwin2.0/dataset \
    dataloader.train_tasks="[adjust_bottle,lift_pot]" \
    dataloader.variant=clean_50 \
    training.max_steps=1995 \
    training.save_steps=500 \
    training.batch_size=1 \
    training.output_dir=outputs/ar_2task
