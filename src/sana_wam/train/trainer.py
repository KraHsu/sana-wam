"""Minimal torchrun DDP + bf16 trainer for the SANA block-AR model.

Replaces openwam's 1143-line accelerate/deepspeed ``OpenWAMTrainer``. The real
train step is two calls — ``architecture.prepare_inputs(batch)`` (VAE-encode +
collate) then ``architecture.compute_loss(**inputs)`` (the AR per-chunk loss);
everything else here is freeze setup, optimizer groups, the loop, and writing
the deploy checkpoint contract.

Distributed: plain ``torchrun``. Because the entry point is ``compute_loss``
(not ``forward``), we cannot rely on DDP's forward hook — instead each rank runs
the full step and we **manually all-reduce trainable gradients** (no-op when
world_size==1). Frozen modules (VAE, text encoder) stay on the no-grad path.
"""

from __future__ import annotations

import logging
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from sana_wam.config import flatten_model_cfg
from sana_wam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset
from sana_wam.model.architecture import DualSystemARArchitecture
from sana_wam.train.checkpointing import manage_checkpoints, save_action_stats, save_config

logger = logging.getLogger(__name__)


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t = cfg.training
        self.lambda_video = float(self.t.get("lambda_video", 1.0))
        self.lambda_action = float(self.t.get("lambda_action", 1.0))

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # --- dataset ---
        self.dataset = MultiTaskRoboTwinDataset.from_config(cfg.dataloader, split="train")

        # --- architecture ---
        self.architecture = DualSystemARArchitecture(flatten_model_cfg(cfg.model))
        self.architecture.set_dtype_device(torch.bfloat16, self.device)
        self.architecture.init_training_schedulers(1000)
        self.architecture.set_training_runtime(
            use_gradient_checkpointing=bool(self.t.get("use_gradient_checkpointing", False)),
        )

        # --- freeze (VAE / text encoder) ---
        freeze_list = list(self.t.get("freeze", []) or [])
        frozen = self.architecture.freeze_modules(freeze_list)
        if _rank() == 0:
            logger.info("Froze %d params via %s", len(frozen), freeze_list)

        # --- action stats into buffers (deploy normalization parity) ---
        self._load_action_stats()

    # ------------------------------------------------------------------ setup
    def _load_action_stats(self):
        stats = getattr(self.dataset, "action_stats", None)
        if callable(stats):
            stats = stats()
        if stats is None:
            return
        mean = torch.from_numpy(stats["mean"].astype(np.float32))
        std = torch.from_numpy(np.maximum(stats["std"].astype(np.float32), 1e-3))
        self.architecture.action_mean.copy_(mean)
        self.architecture.action_std.copy_(std)
        if _rank() == 0:
            logger.info("Loaded action stats into architecture buffers")

    def _param_groups(self):
        """Two LR groups: action_backbone @ action_lr, everything else @ video_lr."""
        mods = self.architecture.get_trainable_modules(freeze_list=())
        action_params, video_params = [], []
        for name, mod in mods.items():
            tgt = action_params if name == "action_backbone" else video_params
            tgt += [p for p in mod.parameters() if p.requires_grad]
        groups = []
        if action_params:
            groups.append({"params": action_params, "lr": float(self.t.action_lr)})
        if video_params:
            groups.append({"params": video_params, "lr": float(self.t.video_lr)})
        return groups

    def _trainable_params(self):
        return [p for g in self._param_groups() for p in g["params"]]

    def _lr_lambda(self, step, total):
        warmup = int(self.t.get("warmup_steps", 0))
        if warmup and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # ------------------------------------------------------------------ step
    def _compute_loss(self, batch):
        inputs = self.architecture.prepare_inputs(batch)
        result = self.architecture.compute_loss(
            **inputs, lambda_video=self.lambda_video, lambda_action=self.lambda_action,
        )
        return result["loss"], result.get("loss_video"), result.get("loss_action")

    def _allreduce_grads(self, params):
        if _world_size() == 1:
            return
        ws = _world_size()
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= ws

    # ------------------------------------------------------------------ train
    def train(self):
        t = self.t
        max_steps = int(t.max_steps)
        save_steps = int(t.get("save_steps", 0) or 0)
        grad_accum = int(t.get("gradient_accumulation_steps", 1))
        batch_size = int(t.get("batch_size", 1))
        grad_clip = float(t.get("grad_clip", 0.0) or 0.0)
        debug = bool(t.get("debug", False))

        ts = time.strftime("%Y%m%d_%H%M%S") if not debug else "debug"
        output_path = os.path.join(str(t.get("output_dir", "outputs/ar_sana")), ts)
        if _rank() == 0:
            os.makedirs(output_path, exist_ok=True)

        sampler = DistributedSampler(self.dataset, shuffle=True) if _is_dist() else None
        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
            num_workers=int(t.get("num_workers", 0)), collate_fn=list, drop_last=True,
            persistent_workers=bool(int(t.get("num_workers", 0)) > 0),
        )

        params = self._trainable_params()
        optimizer = torch.optim.AdamW(
            self._param_groups(), weight_decay=float(t.get("weight_decay", 0.0)), betas=(0.9, 0.95),
        )
        base_lrs = [g["lr"] for g in optimizer.param_groups]

        if _rank() == 0:
            n_train = sum(p.numel() for p in params)
            logger.info("Trainable params: %.1fM | output: %s", n_train / 1e6, output_path)

        self.architecture.train()
        step, epoch = 0, 0
        data_iter = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        while step < max_steps:
            for micro in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    data_iter = iter(loader)
                    batch = next(data_iter)
                loss, lv, la = self._compute_loss(batch)
                (loss / grad_accum).backward()

            self._allreduce_grads(params)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            # cosine-with-warmup LR
            scale = self._lr_lambda(step, max_steps)
            for g, blr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = blr * scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if _rank() == 0 and (step % 10 == 0 or step == 1):
                logger.info(
                    "step %d/%d  loss=%.4f  video=%.4f  action=%.4f  lr=%.2e",
                    step, max_steps, float(loss),
                    float(lv) if lv is not None else 0.0,
                    float(la) if la is not None else 0.0,
                    optimizer.param_groups[0]["lr"],
                )

            if save_steps and step % save_steps == 0:
                self._save(output_path, step)

        # Final save (skip if the last step already saved at the save cadence).
        if not (save_steps and step % save_steps == 0):
            self._save(output_path, step)
        if _is_dist():
            dist.barrier()
        return output_path

    # ------------------------------------------------------------------ save
    def _save(self, output_path: str, step: int):
        if _rank() != 0:
            return
        ckpt = os.path.join(output_path, f"checkpoint_step_{step}.safetensors")
        save_config(output_path, self.cfg)
        self.architecture.save_checkpoint(ckpt)
        save_action_stats(output_path, self.dataset)
        self.architecture.copy_deploy_artifacts(output_path, self.cfg)
        manage_checkpoints(output_path, int(self.t.get("keep_last_k", 2)))
        logger.info("Saved checkpoint: %s", ckpt)
