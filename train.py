#!/usr/bin/env python3
"""
Autoresearch SSL pretraining script — agent modifies this file.

The ONLY file the agent edits. Everything is fair game:
  - backbone architecture / config
  - SSL objective (contrastive, masked pred, XYZ aux, dist aux)
  - optimizer / learning rate / schedule
  - batch size, gradient accumulation
  - preprocessing / augmentation

DO NOT modify evaluate.py or program.md.

Usage (on HPC):
    singularity exec --nv /scratch/sd5963/containers/w2v2_cuda_cu128_ray.sif \
        python3 train.py > run.log 2>&1

The script saves a checkpoint to run_checkpoint.pt and prints a --- summary block.
"""

import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as AF

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "Lfp2vec_benchmarks"
ALPHABRAIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ALPHABRAIN_ROOT))

from backbones import get_backbone
from blind_localization.data.lazyloader_dataset import CompactDataset

# ---------------------------------------------------------------------------
# Experiment config — EDIT THESE (agent modifies this section and below)
# ---------------------------------------------------------------------------

BACKBONE       = "whisper_distill"   # whisper_distill | wav2vec2
DATA_DIR       = "/path/to/your/dataset/train"     # ← set this
VAL_DIR        = "/path/to/your/dataset/val"       # ← set this
CHECKPOINT_OUT = Path(__file__).resolve().parent / "run_checkpoint.pt"

# Training budget — use ONE of these two modes:
#   TIME_BUDGET > 0  → train for exactly TIME_BUDGET seconds (wall clock)
#   TIME_BUDGET = 0  → train for MAX_EPOCHS epochs instead
TIME_BUDGET  = 7200     # seconds — 2h to converge after real-signal-frames fix
MAX_EPOCHS   = 50       # used only if TIME_BUDGET == 0

# Backbone config — whisper-tiny (39M) with best HPs from ws5k_ssl_hebo trial c0cfdf58
BACKBONE_CONFIG = {
    "backbone": BACKBONE,
    "pretrained_model": "openai/whisper-tiny",
    "hidden_size": 384,
    "num_hidden_layers": 4,
    "num_attention_heads": 6,
    "intermediate_size": 1536,
    "hidden_dropout": 0.000304035987688419,
    "attention_dropout": 0.1827156188329001,
    # whisper_distill-specific (EMA teacher-student)
    "ema_decay": 0.9919816400906644,
    "average_top_k_layers": 2,       # top 2 encoder layers only (more semantic per Data2Vec audio ablation)
    "mask_time_prob": 0.65,
    "mask_time_length": 25,
    "ema_anneal_end_step": 3000,
    "num_negatives": 100,
    "real_signal_frames": 150,   # 3s LFP / 30s pad * 1500 frames = 150 real frames
    # Loss weights (raw — whisper_distill normalises internally)
    "ssl_lambda_raw": 1.0,
    "dist_lambda_raw": 0.0,
    "distpred_lambda_raw": 0.0,
    "xyz_lambda": 0.0,
}

# Optimization — adapted from best trial c0cfdf58
BATCH_SIZE      = 16         # MPS memory limit ~47GB; 64 OOMs
GRAD_ACCUM      = 4          # effective batch = 64 (same as GPU config)
LR              = 3.908750354896152e-4
WEIGHT_DECAY    = 0.049235573219082
WARMUP_STEPS    = 50      # was 200 = 2.5 epochs warmup; too slow for short runs
NATIVE_SR       = 1250
TARGET_SR       = 16000

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[autoresearch] backbone={BACKBONE}, device={device}")
print(f"[autoresearch] budget={'%ds' % TIME_BUDGET if TIME_BUDGET > 0 else '%d epochs' % MAX_EPOCHS}")
print(f"[autoresearch] batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, lr={LR}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_ds = CompactDataset(DATA_DIR, atlas_depth=9, include_labels=True, gpu_resample=True)
val_ds   = CompactDataset(VAL_DIR,  atlas_depth=9, include_labels=True, gpu_resample=True)

_pin = device.type == "cuda"
_workers = 0 if device.type == "mps" else 4   # MPS: forked workers cause issues
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=_workers, pin_memory=_pin, drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=_workers, pin_memory=_pin, drop_last=False,
)

print(f"[autoresearch] train={len(train_ds)} samples, val={len(val_ds)} samples")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

backbone = get_backbone(BACKBONE, BACKBONE_CONFIG).to(device)
num_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"[autoresearch] params={num_params/1e6:.1f}M")

# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    backbone.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
)

def get_lr_scale(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    return 1.0

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(loader, train=True):
    backbone.train() if train else backbone.eval()
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            signal, label, coords = batch
            signal = signal.to(device, non_blocking=True)        # [B, T]
            coords = coords.to(device, non_blocking=True).float() # [B, 3]

            # Resample 1250 Hz → 16 kHz
            if backbone.needs_resampling:
                signal = AF.resample(signal, NATIVE_SR, TARGET_SR)

            # SSL forward — backbone.ssl_forward returns scalar loss
            use_coords = BACKBONE_CONFIG.get("xyz_lambda", 0) > 0 or \
                         BACKBONE_CONFIG.get("dist_lambda", 0) > 0 or \
                         BACKBONE_CONFIG.get("distpred_lambda", 0) > 0
            loss = backbone.ssl_forward(signal, coords=coords if use_coords else None)
            loss_val = loss.item() / GRAD_ACCUM

            if train:
                (loss / GRAD_ACCUM).backward()

            total_loss += loss_val
            n_batches  += 1

            if train and n_batches % GRAD_ACCUM == 0:
                step = n_batches // GRAD_ACCUM
                scale = get_lr_scale(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_batches, 1)


t_start = time.time()
t_train_start = None
epoch = 0
best_val_loss = float("inf")
total_training_seconds = 0.0
step_global = 0
peak_vram_mb = 0.0

# GC freeze to avoid stalls
gc.collect()
gc.freeze()
gc.disable()

print(f"[autoresearch] starting training...")

while True:
    t0 = time.time()

    train_loss = run_epoch(train_loader, train=True)

    t1 = time.time()
    epoch_sec = t1 - t0

    # Don't count first epoch in budget (compilation / caching warmup)
    if epoch > 0:
        total_training_seconds += epoch_sec

    val_loss = run_epoch(val_loader, train=False)
    if device.type == "cuda":
        peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)

    epoch += 1
    progress_pct = 100 * total_training_seconds / TIME_BUDGET if TIME_BUDGET > 0 else 100 * epoch / MAX_EPOCHS
    print(f"epoch {epoch:04d} ({progress_pct:.1f}%) | train={train_loss:.4f} | val={val_loss:.4f} | {epoch_sec:.0f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "backbone": BACKBONE,
            "backbone_config": BACKBONE_CONFIG,
            "backbone_state_dict": backbone.state_dict(),
            "epoch": epoch,
            "val_loss": best_val_loss,
        }, CHECKPOINT_OUT)

    # Check stopping condition
    if TIME_BUDGET > 0:
        if epoch > 1 and total_training_seconds >= TIME_BUDGET:
            break
    else:
        if epoch >= MAX_EPOCHS:
            break

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary (parse this with grep "^val_loss:\|^probe_acc:")
# ---------------------------------------------------------------------------
print("---")
print(f"val_loss:          {best_val_loss:.6f}")
print(f"training_seconds:  {total_training_seconds:.1f}")
print(f"total_seconds:     {t_end - t_start:.1f}")
print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
print(f"num_epochs:        {epoch}")
print(f"num_params_M:      {num_params/1e6:.1f}")
print(f"backbone:          {BACKBONE}")
print(f"time_budget:       {TIME_BUDGET}")
print(f"checkpoint:        {CHECKPOINT_OUT}")
