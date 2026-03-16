#!/usr/bin/env python3
"""
Locked evaluation script — DO NOT MODIFY.

This is the ground truth metric for autoresearch experiments.
The agent never touches this file.

Two metrics (choose with --metric):
  ssl_loss   — fast (~2 min): SSL val loss on fixed val set. Use for quick iteration.
  probe_acc  — slower (~5 min): Frozen backbone → sklearn linear probe → balanced accuracy.
               This is the actual downstream metric that matters.

Usage:
    python3 evaluate.py --checkpoint run_checkpoint.pt --metric probe_acc
    python3 evaluate.py --checkpoint run_checkpoint.pt --metric ssl_loss

Prints exactly one line:
    val_loss: 0.123456      (for ssl_loss)
    probe_acc: 0.456789     (for probe_acc)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "Lfp2vec_benchmarks"
ALPHABRAIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ALPHABRAIN_ROOT))

from backbones import get_backbone
from blind_localization.data.lazyloader_dataset import CompactDataset

# ---------------------------------------------------------------------------
# LOCKED CONSTANTS — never change these
# ---------------------------------------------------------------------------

VAL_DIR    = "/Users/neurolab/neuroinformatics/Alphabrain_staging/data/compact_atlas_5k/compact_atlas_5k/val"
BATCH_SIZE = 32
NATIVE_SR  = 1250
TARGET_SR  = 16000
SEED       = 42

# ---------------------------------------------------------------------------

def load_backbone(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone_name   = ckpt["backbone"]
    backbone_config = ckpt["backbone_config"]
    backbone = get_backbone(backbone_name, backbone_config).to(device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.eval()
    print(f"[evaluate] loaded {backbone_name} from epoch {ckpt.get('epoch','?')} "
          f"(ckpt val_loss={ckpt.get('val_loss', float('nan')):.4f})", flush=True)
    return backbone


@torch.no_grad()
def eval_ssl_loss(backbone, device):
    """SSL val loss on fixed val set."""
    val_ds = CompactDataset(VAL_DIR, atlas_depth=9, include_labels=True, gpu_resample=True)
    _pin = device.type == "cuda"
    _workers = 0 if device.type == "mps" else 4
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=_workers, pin_memory=_pin,
    )
    backbone.eval()
    total, n = 0.0, 0
    for batch in loader:
        signal, _, coords = batch
        signal = signal.to(device)
        coords = coords.to(device).float()
        if backbone.needs_resampling:
            signal = AF.resample(signal, NATIVE_SR, TARGET_SR)
        loss = backbone.ssl_forward(signal, coords=None)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def extract_embeddings(backbone, device):
    """Extract mean-pooled embeddings + labels from val set."""
    val_ds = CompactDataset(VAL_DIR, atlas_depth=9, include_labels=True, gpu_resample=True)
    _pin = device.type == "cuda"
    _workers = 0 if device.type == "mps" else 4
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=_workers, pin_memory=_pin,
    )
    all_embeds, all_labels = [], []
    for batch in loader:
        signal, label, _ = batch
        signal = signal.to(device)
        if backbone.needs_resampling:
            signal = AF.resample(signal, NATIVE_SR, TARGET_SR)
        embeds = backbone.encode(signal)          # [B, D]
        all_embeds.append(embeds.cpu().float().numpy())
        all_labels.append(label.numpy())
    X = np.concatenate(all_embeds, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def eval_probe_acc(backbone, device):
    """Balanced accuracy of a frozen linear probe trained on train embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Extract val embeddings (used for final eval)
    print("[evaluate] extracting val embeddings...", flush=True)
    X_val, y_val = extract_embeddings(backbone, device)

    # Extract train embeddings for probe fitting
    train_ds = CompactDataset(VAL_DIR.replace("/val", "/train"), atlas_depth=9, include_labels=True, gpu_resample=True)
    _pin = device.type == "cuda"
    _workers = 0 if device.type == "mps" else 4
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=_workers, pin_memory=_pin,
    )
    print("[evaluate] extracting train embeddings...", flush=True)
    all_embeds, all_labels = [], []
    with torch.no_grad():
        for batch in train_loader:
            signal, label, _ = batch
            signal = signal.to(device)
            if backbone.needs_resampling:
                signal = AF.resample(signal, NATIVE_SR, TARGET_SR)
            embeds = backbone.encode(signal)
            all_embeds.append(embeds.cpu().float().numpy())
            all_labels.append(label.numpy())
    X_train = np.concatenate(all_embeds, axis=0)
    y_train = np.concatenate(all_labels, axis=0)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Fit linear probe
    print(f"[evaluate] fitting linear probe on {len(X_train)} train samples...", flush=True)
    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        random_state=SEED, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = balanced_accuracy_score(y_val, y_pred)
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="run_checkpoint.pt",
                        help="Path to checkpoint saved by train.py")
    parser.add_argument("--metric", choices=["ssl_loss", "probe_acc"],
                        default="probe_acc",
                        help="Metric to evaluate")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    t0 = time.time()

    backbone = load_backbone(args.checkpoint, device)

    if args.metric == "ssl_loss":
        result = eval_ssl_loss(backbone, device)
        print(f"val_loss: {result:.6f}")

    elif args.metric == "probe_acc":
        result = eval_probe_acc(backbone, device)
        print(f"probe_acc: {result:.6f}")

    print(f"[evaluate] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
