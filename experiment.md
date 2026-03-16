# Experiment Log

Human-readable record of each autoresearch iteration.
Updated after every run. Complements `results.tsv` (machine-readable) with reasoning and observations.

---

## Active Experiment

**ID**: 001
**Status**: ✅ COMPLETED
**Branch**: `autoresearch/init`
**Backbone**: `whisper_distill` + `openai/whisper-tiny` (17.2M trainable params, EMA teacher-student)
**Device**: MPS (Apple M3 Max)
**Budget**: 1800s (30 min)

### Hypothesis
Whisper-tiny with EMA teacher-student (Data2Vec-style) is collapse-resistant — unlike wav2vec2 InfoNCE which collapsed to 7.69% (random chance). Use best HPs from HEBO trial c0cfdf58 as the baseline.

### Config
```
lr=3.909e-4, weight_decay=0.0492, batch=16 (grad_accum=4, effective=64)
ema_decay=0.9919, average_top_k_layers=4
mask_time_prob=0.3355, mask_time_length=5
atlas_depth=9 → 105 classes
```

### Result
- **probe_acc**: 0.0592 (5.92%) — 6.2× above random chance (0.95%)
- **val_loss**: 0.0255 (epoch 6, 30 min training)
- **Status**: keep (committed bfedb70)

---

## Completed Experiments

| ID | Description | probe_acc | val_loss | Status |
|----|-------------|-----------|----------|--------|
| ID | Description | probe_acc | val_loss | Status |
|----|-------------|-----------|----------|--------|
| 001 | baseline whisper-tiny HPs from HEBO trial c0cfdf58 | 0.0592 | 0.025500 | keep |
| 1 | baseline whisper-tiny HPs from HEBO trial c0cfdf58 | 0.0566 | 0.021635 | keep |
| 2 | increase time budget for more training | 0.0607 | 0.018964 | keep |
| 3 | increase time budget for more training | 0.0524 | 0.012689 | discard |
| 4 | lr x2 fallback | 0.0556 | 0.016608 | discard |
| 5 | lr x2 fallback | 0.0550 | 0.016609 | discard |
| 6 | increase time budget for more training | 0.0536 | 0.012088 | discard |
| 7 | increase time budget for more training | 0.0540 | 0.012073 | discard |
| 1 | baseline whisper-tiny HPs from HEBO trial c0cfdf58 | 0.0453 | 0.069371 | discard |
| 2 | real_signal_frames=150 — restrict masking to real LFP frames only (fixes 90% padding-mask bug) | 0.0384 | 0.039854 | discard |
| 3 | differential LR: encoder 0.1x via get_optimizer_params — preserve pretrained whisper weights | 0.0562 | 0.047565 | discard |
| 4 | mask_time_prob=0.65 mask_time_length=25 ema_anneal_end_step=3000 — high mask ratio + slow teacher (MAEEG/Data2Vec 2.0) | 0.0682 | 0.254372 | keep |
| 5 | vicreg_var_lambda=0.5 + TIME_BUDGET=7200 — VICReg variance regularization directly prevents collapse (EEG2Rep/VICReg literature) | 0.0593 | 0.154475 | discard |
| 6 | real_signal_frames=150 + TIME_BUDGET=7200 — re-apply padding mask fix with 2h to allow convergence (encode() now pools real frames only) | 0.0698 | 0.237119 | keep |
| 7 | loss_type=cosine + vicreg_var_lambda=0.25 + TIME_BUDGET=7200 — cosine SSL + anti-collapse combo (strongest collapse-resistant config) | 0.0651 | 0.293824 | discard |
| 8 | unfreeze_top_k_layers=2 — freeze bottom 2 encoder layers, train top 2 only; preserve pretrained low-level features | 0.0458 | 0.199497 | discard |
| 9 | TIME_BUDGET=10800 — push baseline HEBO HPs to full 3h convergence (only improvement was more time) | 0.0675 | 0.223190 | discard |
| 10 | attention_dropout=0.0 — current 0.183 adds noise that hurts representations in small 4-layer model | 0.0669 | 0.133962 | discard |
| 11 | ema_decay=0.999 — slower EMA teacher gives more stable targets (BYOL/Data2Vec 2.0: high EMA essential for SSL) | 0.0501 | 0.117499 | discard |
| 12 | WARMUP_STEPS=50 — current 200 = ~2.5 epochs of warmup; with 7 total epochs model barely reaches full LR | 0.0522 | 0.688001 | discard |
| 13 | GRAD_ACCUM=8 + LR=2e-4 — larger effective batch (128) with scaled LR; SSL benefits from large batches | 0.0676 | 0.180306 | discard |
| 14 | cosine LR decay after warmup — warmup 200 steps then cosine anneal to 0; flat LR hurts SSL convergence | 0.0508 | 0.661536 | discard |
| 15 | mask_before_encoder=True — proper Data2Vec: zero mel frames before encoder (not post-hoc on hidden states) | 0.0501 | 0.005362 | discard |
| 16 | input_noise_std=0.05 — student sees noisy mel, teacher sees clean mel; two-view SSL helps learn invariant representations | 0.0532 | 0.648886 | discard |

---

## Key Findings So Far

- **wav2vec2 InfoNCE collapses**: 7.69% balanced accuracy = random chance (1/13). All embeddings become similar.
- **whisper_distill avoids collapse**: EMA teacher-student uses MSE regression — no contrastive degenerate solution possible.
- **Dataset**: compact_atlas_5k, atlas_depth=9, 105 brain region classes, ~5000 train / ~5000 val samples.
- **MPS quirks**: batch_size=64 OOMs on M3 Max (48GB MPS limit). Reduced to 16 + grad_accum=4.

---

## Hypothesis Queue (priority order)

1. ✅ **Baseline** — whisper-tiny + best HEBO HPs, pure SSL loss (running)
2. **XYZ auxiliary loss** — `xyz_lambda=0.1` forces spatially-grounded embeddings. Most promising fix for remaining collapse.
3. **dist_contrastive** — `dist_lambda_raw=0.1` spatial distance-weighted contrastive.
4. **distpred auxiliary** — `distpred_lambda_raw=0.1` distance prediction head.
5. **Stronger masking** — increase `mask_time_prob` to 0.5. More masked → harder SSL → better representations.
6. **Smaller LR** — try 1e-4. Slower learning may prevent EMA instability.
7. **More negatives** — `num_negatives=200`. Harder negatives → better discriminative features.
8. **Combine aux losses** — xyz + distpred together once individual effects known.

---

## How the Loop Works

```
1. TRAIN    → train.py runs for TIME_BUDGET seconds on MPS
              saves best checkpoint to run_checkpoint.pt

2. EVALUATE → evaluate.py --metric probe_acc
              frozen backbone → linear probe on train embeddings → balanced acc on val
              (~5 min for 105-class probe)

3. DECIDE   → improved?  → git commit (keep), log to results.tsv + this file
              worse?      → git reset --hard HEAD~1 (discard)

4. PROPOSE  → qwen3:8b (Ollama) reads results.tsv + run.log + train.py
              outputs: 1-line hypothesis + exact train.py change

5. EDIT     → qwen3:8b writes modified train.py

6. → repeat
```

**Metric**: `probe_acc` (balanced accuracy, 105 classes). Random chance = 1/105 ≈ 0.95%.
**Baseline goal**: Beat 0.95% (anything >1% means representations are non-random).
**Target**: >15% balanced accuracy (competitive with wav2vec2 on 13-class task after full training).
