# autoresearch — LFP Brain Region Decoding

Autonomous SSL pretraining experiments for the Lfp2vec / Alphabrain project.

## The Problem

We are self-supervised pre-training a Whisper-tiny backbone (Data2Vec-style EMA teacher-student) on LFP signals from the Allen Brain Observatory to classify brain regions (105 classes at atlas_depth=9). The SSL pre-training is producing **collapsed representations**:

- Fine-tuning gives ~6% balanced accuracy vs 0.95% random chance (1/105)
- Target is >15%
- Root cause: SSL objective may not be forcing discriminative representations

Root cause candidates:
1. Masking padding frames (3s LFP padded to 30s → 90% of frames are trivial)
2. Post-hoc hidden state masking (not proper Data2Vec pre-encoder masking)
3. EMA decay too fast / too slow → unstable or stale targets
4. Not enough training time — MPS epoch ~80 min, TIME_BUDGET=3600 = ~1 epoch only

## Current Workflow: Autonomous Loop + Claude Supervisor

The experiment loop runs **fully autonomously** via `autoloop.py`. Claude acts as a **supervisor**, checking in every 10 minutes via a cron job to refill the proposal queue and restart if crashed.

### Components

| File | Role |
|------|------|
| `autoloop.py` | Autonomous loop: pop proposal → apply to train.py → train → eval → keep/discard → repeat |
| `train.py` | The ONLY file autoloop modifies. Agent edits BACKBONE_CONFIG, optimizer HPs, TIME_BUDGET. |
| `evaluate.py` | Locked. Runs linear probe, outputs `probe_acc`. Do not modify. |
| `proposal_queue.json` | JSON array of pending experiments. autoloop pops from front. Claude refills when low. |
| `next_proposal.json` | Optional override: if this file exists, autoloop uses it as the NEXT proposal (ignores queue front). Delete after use. |
| `results.tsv` | Tab-separated log of all experiments. |
| `autoloop_stdout.log` | Full loop stdout. Tail this to monitor progress. |
| `monitor.py` | Live dashboard: `uv run python3 monitor.py` |

### How autoloop.py Works

```
while True:
    1. Pop next proposal from proposal_queue.json (or next_proposal.json if exists)
    2. Apply changes to train.py (string replacements)
    3. git commit
    4. Run: uv run python3 train.py
    5. Run: uv run python3 evaluate.py --metric probe_acc
    6. If probe_acc > best → keep commit, update best
       Else → git checkout train.py (revert to last kept commit)
    7. If queue empty → ask qwen3:8b to propose next experiment
    8. Log to results.tsv
```

### How Claude Supervises (Cron every 10 min)

Claude is triggered every 10 minutes with the standard check prompt. Claude:

1. Reads last 30 lines of `autoloop_stdout.log`
2. Reads `results.tsv` and `proposal_queue.json`
3. **If loop crashed** (no "Starting train.py" in recent log + no active process): restarts with
   ```bash
   nohup ~/.local/bin/uv run python3 autoloop.py >> autoloop_stdout.log 2>&1 &
   ```
4. **If queue has <3 items**: adds 2-3 new proposals to `proposal_queue.json` based on results
5. **Otherwise**: does nothing

### Starting the Loop

```bash
cd /Users/neurolab/neuroinformatics/Alphabrain_staging/autoresearch
nohup ~/.local/bin/uv run python3 autoloop.py >> autoloop_stdout.log 2>&1 &
```

### Monitoring

```bash
tail -f autoloop_stdout.log          # live log
uv run python3 monitor.py            # live dashboard (refresh 30s)
```

## Proposal Queue Format

Each proposal is a JSON object:

```json
{
  "description": "one-line description of the experiment",
  "changes": [
    {"old": "exact string in train.py", "new": "replacement string"}
  ]
}
```

**Rules for proposals:**
- Only modify `BACKBONE_CONFIG` values, optimizer HPs (`LR`, `WEIGHT_DECAY`, `WARMUP_STEPS`, `BATCH_SIZE`, `GRAD_ACCUM`), or `TIME_BUDGET`
- Do NOT add auxiliary losses (`xyz_lambda`, `dist_lambda_raw`, `distpred_lambda_raw` stay 0)
- Do NOT repeat experiments already in `results.tsv`
- `old` strings must match train.py **exactly** (check spacing, quotes, comments)
- Base proposals on what improved vs what didn't

### Injecting a One-Off Proposal (Override)

To force a specific experiment as the very next one (bypasses the queue):

```bash
cat > next_proposal.json << 'EOF'
{
  "description": "my experiment",
  "changes": [{"old": "...", "new": "..."}]
}
EOF
```

autoloop picks this up at the start of the next iteration and deletes it.

## Metric Strategy

- **Primary metric**: `probe_acc` — balanced accuracy of frozen linear probe on val set
- **Random chance**: 0.95% (1/105 classes at atlas_depth=9)
- **Target**: >15%
- **Key insight**: SSL val_loss improving does NOT guarantee probe_acc improves. Always use `probe_acc`.

Higher `probe_acc` = better.

## What train.py Can Change

- Any value in `BACKBONE_CONFIG` (backbone architecture, SSL params, dropout, EMA settings)
- `TIME_BUDGET` — wall-clock seconds; `TIME_BUDGET=0` switches to `MAX_EPOCHS` mode
- `LR`, `WEIGHT_DECAY`, `WARMUP_STEPS`, `BATCH_SIZE`, `GRAD_ACCUM`
- The `get_lr_scale()` function (LR schedule)
- New backbone config keys supported by the backbone (e.g. `real_signal_frames`, `input_noise_std`, `loss_type`, `mask_before_encoder`)

## What Cannot Change

- `evaluate.py` — locked
- `program.md` — human's file (update only when human explicitly asks)
- Package installs
- Eval data path or eval procedure

## Logging Results

`results.tsv` — tab-separated. Header:

```
commit	metric	value	peak_vram_mb	status	description
```

- `commit`: 7-char git hash
- `metric`: `probe_acc`
- `value`: float
- `peak_vram_mb`: 0.0 on MPS (not tracked)
- `status`: `keep`, `discard`, `crash`
- `description`: short text, no tabs

## Results So Far (as of 2026-03-14)

| probe_acc | status | description |
|-----------|--------|-------------|
| 5.92% | keep | baseline HEBO HPs |
| 5.66% | keep | baseline HEBO HPs |
| **6.07%** | **keep** | **increase time budget (best)** |
| 5.24% | discard | increase time budget |
| 5.56% | discard | lr x2 fallback |
| 5.50% | discard | lr x2 fallback |
| 5.36% | discard | increase time budget |
| 5.40% | discard | increase time budget |
| 4.53% | discard | baseline HEBO HPs |
| 3.84% | discard | real_signal_frames=150 (needs more time) |
| 5.62% | discard | differential LR encoder 0.1x |

**Key finding**: Only increasing TIME_BUDGET has ever improved probe_acc. Most other changes are neutral or hurt. More training time is the clearest bottleneck (1h ≈ 1 epoch on MPS M3 Max).

## Hypotheses Queue (priority order)

1. **More training time** — TIME_BUDGET=7200 or 10800. Only thing proven to help.
2. **real_signal_frames=150 + more time** — fix is correct but 1h wasn't enough to converge.
3. **ema_decay=0.999** — slower EMA gives more stable targets (BYOL/Data2Vec 2.0).
4. **attention_dropout=0.0** — current 0.183 adds noise that hurts small model.
5. **mask_before_encoder=True** — proper Data2Vec: mask mel before encoder, not post-hoc on hidden states.
6. **loss_type=cosine** — scale-invariant cosine SSL loss, more collapse-resistant.
7. **input_noise_std=0.05** — two-view SSL: noisy student vs clean teacher.
8. **average_top_k_layers=2** — top-2 encoder layers only (more semantic per Data2Vec ablations).
9. **cosine LR schedule** — warmup then cosine decay; flat LR hurts SSL convergence.

## Simplicity Criterion

All else being equal, simpler is better.
- Small improvement + added complexity → discard
- Same performance + deleted code → keep
- Large improvement + ugly code → keep, but note the debt

## NEVER STOP

Once running, do NOT pause to ask if you should continue. Run until the human manually stops you. If stuck on an idea, move to the next hypothesis. You are an autonomous researcher.
