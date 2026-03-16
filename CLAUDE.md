# autoresearch/ — CLAUDE.md

> **For any LLM reading this file (Claude Code, Cursor, Copilot, GPT, etc.):**
> This is a reusable autonomous ML experiment loop. Before doing anything, read the
> [First-Time Setup: Ask the User](#first-time-setup-ask-the-user) section.
> If this project is already configured, skip to [Supervisor Role](#supervisor-role-claude--any-llm-via-cron).

---

## What This Is

An **autonomous ML research loop** that:
1. Pulls experiment proposals from a queue
2. Applies them as string-replacements to `train.py`
3. Trains the model, evaluates it, and keeps or discards the change via git
4. Refills the proposal queue (human supervisor or local LLM fallback)
5. Repeats indefinitely

It is **project-agnostic** — point it at any `train.py` + `evaluate.py` pair. The only things that change per project are the backbone config, dataset paths, and the metric being optimised.

---

## First-Time Setup: Ask the User

> **LLM instruction**: If `program.md` is empty or missing, or if the user says this is a new project, ask the following questions before touching any file. Populate `program.md` with their answers and confirm before starting the loop.

### Questions to ask

```
1. What is the research problem in one sentence?
   (e.g. "SSL pretraining of a Whisper-tiny backbone on LFP signals to classify 105 brain regions")

2. What model / backbone are you training?
   (name, size, any pretrained weights)

3. Where is your dataset?
   - Training data path:
   - Validation data path:
   - How many samples / classes?

4. What is the evaluation metric?
   (e.g. balanced_accuracy, top-1 accuracy, F1 — must match what evaluate.py prints)

5. What is the current best result?
   (e.g. "6.98% probe_acc at iteration 13")

6. What is the target / success threshold?
   (e.g. ">15% probe_acc")

7. What device will training run on?
   (CUDA GPU / Apple MPS / CPU — affects batch size and epoch time)

8. What is your TIME_BUDGET per experiment?
   (seconds of wall-clock training per run, e.g. 7200 = 2 hours)

9. Any hard constraints on train.py?
   (e.g. "do not change BATCH_SIZE", "keep auxiliary lambdas at 0")

10. Do you want Ollama (local LLM) as a fallback proposal generator?
    (only needed if the queue runs dry and you can't supervise manually)
```

Once the user has answered, **write their answers to `program.md`** and confirm before running anything.

---

## How the Loop Works

```
autoloop.py runs forever as a background process:

while True:
    1. Pop next proposal from proposal_queue.json
       (fallback chain: queue → next_proposal.json → arxiv+qwen3 → qwen3:8b)
    2. Apply old→new string replacements to train.py
    3. Run: uv run python3 train.py         (TIME_BUDGET seconds, saves run_checkpoint.pt)
    4. Run: uv run python3 evaluate.py      (prints metric, e.g. probe_acc: 0.069770)
    5. metric > best_ever?
          YES → git commit train.py  (status=keep)
          NO  → git checkout train.py (status=discard, reverts to last kept state)
    6. Log to results.tsv
    7. If queue empty + stuck (≥3 straight discards) → search arxiv → qwen3:8b proposes
    8. repeat
```

**Stuck detection**: after `STUCK_THRESHOLD=3` consecutive discards with an empty queue,
autoloop searches arxiv (5 queries) and feeds abstracts to `qwen3:8b` via Ollama.
**This is optional** — if you keep `proposal_queue.json` filled (via any LLM supervisor),
the loop never needs Ollama.

---

## File Map

```
autoresearch/
├── CLAUDE.md               ← you are here
├── autoloop.py             ← autonomous loop engine (never edit)
├── train.py                ← THE ONLY FILE the loop modifies
├── evaluate.py             ← locked ground-truth metric (never modify)
├── proposal_queue.json     ← pending experiments (FIFO queue, supervisor refills)
├── next_proposal.json      ← one-off override (autoloop pops + deletes it)
├── results.tsv             ← full experiment history
├── autoloop_stdout.log     ← live loop output
├── dashboard.py            ← terminal TUI (ANSI, sparklines, live refresh)
├── monitor.py              ← older simpler dashboard
├── ollama_agent.py         ← standalone qwen3:8b proposal agent (optional)
├── experiment.md           ← human-readable experiment notes
├── program.md              ← problem statement + hypotheses
│                              (human edits this — ask the user, or help them draft it)
├── run_checkpoint.pt       ← best checkpoint from current/last run
├── run.log                 ← stdout from last train.py run
├── eval.log                ← stdout from last evaluate.py run
├── pyproject.toml          ← uv project (torch, sklearn, transformers, etc.)
└── .venv/                  ← managed by uv
```

The model backbone lives outside this folder (project-specific):
```
../Lfp2vec_benchmarks/backbones/whisper_distill_backbone.py   ← the neural network
../Lfp2vec_benchmarks/backbones/__init__.py                   ← get_backbone() factory
../blind_localization/data/lazyloader_dataset.py              ← CompactDataset
```

---

## Machine Setup

1. **Install uv** (Python env manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python deps** (from `autoresearch/`):
   ```bash
   cd autoresearch
   ~/.local/bin/uv sync
   ```

3. **Set data paths in `train.py`** — ask the user for the paths, then edit:
   ```python
   DATA_DIR = "/path/to/dataset/train"
   VAL_DIR  = "/path/to/dataset/val"
   ```

4. **Device**: training loop auto-detects CUDA → MPS → CPU. No config needed.

5. **Ollama (optional)**: only for autoloop's stuck-detection fallback.
   Install [Ollama](https://ollama.com) and run `ollama pull qwen3:8b` if you want it.

6. **Git**: autoloop uses `git commit` / `git checkout train.py` to track experiments.
   The repo must have at least one commit and `git` must be on PATH.

---

## Starting / Stopping / Monitoring

```bash
# Start the loop (from the autoresearch/ directory)
cd /path/to/autoresearch
nohup ~/.local/bin/uv run python3 autoloop.py >> autoloop_stdout.log 2>&1 &

# Check if running
ps aux | grep "autoloop\|train\.py" | grep -v grep

# Live log
tail -f autoloop_stdout.log

# Terminal dashboard (15s refresh, ANSI colors, sparklines)
~/.local/bin/uv run python3 dashboard.py

# Older simple dashboard (30s refresh)
~/.local/bin/uv run python3 monitor.py

# Stop (graceful — current epoch finishes)
kill $(pgrep -f autoloop.py)
```

---

## The Only File You Should Edit: `train.py`

`train.py` is the **agent's sandbox** — autoloop applies all experiments here via string replacement. Everything else is locked.

### What CAN be changed in train.py

```python
# Optimizer HPs
LR              = 3.908750354896152e-4
WEIGHT_DECAY    = 0.049235573219082
WARMUP_STEPS    = 200
BATCH_SIZE      = 16         # device memory limit
GRAD_ACCUM      = 4          # effective batch = BATCH_SIZE * GRAD_ACCUM

# Training budget
TIME_BUDGET     = 7200       # seconds wall-clock (0 = use MAX_EPOCHS instead)
MAX_EPOCHS      = 50         # only if TIME_BUDGET == 0

# LR schedule function
def get_lr_scale(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    return 1.0                # ← can change to cosine decay etc.

# Backbone config (project-specific — all keys passed to the backbone)
BACKBONE_CONFIG = {
    "backbone": "whisper_distill",
    "pretrained_model": "openai/whisper-tiny",
    "hidden_size": 384,
    "num_hidden_layers": 4,
    "num_attention_heads": 6,
    "intermediate_size": 1536,
    "hidden_dropout": 0.000304,        # ← tunable
    "attention_dropout": 0.1827,       # ← tunable
    "ema_decay": 0.9919,               # ← tunable
    "average_top_k_layers": 4,         # ← tunable
    "mask_time_prob": 0.65,            # ← tunable
    "mask_time_length": 25,            # ← tunable
    "ema_anneal_end_step": 3000,       # ← tunable
    "num_negatives": 100,
    "real_signal_frames": 150,         # CRITICAL: 3s LFP / 30s pad = 150/1500 real frames
    "ssl_lambda_raw": 1.0,
    "dist_lambda_raw": 0.0,            # KEEP AT 0
    "distpred_lambda_raw": 0.0,        # KEEP AT 0
    "xyz_lambda": 0.0,                 # KEEP AT 0
}
```

### What CANNOT be changed

- `evaluate.py` — never touch, it's the ground truth metric
- `program.md` — human's problem statement. Ask the user to edit it, or help them draft changes, but do not modify it directly.
- Data paths (`DATA_DIR`, `VAL_DIR`)
- `BACKBONE` name
- Auxiliary loss lambdas (`xyz_lambda`, `dist_lambda_raw`, `distpred_lambda_raw` stay 0)
- Package installs

---

## Proposal Queue Format

```json
[
  {
    "description": "one-line description of the experiment",
    "changes": [
      {"old": "exact string in train.py", "new": "replacement string"}
    ]
  }
]
```

**Rules for writing proposals**:
1. `old` must match `train.py` **exactly** — copy-paste, check spaces, quotes, inline comments
2. Only modify `BACKBONE_CONFIG`, optimizer HPs, `TIME_BUDGET`, or `get_lr_scale()`
3. Do NOT repeat any experiment already in `results.tsv`
4. Do NOT add auxiliary losses (keep `xyz_lambda`, `dist_lambda_raw`, `distpred_lambda_raw` at 0)
5. Base proposals on what improved vs what didn't — read `results.tsv` before proposing

---

## Injecting a One-Off Experiment (Override)

```bash
cat > autoresearch/next_proposal.json << 'EOF'
{
  "description": "my experiment description",
  "changes": [{"old": "exact old string", "new": "replacement"}]
}
EOF
```

autoloop picks it up at the start of the next iteration and deletes the file.

---

## Evaluation: `evaluate.py`

Called by autoloop after every training run. **Never modify this file.**

```bash
# How autoloop calls it:
uv run python3 evaluate.py --checkpoint run_checkpoint.pt --metric probe_acc

# Output (exactly one line):
probe_acc: 0.069770
```

Internally:
1. Loads `run_checkpoint.pt` (saved by train.py)
2. Extracts frozen embeddings via `backbone.encode(signal)`
3. Fits `sklearn.LogisticRegression` linear probe on train embeddings
4. Returns `balanced_accuracy_score` on val set → **this is the metric**

SSL val_loss does NOT correlate with probe_acc — always use probe_acc.

---

## Supervisor Role (Claude / any LLM via cron)

**You do NOT need Ollama or any local LLM to be a supervisor.** Any LLM (Claude, GPT, Gemini, etc.) can supervise by reading loop state and writing proposals. Ollama/qwen3 is only autoloop's last-resort fallback when the queue runs dry and it's been stuck.

A cron job triggers every 10 minutes with the standard check prompt. The supervisor should:

1. Read last 30 lines of `autoloop_stdout.log`
2. Read `results.tsv` and `proposal_queue.json`
3. **If crashed** (no "Starting train.py" in recent lines, no active `python3 train.py` process):
   ```bash
   cd /path/to/autoresearch
   nohup ~/.local/bin/uv run python3 autoloop.py >> autoloop_stdout.log 2>&1 &
   ```
4. **If queue < 3 items**: add 2–3 new proposals based on results history
5. **Otherwise**: do nothing

---

## Current Project: Alphabrain / LFP2Vec

> This section is project-specific. If you are adapting this framework for a new project,
> replace this section with your own project state after completing the setup questions above.

**Problem**: SSL pretraining of a Whisper-tiny backbone (Data2Vec-style EMA teacher-student)
on LFP signals from the Allen Brain Observatory to classify brain regions (105 classes, `atlas_depth=9`).

| | |
|---|---|
| Random chance | 0.95% (1/105) |
| Best so far | **6.98%** (iter 13, `real_signal_frames=150 + TIME_BUDGET=7200`) |
| Target | **>15%** |
| Device | Apple M3 Max MPS (~80 min/epoch, ~2–3 epochs per 2h budget) |

### Backbone: `whisper_distill_backbone.py`

Data2Vec-style EMA teacher-student SSL on mel spectrograms:

```
LFP signal (1250 Hz) → resample to 16 kHz → WhisperFeatureExtractor → mel [B, 80, 3000]
                                                        |
               ┌────────────────────────────────────────┴──────────────────────────────────────┐
               ▼                                                                                ▼
         Teacher encoder (EMA)                                                   Student encoder (trainable)
         θ_t ← decay·θ_t + (1-decay)·θ_s                                        sees MASKED mel frames
               |                                                                                |
         avg top-K hidden layers                                                       ConvDecoder (4-layer
         + instance normalize                                                          depthwise-sep conv)
               |                                                                                |
               └────────────── MSE (or cosine) loss at masked positions ───────────────────────┘
```

**Key**: `real_signal_frames=150` — LFP is 3s padded to 30s for Whisper. Only 150/1500 frames are real.
`encode()` mean-pools only those 150 frames → prevents trivial padding-based representations.

### Data

```
compact_atlas_5k/
├── train/    ~4000 samples, 105 classes at atlas_depth=9
└── val/      ~1000 samples

Each sample: (signal [3750,], meta_str, label_id [0-104], coords [3,])
Signal: 3s LFP @ 1250 Hz → padded to 30s (44100 samples) by CompactDataset
```

Expected location: `<REPO_ROOT>/data/compact_atlas_5k/`

### Results History (as of 2026-03-16)

| # | probe_acc | status | description |
|---|-----------|--------|-------------|
| 1 | 5.92% | keep | baseline HEBO HPs |
| 2 | 5.66% | keep | baseline HEBO HPs |
| 3 | 6.07% | keep | increase time budget |
| 4–10 | 3.8–5.6% | discard | lr changes, time budget variants |
| 11 | 6.82% | **keep** | mask_time_prob=0.65, mask_time_length=25, ema_anneal_end_step=3000 |
| 12 | 5.93% | discard | vicreg_var_lambda=0.5 |
| **13** | **6.98%** | **keep ★ BEST** | **real_signal_frames=150 + TIME_BUDGET=7200** |
| 14 | 6.51% | discard | cosine loss + vicreg |
| 15 | 4.58% | discard | unfreeze_top_k_layers=2 |
| 16 | 6.75% | discard | TIME_BUDGET=10800 |
| 17 | 6.69% | discard | attention_dropout=0.0 |
| 18 | 5.01% | discard | ema_decay=0.999 |
| 19 | 5.22% | discard | WARMUP_STEPS=50 |
| 20 | 6.76% | discard | GRAD_ACCUM=8 + LR=2e-4 |
| 21 | 5.08% | discard | cosine LR decay |
| 22 | 5.01% | discard | mask_before_encoder=True |
| 23+ | running | — | input_noise_std=0.05, average_top_k_layers=2, hidden_dropout=0.1, loss_type=cosine |

**Key findings**:
- Only `real_signal_frames=150` (padding fix) + 2h budget ever reliably improved probe_acc
- Aggressive changes to EMA, LR schedule, masking all hurt
- Model stuck at ~7% — may need fundamentally different approach once queue is exhausted

### Environment

- **Device**: Apple M3 Max MPS (no CUDA)
- **Python env**: managed by `uv` → `~/.local/bin/uv run python3 <script>`
- **Local LLM (optional)**: Ollama at `localhost:11434`, model `qwen3:8b`
- **Git branch**: `autoresearch/init`
- **Remote**: `https://github.com/subro608/Alphabrain_staging.git`

---

## Quick Reference

```bash
# Is the loop running?
ps aux | grep "autoloop\|train\.py" | grep -v grep

# What just happened?
tail -20 autoloop_stdout.log

# What are the results?
cat results.tsv

# What's queued next?
cat proposal_queue.json

# Live dashboard
~/.local/bin/uv run python3 dashboard.py

# Inject a one-off experiment
echo '{"description":"...","changes":[{"old":"...","new":"..."}]}' > next_proposal.json

# Start loop from scratch
nohup ~/.local/bin/uv run python3 autoloop.py >> autoloop_stdout.log 2>&1 &
```
