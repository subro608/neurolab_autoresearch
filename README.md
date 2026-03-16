# neurolab_autoresearch

An **autonomous ML experiment loop** that self-supervises model training, evaluates representations, and continuously proposes + tests new experiments — without human intervention.

Built for the [Alphabrain / LFP2Vec](https://github.com/subro608/Alphabrain_staging) project, but designed to be fully **project-agnostic**.

---

## What It Does

```
while True:
    1. Pop next experiment from proposal_queue.json
    2. Apply string-replacement changes to train.py
    3. Train the model (TIME_BUDGET seconds)
    4. Evaluate (linear probe → metric)
    5. metric improved? → git commit (keep) : git checkout (discard)
    6. Log to results.tsv
    7. If queue empty + stuck → search arxiv → local LLM proposes
    8. repeat
```

A **supervisor LLM** (Claude, GPT, Gemini — via cron) reads the loop state every 10 minutes, restarts if crashed, and refills the proposal queue with new ideas. No Ollama required if the supervisor keeps the queue filled.

---

## Quick Start

### 1. Install dependencies

```bash
# Install uv (Python env manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python deps
uv sync
```

### 2. Set your data paths

Edit `train.py`:
```python
DATA_DIR = "/path/to/your/dataset/train"
VAL_DIR  = "/path/to/your/dataset/val"
```

### 3. Describe your project

Fill in `program.md` with your problem statement, metric, and hypotheses. Or let the LLM supervisor ask you (see CLAUDE.md).

### 4. Add initial proposals

```bash
cat > proposal_queue.json << 'EOF'
[
  {
    "description": "baseline run",
    "changes": []
  }
]
EOF
```

### 5. Start the loop

```bash
nohup ~/.local/bin/uv run python3 autoloop.py >> autoloop_stdout.log 2>&1 &
```

### 6. Monitor

```bash
tail -f autoloop_stdout.log           # live log
uv run python3 dashboard.py           # terminal TUI (15s refresh)
```

---

## File Overview

| File | Role |
|------|------|
| `autoloop.py` | Core loop engine — **never edit** |
| `train.py` | **Only file the loop modifies** — your training script |
| `evaluate.py` | Locked metric evaluator — **never modify** |
| `proposal_queue.json` | FIFO queue of pending experiments |
| `next_proposal.json` | One-off override — autoloop pops + deletes it |
| `results.tsv` | Full experiment history |
| `autoloop_stdout.log` | Live loop output |
| `dashboard.py` | Terminal TUI with sparklines and live config |
| `monitor.py` | Simpler 30s-refresh dashboard |
| `ollama_agent.py` | Standalone Ollama proposal agent (optional) |
| `program.md` | Human-owned problem statement — LLM reads, human edits |
| `experiment.md` | Human-readable experiment notes |
| `CLAUDE.md` | Full onboarding doc for any LLM supervisor |

---

## Proposal Format

```json
[
  {
    "description": "one-line experiment description",
    "changes": [
      {"old": "exact string in train.py", "new": "replacement string"}
    ]
  }
]
```

**Rules:**
- `old` must match `train.py` exactly (spaces, quotes, inline comments)
- Only modify `BACKBONE_CONFIG`, optimizer HPs, or `TIME_BUDGET`
- Never repeat an experiment already in `results.tsv`
- Keep auxiliary loss lambdas at 0 unless you know what you're doing

---

## Supervisor Role (LLM via cron)

Any LLM can act as supervisor — no local Ollama needed. The loop only calls Ollama as a last resort when the queue is empty and it has been stuck for 3+ iterations.

**Cron check (every 10 min):**
1. Read last 30 lines of `autoloop_stdout.log`
2. Read `results.tsv` and `proposal_queue.json`
3. If crashed → restart: `nohup uv run python3 autoloop.py >> autoloop_stdout.log 2>&1 &`
4. If queue < 3 items → add 2–3 new proposals
5. Otherwise → do nothing

See `CLAUDE.md` for the full onboarding guide including setup questions to ask a new user.

---

## Adapting to Your Project

`train.py` and `evaluate.py` are included as working examples from the LFP2Vec project (Whisper-tiny backbone, Data2Vec SSL, 105-class brain region classification). Replace them with your own:

- `train.py` — must save a checkpoint to `run_checkpoint.pt`
- `evaluate.py` — must print exactly one line: `<metric_name>: <float>` (e.g. `probe_acc: 0.0698`)

Everything else (autoloop, dashboard, proposal queue) works unchanged.

---

## Requirements

- Python 3.10+, managed by [uv](https://github.com/astral-sh/uv)
- `git` on PATH (autoloop uses `git commit` / `git checkout` to track experiments)
- PyTorch (CUDA, MPS, or CPU — auto-detected)
- Ollama + `qwen3:8b` (optional — only for stuck-detection fallback)

---

## Example Results (LFP2Vec)

| # | probe_acc | status | description |
|---|-----------|--------|-------------|
| 1–3 | 5.9–6.1% | keep | baseline HEBO HPs |
| 11 | 6.82% | **keep** | mask_time_prob=0.65 + ema_anneal_end_step=3000 |
| **13** | **6.98%** | **keep ★** | **real_signal_frames=150 + TIME_BUDGET=7200** |
| 14–24 | 4.5–6.8% | discard | cosine loss, masking variants, dropout, LR schedules |

Random chance: 0.95% (1/105 classes). Target: >15%.

---

## Citation / Credits

Built at the NYU Neuroinformatics Lab for the Alphabrain project.
If you use this framework, feel free to link back to this repo.
