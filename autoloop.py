#!/usr/bin/env python3
"""
Autonomous autoresearch loop — runs forever without human input.

Cycle:
  1. Train  (train.py, TIME_BUDGET seconds)
  2. Evaluate (evaluate.py --metric probe_acc)
  3. Decide  keep (git commit) or discard (git reset)
  4. Propose next experiment:
       a. proposal_queue.json  — pre-loaded manual/Claude proposals (highest priority)
       b. next_proposal.json   — single manual override
       c. deep research        — arxiv search → qwen3:8b (triggered when stuck)
       d. qwen3:8b             — standard autonomous proposal
  5. Edit train.py (apply changes)
  6. → repeat

Stuck detection: if last 3 results are all discards or same description,
automatically searches arxiv and feeds paper abstracts to qwen3:8b for novel ideas.

Run:
    cd autoresearch
    uv run python3 autoloop.py

Stop: Ctrl+C at any time. Current experiment finishes cleanly.
"""

import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

AUTORESEARCH_DIR = Path(__file__).resolve().parent
TRAIN_PY   = AUTORESEARCH_DIR / "train.py"
EVAL_PY    = AUTORESEARCH_DIR / "evaluate.py"
RESULTS    = AUTORESEARCH_DIR / "results.tsv"
EXPERIMENT = AUTORESEARCH_DIR / "experiment.md"
RUN_LOG    = AUTORESEARCH_DIR / "run.log"
CHECKPOINT = AUTORESEARCH_DIR / "run_checkpoint.pt"
UV         = Path.home() / ".local/bin/uv"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"
ARXIV_API  = "http://export.arxiv.org/api/query"

# Trigger deep research after this many consecutive non-improvements
STUCK_THRESHOLD = 3

ARXIV_QUERIES = [
    "Data2Vec EMA teacher student self-supervised learning improvement audio",
    "masked prediction self-supervised learning time series transformer architecture",
    "EEG LFP neural signal self-supervised representation learning transformer",
    "BYOL momentum encoder SSL linear probe improvement techniques",
    "whisper encoder fine-tuning self-supervised downstream classification",
]

# ── helpers ─────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[autoloop {ts}] {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=AUTORESEARCH_DIR, **kwargs)


def git(*args) -> str:
    r = run(["git"] + list(args), capture_output=True, text=True)
    return r.stdout.strip()


def current_commit() -> str:
    return git("rev-parse", "--short", "HEAD")


# ── step 1: train ────────────────────────────────────────────────────────────

def do_train() -> bool:
    log("Starting train.py ...")
    RUN_LOG.write_text("")  # clear previous log
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r = run(
        [str(UV), "run", "python3", "train.py"],
        env=env,
        stdout=open(RUN_LOG, "w"),
        stderr=subprocess.STDOUT,
    )
    if r.returncode != 0:
        log(f"train.py FAILED (exit {r.returncode})")
        return False
    log("train.py finished")
    return True


# ── step 2: evaluate ─────────────────────────────────────────────────────────

def do_evaluate() -> float | None:
    log("Running evaluate.py --metric probe_acc ...")
    eval_log = AUTORESEARCH_DIR / "eval.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r = run(
        [str(UV), "run", "python3", "evaluate.py",
         "--checkpoint", str(CHECKPOINT), "--metric", "probe_acc"],
        env=env,
        stdout=open(eval_log, "w"),
        stderr=subprocess.STDOUT,
    )
    output = eval_log.read_text()
    # append to run.log for record
    with open(RUN_LOG, "a") as f:
        f.write("\n--- evaluate.py output ---\n")
        f.write(output)

    match = re.search(r"^probe_acc:\s*([0-9.]+)", output, re.MULTILINE)
    if not match:
        log("Could not parse probe_acc from eval output")
        log(output[-500:])
        return None
    acc = float(match.group(1))
    log(f"probe_acc = {acc:.6f}")
    return acc


# ── step 3: keep / discard ───────────────────────────────────────────────────

def read_best_probe_acc() -> float:
    """Read the best probe_acc from results.tsv so far."""
    if not RESULTS.exists():
        return 0.0
    best = 0.0
    for line in RESULTS.read_text().splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == "probe_acc" and parts[4] == "keep":
            try:
                best = max(best, float(parts[2]))
            except ValueError:
                pass
    return best


def parse_val_loss() -> float:
    text = RUN_LOG.read_text()
    m = re.search(r"^val_loss:\s*([0-9.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else 0.0


def parse_training_seconds() -> float:
    text = RUN_LOG.read_text()
    m = re.search(r"^training_seconds:\s*([0-9.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else 0.0


def log_result(commit: str, metric: str, value: float, status: str, description: str):
    if not RESULTS.exists():
        RESULTS.write_text("commit\tmetric\tvalue\tpeak_vram_mb\tstatus\tdescription\n")
    with open(RESULTS, "a") as f:
        f.write(f"{commit}\t{metric}\t{value:.6f}\t0.0\t{status}\t{description}\n")


def do_decide(probe_acc: float, description: str) -> bool:
    """Returns True if we keep the experiment."""
    commit = current_commit()
    best = read_best_probe_acc()
    improved = probe_acc > best

    if improved:
        log(f"IMPROVED {best:.4f} → {probe_acc:.4f} — keeping")
        msg = f"{description} | probe_acc={probe_acc:.4f}"
        run(["git", "add", "train.py"])
        run(["git", "commit", "-m", msg,
             "--author", "autoloop <autoloop@alphabrain>"])
        log_result(current_commit(), "probe_acc", probe_acc, "keep", description)
    else:
        log(f"No improvement ({probe_acc:.4f} ≤ best {best:.4f}) — discarding")
        run(["git", "checkout", "train.py"])  # revert to last committed version
        log_result(commit, "probe_acc", probe_acc, "discard", description)

    return improved


# ── step 4+5: propose + edit ─────────────────────────────────────────────────

import requests  # noqa: E402  (stdlib-compat, always available via uv)


# ── arxiv search + stuck detection ───────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 4) -> list[dict]:
    """Search arxiv and return list of {title, abstract}."""
    try:
        resp = requests.get(ARXIV_API, params={
            "search_query": query,
            "max_results": max_results,
            "sortBy": "relevance",
        }, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("a:entry", ns):
            title    = entry.find("a:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("a:summary", ns).text.strip().replace("\n", " ")[:600]
            papers.append({"title": title, "abstract": abstract})
        return papers
    except Exception as e:
        log(f"arxiv search failed: {e}")
        return []


def is_stuck() -> bool:
    """Return True if last STUCK_THRESHOLD results are all discards or repeating descriptions."""
    if not RESULTS.exists():
        return False
    lines = [l for l in RESULTS.read_text().splitlines()[1:] if l.strip()]
    if len(lines) < STUCK_THRESHOLD:
        return False
    recent = lines[-STUCK_THRESHOLD:]
    statuses     = [l.split("\t")[4] for l in recent if len(l.split("\t")) > 4]
    descriptions = [l.split("\t")[5] for l in recent if len(l.split("\t")) > 5]
    all_discards = all(s in ("discard", "crash") for s in statuses)
    all_same_desc = len(set(descriptions)) == 1
    return all_discards or all_same_desc


RESEARCH_SYSTEM = """You are an autonomous ML research assistant for the Alphabrain LFP brain region decoding project.
You have been given summaries of recent research papers. Your job is to extract ONE concrete experiment idea
from these papers and implement it as a change to train.py.

Dataset: compact_atlas_5k, 105 brain region classes (atlas_depth=9), ~5000 train/val samples.
Backbone: whisper_distill (EMA teacher-student, MSE regression on masked positions, collapse-resistant).
Device: Apple M3 Max MPS. batch_size=16 (MPS limit), GRAD_ACCUM=4, TIME_BUDGET=1800s.
Current best probe_acc: ~6% (random=0.95%, target=15%+).

Rules:
- Only modify BACKBONE_CONFIG values, optimizer HPs (LR/WD/WARMUP), or TIME_BUDGET.
- Do NOT add auxiliary losses (xyz_lambda, dist_lambda_raw, distpred_lambda_raw must stay 0).
- Do NOT change backbone name, data paths, atlas_depth, or batch_size.
- The idea must come directly from one of the provided papers.
- Output EXACTLY this JSON (no markdown, no explanation):
{
  "description": "one-line description citing the paper",
  "changes": [
    {"old": "exact string to replace", "new": "replacement string"}
  ]
}
"""

def research_propose(results_text: str, train_text: str) -> dict | None:
    """Search arxiv for novel ideas, then ask qwen3:8b to implement one."""
    log("Entering deep research mode — searching arxiv...")
    all_papers = []
    for q in ARXIV_QUERIES[:3]:  # 3 queries, 4 papers each = up to 12 papers
        papers = search_arxiv(q, max_results=4)
        all_papers.extend(papers)
        if papers:
            log(f"  arxiv: '{q[:50]}...' → {len(papers)} papers")

    if not all_papers:
        log("arxiv search returned nothing — falling back to ollama_propose")
        return ollama_propose(results_text, train_text)

    # De-duplicate by title
    seen, unique = set(), []
    for p in all_papers:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique.append(p)
    all_papers = unique[:10]

    paper_text = "\n\n".join(
        f"[{i+1}] {p['title']}\n{p['abstract']}" for i, p in enumerate(all_papers)
    )

    user_msg = f"""Recent papers on SSL improvement:

{paper_text}

Current results.tsv:
{results_text}

Current train.py:
{train_text}

Based on one of the above papers, propose the single most promising change to improve probe_acc.
Output only JSON.
"""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": RESEARCH_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            log(f"Research propose returned non-JSON: {content[:200]}")
            return None
        result = json.loads(m.group(0))
        log(f"Research proposal: {result.get('description','?')}")
        return result
    except Exception as e:
        log(f"Research propose failed: {e}")
        return None


PROPOSE_SYSTEM = """You are an autonomous ML research assistant for the Alphabrain LFP brain region decoding project.
Your job: propose and implement the single best next experiment to improve linear probe accuracy.

Dataset: compact_atlas_5k, 105 brain region classes (atlas_depth=9), ~5000 train/val samples.
Backbone: whisper_distill (EMA teacher-student, collapse-resistant MSE loss). NOT contrastive.
Device: Apple M3 Max MPS. batch_size=16 (MPS limit), GRAD_ACCUM=4, TIME_BUDGET=1800s.

Rules:
- Only modify BACKBONE_CONFIG values, optimizer HPs (LR/WD/WARMUP), or TIME_BUDGET.
- Do NOT change the backbone name, data paths, atlas_depth, or batch_size.
- Output EXACTLY this JSON (no markdown, no explanation):
{
  "description": "one-line description of the change",
  "changes": [
    {"old": "exact string to replace", "new": "replacement string"}
  ]
}
"""

def ollama_propose(results_text: str, train_text: str) -> dict | None:
    """Ask qwen3:8b to propose the next experiment. Returns {description, changes}."""
    user_msg = f"""
Current results.tsv:
{results_text}

Current train.py:
{train_text}

Propose the single best next change. Output only JSON.
"""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": PROPOSE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()
        # Strip markdown code fences if present
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        # Extract first JSON object
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            log(f"Ollama returned non-JSON: {content[:200]}")
            return None
        return json.loads(m.group(0))
    except Exception as e:
        log(f"Ollama propose failed: {e}")
        return None


def apply_changes(changes: list[dict]) -> bool:
    """Apply old→new string replacements to train.py."""
    text = TRAIN_PY.read_text()
    for ch in changes:
        old, new = ch.get("old", ""), ch.get("new", "")
        if old not in text:
            log(f"WARNING: could not find string to replace: {old!r}")
            return False
        text = text.replace(old, new, 1)
    TRAIN_PY.write_text(text)
    return True


# ── update experiment.md ─────────────────────────────────────────────────────

def update_experiment_md(iteration: int, description: str, probe_acc: float,
                          val_loss: float, status: str):
    content = EXPERIMENT.read_text() if EXPERIMENT.exists() else ""

    # Update the completed experiments table
    row = f"| {iteration} | {description} | {probe_acc:.4f} | {val_loss:.6f} | {status} |"

    # Replace the placeholder row
    if "| — | — |" in content:
        content = content.replace(
            "| — | — | — | — | — |",
            f"| ID | Description | probe_acc | val_loss | Status |\n|----|-------------|-----------|----------|--------|\n{row}"
        )
    else:
        # Append to the table
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "| Status |" in line and i + 1 < len(lines) and lines[i+1].startswith("|---"):
                # Find end of table
                j = i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                lines.insert(j, row)
                break
        content = "\n".join(lines)

    EXPERIMENT.write_text(content)


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    log("Autoresearch loop starting")
    log(f"Working dir: {AUTORESEARCH_DIR}")

    # Initialise results.tsv if missing
    if not RESULTS.exists():
        RESULTS.write_text("commit\tmetric\tvalue\tpeak_vram_mb\tstatus\tdescription\n")

    iteration = 0
    description = "baseline whisper-tiny HPs from HEBO trial c0cfdf58"

    while True:
        iteration += 1
        log(f"{'='*60}")
        log(f"ITERATION {iteration}: {description}")
        log(f"{'='*60}")

        # 1. Train
        train_ok = do_train()
        if not train_ok:
            log("Training crashed — logging and moving on")
            log_result(current_commit(), "probe_acc", 0.0, "crash", description)
            # revert any changes, propose next
        else:
            # 2. Evaluate
            probe_acc = do_evaluate()
            val_loss  = parse_val_loss()

            if probe_acc is None:
                log("Eval crashed — logging and moving on")
                log_result(current_commit(), "probe_acc", 0.0, "crash", description)
                status = "crash"
                probe_acc = 0.0
            else:
                # 3. Decide keep/discard
                kept = do_decide(probe_acc, description)
                status = "keep" if kept else "discard"

            update_experiment_md(iteration, description, probe_acc, val_loss, status)
            log(f"Iteration {iteration} done: probe_acc={probe_acc:.4f} status={status}")

        # 4+5. Propose next experiment
        # Priority: proposal_queue.json (array, pop front) > next_proposal.json (single) > ollama
        queue_file    = AUTORESEARCH_DIR / "proposal_queue.json"
        override_file = AUTORESEARCH_DIR / "next_proposal.json"
        if queue_file.exists():
            queue = json.loads(queue_file.read_text())
            if queue:
                proposal = queue.pop(0)
                log(f"Using queued proposal ({len(queue)} remaining): {proposal.get('description','?')}")
                queue_file.write_text(json.dumps(queue, indent=2))
            else:
                queue_file.unlink()
                proposal = None
        elif override_file.exists():
            log("Using manual override from next_proposal.json")
            proposal = json.loads(override_file.read_text())
            override_file.unlink()
        elif is_stuck():
            log(f"Stuck detected ({STUCK_THRESHOLD} consecutive discards/repeats) — entering deep research mode")
            proposal = research_propose(
                results_text=RESULTS.read_text(),
                train_text=TRAIN_PY.read_text(),
            )
        else:
            log("Asking qwen3:8b for next experiment...")
            proposal = ollama_propose(
                results_text=RESULTS.read_text(),
                train_text=TRAIN_PY.read_text(),
            )

        if proposal is None:
            log("Ollama failed to propose — using fallback: increase LR by 2x")
            proposal = {
                "description": "lr x2 fallback",
                "changes": []
            }

        description = proposal.get("description", "unknown change")
        changes     = proposal.get("changes", [])
        log(f"Next: {description}")

        if changes:
            ok = apply_changes(changes)
            if not ok:
                log("Failed to apply changes — skipping to next proposal")
                continue
            log(f"Applied {len(changes)} change(s) to train.py")
        else:
            log("No changes to apply (empty changes list)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped by user (Ctrl+C)")
        sys.exit(0)
