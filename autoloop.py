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
       c. deep research        — Claude CLI with WebSearch (arxiv + HuggingFace), triggered when stuck
       d. Claude CLI           — standard autonomous proposal (no search)
  5. Edit train.py (apply changes)
  6. → repeat

Stuck detection: if last 3 results are all discards or same description,
autoloop queries arxiv + HuggingFace Papers directly via search_server.py and
injects results into the LLM prompt — works with any AUTOLOOP_LLM backend.

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
CLAUDE_CLI = Path.home() / ".local/bin/claude"
ARXIV_API  = "http://export.arxiv.org/api/query"

# LLM backend for autonomous proposal generation (fallback when queue is empty).
# Set via env var AUTOLOOP_LLM. Options:
#   claude   — Claude Code CLI (claude --print). Default if claude is on PATH.
#   codex    — OpenAI Codex CLI (codex exec). Requires codex on PATH + login.
#   openai   — OpenAI API directly via OPENAI_API_KEY env var (gpt-4o).
#   ollama   — Local Ollama server. Set OLLAMA_MODEL (default: qwen3:8b).
#   none     — Disable LLM fallback entirely; use lr×2 hardcoded fallback.
AUTOLOOP_LLM   = os.environ.get("AUTOLOOP_LLM", "claude")
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

# Trigger deep research after this many consecutive non-improvements
STUCK_THRESHOLD = 3

PROPOSE_SYSTEM = """You are an autonomous ML research assistant for the Alphabrain LFP brain region decoding project.
Your job: propose and implement the single best next experiment to improve linear probe accuracy.

Setup:
- Dataset: compact_atlas_5k, 105 brain region classes (atlas_depth=9), ~5000 train/val samples
- Backbone: whisper_distill — EMA teacher-student (Data2Vec-style) with distance contrastive loss
- Objective: dist_lambda_raw=1.0 (distance contrastive is the main SSL loss), ssl_lambda_raw=0.0
- Device: Apple M3 Max MPS. BATCH_SIZE=16 (hard MPS limit), GRAD_ACCUM=4, TIME_BUDGET=7200s
- Current best probe_acc: 6.98% (random chance=0.95%, target=15%+)
- Key distc params: dist_temperature, dist_sigma (spatial bandwidth in µm), num_negatives

Rules:
- Only modify BACKBONE_CONFIG values, optimizer HPs (LR/WD/WARMUP_STEPS), or TIME_BUDGET
- Do NOT add or change auxiliary losses — xyz_lambda, distpred_lambda_raw must stay 0
- dist_lambda_raw must stay 1.0 (it is the main objective, not auxiliary)
- Do NOT change backbone name, data paths, atlas_depth, or BATCH_SIZE
- Do NOT repeat any experiment already in results.tsv
- The "old" string must match train.py EXACTLY (copy-paste, including spaces and inline comments)
- Output EXACTLY this JSON (no markdown, no explanation):
{
  "description": "one-line description of the change and why",
  "changes": [
    {"old": "exact string to replace", "new": "replacement string"}
  ]
}
"""

RESEARCH_SYSTEM = """You are an autonomous ML research assistant for the Alphabrain LFP brain region decoding project.
You will search arxiv and HuggingFace papers to find novel ideas, then implement ONE as a train.py change.

Setup:
- Dataset: compact_atlas_5k, 105 brain region classes (atlas_depth=9), ~5000 train/val samples
- Backbone: whisper_distill — EMA teacher-student (Data2Vec-style) with distance contrastive loss
- Objective: dist_lambda_raw=1.0, dist_temperature controls contrastive sharpness, dist_sigma is spatial bandwidth
- Device: Apple M3 Max MPS. BATCH_SIZE=16, GRAD_ACCUM=4, TIME_BUDGET=7200s
- Current best probe_acc: 6.98% (random=0.95%, target=15%+)

Rules:
- Only modify BACKBONE_CONFIG values, optimizer HPs (LR/WD/WARMUP_STEPS), or TIME_BUDGET
- Do NOT add auxiliary losses — xyz_lambda, distpred_lambda_raw must stay 0
- dist_lambda_raw must stay 1.0
- The idea must be grounded in a paper you found
- "old" must match train.py EXACTLY
- Output EXACTLY this JSON (no markdown):
{
  "description": "one-line description citing the paper/idea",
  "changes": [
    {"old": "exact string to replace", "new": "replacement string"}
  ]
}
"""

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

# Search functions from the FastMCP server — called directly so any LLM backend
# can benefit from arxiv/HuggingFace results injected into the prompt.
try:
    from search_server import _search_arxiv, _search_huggingface_papers
except ImportError:
    # Fallback stubs if fastmcp/httpx not yet installed
    def _search_arxiv(query: str, max_results: int = 5) -> str:          # type: ignore[misc]
        return f"[search_server not available — run: uv add fastmcp httpx]"
    def _search_huggingface_papers(query: str, max_results: int = 5) -> str:  # type: ignore[misc]
        return f"[search_server not available — run: uv add fastmcp httpx]"


# ── LLM backend — pluggable via AUTOLOOP_LLM env var ─────────────────────────

def _llm_call_claude(prompt: str, system: str, allow_web_search: bool,
                     timeout: int) -> str | None:
    """Claude Code CLI backend: `claude --print`."""
    cmd = [str(CLAUDE_CLI), "--print", "--output-format", "text", "--model", "sonnet"]
    if system:
        cmd += ["--append-system-prompt", system]
    if allow_web_search:
        cmd += ["--allowedTools", "WebSearch"]
    cmd.append(prompt)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=AUTORESEARCH_DIR)
        if r.returncode != 0:
            log(f"claude call failed (exit {r.returncode}): {r.stderr[:300]}")
            return None
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        log(f"claude call timed out after {timeout}s")
        return None
    except Exception as e:
        log(f"claude call error: {e}")
        return None


def _llm_call_openai(prompt: str, system: str, allow_web_search: bool,
                     timeout: int) -> str | None:
    """OpenAI backend: gpt-4o via OPENAI_API_KEY. Web search via responses API."""
    try:
        from openai import OpenAI  # pip install openai
        client = OpenAI(timeout=timeout)
        if allow_web_search:
            # Responses API with built-in web search tool
            resp = client.responses.create(
                model="gpt-4o",
                instructions=system or None,
                input=prompt,
                tools=[{"type": "web_search_preview"}],
            )
            return resp.output_text.strip()
        else:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    *([ {"role": "system", "content": system}] if system else []),
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
    except ImportError:
        log("openai package not installed — run: uv add openai")
        return None
    except Exception as e:
        log(f"openai call error: {e}")
        return None


def _llm_call_codex(prompt: str, system: str, allow_web_search: bool,
                    timeout: int) -> str | None:
    """OpenAI Codex CLI backend: `codex exec`. Requires codex on PATH + login."""
    import shutil, tempfile
    codex_bin = shutil.which("codex") or "codex"
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    # Write output to a temp file via -o flag (cleanest way to capture final response)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        out_file = f.name
    try:
        cmd = [
            codex_bin, "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ephemeral",
            "-o", out_file,
            full_prompt,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=AUTORESEARCH_DIR)
        if r.returncode != 0:
            log(f"codex call failed (exit {r.returncode}): {r.stderr[:300]}")
            return None
        if os.path.exists(out_file):
            result = open(out_file).read().strip()
            return result if result else r.stdout.strip()
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        log(f"codex call timed out after {timeout}s")
        return None
    except Exception as e:
        log(f"codex call error: {e}")
        return None
    finally:
        try:
            os.unlink(out_file)
        except OSError:
            pass


def _llm_call_ollama(prompt: str, system: str, timeout: int) -> str | None:
    """Ollama backend: local server via REST API. No web search support."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        log(f"Ollama not reachable at {OLLAMA_URL} — is it running?")
        return None
    except Exception as e:
        log(f"ollama call error: {e}")
        return None


def claude_call(prompt: str, system: str = "", allow_web_search: bool = False,
                timeout: int = 180) -> str | None:
    """Dispatch to the configured LLM backend (AUTOLOOP_LLM env var). Retries up to 3 times."""
    if AUTOLOOP_LLM == "none":
        log("LLM disabled (AUTOLOOP_LLM=none) — skipping proposal generation")
        return None
    if AUTOLOOP_LLM not in ("claude", "codex", "openai", "ollama"):
        log(f"Unknown AUTOLOOP_LLM={AUTOLOOP_LLM!r} — valid: claude, codex, openai, ollama, none")
        return None
    if AUTOLOOP_LLM == "codex" and allow_web_search:
        log("codex backend does not support web search — skipping research_propose")
        return None

    for attempt in range(1, 4):  # 3 attempts total
        if attempt > 1:
            log(f"LLM retry {attempt}/3 ...")
            time.sleep(5 * (attempt - 1))  # 5s, then 10s between retries
        if AUTOLOOP_LLM == "claude":
            result = _llm_call_claude(prompt, system, allow_web_search, timeout)
        elif AUTOLOOP_LLM == "codex":
            result = _llm_call_codex(prompt, system, allow_web_search, timeout)
        elif AUTOLOOP_LLM == "openai":
            result = _llm_call_openai(prompt, system, allow_web_search, timeout)
        elif AUTOLOOP_LLM == "ollama":
            result = _llm_call_ollama(prompt, system, timeout)
        if result:
            return result
        log(f"LLM attempt {attempt}/3 returned nothing")

    log("All 3 LLM attempts failed")
    return None


def _parse_proposal_json(content: str) -> dict | None:
    """Extract and parse the first JSON object from a Claude response."""
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        log(f"No JSON found in response: {content[:200]}")
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        log(f"JSON parse error: {e} — content: {content[:200]}")
        return None


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


def claude_propose(results_text: str, train_text: str) -> dict | None:
    """Ask Claude CLI to propose the next experiment (no web search)."""
    log("Asking Claude for next experiment proposal...")
    prompt = f"""Current results.tsv (experiment history):
{results_text}

Current train.py BACKBONE_CONFIG and optimizer HPs:
{train_text}

Propose the single best next experiment to improve probe_acc.
Reason about what patterns you see in results — what worked, what didn't, what hasn't been tried.
Output ONLY valid JSON, no markdown, no explanation."""
    content = claude_call(prompt, system=PROPOSE_SYSTEM, allow_web_search=False)
    if not content:
        return None
    result = _parse_proposal_json(content)
    if result:
        log(f"Claude proposal: {result.get('description', '?')}")
    return result


def research_propose(results_text: str, train_text: str) -> dict | None:
    """Search arxiv + HuggingFace Papers directly, then ask ANY LLM backend to propose.

    Search is done by autoloop itself (backend-agnostic) — results are injected as
    text into the prompt, so codex/openai/claude all get the same paper context.
    """
    log("Entering deep research mode — querying arxiv + HuggingFace Papers...")

    # Step 1: ask the LLM what to search for (short response, no search needed)
    query_prompt = f"""Experiment history (results.tsv):
{results_text}

Current BACKBONE_CONFIG:
{train_text}

We are stuck — last several experiments did not improve probe_acc.
Output ONLY a JSON array of 2-3 targeted arxiv/HuggingFace search queries that might find
novel ideas to unblock us. Be specific (e.g. "distance contrastive loss spatial embeddings").
Example: ["query one", "query two", "query three"]"""

    queries_raw = claude_call(query_prompt, system=RESEARCH_SYSTEM, allow_web_search=False, timeout=60)
    queries: list[str] = []
    if queries_raw:
        m = re.search(r"\[.*?\]", queries_raw, re.DOTALL)
        if m:
            try:
                queries = [q for q in json.loads(m.group(0)) if isinstance(q, str)]
            except json.JSONDecodeError:
                pass
    if not queries:
        # Fallback queries based on current objective
        queries = [
            "distance contrastive self-supervised learning spatial coordinates",
            "contrastive SSL neural signals brain region classification",
        ]
    log(f"Search queries: {queries}")

    # Step 2: run searches directly — no LLM tool calls needed
    search_results = ""
    for q in queries[:3]:
        arxiv_hits = _search_arxiv(q, max_results=3)
        hf_hits    = _search_huggingface_papers(q, max_results=3)
        search_results += f"\n\n### Arxiv: \"{q}\"\n{arxiv_hits}"
        search_results += f"\n\n### HuggingFace Papers: \"{q}\"\n{hf_hits}"

    # Step 3: ask LLM to propose based on injected search results
    propose_prompt = f"""You are improving a distance contrastive SSL model for LFP brain region decoding.

Experiment history (results.tsv):
{results_text}

Current BACKBONE_CONFIG + optimizer:
{train_text}

Papers found by search:
{search_results}

From the papers above, extract ONE concrete idea that:
- Has NOT been tried already (check results.tsv carefully)
- Can be implemented as a BACKBONE_CONFIG or optimizer HP change in train.py
- Is grounded in a specific paper found above

Output ONLY valid JSON (no markdown):
{{
  "description": "one-line description citing the paper",
  "changes": [
    {{"old": "exact string in train.py", "new": "replacement string"}}
  ]
}}"""
    content = claude_call(propose_prompt, system=RESEARCH_SYSTEM, allow_web_search=False, timeout=180)
    if not content:
        log("Research propose failed — falling back to claude_propose")
        return claude_propose(results_text, train_text)
    result = _parse_proposal_json(content)
    if result:
        log(f"Research proposal: {result.get('description', '?')}")
        return result
    log("Research propose returned no valid JSON — falling back to claude_propose")
    return claude_propose(results_text, train_text)


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
        # Priority: proposal_queue.json (array, pop front) > next_proposal.json (single) > Claude
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
            proposal = claude_propose(
                results_text=RESULTS.read_text(),
                train_text=TRAIN_PY.read_text(),
            )

        if proposal is None:
            log("Claude failed to propose — using fallback: increase LR by 2x")
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
