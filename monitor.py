#!/usr/bin/env python3
"""
Live autoresearch monitor — run in a separate terminal.

    cd autoresearch && uv run python3 monitor.py

Refreshes every 30s. Shows:
  - Current iteration status + time elapsed
  - All results so far (probe_acc, status)
  - What's queued next
  - Estimated time remaining
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

AUTORESEARCH_DIR = Path(__file__).resolve().parent
LOG       = AUTORESEARCH_DIR / "autoloop_stdout.log"
RESULTS   = AUTORESEARCH_DIR / "results.tsv"
QUEUE     = AUTORESEARCH_DIR / "proposal_queue.json"
OVERRIDE  = AUTORESEARCH_DIR / "next_proposal.json"
TRAIN_PY  = AUTORESEARCH_DIR / "train.py"

REFRESH_S = 30


def clear():
    os.system("clear")


def parse_log():
    if not LOG.exists():
        return []
    return LOG.read_text().splitlines()


def current_status(lines):
    """Return (iteration, description, phase, started_at, phase_started_at)."""
    iteration, description, phase = 0, "unknown", "idle"
    started_at = phase_started_at = None

    for line in lines:
        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] ITERATION (\d+): (.+)", line)
        if m:
            iteration = int(m.group(2))
            description = m.group(3)
            phase = "training"
            started_at = m.group(1)
            phase_started_at = m.group(1)

        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] train\.py finished", line)
        if m:
            phase = "evaluating"
            phase_started_at = m.group(1)

        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] probe_acc = ([0-9.]+)", line)
        if m:
            phase = "deciding"
            phase_started_at = m.group(1)

        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] Asking qwen3", line)
        if m:
            phase = "proposing"
            phase_started_at = m.group(1)

        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] (Using queued|Using manual|Next:)", line)
        if m:
            phase = "applying"
            phase_started_at = m.group(1)

        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] Iteration \d+ done", line)
        if m:
            phase = "done"
            phase_started_at = m.group(1)

    return iteration, description, phase, started_at, phase_started_at


def elapsed(time_str):
    if not time_str:
        return "?"
    now = datetime.now()
    t = datetime.strptime(time_str, "%H:%M:%S").replace(
        year=now.year, month=now.month, day=now.day
    )
    if t > now:  # crossed midnight
        t -= timedelta(days=1)
    delta = now - t
    m, s = divmod(int(delta.total_seconds()), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def parse_results():
    if not RESULTS.exists():
        return []
    rows = []
    for line in RESULTS.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 6:
            rows.append({
                "commit": parts[0][:7],
                "metric": parts[1],
                "value": float(parts[2]),
                "status": parts[4],
                "description": parts[5][:55],
            })
    return rows


def read_queue():
    items = []
    if QUEUE.exists():
        try:
            items = json.loads(QUEUE.read_text())
        except Exception:
            pass
    if OVERRIDE.exists():
        try:
            items = [json.loads(OVERRIDE.read_text())] + items
        except Exception:
            pass
    return items


def time_budget():
    if not TRAIN_PY.exists():
        return "?"
    m = re.search(r"^TIME_BUDGET\s*=\s*(\d+)", TRAIN_PY.read_text(), re.MULTILINE)
    return f"{int(m.group(1))//60}min" if m else "?"


def main():
    while True:
        clear()
        lines = parse_log()
        iteration, description, phase, started_at, phase_started_at = current_status(lines)
        results = parse_results()
        queue = read_queue()
        best = max((r["value"] for r in results if r["status"] == "keep"), default=0.0)
        total = len(results)

        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"{'='*65}")
        print(f"  Alphabrain Autoresearch Monitor        {now_str}  (refresh {REFRESH_S}s)")
        print(f"{'='*65}")

        # Current iteration
        phase_icons = {
            "training": "🏋️  training",
            "evaluating": "🔬 evaluating",
            "deciding": "⚖️  deciding",
            "proposing": "🤖 proposing",
            "applying": "✏️  applying",
            "done": "✅ done",
            "idle": "💤 idle",
        }
        icon = phase_icons.get(phase, phase)
        print(f"\n  Iteration #{total+1} (overall #{total+1})   TIME_BUDGET={time_budget()}")
        print(f"  Phase     : {icon}  [{elapsed(phase_started_at)} elapsed]")
        print(f"  Experiment: {description[:60]}")
        if phase == "training":
            print(f"  Training started {elapsed(started_at)} ago — est. {time_budget()} total")

        # Results table
        print(f"\n{'─'*65}")
        print(f"  Results ({total} experiments, best probe_acc={best:.4f} = {best*100:.2f}%)")
        print(f"{'─'*65}")
        print(f"  {'#':>3}  {'probe_acc':>9}  {'status':>8}  description")
        for i, r in enumerate(results[-12:], start=max(1, total-11)):
            marker = " ★" if r["value"] == best else "  "
            status_icon = {"keep": "✅", "discard": "❌", "crash": "💥"}.get(r["status"], "?")
            print(f"  {i:>3}  {r['value']*100:>8.2f}%  {status_icon} {r['status']:>6}  {r['description']}")
        print(f"  Random chance: 0.95% (1/105 classes)  Target: >15%")

        # Queue
        print(f"\n{'─'*65}")
        if queue:
            print(f"  Proposal queue ({len(queue)} pending):")
            for i, q in enumerate(queue[:5]):
                src = "Claude" if i < len(queue) else "qwen3"
                print(f"  [{i+1}] {q.get('description','?')[:60]}")
            if len(queue) > 5:
                print(f"  ... +{len(queue)-5} more")
        else:
            print(f"  Proposal queue: empty — qwen3:8b / deep research will propose")

        print(f"\n{'='*65}")
        print(f"  tail -f autoloop_stdout.log   |   Ctrl+C to exit monitor")
        print(f"{'='*65}")

        time.sleep(REFRESH_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
