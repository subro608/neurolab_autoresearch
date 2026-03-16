#!/usr/bin/env python3
"""
Alphabrain Autoresearch Dashboard — terminal TUI inspired by the screenshot style.

    cd autoresearch && uv run python3 dashboard.py

Refreshes every 15s. Shows:
  - Current iteration / phase / elapsed
  - Full experiment history with probe_acc bars + sparkline
  - Current config (train.py key params)
  - Pending proposal queue
  - Last log lines (live feed)
"""

import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
DIR      = Path(__file__).resolve().parent
LOG      = DIR / "autoloop_stdout.log"
RESULTS  = DIR / "results.tsv"
QUEUE    = DIR / "proposal_queue.json"
OVERRIDE = DIR / "next_proposal.json"
TRAIN_PY = DIR / "train.py"

REFRESH_S = 15
RANDOM_CHANCE = 0.0095   # 1/105 classes
TARGET        = 0.15

# ── ANSI colours ───────────────────────────────────────────────────────────────
R = "\033[0m"
BOLD      = "\033[1m"
DIM       = "\033[2m"
TEAL      = "\033[38;5;43m"
TEAL_B    = "\033[1;38;5;43m"
GREEN     = "\033[38;5;82m"
GREEN_B   = "\033[1;38;5;82m"
YELLOW    = "\033[38;5;220m"
YELLOW_B  = "\033[1;38;5;220m"
RED       = "\033[38;5;203m"
RED_B     = "\033[1;38;5;203m"
GRAY      = "\033[38;5;245m"
WHITE     = "\033[38;5;255m"
WHITE_B   = "\033[1;38;5;255m"
CYAN      = "\033[38;5;159m"
CYAN_B    = "\033[1;38;5;159m"
PURPLE    = "\033[38;5;141m"
BG_DARK   = "\033[48;5;234m"
BG_HEADER = "\033[48;5;237m"

# ── sparkline chars ────────────────────────────────────────────────────────────
SPARK = "▁▂▃▄▅▆▇█"

# ── block bar chars ────────────────────────────────────────────────────────────
BAR_FULL  = "█"
BAR_EMPTY = "░"
BAR_HALF  = "▒"

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def hide_cursor():
    sys.stdout.write("\033[?25l")

def show_cursor():
    sys.stdout.write("\033[?25h")

def term_width():
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 100

# ── data parsers ───────────────────────────────────────────────────────────────

def read_log_lines():
    if not LOG.exists():
        return []
    return LOG.read_text().splitlines()


def parse_current(lines):
    """Return dict with current iteration state."""
    state = dict(iteration=0, description="idle", phase="idle",
                 iter_started=None, phase_started=None, last_probe=None)
    for line in lines:
        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] ITERATION (\d+): (.+)", line)
        if m:
            state.update(iteration=int(m.group(2)), description=m.group(3),
                         phase="training", iter_started=m.group(1), phase_started=m.group(1))
        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] train\.py finished", line)
        if m:
            state.update(phase="evaluating", phase_started=m.group(1))
        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] probe_acc = ([0-9.]+)", line)
        if m:
            state.update(phase="deciding", phase_started=m.group(1),
                         last_probe=float(m.group(2)))
        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] Iteration \d+ done: probe_acc=([0-9.]+)", line)
        if m:
            state.update(phase="done", phase_started=m.group(1),
                         last_probe=float(m.group(2)))
        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] (Using queued|Using manual|Applied)", line)
        if m and state["phase"] == "done":
            state.update(phase="applying", phase_started=m.group(1))
        m = re.search(r"\[autoloop (\d{2}:\d{2}:\d{2})\] Asking", line)
        if m:
            state.update(phase="proposing", phase_started=m.group(1))
    return state


def elapsed_str(time_str, reference=None):
    if not time_str:
        return "?"
    now = reference or datetime.now()
    t = datetime.strptime(time_str, "%H:%M:%S").replace(year=now.year, month=now.month, day=now.day)
    if t > now:
        t -= timedelta(days=1)
    delta = now - t
    m, s = divmod(int(delta.total_seconds()), 60)
    h, m2 = divmod(m, 60)
    if h:
        return f"{h}h{m2:02d}m"
    return f"{m}m{s:02d}s"


def parse_results():
    if not RESULTS.exists():
        return []
    rows = []
    for line in RESULTS.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 6:
            try:
                rows.append(dict(commit=parts[0][:7], metric=parts[1],
                                 value=float(parts[2]), status=parts[4],
                                 description=parts[5]))
            except ValueError:
                pass
    return rows


def read_queue():
    items = []
    if OVERRIDE.exists():
        try:
            items = [json.loads(OVERRIDE.read_text())]
        except Exception:
            pass
    if QUEUE.exists():
        try:
            items += json.loads(QUEUE.read_text())
        except Exception:
            pass
    return items


def read_config():
    cfg = {}
    if not TRAIN_PY.exists():
        return cfg
    txt = TRAIN_PY.read_text()
    def get(pat):
        m = re.search(pat, txt, re.MULTILINE)
        return m.group(1).strip() if m else "?"
    cfg["TIME_BUDGET"]  = get(r"^TIME_BUDGET\s*=\s*(.+?)$")
    cfg["LR"]           = get(r"^LR\s*=\s*(.+?)$")
    cfg["WEIGHT_DECAY"] = get(r"^WEIGHT_DECAY\s*=\s*(.+?)$")
    cfg["BATCH_SIZE"]   = get(r"^BATCH_SIZE\s*=\s*(.+?)$")
    cfg["GRAD_ACCUM"]   = get(r"^GRAD_ACCUM\s*=\s*(.+?)$")
    cfg["WARMUP_STEPS"] = get(r"^WARMUP_STEPS\s*=\s*(.+?)$")
    cfg["ema_decay"]    = get(r'"ema_decay":\s*(.+?),')
    cfg["mask_prob"]    = get(r'"mask_time_prob":\s*(.+?),')
    cfg["mask_len"]     = get(r'"mask_time_length":\s*(.+?),')
    cfg["top_k"]        = get(r'"average_top_k_layers":\s*(.+?),')
    cfg["real_frames"]  = get(r'"real_signal_frames":\s*(.+?),')
    cfg["hidden_do"]    = get(r'"hidden_dropout":\s*(.+?),')
    return cfg


# ── rendering helpers ──────────────────────────────────────────────────────────

def bar(value, max_value, width=20, color=TEAL, empty_color=GRAY):
    if max_value <= 0:
        filled = 0
    else:
        filled = min(width, int(round(value / max_value * width)))
    empty = width - filled
    return f"{color}{BAR_FULL * filled}{empty_color}{BAR_EMPTY * empty}{R}"


def sparkline(values, width=24):
    if not values:
        return GRAY + "─" * width + R
    mn, mx = min(values), max(values)
    if mx == mn:
        return TEAL + "▄" * min(len(values), width) + R
    step = (mx - mn) / (len(SPARK) - 1)
    chars = [SPARK[min(len(SPARK)-1, int((v - mn) / step))] for v in values]
    # keep last `width` chars
    chars = chars[-width:]
    # colour: low=gray, mid=teal, high=green
    result = ""
    for i, c in enumerate(chars):
        idx = len(chars) - len(chars) + i
        v = values[-(len(chars) - i)] if len(chars) <= len(values) else values[i]
        if mx > mn:
            pct = (v - mn) / (mx - mn)
        else:
            pct = 0.5
        if pct < 0.33:
            result += GRAY + c
        elif pct < 0.67:
            result += TEAL + c
        else:
            result += GREEN + c
    return result + R


def phase_indicator(phase):
    icons = {
        "training":   (GREEN_B,  "◉", "TRAINING"),
        "evaluating": (YELLOW_B, "◈", "EVALUATING"),
        "deciding":   (CYAN_B,   "◇", "DECIDING"),
        "proposing":  (PURPLE,   "◆", "PROPOSING"),
        "applying":   (TEAL_B,   "◀", "APPLYING"),
        "done":       (GRAY,     "◎", "DONE"),
        "idle":       (GRAY,     "○", "IDLE"),
    }
    color, icon, label = icons.get(phase, (GRAY, "?", phase.upper()))
    return f"{color}{icon} {label}{R}"


def sep_line(W, char="─", color=GRAY):
    return f"{color}{char * W}{R}"


def section_header(title, W, color=TEAL_B):
    dash = "─" * 2
    space = W - len(title) - 6
    return f"{color}{dash} {WHITE_B}{title}{R} {color}{'─' * max(0,space)}{R}"


def trunc(s, n):
    return s if len(s) <= n else s[:n-1] + "…"

# ── main render ────────────────────────────────────────────────────────────────

def render():
    W = min(term_width(), 110)
    now = datetime.now()
    now_str = now.strftime("%H:%M:%S")

    log_lines  = read_log_lines()
    state      = parse_current(log_lines)
    results    = parse_results()
    queue      = read_queue()
    cfg        = read_config()

    best_keep   = max((r["value"] for r in results if r["status"] == "keep"), default=0.0)
    all_values  = [r["value"] for r in results]
    total_runs  = len(results)
    keeps       = sum(1 for r in results if r["status"] == "keep")
    discards    = sum(1 for r in results if r["status"] == "discard")

    lines = []

    # ── HEADER ──────────────────────────────────────────────────────────────────
    phase_str = phase_indicator(state["phase"])
    iter_str  = f"#{state['iteration']}" if state["iteration"] else "#0"
    elapsed_t = elapsed_str(state["iter_started"])
    budget_s  = cfg.get("TIME_BUDGET", "?").split()[0]
    try:
        budget_h = f"{int(budget_s)//3600}h" if int(budget_s) >= 3600 else f"{int(budget_s)//60}min"
    except Exception:
        budget_h = budget_s

    # Top status bar
    left  = f" {TEAL_B}◆ ALPHABRAIN AUTORESEARCH{R}  {GRAY}│{R}  {GREEN_B}{total_runs}{R}{GRAY} runs{R}  {GRAY}│{R}  {YELLOW_B}{keeps}{R}{GRAY} kept{R}  {GRAY}│{R}  {budget_h} budget"
    right = f"{GRAY}best={GREEN_B}{best_keep*100:.2f}%{R}{GRAY}  target={WHITE}{TARGET*100:.0f}%{R}  {GRAY}{now_str}{R} "
    # strip ANSI for width calc
    def vlen(s):
        return len(re.sub(r'\033\[[0-9;]*m', '', s))
    pad = W - vlen(left) - vlen(right)
    lines.append(BG_HEADER + left + " " * max(0, pad) + right + R)
    lines.append(sep_line(W, "═", TEAL))

    # Iteration block
    iter_label = f"  {TEAL_B}ITER {iter_str}{R}  {phase_str}"
    if state["iter_started"]:
        iter_label += f"  {GRAY}started {state['iter_started']}  elapsed {elapsed_t}{R}"
    lines.append(iter_label)

    desc = trunc(state["description"], W - 10)
    lines.append(f"  {GRAY}└─{R} {WHITE}{desc}{R}")
    lines.append("")

    # ── EXPERIMENTS ─────────────────────────────────────────────────────────────
    lines.append(section_header("EXPERIMENTS", W))

    # Sparkline of all values
    if all_values:
        spark = sparkline(all_values, width=min(total_runs + 2, W - 30))
        lines.append(f"  {GRAY}history {R}{spark}  {GRAY}n={total_runs}{R}")
    lines.append("")

    # Table header
    col_w = W - 52
    lines.append(f"  {GRAY}{'#':>3}  {'probe_acc':>9}  {'bar':<20}  {'st':>5}  {'description'}{R}")
    lines.append(f"  {GRAY}{'─'*3}  {'─'*9}  {'─'*20}  {'─'*5}  {'─'*(col_w)}{R}")

    for i, r in enumerate(results[-15:], start=max(1, total_runs - 14)):
        v   = r["value"]
        pct = f"{v*100:5.2f}%"
        b   = bar(v, TARGET, width=20)
        is_best = (v == best_keep)

        if r["status"] == "keep":
            st_str = f"{GREEN}keep{R}"
            num_c  = GREEN_B if is_best else GREEN
        elif r["status"] == "discard":
            st_str = f"{GRAY}skip{R}"
            num_c  = GRAY
        else:
            st_str = f"{RED}FAIL{R}"
            num_c  = RED

        star = f"{YELLOW_B}★{R}" if is_best else " "
        desc_t = trunc(r["description"], col_w)
        val_c = GREEN_B if is_best else (GREEN if r["status"] == "keep" else GRAY)
        lines.append(f"  {num_c}{i:>3}{R}{star} {val_c}{pct}{R}  {b}  {st_str}  {GRAY}{desc_t}{R}")

    # Current running (if active)
    if state["phase"] in ("training", "evaluating"):
        cur_n = total_runs + 1
        bar_anim = TEAL + "█" * 10 + GRAY + "░" * 10 + R
        lines.append(f"  {TEAL_B}{cur_n:>3}{R}  {TEAL}ACTIVE …{R}  {bar_anim}  {TEAL}run {R}  {GRAY}{trunc(state['description'], col_w)}{R}")

    lines.append("")

    # Progress toward target
    pct_of_target = best_keep / TARGET
    prog_bar = bar(best_keep, TARGET, width=W - 30, color=GREEN if pct_of_target >= 0.5 else TEAL)
    lines.append(f"  {GRAY}Progress to target ({TARGET*100:.0f}%): {R}{prog_bar} {WHITE_B}{pct_of_target*100:.0f}%{R}")
    lines.append(f"  {GRAY}Random chance: {RANDOM_CHANCE*100:.2f}%  Best: {GREEN_B}{best_keep*100:.2f}%{R}  {GRAY}Target: {WHITE}{TARGET*100:.0f}%{R}  {GRAY}Gap: {RED_B}{(TARGET-best_keep)*100:.2f}%{R}")
    lines.append("")

    # ── CONFIG ──────────────────────────────────────────────────────────────────
    lines.append(section_header("CONFIG  (current train.py)", W))
    lr_s = cfg.get("LR", "?")
    try:
        lr_f = float(lr_s.split()[0].split('#')[0])
        lr_s = f"{lr_f:.2e}"
    except Exception:
        lr_s = lr_s[:12]
    wd_s = cfg.get("WEIGHT_DECAY","?").split()[0]
    try:
        wd_s = f"{float(wd_s):.3f}"
    except Exception:
        pass

    row1 = (f"  {TEAL}LR{R}={CYAN}{lr_s}{R}  "
            f"{TEAL}WD{R}={CYAN}{wd_s}{R}  "
            f"{TEAL}BS{R}={CYAN}{cfg.get('BATCH_SIZE','?').split()[0]}{R}  "
            f"{TEAL}ACCUM{R}={CYAN}{cfg.get('GRAD_ACCUM','?').split()[0]}{R}  "
            f"{TEAL}WARMUP{R}={CYAN}{cfg.get('WARMUP_STEPS','?').split()[0]}{R}  "
            f"{TEAL}BUDGET{R}={CYAN}{cfg.get('TIME_BUDGET','?').split()[0]}{R}s")
    row2 = (f"  {TEAL}ema_decay{R}={CYAN}{cfg.get('ema_decay','?')[:8]}{R}  "
            f"{TEAL}mask_prob{R}={CYAN}{cfg.get('mask_prob','?')}{R}  "
            f"{TEAL}mask_len{R}={CYAN}{cfg.get('mask_len','?')}{R}  "
            f"{TEAL}top_k_layers{R}={CYAN}{cfg.get('top_k','?')}{R}  "
            f"{TEAL}real_frames{R}={CYAN}{cfg.get('real_frames','?')}{R}")
    lines.append(row1)
    lines.append(row2)
    lines.append("")

    # ── QUEUE ───────────────────────────────────────────────────────────────────
    lines.append(section_header(f"QUEUE  ({len(queue)} pending)", W))
    if queue:
        for i, q in enumerate(queue[:6]):
            tag = f"{TEAL_B}[{i+1}]{R}" if i == 0 else f"{GRAY}[{i+1}]{R}"
            d = trunc(q.get("description", "?"), W - 8)
            nc = q.get("changes", [])
            change_hint = f"  {DIM}{GRAY}({len(nc)} change{'s' if len(nc)!=1 else ''}){R}"
            lines.append(f"  {tag} {WHITE if i==0 else GRAY}{d}{R}{change_hint}")
        if len(queue) > 6:
            lines.append(f"  {GRAY}   … +{len(queue)-6} more proposals{R}")
    else:
        lines.append(f"  {YELLOW}  ⚠  queue empty — qwen3:8b will auto-propose{R}")
    lines.append("")

    # ── LOG ─────────────────────────────────────────────────────────────────────
    lines.append(section_header("LOG", W))
    tail = log_lines[-8:] if log_lines else ["(no log)"]
    for ll in tail:
        # highlight key events
        if "ITERATION" in ll:
            ll_col = TEAL_B
        elif "probe_acc" in ll:
            ll_col = GREEN_B
        elif "Starting train.py" in ll:
            ll_col = YELLOW
        elif "keep" in ll or "improvement" in ll.lower():
            ll_col = GREEN
        elif "discard" in ll or "No improvement" in ll:
            ll_col = GRAY
        elif "crash" in ll.lower() or "error" in ll.lower() or "Error" in ll:
            ll_col = RED_B
        else:
            ll_col = GRAY
        lines.append(f"  {ll_col}{trunc(ll, W-3)}{R}")

    # ── FOOTER ──────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(sep_line(W, "═", TEAL))
    footer = f"  {GRAY}refresh {REFRESH_S}s  │  Ctrl+C to exit  │  tail -f autoloop_stdout.log  │  uv run python3 autoloop.py{R}"
    lines.append(footer)

    return "\n".join(lines)


# ── entrypoint ─────────────────────────────────────────────────────────────────

def handle_exit(sig, frame):
    show_cursor()
    print()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    hide_cursor()
    try:
        while True:
            clear_screen()
            print(render())
            time.sleep(REFRESH_S)
    finally:
        show_cursor()


if __name__ == "__main__":
    main()
