#!/usr/bin/env python3
"""
Local LLM subagent using Ollama — handles token-intensive tasks so Claude doesn't have to.

Usage (Claude calls this via Bash):
    python3 ollama_agent.py --task "analyze run.log and tell me if val_loss improved"
    python3 ollama_agent.py --task "propose a change to train.py to fix SSL collapse" --context train.py
    python3 ollama_agent.py --task "edit train.py: change xyz_lambda to 0.1" --edit train.py
    python3 ollama_agent.py --task "summarize the top 5 results from results.tsv"

The agent has access to tools:
    read_file   — read any local file
    write_file  — write/overwrite a file
    run_command — run a shell command (safe subset only)
    list_files  — list files in a directory

Model: qwen3:8b (default) or llama3.1 (--model llama3.1)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"

AUTORESEARCH_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Tools available to the local LLM
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to file"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "Full content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a safe read-only shell command (grep, tail, cat, wc, head, git log, git diff, git status). No training runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        },
    },
]

# Safe commands whitelist — no training runs, no destructive ops
SAFE_PREFIXES = ("grep", "tail", "head", "cat", "wc", "ls", "git log", "git diff",
                 "git status", "git branch", "echo", "find", "awk", "sed", "sort",
                 "python3 -c", "python3 evaluate.py")

def execute_tool(name: str, args: dict) -> str:
    if name == "read_file":
        path = Path(args["path"])
        if not path.is_absolute():
            path = AUTORESEARCH_DIR / path
        try:
            return path.read_text()
        except Exception as e:
            return f"ERROR: {e}"

    elif name == "write_file":
        path = Path(args["path"])
        if not path.is_absolute():
            path = AUTORESEARCH_DIR / path
        try:
            path.write_text(args["content"])
            return f"OK: wrote {len(args['content'])} chars to {path}"
        except Exception as e:
            return f"ERROR: {e}"

    elif name == "run_command":
        cmd = args["command"].strip()
        if not any(cmd.startswith(p) for p in SAFE_PREFIXES):
            return f"BLOCKED: '{cmd}' not in safe command list. Only read-only commands allowed."
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=AUTORESEARCH_DIR
            )
            out = result.stdout + result.stderr
            return out[:8000] if len(out) > 8000 else out  # cap output
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out (30s)"
        except Exception as e:
            return f"ERROR: {e}"

    elif name == "list_files":
        path = Path(args["path"])
        if not path.is_absolute():
            path = AUTORESEARCH_DIR / path
        try:
            entries = sorted(path.iterdir())
            return "\n".join(str(e.name) + ("/" if e.is_dir() else "") for e in entries)
        except Exception as e:
            return f"ERROR: {e}"

    return f"ERROR: unknown tool {name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a research assistant subagent for the Alphabrain LFP brain region decoding project.
You help with token-intensive tasks: reading logs, analyzing results, proposing code changes, editing files.

You have tools to read/write files and run safe shell commands.
Be concise and direct. Output only what's needed — no padding.
When editing code files, output the complete new file content via write_file.
When analyzing logs or results, extract the key numbers and state your conclusion clearly."""


def run_agent(task: str, model: str, context_files: list[str], max_turns: int = 10) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Preload any context files into the initial message
    user_content = task
    for cf in context_files:
        path = Path(cf)
        if not path.is_absolute():
            path = AUTORESEARCH_DIR / cf
        if path.exists():
            content = path.read_text()
            user_content += f"\n\n--- {cf} ---\n{content}"

    messages.append({"role": "user", "content": user_content})

    for turn in range(max_turns):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": TOOLS,
        }

        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        msg = data["message"]
        messages.append(msg)

        # Check for tool calls
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            # No tool calls — done
            return msg.get("content", "").strip()

        # Execute tool calls and feed results back
        tool_results = []
        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            result = execute_tool(name, args)
            print(f"[tool] {name}({list(args.keys())}) → {len(str(result))} chars", file=sys.stderr)
            tool_results.append({
                "role": "tool",
                "content": result,
            })

        messages.extend(tool_results)

    return "[max_turns reached]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local LLM subagent via Ollama")
    parser.add_argument("--task", required=True, help="Task description for the agent")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["qwen3:8b", "llama3.1:latest"],
                        help="Ollama model to use")
    parser.add_argument("--context", nargs="*", default=[],
                        help="Files to preload as context (e.g. train.py run.log)")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max tool-use turns before stopping")
    args = parser.parse_args()

    result = run_agent(
        task=args.task,
        model=args.model,
        context_files=args.context,
        max_turns=args.max_turns,
    )
    print(result)


if __name__ == "__main__":
    main()
