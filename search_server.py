#!/usr/bin/env python3
"""
FastMCP search server — arxiv + HuggingFace Papers search tools.

Provides two tools any MCP-compatible LLM client can call:
  search_arxiv(query, max_results=5)
  search_huggingface_papers(query, max_results=5)

Usage
-----
Standalone (stdio MCP server):
    uv run python search_server.py

Connect from Claude Code:
    claude --mcp-config mcp_config.json "search arxiv for contrastive SSL"

Connect from Codex:
    codex exec -c 'mcp_servers=[{name="search",command="uv",args=["run","python","search_server.py"]}]' "..."

autoloop.py calls these functions directly (no MCP transport needed) via
the _search_arxiv / _search_huggingface helpers imported from this module.
"""

import xml.etree.ElementTree as ET

import httpx
from fastmcp import FastMCP

mcp = FastMCP("autoresearch-search")

# ── Core HTTP search functions (called by autoloop.py directly) ────────────

def _search_arxiv(query: str, max_results: int = 5) -> str:
    """Hit the arxiv API and return formatted paper titles + abstracts."""
    try:
        r = httpx.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"},
            timeout=30,
        )
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        results = []
        for entry in root.findall("a:entry", ns):
            title    = (entry.find("a:title", ns).text or "").strip()
            abstract = (entry.find("a:summary", ns).text or "").strip()[:400]
            link     = (entry.find("a:id", ns).text or "").strip()
            results.append(f"**{title}**\n{link}\n{abstract}")
        return "\n\n---\n\n".join(results) if results else "No arxiv results found."
    except Exception as e:
        return f"arxiv search error: {e}"


def _search_huggingface_papers(query: str, max_results: int = 5) -> str:
    """Hit the HuggingFace Papers API and return formatted results."""
    try:
        r = httpx.get(
            "https://huggingface.co/api/papers",
            params={"q": query},
            timeout=30,
        )
        r.raise_for_status()
        papers = r.json()[:max_results]
        results = []
        for p in papers:
            title    = p.get("title", "?")
            abstract = p.get("summary", "")[:400]
            pid      = p.get("id", "")
            results.append(f"**{title}**\nhttps://huggingface.co/papers/{pid}\n{abstract}")
        return "\n\n---\n\n".join(results) if results else "No HuggingFace Papers results found."
    except Exception as e:
        return f"HuggingFace Papers search error: {e}"


# ── MCP tool wrappers (used when an LLM connects to this server) ───────────

@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arxiv for ML/neuroscience papers. Returns title, URL, and abstract snippet."""
    return _search_arxiv(query, max_results)


@mcp.tool()
def search_huggingface_papers(query: str, max_results: int = 5) -> str:
    """Search HuggingFace Papers for ML papers. Returns title, URL, and abstract snippet."""
    return _search_huggingface_papers(query, max_results)


if __name__ == "__main__":
    mcp.run()
