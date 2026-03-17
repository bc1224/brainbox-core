"""
Local summary store — saves per-document summaries as JSON.
Used for instant recall of what's been ingested and for multi-source synthesis.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import anthropic
import config

SUMMARY_DIR = os.path.join(os.path.dirname(__file__), ".summaries")


def _ensure_dir():
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def _namespace_path(namespace: str) -> str:
    return os.path.join(SUMMARY_DIR, f"{namespace}.json")


def load_summaries(namespace: str = "default") -> dict:
    """Load all summaries for a namespace. Returns {filename: {summary, ingested_at, ...}}"""
    path = _namespace_path(namespace)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_summary(filename: str, summary: str, file_type: str, chunk_count: int, namespace: str = "default"):
    """Save a document summary."""
    _ensure_dir()
    summaries = load_summaries(namespace)
    summaries[filename] = {
        "summary": summary,
        "file_type": file_type,
        "chunk_count": chunk_count,
        "ingested_at": datetime.now().isoformat(),
    }
    with open(_namespace_path(namespace), "w") as f:
        json.dump(summaries, f, indent=2)


def delete_summaries(namespace: str = "default"):
    """Delete all summaries for a namespace."""
    path = _namespace_path(namespace)
    if os.path.exists(path):
        os.remove(path)


def generate_summary(text: str, filename: str) -> str:
    """Generate a summary of a document using Claude."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Use first ~8000 words to stay within reasonable token limits
    words = text.split()
    truncated = " ".join(words[:8000])
    was_truncated = len(words) > 8000

    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=500,
        system="You summarize documents concisely. Give a 2-4 sentence overview of what this document contains, its key topics, and any notable details. Be specific — mention names, numbers, dates when present. Do not use filler phrases like 'This document contains...' — just state what it covers.",
        messages=[
            {
                "role": "user",
                "content": f"Summarize this document ({filename}){'[truncated]' if was_truncated else ''}:\n\n{truncated}"
            }
        ]
    )
    return message.content[0].text


def synthesize_sources(question: str, namespace: str = "default") -> str | None:
    """
    Use all document summaries to answer high-level synthesis questions.
    Returns None if no summaries exist.
    """
    summaries = load_summaries(namespace)
    if not summaries:
        return None

    summary_text = "\n\n".join(
        f"**{fname}** ({info['file_type']}): {info['summary']}"
        for fname, info in summaries.items()
    )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2000,
        system="""You are a research assistant that synthesizes information across multiple sources. 
You have summaries of all documents in the user's knowledge base. 
Use these to answer high-level questions, compare sources, identify themes, and find connections.
Be specific and cite which documents you're drawing from.
Format in markdown.""",
        messages=[
            {
                "role": "user",
                "content": f"""Document summaries in this knowledge base:

{summary_text}

---

Question: {question}"""
            }
        ]
    )
    return message.content[0].text
