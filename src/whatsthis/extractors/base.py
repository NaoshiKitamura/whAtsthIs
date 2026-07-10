"""
Common types and utilities shared by every extractor.

Rationale: rather than pass a huge file straight to the LLM, each extractor
builds a compact, structured summary that gives the LLM enough context to
write an explanation. ExtractionResult.summary is what actually gets
interpolated into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    category_label: str            # Human-readable category name (shown in the prompt)
    summary: str                   # Text summary handed to the LLM
    raw_excerpt: str = ""          # Optional raw-text excerpt, added as extra context if useful
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra info for CLI display
    warnings: list[str] = field(default_factory=list)       # Notes/caveats surfaced during parsing


def read_text_safely(path: str, max_bytes: int = 5_000_000) -> str:
    """Read a text file safely, truncating to the first max_bytes if it's too large."""
    with open(path, "rb") as f:
        raw = f.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    text = raw[:max_bytes].decode("utf-8", errors="ignore")
    if truncated:
        text += "\n\n... [file is large; remainder omitted] ...\n"
    return text


def head_tail(text: str, n_lines: int = 60) -> str:
    """Return only the head and tail of a long text block."""
    lines = text.splitlines()
    if len(lines) <= 2 * n_lines:
        return text
    head = "\n".join(lines[:n_lines])
    tail = "\n".join(lines[-n_lines:])
    omitted = len(lines) - 2 * n_lines
    return f"{head}\n\n... [{omitted} lines omitted] ...\n\n{tail}"


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
