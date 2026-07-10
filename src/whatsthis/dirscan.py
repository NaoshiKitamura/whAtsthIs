"""
Lightweight recursive directory scanner for whatsthis' directory-analysis mode.

This intentionally does *not* run the full per-file extractors on every file
(that would be slow and blow up the prompt size for large trees) -- it just
walks the tree and classifies each file's category, cheaply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import detect

DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode",
    "dist", "build", ".mypy_cache", ".pytest_cache", "site-packages", ".tox",
}


@dataclass
class ScannedFile:
    relpath: str
    abspath: str
    size: int
    detection: detect.Detection


def scan(root: str, max_files: int = 500) -> tuple[list[ScannedFile], bool]:
    """Walk `root` and classify each file. Returns (files, truncated)."""
    results: list[ScannedFile] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".")
        )
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            if len(results) >= max_files:
                truncated = True
                return results, truncated
            abspath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(abspath)
            except OSError:
                continue
            relpath = os.path.relpath(abspath, root)
            det = detect.detect(abspath)
            results.append(ScannedFile(relpath=relpath, abspath=abspath, size=size, detection=det))

    return results, truncated
