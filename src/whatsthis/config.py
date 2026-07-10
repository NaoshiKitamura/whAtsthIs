"""
Centralized, user-overridable defaults.

Every value here can be overridden by an environment variable, and (where it
makes sense) further overridden per-invocation by a CLI flag, which always
takes precedence. Keeping these in one place avoids the same env var being
parsed slightly differently in multiple modules.
"""

from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Ollama connection / model
DEFAULT_MODEL = os.environ.get("WT_MODEL", "qwen3.5:9b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MAX_TOKENS = int(os.environ.get("WT_MAX_TOKENS", "4000"))
DEFAULT_THINK = _bool_env("WT_THINK", False)

# Output language for the LLM's explanation. Any value the model understands
# as a language name/code works (e.g. "en", "English", "ja", "Japanese").
DEFAULT_LANGUAGE = os.environ.get("WT_LANG", "en")

# Directory-analysis mode limits
DEFAULT_MAX_FILES = int(os.environ.get("WT_MAX_FILES", "500"))
