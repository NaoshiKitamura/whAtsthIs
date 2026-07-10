"""
Thin wrapper around a local Ollama server's chat API.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config

DEFAULT_MODEL = config.DEFAULT_MODEL
DEFAULT_HOST = config.DEFAULT_HOST
DEFAULT_MAX_TOKENS = config.DEFAULT_MAX_TOKENS
DEFAULT_THINK = config.DEFAULT_THINK


class LLMError(RuntimeError):
    pass


def explain(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    host: str | None = None,
    timeout: int = 600,
    think: bool = DEFAULT_THINK,
) -> str:
    host = (host or DEFAULT_HOST).rstrip("/")
    url = f"{host}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": think,  # "thinking" models (e.g. the qwen3 family) can burn the whole
                          # token budget on reasoning and leave content empty if this is True
        "options": {"num_predict": max_tokens},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        hint = ""
        if e.code == 404 or "not found" in body.lower():
            hint = f"\nThe model '{model}' may not be pulled yet. Try `ollama pull {model}`."
        raise LLMError(f"Ollama returned an error (HTTP {e.code}): {body}{hint}") from e
    except urllib.error.URLError as e:
        raise LLMError(
            f"Failed to connect to the Ollama server ({url}).\n"
            "  - Check that `ollama serve` is running\n"
            "  - If OLLAMA_HOST is non-default, pass it with --host\n"
            f"Details: {e.reason}"
        ) from e

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Could not parse Ollama's response as JSON: {e}\nResponse: {raw[:500]}") from e

    message = body.get("message", {})
    content = (message.get("content") or "").strip()
    thinking = (message.get("thinking") or "").strip()
    done_reason = body.get("done_reason")

    if not content:
        if "error" in body:
            raise LLMError(f"Ollama returned an error: {body['error']}")
        if thinking and done_reason == "length":
            raise LLMError(
                f"'{model}' used up the entire max_tokens budget ({max_tokens}) on its internal "
                "reasoning ('thinking') and never produced actual content.\n"
                "  Try:\n"
                "  - increasing --max-tokens (e.g. --max-tokens 8000)\n"
                "  - thinking is disabled by default here; if it's still happening, this model may "
                "not support disabling it via the 'think' API field."
            )
        raise LLMError(f"Ollama's response had no content: {body}")

    return content.strip()
