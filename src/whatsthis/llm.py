"""
Ollamaのローカルサーバ (デフォルト http://localhost:11434) を呼び出すだけの薄いラッパー。

想定する使い方:
    ollama serve            # サーバ起動（バックグラウンドで動いていればOK）
    ollama pull qwen2.5     # 使うモデルを取得

標準ライブラリの urllib のみを使用し、追加の依存パッケージを増やさない設計。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = os.environ.get("WT_MODEL", "qwen2.5")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class LLMError(RuntimeError):
    pass


def explain(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1500,
    host: str | None = None,
    timeout: int = 600,
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
            hint = f"\nモデル '{model}' が未取得の可能性があります。`ollama pull {model}` を試してください。"
        raise LLMError(f"Ollamaがエラーを返しました (HTTP {e.code}): {body}{hint}") from e
    except urllib.error.URLError as e:
        raise LLMError(
            f"Ollamaサーバへの接続に失敗しました ({url})。\n"
            "  - `ollama serve` が起動しているか確認してください\n"
            f"  - OLLAMA_HOST を変更している場合は --host で指定してください\n"
            f"詳細: {e.reason}"
        ) from e

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Ollamaからの応答をJSONとして解析できませんでした: {e}\n応答: {raw[:500]}") from e

    content = body.get("message", {}).get("content", "")
    if not content:
        if "error" in body:
            raise LLMError(f"Ollamaがエラーを返しました: {body['error']}")
        raise LLMError(f"Ollamaからの応答にcontentが含まれていません: {body}")

    return content.strip()
