"""
抽出器（Extractor）共通の型とユーティリティ。

方針:
  巨大ファイルをそのままLLMに渡さず、各Extractorが
  「LLMが説明を書くのに十分な、要約された構造化情報」を作る。
  ExtractionResult.summary が実際にプロンプトへ差し込まれる文字列。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    category_label: str            # 人間可読なカテゴリ名（プロンプトに出す）
    summary: str                   # LLMに渡す要約テキスト（構造化情報を文字列化したもの）
    raw_excerpt: str = ""          # 必要なら生テキストの抜粋（あれば追加コンテキストとして渡す）
    metadata: dict[str, Any] = field(default_factory=dict)  # CLI表示用の追加情報
    warnings: list[str] = field(default_factory=list)       # パース中に気づいた注意点


def read_text_safely(path: str, max_bytes: int = 5_000_000) -> str:
    """テキストファイルを安全に読む。大きすぎる場合は先頭のみ。"""
    with open(path, "rb") as f:
        raw = f.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    text = raw[:max_bytes].decode("utf-8", errors="ignore")
    if truncated:
        text += "\n\n... [ファイルが大きいため以降は省略] ...\n"
    return text


def head_tail(text: str, n_lines: int = 60) -> str:
    """長いテキストの先頭と末尾だけを抜粋する。"""
    lines = text.splitlines()
    if len(lines) <= 2 * n_lines:
        return text
    head = "\n".join(lines[:n_lines])
    tail = "\n".join(lines[-n_lines:])
    omitted = len(lines) - 2 * n_lines
    return f"{head}\n\n... [中間 {omitted} 行を省略] ...\n\n{tail}"


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
