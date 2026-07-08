"""
設定ファイル・ドキュメント系フォーマット（JSON / YAML / TOML / Markdown）の抽出器。
"""

from __future__ import annotations

import json
import re

from .base import ExtractionResult, read_text_safely, head_tail


def _describe_value(v, depth: int = 0) -> str:
    if isinstance(v, dict):
        return f"object({len(v)} keys)"
    if isinstance(v, list):
        return f"array(len={len(v)})"
    if isinstance(v, str):
        s = v if len(v) <= 40 else v[:37] + "..."
        return f'"{s}"'
    return repr(v)


def _summarize_mapping(obj, max_depth: int = 2, depth: int = 0, prefix: str = "") -> list[str]:
    lines = []
    if not isinstance(obj, dict) or depth > max_depth:
        return lines
    for k, v in obj.items():
        lines.append(f"{prefix}{k}: {_describe_value(v)}")
        if isinstance(v, dict) and depth < max_depth:
            lines.extend(_summarize_mapping(v, max_depth, depth + 1, prefix + "  "))
    return lines


def extract_json(path: str) -> ExtractionResult:
    text = read_text_safely(path)
    warnings = []
    try:
        data = json.loads(text)
        struct = "\n".join(_summarize_mapping(data)) if isinstance(data, dict) else _describe_value(data)
        summary = f"トップレベルの型: {type(data).__name__}\n\n主なキーと値の型/概要:\n{struct}"
    except json.JSONDecodeError as e:
        warnings.append(f"JSONとしてパースできませんでした: {e}")
        summary = head_tail(text)

    return ExtractionResult(
        category_label="JSONファイル",
        summary=summary,
        raw_excerpt=head_tail(text, n_lines=30),
        warnings=warnings,
    )


def extract_yaml(path: str) -> ExtractionResult:
    text = read_text_safely(path)
    warnings = []
    try:
        import yaml  # PyYAML
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            struct = "\n".join(_summarize_mapping(data))
            summary = f"トップレベルの型: dict\n\n主なキーと値の型/概要:\n{struct}"
        else:
            summary = f"トップレベルの型: {type(data).__name__}\n{_describe_value(data)}"
    except Exception as e:  # noqa: BLE001
        warnings.append(f"YAMLとしてパースできませんでした: {e}")
        summary = head_tail(text)

    return ExtractionResult(
        category_label="YAMLファイル",
        summary=summary,
        raw_excerpt=head_tail(text, n_lines=30),
        warnings=warnings,
    )


def extract_toml(path: str) -> ExtractionResult:
    text = read_text_safely(path)
    warnings = []
    try:
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(text)
        struct = "\n".join(_summarize_mapping(data))
        summary = f"主なキーと値の型/概要:\n{struct}"
    except Exception as e:  # noqa: BLE001
        warnings.append(f"TOMLとしてパースできませんでした: {e}")
        summary = head_tail(text)

    return ExtractionResult(
        category_label="TOMLファイル",
        summary=summary,
        raw_excerpt=head_tail(text, n_lines=30),
        warnings=warnings,
    )


def extract_markdown(path: str) -> ExtractionResult:
    text = read_text_safely(path)
    headings = re.findall(r"^(#{1,6})\s+(.*)$", text, re.MULTILINE)
    links = re.findall(r"\[.*?\]\((.*?)\)", text)
    n_lines = len(text.splitlines())
    n_words = len(text.split())

    heading_lines = [f"{h[0]} {h[1]}" for h in headings[:40]]
    parts = [
        f"総行数: {n_lines}, 概算語数: {n_words}",
        "見出し構造:\n  " + ("\n  ".join(heading_lines) if heading_lines else "(見出しなし)"),
        f"リンク数: {len(links)}",
    ]

    return ExtractionResult(
        category_label="Markdownドキュメント",
        summary="\n\n".join(parts),
        raw_excerpt=head_tail(text, n_lines=40),
        metadata={"lines": n_lines, "words": n_words},
    )
