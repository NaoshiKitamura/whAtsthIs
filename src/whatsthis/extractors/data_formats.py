"""
Extractor for config/document formats: JSON / YAML / TOML / Markdown.
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
        summary = f"Top-level type: {type(data).__name__}\n\nKeys / value types:\n{struct}"
    except json.JSONDecodeError as e:
        warnings.append(f"Could not parse as JSON: {e}")
        summary = head_tail(text)

    return ExtractionResult(
        category_label="JSON file",
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
            summary = f"Top-level type: dict\n\nKeys / value types:\n{struct}"
        else:
            summary = f"Top-level type: {type(data).__name__}\n{_describe_value(data)}"
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Could not parse as YAML: {e}")
        summary = head_tail(text)

    return ExtractionResult(
        category_label="YAML file",
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
        summary = f"Keys / value types:\n{struct}"
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Could not parse as TOML: {e}")
        summary = head_tail(text)

    return ExtractionResult(
        category_label="TOML file",
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
        f"Total lines: {n_lines}, approx. word count: {n_words}",
        "Heading structure:\n  " + ("\n  ".join(heading_lines) if heading_lines else "(no headings)"),
        f"Link count: {len(links)}",
    ]

    return ExtractionResult(
        category_label="Markdown document",
        summary="\n\n".join(parts),
        raw_excerpt=head_tail(text, n_lines=40),
        metadata={"lines": n_lines, "words": n_words},
    )
