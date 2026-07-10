"""
The `wt` command.

    wt <file>         explain a single file
    wt <directory>     explain a whole folder (workflow overview, file roles, relationships)
"""

from __future__ import annotations

import os
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import config
from . import detect
from . import dirscan
from . import extractors
from . import prompts
from . import relations
from . import llm

console = Console()
err_console = Console(stderr=True)


def _run_llm(system_prompt: str, user_prompt: str, model: str, host: str, max_tokens: int, think: bool) -> str | None:
    with console.status(f"[bold cyan]Asking Ollama ({model})..."):
        try:
            return llm.explain(system_prompt, user_prompt, model=model, max_tokens=max_tokens, host=host, think=think)
        except llm.LLMError as e:
            err_console.print(f"[bold red]LLM call failed:[/bold red] {e}")
            err_console.print("\n[dim]Hint: use --no-llm to inspect the extracted summary without calling the LLM.[/dim]")
            sys.exit(1)


def _analyze_file(filepath: str, model: str, host: str, no_llm: bool, verbose: bool, max_tokens: int, think: bool, lang: str) -> None:
    filename = os.path.basename(filepath)

    with console.status(f"[bold cyan]Detecting file type for {filename}..."):
        detection = detect.detect(filepath)

    with console.status(f"[bold cyan]Extracting info as {detection.category.name}..."):
        try:
            extraction = extractors.extract(filepath, detection)
        except Exception as e:  # noqa: BLE001
            err_console.print(f"[bold red]Extraction failed:[/bold red] {e}")
            sys.exit(1)

    if verbose:
        console.print(Panel.fit(
            f"[bold]Category:[/bold] {extraction.category_label}\n"
            f"[bold]Reason:[/bold] {detection.reason}",
            title="Detection result", border_style="dim"
        ))
        for w in extraction.warnings:
            console.print(f"[yellow]! {w}[/yellow]")

    if no_llm:
        console.print(Panel(extraction.summary or "(no summary)", title=f"Extracted summary: {filename}", border_style="cyan"))
        return

    system_prompt, user_prompt = prompts.build_prompt(filename, detection, extraction, lang)

    if verbose:
        console.print(Panel.fit(user_prompt, title="Prompt sent to the LLM (debug)", border_style="dim"))

    answer = _run_llm(system_prompt, user_prompt, model, host, max_tokens, think)
    console.print(Panel(Markdown(answer), title=f"[bold]wt: {filename}[/bold]", border_style="green"))


def _analyze_directory(dirpath: str, model: str, host: str, no_llm: bool, verbose: bool, max_tokens: int, think: bool, lang: str, max_files: int) -> None:
    root_label = os.path.basename(os.path.abspath(dirpath.rstrip(os.sep))) or dirpath

    with console.status(f"[bold cyan]Scanning directory {root_label}..."):
        files, truncated = dirscan.scan(dirpath, max_files=max_files)

    if not files:
        err_console.print("[yellow]No files found in this directory.[/yellow]")
        return

    with console.status("[bold cyan]Looking for relationships between files..."):
        notes = relations.build_all_notes(files)

    tree_lines = [
        f"{f.relpath} : {f.detection.category.name} : {extractors.base.human_size(f.size)}"
        for f in files
    ]

    if verbose:
        console.print(Panel.fit(
            f"Files scanned: {len(files)}" + (" (truncated)" if truncated else ""),
            title="Scan result", border_style="dim"
        ))
        for n in notes:
            console.print(f"[cyan]* {n}[/cyan]")

    if no_llm:
        body = "\n".join(tree_lines) + ("\n\n--- Relationship notes ---\n" + "\n".join(notes) if notes else "")
        console.print(Panel(body, title=f"Directory scan: {root_label}", border_style="cyan"))
        return

    system_prompt, user_prompt = prompts.build_directory_prompt(root_label, tree_lines, notes, lang, truncated)

    if verbose:
        console.print(Panel.fit(user_prompt, title="Prompt sent to the LLM (debug)", border_style="dim"))

    answer = _run_llm(system_prompt, user_prompt, model, host, max_tokens, think)
    console.print(Panel(Markdown(answer), title=f"[bold]wt: {root_label}/[/bold]", border_style="green"))


@click.command()
@click.argument("target", type=click.Path(exists=True))
@click.option("--model", default=config.DEFAULT_MODEL, show_default=True, help="Ollama model name (e.g. qwen3.5:9b, qwen2.5:14b)")
@click.option("--host", default=config.DEFAULT_HOST, show_default=True, help="Ollama server URL")
@click.option("--lang", "--language", "lang", default=config.DEFAULT_LANGUAGE, show_default=True, help="Language for the LLM's answer (e.g. en, ja, English, Japanese)")
@click.option("--no-llm", is_flag=True, help="Skip the LLM call and just show the extracted summary (no Ollama needed)")
@click.option("-v", "--verbose", is_flag=True, help="Show extra detail: detection reason, extraction warnings, the raw prompt")
@click.option("--max-tokens", default=config.DEFAULT_MAX_TOKENS, show_default=True, help="Max tokens for the LLM's response")
@click.option("--think", is_flag=True, default=config.DEFAULT_THINK, help="Enable 'thinking' mode for models that support it (default: off)")
@click.option("--max-files", default=config.DEFAULT_MAX_FILES, show_default=True, help="Directory mode only: max number of files to scan")
def main(target: str, model: str, host: str, lang: str, no_llm: bool, verbose: bool, max_tokens: int, think: bool, max_files: int) -> None:
    """Explain TARGET, which can be a single file or a directory."""
    if os.path.isdir(target):
        _analyze_directory(target, model, host, no_llm, verbose, max_tokens, think, lang, max_files)
    else:
        _analyze_file(target, model, host, no_llm, verbose, max_tokens, think, lang)


if __name__ == "__main__":
    main()
