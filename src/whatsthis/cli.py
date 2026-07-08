"""
`wt` コマンドの本体。

    wt <filename>

だけで、ファイルの種類判定 → 要約抽出 → LLMへの説明依頼 → 結果表示、を行う。
"""

from __future__ import annotations

import os
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import detect
from . import extractors
from . import prompts
from . import llm

console = Console()
err_console = Console(stderr=True)


@click.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default=llm.DEFAULT_MODEL, show_default=True, help="使用するOllamaのモデル名（例: qwen2.5, qwen2.5:14b）")
@click.option("--host", default=llm.DEFAULT_HOST, show_default=True, help="OllamaサーバのURL")
@click.option("--no-llm", is_flag=True, help="LLMを呼ばず、抽出した要約情報のみを表示する（動作確認・Ollama不要）")
@click.option("-v", "--verbose", is_flag=True, help="判定理由や抽出時の警告など、詳細情報も表示する")
@click.option("--max-tokens", default=1500, show_default=True, help="LLM応答の最大トークン数")
def main(filepath: str, model: str, host: str, no_llm: bool, verbose: bool, max_tokens: int) -> None:
    """FILEPATH を解析し、LLMがその内容を説明します。"""

    filename = os.path.basename(filepath)

    with console.status(f"[bold cyan]{filename} の種類を判定中..."):
        detection = detect.detect(filepath)

    with console.status(f"[bold cyan]{detection.category.name} として情報を抽出中..."):
        try:
            extraction = extractors.extract(filepath, detection)
        except Exception as e:  # noqa: BLE001
            err_console.print(f"[bold red]抽出中にエラーが発生しました:[/bold red] {e}")
            sys.exit(1)

    if verbose:
        console.print(Panel.fit(
            f"[bold]カテゴリ:[/bold] {extraction.category_label}\n"
            f"[bold]判定理由:[/bold] {detection.reason}",
            title="判定結果", border_style="dim"
        ))
        if extraction.warnings:
            for w in extraction.warnings:
                console.print(f"[yellow]⚠ {w}[/yellow]")

    if no_llm:
        console.print(Panel(extraction.summary or "(要約情報なし)", title=f"抽出された要約: {filename}", border_style="cyan"))
        return

    system_prompt, user_prompt = prompts.build_prompt(filename, detection, extraction)

    if verbose:
        console.print(Panel.fit(user_prompt, title="LLMへ送るプロンプト（デバッグ表示）", border_style="dim"))

    with console.status(f"[bold cyan]Ollama ({model}) に説明を依頼中..."):
        try:
            answer = llm.explain(system_prompt, user_prompt, model=model, max_tokens=max_tokens, host=host)
        except llm.LLMError as e:
            err_console.print(f"[bold red]LLM呼び出しエラー:[/bold red] {e}")
            err_console.print("\n[dim]ヒント: --no-llm オプションを使うとLLMを呼ばずに抽出結果だけ確認できます。[/dim]")
            sys.exit(1)

    console.print(Panel(Markdown(answer), title=f"[bold]wt: {filename}[/bold]", border_style="green"))


if __name__ == "__main__":
    main()
