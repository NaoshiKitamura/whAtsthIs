"""
プログラミング言語ファイル（Python / C / C++ / Fortran / Bash）用の抽出器。

方針:
  小さいファイルはそのまま中身を渡す。
  大きいファイルは、関数/クラス定義・import・docstring等の
  「構造」を抜き出してLLMへの入力を圧縮する。
"""

from __future__ import annotations

import ast
import re

from .base import ExtractionResult, read_text_safely, head_tail

# このサイズを超えたら「構造抽出モード」に切り替える
_STRUCTURE_MODE_THRESHOLD_LINES = 150


def _extract_python_structure(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return f"[Pythonとしてパース失敗: {e}]\n\n" + head_tail(text)

    module_doc = ast.get_docstring(tree)
    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            methods = [
                n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            doc_part = f" - {doc.splitlines()[0]}" if doc else ""
            classes.append(f"class {node.name}({', '.join(m.id if isinstance(m, ast.Name) else ast.unparse(m) for m in node.bases)}){doc_part}\n"
                            f"    methods: {', '.join(methods) if methods else '(なし)'}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            args = [a.arg for a in node.args.args]
            doc_part = f" - {doc.splitlines()[0]}" if doc else ""
            functions.append(f"def {node.name}({', '.join(args)}){doc_part}")

    n_lines = len(text.splitlines())
    parts = [f"総行数: {n_lines}"]
    if module_doc:
        parts.append(f"モジュールdocstring: {module_doc.strip().splitlines()[0]}")
    parts.append("import文:\n  " + ("\n  ".join(imports) if imports else "(なし)"))
    parts.append("トップレベルのクラス:\n  " + ("\n  ".join(classes) if classes else "(なし)"))
    parts.append("トップレベルの関数:\n  " + ("\n  ".join(functions) if functions else "(なし)"))

    # __main__ ブロックの有無やCLI引数っぽい記述があるかも軽くチェック
    if 'if __name__ == "__main__"' in text or "if __name__ == '__main__'" in text:
        parts.append("スクリプトとして直接実行可能（__main__ ブロックあり）")

    return "\n\n".join(parts)


_C_FUNC_RE = re.compile(
    r"^[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", re.MULTILINE
)
_C_INCLUDE_RE = re.compile(r"^\s*#include\s*[<\"](.+?)[>\"]", re.MULTILINE)

_FORTRAN_PROC_RE = re.compile(
    r"^\s*(?:recursive\s+)?(?:subroutine|function)\s+(\w+)", re.IGNORECASE | re.MULTILINE
)
_FORTRAN_MODULE_RE = re.compile(r"^\s*module\s+(\w+)", re.IGNORECASE | re.MULTILINE)
_FORTRAN_USE_RE = re.compile(r"^\s*use\s+([\w,: ]+)", re.IGNORECASE | re.MULTILINE)

_BASH_FUNC_RE = re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\)\s*\{", re.MULTILINE)


def _extract_c_like_structure(text: str) -> str:
    includes = _C_INCLUDE_RE.findall(text)
    funcs = _C_FUNC_RE.findall(text)
    n_lines = len(text.splitlines())
    parts = [f"総行数: {n_lines}"]
    parts.append("#include:\n  " + ("\n  ".join(includes) if includes else "(なし)"))
    parts.append("検出された関数（簡易パースのため漏れの可能性あり）:\n  " +
                 ("\n  ".join(sorted(set(funcs))) if funcs else "(検出できず)"))
    return "\n\n".join(parts)


def _extract_fortran_structure(text: str) -> str:
    modules = _FORTRAN_MODULE_RE.findall(text)
    procs = _FORTRAN_PROC_RE.findall(text)
    uses = _FORTRAN_USE_RE.findall(text)
    n_lines = len(text.splitlines())
    parts = [f"総行数: {n_lines}"]
    parts.append("module:\n  " + ("\n  ".join(modules) if modules else "(なし)"))
    parts.append("use文（依存モジュール）:\n  " + ("\n  ".join(u.strip() for u in uses) if uses else "(なし)"))
    parts.append("subroutine/function:\n  " + ("\n  ".join(procs) if procs else "(検出できず)"))
    return "\n\n".join(parts)


def _extract_bash_structure(text: str) -> str:
    funcs = _BASH_FUNC_RE.findall(text)
    shebang = ""
    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        shebang = lines[0]
    n_lines = len(lines)
    parts = [f"総行数: {n_lines}", f"shebang: {shebang or '(なし)'}"]
    parts.append("定義されている関数:\n  " + ("\n  ".join(funcs) if funcs else "(なし)"))
    return "\n\n".join(parts)


_LANG_LABELS = {
    "python": "Pythonスクリプト",
    "c": "Cソースコード",
    "cpp": "C++ソースコード",
    "fortran": "Fortranソースコード",
    "bash": "Bash/シェルスクリプト",
}


def extract(path: str, lang: str) -> ExtractionResult:
    text = read_text_safely(path)
    n_lines = len(text.splitlines())
    warnings: list[str] = []

    use_structure_mode = n_lines > _STRUCTURE_MODE_THRESHOLD_LINES

    if use_structure_mode:
        if lang == "python":
            summary = _extract_python_structure(text)
        elif lang in ("c", "cpp"):
            summary = _extract_c_like_structure(text)
        elif lang == "fortran":
            summary = _extract_fortran_structure(text)
        elif lang == "bash":
            summary = _extract_bash_structure(text)
        else:
            summary = head_tail(text)
        raw_excerpt = head_tail(text, n_lines=40)
        warnings.append(f"{n_lines}行と大きいため、構造（関数・クラス・importなど）を抽出して要約しています。"
                         "全文はraw_excerptに一部のみ含まれます。")
    else:
        summary = text
        raw_excerpt = ""

    return ExtractionResult(
        category_label=_LANG_LABELS.get(lang, lang),
        summary=summary,
        raw_excerpt=raw_excerpt,
        metadata={"lines": n_lines, "structure_mode": use_structure_mode},
        warnings=warnings,
    )
