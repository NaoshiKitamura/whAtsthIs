"""
Extractor for programming-language files (Python / C / C++ / Fortran / Bash).

Small files are passed through as-is. Large files are switched into
"structure mode", which pulls out function/class definitions, imports, and
docstrings instead of the full text, to keep the LLM input compact.
"""

from __future__ import annotations

import ast
import re

from .base import ExtractionResult, read_text_safely, head_tail

# Files larger than this many lines switch to structure-extraction mode.
_STRUCTURE_MODE_THRESHOLD_LINES = 150


def _extract_python_structure(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return f"[Failed to parse as Python: {e}]\n\n" + head_tail(text)

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
                            f"    methods: {', '.join(methods) if methods else '(none)'}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            args = [a.arg for a in node.args.args]
            doc_part = f" - {doc.splitlines()[0]}" if doc else ""
            functions.append(f"def {node.name}({', '.join(args)}){doc_part}")

    n_lines = len(text.splitlines())
    parts = [f"Total lines: {n_lines}"]
    if module_doc:
        parts.append(f"Module docstring: {module_doc.strip().splitlines()[0]}")
    parts.append("Imports:\n  " + ("\n  ".join(imports) if imports else "(none)"))
    parts.append("Top-level classes:\n  " + ("\n  ".join(classes) if classes else "(none)"))
    parts.append("Top-level functions:\n  " + ("\n  ".join(functions) if functions else "(none)"))

    if 'if __name__ == "__main__"' in text or "if __name__ == '__main__'" in text:
        parts.append("Has a __main__ block (runnable directly as a script).")

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
    parts = [f"Total lines: {n_lines}"]
    parts.append("#include:\n  " + ("\n  ".join(includes) if includes else "(none)"))
    parts.append("Detected functions (heuristic regex parse, may miss some):\n  " +
                 ("\n  ".join(sorted(set(funcs))) if funcs else "(none detected)"))
    return "\n\n".join(parts)


def _extract_fortran_structure(text: str) -> str:
    modules = _FORTRAN_MODULE_RE.findall(text)
    procs = _FORTRAN_PROC_RE.findall(text)
    uses = _FORTRAN_USE_RE.findall(text)
    n_lines = len(text.splitlines())
    parts = [f"Total lines: {n_lines}"]
    parts.append("Modules:\n  " + ("\n  ".join(modules) if modules else "(none)"))
    parts.append("use statements (dependencies):\n  " + ("\n  ".join(u.strip() for u in uses) if uses else "(none)"))
    parts.append("Subroutines/functions:\n  " + ("\n  ".join(procs) if procs else "(none detected)"))
    return "\n\n".join(parts)


def _extract_bash_structure(text: str) -> str:
    funcs = _BASH_FUNC_RE.findall(text)
    shebang = ""
    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        shebang = lines[0]
    n_lines = len(lines)
    parts = [f"Total lines: {n_lines}", f"shebang: {shebang or '(none)'}"]
    parts.append("Defined functions:\n  " + ("\n  ".join(funcs) if funcs else "(none)"))
    return "\n\n".join(parts)


_LANG_LABELS = {
    "python": "Python script",
    "c": "C source code",
    "cpp": "C++ source code",
    "fortran": "Fortran source code",
    "bash": "Bash/shell script",
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
        warnings.append(f"File has {n_lines} lines; extracted structure (functions/classes/imports) instead of "
                         "the full text. raw_excerpt contains only a partial excerpt.")
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
