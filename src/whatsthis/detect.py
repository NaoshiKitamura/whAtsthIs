"""
ファイル種別の判定モジュール。

ファイル名・拡張子・（必要なら）中身の先頭数行を見て、
このファイルが「何のカテゴリ」に属するかを推定する。
判定結果は extractors/ 以下の対応する抽出器へ渡される。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum, auto


class Category(Enum):
    # --- プログラミング言語 / 汎用テキスト ---
    PYTHON = auto()
    C = auto()
    CPP = auto()
    FORTRAN = auto()
    BASH = auto()
    JSON = auto()
    YAML = auto()
    TOML = auto()
    MARKDOWN = auto()

    # --- VASP 関連 ---
    VASP_OUTCAR = auto()
    VASP_VASPRUN = auto()
    VASP_POSCAR = auto()       # POSCAR / CONTCAR
    VASP_INCAR = auto()
    VASP_KPOINTS = auto()
    VASP_XDATCAR = auto()

    # --- LAMMPS 関連 ---
    LAMMPS_LOG = auto()
    LAMMPS_DATA = auto()
    LAMMPS_INPUT = auto()

    # --- Quantum ESPRESSO ---
    QE_INPUT = auto()

    # --- ASE が扱える構造ファイル全般 ---
    STRUCTURE_CIF = auto()
    STRUCTURE_XYZ = auto()
    STRUCTURE_EXTXYZ = auto()
    STRUCTURE_GENERIC_ASE = auto()   # 上記以外でase.io.readに投げてみるもの

    # --- 不明 ---
    UNKNOWN_TEXT = auto()
    UNKNOWN_BINARY = auto()


@dataclass
class Detection:
    category: Category
    reason: str  # なぜそう判定したかの説明（デバッグ・透明性のため）


# 拡張子ベースの単純なマッピング（優先度は detect() の中で制御）
_CODE_EXT_MAP = {
    ".py": Category.PYTHON,
    ".c": Category.C,
    ".h": Category.C,
    ".cpp": Category.CPP,
    ".cxx": Category.CPP,
    ".cc": Category.CPP,
    ".hpp": Category.CPP,
    ".hxx": Category.CPP,
    ".f90": Category.FORTRAN,
    ".f95": Category.FORTRAN,
    ".f": Category.FORTRAN,
    ".for": Category.FORTRAN,
    ".sh": Category.BASH,
    ".bash": Category.BASH,
    ".json": Category.JSON,
    ".yaml": Category.YAML,
    ".yml": Category.YAML,
    ".toml": Category.TOML,
    ".md": Category.MARKDOWN,
    ".markdown": Category.MARKDOWN,
}

_STRUCTURE_EXT_MAP = {
    ".cif": Category.STRUCTURE_CIF,
    ".xyz": Category.STRUCTURE_XYZ,
    ".extxyz": Category.STRUCTURE_EXTXYZ,
    ".vasp": Category.VASP_POSCAR,
    ".pwi": Category.QE_INPUT,
    ".pwo": Category.QE_INPUT,
}

# 完全一致（大文字小文字区別しない）で判定するVASP系の特別なファイル名
_VASP_EXACT_NAMES = {
    "outcar": Category.VASP_OUTCAR,
    "poscar": Category.VASP_POSCAR,
    "contcar": Category.VASP_POSCAR,
    "incar": Category.VASP_INCAR,
    "kpoints": Category.VASP_KPOINTS,
    "xdatcar": Category.VASP_XDATCAR,
}

_LAMMPS_LOG_PATTERNS = [
    re.compile(r"^log\..*lammps.*$", re.IGNORECASE),
    re.compile(r"^log\.lammps$", re.IGNORECASE),
]


def _sniff_head(path: str, n_bytes: int = 4096) -> str:
    """ファイル先頭の一部をテキストとして読む（バイナリなら空文字を返す）。"""
    try:
        with open(path, "rb") as f:
            raw = f.read(n_bytes)
        return raw.decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _looks_like_lammps_data(head: str) -> bool:
    # LAMMPS data ファイルは1行目がコメントで、その後に
    # "N atoms" / "N atom types" 等の行が続く
    return bool(re.search(r"\batoms\b", head) and re.search(r"\batom types\b", head))


def _looks_like_qe_input(head: str) -> bool:
    return "&control" in head.lower() or "&system" in head.lower()


def _looks_like_vasprun(head: str) -> bool:
    return "<modeling>" in head or ("<?xml" in head and "vasp" in head.lower())


def detect(path: str) -> Detection:
    filename = os.path.basename(path)
    stem, ext = os.path.splitext(filename)
    ext = ext.lower()
    lower_name = filename.lower()

    # 1) ファイル名の完全一致（VASP系）
    if lower_name in _VASP_EXACT_NAMES:
        return Detection(_VASP_EXACT_NAMES[lower_name], f"ファイル名 '{filename}' がVASP標準名と一致")

    # 2) vasprun.xml
    if lower_name == "vasprun.xml":
        return Detection(Category.VASP_VASPRUN, "ファイル名が vasprun.xml")

    # 3) LAMMPS ログ
    for pat in _LAMMPS_LOG_PATTERNS:
        if pat.match(lower_name):
            return Detection(Category.LAMMPS_LOG, f"ファイル名パターン '{pat.pattern}' に一致")

    # 4) LAMMPS input script （in.* や *.lmp / *.in で中身にLAMMPSコマンドがある場合）
    if lower_name.startswith("in.") or ext in (".lmp",):
        return Detection(Category.LAMMPS_INPUT, f"ファイル名/拡張子がLAMMPS入力スクリプトの慣例に一致")

    # 5) 拡張子で構造ファイル判定
    if ext in _STRUCTURE_EXT_MAP:
        cat = _STRUCTURE_EXT_MAP[ext]
        return Detection(cat, f"拡張子 '{ext}' に一致")

    # 6) 拡張子でプログラミング言語/汎用テキスト判定
    if ext in _CODE_EXT_MAP:
        return Detection(_CODE_EXT_MAP[ext], f"拡張子 '{ext}' に一致")

    # 7) 拡張子なし or 不明拡張子 → 中身を軽く覗いて判定
    head = _sniff_head(path)

    if not head:
        return Detection(Category.UNKNOWN_BINARY, "テキストとして読めなかった（バイナリの可能性）")

    if _looks_like_vasprun(head):
        return Detection(Category.VASP_VASPRUN, "中身がVASPのXML出力に類似")

    if lower_name.startswith("data.") or ext == ".data":
        if _looks_like_lammps_data(head):
            return Detection(Category.LAMMPS_DATA, "ファイル名/中身がLAMMPS dataファイルに類似")

    if _looks_like_lammps_data(head):
        return Detection(Category.LAMMPS_DATA, "中身がLAMMPS dataファイルの構造に類似")

    if _looks_like_qe_input(head):
        return Detection(Category.QE_INPUT, "中身に &control / &system が含まれQuantum ESPRESSO入力に類似")

    if ext in (".data", ".lammps"):
        return Detection(Category.LAMMPS_DATA, f"拡張子 '{ext}' がLAMMPS関連")

    # 8) それでも分からなければ、ASEに読ませてみる価値があるかもしれないテキスト
    #    としてジェネリックな構造ファイル候補にする
    return Detection(Category.UNKNOWN_TEXT, "既知のパターンに一致しない汎用テキストファイル")
