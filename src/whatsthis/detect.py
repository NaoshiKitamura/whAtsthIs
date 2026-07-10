"""
File-category detection.

Given a file path, guess which category it belongs to based on filename,
extension, and (if needed) a peek at the first few bytes of content. The
result is handed to the matching extractor in extractors/.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum, auto


class Category(Enum):
    # --- Programming languages / generic text ---
    PYTHON = auto()
    C = auto()
    CPP = auto()
    FORTRAN = auto()
    BASH = auto()
    JSON = auto()
    YAML = auto()
    TOML = auto()
    MARKDOWN = auto()

    # --- VASP ---
    VASP_OUTCAR = auto()
    VASP_VASPRUN = auto()
    VASP_POSCAR = auto()       # POSCAR / CONTCAR
    VASP_INCAR = auto()
    VASP_KPOINTS = auto()
    VASP_XDATCAR = auto()

    # --- LAMMPS ---
    LAMMPS_LOG = auto()
    LAMMPS_DATA = auto()
    LAMMPS_INPUT = auto()

    # --- Quantum ESPRESSO ---
    QE_INPUT = auto()

    # --- Structure files ASE can read ---
    STRUCTURE_CIF = auto()
    STRUCTURE_XYZ = auto()
    STRUCTURE_EXTXYZ = auto()
    STRUCTURE_GENERIC_ASE = auto()   # anything else we try feeding to ase.io.read

    # --- Unknown ---
    UNKNOWN_TEXT = auto()
    UNKNOWN_BINARY = auto()


@dataclass
class Detection:
    category: Category
    reason: str  # why we classified it this way (kept for transparency/debugging)


# Simple extension -> category maps. Priority between these is handled in detect().
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

# Exact (case-insensitive) filename matches for VASP's fixed-name files.
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
    """Read a small chunk from the start of a file as text (empty string if binary)."""
    try:
        with open(path, "rb") as f:
            raw = f.read(n_bytes)
        return raw.decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _looks_like_lammps_data(head: str) -> bool:
    # LAMMPS data files start with a comment line, followed by lines like
    # "N atoms" / "N atom types".
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

    # 1) Exact filename match (VASP fixed-name files)
    if lower_name in _VASP_EXACT_NAMES:
        return Detection(_VASP_EXACT_NAMES[lower_name], f"filename '{filename}' matches a standard VASP filename")

    # 2) vasprun.xml
    if lower_name == "vasprun.xml":
        return Detection(Category.VASP_VASPRUN, "filename is vasprun.xml")

    # 3) LAMMPS log
    for pat in _LAMMPS_LOG_PATTERNS:
        if pat.match(lower_name):
            return Detection(Category.LAMMPS_LOG, f"filename matches LAMMPS log pattern '{pat.pattern}'")

    # 4) LAMMPS input script (in.* or *.lmp)
    if lower_name.startswith("in.") or ext in (".lmp",):
        return Detection(Category.LAMMPS_INPUT, "filename/extension matches common LAMMPS input-script conventions")

    # 5) Structure file by extension
    if ext in _STRUCTURE_EXT_MAP:
        cat = _STRUCTURE_EXT_MAP[ext]
        return Detection(cat, f"extension '{ext}' matched")

    # 6) Programming language / generic text by extension
    if ext in _CODE_EXT_MAP:
        return Detection(_CODE_EXT_MAP[ext], f"extension '{ext}' matched")

    # 7) No/unknown extension -> peek at content
    head = _sniff_head(path)

    if not head:
        return Detection(Category.UNKNOWN_BINARY, "could not be read as text (likely binary)")

    if _looks_like_vasprun(head):
        return Detection(Category.VASP_VASPRUN, "content resembles VASP's XML output")

    if lower_name.startswith("data.") or ext == ".data":
        if _looks_like_lammps_data(head):
            return Detection(Category.LAMMPS_DATA, "filename/content resembles a LAMMPS data file")

    if _looks_like_lammps_data(head):
        return Detection(Category.LAMMPS_DATA, "content structure resembles a LAMMPS data file")

    if _looks_like_qe_input(head):
        return Detection(Category.QE_INPUT, "content contains &control / &system, resembling a Quantum ESPRESSO input")

    if ext in (".data", ".lammps"):
        return Detection(Category.LAMMPS_DATA, f"extension '{ext}' is LAMMPS-related")

    # 8) Fall back to a generic text category
    return Detection(Category.UNKNOWN_TEXT, "did not match any known pattern; treated as generic text")
