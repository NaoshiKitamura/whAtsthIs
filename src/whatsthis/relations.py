"""
Purely rule-based (no LLM call) relationship detection between files in a
scanned directory. Kept deterministic and fast; the LLM only turns these
hints into a narrative later.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

from .detect import Category
from .dirscan import ScannedFile

_VASP_INPUT_NAMES = {"incar", "poscar", "kpoints", "potcar"}
_VASP_OUTPUT_NAMES = {"outcar", "contcar", "vasprun.xml", "xdatcar"}
_LAMMPS_CATEGORIES = (Category.LAMMPS_LOG, Category.LAMMPS_INPUT, Category.LAMMPS_DATA)

_FILENAME_TOKEN_RE = re.compile(r"[\"']([\w./\\-]+?\.\w+)[\"']")


def group_by_directory(files: list[ScannedFile]) -> dict[str, list[ScannedFile]]:
    groups: dict[str, list[ScannedFile]] = defaultdict(list)
    for f in files:
        groups[os.path.dirname(f.relpath) or "."].append(f)
    return groups


def detect_calculation_groups(files: list[ScannedFile]) -> list[str]:
    """Notes about VASP / LAMMPS calculation setups detected per subdirectory."""
    notes: list[str] = []
    for dirname, members in group_by_directory(files).items():
        names_lower = {os.path.basename(m.relpath).lower() for m in members}

        vasp_in = sorted(names_lower & _VASP_INPUT_NAMES)
        vasp_out = sorted(names_lower & _VASP_OUTPUT_NAMES)
        if vasp_in or vasp_out:
            notes.append(
                f"'{dirname}' looks like a VASP calculation directory "
                f"(input files present: {vasp_in or 'none'}; output files present: {vasp_out or 'none'})."
            )

        lammps_members = [m for m in members if m.detection.category in _LAMMPS_CATEGORIES]
        if lammps_members:
            kinds = sorted({m.detection.category.name for m in lammps_members})
            notes.append(f"'{dirname}' looks like a LAMMPS simulation directory (file kinds present: {kinds}).")

        if "poscar" in names_lower and "contcar" in names_lower:
            notes.append(
                f"'{dirname}' contains both POSCAR and CONTCAR, suggesting a completed relaxation/MD run "
                "(CONTCAR is the resulting final structure)."
            )

        cif_or_xyz = [m for m in members if m.detection.category.name in
                      ("STRUCTURE_CIF", "STRUCTURE_XYZ", "STRUCTURE_EXTXYZ")]
        ml_scripts = [m for m in members if m.detection.category == Category.PYTHON and
                      re.search(r"train|model|gnn|potential", m.relpath, re.IGNORECASE)]
        if cif_or_xyz and ml_scripts:
            notes.append(
                f"'{dirname}' has both structure files ({[m.relpath for m in cif_or_xyz]}) and a "
                f"training-related Python script ({[m.relpath for m in ml_scripts]}), suggesting a "
                "machine-learning potential training workflow."
            )

    return notes


def detect_code_references(files: list[ScannedFile], max_script_bytes: int = 300_000) -> list[str]:
    """Scan Python/Bash scripts for string literals matching other filenames in the
    tree, as a lightweight proxy for 'this script reads/writes that file'."""
    all_names: dict[str, str] = {}
    for f in files:
        all_names.setdefault(os.path.basename(f.relpath), f.relpath)

    notes: list[str] = []
    for f in files:
        if f.detection.category.name not in ("PYTHON", "BASH"):
            continue
        if f.size > max_script_bytes:
            continue
        try:
            with open(f.abspath, "r", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue

        referenced = set()
        for m in _FILENAME_TOKEN_RE.finditer(text):
            candidate = os.path.basename(m.group(1))
            target = all_names.get(candidate)
            if target and target != f.relpath:
                referenced.add(target)

        if referenced:
            notes.append(f"'{f.relpath}' appears to reference: {sorted(referenced)}")

    return notes


def build_all_notes(files: list[ScannedFile]) -> list[str]:
    return detect_calculation_groups(files) + detect_code_references(files)
