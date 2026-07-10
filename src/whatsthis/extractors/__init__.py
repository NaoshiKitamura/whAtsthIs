from __future__ import annotations

from ..detect import Category, Detection
from .base import ExtractionResult, read_text_safely, head_tail
from . import code, data_formats, structure, vasp, lammps

_CODE_LANG_MAP = {
    Category.PYTHON: "python",
    Category.C: "c",
    Category.CPP: "cpp",
    Category.FORTRAN: "fortran",
    Category.BASH: "bash",
}

_STRUCTURE_LABEL_MAP = {
    Category.VASP_POSCAR: "VASP POSCAR/CONTCAR (crystal structure file)",
    Category.STRUCTURE_CIF: "CIF (crystal structure file)",
    Category.STRUCTURE_XYZ: "XYZ (molecule/structure coordinate file)",
    Category.STRUCTURE_EXTXYZ: "extended XYZ (structure + attached properties file)",
    Category.VASP_XDATCAR: "VASP XDATCAR (MD trajectory file)",
    Category.STRUCTURE_GENERIC_ASE: "ASE-readable structure file",
}

_ASE_FORMAT_HINT = {
    Category.VASP_POSCAR: "vasp",
    Category.STRUCTURE_CIF: "cif",
    Category.STRUCTURE_XYZ: "xyz",
    Category.STRUCTURE_EXTXYZ: "extxyz",
    Category.VASP_XDATCAR: "vasp-xdatcar",
}


def extract(path: str, detection: Detection) -> ExtractionResult:
    cat = detection.category

    if cat in _CODE_LANG_MAP:
        return code.extract(path, _CODE_LANG_MAP[cat])

    if cat == Category.JSON:
        return data_formats.extract_json(path)
    if cat == Category.YAML:
        return data_formats.extract_yaml(path)
    if cat == Category.TOML:
        return data_formats.extract_toml(path)
    if cat == Category.MARKDOWN:
        return data_formats.extract_markdown(path)

    if cat == Category.VASP_OUTCAR:
        return vasp.extract_outcar(path)
    if cat == Category.VASP_VASPRUN:
        return vasp.extract_vasprun(path)

    if cat in (Category.VASP_INCAR, Category.VASP_KPOINTS):
        text = read_text_safely(path)
        label = "VASP INCAR (calculation settings file)" if cat == Category.VASP_INCAR else "VASP KPOINTS (k-point settings file)"
        return ExtractionResult(category_label=label, summary=text)

    if cat in _STRUCTURE_LABEL_MAP:
        return structure.extract(path, _STRUCTURE_LABEL_MAP[cat], _ASE_FORMAT_HINT.get(cat))

    if cat == Category.LAMMPS_LOG:
        return lammps.extract_log(path)
    if cat == Category.LAMMPS_INPUT:
        return lammps.extract_input_script(path)
    if cat == Category.LAMMPS_DATA:
        return lammps.extract_data(path)

    if cat == Category.QE_INPUT:
        text = read_text_safely(path)
        return ExtractionResult(category_label="Quantum ESPRESSO input file", summary=head_tail(text, 80))

    if cat == Category.UNKNOWN_TEXT:
        text = read_text_safely(path)
        # last resort: see if ASE can read it as a structure file
        try:
            result = structure.extract(path, "Unknown format (attempted ASE structure read)")
            if not result.warnings:
                return result
        except Exception:  # noqa: BLE001
            pass
        return ExtractionResult(
            category_label="Unknown-format text file",
            summary=head_tail(text, 60),
            warnings=["Did not match any known file category; passing raw text through as-is."],
        )

    return ExtractionResult(
        category_label="Unknown (possibly binary)",
        summary="",
        warnings=["This file could not be read as text and has no matching extractor."],
    )
