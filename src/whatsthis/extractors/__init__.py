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
    Category.VASP_POSCAR: "VASP POSCAR/CONTCAR（結晶構造ファイル）",
    Category.STRUCTURE_CIF: "CIF（結晶構造ファイル）",
    Category.STRUCTURE_XYZ: "XYZ（分子/構造座標ファイル）",
    Category.STRUCTURE_EXTXYZ: "extended XYZ（構造+付加情報ファイル）",
    Category.VASP_XDATCAR: "VASP XDATCAR（MD軌跡ファイル）",
    Category.STRUCTURE_GENERIC_ASE: "ASE対応の構造ファイル",
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
        label = "VASP INCAR（計算設定ファイル）" if cat == Category.VASP_INCAR else "VASP KPOINTS（k点設定ファイル）"
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
        return ExtractionResult(category_label="Quantum ESPRESSO 入力ファイル", summary=head_tail(text, 80))

    if cat == Category.UNKNOWN_TEXT:
        text = read_text_safely(path)
        # 最後の手段としてASEに構造ファイルとして読めるか試す
        try:
            result = structure.extract(path, "不明フォーマット（ASEで構造として読み込み試行）")
            if not result.warnings:
                return result
        except Exception:  # noqa: BLE001
            pass
        return ExtractionResult(
            category_label="不明フォーマットのテキストファイル",
            summary=head_tail(text, 60),
            warnings=["既知のファイル種別と一致しませんでした。テキストとしてそのまま渡しています。"],
        )

    return ExtractionResult(
        category_label="不明（バイナリの可能性）",
        summary="",
        warnings=["このファイルはテキストとして読めず、対応する抽出器もありません。"],
    )
