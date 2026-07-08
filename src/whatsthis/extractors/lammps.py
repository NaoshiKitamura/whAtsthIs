"""
LAMMPS関連ファイル（log.lammps / 入力スクリプト / data ファイル）用の抽出器。
"""

from __future__ import annotations

import re
import os

from .base import ExtractionResult, human_size, read_text_safely, head_tail


def extract_log(path: str) -> ExtractionResult:
    """log.lammps: thermo出力ブロックを検出し、run条件・統計を要約する。"""
    size = os.path.getsize(path)
    warnings: list[str] = []

    units = None
    atom_style = None
    pair_style = None
    fixes: list[str] = []
    run_commands: list[str] = []
    thermo_headers: list[str] = []
    thermo_blocks: list[list[dict[str, float]]] = []
    current_header: list[str] | None = None
    current_block: list[dict[str, float]] = []

    def _flush_block():
        nonlocal current_block, current_header
        if current_header and current_block:
            thermo_blocks.append(current_block)
        current_block = []

    with open(path, "r", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("units") and units is None:
                units = stripped.split(maxsplit=1)[-1]
            if stripped.startswith("atom_style") and atom_style is None:
                atom_style = stripped.split(maxsplit=1)[-1]
            if stripped.startswith("pair_style") and pair_style is None:
                pair_style = stripped.split(maxsplit=1)[-1]
            if stripped.startswith("fix "):
                fixes.append(stripped)
            if stripped.startswith("run "):
                run_commands.append(stripped)

            # thermoヘッダー行: "Step Temp E_pair E_mol TotEng Press" のような形
            if re.match(r"^Step\s+\w", stripped):
                _flush_block()
                current_header = stripped.split()
                thermo_headers.append(stripped)
                continue

            if current_header is not None:
                if stripped.startswith("Loop time"):
                    _flush_block()
                    current_header = None
                    continue
                tokens = stripped.split()
                if len(tokens) == len(current_header):
                    try:
                        row = {k: float(v) for k, v in zip(current_header, tokens)}
                        current_block.append(row)
                        continue
                    except ValueError:
                        pass
        _flush_block()

    lines = [f"ファイルサイズ: {human_size(size)}"]
    lines.append(f"units: {units or '不明'}")
    lines.append(f"atom_style: {atom_style or '不明'}")
    lines.append(f"pair_style: {pair_style or '不明'}")

    ensemble_hits = set()
    for fx in fixes:
        for kw in ("nve", "nvt", "npt", "nph", "langevin", "berendsen"):
            if kw in fx.lower():
                ensemble_hits.add(kw)
    if ensemble_hits:
        lines.append(f"検出されたアンサンブル/積分手法 (fixコマンドから): {', '.join(sorted(ensemble_hits))}")
    lines.append(f"run コマンド数: {len(run_commands)}" + (f"（例: {run_commands[0]}）" if run_commands else ""))

    if thermo_blocks:
        lines.append(f"thermo出力ブロック数: {len(thermo_blocks)}")
        for i, block in enumerate(thermo_blocks):
            keys = list(block[0].keys())
            n_steps = len(block)
            first_row = block[0]
            last_row = block[-1]
            lines.append(f"  ブロック{i+1}: {n_steps}行, カラム={keys}")
            lines.append(f"    最初のstep: {first_row}")
            lines.append(f"    最後のstep: {last_row}")
            if "Temp" in keys:
                temps = [r["Temp"] for r in block]
                lines.append(f"    Temp範囲: {min(temps):.3f} ~ {max(temps):.3f}")
    else:
        warnings.append("thermo出力ブロックが検出できませんでした（ログの途中/カスタムthermo_styleの可能性）。")

    return ExtractionResult(
        category_label="LAMMPS ログファイル (log.lammps)",
        summary="\n".join(lines),
        metadata={
            "units": units,
            "pair_style": pair_style,
            "n_thermo_blocks": len(thermo_blocks),
        },
        warnings=warnings,
    )


def extract_input_script(path: str) -> ExtractionResult:
    """LAMMPS入力スクリプト（in.* など）を要約する。"""
    text = read_text_safely(path)
    lines_raw = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]

    def _grab(cmd: str) -> list[str]:
        return [l.strip() for l in lines_raw if l.strip().startswith(cmd)]

    summary_parts = [
        f"総有効行数（コメント/空行除く）: {len(lines_raw)}",
        "units: " + ("\n  ".join(_grab("units")) or "(なし)"),
        "atom_style: " + ("\n  ".join(_grab("atom_style")) or "(なし)"),
        "read_data / read_restart: " + ("\n  ".join(_grab("read_data") + _grab("read_restart")) or "(なし)"),
        "pair_style / pair_coeff: " + ("\n  ".join(_grab("pair_style") + _grab("pair_coeff")) or "(なし)"),
        "fix: " + ("\n  ".join(_grab("fix")) or "(なし)"),
        "run / minimize: " + ("\n  ".join(_grab("run") + _grab("minimize")) or "(なし)"),
        "dump: " + ("\n  ".join(_grab("dump")) or "(なし)"),
    ]

    return ExtractionResult(
        category_label="LAMMPS 入力スクリプト",
        summary="\n\n".join(summary_parts),
        raw_excerpt=head_tail(text, n_lines=40),
    )


def extract_data(path: str) -> ExtractionResult:
    """LAMMPS data ファイル: ヘッダ部分（原子数・タイプ数・box等）を要約する。"""
    warnings: list[str] = []
    header_info: dict[str, str] = {}
    section_names: list[str] = []

    patterns = {
        "atoms": re.compile(r"^\s*(\d+)\s+atoms\s*$"),
        "atom types": re.compile(r"^\s*(\d+)\s+atom types\s*$"),
        "bonds": re.compile(r"^\s*(\d+)\s+bonds\s*$"),
        "angles": re.compile(r"^\s*(\d+)\s+angles\s*$"),
        "xlo xhi": re.compile(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+xlo xhi\s*$"),
        "ylo yhi": re.compile(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+ylo yhi\s*$"),
        "zlo zhi": re.compile(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+zlo zhi\s*$"),
    }
    section_re = re.compile(r"^\s*(Atoms|Velocities|Masses|Bonds|Angles|Dihedrals|Pair Coeffs)\b")

    with open(path, "r", errors="ignore") as f:
        first_line = f.readline().strip()
        for line in f:
            if section_re.match(line):
                section_names.append(line.strip())
                if len(section_names) > 2 and "Atoms" in section_names:
                    break  # ヘッダ情報は十分集まったので打ち切り
                continue
            for key, pat in patterns.items():
                m = pat.match(line.strip())
                if m:
                    header_info[key] = line.strip()

    lines = [f"コメント行（1行目）: {first_line}"]
    for k in ("atoms", "atom types", "bonds", "angles", "xlo xhi", "ylo yhi", "zlo zhi"):
        if k in header_info:
            lines.append(f"{k}: {header_info[k]}")
    lines.append("検出されたセクション: " + (", ".join(section_names) if section_names else "(検出できず)"))

    return ExtractionResult(
        category_label="LAMMPS data ファイル（構造・トポロジー定義）",
        summary="\n".join(lines),
        warnings=warnings,
    )
