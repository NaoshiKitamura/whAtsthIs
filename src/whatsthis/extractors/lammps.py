"""
Extractor for LAMMPS-related files (log.lammps / input scripts / data files).
"""

from __future__ import annotations

import re
import os

from .base import ExtractionResult, human_size, read_text_safely, head_tail


def extract_log(path: str) -> ExtractionResult:
    """log.lammps: detect thermo output blocks and summarize run conditions/stats."""
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

            # thermo header line, e.g. "Step Temp E_pair E_mol TotEng Press"
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

    lines = [f"File size: {human_size(size)}"]
    lines.append(f"units: {units or 'unknown'}")
    lines.append(f"atom_style: {atom_style or 'unknown'}")
    lines.append(f"pair_style: {pair_style or 'unknown'}")

    ensemble_hits = set()
    for fx in fixes:
        for kw in ("nve", "nvt", "npt", "nph", "langevin", "berendsen"):
            if kw in fx.lower():
                ensemble_hits.add(kw)
    if ensemble_hits:
        lines.append(f"Detected ensemble/integrator (from fix commands): {', '.join(sorted(ensemble_hits))}")
    lines.append(f"Number of run commands: {len(run_commands)}" + (f" (e.g. {run_commands[0]})" if run_commands else ""))

    if thermo_blocks:
        lines.append(f"Thermo output blocks: {len(thermo_blocks)}")
        for i, block in enumerate(thermo_blocks):
            keys = list(block[0].keys())
            n_steps = len(block)
            first_row = block[0]
            last_row = block[-1]
            lines.append(f"  Block {i+1}: {n_steps} rows, columns={keys}")
            lines.append(f"    First step: {first_row}")
            lines.append(f"    Last step: {last_row}")
            if "Temp" in keys:
                temps = [r["Temp"] for r in block]
                lines.append(f"    Temp range: {min(temps):.3f} ~ {max(temps):.3f}")
    else:
        warnings.append("No thermo output blocks detected (this could be a partial log or a custom thermo_style).")

    return ExtractionResult(
        category_label="LAMMPS log file (log.lammps)",
        summary="\n".join(lines),
        metadata={
            "units": units,
            "pair_style": pair_style,
            "n_thermo_blocks": len(thermo_blocks),
        },
        warnings=warnings,
    )


def extract_input_script(path: str) -> ExtractionResult:
    """Summarize a LAMMPS input script (in.* etc.)."""
    text = read_text_safely(path)
    lines_raw = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]

    def _grab(cmd: str) -> list[str]:
        return [l.strip() for l in lines_raw if l.strip().startswith(cmd)]

    summary_parts = [
        f"Effective lines (excluding comments/blank lines): {len(lines_raw)}",
        "units: " + ("\n  ".join(_grab("units")) or "(none)"),
        "atom_style: " + ("\n  ".join(_grab("atom_style")) or "(none)"),
        "read_data / read_restart: " + ("\n  ".join(_grab("read_data") + _grab("read_restart")) or "(none)"),
        "pair_style / pair_coeff: " + ("\n  ".join(_grab("pair_style") + _grab("pair_coeff")) or "(none)"),
        "fix: " + ("\n  ".join(_grab("fix")) or "(none)"),
        "run / minimize: " + ("\n  ".join(_grab("run") + _grab("minimize")) or "(none)"),
        "dump: " + ("\n  ".join(_grab("dump")) or "(none)"),
    ]

    return ExtractionResult(
        category_label="LAMMPS input script",
        summary="\n\n".join(summary_parts),
        raw_excerpt=head_tail(text, n_lines=40),
    )


def extract_data(path: str) -> ExtractionResult:
    """LAMMPS data file: summarize header info (atom/type counts, box, etc.)."""
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
                    break  # enough header info collected, stop early
                continue
            for key, pat in patterns.items():
                m = pat.match(line.strip())
                if m:
                    header_info[key] = line.strip()

    lines = [f"Comment line (line 1): {first_line}"]
    for k in ("atoms", "atom types", "bonds", "angles", "xlo xhi", "ylo yhi", "zlo zhi"):
        if k in header_info:
            lines.append(f"{k}: {header_info[k]}")
    lines.append("Detected sections: " + (", ".join(section_names) if section_names else "(none detected)"))

    return ExtractionResult(
        category_label="LAMMPS data file (structure/topology definition)",
        summary="\n".join(lines),
        warnings=warnings,
    )
