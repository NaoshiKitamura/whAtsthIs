"""
Extractor for VASP result files (OUTCAR / vasprun.xml).

OUTCAR and vasprun.xml can be hundreds of MB, so both are processed as a
stream (line-by-line / iterparse) rather than loaded fully into memory; only
the statistics and final-state values needed for the summary are kept.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .base import ExtractionResult, human_size
import os


_INCAR_INT_KEYS = ("IBRION", "NSW", "ISIF", "ISMEAR", "ISPIN", "LORBIT")
_INCAR_FLOAT_KEYS = ("POTIM", "ENCUT", "EDIFF", "TEBEG", "TEEND")

_KEY_RE = {k: re.compile(rf"\b{k}\s*=\s*(-?\d+)") for k in _INCAR_INT_KEYS}
_KEY_RE.update({k: re.compile(rf"\b{k}\s*=\s*(-?[\d.]+E?[+-]?\d*)") for k in _INCAR_FLOAT_KEYS})

_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*(-?\d+\.\d+)")
_TEMP_RE = re.compile(r"temperature\s+([\d.]+)\s*K")
_NIONS_RE = re.compile(r"NIONS\s*=\s*(\d+)")


def _ibrion_nsw_description(ibrion, nsw) -> str:
    if ibrion is None or nsw is None:
        return "IBRION/NSW could not be detected, so it's unclear whether this is a static, relaxation, or MD run."
    if nsw == 0:
        return "NSW=0, so this is a static (single-point) calculation with no ionic motion."
    if ibrion == 0:
        return f"IBRION=0, NSW={nsw}: this is an ab initio molecular dynamics (AIMD) run."
    if ibrion in (1, 2, 3):
        return f"IBRION={ibrion}, NSW={nsw}: this is a geometry/structure relaxation run."
    return f"IBRION={ibrion}, NSW={nsw} (an unusual combination; hard to classify automatically)."


def extract_outcar(path: str) -> ExtractionResult:
    size = os.path.getsize(path)
    warnings: list[str] = []

    incar_vals: dict[str, str] = {}
    potcar_elements: list[str] = []
    nions = None
    n_ionic_steps = 0
    last_toten = None
    first_temp = None
    last_temp = None
    n_temp_samples = 0
    last_stress_line = ""
    saw_force_block = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "VRHFIN" in line:
                m = re.search(r"VRHFIN\s*=\s*([A-Za-z]+)", line)
                if m:
                    potcar_elements.append(m.group(1))
            if nions is None and "NIONS" in line:
                m = _NIONS_RE.search(line)
                if m:
                    nions = int(m.group(1))
            for k, pat in _KEY_RE.items():
                if k not in incar_vals and k in line:
                    m = pat.search(line)
                    if m:
                        incar_vals[k] = m.group(1)
            if "TOTAL-FORCE" in line and "POSITION" in line:
                n_ionic_steps += 1
                saw_force_block = True
            m = _TOTEN_RE.search(line)
            if m:
                last_toten = float(m.group(1))
            m = _TEMP_RE.search(line)
            if m:
                t = float(m.group(1))
                if first_temp is None:
                    first_temp = t
                last_temp = t
                n_temp_samples += 1
            if "in kB" in line:
                last_stress_line = line.strip()

    ibrion = int(incar_vals["IBRION"]) if "IBRION" in incar_vals else None
    nsw = int(incar_vals["NSW"]) if "NSW" in incar_vals else None

    lines = [f"File size: {human_size(size)}"]
    lines.append(f"Number of atoms (NIONS): {nions if nions is not None else 'unknown'}")
    lines.append("Elements (detected from POTCAR): " + (", ".join(dict.fromkeys(potcar_elements)) if potcar_elements else "unknown"))
    lines.append(_ibrion_nsw_description(ibrion, nsw))
    lines.append(f"Detected ionic steps (POSITION/TOTAL-FORCE blocks): {n_ionic_steps}")

    for k in _INCAR_INT_KEYS + _INCAR_FLOAT_KEYS:
        if k in incar_vals:
            lines.append(f"  {k} = {incar_vals[k]}")

    if last_toten is not None:
        lines.append(f"Final-step free energy (TOTEN): {last_toten} eV")
    else:
        warnings.append("Could not find a TOTEN value.")

    if n_temp_samples > 0:
        lines.append(f"Temperature samples: {n_temp_samples}, first: {first_temp} K, last: {last_temp} K "
                      "(indicates the system temperature over an MD run)")

    if last_stress_line:
        lines.append(f"Final-step stress tensor line (in kB): {last_stress_line}")

    if saw_force_block:
        lines.append("Per-atom forces are available for each ionic step.")

    return ExtractionResult(
        category_label="VASP OUTCAR (calculation log)",
        summary="\n".join(lines),
        metadata={
            "nions": nions,
            "ibrion": ibrion,
            "nsw": nsw,
            "n_ionic_steps": n_ionic_steps,
            "final_energy_eV": last_toten,
        },
        warnings=warnings,
    )


# --- vasprun.xml ---

_NS_STRIP_RE = re.compile(r"\{.*\}")


def extract_vasprun(path: str) -> ExtractionResult:
    size = os.path.getsize(path)
    warnings: list[str] = []

    incar: dict[str, str] = {}
    atom_symbols: list[str] = []
    n_calculations = 0
    last_energy = None
    last_volume = None

    try:
        context = ET.iterparse(path, events=("start", "end"))
        in_incar = False

        for event, elem in context:
            tag = _NS_STRIP_RE.sub("", elem.tag)

            if event == "start" and tag == "incar":
                in_incar = True
            if event == "end" and tag == "incar":
                in_incar = False

            if event == "end" and tag == "i" and in_incar:
                name = elem.get("name")
                if name:
                    incar[name] = (elem.text or "").strip()

            if event == "end" and tag == "atominfo":
                for arr in elem.iter():
                    arr_tag = _NS_STRIP_RE.sub("", arr.tag)
                    if arr_tag == "array" and arr.get("name") == "atoms":
                        for rc in arr.iter():
                            rc_tag = _NS_STRIP_RE.sub("", rc.tag)
                            if rc_tag == "rc":
                                cs = [c.text.strip() for c in rc if (c.text or "").strip()]
                                if cs:
                                    atom_symbols.append(cs[0])
                elem.clear()

            if event == "end" and tag == "calculation":
                n_calculations += 1
                for en in elem.iter():
                    en_tag = _NS_STRIP_RE.sub("", en.tag)
                    if en_tag == "i" and en.get("name") == "e_fr_energy":
                        try:
                            last_energy = float((en.text or "").strip())
                        except ValueError:
                            pass
                for st in elem.iter():
                    st_tag = _NS_STRIP_RE.sub("", st.tag)
                    if st_tag == "crystal":
                        for v in st.iter():
                            v_tag = _NS_STRIP_RE.sub("", v.tag)
                            if v_tag == "i" and v.get("name") == "volume":
                                try:
                                    last_volume = float((v.text or "").strip())
                                except ValueError:
                                    pass
                elem.clear()  # free memory for already-processed calculation blocks

    except ET.ParseError as e:
        warnings.append(f"Error while parsing XML (result may be partial): {e}")

    nions = len(atom_symbols)
    from collections import Counter
    elem_counts = Counter(atom_symbols)

    ibrion = incar.get("IBRION")
    nsw = incar.get("NSW")
    ibrion_i = int(ibrion) if ibrion not in (None, "") else None
    nsw_i = int(nsw) if nsw not in (None, "") else None

    lines = [f"File size: {human_size(size)}"]
    lines.append(f"Number of atoms: {nions if nions else 'unknown'}")
    if elem_counts:
        lines.append("Elements and counts: " + ", ".join(f"{el}:{n}" for el, n in sorted(elem_counts.items())))
    lines.append(_ibrion_nsw_description(ibrion_i, nsw_i))
    lines.append(f"<calculation> block count (~ number of ionic steps): {n_calculations}")

    interesting_incar = ["ENCUT", "EDIFF", "ISMEAR", "ISPIN", "ALGO", "PREC", "LORBIT", "TEBEG", "TEEND"]
    for k in interesting_incar:
        if k in incar:
            lines.append(f"  INCAR {k} = {incar[k]}")

    if last_energy is not None:
        lines.append(f"Final-step free energy (e_fr_energy): {last_energy} eV")
    if last_volume is not None:
        lines.append(f"Final-step cell volume: {last_volume} A^3")

    return ExtractionResult(
        category_label="VASP vasprun.xml (calculation result XML)",
        summary="\n".join(lines),
        metadata={
            "nions": nions,
            "n_calculations": n_calculations,
            "final_energy_eV": last_energy,
        },
        warnings=warnings,
    )
