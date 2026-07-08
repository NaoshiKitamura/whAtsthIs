"""
VASPの計算結果ファイル（OUTCAR / vasprun.xml）専用の抽出器。

方針:
  OUTCARやvasprun.xmlは数百MBになることがあるため、
  ファイル全体をメモリに載せず1行/1要素ずつストリーム処理し、
  必要な統計・最終状態のみを抽出する。
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
        return "IBRION/NSWが検出できず、静的計算かMD/構造最適化かは判別できませんでした。"
    if nsw == 0:
        return f"NSW=0 のため、構造は動かさない静的（single-point）計算です。"
    if ibrion == 0:
        return f"IBRION=0, NSW={nsw} のため、これは分子動力学（AIMD）計算です。"
    if ibrion in (1, 2, 3):
        return f"IBRION={ibrion}, NSW={nsw} のため、これは構造最適化（ジオメトリ緩和）計算です。"
    return f"IBRION={ibrion}, NSW={nsw}（判別が難しい設定です）。"


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

    lines = [f"ファイルサイズ: {human_size(size)}"]
    lines.append(f"原子数 (NIONS): {nions if nions is not None else '不明'}")
    lines.append("元素種 (POTCARから検出): " + (", ".join(dict.fromkeys(potcar_elements)) if potcar_elements else "不明"))
    lines.append(_ibrion_nsw_description(ibrion, nsw))
    lines.append(f"検出されたイオンステップ数（POSITION/TOTAL-FORCEブロック）: {n_ionic_steps}")

    for k in _INCAR_INT_KEYS + _INCAR_FLOAT_KEYS:
        if k in incar_vals:
            lines.append(f"  {k} = {incar_vals[k]}")

    if last_toten is not None:
        lines.append(f"最終ステップの自由エネルギー (free energy TOTEN): {last_toten} eV")
    else:
        warnings.append("TOTENの値が検出できませんでした。")

    if n_temp_samples > 0:
        lines.append(f"温度サンプル数: {n_temp_samples}, 先頭: {first_temp} K, 最終: {last_temp} K"
                      "（MD計算の場合、系の温度推移を示す）")

    if last_stress_line:
        lines.append(f"最終ステップの応力テンソル行 (in kB): {last_stress_line}")

    if saw_force_block:
        lines.append("原子に働く力（Force）の情報が各イオンステップで取得可能です。")

    return ExtractionResult(
        category_label="VASP OUTCAR（計算結果ログ）",
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
    last_cell = None
    last_volume = None

    try:
        context = ET.iterparse(path, events=("start", "end"))
        current_calc_energy = {}
        in_incar = False
        in_atominfo_array = False
        atom_array_field_names: list[str] = []

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
                # <array name="atoms"> ... <set><rc><c>Element</c><c>1</c></rc>...
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
                # エネルギー抽出
                for en in elem.iter():
                    en_tag = _NS_STRIP_RE.sub("", en.tag)
                    if en_tag == "i" and en.get("name") == "e_fr_energy":
                        try:
                            last_energy = float((en.text or "").strip())
                        except ValueError:
                            pass
                # 最終構造のセル・体積
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
                elem.clear()  # メモリ節約：処理済みのcalculationは破棄

    except ET.ParseError as e:
        warnings.append(f"XMLパース中にエラー（部分的な結果の可能性）: {e}")

    nions = len(atom_symbols)
    from collections import Counter
    elem_counts = Counter(atom_symbols)

    ibrion = incar.get("IBRION")
    nsw = incar.get("NSW")
    ibrion_i = int(ibrion) if ibrion not in (None, "") else None
    nsw_i = int(nsw) if nsw not in (None, "") else None

    lines = [f"ファイルサイズ: {human_size(size)}"]
    lines.append(f"原子数: {nions if nions else '不明'}")
    if elem_counts:
        lines.append("元素と数: " + ", ".join(f"{el}:{n}" for el, n in sorted(elem_counts.items())))
    lines.append(_ibrion_nsw_description(ibrion_i, nsw_i))
    lines.append(f"<calculation> ブロック数（≒イオンステップ数）: {n_calculations}")

    interesting_incar = ["ENCUT", "EDIFF", "ISMEAR", "ISPIN", "ALGO", "PREC", "LORBIT", "TEBEG", "TEEND"]
    for k in interesting_incar:
        if k in incar:
            lines.append(f"  INCAR {k} = {incar[k]}")

    if last_energy is not None:
        lines.append(f"最終ステップのフリーエネルギー (e_fr_energy): {last_energy} eV")
    if last_volume is not None:
        lines.append(f"最終ステップのセル体積: {last_volume} Å^3")

    return ExtractionResult(
        category_label="VASP vasprun.xml（計算結果XML）",
        summary="\n".join(lines),
        metadata={
            "nions": nions,
            "n_calculations": n_calculations,
            "final_energy_eV": last_energy,
        },
        warnings=warnings,
    )
