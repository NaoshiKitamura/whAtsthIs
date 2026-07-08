"""
ASEで読める構造ファイル（POSCAR / CONTCAR / CIF / XYZ / extxyz / LAMMPS data など）用の抽出器。
"""

from __future__ import annotations

from collections import Counter

from .base import ExtractionResult


def _describe_atoms(atoms) -> dict:
    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)
    formula = atoms.get_chemical_formula()
    cell = atoms.get_cell()
    lengths_angles = cell.cellpar() if cell.rank == 3 else None
    pbc = atoms.get_pbc().tolist()

    info = {
        "formula": formula,
        "n_atoms": len(atoms),
        "elements": dict(sorted(counts.items())),
        "pbc": pbc,
    }
    if lengths_angles is not None:
        a, b, c, alpha, beta, gamma = lengths_angles
        info["cell_lengths_angstrom"] = {"a": round(float(a), 4), "b": round(float(b), 4), "c": round(float(c), 4)}
        info["cell_angles_degree"] = {"alpha": round(float(alpha), 2), "beta": round(float(beta), 2), "gamma": round(float(gamma), 2)}
        try:
            info["volume_angstrom3"] = round(float(atoms.get_volume()), 3)
        except Exception:  # noqa: BLE001
            pass
    return info


def extract(path: str, category_label: str, ase_format: str | None = None) -> ExtractionResult:
    from ase.io import read

    warnings: list[str] = []
    frames = None

    try:
        frames = read(path, index=":", format=ase_format)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"ase.io.read(index=':')での読み込みに失敗: {e}")
        try:
            single = read(path, format=ase_format)
            frames = [single]
        except Exception as e2:  # noqa: BLE001
            return ExtractionResult(
                category_label=category_label,
                summary=f"ASEでの構造読み込みに失敗しました: {e2}",
                warnings=[str(e2)],
            )

    n_frames = len(frames)
    first = frames[0]
    last = frames[-1]

    info = _describe_atoms(first)

    lines = [f"検出フレーム数: {n_frames}" + ("（複数フレーム = 軌跡/トラジェクトリの可能性）" if n_frames > 1 else "（単一構造）")]
    lines.append(f"化学式: {info['formula']}")
    lines.append(f"原子数: {info['n_atoms']}")
    lines.append("元素と数: " + ", ".join(f"{el}:{n}" for el, n in info["elements"].items()))
    lines.append(f"周期境界条件 (PBC): {info['pbc']}")
    if "cell_lengths_angstrom" in info:
        la = info["cell_lengths_angstrom"]
        an = info["cell_angles_degree"]
        lines.append(f"セル長 (Å): a={la['a']}, b={la['b']}, c={la['c']}")
        lines.append(f"セル角 (deg): alpha={an['alpha']}, beta={an['beta']}, gamma={an['gamma']}")
        if "volume_angstrom3" in info:
            lines.append(f"セル体積: {info['volume_angstrom3']} Å^3")

    if n_frames > 1:
        last_info = _describe_atoms(last)
        if last_info["formula"] != info["formula"]:
            lines.append(f"最終フレームの化学式: {last_info['formula']}（先頭フレームと異なる＝原子数/種類が変化）")
        lines.append("複数フレームがあるため、MD軌跡・構造最適化の履歴・NEB経路などの可能性があります。")

    # calculator付き（エネルギー計算済み）かどうか
    try:
        energy = first.get_potential_energy()
        lines.append(f"先頭フレームに計算済みポテンシャルエネルギーが付随: {energy:.6f} eV")
    except Exception:  # noqa: BLE001
        pass

    return ExtractionResult(
        category_label=category_label,
        summary="\n".join(lines),
        metadata={"n_frames": n_frames, **info},
        warnings=warnings,
    )
