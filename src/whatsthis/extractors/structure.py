"""
Extractor for structure files ASE can read (POSCAR / CONTCAR / CIF / XYZ /
extxyz / LAMMPS data, etc.).
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
        warnings.append(f"ase.io.read(index=':') failed: {e}")
        try:
            single = read(path, format=ase_format)
            frames = [single]
        except Exception as e2:  # noqa: BLE001
            return ExtractionResult(
                category_label=category_label,
                summary=f"ASE failed to read this structure file: {e2}",
                warnings=[str(e2)],
            )

    n_frames = len(frames)
    first = frames[0]
    last = frames[-1]

    info = _describe_atoms(first)

    lines = [f"Detected frames: {n_frames}" + (" (multiple frames -> possibly a trajectory)" if n_frames > 1 else " (single structure)")]
    lines.append(f"Chemical formula: {info['formula']}")
    lines.append(f"Number of atoms: {info['n_atoms']}")
    lines.append("Elements and counts: " + ", ".join(f"{el}:{n}" for el, n in info["elements"].items()))
    lines.append(f"Periodic boundary conditions (PBC): {info['pbc']}")
    if "cell_lengths_angstrom" in info:
        la = info["cell_lengths_angstrom"]
        an = info["cell_angles_degree"]
        lines.append(f"Cell lengths (A): a={la['a']}, b={la['b']}, c={la['c']}")
        lines.append(f"Cell angles (deg): alpha={an['alpha']}, beta={an['beta']}, gamma={an['gamma']}")
        if "volume_angstrom3" in info:
            lines.append(f"Cell volume: {info['volume_angstrom3']} A^3")

    if n_frames > 1:
        last_info = _describe_atoms(last)
        if last_info["formula"] != info["formula"]:
            lines.append(f"Last frame's formula: {last_info['formula']} (differs from first frame -> composition changes over frames)")
        lines.append("Multiple frames present, possibly an MD trajectory, optimization history, or NEB path.")

    try:
        energy = first.get_potential_energy()
        lines.append(f"First frame has an attached potential energy: {energy:.6f} eV")
    except Exception:  # noqa: BLE001
        pass

    return ExtractionResult(
        category_label=category_label,
        summary="\n".join(lines),
        metadata={"n_frames": n_frames, **info},
        warnings=warnings,
    )
