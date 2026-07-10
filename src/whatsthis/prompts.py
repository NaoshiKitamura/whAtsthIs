"""
Builds the prompts sent to the LLM.

Design:
  - A fixed system prompt establishes the "computational materials science
    assistant" persona and the basic output format, with the answer language
    injected as a parameter (see config.DEFAULT_LANGUAGE / --lang).
  - The user prompt carries the extractor's summary, not the raw file (to
    keep large files out of the context window).
  - Per-category hints tell the model what to focus on for that file type.
"""

from __future__ import annotations

from .detect import Category, Detection
from .extractors.base import ExtractionResult

SYSTEM_PROMPT_TEMPLATE = """\
You are an assistant that helps computational materials science / computational chemistry \
researchers understand files in their research folders.

The user selects a single file, and you are given a summary (extracted metadata, not the \
raw file) describing it. Your job is to explain, concisely and concretely:

- What kind of file this is (format, and which software produced it)
- What this file does / represents
- What information it contains
- What role it typically plays in a research workflow

Respond in {language}, using Markdown with headers and bullet points where useful.
Do not state anything as fact that isn't supported by the provided summary -- if something \
is unclear or missing from the summary, say so explicitly ("unknown" / "cannot be determined \
from the given information") rather than guessing confidently.
Briefly clarify technical terms for a non-expert reader, but keep the overall explanation concise.
"""

_CATEGORY_HINTS: dict[Category, str] = {
    Category.PYTHON: (
        "This is a Python script. Cover: (1) the overall purpose of the script, "
        "(2) the main functions/classes and their roles, (3) the rough execution flow, "
        "(4) key libraries used and what they're used for, "
        "(5) which stage of a research workflow this likely belongs to "
        "(preprocessing / running a calculation / postprocessing-analysis / visualization, etc.)."
    ),
    Category.C: "This is C source code. Explain its purpose, main functions, and likely use (numerical routine / driver / utility, etc.).",
    Category.CPP: "This is C++ source code. Explain its purpose, main classes/functions, and likely use.",
    Category.FORTRAN: "This is Fortran source code, likely part of a numerical simulation code (e.g. first-principles or MD). Explain module structure, key subroutines/functions, and likely use.",
    Category.BASH: "This is a shell script, likely a job-submission or workflow-automation script. Explain what it runs and which tools it invokes.",
    Category.JSON: "This is a JSON config/data file. Infer and explain the purpose of the main keys.",
    Category.YAML: "This is a YAML config file. Infer and explain the purpose of the main keys (CI config, calculation parameters, environment definition, etc.).",
    Category.TOML: "This is a TOML config file. Explain the purpose of the main keys (e.g. pyproject.toml -> Python package configuration).",
    Category.MARKDOWN: "This is a Markdown document. Infer what kind of document it is (README / report / notes / etc.) from its heading structure.",
    Category.VASP_OUTCAR: (
        "This is a VASP OUTCAR file. Make clear: (1) that it's a VASP calculation log, "
        "(2) whether it's a static calculation / structure relaxation / molecular dynamics (MD) run, "
        "(3) what information is available (energy, forces, stress, temperature, etc.), "
        "(4) what research purpose this calculation likely served (structure relaxation, property "
        "calculation, MD simulation, etc.)."
    ),
    Category.VASP_VASPRUN: (
        "This is a VASP vasprun.xml file. Cover the same points as OUTCAR, and also note the benefit "
        "of its machine-readable XML format (e.g. easy to parse programmatically with pymatgen/ASE)."
    ),
    Category.VASP_POSCAR: (
        "This is a VASP structure file (POSCAR/CONTCAR). Explain the number of atoms, elemental "
        "composition, cell information, and (as far as can be inferred) the nature of the structure "
        "(bulk crystal / surface slab / molecule / defect structure, etc.). If it's a CONTCAR, note "
        "that it represents the final/current structure from a relaxation or MD run."
    ),
    Category.VASP_INCAR: "This is a VASP settings file (INCAR). Based on the parameters set, explain what kind of calculation is intended.",
    Category.VASP_KPOINTS: "This is a VASP k-point sampling settings file (KPOINTS). Explain the k-point mesh/sampling scheme.",
    Category.VASP_XDATCAR: "This is a VASP XDATCAR file (a trajectory of atomic coordinates). Explain that it's a time series of coordinates from an MD run or relaxation.",
    Category.LAMMPS_LOG: (
        "This is a LAMMPS log file. Cover: (1) the simulation setup (units, pair_style, ensemble), "
        "(2) what the thermo output indicates was run (how many steps, temperature/energy trends), "
        "(3) how this log is likely used in a research workflow (property evaluation, equilibration, "
        "production run, etc.)."
    ),
    Category.LAMMPS_INPUT: "This is a LAMMPS input script. Explain the simulation setup (system definition, potential, ensemble, what is executed).",
    Category.LAMMPS_DATA: "This is a LAMMPS data file (atom configuration/topology definition). Explain the system's scale/type based on atom count, atom types, and box size.",
    Category.QE_INPUT: "This is a Quantum ESPRESSO input file. Based on the &CONTROL, &SYSTEM, etc. namelists, explain what kind of calculation is intended (scf, relax, md, etc.).",
    Category.STRUCTURE_CIF: "This is a CIF-format crystal structure file. Explain the structural information it contains: composition, cell info, symmetry, etc.",
    Category.STRUCTURE_XYZ: "This is an XYZ-format structure file. Explain the number of atoms, elemental composition, and (if multiple frames) that it may be a trajectory.",
    Category.STRUCTURE_EXTXYZ: "This is an extended-XYZ structure file. Note that, beyond coordinates, it often carries extra info like energies/forces, and is commonly used as machine-learning potential training data.",
}


def build_prompt(filename: str, detection: Detection, extraction: ExtractionResult, language: str) -> tuple[str, str]:
    hint = _CATEGORY_HINTS.get(detection.category, "")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language)

    parts = [
        f"# Target file\n`{filename}`",
        f"# Detected category\n{extraction.category_label}\n(reason: {detection.reason})",
    ]
    if hint:
        parts.append(f"# What to focus on for this category\n{hint}")

    parts.append(f"# Extracted summary\n```\n{extraction.summary}\n```")

    if extraction.raw_excerpt:
        parts.append(f"# Raw text excerpt (for reference; not the full file)\n```\n{extraction.raw_excerpt}\n```")

    if extraction.warnings:
        w = "\n".join(f"- {w}" for w in extraction.warnings)
        parts.append(f"# Caveats from extraction\n{w}")

    user_prompt = "\n\n".join(parts)
    return system_prompt, user_prompt


# --- Directory-analysis mode ---

DIRECTORY_SYSTEM_PROMPT_TEMPLATE = """\
You are an assistant that helps computational materials science / computational chemistry \
researchers understand a folder of files -- which they may not have created themselves \
(e.g. handed off by a collaborator, or an old project being revisited).

You are given:
- A directory tree listing (relative path, detected category, file size)
- Short, automatically generated notes about detected calculation setups \
(e.g. "this looks like a VASP run directory")
- Notes about which scripts appear to reference which other files

Your job is to explain, in {language}, using Markdown with headers and bullet points:
- What kind of research project or calculation workflow this directory represents
- The role of each major file or group of files
- How the files relate to each other (inputs -> outputs, scripts -> data, etc.)
- A suggested reading order / entry point for someone new to this folder

Do not invent facts beyond what the provided information supports. If something is unclear, \
say so explicitly rather than guessing confidently. Keep the explanation concise but complete.
"""


def build_directory_prompt(
    root_label: str,
    tree_lines: list[str],
    relation_notes: list[str],
    language: str,
    truncated: bool = False,
) -> tuple[str, str]:
    system_prompt = DIRECTORY_SYSTEM_PROMPT_TEMPLATE.format(language=language)

    parts = [
        f"# Root directory\n`{root_label}`",
        "# File tree (relative_path : category : size)\n```\n" + "\n".join(tree_lines) + "\n```",
    ]
    if truncated:
        parts.append("# Note\nThe file listing was truncated (too many files); the tree above is only a subset.")
    if relation_notes:
        parts.append("# Automatically detected relationships / hints\n" + "\n".join(f"- {n}" for n in relation_notes))
    else:
        parts.append("# Automatically detected relationships / hints\n(none detected)")

    user_prompt = "\n\n".join(parts)
    return system_prompt, user_prompt
