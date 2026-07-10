# whatsthis (wt)

Point `wt` at a single file (or now, a whole directory) from a computational
materials science / computational chemistry project, and an LLM explains:

- what kind of file it is
- what it does
- what information it contains
- what role it plays in a research workflow

```bash
wt OUTCAR
wt vasprun.xml
wt log.lammps
wt train.py
wt POSCAR
wt ./my_project_dir/     # NEW: directory mode
```

## Design

Two modes:

- **File mode** (default when TARGET is a file): understand one file in isolation.
- **Directory mode** (when TARGET is a directory): scan the whole tree, detect
  likely calculation setups (VASP run, LAMMPS run, ...) and cross-file
  references with cheap heuristics (no LLM), then ask the LLM to turn that
  into a narrative: what the project is, the role of each file, how files
  relate, and a suggested reading order.

Pipeline:

1. **`detect.py`** — classify a file's category from its name, extension, and
   (if needed) a peek at its content (e.g. VASP OUTCAR / POSCAR, LAMMPS log,
   Python script, CIF, ...).
2. **`extractors/`** — category-specific logic that turns a file into "just
   enough structured summary for the LLM to write an explanation". Huge files
   (hundreds of MB of OUTCAR / vasprun.xml) are streamed rather than loaded
   fully into memory.
   - `code.py`: Python/C/C++/Fortran/Bash. Small files pass through whole;
     large ones get their structure extracted (functions, classes, imports).
   - `data_formats.py`: JSON/YAML/TOML/Markdown. Extracts top-level key
     structure or heading structure.
   - `structure.py`: ASE-readable structure files (POSCAR/CONTCAR/CIF/XYZ/
     extxyz, etc.). Extracts atom count, elemental composition, cell info.
   - `vasp.py`: OUTCAR/vasprun.xml. Streams INCAR parameters, ionic-step
     count, final energy, temperature history, etc., and classifies the run
     as MD / relaxation / static.
   - `lammps.py`: log.lammps (thermo-block parsing), LAMMPS input scripts,
     LAMMPS data files.
3. **`dirscan.py`** / **`relations.py`** (directory mode only) — recursively
   list files with their detected category, and apply simple heuristics to
   spot calculation groups (e.g. INCAR+POSCAR+OUTCAR in the same folder) and
   script-to-file references (e.g. a Python script that opens `"POSCAR"`).
4. **`prompts.py`** — builds the system/user prompt, with category-specific
   hints about what to focus on, and the target answer language injected as
   a parameter.
5. **`llm.py`** — calls a local [Ollama](https://ollama.com) server
   (default `http://localhost:11434`) to generate the explanation. No extra
   dependency beyond the standard library (`urllib`). Default model:
   `qwen3.5:9b`.
6. **`cli.py`** — wires the above together and renders output with `rich`.

## Install

```bash
# 1. Install Ollama if you haven't: https://ollama.com
ollama serve &                # start the server (skip if already running)
ollama pull qwen3.5:9b        # pull the default model

# 2. Install whatsthis (wt)
cd whatsthis
pip install -e .
```

## Usage

```bash
wt OUTCAR                            # explain a file with Ollama (qwen3.5:9b)
wt OUTCAR --no-llm                   # just show the extracted summary (no Ollama needed)
wt train.py -v                       # also show detection reason / prompt sent to the LLM
wt POSCAR --model qwen2.5:14b        # use a different model
wt OUTCAR --host http://192.168.1.10:11434   # use a remote Ollama server
wt ./my_project_dir/                 # explain a whole directory
wt ./my_project_dir/ --max-files 200 # cap how many files are scanned

wt OUTCAR --lang ja                  # answer in Japanese
wt OUTCAR --lang "English"           # answer in English (also the default)
```

Defaults can also be set via environment variables (a CLI flag always wins):

| Env var         | Meaning                        | Default                  |
|------------------|---------------------------------|---------------------------|
| `WT_MODEL`       | Ollama model name               | `qwen3.5:9b`              |
| `OLLAMA_HOST`    | Ollama server URL               | `http://localhost:11434`  |
| `WT_MAX_TOKENS`  | Max tokens for the LLM response | `4000`                    |
| `WT_THINK`       | Enable "thinking" mode          | `false`                   |
| `WT_LANG`        | Answer language                 | `en`                      |
| `WT_MAX_FILES`   | Directory mode file-scan cap    | `500`                     |

## Supported files

- **Programming languages**: Python, C, C++, Fortran, Bash, JSON, YAML, TOML, Markdown
- **Computational-chemistry structure files (via ASE)**: POSCAR, CONTCAR, CIF, XYZ, extXYZ, LAMMPS data, and more
- **Calculation result files**: OUTCAR, vasprun.xml, log.lammps, LAMMPS input scripts
- **Directories**: any mix of the above, with heuristic relationship detection

## Known limitations / future work

- Directory mode's relationship detection is heuristic (filename/regex-based), not a real
  dependency analysis; it can miss or mis-detect relationships in complex projects.
- Quantum ESPRESSO input detection is basic (no dedicated parser yet).
- Very large files rely on summaries, so some fine-grained detail never reaches the LLM.
- Future ideas: missing-file inference, workflow inference, aider integration, a research agent.
