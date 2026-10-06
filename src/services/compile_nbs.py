#!/usr/bin/env python3
"""
compile_nbs.py — QC-VORONOI Notebook Source Compiler
====================================================
Compiles all notebooks in src/ directly (root-level only, excluding tests/)
into segregated .txt files by project phase in src/docs/telemetry/.
"""

import json
import sys
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent  # src/services/
SRC_DIR = SCRIPT_DIR.parent                   # src/
DOCS_DIR = SRC_DIR / "docs"
TELEMETRY_DIR = DOCS_DIR / "telemetry"

PHASE_MAP = {
    "1": "Phase 1 - Data Engineering",
    "2": "Phase 2 - Spatial Quality Control",
    "3": "Phase 3 - Parameter Optimization",
    "4": "Phase 4 - Statistical Analytics and Validation",
}


def compile_notebooks():
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(SRC_DIR.glob("[1-4].*.ipynb"))
    if not notebooks:
        print("[INFO] No numbered notebooks found in src/ yet.")
        return

    phase_code = defaultdict(list)

    for nb_path in notebooks:
        match = re.match(r"^([1-4])\.", nb_path.name)
        if not match:
            continue
        phase = match.group(1)

        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            print(f"[WARN] Error reading {nb_path.name}: {e}")
            continue

        nb_lines = [f"\n{'='*70}\n# NOTEBOOK: {nb_path.name}\n{'='*70}\n"]
        for cell_idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                if src.strip():
                    nb_lines.append(f"\n# --- Code Cell {cell_idx + 1} ---\n{src}\n")

        phase_code[phase].extend(nb_lines)

    for phase, lines in phase_code.items():
        phase_title = PHASE_MAP.get(phase, f"Phase {phase}")
        out_file = TELEMETRY_DIR / f"compiled_phase_{phase}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"=== QC-VORONOI: {phase_title} ===\n")
            f.write(f"Compiled on: {datetime.now().isoformat()}\n")
            f.writelines(lines)
        print(f"[OK] Compiled {phase_title} -> {out_file.name}")


if __name__ == "__main__":
    compile_notebooks()
