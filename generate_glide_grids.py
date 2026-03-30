#!/usr/bin/env python3
"""
generate_grids.py

For each .maegz in the parent directory (“../”):
  • Compute centroid of provided residues.
  • Create a subfolder "<basename>_grid".
  • Write a DICE‐style "<basename>.in" containing:

      FORCEFIELD   OPLS_2005
      GRID_CENTER   <cx>, <cy>, <cz>
      GRIDFILE      <basename>_grid.zip
      INNERBOX      20, 20, 20
      OUTERBOX      30, 30, 30
      RECEP_FILE    /full/path/to/<basename>.maegz
      USEFLEXMAE    YES

  • Run:
      $SCHRODINGER/run -WAIT glide <basename>.in

  so that each grid appears as "<basename>_grid/<basename>_grid.zip".

All Glide jobs run serially (one license at a time).
"""

import os
import sys
import subprocess
from schrodinger.structure import StructureReader

# ---------------------------------------
# USER‐CONFIGURABLE PARAMETERS
# ---------------------------------------
# Residues for GRID_CENTER_ASL:
CENTROID_RESIDUES = [71, 96, 99, 103, 194, 195, 198, 199, 202, 283, 286]

# Inner and outer grid box sizes (in Å):
INNER_BOX = [20, 20, 20]
OUTER_BOX = [30, 30, 30]
# ---------------------------------------

def find_schrodinger_root():
    s = os.environ.get("SCHRODINGER")
    if not s or not os.path.isdir(s):
        print("ERROR: $SCHRODINGER must point to your Schrodinger install.", file=sys.stderr)
        sys.exit(1)
    return s

def find_glide_exec(schrod_root):
    glide_path = os.path.join(schrod_root, "glide")
    if not os.path.isfile(glide_path) or not os.access(glide_path, os.X_OK):
        print(f"ERROR: cannot find or execute '{glide_path}'", file=sys.stderr)
        sys.exit(1)
    return glide_path

def get_residue_index(atom):
    if hasattr(atom, "resi"):
        return atom.resi
    if hasattr(atom, "resnum"):
        return atom.resnum
    return None

def compute_centroid(maegz_file):
    coords = []
    try:
        reader = StructureReader(maegz_file)
        struct = next(reader)
    except Exception as e:
        print(f"ERROR: Could not read '{maegz_file}': {e}", file=sys.stderr)
        return None

    for atom in struct.atom:
        idx = get_residue_index(atom)
        if idx in CENTROID_RESIDUES:
            coords.append((atom.x, atom.y, atom.z))

    if not coords:
        print(f"WARNING: No atoms found in residues {CENTROID_RESIDUES} for '{maegz_file}'", file=sys.stderr)
        return None

    cx = sum(x for x, _, _ in coords) / len(coords)
    cy = sum(y for _, y, _ in coords) / len(coords)
    cz = sum(z for _, _, z in coords) / len(coords)
    return cx, cy, cz

def write_in_file(subdir, base, abs_rec, centroid):
    """
    Write "<base>.in" inside subdir using the Glide GUI‐compatible format.
    """
    cx, cy, cz = centroid
    in_path = os.path.join(subdir, f"{base}.in")

    # Format GRID_CENTER with commas and spaces:
    grid_center_line = f"{cx:.6f}, {cy:.6f}, {cz:.6f}"

    with open(in_path, "w") as f:
        f.write("FORCEFIELD   OPLS_2005\n")
        f.write(f"GRID_CENTER   {grid_center_line}\n")
        f.write(f"GRIDFILE      {base}_grid.zip\n")
        f.write(f"INNERBOX      {INNER_BOX[0]}, {INNER_BOX[1]}, {INNER_BOX[2]}\n")
        f.write(f"OUTERBOX      {OUTER_BOX[0]}, {OUTER_BOX[1]}, {OUTER_BOX[2]}\n")
        f.write(f"RECEP_FILE    {abs_rec}\n")
        f.write("USEFLEXMAE    YES\n")
    return in_path

def main():
    schrod_root = find_schrodinger_root()
    glide_exe = find_glide_exec(schrod_root)

    # 1) Find all .maegz files in the parent directory ("../"):
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    all_maegz = sorted(fname for fname in os.listdir(parent_dir) if fname.lower().endswith(".maegz"))
    if not all_maegz:
        print("ERROR: No '.maegz' files found in ../", file=sys.stderr)
        sys.exit(1)

    for rec in all_maegz:
        base = os.path.splitext(rec)[0]
        subdir = f"{base}_grid"
        os.makedirs(subdir, exist_ok=True)

        abs_rec = os.path.join(parent_dir, rec)
        print(f"\n=== Processing {rec} → Subdir: {subdir} ===")

        centroid = compute_centroid(abs_rec)
        if centroid is None:
            print(f"SKIPPING {rec}: cannot compute centroid", file=sys.stderr)
            continue
        cx, cy, cz = centroid
        print(f"Computed centroid (logging): {cx:.6f}, {cy:.6f}, {cz:.6f}  (will use for GRID_CENTER)")

        in_path = write_in_file(subdir, base, abs_rec, centroid)
        print(f"→ Wrote input file: {in_path}")

        cmd = [
            os.path.join(schrod_root, "run"),
            "-WAIT",                      # <— wait for the grid to finish
            glide_exe,
            os.path.basename(in_path)
        ]
        print(f"→ Launching Glide: {' '.join(cmd)}\n  (inside {subdir})")
        proc = subprocess.Popen(
            cmd, cwd=subdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
        proc.wait()

        if proc.returncode != 0:
            print(f"ERROR: Glide returned exit code {proc.returncode} for {rec}", file=sys.stderr)
        else:
            print(f"SUCCESS: Grid created for {rec}. Check '{subdir}/{base}_grid.zip'")

    print("\n>>> All done. Inspect each *_grid/ folder for the *_grid.zip files. <<<\n")

if __name__ == "__main__":
    main()

