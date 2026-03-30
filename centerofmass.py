from pymol import cmd
import glob
import csv
import os

# Run: pymol -cq -r centerofmass.py

def main():
    mol2_files = sorted(glob.glob("*.mol2"))

    with open("ligand_com.csv", "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["file", "state", "x", "y", "z"])

        for i, mol2_file in enumerate(mol2_files, start=1):
            obj = f"obj_{i}"
            cmd.load(mol2_file, obj)

            n_states = cmd.count_states(obj)
            if n_states < 1:
                n_states = 1

            for state in range(1, n_states + 1):
                com = cmd.centerofmass(obj, state=state)
                writer.writerow([mol2_file, state, com[0], com[1], com[2]])

            cmd.delete(obj)

    cmd.quit()

if __name__ == "__main__":
    main()
