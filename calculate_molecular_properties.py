#!/usr/bin/env python3
"""
Compute RDKit properties from SMILES in a CSV using only the stdlib csv module.

Outputs columns:
MW, LogP, HBD, HBA, RotB, FormalCharge, InvalidSMILES

python3 calculate_molecular_proterties.py input.csv
"""

import argparse
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def calc_props(smiles: str):
    if smiles is None:
        return None
    smiles = smiles.strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input CSV path")
    ap.add_argument("-o", "--output", default="out_with_props.csv", help="Output CSV path")
    ap.add_argument("-s", "--smiles_col", default="SMILES", help="SMILES column header (default: SMILES)")
    args = ap.parse_args()

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("ERROR: Input CSV appears to have no header row.")

        if args.smiles_col not in reader.fieldnames:
            raise SystemExit(f"ERROR: smiles_col '{args.smiles_col}' not found. Columns: {reader.fieldnames}")

        out_fields = list(reader.fieldnames) + ["MW", "LogP", "HBD", "HBA", "RotB", "FormalCharge", "InvalidSMILES"]

        with open(args.output, "w", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                smi = row.get(args.smiles_col, "")
                props = calc_props(smi)
                if props is None:
                    row.update({"MW": "", "LogP": "", "HBD": "", "HBA": "", "RotB": "", "FormalCharge": ""})
                    row["InvalidSMILES"] = "True"
                else:
                    # write numbers with reasonable formatting
                    row.update({
                        "MW": f"{props['MW']:.4f}",
                        "LogP": f"{props['LogP']:.4f}",
                        "HBD": str(int(props["HBD"])),
                        "HBA": str(int(props["HBA"])),
                        "RotB": str(int(props["RotB"])),
                        "FormalCharge": str(int(props["FormalCharge"])),
                    })
                    row["InvalidSMILES"] = "False"

                writer.writerow(row)

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

