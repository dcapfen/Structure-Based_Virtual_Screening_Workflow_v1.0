#!/usr/bin/env python3
import os
import glob
import re
import textwrap
import argparse
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def parse_args():
    p = argparse.ArgumentParser(
        description="ROC/AUC, LogAUC, Enrichment Score, and EF1% per grid, mapping ligand IDs via i_i_glide_lignum"
    )
    p.add_argument("-d", "--data_dir",    required=True,
                   help="Directory containing both *-DOCK_SP_*001.csv and *SP_OUT*.csv")
    p.add_argument("-l", "--ligand_file", required=True,
                   help="CSV of ALL ligands, columns=LIGAND_ID,ACTIVITY")
    p.add_argument("-o", "--output_dir",  default="roc_logauc_results",
                   help="Where to save ROC plots + summary CSV")
    p.add_argument("-m", "--missing_score", type=float, default=0.0,
                   help="Score to assign to any ligand missing from a grid")
    return p.parse_args()


def sanitize(name):
    # safe filename: only letters, digits, dot, underscore, hyphen
    return re.sub(r"[^\w\.-]", "_", name)


def wrap(text, width=30):
    # wrap long legend strings
    return "\n".join(textwrap.wrap(text, width=width))


def compute_logauc(fpr, tpr, a):
    """
    Paper definition (Knight & Naprienko):
      LogAUC(f; a) = (1 / -ln(a)) * integral_a^1 ROC(x) dx/x
    Treat ROC as a right-continuous step function.
    """
    if not (0.0 < a < 1.0):
        raise ValueError("cutoff a must be in (0,1)")

    area = 0.0

    # Step integral: on (fpr[i-1], fpr[i]] the ROC value is tpr[i]
    for i in range(1, len(fpr)):
        x_end = fpr[i]
        if x_end <= a:
            continue
        x_start = max(a, fpr[i - 1])
        if x_end > x_start:
            area += tpr[i] * (math.log(x_end) - math.log(x_start))

    # If ROC doesn't explicitly end at 1.0, extend the final step to 1.0
    last = max(a, fpr[-1])
    if last < 1.0:
        area += tpr[-1] * (math.log(1.0) - math.log(last))

    return area / (-math.log(a))


def find_sp_file(args, dock_fp, grid_name, all_csvs):
    # first try the original grid-index matching from roc_auc.py
    m = re.search(r"-DOCK_SP_(\d+)-001\.csv$", os.path.basename(dock_fp))
    if m:
        idx = m.group(1)
        sp_pattern = os.path.join(args.data_dir, f"*SP_OUT*_{idx}.csv")
        sp_files = glob.glob(sp_pattern)
        if sp_files:
            return sp_files[0]

    # fallback to the original grid-name matching from logauc_enrichment_score_v2.py
    for candidate in all_csvs:
        if candidate == dock_fp:
            continue
        try:
            hdr = pd.read_csv(candidate, nrows=0, dtype=str)
        except Exception:
            continue
        cols = set(hdr.columns)
        if not {"s_sm_ligand", "i_i_glide_lignum", "s_i_glide_gridfile"}.issubset(cols):
            continue
        try:
            tmp = pd.read_csv(candidate, usecols=["s_i_glide_gridfile"], nrows=1, dtype=str)
        except Exception:
            continue
        if tmp["s_i_glide_gridfile"].iat[0] == grid_name:
            return candidate

    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) load your full ligand library
    lib = pd.read_csv(args.ligand_file, dtype=str)
    if set(lib.columns) != {"LIGAND_ID", "ACTIVITY"}:
        raise ValueError("ligand_file must have exactly: LIGAND_ID,ACTIVITY")
    lib["ACTIVITY"] = lib["ACTIVITY"].astype(int)
    total = len(lib)

    # Determine lambda = 1 / (e * # decoys)
    n_decoy = lib["ACTIVITY"].value_counts().get(0, 0)
    if n_decoy == 0:
        raise ValueError("No decoys (ACTIVITY=0) found in ligand_file")
    lam = 1.0 / (math.e * n_decoy)

    # RawLogAUC_random = (1 - lam) / ( -ln(lam) )
    logauc_rand = (1.0 - lam) / (-math.log(lam))

    # 2) find all DOCK_SP files
    dock_paths = sorted(glob.glob(os.path.join(args.data_dir, "*-DOCK_SP_*001.csv")))
    if not dock_paths:
        raise FileNotFoundError(f"No DOCK_SP files found in {args.data_dir}")

    # pre-list all CSVs once for fallback SP_OUT matching
    all_csvs = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))

    summary = []
    for dock_fp in dock_paths:
        # 3) load DOCK_SP: i_i_glide_lignum, docking_score, gridfile
        dock_df = pd.read_csv(dock_fp, dtype=str)
        required_dock = {"i_i_glide_lignum", "r_i_docking_score", "s_i_glide_gridfile"}
        if not required_dock.issubset(dock_df.columns):
            print(f"⏭ Skipping {os.path.basename(dock_fp)} (missing {required_dock - set(dock_df.columns)})")
            continue

        grid_name = dock_df["s_i_glide_gridfile"].iat[0]
        safe_name = sanitize(grid_name)

        # 4) find & load SP_OUT partner
        sp_fp = find_sp_file(args, dock_fp, grid_name, all_csvs)
        if sp_fp is None:
            print(f"⚠️  No SP_OUT file found for grid '{grid_name}', skipping")
            continue

        sp_df = pd.read_csv(sp_fp, dtype=str)
        required_sp = {"s_sm_ligand", "i_i_glide_lignum"}
        if not required_sp.issubset(sp_df.columns):
            print(f"⏭ Skipping {os.path.basename(sp_fp)} (missing {required_sp - set(sp_df.columns)})")
            continue

        # 5) CORE MERGE STEP
        # map ligand ID -> internal index
        id_map = sp_df[["s_sm_ligand", "i_i_glide_lignum"]].drop_duplicates()
        merged = lib.merge(
            id_map,
            left_on="LIGAND_ID",
            right_on="s_sm_ligand",
            how="left"
        )
        n_id_mapped = merged["i_i_glide_lignum"].notnull().sum()

        # map index -> docking score from DOCK_SP
        dock_map = dock_df[["i_i_glide_lignum", "r_i_docking_score"]]
        merged = merged.merge(
            dock_map,
            on="i_i_glide_lignum",
            how="left"
        )
        n_score = merged["r_i_docking_score"].notnull().sum()
        print(f"[{grid_name}]: IDs→index: {n_id_mapped}/{total}, scores: {n_score}/{total}")

        # 6) fill missing scores & convert type
        merged["r_i_docking_score"] = (
            merged["r_i_docking_score"]
                  .astype(float)
                  .fillna(args.missing_score)
        )

        # 7) sort by descending docking performance (more negative = better)
        merged["y_score"] = -merged["r_i_docking_score"]
        merged.sort_values("y_score", ascending=False, inplace=True)

        # 8) compute ROC, AUC
        y_true = merged["ACTIVITY"].astype(int).values
        y_score = merged["y_score"].astype(float).values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # 9) compute EF1% (top 1% of ranked list)
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        A = int(y_true.sum())
        N = int(y_true.shape[0])
        top_frac = 0.01

        k = max(1, int(math.ceil(top_frac * N)))
        order = np.argsort(y_score)[::-1]
        a_top = int(y_true[order][:k].sum())

        ef1 = (a_top / (top_frac * A)) if A > 0 else float("nan")
        hitrate1 = a_top / k

        # 10) compute raw LogAUC & Enrichment Score
        raw_lauc = compute_logauc(fpr, tpr, lam)
        enr_score = (raw_lauc - logauc_rand) / (1.0 - logauc_rand)

        # 11) plot ROC curve with wrapped legend and all metrics
        plt.figure()
        legend_label = (
            f"{wrap(grid_name)}\n"
            f"AUC={roc_auc:.3f}\n"
            f"LogAUC={raw_lauc * 100:.1f}\n"
            f"EnrScore={enr_score * 100:.1f}\n"
            f"EF1%={ef1:.2f}"
        )
        plt.plot(fpr, tpr, color="green", lw=2, label=legend_label)
        plt.plot([0, 1], [0, 1], ls="--", lw=2, color="red", label="random (AUC=0.5)")
        plt.title("ROC Curve", fontsize=14)
        plt.xlabel("FPR", fontsize=12)
        plt.ylabel("TPR", fontsize=12)
        plt.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        out_png = os.path.join(args.output_dir, f"ROC_{safe_name}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()

        summary.append({
            "Grid":      grid_name,
            "Filename":  os.path.basename(out_png),
            "ID_mapped": n_id_mapped,
            "Scored":    n_score,
            "AUC":       roc_auc,
            "RawLogAUC": raw_lauc * 100,
            "EnrScore":  enr_score * 100,
            "EF1%":      ef1,
            "Top1%N":    k,
            "Top1%Act":  a_top,
            "HitRate1%": hitrate1
        })

    # 12) write summary
    if not summary:
        print("⚠️  No grids processed—check your CSVs and column names.")
        return

    pd.DataFrame(summary) \
      .sort_values("EnrScore", ascending=False) \
      .to_csv(os.path.join(args.output_dir, "enrichment_summary.csv"), index=False)

    print(f"\n✅ All done!  See '{args.output_dir}/' for your plots & summary.")


if __name__ == "__main__":
    main()
