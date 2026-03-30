#!/usr/bin/env python3
"""
process_vsw_hits.py

Scan a **VSW (Virtual Screening Workflow)** job directory, aggregate subjob CSVs (e.g., *vDOCK_SP*.csv)
that include **SMILES**, **title**, and **r_i_docking_score**, then:
  • Rank by docking score (lower/more negative = better) → `orig_rank`
  • Keep only the top X percentile
  • Collapse alternate conformers by SMILES (keep best score per SMILES)
  • Apply PAINS and Lipinski Rule-of-5 filters
  • Cluster by Morgan fingerprints (Butina on Tanimoto distance) and keep the best per cluster

Output: a CSV sorted by best binders (most negative `dock_score` first) that includes `ID`, `SMILES`, `dock_score`, `orig_rank`, and `subjob`.
A companion **metrics text file** is also written next to the CSV with the same name plus `_data.txt`,
containing the step-by-step counts (initial size, after filters, clusters, etc.).

Live metrics are printed after each stage: initial size, after percentile, after conformer de-dup,
after PAINS, after Lipinski, and number of clusters.

─────────────────────────────────────────────────────────────────────────────────────────────
DEPENDENCIES & SETUP
─────────────────────────────────────────────────────────────────────────────────────────────
Conda (recommended for RDKit):
    conda create -n vls-pipeline python=3.10 numpy=1.24 pandas rdkit -c conda-forge
    conda activate vls-pipeline

Python venv (works on many systems; RDKit wheels may vary):
    python3 -m venv ~/venvs/vls-pipeline
    source ~/venvs/vls-pipeline/bin/activate
    pip install --upgrade pip
    pip install "numpy<2.0" pandas rdkit-pypi

Verify imports resolve **inside** the env (paths should point into your env, not Anaconda):
    python - << 'PY'
import sys, numpy, pandas
print("python:", sys.executable)
print("numpy:", numpy.__file__)
print("pandas:", pandas.__file__)
PY

Notes:
  • If you hit glibc / binary-ABI errors with RDKit wheels, prefer the conda-forge env.
  • Pandas does not require pyarrow; omitting it avoids extra ABI constraints.

─────────────────────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────────────────────
    # Minimal: point to the VSW directory; the script searches *DOCK_SP*.csv at the top level first.
    # If none are found, it automatically falls back to a recursive search (**/*DOCK_SP*.csv).
    # Default expected columns in subjobs: **SMILES**, **title**, **r_i_docking_score**.
    python process_vsw_hits.py \
      --vsw-dir /path/to/VSW_RUN_DIR \
      -p 20 -c 0.25 -o results_processed.csv

    # Provide a custom glob (relative to --vsw-dir). By default, *_skip.csv files are excluded.
    python process_vsw_hits.py \
      --vsw-dir /path/to/VSW_RUN_DIR \
      --subjobs-glob "**/*DOCK_SP*.csv" \
      --score-col r_i_docking_score --id-col title \
      -p 20 -c 0.25 -o results_processed.csv

    # Debug/audit mode: write per-file diagnostics and skip clustering for speed
    python process_vsw_hits.py \
      --vsw-dir /path/to/VSW_RUN_DIR \
      --audit audit_subjobs.csv --no-cluster --expect 36042 \
      -p 100 -o aggregated_only.csv

The input files **do not need to be pre-sorted**. `orig_rank` is computed from score values across
all subjobs after aggregation, independent of file order.

─────────────────────────────────────────────────────────────────────────────────────────────
ARGUMENTS
─────────────────────────────────────────────────────────────────────────────────────────────
  --vsw-dir              Path to the VSW job directory (the script searches for *DOCK_SP*.csv)
  --subjobs-glob         (Optional) Glob pattern relative to --vsw-dir. If omitted, search top level first
                         with "*DOCK_SP*.csv" (excluding *_skip.csv), then fall back to recursive "**/*DOCK_SP*.csv".
  --include-skip         Include files matching *_skip.csv (default: excluded)
  -p, --percentile       Top percentile of best (most negative) scores to keep (default: 100)
  -c, --cluster-cutoff   Tanimoto **distance** cutoff for Butina (default: 0.2 ⇒ ~0.8 similarity)
  -o, --output           Output CSV filename (default: <vsw_dir_basename>_processed.csv)
  --score-col            Docking score column name (default auto-detect; prefers r_i_docking_score)
  --smiles-col           SMILES column name (default auto-detect; prefers SMILES)
  --id-col               Title/ID column name (default: title)
  --strict-cols          Require presence of the specified/expected columns in each subjob; skip file otherwise
  --audit                Write a per-file audit CSV with column detection and row counts
  --no-cluster           Skip Butina clustering (useful for debugging and speed)
  --expect               Optional integer; warn if aggregated input row count differs from this

"""

import argparse
import glob
import os
import re
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.DataStructs import BulkTanimotoSimilarity, FingerprintSimilarity
from rdkit.ML.Cluster import Butina

# ------------------------------ Column detection ------------------------------
SMILES_CANDIDATES = [
    "SMILES", "smiles", "s_smiles", "s_m_smiles", "r_smiles"
]  # prefers SMILES
SCORE_CANDIDATES = [
    "r_i_docking_score", "r_i_glide_gscore", "dock_score", "docking_score", "score"
]  # prefers r_i_docking_score
ID_CANDIDATES = [
    "title", "Title", "_Name",  # common in subjobs
    "s_m_title",                  # seen in some exports
    "name", "Name", "ID", "id", "compound", "Compound",
    "ligand", "Ligand", "molecule", "Molecule",
    "entry", "Entry", "cmpd_id",
    "r_i_ligand_name", "s_title", "r_title"
]

SMILES_TOKEN = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#%\\/\.]+$")


# Extract a readable subjob tag from a filename (e.g., '1-016' or '016')
def _extract_subjob_tag(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split('-')
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return f"{parts[-2]}-{parts[-1]}"
    if parts and parts[-1].isdigit():
        return parts[-1]
    return name

def detect_smiles_col(df: pd.DataFrame, fallback: Optional[str] = None) -> Optional[str]:
    if fallback and fallback in df.columns:
        return fallback
    for c in SMILES_CANDIDATES:
        if c in df.columns:
            return c
    # heuristic scan
    best, best_ok = None, 0.0
    for c in df.select_dtypes(include="object").columns:
        s = df[c].dropna().astype(str).head(200)
        if s.empty:
            continue
        ok = s.map(lambda x: bool(SMILES_TOKEN.match(x))).mean()
        if ok > best_ok:
            best, best_ok = c, ok
    return best if best_ok >= 0.8 else None


def detect_score_col(df: pd.DataFrame, fallback: Optional[str] = None) -> Optional[str]:
    if fallback and fallback in df.columns:
        return fallback
    for c in SCORE_CANDIDATES:
        if c in df.columns:
            return c
    # last resort: any float-like column with negative values
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any() and (vals < 0).mean() > 0.2:
            return c
    return None


def detect_id_col(df: pd.DataFrame, fallback: Optional[str] = None) -> Optional[str]:
    if fallback and fallback in df.columns:
        return fallback
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    # heuristic: ID-like string column (high uniqueness, not SMILES-like)
    obj_cols = list(df.select_dtypes(include="object").columns)
    best, best_score = None, -1.0
    for c in obj_cols:
        s = df[c].astype(str).str.strip()
        if s.empty:
            continue
        uniq_ratio = s.nunique(dropna=True) / max(len(s), 1)
        not_smiles = (~s.str.match(SMILES_TOKEN)).mean()
        score = 0.7 * uniq_ratio + 0.3 * not_smiles
        if score > best_score:
            best, best_score = c, score
    return best

# ------------------------------ Aggregation from subjobs ------------------------------

def collect_subjob_files(vsw_dir: str, pattern: Optional[str], include_skip: bool) -> List[str]:
    """Return a sorted list of subjob CSVs.
    If no pattern is provided, search top-level "*DOCK_SP*.csv" (excluding *_skip.csv),
    else fall back to recursive "**/*DOCK_SP*.csv" (still excluding *_skip.csv). If pattern is provided,
    use it as-is (relative to vsw_dir) with recursive globbing. Optionally include *_skip.csv.
    """
    def _filter(files: List[str]) -> List[str]:
        if include_skip:
            return files
        return [f for f in files if not f.endswith("_skip.csv")]

    if pattern is None:
        top = glob.glob(os.path.join(vsw_dir, "*DOCK_SP*.csv"))
        top = _filter(top)
        if top:
            return sorted(top)
        rec = glob.glob(os.path.join(vsw_dir, "**/*DOCK_SP*.csv"), recursive=True)
        return sorted(_filter(rec))
    # user provided a pattern; respect it
    files = glob.glob(os.path.join(vsw_dir, pattern), recursive=True)
    return sorted(_filter(files))


def read_minimal_from_subjob(csv_path: str, id_hint: Optional[str], smiles_hint: Optional[str], score_hint: Optional[str], strict: bool) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"[warn] failed to read {csv_path}: {e}")
        return None

    smi_col = detect_smiles_col(df, smiles_hint)
    sc_col = detect_score_col(df, score_hint)
    id_col = detect_id_col(df, id_hint)

    # strict mode: all three must exist
    if strict and (not smi_col or not sc_col or not id_col):
        print(f"[warn] {csv_path}: missing required columns (ID={id_col}, SMILES={smi_col}, SCORE={sc_col}), skipping")
        return None

    # Non-strict: we must have at least SMILES + SCORE. If ID missing, synthesize one from filename + row index.
    if not smi_col or not sc_col:
        print(f"[warn] {csv_path}: could not find SMILES/score columns, skipping")
        return None

    base = os.path.splitext(os.path.basename(csv_path))[0]
    tag = _extract_subjob_tag(csv_path)

    if id_col:
        sub = df[[id_col, smi_col, sc_col]].copy()
        sub.columns = ["ID", "SMILES", "dock_score"]
    else:
        sub = df[[smi_col, sc_col]].copy()
        sub.columns = ["SMILES", "dock_score"]
        sub["ID"] = [f"{base}:{i}" for i in range(len(sub))]

    sub["subjob"] = tag
    sub["dock_score"] = pd.to_numeric(sub["dock_score"], errors="coerce")
    sub = sub.dropna(subset=["SMILES", "dock_score"])  # ensure usable rows
    return sub


def aggregate_from_subjobs(files: List[str], id_hint: Optional[str], smiles_hint: Optional[str], score_hint: Optional[str], strict: bool, audit_path: Optional[str]) -> pd.DataFrame:
    frames = []
    audit_rows = []
    for f in files:
        df0 = None
        try:
            df0 = pd.read_csv(f, nrows=5)
        except Exception:
            pass
        smi0 = detect_smiles_col(df0, smiles_hint) if df0 is not None else None
        sc0 = detect_score_col(df0, score_hint) if df0 is not None else None
        id0 = detect_id_col(df0, id_hint) if df0 is not None else None

        sub = read_minimal_from_subjob(f, id_hint, smiles_hint, score_hint, strict)
        used = 0 if sub is None else len(sub)
        frames.append(sub) if sub is not None and used else None
        # full counts
        try:
            full = pd.read_csv(f, low_memory=False)
            n_full = len(full)
            n_nonnull = len(full.dropna(subset=[smi0, sc0])) if smi0 and sc0 and smi0 in full.columns and sc0 in full.columns else None
        except Exception:
            n_full, n_nonnull = None, None
        audit_rows.append({
            "file": f,
            "id_detected": id0,
            "smiles_detected": smi0,
            "score_detected": sc0,
            "rows_total": n_full,
            "rows_smi+score_nonnull": n_nonnull,
            "rows_used": used,
        })

    if audit_path:
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        print(f"[audit] Wrote per-file diagnostics to: {audit_path}")

    frames = [fr for fr in frames if fr is not None and len(fr)]
    if not frames:
        sys.exit("No usable subjob CSVs were found. Check --vsw-dir/--subjobs-glob and column names.")

    df = pd.concat(frames, ignore_index=True)
    # Do NOT drop duplicates globally here; keep all rows and let later steps deduplicate by SMILES if desired
    return df

# ------------------------------ Filtering helper functions ------------------------------

def filter_percentile(df: pd.DataFrame, percentile: float) -> pd.DataFrame:
    thr = np.percentile(df["dock_score"].values, percentile)
    return df[df["dock_score"] <= thr]


def remove_conformers(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("SMILES")["dock_score"].idxmin()
    return df.loc[idx]


def attach_mols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Mol"] = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
    df = df[df["Mol"].notna()].copy()
    return df


def apply_pains_filter(df: pd.DataFrame) -> pd.DataFrame:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    keep = [i for i, m in enumerate(df["Mol"]) if m and not catalog.HasMatch(m)]
    return df.iloc[keep].copy()


def apply_lipinski_filter(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for i, m in enumerate(df["Mol"]):
        mw = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        hbd = Descriptors.NumHDonors(m)
        hba = Descriptors.NumHAcceptors(m)
        if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
            keep.append(i)
    return df.iloc[keep].copy()

# ------------------------------ Main pipeline ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Process a VSW run directory of Glide subjob CSVs into ranked, filtered, clustered hits")
    ap.add_argument("--vsw-dir", required=True, help="Path to VSW directory containing subjob CSVs")
    ap.add_argument("--subjobs-glob", default=None, help="Glob relative to --vsw-dir (default shallow then recursive)")
    ap.add_argument("--include-skip", action="store_true", help="Include files matching *_skip.csv (default: excluded)")
    ap.add_argument("-p", "--percentile", type=float, default=100.0, help="Top percentile to retain (lower is better)")
    ap.add_argument("-c", "--cluster-cutoff", type=float, default=0.2, help="Tanimoto distance cutoff for Butina")
    ap.add_argument("-o", "--output", default=None, help="Output CSV filename")
    ap.add_argument("--score-col", default=None, help="Docking score column name (auto-detect per file if omitted)")
    ap.add_argument("--smiles-col", default=None, help="SMILES column name (auto-detect per file if omitted)")
    ap.add_argument("--id-col", default="title", help="Ligand title/ID column name (default: title)")
    ap.add_argument("--strict-cols", action="store_true", help="Require ID/SMILES/score to be present; skip file otherwise")
    ap.add_argument("--audit", default=None, help="Write per-file audit CSV with column detection and row counts")
    ap.add_argument("--no-cluster", action="store_true", help="Skip clustering (debug/speed)")
    ap.add_argument("--expect", type=int, default=None, help="Warn if aggregated input count differs from this")
    args = ap.parse_args()

    # metrics collector
    metrics: list[str] = []
    def metric(msg: str):
        print(msg)
        metrics.append(str(msg))

    files = collect_subjob_files(args.vsw_dir, args.subjobs_glob, args.include_skip)
    if not files:
        sys.exit(f"No subjob CSVs found in {args.vsw_dir} with pattern {args.subjobs_glob or '*DOCK_SP*.csv'}")

    excluded_note = " (including *_skip.csv)" if args.include_skip else " (excluding *_skip.csv)"
    metric(f"Found {len(files)} subjob CSVs{excluded_note}. Aggregating minimal columns…")

    df = aggregate_from_subjobs(files, args.id_col, args.smiles_col, args.score_col, args.strict_cols, args.audit)

    if args.expect is not None and len(df) != args.expect:
        print(f"[warn] Aggregated {len(df)} rows; differs from --expect {args.expect}")

    # Original rank by dock_score across aggregated set
    df = df.sort_values("dock_score", ascending=True, kind="mergesort").reset_index(drop=True)
    df["orig_rank"] = np.arange(1, len(df) + 1, dtype=int)
    metric(f"Initial library size (aggregated): {len(df)}")

    # Top percentile filter
    df = filter_percentile(df, args.percentile)
    metric(f"After top {args.percentile}% filter: {len(df)}")

    # Conformer de-dup (by SMILES)
    df = remove_conformers(df)
    metric(f"After conformer deduplication: {len(df)}")

    # Build RDKit Mol once
    df = attach_mols(df)

    # PAINS
    df = apply_pains_filter(df)
    metric(f"After PAINS filter: {len(df)}")

    # Lipinski
    df = apply_lipinski_filter(df)
    metric(f"After Lipinski filter: {len(df)}")

    # Clustering (O(N^2))
    if not args.no_cluster:
        n = len(df)
        if n == 0:
            sys.exit("No molecules remain after filtering.")
        gen = GetMorganGenerator(radius=2, fpSize=2048)
        fps = [gen.GetFingerprint(m) for m in df["Mol"]]

        # Build condensed distance list in the order expected by Butina
        expected = n * (n - 1) // 2
        dists: list[float] = []
        for i in range(1, n):
            sims = BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1.0 - float(x) for x in sims])

        # Safety fallback: if anything went off (rare), rebuild pairwise distances
        if len(dists) != expected:
            metric(f"[warn] distance list length {len(dists)} != expected {expected}; rebuilding pairwise distances (fallback)")
            dists = []
            for i in range(1, n):
                for j in range(i):
                    dists.append(1.0 - float(FingerprintSimilarity(fps[i], fps[j])))

        clusters = Butina.ClusterData(dists, n, args.cluster_cutoff, isDistData=True)
        metric(f"Number of clusters formed: {len(clusters)}")

        # Pick best-scoring representative of each cluster
        idxs = [min(c, key=lambda j: df.iloc[j]["dock_score"]) for c in clusters]
        df = df.iloc[idxs].copy()
    else:
        metric("[info] Skipping clustering (--no-cluster)")

    # Clean columns (drop helper Mol)
    df.drop(columns=["Mol"], inplace=True, errors="ignore")

    df.drop(columns=["Mol"], inplace=True, errors="ignore")

    # Final sort: best binders first (most negative score)
    df = df.sort_values("dock_score", ascending=True, kind="mergesort").reset_index(drop=True)

    # Write output
    out_path = args.output or f"{os.path.basename(os.path.normpath(args.vsw_dir))}_processed.csv"
    df.to_csv(out_path, index=False)

    # Write metrics sidecar file
    data_path = out_path[:-4] + "_data.txt" if out_path.lower().endswith(".csv") else out_path + "_data.txt"
    with open(data_path, "w") as fh:
        fh.write("\n".join(metrics) + "\n")

    metric(f"Processed hits written to: {out_path}")
    metric(f"Metrics written to: {data_path}")
    # End of metrics sidecar write


if __name__ == "__main__":
    # Silence harmless RDKit warnings if desired
    import warnings as _warnings
    _warnings.filterwarnings("ignore", message="to-Python converter for boost::shared_ptr")
    try:
        main()
    except Exception as e:
        # Ensure a clean message if something unexpected blows up late
        print(f"[error] {e}")
        raise

