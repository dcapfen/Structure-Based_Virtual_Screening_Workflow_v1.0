#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Run: python3 glide_score_histogram_fixed.py --vsw-dir /path/to/launch_directory --score-col r_i_docking_score --dedupe-by title --dedupe-policy best --bin-width 1.0 --counts-out dockscore_counts.csv --debug-report dockscore_debug.csv

SCORE_CANDIDATES = [
    "r_i_docking_score",
    "r_i_glide_gscore",
    "dock_score",
    "docking_score",
    "score",
]
SNIFF_MAX_ROWS = 5000
KEY_CANDIDATES = {
    "title": ["title"],
    "smiles": ["SMILES", "smiles"],
}


@dataclass
class LigandRecord:
    best_score: Optional[float] = None
    n_rows: int = 0
    n_scored_rows: int = 0
    n_skip_rows: int = 0


# --------------------------- Discovery ---------------------------
def _discover_csvs_single(root: str, pattern: Optional[str]) -> Tuple[List[str], List[str]]:
    """Return (scored_files, skip_files) for a single root (dir or file)."""

    def split_skip(paths: List[str]) -> Tuple[List[str], List[str]]:
        scored, skipped = [], []
        for p in paths:
            (skipped if p.endswith("_skip.csv") else scored).append(p)
        return sorted(scored), sorted(skipped)

    if os.path.isfile(root):
        return split_skip([root])
    if not os.path.isdir(root):
        return ([], [])
    if pattern:
        return split_skip(glob.glob(os.path.join(root, pattern), recursive=True))
    top = glob.glob(os.path.join(root, "*DOCK*.csv"))
    if top:
        return split_skip(top)
    return split_skip(glob.glob(os.path.join(root, "**/*DOCK*.csv"), recursive=True))


def _discover_csvs(roots: List[str], pattern: Optional[str]) -> Tuple[List[str], List[str]]:
    scored, skipped = set(), set()
    for root in roots:
        s, k = _discover_csvs_single(root, pattern)
        scored.update(s)
        skipped.update(k)
    return sorted(scored), sorted(skipped)


# --------------------------- Helpers ---------------------------
def _parse_float(val: str) -> Optional[float]:
    try:
        v = float(str(val).strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _normalize_key(val: str) -> Optional[str]:
    key = (val or "").strip()
    return key if key else None


def _sniff_score_column(csv_path: str, hint: Optional[str]) -> Optional[str]:
    """Pick a score column by: --score-col > known candidates > heuristic."""
    try:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return None
            columns = list(reader.fieldnames)
            if hint and hint in columns:
                return hint
            norm = {c.strip().lower(): c for c in columns}
            for want in [c.lower() for c in SCORE_CANDIDATES]:
                if want in norm:
                    return norm[want]

            num_counts = {c: 0 for c in columns}
            neg_counts = {c: 0 for c in columns}
            for i, row in enumerate(reader):
                if i >= SNIFF_MAX_ROWS:
                    break
                for c in columns:
                    v = _parse_float(row.get(c, ""))
                    if v is None:
                        continue
                    num_counts[c] += 1
                    if v < 0:
                        neg_counts[c] += 1

            best, best_neg_frac = None, 0.0
            for c in columns:
                n = num_counts[c]
                if n < 10:
                    continue
                frac = neg_counts[c] / float(n)
                if frac > 0.2 and frac > best_neg_frac:
                    best_neg_frac = frac
                    best = c
            return best
    except Exception as e:
        print(f"[warn] Could not inspect {csv_path}: {e}")
        return None


def _find_key_column(csv_path: str, dedupe_by: str) -> Optional[str]:
    targets = {name.strip().lower() for name in KEY_CANDIDATES[dedupe_by]}
    try:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return None
            for c in reader.fieldnames:
                if c.strip().lower() in targets:
                    return c
    except Exception as e:
        print(f"[warn] Could not read header in {csv_path}: {e}")
    return None


def _update_score(old: Optional[float], new: float, policy: str) -> float:
    if old is None:
        return new
    if policy == "best":
        return min(old, new)
    if policy == "worst":
        return max(old, new)
    return old


# --------------------------- Collect / dedupe ---------------------------
def _collect_ligands(
    scored_files: List[str],
    skip_files: List[str],
    score_hint: Optional[str],
    *,
    dedupe_by: str,
    policy: str,
    allow_missing_key: bool,
    stats: Optional[dict] = None,
) -> Tuple[Dict[str, LigandRecord], int, int]:
    """
    Build a single global ligand table across all scored and skip files.

    Returns:
      records: ligand_key -> LigandRecord
      dupes_deleted_scored: duplicate scored rows encountered and not retained as distinct ligands
      missing_key_rows: rows skipped because dedupe key was missing/blank (unless allow_missing_key=True)
    """
    records: Dict[str, LigandRecord] = {}
    dupes_deleted_scored = 0
    missing_key_rows = 0

    for csv_path in scored_files:
        file_key = os.path.basename(csv_path)
        if stats is not None and file_key not in stats:
            stats[file_key] = {
                "rows": 0,
                "scorable": 0,
                "kept": 0,
                "dupes_deleted": 0,
                "missing_key": 0,
                "skip_only_counted": 0,
                "reason": "ok",
            }

        col = _sniff_score_column(csv_path, score_hint)
        if not col:
            if stats is not None:
                stats[file_key]["reason"] = "no_score_col"
            print(f"[warn] No plausible score column in {file_key}; skipping file")
            continue

        key_header = _find_key_column(csv_path, dedupe_by)
        if not key_header and not allow_missing_key:
            if stats is not None:
                stats[file_key]["reason"] = "no_key_col"
            print(f"[warn] No '{dedupe_by}' column in {file_key}; skipping file")
            continue

        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row_idx, row in enumerate(reader, start=1):
                    if stats is not None:
                        stats[file_key]["rows"] += 1

                    score = _parse_float(row.get(col, ""))
                    if score is None:
                        continue
                    if stats is not None:
                        stats[file_key]["scorable"] += 1

                    key = _normalize_key(row.get(key_header, "") if key_header else "")
                    if key is None:
                        missing_key_rows += 1
                        if stats is not None:
                            stats[file_key]["missing_key"] += 1
                        if not allow_missing_key:
                            continue
                        key = f"__missing_key__{file_key}__row{row_idx}"

                    rec = records.get(key)
                    if rec is None:
                        rec = LigandRecord(best_score=score, n_rows=1, n_scored_rows=1, n_skip_rows=0)
                        records[key] = rec
                        if stats is not None:
                            stats[file_key]["kept"] += 1
                    else:
                        rec.n_rows += 1
                        rec.n_scored_rows += 1
                        dupes_deleted_scored += 1
                        if stats is not None:
                            stats[file_key]["dupes_deleted"] += 1
                        rec.best_score = _update_score(rec.best_score, score, policy)
        except Exception as e:
            if stats is not None:
                stats[file_key]["reason"] = f"error: {e}"[:160]
            print(f"[warn] Could not read {csv_path}: {e}")

    for csv_path in skip_files:
        file_key = os.path.basename(csv_path)
        if stats is not None and file_key not in stats:
            stats[file_key] = {
                "rows": 0,
                "scorable": 0,
                "kept": 0,
                "dupes_deleted": 0,
                "missing_key": 0,
                "skip_only_counted": 0,
                "reason": "ok",
            }

        key_header = _find_key_column(csv_path, dedupe_by)
        if not key_header and not allow_missing_key:
            if stats is not None:
                stats[file_key]["reason"] = "no_key_col"
            print(f"[warn] No '{dedupe_by}' column in {file_key}; skipping *_skip.csv file")
            continue

        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row_idx, row in enumerate(reader, start=1):
                    if stats is not None:
                        stats[file_key]["rows"] += 1

                    key = _normalize_key(row.get(key_header, "") if key_header else "")
                    if key is None:
                        missing_key_rows += 1
                        if stats is not None:
                            stats[file_key]["missing_key"] += 1
                        if not allow_missing_key:
                            continue
                        key = f"__missing_key__{file_key}__row{row_idx}"

                    rec = records.get(key)
                    if rec is None:
                        records[key] = LigandRecord(best_score=None, n_rows=1, n_scored_rows=0, n_skip_rows=1)
                        if stats is not None:
                            stats[file_key]["skip_only_counted"] += 1
                    else:
                        rec.n_rows += 1
                        rec.n_skip_rows += 1
                        if stats is not None and rec.best_score is None:
                            stats[file_key]["dupes_deleted"] += 1
        except Exception as e:
            if stats is not None:
                stats[file_key]["reason"] = f"error: {e}"[:160]
            print(f"[warn] Could not read {csv_path}: {e}")

    return records, dupes_deleted_scored, missing_key_rows


# --------------------------- Histogram ---------------------------
def build_histogram(scores: np.ndarray, width: float):
    if width <= 0:
        raise ValueError("Bin width must be > 0")
    if scores.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), 0

    non_positive = scores[scores <= 0.0]
    positive_n = int((scores > 0.0).sum())

    if non_positive.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), positive_n

    start = width * math.floor(float(np.nanmin(non_positive)) / width)
    edges = np.arange(start, 0.0 + width, width)
    if edges.size < 2:
        edges = np.array([-width, 0.0])

    counts, edges = np.histogram(non_positive, bins=edges)
    return counts.astype(int), edges, positive_n


# --------------------------- Output helpers ---------------------------
def _format_bin_labels(edges: np.ndarray, counts: np.ndarray, bin_width: float, positive_n: int, failed_n: int):
    labels: List[str] = []
    heights: List[int] = []
    positions: List[float] = []

    if edges.size >= 2:
        lefts, rights = edges[:-1], edges[1:]
        centers = 0.5 * (lefts + rights)
        for left, right, center, count in zip(lefts, rights, centers, counts):
            labels.append(f"{left:.1f} to {right:.1f}")
            heights.append(int(count))
            positions.append(float(center))

    last_edge = float(edges[-1]) if edges.size else 0.0
    positive_x = last_edge + bin_width * 0.85
    failed_x = positive_x + bin_width * 1.15

    labels.append("> 0")
    heights.append(int(positive_n))
    positions.append(float(positive_x))

    labels.append("Failed")
    heights.append(int(failed_n))
    positions.append(float(failed_x))

    return labels, heights, positions


def _plot_histogram(
    positions: List[float],
    heights: List[int],
    labels: List[str],
    bin_width: float,
    title: str,
    out_png: str,
    dpi: int,
):
    plt.figure(figsize=(11, 6))
    bar_width = bin_width * 0.9 if len(heights) > 1 else 0.8
    plt.bar(positions, heights, width=bar_width, align="center", edgecolor="black")

    ax = plt.gca()
    ymax = max(heights) if heights else 0
    if ymax > 0:
        exp = int(np.floor(np.log10(ymax)))
        step = 10 ** exp
        n = int(np.ceil(ymax / step))
        if n < 4 and exp > 0:
            exp -= 1
            step = 10 ** exp
            n = int(np.ceil(ymax / step))
        elif n > 12:
            exp += 1
            step = 10 ** exp
            n = int(np.ceil(ymax / step))
        ax.set_yticks((np.arange(1, n + 1) * step).tolist())
        from matplotlib.ticker import FuncFormatter

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, pos: "0" if v <= 0 else f"{int(round(v / (10 ** exp)))}e{exp}")
        )

    plt.xlabel("Dock score (kcal/mol)")
    plt.ylabel("Ligand count")
    plt.title(title)
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.tight_layout()

    bump = max(1, 0.02 * ymax)
    for x, h in zip(positions, heights):
        plt.text(x, h + bump, str(h), ha="center", va="bottom", fontsize=9)

    plt.savefig(out_png, dpi=dpi)
    plt.close()


# --------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Glide dock score histogram with global ligand de-duplication across all subjob CSVs. "
            "Default behavior keeps the most negative score per unique title and places positive scores "
            "and failed ligands into separate bins."
        )
    )
    ap.add_argument("--vsw-dir", dest="vsw_dirs", action="append", required=True, help="Launch dir or CSV path; repeatable")
    ap.add_argument("--subjobs-glob", default=None, help="Glob relative to each dir, e.g. '**/*DOCK*.csv'")
    ap.add_argument("--score-col", default=None, help="Force score column, e.g. r_i_docking_score")
    ap.add_argument("--bin-width", type=float, default=1.0, help="Histogram bin width in kcal/mol")
    ap.add_argument("--out", default=None, help="Output PNG path")
    ap.add_argument("--counts-out", default=None, help="Optional CSV for per-bin counts plus summary totals")
    ap.add_argument("--title", default=None, help="Plot title")
    ap.add_argument("--dpi", type=int, default=160, help="PNG DPI")
    ap.add_argument("--no-dedupe", action="store_true", help="Disable de-duplication entirely")
    ap.add_argument("--dedupe-by", choices=["smiles", "title"], default="title", help="Key used for de-duplication")
    ap.add_argument(
        "--dedupe-policy",
        choices=["first", "best", "worst"],
        default="best",
        help="For duplicates, keep the first, best (most negative), or worst score",
    )
    ap.add_argument(
        "--allow-missing-key",
        action="store_true",
        help=(
            "Keep rows with missing dedupe key by treating each missing-key row as unique. "
            "Default is to skip them so total counts reflect valid ligand keys only."
        ),
    )
    ap.add_argument("--debug-report", default=None, help="Optional per-file debug CSV")
    args = ap.parse_args()

    scored_files, skip_files = _discover_csvs(args.vsw_dirs, args.subjobs_glob)
    if not scored_files and not skip_files:
        raise SystemExit("No subjob CSVs found")

    stats = {} if args.debug_report else None

    if args.no_dedupe:
        scores_all: List[float] = []
        failed_n = 0

        for csv_path in scored_files:
            file_key = os.path.basename(csv_path)
            col = _sniff_score_column(csv_path, args.score_col)
            if not col:
                print(f"[warn] No plausible score column in {file_key}; skipping file")
                continue
            try:
                with open(csv_path, newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        score = _parse_float(row.get(col, ""))
                        if score is not None:
                            scores_all.append(score)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")

        for csv_path in skip_files:
            try:
                with open(csv_path, newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)
                    failed_n += sum(1 for _ in reader)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")

        scores_np = np.array(scores_all, dtype=float) if scores_all else np.array([], dtype=float)
        counts, edges, positive_n = build_histogram(scores_np, args.bin_width)
        unique_kept = len(scores_all)
        missing_key_rows = 0
        dupes_deleted_scored = 0
        total_unique_ligands = len(scores_all) + failed_n
        title = args.title or "Dock Score Distribution (no dedupe)"
    else:
        records, dupes_deleted_scored, missing_key_rows = _collect_ligands(
            scored_files,
            skip_files,
            args.score_col,
            dedupe_by=args.dedupe_by,
            policy=args.dedupe_policy,
            allow_missing_key=args.allow_missing_key,
            stats=stats,
        )

        scores_np = np.array(
            [rec.best_score for rec in records.values() if rec.best_score is not None],
            dtype=float,
        )
        failed_n = sum(1 for rec in records.values() if rec.best_score is None)
        counts, edges, positive_n = build_histogram(scores_np, args.bin_width)
        unique_kept = len(records)
        total_unique_ligands = len(records)
        title = args.title or (
            f"Dock Score Distribution (dedup={args.dedupe_by}, policy={args.dedupe_policy}; "
            f"unique_ligands={unique_kept:,})"
        )

    labels, heights, positions = _format_bin_labels(edges, counts, args.bin_width, positive_n, failed_n)
    total_all_bins = int(sum(heights))

    base = os.path.basename(os.path.normpath(args.vsw_dirs[0]))
    out_png = args.out or f"{base}_dockscore_hist.png"
    _plot_histogram(positions, heights, labels, args.bin_width, title, out_png, args.dpi)
    print(f"[ok] Wrote histogram: {out_png}")

    if args.counts_out:
        with open(args.counts_out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["bin_label", "count"])
            for label, count in zip(labels, heights):
                writer.writerow([label, count])
            writer.writerow([])
            writer.writerow(["TOTAL_LIGANDS_ALL_BINS", total_all_bins])
            writer.writerow([f"UNIQUE_{args.dedupe_by.upper()}_KEPT" if not args.no_dedupe else "TOTAL_ROWS_KEPT", unique_kept])
            writer.writerow(["POSITIVE_SCORE_BIN_COUNT", positive_n])
            writer.writerow(["FAILED_LIGANDS_BIN_COUNT", failed_n])
            writer.writerow(["DUPLICATES_DELETED_SCORED", dupes_deleted_scored])
            writer.writerow(["MISSING_DEDUPE_KEY_ROWS_SKIPPED", missing_key_rows])
        print(f"[ok] Wrote counts CSV: {args.counts_out}")

    if args.debug_report and stats is not None:
        with open(args.debug_report, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "rows", "scorable", "kept", "dupes_deleted", "missing_key", "skip_only_counted", "reason"])
            for file_name, st in sorted(stats.items()):
                writer.writerow([
                    file_name,
                    st.get("rows", ""),
                    st.get("scorable", ""),
                    st.get("kept", ""),
                    st.get("dupes_deleted", ""),
                    st.get("missing_key", ""),
                    st.get("skip_only_counted", ""),
                    st.get("reason", "ok"),
                ])
        print(f"[ok] Wrote debug report: {args.debug_report}")

    print("[summary]")
    print(f"  scored_files_found: {len(scored_files)}")
    print(f"  skip_files_found: {len(skip_files)}")
    print(f"  positive_score_bin: {positive_n}")
    print(f"  failed_ligands_bin: {failed_n}")
    print(f"  sum_of_all_bins: {total_all_bins}")
    if args.no_dedupe:
        print(f"  total_rows_counted: {total_unique_ligands}")
    else:
        print(f"  unique_ligands_counted: {total_unique_ligands}")
        print(f"  duplicates_deleted_scored: {dupes_deleted_scored}")
        print(f"  missing_key_rows_skipped: {missing_key_rows}")

    if total_all_bins == total_unique_ligands:
        print("  sanity_check: PASS (sum of all bins matches total ligand count)")
    else:
        print("  sanity_check: FAIL (sum of all bins does not match total ligand count)")


if __name__ == "__main__":
    main()
