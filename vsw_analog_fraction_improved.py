#!/usr/bin/env python3
"""

Inputs
------
--inputs-csv : CSV with headers including:
  - name       : seed's parent title in Glide CSV (exact match for parent row)
  - ligand_id  : seed's L#### identifier (e.g., L0886); present in analog titles
--vsw-dir     : directory with many Glide subjob CSVs (scored + *_skip.csv)
--subjobs-glob: pattern like "**/*vsw-DOCK*.csv"
--score-col   : e.g., r_i_docking_score
--id-col      : e.g., title (auto-detects if omitted)
--tools       : comma-separated tool names, e.g. "spacelight,spacemacs,ftrees"

------------
- Reads seeds from inputs.csv and builds maps:
    name.lower() -> seed_index   AND   ligand_id (L####) -> seed_index
- Scans ALL subjob CSVs. Parent rows are:
    title == name   OR   title == ligand_id
  Parent score = most negative across all files; if only in *_skip.csv, score=0.
- Analog rows:
    title contains ligand_id AND we can infer the tool token:
      (a) substring match against tools, or
      (b) pattern L####_<tool>_#### (middle token), or
      (c) short aliases: sl→spacelight, sm→spacemacs, ft→ftrees
  For each (seed, tool, analog_title) keep the most negative score (dedup).
- For each seed/tool: fraction = (# analogs with analog_score < parent_score) / (total analogs)
  If total=0, fraction=0 (still plotted at the seed score).
- Outputs 3 scatter plots and a details CSV.

"""

import argparse, csv, glob, os, re
from collections import defaultdict

# candidate column names
ID_CANDIDATES    = ["title","Title","_Name","name","Name","r_i_ligand_name","s_title","r_title","ID","id"]
SCORE_CANDIDATES = ["r_i_docking_score","r_i_glide_gscore","dock_score","docking_score","score"]

TOOL_ALIASES = {
    "sl": "spacelight",
    "sm": "spacemacs",
    "ft": "ftrees",
}

def autodetect_cols(header, score_hint, id_hint):
    id_col = id_hint if (id_hint and id_hint in header) else None
    if id_col is None:
        for c in ID_CANDIDATES:
            if c in header:
                id_col = c; break
    score_col = score_hint if (score_hint and score_hint in header) else None
    if score_col is None:
        for c in SCORE_CANDIDATES:
            if c in header:
                score_col = c; break
    return id_col, score_col

def read_inputs_csv(path, skip_rows=0):
    """Return (seeds_order, name_to_idx, lid_to_idx) from a CSV with 'name' and 'ligand_id'."""
    seeds_order = []  # list of (idx, name, ligand_id)
    name_to_idx = {}
    lid_to_idx  = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        if skip_rows:
            rows = rows[skip_rows:]
        for i, row in enumerate(rows, start=1):
            nm  = (row.get("name") or "").strip()
            lid = (row.get("ligand_id") or "").strip()
            if not nm or not lid:
                continue
            seeds_order.append((i, nm, lid))
            name_to_idx[nm.lower()] = i
            lid_to_idx[lid] = i
    return seeds_order, name_to_idx, lid_to_idx

def iter_rows(path, id_col, score_col, default_score=None):
    """Yield (title, score, subjob). If score missing and default_score is not None, use it."""
    subjob = os.path.basename(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            title = (row.get(id_col) or "").strip()
            if not title:
                for alt in ID_CANDIDATES:
                    if alt in row and str(row[alt]).strip():
                        title = str(row[alt]).strip(); break
            if not title:
                continue
            if score_col and score_col in row and str(row[score_col]).strip() != "":
                try:
                    sc = float(str(row[score_col]).strip())
                except Exception:
                    continue
            else:
                if default_score is None:
                    continue
                sc = float(default_score)
            yield title, sc, subjob

def infer_tool_from_title(title_low, tools_low):
    """Try multiple strategies to infer tool from title."""
    # 1) direct substring
    for t in tools_low:
        if t in title_low:
            return t
    # 2) take the middle token if looks like L####_<tool>_####
    parts = re.split(r"[\s_\-\.]+", title_low)
    if len(parts) >= 3 and re.fullmatch(r"l\d{4}", parts[0]):
        mid = parts[1]
        if mid in tools_low:
            return mid
        # 3) aliases like 'sl','sm','ft'
        if mid in TOOL_ALIASES and TOOL_ALIASES[mid] in tools_low:
            return TOOL_ALIASES[mid]
    # 4) look for any token that maps via aliases
    for p in parts:
        if p in TOOL_ALIASES and TOOL_ALIASES[p] in tools_low:
            return TOOL_ALIASES[p]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsw-dir", required=True)
    ap.add_argument("--inputs-csv", required=True, help="CSV with headers: name, ligand_id")
    ap.add_argument("--inputs-skip-rows", type=int, default=0)
    ap.add_argument("--subjobs-glob", default="**/*vsw-DOCK*.csv")
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--tools", default="spacelight,spacemacs,ftrees")
    ap.add_argument("--details-csv", default="fraction_improved_details_dedup.csv")
    args = ap.parse_args()

    tools_low = [t.strip().lower() for t in args.tools.split(",") if t.strip()]
    if not tools_low:
        tools_low = ["spacelight","spacemacs","ftrees"]

    # 1) Seeds
    seeds_order, name_to_idx, lid_to_idx = read_inputs_csv(args.inputs_csv, skip_rows=args.inputs_skip_rows)
    n_seeds = len(seeds_order)
    if n_seeds == 0:
        raise SystemExit("No seeds parsed from inputs CSV (need 'name' and 'ligand_id').")

    # 2) CSV list
    csv_paths = sorted(glob.glob(os.path.join(args.vsw_dir, args.subjobs_glob), recursive=True))
    if not csv_paths:
        raise SystemExit(f"No subjob CSVs found in {args.vsw_dir} with pattern {args.subjobs_glob}")

    # 3) Data stores
    parent_best = {}                              # idx -> best parent score
    parent_subjobs = defaultdict(set)             # idx -> {subjobs}
    analog_best = defaultdict(dict)               # (idx, tool) -> { analog_title_lower : best_score }
    analog_subjobs = defaultdict(set)             # (idx, tool) -> {subjobs}

    # small audit to help debug analog matching
    audit_unknown_tool = 0
    audit_unknown_examples = []

    # 4) scan all CSVs
    for path in csv_paths:
        # detect columns
        with open(path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), None)
        if not header:
            continue
        id_col, score_col = autodetect_cols(header, args.score_col, args.id_col)

        is_skip = path.endswith("_skip.csv")
        default_score = 0.0 if is_skip else None

        for title, score, subjob in iter_rows(path, id_col, score_col, default_score=default_score):
            t_low = title.lower()

            # Parent detection: exact match by seed name, OR bare ligand_id
            idx = name_to_idx.get(t_low)
            if idx is None:
                # try exact bare ligand_id
                for _, nm, lid in seeds_order:
                    if t_low == lid.lower():
                        idx = lid_to_idx[lid]
                        break

            if idx is not None:
                # If title is exactly the seed name or bare ligand_id → parent row
                # (Only treat as parent when EXACT equality, not just containing)
                if t_low == seeds_order[idx-1][1].lower() or t_low == seeds_order[idx-1][2].lower():
                    prev = parent_best.get(idx)
                    if prev is None or score < prev:
                        parent_best[idx] = score
                    parent_subjobs[idx].add(subjob)
                    continue  # handled as parent

            # Analog detection: needs a ligand_id present anywhere
            # Find which ligand_id this title belongs to
            idx = None
            for _, nm, lid in seeds_order:
                if lid.lower() in t_low:
                    idx = lid_to_idx[lid]
                    break

            if idx is None:
                continue  # neither parent nor analog we can map

            # Infer tool
            tool = infer_tool_from_title(t_low, tools_low)
            if tool is None:
                audit_unknown_tool += 1
                if len(audit_unknown_examples) < 10:
                    audit_unknown_examples.append(title)
                continue  # skip analogs whose tool we cannot map

            key = (idx, tool)
            atitle = t_low
            # dedup by analog title: keep most negative
            store = analog_best[key]
            if atitle in store:
                if score < store[atitle]:
                    store[atitle] = score
            else:
                store[atitle] = score
            analog_subjobs[key].add(subjob)

    # Any seed with no parent seen → parent score = 0 (fully skipped parents)
    for idx, nm, lid in seeds_order:
        if idx not in parent_best:
            parent_best[idx] = 0.0

    # 5) Compute fractions; write details CSV
    details = []
    fractions_by_tool = {t: [] for t in tools_low}
    seed_scores_by_tool = {t: [] for t in tools_low}

    for tool in tools_low:
        for idx, nm, lid in seeds_order:
            psc = parent_best[idx]
            key = (idx, tool)
            analogs = analog_best.get(key, {})
            total = len(analogs)
            improved = sum(1 for sc in analogs.values() if sc < psc)
            worse_eq = total - improved
            frac = (improved / total) if total > 0 else 0.0

            fractions_by_tool[tool].append(frac)
            seed_scores_by_tool[tool].append(psc)

            details.append({
                "tool": tool,
                "seed_index": idx,
                "seed_name": nm,
                "ligand_id": lid,
                "parent_score": f"{psc:.3f}",
                "n_analogs": total,
                "n_improved": improved,
                "n_worse_or_equal": worse_eq,
                "subjobs_seen": ";".join(sorted(analog_subjobs.get(key, set()) | parent_subjobs.get(idx, set())))
            })

    # save details
    with open(args.details_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(details[0].keys()))
        w.writeheader()
        w.writerows(details)
    print(f"[ok] Wrote details CSV: {args.details_csv}")

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = os.path.basename(os.path.normpath(args.vsw_dir))
    for tool in tools_low:
        xs = seed_scores_by_tool[tool]
        ys = fractions_by_tool[tool]
        plt.figure(figsize=(14,3.6), dpi=150)
        plt.scatter(xs, ys, s=10)
        plt.xlabel("Seed dock score (kcal/mol)")
        plt.ylabel("Fraction of analogs improved")
        plt.title(f"Improved analog fraction vs seed score — {tool}")
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        out_png = f"{base}_improvement_fraction_{tool}.png"
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"[ok] Wrote {out_png}  (n={len(xs)} seeds)")

    # summaries
    print("\n[summary] Post-deduplication sanity check")
    print(f"[summary] total seeds (from inputs.csv): {len(seeds_order)}")
    for tool in tools_low:
        atot = sum(len(analog_best[k]) for k in analog_best if k[1]==tool)
        avg = atot/len(seeds_order) if seeds_order else 0.0
        print(f"[summary] {tool}: seeds_plotted={len(seeds_order)}, analogs_dedup_total={atot}, avg_per_seed={avg:.1f}")

    if audit_unknown_tool:
        print(f"[warn] {audit_unknown_tool} analog rows skipped due to unknown tool in title.")
        if audit_unknown_examples:
            print("[warn] examples:")
            for ex in audit_unknown_examples:
                print(f"       - {ex}")

if __name__ == "__main__":
    main()

