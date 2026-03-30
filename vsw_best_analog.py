#!/usr/bin/env python3
"""
Best analog per tool vs seed score (matches original analog mapping logic).

- Reads seeds from --inputs-csv with headers: name, ligand_id
- Scans Glide subjob CSVs under --vsw-dir (including *_skip.csv)
- Parent/seed score: best (most negative) score where title == seed name OR title == ligand_id
  If only present in *_skip.csv -> score=0.0 (same behavior as original script)
- Analog rows: title contains ligand_id (substring match, same as original)
- Tool inferred from title using the same logic as original:
    (a) substring match against full tool names
    (b) pattern L####_<tool>_#### middle token
    (c) aliases: sl->spacelight, sm->spacemacs, ft->ftrees
- For each (seed, tool), keep the single best (most negative) analog score and record its title.
  The analog ligand ID is the title string (per your note).

Outputs:
- Combined CSV: best_analogs_by_tool.csv (one row per seed per tool)
- One plot per tool: <vsw_dir_basename>_best_analog_vs_seed_<tool>.png
"""

import argparse, csv, glob, os, re
from collections import defaultdict

ID_CANDIDATES    = ["title","Title","_Name","name","Name","r_i_ligand_name","s_title","r_title","ID","id"]
SCORE_CANDIDATES = ["r_i_docking_score","r_i_glide_gscore","dock_score","docking_score","score"]

TOOL_ALIASES = {"sl": "spacelight", "sm": "spacemacs", "ft": "ftrees"}

def autodetect_cols(header, score_hint, id_hint):
    id_col = id_hint if (id_hint and id_hint in header) else None
    if id_col is None:
        for c in ID_CANDIDATES:
            if c in header:
                id_col = c
                break

    score_col = score_hint if (score_hint and score_hint in header) else None
    if score_col is None:
        for c in SCORE_CANDIDATES:
            if c in header:
                score_col = c
                break

    return id_col, score_col

def read_inputs_csv(path, skip_rows=0):
    seeds_order = []  # list of (idx, name, ligand_id)
    name_to_idx = {}
    lid_to_idx  = {}

    with open(path, newline="", encoding="utf-8-sig") as fh:
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
            lid_to_idx[lid] = i  # keep original-cased key like the original script

    return seeds_order, name_to_idx, lid_to_idx

def iter_rows(path, id_col, score_col, default_score=None):
    subjob = os.path.basename(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            title = (row.get(id_col) or "").strip() if id_col else ""
            if not title:
                for alt in ID_CANDIDATES:
                    if alt in row and str(row[alt]).strip():
                        title = str(row[alt]).strip()
                        break
            if not title:
                continue

            if score_col and score_col in row and str(row[score_col]).strip() != "":
                try:
                    score = float(str(row[score_col]).strip())
                except Exception:
                    continue
            else:
                if default_score is None:
                    continue
                score = float(default_score)

            yield title, score, subjob

def infer_tool_from_title(title_low, tools_low):
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
        if mid in TOOL_ALIASES and TOOL_ALIASES[mid] in tools_low:
            return TOOL_ALIASES[mid]

    # 3) look for any token that maps via aliases
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
    ap.add_argument("--tools", default="ftrees,spacelight,spacemacs")
    ap.add_argument("--out-csv", default="best_analogs_by_tool.csv")
    args = ap.parse_args()

    tools_low = [t.strip().lower() for t in args.tools.split(",") if t.strip()]
    if not tools_low:
        tools_low = ["ftrees", "spacelight", "spacemacs"]

    seeds_order, name_to_idx, lid_to_idx = read_inputs_csv(args.inputs_csv, skip_rows=args.inputs_skip_rows)
    if not seeds_order:
        raise SystemExit("No seeds parsed from inputs CSV (need 'name' and 'ligand_id').")

    idx_to_seed = {idx: (nm, lid) for idx, nm, lid in seeds_order}

    csv_paths = sorted(glob.glob(os.path.join(args.vsw_dir, args.subjobs_glob), recursive=True))
    if not csv_paths:
        raise SystemExit(f"No subjob CSVs found in {args.vsw_dir} with pattern {args.subjobs_glob}")

    parent_best = {}  # idx -> most negative parent score
    best_by_tool = {} # (idx, tool) -> dict(score, analog_title, subjob)

    audit_unknown_tool = 0
    audit_unknown_examples = []

    for path in csv_paths:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), None)
        if not header:
            continue

        id_col, score_col = autodetect_cols(header, args.score_col, args.id_col)

        is_skip = path.endswith("_skip.csv")
        default_score = 0.0 if is_skip else None

        for title, score, subjob in iter_rows(path, id_col, score_col, default_score=default_score):
            t_low = title.lower()

            # --- Parent detection (matches original script approach) ---
            idx = name_to_idx.get(t_low)
            if idx is None:
                # exact bare ligand_id
                for _, nm, lid in seeds_order:
                    if t_low == lid.lower():
                        idx = lid_to_idx[lid]
                        break

            if idx is not None:
                nm, lid = idx_to_seed[idx]
                if t_low == nm.lower() or t_low == lid.lower():
                    prev = parent_best.get(idx)
                    if prev is None or score < prev:
                        parent_best[idx] = score
                    continue  # handled as parent

            # --- Analog mapping (the key fix): substring lid match like original ---
            idx = None
            seed_lid = None
            for _, nm, lid in seeds_order:
                if lid.lower() in t_low:
                    idx = lid_to_idx[lid]
                    seed_lid = lid
                    break
            if idx is None:
                continue

            # --- Tool inference ---
            tool = infer_tool_from_title(t_low, tools_low)
            if tool is None:
                audit_unknown_tool += 1
                if len(audit_unknown_examples) < 10:
                    audit_unknown_examples.append(title)
                continue

            key = (idx, tool)
            prev = best_by_tool.get(key)
            if prev is None or score < prev["score"]:
                best_by_tool[key] = {
                    "score": score,
                    "analog_title": title,   # title IS the analog ligand id, per your note
                    "subjob": subjob,
                }

    # If a seed has no parent row anywhere, keep original behavior: parent score = 0.0
    for idx, nm, lid in seeds_order:
        if idx not in parent_best:
            parent_best[idx] = 0.0

    # --- Write combined CSV: one row per (seed, tool) ---
    out_rows = []
    missing = defaultdict(int)

    for tool in tools_low:
        for idx, nm, lid in seeds_order:
            seed_score = parent_best[idx]
            rec = best_by_tool.get((idx, tool))

            if rec is None:
                missing[tool] += 1
                out_rows.append({
                    "tool": tool,
                    "seed_index": idx,
                    "seed_name": nm,
                    "seed_ligand_id": lid,
                    "seed_score": f"{seed_score:.3f}",
                    "best_analog_score": "",
                    "delta_best_minus_seed": "",
                    "has_analog": 0,
                    "best_analog_ligand_id": "",
                    "best_analog_title": "",
                    "best_analog_subjob": "",
                })
            else:
                best_score = rec["score"]
                out_rows.append({
                    "tool": tool,
                    "seed_index": idx,
                    "seed_name": nm,
                    "seed_ligand_id": lid,
                    "seed_score": f"{seed_score:.3f}",
                    "best_analog_score": f"{best_score:.3f}",
                    "delta_best_minus_seed": f"{(best_score - seed_score):.3f}",  # negative => improved
                    "has_analog": 1,
                    "best_analog_ligand_id": rec["analog_title"],  # title is the ligand id
                    "best_analog_title": rec["analog_title"],
                    "best_analog_subjob": rec["subjob"],
                })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[ok] Wrote CSV: {args.out_csv}")

    # --- Plots: one per tool; only plot seeds that actually have an analog for that tool ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = os.path.basename(os.path.normpath(args.vsw_dir))

    for tool in tools_low:
        xs, ys = [], []
        for idx, nm, lid in seeds_order:
            rec = best_by_tool.get((idx, tool))
            if rec is None:
                continue
            xs.append(parent_best[idx])
            ys.append(rec["score"])

        out_png = f"{base}_best_analog_vs_seed_{tool}.png"
        plt.figure(figsize=(6.8, 5.6), dpi=150)
        if xs:
            plt.scatter(xs, ys, s=14)
            lo = min(xs + ys)
            hi = max(xs + ys)
            plt.plot([lo, hi], [lo, hi])  # y=x reference
        plt.xlabel("Seed dock score (kcal/mol)")
        plt.ylabel("Best analog dock score (kcal/mol, most negative)")
        plt.title(f"Best analog vs seed — {tool}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

        if missing[tool]:
            print(f"[warn] {tool}: {missing[tool]} seeds had no analogs for this tool (not plotted).")
        print(f"[ok] Wrote plot: {out_png}")

    if audit_unknown_tool:
        print(f"[warn] {audit_unknown_tool} analog rows skipped due to unknown tool in title.")
        if audit_unknown_examples:
            print("[warn] examples:")
            for ex in audit_unknown_examples:
                print(f"       - {ex}")

if __name__ == "__main__":
    main()
