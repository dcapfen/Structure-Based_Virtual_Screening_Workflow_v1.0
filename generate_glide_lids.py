#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from schrodinger.application.glide import poseviewconvert
from schrodinger.application.livedesign import lid
from schrodinger.utils import qapplication  # manages global QApplication :contentReference[oaicite:2]{index=2}


def safe_name(s: str, maxlen: int = 160) -> str:
    s = (s or "").strip() or "ligand"
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:maxlen] or "ligand"


def main():
    ap = argparse.ArgumentParser(
        description="Export Ligand Interaction Diagram PNGs from a Glide poseviewer file (*_pv.mae/.maegz)."
    )
    ap.add_argument("pvfile", help="Input poseviewer file, e.g. *_pv.maegz")
    ap.add_argument("outdir", help="Output directory for PNGs")
    ap.add_argument("--top", type=int, default=250, help="Export top N ligands (in file order)")
    ap.add_argument("--radius", type=float, default=None,
                    help="Optional: keep only receptor residues within this Å of ligand (often speeds + stabilizes)")
    ap.add_argument("--offscreen", action="store_true",
                    help="Force -platform offscreen for Qt (Linux only).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure a Qt application exists (needed by sketcher/LID) :contentReference[oaicite:3]{index=3}
    qapplication.get_application(create=True, offscreen=args.offscreen)

    n = 0
    for i, complex_st in enumerate(
        poseviewconvert.get_pv_file_merged_structures(args.pvfile, radius=args.radius),
        start=1
    ):
        if n >= args.top:
            break

        comp = poseviewconvert.Complex(complex_st)
        title = comp.ligand.title or complex_st.title or f"pose_{i}"
        fname = outdir / f"{n+1:04d}_{safe_name(title)}.png"

        img = lid.generate_lid(comp.ligand, comp.receptor)
        img.save(str(fname))

        n += 1
        if n % 50 == 0:
            print(f"Wrote {n} diagrams...")

    print(f"Done. Wrote {n} PNGs to {outdir.resolve()}")


if __name__ == "__main__":
    main()

