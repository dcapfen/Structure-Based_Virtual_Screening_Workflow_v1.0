#!/usr/bin/env bash
# Strict mode + error diagnostics
set -Eeuo pipefail
trap 'echo "ERROR @ line $LINENO: $BASH_COMMAND" >&2' ERR

# =============== Config ===============
TOPN=100                  # hits per engine, per input (use 500 if you prefer)
FT_SIM=0.00               # FTrees min similarity
SL_SIM=0.00               # SpaceLight min similarity
SM_SIM=0.00               # SpaceMACS min similarity (only for -t 3)
SL_FINGERPRINT="ecfp4"    # (optional second pass: "fCSFP4" with smaller TOPN)
THREADS=8                 # intra-process threads; runs stay sequential by ligand

# Paths to the binaries (adjust to your folders)
FT_BIN="../ftrees-7.0.0-Linux-x64/ftrees"
SL_BIN="../spacelight-2.0.0-Linux-x64/spacelight"
SM_BIN="../spacemacs-2.0.0-Linux-x64/spacemacs"

# Optional SpaceLight .tfsdb; leave empty to reuse the .space passed on CLI
SL_TARGET=""

# Uncomment for live shell trace if you want to see every command:
# set -x
# ======================================

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.smi> <Space.space|SpaceLight.tfsdb> [out_dir]" >&2
  echo "  input.smi: SMILES [optional_name] per line (tabs or spaces OK; CRLF OK)." >&2
  exit 1
fi

INPUT_SMI="$1"
SPACE="$2"
OUTDIR="${3:-out}"

mkdir -p "$OUTDIR"

# Prefer a local TMPDIR if available (helps when /home is NFS)
: "${TMPDIR:=$(mktemp -d)}"
if [[ ! -d "$TMPDIR" ]]; then TMPDIR="$(mktemp -d)"; fi
trap 'rm -rf "$TMPDIR"' EXIT

# Sanity checks
for b in "$FT_BIN" "$SL_BIN" "$SM_BIN"; do
  [[ -x "$b" ]] || { echo "ERROR: binary not executable: $b" >&2; exit 2; }
done
[[ -f "$INPUT_SMI" ]] || { echo "ERROR: input .smi not found: $INPUT_SMI" >&2; exit 2; }
[[ -f "$SPACE" ]] ||   { echo "ERROR: space/tfsdb not found: $SPACE" >&2; exit 2; }

# Select SpaceLight target
if [[ -n "$SL_TARGET" ]]; then
  SL_SPACE="$SL_TARGET"
else
  SL_SPACE="$SPACE"
fi

echo "Input file : $INPUT_SMI"
echo "FTrees bin : $FT_BIN"
echo "SpaceLight : $SL_BIN (target: $SL_SPACE ; FP: $SL_FINGERPRINT)"
echo "SpaceMACS  : $SM_BIN"
echo "Space file : $SPACE"
echo "Output dir : $OUTDIR"
echo "TOPN=$TOPN; FT_SIM=$FT_SIM; SL_SIM=$SL_SIM; SM_SIM=$SM_SIM; THREADS=$THREADS"
echo

# inputs.csv (catalog of seeds)
INPUTS_CSV="$OUTDIR/inputs.csv"
echo "ligand_id,smiles,name,source_id" > "$INPUTS_CSV"

# Helper to append a 'source_id' column
add_source_id () {
  local infile="$1" outfile="$2" lid="$3" tool="$4"
  awk -v id="$lid" -v tool="$tool" 'BEGIN{FS=OFS=","}
    NR==1 { $(NF+1)="source_id"; print; next }
    NR>1  { $(NF+1)=sprintf("%s_%s_%04d", id, tool, NR-1); print }
  ' "$infile" > "$outfile"
}

# ------------- Main loop (robust to CRLF/TAB & no newline at EOF) -------------
LIDX=0
line=""

# Ensure predictable IFS for read
IFS=

while IFS= read -r line || [[ -n "${line:-}" ]]; do
  # Trim trailing CR (handle CRLF)
  line="${line%$'\r'}"

  # Skip blank/comment
  [[ -z "${line//[[:space:]]/}" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  # Normalize tabs to single spaces so awk fields work
  line="${line//$'\t'/ }"

  : "$(( LIDX+=1 ))"
  LID=$(printf "L%04d" "$LIDX")

  # First field = SMILES; second (optional) = name/ID
  SMI="$(awk '{print $1}' <<<"$line")"
  NAME="$(awk 'NF>=2{print $2}' <<<"$line")"
  [[ -z "${SMI// }" ]] && { echo "Skipping malformed line $LIDX" >&2; continue; }
  [[ -z "${NAME// }" ]] && NAME="$LID"

  echo ">> [$LID] $NAME"

  # Record input ligand
  echo "$LID,$SMI,$NAME,${LID}_input_0000" >> "$INPUTS_CSV"

  # Per-ligand 1-line query file
  QFILE="$TMPDIR/${LID}.smi"
  printf "%s %s\n" "$SMI" "$NAME" > "$QFILE"

  # ---------- FTrees ----------
  FT_RAW="$TMPDIR/${LID}_ftrees.raw.csv"
  FT_OUT="$OUTDIR/${LID}_ftrees.csv"
  "$FT_BIN" -i "$QFILE" -s "$SPACE" \
    --max-nof-results "$TOPN" \
    --min-similarity-threshold "$FT_SIM" \
    --thread-count "$THREADS" \
    -O "$FT_RAW"
  add_source_id "$FT_RAW" "$FT_OUT" "$LID" "ftrees"

  # ---------- SpaceLight ----------
  SL_RAW="$TMPDIR/${LID}_spacelight.raw.csv"
  SL_OUT="$OUTDIR/${LID}_spacelight.csv"
  "$SL_BIN" -i "$QFILE" -s "$SL_SPACE" \
    -f "$SL_FINGERPRINT" \
    --max-nof-results "$TOPN" \
    --min-similarity-threshold "$SL_SIM" \
    --thread-count "$THREADS" \
    -O "$SL_RAW"
  add_source_id "$SL_RAW" "$SL_OUT" "$LID" "spacelight"

  # ---------- SpaceMACS (MCS similarity mode) ----------
  SM_RAW="$TMPDIR/${LID}_spacemacs.raw.csv"
  SM_OUT="$OUTDIR/${LID}_spacemacs.csv"
  "$SM_BIN" -i "$QFILE" -s "$SPACE" \
    -t 3 \
    --max-nof-results "$TOPN" \
    --min-similarity-threshold "$SM_SIM" \
    --thread-count "$THREADS" \
    -O "$SM_RAW"
  add_source_id "$SM_RAW" "$SM_OUT" "$LID" "spacemacs"

done < "$INPUT_SMI"

if [[ "$LIDX" -eq 0 ]]; then
  echo "No ligands were processed (input empty or mis-formatted)." >&2
fi

echo
echo "Done. Per-input CSVs are in: $OUTDIR"
echo "Files: L####_ftrees.csv, L####_spacelight.csv, L####_spacemacs.csv, plus inputs.csv"
echo "Each engine CSV has a uniform 'source_id' column (L####_<tool>_<rank>)."

