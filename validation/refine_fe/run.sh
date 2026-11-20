#!/usr/bin/env bash
# Fe bcc ferromagnetic (VASP PBE, ISPIN=2, metal): one EHT fit per spin channel (objective metal),
# then spin_combine.py joins them into huckel-SP.dat. Protocol name = INI name ($1, default Fe_metal).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$(dirname "$HERE")")"
EX="$ROOT/validation/vasp/Fe_bcc_pbe_spin"; P="${1:-Fe_metal}"; TAG="${2:-metal-cmaes}"
WORK="${WORK:-$(mktemp -d)}"; mkdir -p "$WORK"
# the two channels run at the same time (one process each)
for ch in up down; do (
    echo "== $ch $(date +%T)"
    python "$ROOT/USER_WORKFLOW/bands_fit.py" "$HERE/$P.ini" --dft-dir "$EX" --spin-channel $ch \
        --name Fe_bcc_$ch --outdir "$WORK/$TAG-$ch" --fitting-dir "$EX/fitting/$TAG/spin_$ch" \
        > "$WORK/$TAG-$ch.log" 2>&1 && grep USER_FIT_DONE "$WORK/$TAG-$ch.log" || { echo "   failed:"; tail -3 "$WORK/$TAG-$ch.log"; }
) & done; wait
python "$ROOT/USER_WORKFLOW/spin_combine.py" --up "$EX/fitting/$TAG/spin_up" --down "$EX/fitting/$TAG/spin_down" \
    --outdir "$EX/fitting/$TAG/spin_polarized"
echo "work files: $WORK"
