#!/usr/bin/env bash
# Al2O3 (SIESTA bilayer): fit only v1 + c1..c3 (validation/refine_v1c3/*.ini).
# Each protocol writes validation/siesta/Al2O2_bilayer_test/fitting/v1c3-<protocol>.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$(dirname "$HERE")")"
EX="$ROOT/validation/siesta/Al2O2_bilayer_test"
WORK="${WORK:-$(mktemp -d)}"; mkdir -p "$WORK"
for p in "${@:-cmaes cmaes-start cmaes-basis ga-basis}"; do for q in $p; do
    echo "== $q $(date +%T)"
    python "$ROOT/USER_WORKFLOW/bands_fit.py" "$HERE/$q.ini" --dft-dir "$EX" --dft-code siesta \
        --name Al2O3_bilayer --outdir "$WORK/$q" --fitting-dir "$EX/fitting/v1c3-$q" \
        > "$WORK/$q.log" 2>&1 && grep USER_FIT_DONE "$WORK/$q.log" || { echo "   failed:"; tail -3 "$WORK/$q.log"; }
done; done
echo "work files: $WORK"
