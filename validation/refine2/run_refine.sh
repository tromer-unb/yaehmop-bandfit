#!/usr/bin/env bash
# Second round (validation/refine2/*.ini): weights concentrated on v1/c1; each starts from an earlier fit and
# writes validation/<code>/<example>/fitting/refine-bands2 (or $PROTOCOL). Work files in $WORK.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$(dirname "$HERE")")"
WORK="${WORK:-$(mktemp -d)}"; mkdir -p "$WORK"
declare -A DIR=([Al2O2_bilayer_test]=siesta [In2S2_monolayer_pbe]=vasp)
for ini in "$HERE"/*.ini; do
    name="$(basename "$ini" .ini)"; code="${DIR[$name]}"; ex="$ROOT/validation/$code/$name"
    echo "== $code/$name"
    python "$ROOT/USER_WORKFLOW/bands_fit.py" "$ini" --dft-dir "$ex" --dft-code "$code" --name "$name" \
        --outdir "$WORK/$code/$name" --fitting-dir "$ex/fitting/${PROTOCOL:-refine-bands2}" "$@" \
        > "$WORK/$code-$name.log" 2>&1 && grep USER_FIT_DONE "$WORK/$code-$name.log" \
        || { echo "   failed:"; grep -E "rror" "$WORK/$code-$name.log" | tail -2; }
done
echo "work files: $WORK"
