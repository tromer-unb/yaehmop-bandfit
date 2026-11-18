#!/usr/bin/env bash
# Fit every DFT example of validation/<code>/<example> and write the standard
# outputs to validation/<code>/<example>/fitting/<protocol>. Extra arguments are passed to
# bands_fit.py, e.g.:  ./run_validation.sh --optimizer cmaes --budget 600
# Work files go to $WORK (default: a temporary folder).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
WORK="${WORK:-$(mktemp -d)}"; mkdir -p "$WORK"
for dir in "$HERE"/*/*/; do
    dir="${dir%/}"; code="$(basename "$(dirname "$dir")")"; name="$(basename "$dir")"
    case "$code" in vasp|qe|siesta|aims) ;; *) continue ;; esac
    [ -e "$dir/SOC_ONLY" ] && continue    # spin-orbit references: soc_fit.py only
    echo "== $code/$name"
    python "$ROOT/USER_WORKFLOW/bands_fit.py" "$HERE/fit_settings.ini" \
        --dft-dir "$dir" --dft-code "$code" --name "$name" \
        --outdir "$WORK/$code/$name" --fitting-dir "$dir/fitting/${PROTOCOL:-custom}" "$@" \
        > "$WORK/$code-$name.log" 2>&1 \
      && grep -E "USER_FIT_DONE" "$WORK/$code-$name.log" \
      || { echo "   not fitted:"; grep -E "Error|error:" "$WORK/$code-$name.log" | tail -1; }
done
echo "logs and work files: $WORK"
