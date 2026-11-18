#!/usr/bin/env python3
"""Compare fitting strategies on every DFT example of validation/.

Each strategy is a list of bands_fit.py options. All runs are scored with the
same measures, whatever objective was optimized (band_objective.py, VBM-aligned,
Fermi level in the middle of the gap): RMS of the states within 2 eV of E_F,
RMS weighted by exp(-|E-E_F|/1 eV), gap errors, errors at the local extrema and
high-symmetry points (energy-weighted), positions of the band edges.

    python validation/compare_strategies.py --work /tmp/strategies
    python validation/compare_strategies.py --only vasp/TiBr2_monolayer_pbe --strategies window-ga

Writes WORK/strategies.csv and WORK/strategies.md (one row per example and
strategy). With --collect, the fitting/ folder of every finished run is copied to
validation/<code>/<example>/fitting/<strategy>/ (paths in fit_summary.json made
relative to the new place); --collect-only does just that for an existing WORK.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STRATEGIES = {
    "frontier-de": ["--objective", "frontier", "--optimizer", "nelder-mead"],
    "window-de": ["--objective", "window", "--optimizer", "nelder-mead"],
    "window-ga": ["--objective", "window", "--optimizer", "genetic", "--polish", "nelder-mead"],
    "window-cmaes": ["--objective", "window", "--optimizer", "cmaes", "--polish", "nelder-mead"],
    "window-ga-basis": ["--objective", "window", "--optimizer", "genetic", "--polish", "nelder-mead",
                        "--basis-add", "auto"],
}
KEYS = ("rms_within_2eV_of_EF_eV", "fermi_weighted_rms_eV", "gap_error_eV", "direct_gap_error_eV",
        "rms_extrema_eV", "rms_high_symmetry_eV", "vbm_position_error", "cbm_position_error", "window_rms_eV")


def examples(only):
    for d in sorted(HERE.glob("*/*/")):
        code = d.parent.name
        if code not in ("vasp", "qe", "siesta", "aims", "wannier90"):
            continue
        rel = f"{code}/{d.name}"
        if only and rel not in only:
            continue
        if (d / "SOC_ONLY").exists():      # spin-orbit references: soc_fit.py only
            continue
        yield code, d, rel


def collect(work):
    """Copy WORK/<code>/<example>/<strategy>/fitting -> validation/.../fitting/<strategy>."""
    import os, shutil
    sys.path.insert(0, str(ROOT / "USER_WORKFLOW"))
    from bands_fit import portable_paths
    n = 0
    for src in sorted(Path(work).glob("*/*/*/fitting")):
        strategy, example, code = src.parent.name, src.parent.parent.name, src.parent.parent.parent.name
        if not (src / "fit_summary.json").is_file() or not (HERE / code / example).is_dir():
            continue
        dst = HERE / code / example / "fitting" / strategy
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        def absolute(o):
            if isinstance(o, dict): return {k: absolute(v) for k, v in o.items()}
            if isinstance(o, list): return [absolute(v) for v in o]
            if isinstance(o, str) and o.startswith(".."):
                return str((src / o).resolve())
            return o
        s = json.loads((dst / "fit_summary.json").read_text())
        s = portable_paths(absolute(s), dst)
        s["protocol"] = strategy
        (dst / "fit_summary.json").write_text(json.dumps(s, indent=2) + "\n")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--only", nargs="+", help="examples as code/name")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--budget-large", type=int, default=1000,
                    help="budget for systems with more than 100 EHT orbitals")
    ap.add_argument("--settings", type=Path, default=HERE / "fit_settings.ini")
    ap.add_argument("--collect", action="store_true",
                    help="copy each finished fitting/ to validation/<code>/<example>/fitting/<strategy>")
    ap.add_argument("--collect-only", action="store_true", help="only copy the finished runs of WORK")
    a = ap.parse_args()
    if a.collect_only:
        print("COLLECTED", collect(a.work))
        return
    a.work.mkdir(parents=True, exist_ok=True)
    rows = []
    for code, d, rel in examples(a.only):
        for name in a.strategies:
            out = a.work / rel / name
            budget = a.budget
            info = d / "fitting" / "fit_summary.json"
            if info.is_file():
                n = json.loads(info.read_text()).get("tight_binding_files", {}).get("n_orbitals", 0)
                if n > 100:
                    budget = a.budget_large
            cmd = [sys.executable, str(ROOT / "USER_WORKFLOW/bands_fit.py"), str(a.settings),
                   "--dft-dir", str(d), "--dft-code", code, "--name", d.name, "--budget", str(budget),
                   "--outdir", str(out), "--fitting-dir", str(out / "fitting")] + STRATEGIES[name]
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True)
            row = {"example": rel, "strategy": name, "budget": budget, "seconds": round(time.time() - t0, 1)}
            if p.returncode:
                err = [l for l in (p.stderr + p.stdout).splitlines() if "rror" in l]
                row["error"] = err[-1][:160] if err else f"exit {p.returncode}"
            else:
                st = json.loads((out / "run_summary.json").read_text())
                w = st.get("window_metrics") or {}
                row |= {k: round(float(w[k]), 4) for k in KEYS if k in w}
                row["frontier_pair_rmse_eV"] = round(float(st["selected_rmse_eV"]), 4)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if a.collect and "error" not in row:
                collect(a.work)
    cols = ["example", "strategy", "budget", "seconds", *KEYS, "frontier_pair_rmse_eV", "error"]
    with (a.work / "strategies.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    md = ["| " + " | ".join(cols[:-1]) + " |", "|" + "---|" * (len(cols) - 1)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, r.get("error", "")) if c != "budget" else r[c]) for c in cols[:-1]) + " |")
    (a.work / "strategies.md").write_text("\n".join(md) + "\n")
    print("TABLE", a.work / "strategies.md")


if __name__ == "__main__":
    main()
