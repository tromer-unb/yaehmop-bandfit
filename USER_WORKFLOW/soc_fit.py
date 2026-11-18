#!/usr/bin/env python3
"""Fit spin-orbit constants on top of an EHT (YAeHMOP) fit, against DFT bands with SOC.

YAeHMOP has no spin-orbit term; it is added as on-site lambda L.S per element and
shell (parameter_fitter/soc.py). The EHT parameters of an earlier spinless fit
stay fixed; only the lambdas are fitted, with the band-window objective (bands
near the Fermi level, taken in the middle of the gap, weigh more).

    python USER_WORKFLOW/soc_fit.py --dft-dir path/to/bands-soc --dft-code vasp \\
        --from-fitting outputs/material/fitting --outdir outputs/material_soc

Outputs in OUTDIR/fitting-soc (or --fitting-dir): soc_parameters.txt (lambdas),
parameters_huckel.dat (the fixed EHT table), huckel-SOC.dat + basis_set-SOC
(WanTiBEXOS SOC format, spin-up orbitals first, then spin-down), orbitals.dat,
POSCAR, figures with the DFT (blue) and EHT+SOC (red, dashed) bands and errors,
fit_summary.json.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, shutil, sys, tempfile
import numpy as np
import scipy.linalg as sl
from scipy.optimize import differential_evolution, minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "parameter_fitter"), str(ROOT / "source"), str(HERE)]
from yaehmop_generic import (resolve_yaehmop_assets, parse_parameter_table, subset_for_elements,
                             valence_electrons, write_parameter_file, write_periodic_input_generic,
                             run_bind, read_matrix_dump, overlap_images_for, orbital_labels,
                             real_space_matrices, gamma_mesh, DEFAULT_OVERLAP_RADIUS)
from dft_bands import read_dft_bands, eht_reference, detect_periodic_dim, DFT_CODES
from band_objective import WindowSettings, window_error
from soc import soc_matrix, soc_shells, soc_bands, write_wtb_soc
from bands_fit import path_kpoints, write_bands_plot, write_full_comparison, portable_paths

README = """Spin-orbit outputs (EHT + on-site lambda L.S)
=============================================
soc_parameters.txt     fitted lambda (eV) per element and shell
parameters_huckel.dat  the EHT table used (fixed, from the spinless fit)
huckel-SOC.dat         spinor H(R), S(R) in the WanTiBEXOS SOC format: SOC / scissors /
                       Fermi level (0) / 3 lattice vectors [A] / 2n orbitals / n cells /
                       R_x R_y R_z [A]  i  j  ReH  ImH  S. Orbitals 1..n are spin up,
                       n+1..2n the same orbitals spin down. ReH + i ImH is
                       H(R) - E_mid*S(R) (x) I2 plus lambda L.S on R = 0 (E_mid in
                       fit_summary.json), so energies are relative to the EHT midgap.
basis_set-SOC          WanTiBEXOS basis file (spin column 1 / -1)
orbitals.dat           spinless orbital list (index, atom, orbital, l, m)
bands_comparison.*     2 VB + 2 CB with SOC: DFT blue, EHT+SOC red dashed, errors
bands_full_comparison.* all bands in the window
POSCAR                 structure used
fit_summary.json       lambdas, errors with and without SOC, settings
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dft-dir", type=Path, required=True, help="DFT band calculation WITH spin-orbit")
    ap.add_argument("--dft-code", choices=DFT_CODES, required=True)
    ap.add_argument("--dft-file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-fitting", type=Path, help="fitting/ folder of a spinless EHT fit")
    src.add_argument("--parameters", type=Path, help="YAeHMOP parameter table (with --k)")
    ap.add_argument("--k", type=float, help="K_EHT with --parameters")
    ap.add_argument("--dimension", type=int, choices=(1, 2, 3))
    ap.add_argument("--overlap-radius", type=int)
    ap.add_argument("--points-per-segment", type=int, default=15)
    ap.add_argument("--lambda-max", type=float, default=1.5, help="upper bound of every lambda (eV)")
    ap.add_argument("--maxiter", type=int, default=60, help="differential-evolution generations")
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--w-split", type=float, default=100.0,
                    help="weight of the spin-orbit splittings of the band pairs near E_F")
    ap.add_argument("--split-pairs", type=int, default=3, help="valence and conduction pairs whose splitting is fitted")
    ap.add_argument("--soc-shells", help="shells with a fitted lambda, e.g. 'Mo:d S:p' (default: d of transition "
                                         "metals, none for Zn/Cd/Hg, p and d of the other elements)")
    ap.add_argument("--emin", type=float, default=-6.0); ap.add_argument("--emax", type=float, default=6.0)
    ap.add_argument("--name"); ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--fitting-dir", type=Path)
    a = ap.parse_args()

    outdir = a.outdir.resolve(); outdir.mkdir(parents=True, exist_ok=True)
    fdir = (a.fitting_dir or outdir / "fitting-soc").resolve(); fdir.mkdir(parents=True, exist_ok=True)
    if a.from_fitting:
        fs = json.loads((a.from_fitting / "fit_summary.json").read_text())
        table = a.from_fitting / "parameters_huckel.dat"
        K = float(fs["K_EHT"]); radius = a.overlap_radius or int(fs.get("overlap_radius", DEFAULT_OVERLAP_RADIUS))
        dim0 = fs.get("periodic_dim")
    else:
        if a.k is None:
            ap.error("--k is required with --parameters")
        table, K, radius, dim0 = a.parameters, a.k, a.overlap_radius or DEFAULT_OVERLAP_RADIUS, None
    ref = read_dft_bands(a.dft_dir, a.dft_code, a.dft_file, soc=True)
    atoms = ref.atoms; syms = atoms.get_chemical_symbols()
    dim = a.dimension or dim0 or detect_periodic_dim(atoms)
    name = a.name or Path(a.dft_dir).resolve().name
    from ase.io import write
    write(fdir / "POSCAR", atoms, format="vasp", direct=True)
    structure = fdir / "POSCAR"
    params = subset_for_elements(parse_parameter_table(table), list(dict.fromkeys(syms)))
    nelec = valence_electrons(params, syms)
    labels = orbital_labels(params, syms); n = len(labels)
    shells = soc_shells(labels)
    if a.soc_shells:
        from band_objective import parse_basis_add
        want = {(el, o) for el, orbs in parse_basis_add(a.soc_shells).items() for o in orbs}
        shells = [s for s in shells if s in want]
    else:   # transition metals: only d (their p shell is empty, high above E_F);
            # d10 metals (Zn, Cd, Hg): none (filled, deep d shell; empty p shell)
        tm = lambda z: 21 <= z <= 29 or 39 <= z <= 47 or 57 <= z <= 79
        d10 = lambda z: z in (30, 48, 80)
        shells = [(el, sh) for el, sh in shells
                  if not d10(params[el]["atnum"]) and not (tm(params[el]["atnum"]) and sh != "d")]
    if not shells:
        ap.error("no shell with l >= 1 selected for spin-orbit")
    er = eht_reference(ref, a.points_per_segment, dim)
    kpts = path_kpoints(er.nodes, a.points_per_segment)[er.mask]
    bind, _ = resolve_yaehmop_assets()
    images = overlap_images_for(dim, radius)
    mesh = gamma_mesh(dim, radius)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); write_parameter_file(tmp / "p.parms", params)
        def dump(ks, tag):
            inp = write_periodic_input_generic(structure, tmp / f"{tag}.bind", None, K, True, nelec,
                                               overlap_images=images, periodic_dim=dim, structure_format="vasp",
                                               kpoints=ks)
            run_bind(bind, inp, tmp / "p.parms", tmp / f"{tag}.log")
            return read_matrix_dump(str(inp) + ".HAM"), read_matrix_dump(str(inp) + ".OV")
        Hk, Sk = dump(kpts, "path")
        Hm, Sm = dump(mesh, "mesh")
    wset = WindowSettings(window_min=a.emin, window_max=a.emax)
    ticks_x = [x for x, _ in er.ticks]

    def bands_for(lams):
        return soc_bands(Hk, Sk, soc_matrix(labels, dict(zip(shells, lams))))

    def splittings(B, nocc):
        """Splitting E[2j+1]-E[2j] of the spinor pairs counted from the gap, and their energies."""
        out, en = [], []
        for j in range(a.split_pairs):
            lo, hi = nocc - 2 - 2 * j, nocc - 1 - 2 * j          # valence pair j
            if lo >= 0: out.append(B[:, hi] - B[:, lo]); en.append(.5 * (B[:, hi] + B[:, lo]))
        for j in range(a.split_pairs):
            lo, hi = nocc + 2 * j, nocc + 1 + 2 * j              # conduction pair j
            if hi < B.shape[1]: out.append(B[:, hi] - B[:, lo]); en.append(.5 * (B[:, hi] + B[:, lo]))
        return np.array(out), np.array(en)
    sd, ed = splittings(er.energies, ref.nocc)
    emid_ref = .5 * (er.energies[:, ref.nocc].min() + er.energies[:, ref.nocc - 1].max())
    gw = wset.far_weight + (1 - wset.far_weight) * np.exp(-np.abs(ed - emid_ref) / wset.energy_scale)

    def split_rms(B):
        se, _ = splittings(B, nelec)
        m = min(len(se), len(sd))
        return float(np.sqrt(np.sum(gw[:m] * (se[:m] - sd[:m]) ** 2) / np.sum(gw[:m])))

    def metrics(lams):
        B = bands_for(lams)
        m = window_error(er.energies, ref.nocc, B, nelec, er.x, ticks_x, wset)
        m["rms_soc_splitting_eV"] = split_rms(B)
        m["dft_max_splitting_eV"] = float(sd.max())
        m["eht_max_splitting_eV"] = float(splittings(B, nelec)[0].max())
        return m

    def cost(lams):
        try:
            B = bands_for(lams)
            w = window_error(er.energies, ref.nocc, B, nelec, er.x, ticks_x, wset)["total"]
            return float(np.sqrt(w ** 2 + a.w_split * split_rms(B) ** 2))
        except Exception:
            return 1e3

    m0 = metrics(np.zeros(len(shells)))
    bounds = [(0.0, a.lambda_max)] * len(shells)
    de = differential_evolution(cost, bounds, seed=a.seed, maxiter=a.maxiter, popsize=10, polish=False, tol=1e-7)
    loc = minimize(cost, de.x, method="Nelder-Mead", bounds=bounds,
                   options={"xatol": 1e-5, "fatol": 1e-7, "maxfev": 600})
    lam = loc.x if loc.fun < de.fun else de.x
    lambdas = dict(zip(shells, [float(v) for v in lam]))
    Hsoc = soc_matrix(labels, lambdas)
    E = soc_bands(Hk, Sk, Hsoc)
    m1 = metrics(lam)
    # Figures (both aligned at their own midgap, as in bands_fit)
    p = 2
    T = er.energies; dn = ref.nocc
    rmid = .5 * (T[:, dn].min() + T[:, dn - 1].max()); emid = .5 * (E[:, nelec].min() + E[:, nelec - 1].max())
    npz = fdir / f"{name}_soc_bands.npz"
    np.savez_compressed(npz, k=er.x, reference=T[:, dn - p:dn + p] - rmid, YAeHMOP=E[:, nelec - p:nelec + p] - emid,
                        reference_all=T - rmid, YAeHMOP_all=E - emid,
                        tick_x=np.array(ticks_x, float), tick_labels=np.array([l for _, l in er.ticks]))
    write_bands_plot(npz, fdir, 2 * p, title=f"{name} — EHT + SOC vs DFT (2 VB + 2 CB)")
    write_full_comparison(npz, fdir, title=f"{name} — all bands, EHT + SOC vs DFT", emin=a.emin, emax=a.emax,
                          window_metrics=m1)
    # WanTiBEXOS SOC file from H(R), S(R) (exact, from the mesh) with the same energy zero
    cells = real_space_matrices(Hm, Sm, mesh, dim, radius)
    kept = [(R, H.real, S.real) for R, H, S in cells if max(np.abs(H).max(), np.abs(S).max()) > 1e-7]
    Em = soc_bands(Hm, Sm, Hsoc)
    allE = np.vstack([E, Em]); e_mid = .5 * (allE[:, nelec].min() + allE[:, nelec - 1].max())
    write_wtb_soc(fdir / "huckel-SOC.dat", kept, atoms.cell, labels, Hsoc, e_mid, atoms.positions, syms,
                  fdir / "basis_set-SOC")
    # check: the written file reproduces the bands on the path
    I2 = np.eye(2); errs = []
    for idx in range(0, len(kpts), max(1, len(kpts) // 12)):
        kq = kpts[idx]
        Hr = sum(np.exp(2j * np.pi * (kq @ R)) * np.kron(I2, H - e_mid * S) for R, H, S in kept) + Hsoc
        Sr = sum(np.exp(2j * np.pi * (kq @ R)) * np.kron(I2, S) for R, H, S in kept)
        ev = sl.eigh(Hr, Sr, eigvals_only=True)
        errs.append(np.abs(ev - (E[idx] - e_mid)).max())
    from yaehmop_generic import ORBITAL_LM
    with open(fdir / "orbitals.dat", "w") as f:
        f.write("# index  atom  orbital  l  m   (YAeHMOP order; spinor basis: these n spin up, then spin down)\n")
        for b, lab in enumerate(labels):
            atom, orb = lab.split(":"); l, m = ORBITAL_LM[orb]
            f.write(f"{b + 1:6d}  {atom:8s} {orb:12s} {l} {m:2d}\n")
    shutil.copy(table, fdir / "parameters_huckel.dat")
    with open(fdir / "soc_parameters.txt", "w") as f:
        f.write("# on-site spin-orbit constants lambda (eV): H_SOC = lambda L.S\n# element shell lambda_eV\n")
        for (el, sh), v in lambdas.items():
            f.write(f"{el:3s} {sh}  {v:.6f}\n")
    summary = dict(name=name, dft_dir=str(Path(a.dft_dir).resolve()), dft_code=a.dft_code,
                   eht_parameters=str(Path(table).resolve()), K_EHT=K, periodic_dim=dim, overlap_radius=radius,
                   lambdas_eV={f"{el}-{sh}": v for (el, sh), v in lambdas.items()},
                   metrics_without_soc=m0, metrics_with_soc=m1, eht_midgap_eV=float(e_mid),
                   spinor_basis="2n functions: orbitals 1..n spin up, n+1..2n spin down",
                   huckel_SOC_check_max_error_eV=float(max(errs)), n_orbitals_spinless=n,
                   w_split=a.w_split, split_pairs=a.split_pairs,
                   n_cells=len(kept), objective_settings=wset.to_dict())
    summary = portable_paths(json.loads(json.dumps(summary, default=float)), fdir)
    (fdir / "fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (fdir / "README.txt").write_text(README)
    print("SOC_FIT_DONE", name, "lambdas", {f"{el}-{sh}": round(v, 4) for (el, sh), v in lambdas.items()})
    print(f"  RMS within 2 eV of E_F: without SOC {m0['rms_within_2eV_of_EF_eV']:.4f} eV, "
          f"with SOC {m1['rms_within_2eV_of_EF_eV']:.4f} eV; gap error {m0['gap_error_eV']:+.3f} -> {m1['gap_error_eV']:+.3f} eV")
    print(f"  spin-orbit splittings near E_F: RMS error {m0['rms_soc_splitting_eV']:.4f} -> {m1['rms_soc_splitting_eV']:.4f} eV; "
          f"largest splitting DFT {m1['dft_max_splitting_eV']:.3f} eV, EHT {m1['eht_max_splitting_eV']:.3f} eV")
    print("  huckel-SOC.dat check (max |dE|):", f"{max(errs):.1e} eV")
    print("FITTING_DIR", fdir)


if __name__ == "__main__":
    main()
