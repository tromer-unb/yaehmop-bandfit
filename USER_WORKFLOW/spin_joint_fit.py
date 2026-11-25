#!/usr/bin/env python3
"""Joint EHT fit of both spin channels of a collinear spin-polarized DFT run, with one energy zero.

    python USER_WORKFLOW/spin_joint_fit.py --dft-dir validation/vasp/Fe_bcc_pbe_spin --dft-code vasp \\
        --start-up <fitting dir of the up fit> --start-down <fitting dir of the down fit> \\
        --optimizer cmaes --budget 2000 --polish nelder-mead --outdir outputs/Fe_joint

Unlike two separate fits (bands_fit.py --spin-channel up / down, then spin_combine.py), the two
parameter tables are fitted together and each candidate is run with the native spin-polarized bind
of the fork github.com/ac-dias/yaehmop (--parm-up / --parm-down): the common Fermi level of the model
(from both channels, with the smearing of the DFT run) is the energy zero of both channels, so the
exchange splitting, electron counts and magnetic moment come from the model, with no alignment
afterwards. See parameter_fitter/spin_joint.py for the objective.

The reference must be metallic or half-metallic (bands of both channels relative to the DFT Fermi
level, objective 'metal'). The starting tables are those of two earlier fits (--start-up,
--start-down: their fitting/ folders) or the stock table for both channels.

Writes, in --outdir:
  dft_reference/up, dft_reference/down   the DFT bands of each channel (as bands_fit.py)
  fit/fit.json, history.csv, progress.csv, param_best_up.dat, param_best_down.dat
  up/, down/                             figures of each channel against its DFT bands
  fitting/spin_up, fitting/spin_down     standard fitting/ folders, H(R) - E_F*S(R) with the COMMON E_F
  fitting/spin_polarized                 huckel-SP.dat, basis_set-SP for WanTiBEXOS (spin_combine.py,
                                         no alignment) and the native common-Fermi check
"""
from __future__ import annotations
from pathlib import Path
import argparse, csv, json, os, subprocess, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "parameter_fitter"), str(ROOT / "source"), str(HERE)]
from yaehmop_generic import (resolve_yaehmop_assets, bind_features, parse_parameter_table, subset_for_elements,
                             valence_electrons, orbital_count, write_parameter_file, gamma_mesh)
from structure_io import read_structure
from band_objective import WindowSettings
from fit_sparse_window import parse_metal_sample
from global_optimizers import run_global, GLOBAL_METHODS, POLISH_METHODS
from p3_objectives import bounded_nelder_mead
from spin_joint import SpinJointFitter, CHANNELS, SMEARINGS, common_fermi, read_dft_moment
import bands_fit as bf

METHODS = tuple(m for m in GLOBAL_METHODS if m != "alternating") + ("nelder-mead",)


def start_tables(args, elements):
    """Starting tables, K and objective settings: from the fitting/ folders of earlier fits, or stock."""
    _, table = resolve_yaehmop_assets()
    stock = subset_for_elements(parse_parameter_table(table), elements)
    tables, Ks, settings, radius = {}, {}, None, None
    for ch, d in (("up", args.start_up), ("down", args.start_down)):
        if d is None:
            tables[ch] = stock
            continue
        d = Path(d).resolve()
        tables[ch] = subset_for_elements(parse_parameter_table(d / "parameters_huckel.dat"), elements)
        sj = d / "fit_summary.json"
        if sj.is_file():
            s = json.loads(sj.read_text())
            if s.get("K_EHT") is not None:
                Ks[ch] = float(s["K_EHT"])
            settings = settings or s.get("objective_settings")
            radius = radius or s.get("overlap_radius") or (s.get("tight_binding_files") or {}).get("overlap_radius")
    mean = float(np.mean(list(Ks.values()))) if Ks else 1.75
    return tables, {ch: Ks.get(ch, mean) for ch in ("up", "down")}, settings, radius


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dft-dir", type=Path, required=True, help="folder of the spin-polarized DFT band run")
    ap.add_argument("--dft-code", required=True, help="vasp, qe, siesta or aims")
    ap.add_argument("--dft-file", default=None, help="eigenvalue file inside --dft-dir, if not the default")
    ap.add_argument("--fermi", type=float, default=None, help="DFT Fermi level (eV) instead of the file's")
    ap.add_argument("--structure", type=Path, default=None, help="structure file (default: the DFT structure)")
    ap.add_argument("--dimension", type=int, choices=(1, 2, 3), default=None, help="periodic dimension")
    ap.add_argument("--points-per-segment", type=int, default=20)
    ap.add_argument("--start-up", type=Path, default=None, help="fitting/ folder of an earlier spin-up fit")
    ap.add_argument("--start-down", type=Path, default=None, help="fitting/ folder of an earlier spin-down fit")
    ap.add_argument("--overlap-radius", type=int, default=None, help="lattice sum radius (default: the start "
                    "fits', else 5)")
    ap.add_argument("--mesh-radius", type=int, default=7,
                    help="k mesh of the common Fermi level: Gamma-centred (2r+1)^dim (default 7)")
    ap.add_argument("--smearing", choices=SMEARINGS, default="mp1",
                    help="occupations of the common Fermi level (default mp1 = Methfessel-Paxton 1, VASP ISMEAR=1)")
    ap.add_argument("--sigma", type=float, default=0.1, help="smearing width in eV (default 0.1)")
    ap.add_argument("--moment", type=float, default=None,
                    help="DFT magnetic moment per cell (default: read from the DFT files when possible)")
    ap.add_argument("--w-moment", type=float, default=0.0,
                    help="weight of |m_model - m_DFT| in the objective (eV per Bohr magneton; default 0 = report only)")
    ap.add_argument("--shared-zetas", action="store_true", help="the same Slater exponents for both channels")
    ap.add_argument("--k-per-channel", action="store_true",
                    help="one Wolfsberg-Helmholz constant K per channel (default: one K for both)")
    ap.add_argument("--objective", action="append", default=[], metavar="KEY=VALUE",
                    help="objective setting for both channels (repeatable), e.g. band_weights=v1:5 or window_max=3.5; "
                         "default: the settings of the start fits")
    ap.add_argument("--objective-up", action="append", default=[], metavar="KEY=VALUE",
                    help="objective setting for the spin-up channel only (repeatable)")
    ap.add_argument("--objective-down", action="append", default=[], metavar="KEY=VALUE",
                    help="objective setting for the spin-down channel only (repeatable)")
    ap.add_argument("--hii-window", type=float, default=None,
                    help="bounds of each Hii: start +- this (eV); default 1.5 when starting from earlier fits, else 5")
    ap.add_argument("--zeta-window", type=float, default=None,
                    help="bounds of the exponents: start +- this fraction; default 0.25 from earlier fits, "
                         "else the stock bounds")
    ap.add_argument("--sigma0", type=float, default=None,
                    help="initial CMA-ES step in the unit cube; default 0.02 from earlier fits (small: the metal objective has narrow valleys near a good fit), else 0.25")
    ap.add_argument("--optimizer", choices=METHODS, default="cmaes")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--polish", choices=POLISH_METHODS, default="nelder-mead")
    ap.add_argument("--polish-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--parallel", type=int, default=None, help="concurrent candidates (default: CPUs / 2)")
    ap.add_argument("--reg-weight", type=float, default=0.04)
    ap.add_argument("--name", default=None)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--no-rmatrix", action="store_true", help="do not write huckel-*_r.dat, the position matrix of the Slater basis (WanTiBEXOS BSE_CENTER_FILE; it takes seconds to minutes for large cells)")
    ap.add_argument("--no-native-check", action="store_true", help="skip spin_combine's native bind check")
    a = ap.parse_args()
    if a.no_rmatrix:
        os.environ["BANDFIT_RMATRIX"]="0"

    bind, _ = resolve_yaehmop_assets()
    feats = bind_features(bind)
    if not {"spin", "common_fermi"} <= feats:
        ap.error(f"{bind} lacks native spin polarization / --common-fermi: use the bind of github.com/ac-dias/yaehmop")
    try:
        code = bf.normalize_code(a.dft_code)
    except ValueError as exc:
        ap.error(str(exc))
    out = a.outdir.resolve(); out.mkdir(parents=True, exist_ok=True)
    name = a.name or a.dft_dir.resolve().name
    t0 = time.time()

    # ---- DFT bands of both channels, on the same YAeHMOP path ----
    L = {}
    for ch in CHANNELS:
        refdir = out / "dft_reference" / ch; refdir.mkdir(parents=True, exist_ok=True)
        L[ch] = bf.load_dft_reference(a.dft_dir, code, refdir, name=f"{name}_{ch}", dft_file=a.dft_file,
                                      spin_channel=ch, metallic=True, fermi=a.fermi, semicore="auto",
                                      structure=a.structure, structure_format=None, dimension=a.dimension,
                                      step=a.points_per_segment, pairs=1)
    if abs(L["up"]["ref"].fermi - L["down"]["ref"].fermi) > 1e-6:
        raise SystemExit("the two channels have different DFT Fermi levels")
    cif, sfmt, dim = L["up"]["cif"], L["up"]["sfmt"], L["up"]["dim"]
    atoms = read_structure(cif, sfmt); syms = atoms.get_chemical_symbols()
    elements = list(dict.fromkeys(syms))
    tables, K0, settings, radius = start_tables(a, elements)
    radius = int(a.overlap_radius or radius or 5)
    nelec = valence_electrons(tables["up"], syms)
    if valence_electrons(tables["down"], syms) != nelec:
        raise SystemExit("the two starting tables count different valence electrons")
    norb = orbital_count(tables["up"], syms)
    if orbital_count(tables["down"], syms) != norb:
        raise SystemExit("the two starting tables have different bases")
    samples = {}
    for ch in CHANNELS:
        e = dict(L[ch]["entry"], overlap_radius=radius)
        samples[ch] = parse_metal_sample(e, {}, {}, cif, Path(e["reference"]), sfmt, atoms, syms, norb, nelec)
    def kv(items, opt):
        out = {}
        for it in items:
            if "=" not in it:
                ap.error(f"{opt} takes KEY=VALUE, not {it!r}")
            k_, v_ = it.split("=", 1)
            out[k_.strip()] = v_.strip()
        return out
    base = {**(settings or {}), **kv(a.objective, "--objective")}
    try:
        wsets = {"up": WindowSettings.from_dict({**base, **kv(a.objective_up, "--objective-up")}),
                 "down": WindowSettings.from_dict({**base, **kv(a.objective_down, "--objective-down")})}
    except ValueError as exc:
        ap.error(str(exc))
    moment, msrc = (a.moment, "--moment") if a.moment is not None else read_dft_moment(a.dft_dir, code)
    if a.w_moment > 0 and moment is None:
        ap.error("--w-moment needs the DFT moment (--moment)")
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 2
    par = a.parallel or max(1, ncpu // 2)
    refine = a.start_up is not None and a.start_down is not None
    hii_w = a.hii_window if a.hii_window is not None else (1.5 if refine else 5.0)
    zeta_w = a.zeta_window if a.zeta_window is not None else (0.25 if refine else None)
    sigma0 = a.sigma0 if a.sigma0 is not None else (0.02 if refine else 0.25)
    if not a.k_per_channel:
        K0 = float(np.mean(list(K0.values())))
    f = SpinJointFitter(samples, tables, K0, wsets, bind=bind, nelec=nelec, mesh_radius=a.mesh_radius,
                        smearing=a.smearing, sigma=a.sigma, moment_ref=moment, w_moment=a.w_moment,
                        shared_zetas=a.shared_zetas, reg_weight=a.reg_weight, workdir=out / "fit" / "work",
                        parallel=par, hii_window=hii_w, zeta_window=zeta_w, k_window=0.3 if refine else 0.6,
                        k_per_channel=a.k_per_channel)
    print(f"SPIN_JOINT {name}: {len(f.names)} parameters, {nelec} electrons, mesh {2 * a.mesh_radius + 1}^{dim}, "
          f"smearing {a.smearing} {a.sigma} eV, DFT moment {moment} ({msrc}), {par} candidates in parallel", flush=True)
    c0, line0 = f.objective(f.stock)
    print(f"START cost {c0:.4f}: up window RMS {line0.get('up_window_rms_eV', float('nan')):.4f} eV, down "
          f"{line0.get('down_window_rms_eV', float('nan')):.4f} eV, moment {line0.get('moment', float('nan')):.3f}",
          flush=True)

    # ---- optimization ----
    if a.optimizer == "nelder-mead":
        res = bounded_nelder_mead(f.objective, f.stock, f.bounds, budget=a.budget)
        res["optimizer"] = "nelder-mead"
    else:
        res = run_global(a.optimizer, f.objective, f.stock, f.bounds, a.budget, seed=a.seed,
                         polish_method=a.polish, polish_fraction=a.polish_fraction,
                         objective_many=f.objective_many if par > 1 else None, parallel=par, sigma0=sigma0)
    x = res["best_x"] if res["best_x"] is not None and res["best_loss"] < c0 else f.stock
    best, line = f.objective(x)
    f.close()

    # ---- final model: common Fermi level, also on a dense mesh ----
    r = f.run(x, "final")
    mu, nu, nd, m, per = f.scores(r)
    dense = f.run(x, "final_dense", mesh=gamma_mesh(dim, 12))
    mu_d, nu_d, nd_d, m_d = common_fermi(dense["eig"]["up"], dense["eig"]["down"], nelec, a.sigma, a.smearing)
    tables_best, K = r["tables"], r["K"]
    fitdir = out / "fit"; fitdir.mkdir(exist_ok=True)
    for ch in CHANNELS:
        write_parameter_file(fitdir / f"param_best_{ch}.dat", tables_best[ch])
        (fitdir / f"param_best_{ch}.txt").write_text(
            f"K_EHT {K[ch]:.12g}\n" + "".join(f"{n_} {float(v):.12g}\n" for n_, v in zip(f.names, x)))
    hist = res["history"]
    keys = list(dict.fromkeys(k for row in hist for k in row))
    with (fitdir / "history.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=keys); w.writeheader(); w.writerows(hist)
    joint = dict(method="joint fit of both spin channels with the common Fermi level of the model "
                        "(native spin-polarized bind, --parm-up/--parm-down)",
                 eht_fermi_eV=mu, electrons_up=nu, electrons_down=nd, moment=m,
                 mesh=f"Gamma-centred {2 * a.mesh_radius + 1}^{dim}", smearing=a.smearing, sigma_eV=a.sigma,
                 dense_mesh=dict(mesh=f"Gamma-centred 25^{dim}", eht_fermi_eV=mu_d, electrons_up=nu_d,
                                 electrons_down=nd_d, moment=m_d),
                 bind_common_fermi_T0_eV=r["bind_fermi"], dft_moment=moment, dft_moment_source=msrc,
                 w_moment=a.w_moment, shared_zetas=a.shared_zetas, K_EHT=K, k_per_channel=a.k_per_channel,
                 overlap_min_eig=r["smin"],
                 bounds=dict(hii_window_eV=hii_w, zeta_window=zeta_w, cmaes_sigma0=sigma0))
    state = dict(name=name, optimizer=res["optimizer"], polish=res.get("polish"), budget=a.budget,
                 objective_evaluations=res["evaluations"], rejected_candidates=f.rejected,
                 start_cost=c0, best_cost=best, start_metrics=line0, best_metrics=line,
                 best_x={n_: float(v) for n_, v in zip(f.names, x)},
                 stock_x={n_: float(v) for n_, v in zip(f.names, f.stock)},
                 start_up=str(a.start_up) if a.start_up else None, start_down=str(a.start_down) if a.start_down else None,
                 objective_settings={ch: wsets[ch].to_dict() for ch in CHANNELS}, overlap_radius=radius, periodic_dim=dim, joint=joint,
                 metrics={ch: per[ch] for ch in CHANNELS}, time_s=time.time() - t0)
    (fitdir / "fit.json").write_text(json.dumps(state, indent=2, default=float) + "\n")

    # ---- figures and fitting/ folders of each channel (energy zero: the common E_F) ----
    er = L["up"]["er"]
    kpts = bf.path_kpoints(er.nodes, a.points_per_segment)[er.mask]
    nodes = bf.node_positions(len(er.nodes), a.points_per_segment, er.mask)
    for ch in CHANNELS:
        s = samples[ch]; d = out / ch; d.mkdir(exist_ok=True)
        E = r["bands"][ch][s["kmask"]] - mu
        nb = s["target"].shape[1]
        npz = d / f"{name}_{ch}_bands.npz"
        np.savez_compressed(npz, k=s["kx"], reference=s["target"], YAeHMOP=E[:, :nb], reference_all=s["target_all"],
                            YAeHMOP_all=E, tick_x=np.array([t for t, _ in s["ticks"]], float),
                            tick_labels=np.array([l for _, l in s["ticks"]]), metallic=True,
                            dft_fermi=s["dft_fermi"], eht_fermi=mu,
                            window=np.array([wsets[ch].window_min, wsets[ch].window_max]))
        pm = bf.write_metal_plots(npz, d, title=f"{name} — joint spin fit vs DFT (spin {ch})",
                                  align_note="common E$_F$ of the spin-polarized model")
        info = L[ch]["info"]
        summary = dict(name=f"{name}_{ch}", reference_type="dft", metallic=True, dft_reference=info,
                       objective="metal (joint spin fit, common Fermi level)", objective_settings=wsets[ch].to_dict(),
                       metrics=per[ch], window_metrics=per[ch], selected_rmse_eV=pm["rmse"],
                       rms_within_1eV_of_EF_eV=pm["near"], eht_fermi_eV=mu, K_EHT=K[ch], periodic_dim=dim,
                       overlap_radius=radius, joint_spin_fit=joint, optimizer_choice=a.optimizer,
                       polish=a.polish, requested_total_budget=a.budget,
                       optimizer_evaluations=res["evaluations"],
                       note="Both spin channels fitted together; the energy zero of huckel-NP.dat is the common "
                            "Fermi level of the model (H(R) - E_F*S(R)), the same for both channels")
        bf.write_fitting_dir(out / "fitting" / f"spin_{ch}", d, structure=cif, sfmt=sfmt,
                             native=fitdir / f"param_best_{ch}.dat", param_txt=fitdir / f"param_best_{ch}.txt",
                             K=K[ch], kpoints=kpts, dim=dim, radius=radius, summary=summary, node_index=nodes,
                             energy_zero=mu)
        print(f"CHANNEL {ch}: window RMS {per[ch]['window_rms_eV']:.4f} eV, within 1 eV of E_F "
              f"{per[ch]['rms_within_1eV_of_EF_eV']:.4f} eV", flush=True)

    # ---- spin-polarized Hamiltonian for WanTiBEXOS (no alignment: one energy zero already) ----
    cmd = [sys.executable, str(HERE / "spin_combine.py"), "--up", str(out / "fitting/spin_up"),
           "--down", str(out / "fitting/spin_down"), "--outdir", str(out / "fitting/spin_polarized"), "--no-align"]
    if a.no_native_check:
        cmd.append("--no-native")
    subprocess.run(cmd, check=True)
    print(f"SPIN_JOINT_DONE {name}: cost {c0:.4f} -> {best:.4f}; E_F(model) {mu:.4f} eV; N_up {nu:.3f}, N_down "
          f"{nd:.3f}, moment {m:.3f} (dense mesh {m_d:.3f}; DFT {moment}); {res['evaluations']} evaluations, "
          f"{time.time() - t0:.0f} s", flush=True)
    print("FITTING", out / "fitting")


if __name__ == "__main__":
    main()
