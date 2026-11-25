#!/usr/bin/env python3
"""Join two spin-channel EHT fits into one spin-polarized Hamiltonian for WanTiBEXOS.

A collinear spin-polarized DFT run is fitted channel by channel
(bands_fit.py --spin-channel up / down), giving one EHT parameter set per spin.
This tool puts both in one spinor basis of 2n functions, ordered as all n
orbitals with spin up, then the same n orbitals with spin down:

    H = | H_up   0     |      S = | S_up  0    |
        | 0      H_dn  |          | 0     S_dn |

The up-down blocks are zero (no spin-orbit coupling). Each channel of the fits
is written relative to its own EHT midgap; here it is shifted by
(DFT midgap of that channel - DFT Fermi level) * S, so that the two channels keep
the relative position (exchange splitting) of the DFT bands, with the Fermi
level at 0.

    python USER_WORKFLOW/spin_combine.py --up outputs/up/fitting --down outputs/down/fitting \\
        --outdir outputs/spin

Writes huckel-SP.dat (WanTiBEXOS 'SP' format, as siesta2wtb.py), basis_set-SP,
parameters_huckel_up.dat / _down.dat, orbitals.dat, POSCAR, fit_summary.json.

With the bind of the fork github.com/ac-dias/yaehmop (native spin polarization,
--parm-up / --parm-down), the two tables are also run together by bind: the bands
of each channel are checked against huckel-SP.dat, and --common-fermi gives the
Fermi level shared by both channels, the electrons of each channel and the magnetic
moment of the EHT model (fit_summary.json "native_spin"). yaehmop_spin/ holds the
input to repeat that run (bind --parm-up ... --parm-down ... --common-fermi).
"""
from __future__ import annotations
from pathlib import Path
import argparse, gzip, json, os, shutil, sys, tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parameter_fitter"))


def read_tb(path):
    """huckel-NP.dat (or .gz) -> (cell, n, {R_cart (rounded): (H, S)})."""
    path = Path(path)
    if not path.is_file() and Path(str(path) + ".gz").is_file():
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        lines = f.read().splitlines()
    cell = np.array([[float(x) for x in lines[3 + i].split()] for i in range(3)])
    n, nc = int(lines[6]), int(lines[7])
    data = np.array([[float(x) for x in l.split()] for l in lines[9:9 + nc * n * n]])
    if len(data) != nc * n * n:
        raise ValueError(f"{path}: expected {nc * n * n} matrix lines, found {len(data)}")
    blocks = {}
    for c in range(nc):
        d = data[c * n * n:(c + 1) * n * n]
        R = tuple(np.round(d[0, :3], 5))
        blocks[R] = (d[:, 5].reshape(n, n), d[:, 7].reshape(n, n))
    return cell, n, blocks


def native_spin(up, dn, out, mesh_radius=4, align=True):
    """Both channels in one native spin-polarized bind run (fork ac-dias/yaehmop), on a
    Gamma-centred mesh: per-channel bands (from the dumped H(k), S(k)) against huckel-SP.dat,
    common Fermi level, electrons per channel and magnetic moment.

    The two channel tables are fitted separately, each aligned at the DFT Fermi level with its own
    energy offset, so their ABSOLUTE EHT energies differ by an arbitrary constant and a common-Fermi
    run with them is not physical. With align=True all Hii of the spin-up table are shifted by -delta/2
    and those of the spin-down table by +delta/2 (secant search on bind runs) until both channels have
    the same offset, i.e. both tables put the DFT Fermi level at the same EHT energy; the common-Fermi
    run is then repeated with the aligned tables (yaehmop_spin/up.dat, down.dat; the fitted ones as
    up_fit.dat, down_fit.dat). A uniform
    Hii shift is not an exact rigid shift in EHT (K and THE CONST DIAG weigh it), so the remaining
    change of the band shape is reported."""
    import scipy.linalg as sl
    from yaehmop_generic import (resolve_yaehmop_assets, bind_features, run_bind_args, run_bind, gamma_mesh,
                                 write_periodic_input_generic, overlap_images_for, read_matrix_dump,
                                 parse_parameter_table, valence_electrons, write_parameter_file)
    from structure_io import read_structure
    bind, _ = resolve_yaehmop_assets()
    if "spin" not in bind_features(bind):
        return {"skipped": f"{bind} has no native spin polarization (--parm-up); use github.com/ac-dias/yaehmop"}
    tb = up["summary"].get("tight_binding_files") or {}
    dim = int(up["summary"].get("periodic_dim") or 3)
    radius = int(tb.get("overlap_radius") or up["summary"].get("overlap_radius") or 5)
    poscar = up["dir"] / "POSCAR"
    syms = read_structure(poscar, "vasp").get_chemical_symbols()
    nelec = valence_electrons(parse_parameter_table(up["dir"] / "parameters_huckel.dat"), syms)
    mesh = gamma_mesh(dim, mesh_radius)
    images = overlap_images_for(dim, radius)
    keep = out / "yaehmop_spin"; keep.mkdir(exist_ok=True)
    shutil.copy(poscar, keep / "POSCAR")
    K = {"up": float(up["summary"]["K_EHT"]), "down": float(dn["summary"]["K_EHT"])}
    tables = {"up": parse_parameter_table(up["dir"] / "parameters_huckel.dat"),
              "down": parse_parameter_table(dn["dir"] / "parameters_huckel.dat")}

    def sp_bands(ch):
        frac_R = {R: np.linalg.solve(ch["cell"].T, np.array(R)) for R in ch["blocks"]}
        out_ = []
        for kq in mesh:
            ph = {R: np.exp(2j * np.pi * (kq @ f)) for R, f in frac_R.items()}
            Hk = sum(ph[R] * (B[0] + ch["shift"] * B[1]) for R, B in ch["blocks"].items())
            Sk = sum(ph[R] * B[1] for R, B in ch["blocks"].items())
            out_.append(sl.eigh(Hk, Sk, eigvals_only=True))
        return np.array(out_)
    SP = {"up": sp_bands(up), "down": sp_bands(dn)}
    near = {t: np.abs(SP[t]) < 3.0 for t in SP}                 # states within 3 eV of E_F (huckel-SP frame)

    def eig_dump(prefix):
        H = read_matrix_dump(f"{prefix}.HAM"); S = read_matrix_dump(f"{prefix}.OV")
        return np.array([sl.eigh(h, s, eigvals_only=True) for h, s in zip(H, S)])

    def compare(nat, tag):
        d = nat - SP[tag]
        dn_ = d[near[tag]]
        off = float(dn_.mean())
        ok = SP[tag] < 15.0                                     # above: near-null directions of S(k)
        return dict(energy_offset_eV=off, max_deviation_within_3eV_eV=float(np.abs(dn_ - off).max()),
                    max_deviation_up_to_15eV_eV=float(np.abs(d[ok] - off).max()))

    def channel(table, k, tag, tmp):
        """Spinless bind run of one channel table on the mesh (same H as that channel of the spin run)."""
        write_parameter_file(tmp / f"{tag}.dat", table)
        inp = write_periodic_input_generic(poscar, tmp / f"{tag}.bind", None, k, True, nelec, overlap_images=images,
                                           periodic_dim=dim, structure_format="vasp", kpoints=mesh)
        run_bind(bind, inp, tmp / f"{tag}.dat", tmp / f"{tag}.log")
        return eig_dump(tmp / f"{tag}.bind")

    def spin_run(t_up, t_dn, tmp, label):
        d = tmp / label; d.mkdir()
        write_parameter_file(d / "up.dat", t_up); write_parameter_file(d / "down.dat", t_dn)
        write_periodic_input_generic(poscar, d / "spin.bind", None, K["up"], True, nelec, overlap_images=images,
                                     periodic_dim=dim, structure_format="vasp", kpoints=mesh, average_properties=True)
        args = ["--parm-up", "up.dat", "--parm-down", "down.dat", "--const-up", f"{K['up']:.12g}",
                "--const-down", f"{K['down']:.12g}", "--common-fermi"]
        cwd = os.getcwd()
        try:
            os.chdir(d)
            run_bind_args(bind, args, "spin.bind", None, d / "spin.log")
        finally:
            os.chdir(cwd)
        spin = {}
        for line in (d / "spin.bind.spin").read_text().splitlines():
            if line.startswith("#") and ":" in line:
                k, v = line[1:].split(":", 1)
                try:
                    spin[k.strip()] = float(v.split()[0])
                except (ValueError, IndexError):
                    spin[k.strip()] = v.strip()
        checks = {tag: compare(eig_dump(d / f"spin_{tag}.bind"), tag) for tag in ("up", "down")}
        ef = spin.get("Fermi_Energy")
        r = dict(command="bind " + " ".join(args) + " spin.bind", common_fermi_eV=ef,
                 electrons_up=spin.get("ELECTRONS_UP"), electrons_down=spin.get("ELECTRONS_DOWN"),
                 magnetic_moment_per_cell=spin.get("MAGNETIC_MOMENT"), bands_vs_huckel_SP=checks)
        if ef is not None:
            r["common_fermi_in_huckel_SP_frame_eV"] = {t: float(ef) - c["energy_offset_eV"] for t, c in checks.items()}
        return r, d

    def shifted(table, delta):
        import copy
        t2 = copy.deepcopy(table)
        for el in t2.values():
            for o in ("s", "p", "d", "f"):
                if o in el:
                    el[o]["Hii"] = float(el[o]["Hii"]) + delta
        return t2

    counts = {t: float((SP[t] < 0).sum() / len(mesh)) for t in SP}
    res = dict(k_mesh=f"Gamma-centred {2 * mesh_radius + 1}^{dim}",
               huckel_SP_electrons_below_EF={**counts, "moment": counts["up"] - counts["down"]})
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw, raw_dir = spin_run(tables["up"], tables["down"], tmp, "fitted")
        res["fitted_tables"] = raw
        final_dir, t_dn = raw_dir, tables["down"]
        if align:
            # the gap between the two offsets is split: up Hii - delta/2, down Hii + delta/2, so that each
            # table moves half as much (smaller change of the band shape than shifting one channel)
            def gap(delta):
                ou = compare(channel(shifted(tables["up"], -delta / 2), K["up"], f"up{len(hist)}", tmp), "up")
                od = compare(channel(shifted(tables["down"], delta / 2), K["down"], f"dn{len(hist)}", tmp), "down")
                return od["energy_offset_eV"] - ou["energy_offset_eV"]
            b = raw["bands_vs_huckel_SP"]
            d0, f0 = 0.0, b["down"]["energy_offset_eV"] - b["up"]["energy_offset_eV"]
            hist = [(d0, f0)]
            d1 = -f0
            for _ in range(8):
                f1 = gap(d1)
                hist.append((d1, f1))
                if abs(f1) < 2e-4 or abs(f1 - f0) < 1e-12:
                    break
                d0, f0, d1 = d1, f1, d1 - f1 * (d1 - d0) / (f1 - f0)
            delta = min(hist, key=lambda h: abs(h[1]))[0]
            t_up, t_dn = shifted(tables["up"], -delta / 2), shifted(tables["down"], delta / 2)
            ali, final_dir = spin_run(t_up, t_dn, tmp, "aligned")
            res["aligned_tables"] = ali
            res["alignment"] = dict(delta_Hii_up_eV=float(-delta / 2), delta_Hii_down_eV=float(delta / 2),
                                    remaining_offset_difference_eV=float(min(abs(h[1]) for h in hist)),
                                    search=[[float(a), float(c)] for a, c in hist])
        for f_ in ("up.dat", "down.dat", "spin.bind", "spin.bind.spin"):
            shutil.copy(final_dir / f_, keep / f_)
        if align:
            write_parameter_file(keep / "up_fit.dat", tables["up"])
            write_parameter_file(keep / "down_fit.dat", tables["down"])
    res["note"] = ("energy_offset_eV: native EHT energy minus huckel-SP.dat energy (per channel, states within 3 eV of "
                   "E_F); fitted_tables: the two tables as fitted, whose absolute energies differ by an arbitrary constant "
                   "(common Fermi level not physical); aligned_tables: Hii of each channel shifted (delta_Hii_up_eV, "
                   "delta_Hii_down_eV) so both channels put the DFT Fermi level at the same EHT energy "
                   "(yaehmop_spin/up.dat, down.dat); "
                   "max_deviation_*: change of band shape against huckel-SP.dat")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--up", type=Path, required=True, help="fitting/ folder of the spin-up fit")
    ap.add_argument("--down", type=Path, required=True, help="fitting/ folder of the spin-down fit")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--no-native", action="store_true",
                    help="skip the native spin-polarized bind run (fork github.com/ac-dias/yaehmop)")
    ap.add_argument("--no-align", action="store_true",
                    help="native run with the two tables as fitted (their absolute EHT energies differ by an arbitrary "
                         "constant); default: shift the spin-down Hii so that both channels share the energy zero")
    ap.add_argument("--mesh-radius", type=int, default=4,
                    help="native run: Gamma-centred k mesh of (2r+1)^dim points (default 4)")
    a = ap.parse_args()
    out = a.outdir.resolve(); out.mkdir(parents=True, exist_ok=True)
    chans = {}
    for tag, d in (("up", a.up), ("down", a.down)):
        s = json.loads((d / "fit_summary.json").read_text())
        dr = s.get("dft_reference") or {}
        if dr.get("spin_channel") not in (None, tag):
            raise SystemExit(f"{d}: fitted spin channel is {dr.get('spin_channel')}, not {tag}")
        # zero of huckel-NP.dat: the DFT midgap (gapped channel) or the DFT Fermi level (metal)
        zero = dr.get("energy_zero_eV", dr.get("midgap_eV"))
        if zero is None or dr.get("fermi_eV") is None:
            raise SystemExit(f"{d}: fit_summary.json lacks the DFT midgap / Fermi level (refit with this version)")
        cell, n, blocks = read_tb(d / "huckel-NP.dat")
        chans[tag] = dict(dir=d, summary=s, cell=cell, n=n, blocks=blocks,
                          shift=float(zero) - float(dr["fermi_eV"]))
    up, dn = chans["up"], chans["down"]
    if up["n"] != dn["n"]:
        raise SystemExit(f"Different basis sizes: up {up['n']}, down {dn['n']} orbitals (use the same basis)")
    if not np.allclose(up["cell"], dn["cell"], atol=1e-5):
        raise SystemExit("The two fits have different cells")
    n = up["n"]; Z = np.zeros((n, n))
    cells = sorted(set(up["blocks"]) | set(dn["blocks"]), key=lambda R: (abs(np.array(R)).sum(), R))
    with open(out / "huckel-SP.dat", "w") as f:
        f.write("SP\n0.000000\n0.000000\n")
        for v in up["cell"]:
            f.write(f"{v[0]:.8f}  {v[1]:.8f}  {v[2]:.8f}\n")
        f.write(f"{2 * n}\n{len(cells)}\n#rcell x   rcell y   rcell z   i   j   ReH   ImH   S\n")
        for R in cells:
            Hu, Su = up["blocks"].get(R, (Z, Z)); Hd, Sd = dn["blocks"].get(R, (Z, Z))
            H = np.block([[Hu + up["shift"] * Su, Z], [Z, Hd + dn["shift"] * Sd]])
            S = np.block([[Su, Z], [Z, Sd]])
            for i in range(2 * n):
                for j in range(2 * n):
                    f.write(f"{R[0]:.8f}   {R[1]:.8f}   {R[2]:.8f}   {i + 1}   {j + 1}   "
                            f"{H[i, j]:.10f}   0.0000000000   {S[i, j]:.10f}\n")
    rows = [l for l in (up["dir"] / "basis_set-NP").read_text().splitlines()[1:] if l.strip()]
    with open(out / "basis_set-SP", "w") as f:
        f.write("#bindex aspecie ax ay az l m spin\n")
        for spin, off in ((1, 0), (-1, n)):
            for r in rows:
                p = r.split()
                f.write(f"{int(p[0]) + off}   {'   '.join(p[1:7])}   {spin}\n")
    # position matrix: the X of each channel in its spin block (r is spin diagonal)
    rfiles = [c["dir"] / "huckel-NP_r.dat" for c in (up, dn)]
    rmatrix = None
    if all(f.is_file() or Path(str(f) + ".gz").is_file() for f in rfiles):
        from position_matrix import read_rmatrix, write_rmatrix
        (_, Xu), (_, Xd) = read_rmatrix(rfiles[0]), read_rmatrix(rfiles[1])
        Rs = sorted(set(Xu) | set(Xd), key=lambda R: (sum(map(abs, R)), R))
        zero = np.zeros((n, n, 3))
        write_rmatrix(out / "huckel-SP_r.dat", Rs, [np.array([Xu.get(R, zero) for R in Rs]),
                                                    np.array([Xd.get(R, zero) for R in Rs])], "two spin channels")
        rmatrix = "huckel-SP_r.dat"
        print("RMATRIX huckel-SP_r.dat (from the huckel-NP_r.dat of the two channels)", flush=True)
    shutil.copy(up["dir"] / "parameters_huckel.dat", out / "parameters_huckel_up.dat")
    shutil.copy(dn["dir"] / "parameters_huckel.dat", out / "parameters_huckel_down.dat")
    for fname in ("orbitals.dat", "POSCAR"):
        if (up["dir"] / fname).is_file():
            shutil.copy(up["dir"] / fname, out / fname)
    summary = dict(spin_up_fit=os.path.relpath(up["dir"].resolve(), out), spin_down_fit=os.path.relpath(dn["dir"].resolve(), out),
                   n_orbitals_per_spin=n, n_cells=len(cells),
                   shift_up_eV=up["shift"], shift_down_eV=dn["shift"],
                   K_EHT_up=up["summary"].get("K_EHT"), K_EHT_down=dn["summary"].get("K_EHT"),
                   energy_zero="DFT Fermi level; each channel shifted by (DFT midgap of the channel - E_F); 0 for a metal (already at E_F)",
                   spinor_basis="2n functions: orbitals 1..n spin up, n+1..2n spin down; up-down blocks zero",
                   position_matrix=rmatrix)
    if not a.no_native:
        try:
            summary["native_spin"] = native_spin(up, dn, out, a.mesh_radius, align=not a.no_align)
        except Exception as exc:          # the joined file is already written; report, do not fail
            summary["native_spin"] = {"error": str(exc)}
        ns = summary["native_spin"]
        if "fitted_tables" in ns:
            hs = ns["huckel_SP_electrons_below_EF"]
            for key in ("fitted_tables", "aligned_tables"):
                if key not in ns:
                    continue
                r = ns[key]
                dev = max(c["max_deviation_within_3eV_eV"] for c in r["bands_vs_huckel_SP"].values())
                print(f"NATIVE_SPIN {key}: N_up {r['electrons_up']}, N_down {r['electrons_down']}, moment "
                      f"{r['magnetic_moment_per_cell']} per cell; band shape vs huckel-SP.dat within {dev:.1e} eV "
                      f"(+-3 eV of E_F)", flush=True)
            if "alignment" in ns:
                print(f"NATIVE_SPIN alignment: Hii shifted by {ns['alignment']['delta_Hii_up_eV']:+.4f} eV (up), "
                      f"{ns['alignment']['delta_Hii_down_eV']:+.4f} eV (down); "
                      f"huckel-SP.dat at E_F = 0: N_up {hs['up']:.3f}, N_down {hs['down']:.3f}, moment {hs['moment']:.3f}",
                      flush=True)
        else:
            print("NATIVE_SPIN", ns, flush=True)
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.txt").write_text(
        "Spin-polarized EHT Hamiltonian (two spin-channel fits joined)\n"
        "huckel-SP.dat    WanTiBEXOS 'SP' file: SP / scissors / Fermi level (0) / 3 lattice vectors [A] /\n"
        "                 2n orbitals / n cells / R_x R_y R_z [A] i j ReH ImH S; orbitals 1..n spin up,\n"
        "                 n+1..2n spin down; up-down blocks zero; energies relative to the DFT Fermi level\n"
        "basis_set-SP     WanTiBEXOS basis file (spin column 1 / -1)\n"
        "huckel-SP_r.dat  position matrix <m,0|r|n,R> (A) of the two channels (WanTiBEXOS BSE_CENTER_FILE),\n"
        "                 when the channel fits wrote huckel-NP_r.dat\n"
        "parameters_huckel_up.dat / _down.dat   the EHT parameters of each channel\n")
    print("SPIN_COMBINE_DONE", out / "huckel-SP.dat", f"2n = {2 * n}, cells = {len(cells)}",
          f"shifts up {up['shift']:+.3f} eV, down {dn['shift']:+.3f} eV")


if __name__ == "__main__":
    main()
