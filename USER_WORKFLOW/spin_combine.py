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


def native_spin(up, dn, out, mesh_radius=4):
    """Both channels in one native spin-polarized bind run (fork ac-dias/yaehmop):
    per-channel bands from the dumped H(k), S(k) against huckel-SP.dat (same k mesh), common
    Fermi level, electrons per channel and magnetic moment. None if the bind lacks the option."""
    from yaehmop_generic import (resolve_yaehmop_assets, bind_features, run_bind_args, gamma_mesh,
                                 write_periodic_input_generic, overlap_images_for, read_matrix_dump,
                                 parse_parameter_table, valence_electrons)
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
    keep = out / "yaehmop_spin"; keep.mkdir(exist_ok=True)
    shutil.copy(up["dir"] / "parameters_huckel.dat", keep / "up.dat")
    shutil.copy(dn["dir"] / "parameters_huckel.dat", keep / "down.dat")
    shutil.copy(poscar, keep / "POSCAR")
    Ku, Kd = float(up["summary"]["K_EHT"]), float(dn["summary"]["K_EHT"])
    args = ["--parm-up", "up.dat", "--parm-down", "down.dat", "--const-up", f"{Ku:.12g}", "--const-down", f"{Kd:.12g}",
            "--common-fermi"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in ("up.dat", "down.dat"):
            shutil.copy(keep / f, tmp / f)
        inp = write_periodic_input_generic(poscar, tmp / "spin.bind", None, Ku, True, nelec,
                                           overlap_images=overlap_images_for(dim, radius), periodic_dim=dim,
                                           structure_format="vasp", kpoints=mesh, average_properties=True)
        shutil.copy(inp, keep / "spin.bind")
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            run_bind_args(bind, args, "spin.bind", None, tmp / "spin.log")
        finally:
            os.chdir(cwd)
        spin = {}
        for line in (tmp / "spin.bind.spin").read_text().splitlines():
            if line.startswith("#") and ":" in line:
                k, v = line[1:].split(":", 1)
                try:
                    spin[k.strip()] = float(v.split()[0])
                except (ValueError, IndexError):
                    spin[k.strip()] = v.strip()
        shutil.copy(tmp / "spin.bind.spin", keep / "spin.bind.spin")
        checks = {}; counts = {}
        for tag, ch in (("up", up), ("down", dn)):
            H = read_matrix_dump(tmp / f"spin_{tag}.bind.HAM"); S = read_matrix_dump(tmp / f"spin_{tag}.bind.OV")
            import scipy.linalg as sl
            nat = np.array([sl.eigh(h, s, eigvals_only=True) for h, s in zip(H, S)])
            cell = ch["cell"]; n = ch["n"]
            frac_R = {R: np.linalg.solve(cell.T, np.array(R)) for R in ch["blocks"]}
            sp = []
            for kq in mesh:
                Hk = sum(np.exp(2j * np.pi * (kq @ frac_R[R])) * (B[0] + ch["shift"] * B[1]) for R, B in ch["blocks"].items())
                Sk = sum(np.exp(2j * np.pi * (kq @ frac_R[R])) * B[1] for R, B in ch["blocks"].items())
                sp.append(sl.eigh(Hk, Sk, eigvals_only=True))
            # bands up to 15 eV above E_F: states far above come from the near-null directions of S(k)
            # (smallest eigenvalue ~1e-3) and are very sensitive to the rounding of the written file
            ef = spin.get("Fermi_Energy", spin.get(f"Fermi_Energy_{tag}"))
            ok = nat <= (float(ef) if isinstance(ef, float) else np.median(nat)) + 15.0
            counts[tag] = float((np.array(sp) < 0).sum() / len(mesh))   # states below E_F = 0 of huckel-SP.dat
            d = (nat - np.array(sp))[ok]
            checks[tag] = dict(energy_offset_eV=float(d.mean()), max_deviation_eV=float(np.abs(d - d.mean()).max()),
                               bands_compared="up to 15 eV above the native Fermi level")
    efer = spin.get("Fermi_Energy")
    res = dict(command="bind " + " ".join(args) + " spin.bind", k_mesh=f"Gamma-centred {2 * mesh_radius + 1}^{dim}",
               common_fermi_eV=efer, electrons_up=spin.get("ELECTRONS_UP"), electrons_down=spin.get("ELECTRONS_DOWN"),
               magnetic_moment_per_cell=spin.get("MAGNETIC_MOMENT"), bands_vs_huckel_SP=checks,
               huckel_SP_electrons_below_EF={**counts, "moment": counts["up"] - counts["down"]},
               note="energy_offset_eV: native EHT energy minus huckel-SP.dat energy (constant per channel: the "
                    "energy zero of the channel); max_deviation_eV checks that both give the same bands. The native "
                    "--common-fermi run uses the ABSOLUTE EHT energies of the two tables, fitted separately; when the "
                    "two offsets differ, its Fermi level, electron counts and moment differ from those of "
                    "huckel-SP.dat, where both channels are aligned at the DFT Fermi level "
                    "(huckel_SP_electrons_below_EF, same mesh)")
    if efer is not None:
        res["common_fermi_in_huckel_SP_frame_eV"] = {t: float(efer) - c["energy_offset_eV"] for t, c in checks.items()}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--up", type=Path, required=True, help="fitting/ folder of the spin-up fit")
    ap.add_argument("--down", type=Path, required=True, help="fitting/ folder of the spin-down fit")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--no-native", action="store_true",
                    help="skip the native spin-polarized bind run (fork github.com/ac-dias/yaehmop)")
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
                   spinor_basis="2n functions: orbitals 1..n spin up, n+1..2n spin down; up-down blocks zero")
    if not a.no_native:
        try:
            summary["native_spin"] = native_spin(up, dn, out, a.mesh_radius)
        except Exception as exc:          # the joined file is already written; report, do not fail
            summary["native_spin"] = {"error": str(exc)}
        ns = summary["native_spin"]
        if "bands_vs_huckel_SP" in ns:
            dev = max(c["max_deviation_eV"] for c in ns["bands_vs_huckel_SP"].values())
            print(f"NATIVE_SPIN bind --parm-up/--parm-down: bands = huckel-SP.dat within {dev:.1e} eV; "
                  f"common Fermi {ns['common_fermi_eV']}, N_up {ns['electrons_up']}, N_down {ns['electrons_down']}, "
                  f"moment {ns['magnetic_moment_per_cell']} per cell (huckel-SP.dat at E_F = 0: "
                  f"N_up {ns['huckel_SP_electrons_below_EF']['up']:.3f}, N_down {ns['huckel_SP_electrons_below_EF']['down']:.3f})",
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
        "parameters_huckel_up.dat / _down.dat   the EHT parameters of each channel\n")
    print("SPIN_COMBINE_DONE", out / "huckel-SP.dat", f"2n = {2 * n}, cells = {len(cells)}",
          f"shifts up {up['shift']:+.3f} eV, down {dn['shift']:+.3f} eV")


if __name__ == "__main__":
    main()
