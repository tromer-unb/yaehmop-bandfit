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
"""
from __future__ import annotations
from pathlib import Path
import argparse, gzip, json, os, shutil
import numpy as np


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--up", type=Path, required=True, help="fitting/ folder of the spin-up fit")
    ap.add_argument("--down", type=Path, required=True, help="fitting/ folder of the spin-down fit")
    ap.add_argument("--outdir", type=Path, required=True)
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
