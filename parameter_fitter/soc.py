"""Spin-orbit coupling added on top of a spinless YAeHMOP Hamiltonian.

YAeHMOP has no spin-orbit term. Here it is added in the usual on-site
approximation (as in DFTB+ and in tight-binding models):

    H_SOC = sum over atoms A and shells l >= 1 of  lambda_{A,l} L.S

acting only between orbitals of the same shell of the same atom. The spinor
basis has 2n functions ordered as all n orbitals with spin up, then the same n
orbitals with spin down. With S = sigma/2,

    up-up   = +(lambda/2) Lz        dn-dn = -(lambda/2) Lz
    up-dn   = (lambda/2) (Lx - i Ly) dn-up = (lambda/2) (Lx + i Ly)

H(k) and S(k) of YAeHMOP become H(k) (x) I2 + H_SOC and S(k) (x) I2 (the
on-site overlap of different orbitals of one atom is zero, so H_SOC needs no
overlap correction). For a p shell the levels split into j = 3/2 (+lambda/2,
4 states) and j = 1/2 (-lambda, 2 states); for d into j = 5/2 (+lambda) and
j = 3/2 (-3 lambda/2).

The real orbitals are YAeHMOP's (s; px py pz; dx2-y2 dz2 dxy dxz dyz; f in the
order of yaehmop_generic.orbital_labels), built from complex spherical harmonics
with the Condon-Shortley phase.
"""
from __future__ import annotations
import numpy as np

SHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3}
# real orbital -> (m, kind): kind 'c' = cos-like, 's' = sin-like, '0' = m = 0
_REAL = {
    "px": (1, "c"), "py": (1, "s"), "pz": (0, "0"),
    "dx2-y2": (2, "c"), "dz2": (0, "0"), "dxy": (2, "s"), "dxz": (1, "c"), "dyz": (1, "s"),
    "fz3": (0, "0"), "fxz2": (1, "c"), "fyz2": (1, "s"), "fxyz": (2, "s"), "fz(x2-y2)": (2, "c"),
    "fx(x2-3y2)": (3, "c"), "fy(3x2-y2)": (3, "s"),
}
SHELL_ORBITALS = {1: ["px", "py", "pz"], 2: ["dx2-y2", "dz2", "dxy", "dxz", "dyz"],
                  3: ["fz3", "fxz2", "fyz2", "fxyz", "fz(x2-y2)", "fx(x2-3y2)", "fy(3x2-y2)"]}


def real_harmonic_matrix(l, names):
    """U with real_a = sum_m U[a, m] Y_{l,m} (columns m = -l..l)."""
    U = np.zeros((len(names), 2 * l + 1), complex)
    for a, nm in enumerate(names):
        m, kind = _REAL[nm]
        if kind == "0":
            U[a, l] = 1.0
        elif kind == "c":     # (1/sqrt2) [ (-1)^m Y_{l,m} + Y_{l,-m} ]
            U[a, l + m] = (-1) ** m / np.sqrt(2); U[a, l - m] = 1 / np.sqrt(2)
        else:                 # (i/sqrt2) [ Y_{l,-m} - (-1)^m Y_{l,m} ]
            U[a, l - m] = 1j / np.sqrt(2); U[a, l + m] = -1j * (-1) ** m / np.sqrt(2)
    return U


def angular_momentum(l, names=None):
    """Lx, Ly, Lz (hbar = 1) in the real orbitals `names` of shell l."""
    names = names or SHELL_ORBITALS[l]
    m = np.arange(-l, l + 1)
    Lz = np.diag(m).astype(complex)
    Lp = np.zeros((2 * l + 1, 2 * l + 1), complex)
    for i, mm in enumerate(m[:-1]):
        Lp[i + 1, i] = np.sqrt(l * (l + 1) - mm * (mm + 1))   # L+ |m> = c |m+1>
    Lx = (Lp + Lp.conj().T) / 2
    Ly = (Lp - Lp.conj().T) / 2j
    U = real_harmonic_matrix(l, names)
    to_real = lambda L: U.conj() @ L @ U.T
    return to_real(Lx), to_real(Ly), to_real(Lz)


def soc_matrix(labels, lambdas):
    """H_SOC (2n x 2n, spin-up block first) for orbital labels 'Mo1:dxy' and
    lambdas {(element, shell): lambda_eV}, e.g. {('Mo','d'): 0.08}."""
    n = len(labels)
    H = np.zeros((2 * n, 2 * n), complex)
    groups = {}
    for i, lab in enumerate(labels):
        atom, orb = lab.split(":")
        shell = orb[0]
        if SHELL_L[shell] == 0:
            continue
        groups.setdefault((atom, shell), []).append((i, orb))
    for (atom, shell), items in groups.items():
        el = atom.rstrip("0123456789")
        lam = float(lambdas.get((el, shell), 0.0))
        if lam == 0.0:
            continue
        idx = [i for i, _ in items]; names = [o for _, o in items]
        Lx, Ly, Lz = angular_momentum(SHELL_L[shell], names)
        ix = np.ix_(idx, idx)
        H[ix] += 0.5 * lam * Lz
        H[np.ix_([n + i for i in idx], [n + i for i in idx])] += -0.5 * lam * Lz
        H[np.ix_(idx, [n + i for i in idx])] += 0.5 * lam * (Lx - 1j * Ly)
        H[np.ix_([n + i for i in idx], idx)] += 0.5 * lam * (Lx + 1j * Ly)
    return H


def soc_shells(labels):
    """(element, shell) pairs with l >= 1 present in the basis."""
    out = []
    for lab in labels:
        atom, orb = lab.split(":")
        key = (atom.rstrip("0123456789"), orb[0])
        if SHELL_L[orb[0]] >= 1 and key not in out:
            out.append(key)
    return out


def spinor(Hk, Sk, Hsoc):
    """Spinor H(k), S(k) (2n x 2n) from spinless H(k), S(k) and the on-site H_SOC."""
    n = Hk.shape[-1]
    I2 = np.eye(2)
    return np.kron(I2, Hk) + Hsoc, np.kron(I2, Sk)


def soc_bands(Hk, Sk, Hsoc):
    import scipy.linalg as sl
    out = []
    for H, S in zip(Hk, Sk):
        Hs, Ss = spinor(H, S, Hsoc)
        out.append(sl.eigh(Hs, Ss, eigvals_only=True))
    return np.array(out)


def write_wtb_soc(path, cells, cell, labels, Hsoc, e_mid, positions, symbols, basis_path):
    """WanTiBEXOS 'SOC' file (as siesta2wtb.py) and its basis file.

    cells: [(R (3 ints), H(R), S(R))] spinless real matrices. The spinor H(R) is
    H(R) (x) I2, plus H_SOC on R = 0; written as H - E_mid*S with Fermi level 0
    (wtb.x diagonalizes (H - E_F I, S)). Spin-up orbitals first, then spin-down.
    """
    n = len(labels)
    with open(path, "w") as f:
        f.write("SOC\n0.000000\n0.000000\n")
        for v in np.asarray(cell):
            f.write(f"{v[0]:.8f}  {v[1]:.8f}  {v[2]:.8f}\n")
        f.write(f"{2 * n}\n{len(cells)}\n#rcell x   rcell y   rcell z   i   j   ReH   ImH   S\n")
        for R, H, S in cells:
            Hs = np.kron(np.eye(2), H - e_mid * S).astype(complex)
            Ss = np.kron(np.eye(2), S)
            if not np.any(R):
                Hs = Hs + Hsoc
            r = np.asarray(R) @ np.asarray(cell)
            for i in range(2 * n):
                for j in range(2 * n):
                    f.write(f"{r[0]:.8f}   {r[1]:.8f}   {r[2]:.8f}   {i + 1}   {j + 1}   "
                            f"{Hs[i, j].real:.10f}   {Hs[i, j].imag:.10f}   {Ss[i, j]:.10f}\n")
    from yaehmop_generic import ORBITAL_LM
    with open(basis_path, "w") as f:
        f.write("#bindex aspecie ax ay az l m spin\n")
        for spin, off in ((1, 0), (-1, n)):
            for b, lab in enumerate(labels):
                atom, orb = lab.split(":")
                ia = int("".join(c for c in atom if c.isdigit())) - 1
                l, m = ORBITAL_LM[orb]
                p = positions[ia]
                f.write(f"{off + b + 1}   {symbols[ia]}   {p[0]:.8f}   {p[1]:.8f}   {p[2]:.8f}   {l}   {m}   {spin}\n")
