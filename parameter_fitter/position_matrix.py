"""Position matrix of the Slater (EHT) basis for WanTiBEXOS (BSE_CENTER_FILE): huckel-*_r.dat.

    X_ij(R) = <phi_i, cell 0 | r | phi_j, cell R>      (Angstrom, absolute coordinates)

written in the layout of Wannier90's seedname_r.dat, as siesta2wtb.py --rmatrix and the YAeHMOP
interface yaehmop2wtb.py --rmatrix (github.com/ac-dias/yaehmop, interface/) write it: a comment line,
the basis size, the number of R vectors, then 'R1 R2 R3 m n Re(x) Im(x) Re(y) Im(y) Re(z) Im(z)'
with m (the home-cell orbital) running fastest. r is spin diagonal: for SOC/SP the up-down blocks
are zero.

The orbitals are the Slater functions YAeHMOP uses for its overlaps, from the parameter table:
c1 r^(n-1) e^(-z1 r) [+ c2 r^(n-1) e^(-z2 r)] times real spherical harmonics (YAeHMOP's order:
s; px py pz; dx2-y2 dz2 dxy dxz dyz; the seven real f), the coefficients normalized as bind does
(double zeta: c1^2 + c2^2 + 2 c1 c2 S12 = 1; s and single-zeta p: c = 1; single-zeta d/f: c1), the
exponents (bohr^-1) converted with the bohr of YAeHMOP's overlaps (AUI = 1.889644746 bohr/A,
R_overlap_mat.c). The integrals are numerical: bipolar coordinates (r_a, r_b, phi) for two
centres, spherical for one; phi and cos(theta) rules exact for the angular part, Gauss-Legendre in
r on pieces graded near the nuclei and split at the kinks r_a = d, r_a = Rc(B) - d. The same
quadrature gives S, compared with YAeHMOP's S(R) (the returned check, ~1e-7 with 6-decimal tables).
"""
from __future__ import annotations
from math import factorial
from pathlib import Path
import numpy as np

BOHR = 1 / 1.889644746
SHELLS = {"s": ["s"], "p": ["px", "py", "pz"],
          "d": ["dx2-y2", "dz2", "dxy", "dxz", "dyz"],
          "f": ["fz3", "fxz2", "fyz2", "fxyz", "fz(x2-y2)", "fx(x2-3y2)", "fy(3x2-y2)"]}


def slater_functions(shell, orb):
    """[(c, zeta)] of one shell of a parameter table (parse_parameter_table), normalized as bind."""
    n, nz = int(shell["n"]), int(shell.get("nzeta", 1))
    z1, z2 = float(shell["zeta1"]), float(shell.get("zeta2", 0.0))
    c1, c2 = float(shell.get("c1", 1.0)), float(shell.get("c2", 0.0))
    if nz == 2 and c2 != 0.0:
        s12 = (4 * z1 * z2 / (z1 + z2) ** 2) ** (n + 0.5)
        norm = np.sqrt(c1 * c1 + c2 * c2 + 2 * s12 * c1 * c2)
        return [(c1 / norm, z1), (c2 / norm, z2)]
    return [(1.0 if orb in "sp" else c1, z1)]


def _ylm(orb, v, r):
    x, y, z = (v[:, i] / r for i in range(3))
    pi = np.pi
    return {"s": 0 * x + np.sqrt(1 / (4 * pi)),
            "px": np.sqrt(3 / (4 * pi)) * x, "py": np.sqrt(3 / (4 * pi)) * y, "pz": np.sqrt(3 / (4 * pi)) * z,
            "dx2-y2": np.sqrt(15 / (16 * pi)) * (x * x - y * y), "dz2": np.sqrt(5 / (16 * pi)) * (3 * z * z - 1),
            "dxy": np.sqrt(15 / (4 * pi)) * x * y, "dxz": np.sqrt(15 / (4 * pi)) * x * z,
            "dyz": np.sqrt(15 / (4 * pi)) * y * z,
            "fz3": np.sqrt(7 / (16 * pi)) * z * (5 * z * z - 3),
            "fxz2": np.sqrt(21 / (32 * pi)) * x * (5 * z * z - 1),
            "fyz2": np.sqrt(21 / (32 * pi)) * y * (5 * z * z - 1),
            "fxyz": np.sqrt(105 / (4 * pi)) * x * y * z,
            "fz(x2-y2)": np.sqrt(105 / (16 * pi)) * z * (x * x - y * y),
            "fx(x2-3y2)": np.sqrt(35 / (32 * pi)) * x * (x * x - 3 * y * y),
            "fy(3x2-y2)": np.sqrt(35 / (32 * pi)) * y * (3 * x * x - y * y)}[orb]


class Atom:
    """Slater orbitals of one element (Angstrom units) and their radius Rc (|R(r)| r < tol beyond it)."""

    def __init__(self, element_table, tol=1e-7):
        self.orbs, self.radial, rc = [], [], 0.0
        for sh in "spdf":
            if sh not in element_table:
                continue
            n = int(element_table[sh]["n"])
            stos = [(c, (2 * z / BOHR) ** (n + 0.5) / np.sqrt(factorial(2 * n)), z / BOHR)
                    for c, z in slater_functions(element_table[sh], sh)]
            for orb in SHELLS[sh]:
                self.orbs.append(orb)
                self.radial.append((n, stos))
            r = np.linspace(0.01, 60.0, 6000)
            f = np.abs(sum(c * N * r ** (n - 1) * np.exp(-z * r) for c, N, z in stos)) * r
            rc = max(rc, r[np.nonzero(f > tol)[0][-1]] if np.any(f > tol) else 1.0)
        self.rc = rc

    def values(self, v):
        r = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
        return np.array([sum(c * N * r ** (n - 1) * np.exp(-z * r) for c, N, z in stos) * _ylm(orb, v, r)
                         for orb, (n, stos) in zip(self.orbs, self.radial)])


def _gauss(lo, hi, h, n, cuts=()):
    e = np.concatenate([0.05 * 2.0 ** np.arange(int(np.log2(h / 0.05)) + 1),
                        np.arange(h, hi + h, h), np.asarray(cuts, float)])
    e = np.unique(np.concatenate([[lo, hi], e[(e > lo + 1e-10) & (e < hi - 1e-10)]]))
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (e[1:] - e[:-1])[:, None]
    return (half * x + 0.5 * (e[1:] + e[:-1])[:, None]).ravel(), (half * w).ravel()


def _quadrature(A, B, Ra, Rb, h, n, nphi=12):
    phi = 2 * np.pi * np.arange(nphi) / nphi
    d = np.linalg.norm(B - A)
    if d < 1e-8:
        r, wr = _gauss(0.0, min(Ra, Rb), h, n)
        ct, wt = np.polynomial.legendre.leggauss(10)
        st = np.sqrt(1 - ct ** 2)
        u = np.stack(np.broadcast_arrays(st[:, None] * np.cos(phi), st[:, None] * np.sin(phi),
                                         ct[:, None] + 0 * phi), -1).reshape(-1, 3)
        w = (wt[:, None] + 0 * phi).ravel() * 2 * np.pi / nphi
        return (A + r[:, None, None] * u[None]).reshape(-1, 3), (wr[:, None] * r[:, None] ** 2 * w).ravel()
    ez = (B - A) / d
    t = np.array([1.0, 0, 0]) if abs(ez[0]) < 0.9 else np.array([0, 1.0, 0])
    ex = t - ez * (t @ ez)
    ex /= np.linalg.norm(ex)
    circle = np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * np.cross(ez, ex)
    P, W = [], []
    ra, wa = _gauss(0.0, Ra, h, n, (d, Rb - d))     # kinks of the r_b range
    for x, wx in zip(ra, wa):
        lo, hi = abs(x - d), min(x + d, Rb)
        if hi - lo < 1e-10:
            continue
        rb, wb = _gauss(lo, hi, h, n)
        z = (x * x - rb * rb + d * d) / (2 * d)
        rho = np.sqrt(np.clip(x * x - z * z, 0, None))
        P.append((A + z[:, None, None] * ez + rho[:, None, None] * circle).reshape(-1, 3))
        W.append(np.repeat(wx * x * rb * wb / d * 2 * np.pi / nphi, nphi))
    return np.concatenate(P), np.concatenate(W)


_JOB = {}


def _cells(job_idx):
    """X and Sq of the cells job_idx (worker of position_matrix)."""
    j = _JOB
    sites, first, cell, pos, Rs, Sref, h, n, thr = (j[k] for k in
        ("sites", "first", "cell", "pos", "Rs", "Sref", "h", "n", "thr"))
    out = []
    for iR in job_idx:
        T = np.asarray(Rs[iR], float) @ cell
        no = first[-1]
        X, Sq = np.zeros((no, no, 3)), np.zeros((no, no))
        for a, A in enumerate(pos):
            ia = slice(first[a], first[a + 1])
            for b, B0 in enumerate(pos):
                ib = slice(first[b], first[b + 1])
                if np.abs(np.asarray(Sref[iR])[ia, ib]).max() <= thr:
                    continue                      # no overlap in YAeHMOP (or below thr): X ~ 0
                B = B0 + T
                if np.linalg.norm(B - A) >= sites[a].rc + sites[b].rc:
                    continue
                pts, w = _quadrature(A, B, sites[a].rc, sites[b].rc, h, n)
                fa = sites[a].values(pts - A) * w
                fb = sites[b].values(pts - B)
                Sq[ia, ib] = fa @ fb.T
                for c in range(3):
                    X[ia, ib, c] = (fa * pts[:, c]) @ fb.T
        out.append((iR, X, Sq))
    return out


def position_matrix(positions, symbols, table, cell, Rs, Sref, h=2.0, n=10, threshold=1e-8, nproc=None):
    """X[iR, i, j, :] and the overlap of the same quadrature, Sq[iR, i, j], for the atom pairs whose
    block of YAeHMOP's S(R) (Sref[iR], n_orb x n_orb) has an element above threshold.
    positions: Cartesian (A), in the order of the basis, in the frame of the YAeHMOP run (see
    yaehmop_generic.yaehmop_frame); Rs: integer cell vectors; cell: lattice vectors (rows, A).

    Only one of each pair R, -R is integrated: X_ij(-R) = X_ji(R) - T_R S_ji(R) (r absolute),
    S_ij(-R) = S_ji(R). The cells are shared among nproc processes (default: the CPUs)."""
    import os
    sites = [Atom(table[s.capitalize()]) for s in symbols]
    first = np.cumsum([0] + [len(a.orbs) for a in sites])
    no = first[-1]
    cell = np.asarray(cell, float)
    Rs = [tuple(int(v) for v in R) for R in Rs]
    index = {R: i for i, R in enumerate(Rs)}
    todo, mirror = [], {}
    for i, R in enumerate(Rs):
        mR = tuple(-v for v in R)
        if mR in index and mR != R and index[mR] in mirror.values():
            continue
        if mR in index and mR != R:
            mirror[index[mR]] = i                # -R is derived from R
        todo.append(i)
    _JOB.update(sites=sites, first=first, cell=cell, pos=np.asarray(positions, float), Rs=Rs, Sref=Sref,
                h=h, n=n, thr=threshold)
    if nproc is None:
        try:
            nproc = len(os.sched_getaffinity(0))
        except AttributeError:
            nproc = os.cpu_count() or 1
    nproc = max(1, min(int(nproc), len(todo)))
    chunks = [todo[k::nproc] for k in range(nproc)]
    if nproc > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(nproc) as pool:
            parts = pool.map(_cells, chunks)
    else:
        parts = [_cells(todo)]
    X, Sq = np.zeros((len(Rs), no, no, 3)), np.zeros((len(Rs), no, no))
    for part in parts:
        for iR, x, sq in part:
            X[iR], Sq[iR] = x, sq
    for iM, iR in mirror.items():                # cell -R from cell R
        T = np.asarray(Rs[iR], float) @ cell
        Sq[iM] = Sq[iR].T
        X[iM] = np.transpose(X[iR], (1, 0, 2)) - Sq[iR].T[:, :, None] * T
    _JOB.clear()
    return X, Sq


def write_rmatrix(path, Rs, Xs, source):
    """seedname_r.dat layout; Xs: one X (nR, n, n, 3) per spin block (1 for NP, 2 for SOC/SP)."""
    nspin, no = len(Xs), Xs[0].shape[1]
    zero = np.zeros(3)
    with open(path, "w") as f:
        f.write(f"yaehmop-bandfit: <m,0|r|n,R> (Angstrom) of the Slater basis of {source}\n")
        f.write(f"{nspin * no}\n{len(Rs)}\n")
        for iR, R in enumerate(Rs):
            r1, r2, r3 = (int(round(v)) for v in R)
            for s in range(nspin):
                for n_ in range(no):
                    for s2 in range(nspin):
                        for m in range(no):
                            x = Xs[s][iR, m, n_] if s2 == s else zero
                            f.write("%5d%5d%5d%5d%5d" % (r1, r2, r3, s2 * no + m + 1, s * no + n_ + 1)
                                    + "".join("%16.10f%16.10f" % (v, 0.0) for v in x) + "\n")


def read_rmatrix(path):
    """huckel-*_r.dat (or .gz) -> (n_basis, {R (tuple of ints): X (n, n, 3)})."""
    import gzip
    p = Path(path)
    if not p.is_file() and Path(str(p) + ".gz").is_file():
        p = Path(str(p) + ".gz")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        f.readline()
        nb = int(f.readline()); f.readline()
        data = np.loadtxt(f)
    out = {}
    for row in np.atleast_2d(data):
        R = tuple(int(v) for v in row[:3])
        X = out.setdefault(R, np.zeros((nb, nb, 3)))
        X[int(row[3]) - 1, int(row[4]) - 1] = row[5::2]
    return nb, out


def rmatrix_enabled():
    """The export writes huckel-*_r.dat unless BANDFIT_RMATRIX=0 (the --no-rmatrix options set it)."""
    import os
    return os.environ.get("BANDFIT_RMATRIX", "1").strip() not in ("0", "no", "false", "off")
