"""Joint EHT fit of the two spin channels of a collinear spin-polarized DFT run, with ONE energy zero.

Fitting the channels separately (bands_fit.py --spin-channel up / down) aligns each channel with its
own constant, so the absolute EHT energies of the two tables differ by an arbitrary amount: their
common Fermi level, electron counts and magnetic moment are not physical, and shifting the tables
afterwards (spin_combine.py) distorts the bands.

Here both tables are fitted at once, and every candidate is evaluated with the native
spin-polarized bind of the fork github.com/ac-dias/yaehmop (--parm-up / --parm-down):

1. one bind run on the band path gives the EHT bands of both channels;
2. one bind run on a Gamma-centred k mesh (--common-fermi) gives H(k), S(k) of both channels; their
   eigenvalues give the common Fermi level E_F of the MODEL, with the smearing of the DFT run
   (Methfessel-Paxton order 1, 0.1 eV by default), as a smooth function of the parameters. bind's own
   common Fermi level (a T = 0 count, which jumps on a finite mesh) is recorded as a check;
3. each channel is compared with its DFT bands (relative to the DFT Fermi level) with the 'metal'
   objective of band_objective.metal_error, with the shift FIXED at -E_F (no free constant per
   channel): the two channels share one energy zero, and their relative position (exchange
   splitting), electron counts and moment come out of the model;
4. optional term w_moment * |m_model - m_DFT| (magnetic moment per cell, Bohr magnetons).

Parameters: Hii and Slater exponents of each channel (shared exponents with shared_zetas), one
Wolfsberg-Helmholz constant K for both. Candidates whose S(k) is not positive definite enough on the
mesh (smallest eigenvalue below overlap_min_eig) are rejected.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy, re, shutil, threading
import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc

from band_objective import metal_error
from fit_from_image import build_specs
from yaehmop_generic import (run_bind_args, parse_band, read_matrix_dump, write_parameter_file,
                             write_periodic_input_generic, gamma_mesh)
from fit_sparse_window import write_sample_input

CHANNELS = ("up", "down")
SMEARINGS = ("mp1", "gaussian", "fermi")


def occupation(E, mu, sigma, kind="mp1"):
    """Occupation of states at E (one electron each) for the Fermi level mu."""
    x = (np.asarray(E) - mu) / sigma
    if kind == "gaussian":
        return 0.5 * erfc(x)
    if kind == "fermi":
        return 0.5 * (1.0 - np.tanh(0.5 * x))
    if kind == "mp1":           # Methfessel-Paxton order 1 (VASP ISMEAR = 1)
        return 0.5 * erfc(x) - x * np.exp(-x * x) / (2.0 * np.sqrt(np.pi))
    raise ValueError(f"smearing must be one of {SMEARINGS}")


def common_fermi(eig_up, eig_dn, nelec, sigma=0.1, kind="mp1"):
    """Common Fermi level of the two channels on an equal-weight k mesh: N_up + N_down = nelec.
    Returns (E_F, N_up, N_down, moment = N_up - N_down)."""
    nk = len(eig_up)
    def count(mu):
        return (occupation(eig_up, mu, sigma, kind).sum() + occupation(eig_dn, mu, sigma, kind).sum()) / nk
    lo = min(eig_up.min(), eig_dn.min()) - 10 * sigma
    hi = max(eig_up.max(), eig_dn.max()) + 10 * sigma
    mu = brentq(lambda m: count(m) - nelec, lo, hi, xtol=1e-9)
    nu = occupation(eig_up, mu, sigma, kind).sum() / nk
    nd = occupation(eig_dn, mu, sigma, kind).sum() / nk
    return float(mu), float(nu), float(nd), float(nu - nd)


def generalized_eigenvalues(H, S):
    """Eigenvalues of H(k) c = E S(k) c for a stack of k points, and the smallest eigenvalue of S."""
    smin = float(np.linalg.eigvalsh(S).min())
    L = np.linalg.cholesky(S)
    Li = np.linalg.inv(L)
    return np.linalg.eigvalsh(Li @ H @ np.conj(np.swapaxes(Li, 1, 2))), smin


def read_dft_moment(dft_dir, code):
    """Magnetic moment per cell of the DFT run, when the files give it (VASP: last 'mag=' of OSZICAR)."""
    dft_dir = Path(dft_dir)
    if code == "vasp":
        for f in (dft_dir / "OSZICAR", dft_dir / "scf" / "OSZICAR"):
            if f.is_file():
                v = re.findall(r"mag=\s*(\S+)", f.read_text())
                if v:
                    return float(v[-1]), str(f)
    return None, None


class SpinJointFitter:
    """Objective of the joint fit; the interface (objective, objective_many, stock, bounds, names)
    is the one global_optimizers.run_global uses."""

    def __init__(self, samples, tables, K0, wset, *, bind, nelec, mesh_radius=7, smearing="mp1",
                 sigma=0.1, moment_ref=None, w_moment=0.0, shared_zetas=False, reg_weight=0.04,
                 overlap_min_eig=1e-3, workdir, parallel=1, hii_window=5.0, zeta_window=None, k_window=0.6,
                 k_per_channel=False):
        self.s = samples                      # {"up": sample, "down": sample} (fit_sparse_window format)
        s0 = samples["up"]
        self.elements = list(s0["elements"])
        self.bind, self.nelec = bind, nelec
        # objective settings: one WindowSettings for both channels, or {"up": ..., "down": ...}
        self.wset = wset if isinstance(wset, dict) else {ch: wset for ch in CHANNELS}
        self.smearing, self.sigma = smearing, float(sigma)
        self.moment_ref, self.w_moment = moment_ref, float(w_moment)
        self.shared_zetas, self.reg_w = bool(shared_zetas), float(reg_weight)
        self.overlap_min_eig = float(overlap_min_eig)
        self.mesh = gamma_mesh(s0["periodic_dim"], mesh_radius)
        self.mesh_radius = int(mesh_radius)
        self.work = Path(workdir).resolve(); self.work.mkdir(parents=True, exist_ok=True)
        self.parallel = max(1, int(parallel))
        self.tables = {ch: {el: copy.deepcopy(tables[ch][el]) for el in self.elements} for ch in CHANNELS}
        specs, x0, bounds, scales = [], [], [], []
        for ch in CHANNELS:
            sp, xs, bd, sc = build_specs(self.tables[ch], self.elements)
            for spec, v, b, c in zip(sp, xs, bd, sc):
                el, orb, kind = spec
                if kind == "K":
                    continue                      # one K for both channels, added below
                if ch == "down" and self.shared_zetas and kind != "Hii":
                    continue                      # exponents follow the spin-up ones
                if kind == "Hii":
                    b = (v - hii_window, v + hii_window)
                elif zeta_window is not None and kind in ("zeta1", "zeta2"):
                    # refining earlier fits: exponents within +-zeta_window (relative) of the start
                    b = (max(b[0], v * (1 - zeta_window)), min(b[1], v * (1 + zeta_window)))
                specs.append((ch, el, orb, kind)); x0.append(v); bounds.append(b); scales.append(c)
        # one K for both channels, or one per channel (k_per_channel; GLOBAL = spin up)
        K0 = K0 if isinstance(K0, dict) else {ch: float(K0) for ch in CHANNELS}
        for ch in (CHANNELS if k_per_channel else ("up",)):
            specs.append(("GLOBAL" if ch == "up" else "down", "", "", "K")); x0.append(float(K0[ch]))
            bounds.append((max(1.0, K0[ch] - k_window), min(2.8, K0[ch] + k_window))); scales.append(0.4)
        self.k_per_channel = bool(k_per_channel)
        self.specs, self.stock = specs, np.asarray(x0, float)
        self.bounds, self.scales = bounds, np.asarray(scales, float)
        self.names = [f"{ch}_{el}_{orb}_{kind}".replace("GLOBAL___", "GLOBAL_").replace("down___K", "down_K")
                      for ch, el, orb, kind in specs]
        self.lock = threading.Lock()
        self.calls, self.best_loss = 0, float("inf")
        self.rejected = 0
        self.progress = open(self.work.parent / "progress.csv", "w", buffering=1)
        self.progress.write("call,loss,best\n")

    # ---------------- parameters ----------------
    def tables_from_x(self, x):
        t = {ch: copy.deepcopy(self.tables[ch]) for ch in CHANNELS}
        K = {}
        for (ch, el, orb, kind), v in zip(self.specs, np.asarray(x, float)):
            if kind == "K":
                K["down" if ch == "down" else "up"] = float(v)
                continue
            targets = [ch] if not (self.shared_zetas and kind != "Hii") else list(CHANNELS)
            for c in targets:
                if kind == "ratio":
                    t[c][el][orb]["c1"] = 1.0; t[c][el][orb]["c2"] = float(v)
                else:
                    t[c][el][orb][kind] = float(v)
        for c in CHANNELS:                       # double zeta: the first exponent stays the compact one
            for el, sp in t[c].items():
                for orb in ("s", "p", "d", "f"):
                    if orb in sp and int(sp[orb].get("nzeta", 1)) == 2 and sp[orb]["zeta1"] <= sp[orb]["zeta2"] + 0.12:
                        raise ValueError("invalid double-zeta parameters")
        K.setdefault("down", K["up"])           # one K for both channels
        return t, K

    # ---------------- bind runs ----------------
    def run(self, x, tag, mesh=None):
        """Both channels with the native spin-polarized bind: bands on the path, and H(k), S(k) on the
        mesh. Returns dict(bands={ch: (nk, norb)}, eig={ch: (nmesh, norb)}, smin, bind_fermi)."""
        tables, K = self.tables_from_x(x)
        d = self.work / tag
        d.mkdir(parents=True, exist_ok=True)
        s = self.s["up"]
        try:
            write_parameter_file(d / "up.dat", tables["up"])
            write_parameter_file(d / "down.dat", tables["down"])
            write_sample_input(s, d / "path.bind", K["up"])
            write_periodic_input_generic(s["cif"], d / "mesh.bind", None, K["up"], True, self.nelec,
                                         overlap_images=s["overlap_images"], periodic_dim=s["periodic_dim"],
                                         structure_format=s.get("structure_format"),
                                         kpoints=self.mesh if mesh is None else mesh, average_properties=True)
            args = ["--parm-up", "up.dat", "--parm-down", "down.dat",
                    "--const-up", f"{K['up']:.12g}", "--const-down", f"{K['down']:.12g}"]
            run_bind_args(self.bind, args, "path.bind", None, d / "path.log", cwd=d)
            run_bind_args(self.bind, args + ["--common-fermi"], "mesh.bind", None, d / "mesh.log", cwd=d)
            out = dict(bands={}, eig={}, smin=np.inf)
            for ch in CHANNELS:
                _, out["bands"][ch] = parse_band(str(d / f"path_{ch}.bind.band"))
                H = read_matrix_dump(str(d / f"mesh_{ch}.bind.HAM"))
                S = read_matrix_dump(str(d / f"mesh_{ch}.bind.OV"))
                try:
                    out["eig"][ch], smin = generalized_eigenvalues(H, S)
                except np.linalg.LinAlgError:
                    out["eig"][ch], smin = None, -1.0
                out["smin"] = min(out["smin"], smin)
            v = re.findall(r"^#Fermi_Energy:\s*(\S+)", (d / "mesh.bind.spin").read_text(), re.M)
            out["bind_fermi"] = float(v[-1]) if v else None
            out["tables"], out["K"] = tables, K
            return out
        finally:
            shutil.rmtree(d, ignore_errors=True)

    # ---------------- objective ----------------
    def scores(self, r):
        """Metrics of one evaluated candidate (the dict returned by run)."""
        mu, nu, nd, m = common_fermi(r["eig"]["up"], r["eig"]["down"], self.nelec, self.sigma, self.smearing)
        per = {}
        for ch in CHANNELS:
            s = self.s[ch]
            E = r["bands"][ch][s["kmask"]]
            if E.shape != (len(s["kx"]), s["norb"]):
                raise ValueError("physical EHT spectrum shape mismatch")
            per[ch] = metal_error(s["target_all"], E, s["kx"], [x for x, _ in s["ticks"]], self.wset[ch], shift=-mu)
        return mu, nu, nd, m, per

    def evaluate(self, x, call):
        r = self.run(x, f"trial_{call:07d}")
        if r["smin"] < self.overlap_min_eig:
            raise ValueError(f"overlap S(k) not positive definite enough (min eigenvalue {r['smin']:.2e})")
        mu, nu, nd, m, per = self.scores(r)
        cost = float(np.mean([per[ch]["total"] for ch in CHANNELS]))
        line = dict(K_up=r["K"]["up"], K_down=r["K"]["down"], eht_fermi_eV=mu, electrons_up=nu, electrons_down=nd, moment=m,
                    bind_fermi_eV=r["bind_fermi"], overlap_min_eig=r["smin"])
        for ch in CHANNELS:
            line[f"{ch}_total"] = per[ch]["total"]
            line[f"{ch}_window_rms_eV"] = per[ch]["window_rms_eV"]
            line[f"{ch}_rms_1eV_EF"] = per[ch]["rms_within_1eV_of_EF_eV"]
        if self.moment_ref is not None and self.w_moment > 0:
            line["moment_term"] = self.w_moment * abs(m - self.moment_ref)
            cost += line["moment_term"]
        reg = float(np.sqrt(np.mean(((np.asarray(x) - self.stock) / self.scales) ** 2)))
        line["regularization"] = reg
        return cost + self.reg_w * reg, line

    def _log(self, call, cost):
        with self.lock:
            self.best_loss = min(self.best_loss, cost)
            self.progress.write(f"{call},{cost:.6g},{self.best_loss:.6g}\n")

    def next_call(self):
        with self.lock:
            self.calls += 1
            return self.calls

    def objective(self, x, eid=None, stage=None):
        call = self.next_call()
        try:
            cost, line = self.evaluate(x, call)
        except Exception as exc:
            with self.lock:
                self.rejected += 1
            self._log(call, 1e4)
            return 1e4, {"error": str(exc)}
        self._log(call, cost)
        return cost, line

    def objective_many(self, xs, eids=None, stage=None):
        with ThreadPoolExecutor(max(1, min(self.parallel, len(xs)))) as pool:
            return list(pool.map(lambda x: self.objective(x), xs))

    def close(self):
        self.progress.close()
