"""Band-window objective for numerical DFT references (ported from dftb_sk_gen).

Instead of only n_pairs bands around the gap, every DFT band inside an energy
window around the gap is compared with the EHT band at the same position counted
from the gap (valence 1 = top of the valence, conduction 1 = bottom of the
conduction), both aligned at their own VBM.

Priority of the bands near the Fermi level (taken in the middle of the gap,
E_F = E_mid): every state counts with g(E) = far_weight + (1 - far_weight) *
exp(-|E - E_F| / energy_scale) in all RMS terms below, so distant bands matter
little. The squared terms, each with its own weight, are

  valence / conduction   RMS of the states inside the window
  frontier               RMS weighted by exp(-|E - E_mid| / frontier_scale):
                         priority of the states closest to the gap / Fermi level
  gap_indirect, gap_direct  fundamental and direct gap errors
  vbm, cbm               EHT energy at the k of the DFT band edge, position of the
                         edge along the path, RMS of the states within edge_window
  high_symmetry          energies at the high-symmetry points
  extrema                energies at the local extrema of every band (+ position)

total = sqrt(sum_i w_i * term_i**2). All energies in eV, positions as fractions
of the path (edges) or of the segment (extrema).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import numpy as np

OBJECTIVES = ("frontier", "window", "metal")


@dataclass
class WindowSettings:
    window_min: float = -6.0   # eV relative to the DFT VBM
    window_max: float = 4.0    # eV relative to the DFT VBM
    n_cond: int = 6            # at most this many conduction bands
    energy_scale: float = 2.0  # eV: decay of g(E) away from E_F (middle of the gap)
    far_weight: float = 0.05   # floor of g(E) for distant bands
    w_valence: float = 1.0
    w_conduction: float = 1.0
    w_frontier: float = 3.0    # extra, sharper priority of the states nearest E_F
    frontier_scale: float = 1.0
    w_gap_indirect: float = 5.0
    w_gap_direct: float = 3.0
    w_vbm: float = 4.0
    w_cbm: float = 4.0
    edge_window: float = 1.0
    w_high_symmetry: float = 4.0
    w_extrema: float = 4.0     # energies at the local maxima/minima of every band
    w_position: float = 2.0    # positions of those extrema along their segment
    # Priority of named bands counted from the gap (v1 = top of the valence, c1 = bottom of
    # the conduction): "v1:10 v2:6 v3:3 c1:10 c2:6 c3:3"; w_bands multiplies their RMS term
    band_weights: str = ""
    w_bands: float = 0.0
    # Dispersion: differences between consecutive k points (dE along the path) of the
    # compared bands, g- and band-weighted; and curvature (effective mass) at the VBM/CBM
    w_slope: float = 0.0
    w_mass: float = 0.0
    # Splittings: differences between neighbouring bands (E_j+1 - E_j) at every k point, EHT vs DFT,
    # g-weighted; targets the splitting of the N copies of a band in an N-layer slab
    w_split: float = 0.0
    # Extra weight of path segments, by index along the path (0 = first segment, as in the figure
    # ticks): "0:4 3:4" multiplies the state weights g(E) of segments 0 and 3 by 4
    segment_weights: str = ""

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown objective settings: {sorted(unknown)}; valid: {sorted(names)}")
        conv = {"n_cond": int, "band_weights": str, "segment_weights": str}
        s = cls(**{k: conv.get(k, float)(v) for k, v in d.items()})
        parse_band_weights(s.band_weights)
        parse_segment_weights(s.segment_weights)
        return s

    def to_dict(self):
        return asdict(self)


def parse_band_weights(text):
    """'v1:10 v2:6 c1:10' -> {('v', 1): 10.0, ('v', 2): 6.0, ('c', 1): 10.0}."""
    out = {}
    for item in str(text or "").replace(",", " ").split():
        key, _, w = item.partition(":")
        key = key.strip().lower()
        if len(key) < 2 or key[0] not in "vc" or not key[1:].isdigit() or not w:
            raise ValueError(f"band weight {item!r}: use v1:10 c2:5 (valence/conduction band counted from the gap)")
        out[(key[0], int(key[1:]))] = float(w)
    return out


def parse_segment_weights(text):
    """'0:4 3:4' -> {0: 4.0, 3: 4.0} (segment index along the path: weight factor)."""
    out = {}
    for item in str(text or "").replace(",", " ").split():
        key, _, w = item.partition(":")
        if not key.strip().isdigit() or not w:
            raise ValueError(f"segment weight {item!r}: use 0:4 3:2 (segment index along the path:factor)")
        out[int(key)] = float(w)
    return out


def _segment_factor(xs, nodes_x, spec):
    """Weight factor of every k point from segment_weights (1 elsewhere)."""
    f = np.ones(len(xs))
    sw = parse_segment_weights(spec)
    if sw:
        for i, (a, b) in enumerate(_segments(np.asarray(xs, float), nodes_x)):
            if i in sw:
                f[a:b + 1] = np.maximum(f[a:b + 1], sw[i])   # shared end points: the larger factor
    return f


def _split_rms(A, B, inside, G):
    """g-weighted RMS of the error in the gaps between neighbouring bands (splittings)."""
    dA = A[:, 1:] - A[:, :-1]; dB = B[:, 1:] - B[:, :-1]
    W = 0.5 * (G[:, 1:] + G[:, :-1]) * (inside[:, 1:] & inside[:, :-1])
    return float(np.sqrt(np.sum(W * (dB - dA) ** 2) / np.sum(W))) if np.sum(W) > 0 else 0.0


def _segments(x, nodes_x):
    """(start, end) index ranges of the path segments between consecutive node positions."""
    idx = sorted({int(np.argmin(np.abs(x - t))) for t in nodes_x} | {0, len(x) - 1})
    return [(a, b) for a, b in zip(idx[:-1], idx[1:]) if b > a]


def _local_extrema(e, segs):
    out = []
    for a, b in segs:
        for i in range(a + 1, b):
            if e[i] > e[i - 1] and e[i] >= e[i + 1]:
                out.append((i, "max"))
            elif e[i] < e[i - 1] and e[i] <= e[i + 1]:
                out.append((i, "min"))
    return out


def window_error(ref, n_ref, eht, n_eht, x, nodes_x, s: WindowSettings):
    """ref (nk, nb_dft), eht (nk, nb_eht) on the same k points; n_*: occupied bands."""
    a = ref - ref[:, n_ref - 1].max()          # both aligned at their VBM
    b = eht - eht[:, n_eht - 1].max()
    nv = min(n_ref, n_eht)
    nc = min(ref.shape[1] - n_ref, eht.shape[1] - n_eht, s.n_cond)
    if nc < 1:
        raise ValueError("No conduction band to compare")
    A = np.hstack([a[:, n_ref - nv:n_ref], a[:, n_ref:n_ref + nc]])
    B = np.hstack([b[:, n_eht - nv:n_eht], b[:, n_eht:n_eht + nc]])
    ov = nv                                    # index of conduction 1 in A, B
    inside = (A >= s.window_min) & (A <= s.window_max)
    inside[:, ov - 1] = True                   # the frontier bands always count
    inside[:, ov] = True
    D = B - A
    gap_ref = float(A[:, ov].min()); gap_eht = float(B[:, ov].min())
    e_mid = 0.5 * gap_ref                      # Fermi level in the middle of the gap
    G = (s.far_weight + (1 - s.far_weight) * np.exp(-np.abs(A - e_mid) / max(s.energy_scale, 1e-6))) * inside
    G = G * _segment_factor(x, nodes_x, s.segment_weights)[:, None]

    def wrms(d, w):
        return float(np.sqrt(np.sum(w * d ** 2) / np.sum(w))) if np.sum(w) > 0 else 0.0
    rv = wrms(D[:, :ov], G[:, :ov])
    rc = wrms(D[:, ov:], G[:, ov:])
    wf = np.exp(-np.abs(A - e_mid) / max(s.frontier_scale, 1e-6)) * inside
    rf = float(np.sqrt(np.sum(wf * D ** 2) / np.sum(wf)))
    dgap = gap_eht - gap_ref
    dgap_dir = float((B[:, ov] - B[:, ov - 1]).min() - (A[:, ov] - A[:, ov - 1]).min())
    xs = np.asarray(x, float); L = max(xs[-1] - xs[0], 1e-9)

    def edge(j, kind):
        f = np.argmax if kind == "max" else np.argmin
        i_ref, i_eht = int(f(A[:, j])), int(f(B[:, j]))
        e_at = float(B[i_ref, j] - A[i_ref, j])
        pos = abs(xs[i_ref] - xs[i_eht]) / L
        e0 = A[i_ref, j]
        near = (np.abs(A[:, j] - e0) <= s.edge_window)
        rw = float(np.sqrt(np.mean(D[near, j] ** 2))) if near.any() else 0.0
        return e_at, pos, rw

    vb = edge(ov - 1, "max"); cb = edge(ov, "min")
    segs = _segments(xs, nodes_x)
    hs_idx = sorted({int(np.argmin(np.abs(xs - t))) for t in nodes_x})
    hs = [(D[i, j], G[i, j]) for i in hs_idx for j in range(A.shape[1]) if inside[i, j]]
    rhs = wrms(np.array([h[0] for h in hs]), np.array([h[1] for h in hs])) if hs else 0.0
    ext_e, ext_p, ext_w = [], [], []
    for j in range(A.shape[1]):
        cand = _local_extrema(B[:, j], segs)
        for i, kind in _local_extrema(A[:, j], segs):
            if not inside[i, j]:
                continue
            seg = next((sg for sg in segs if sg[0] <= i <= sg[1]), (0, len(xs) - 1))
            same = [ii for ii, kk in cand if kk == kind and seg[0] <= ii <= seg[1]]
            if same:
                ii = min(same, key=lambda c: abs(xs[c] - xs[i]))
                ext_e.append(B[ii, j] - A[i, j])
                ext_p.append(abs(xs[ii] - xs[i]) / max(xs[seg[1]] - xs[seg[0]], 1e-9))
            else:
                ext_e.append(D[i, j]); ext_p.append(0.5)
            ext_w.append(G[i, j])
    rext = wrms(np.array(ext_e), np.array(ext_w)) if ext_e else 0.0
    rpos = wrms(np.array(ext_p), np.array(ext_w)) if ext_p else 0.0
    # --- named bands (v1, v2..., c1, c2...) and dispersion
    bw = parse_band_weights(s.band_weights)
    col_w = np.ones(A.shape[1])                # per-band weight used by the slope term
    per_band = {}
    nb_num, nb_den = 0.0, 0.0
    for (kind, k), w in bw.items():
        j = ov - k if kind == "v" else ov + k - 1
        if not 0 <= j < A.shape[1]:
            continue
        col_w[j] = max(col_w[j], w)
        r = float(np.sqrt(np.mean(D[:, j] ** 2)))
        per_band[f"rms_{kind}{k}_eV"] = r
        nb_num += w * r ** 2; nb_den += w
    rbands = float(np.sqrt(nb_num / nb_den)) if nb_den else 0.0
    dA = np.diff(A, axis=0); dB = np.diff(B, axis=0)
    same = np.diff(xs) > 1e-12                  # no step across a collapsed discontinuity
    Wd = 0.5 * (G[1:] + G[:-1]) * col_w[None, :] * same[:, None]
    rslope = wrms(dB - dA, Wd)

    def curvature(e, i):
        if 0 < i < len(e) - 1 and same[i - 1] and same[i]:
            h1, h2 = xs[i] - xs[i - 1], xs[i + 1] - xs[i]
            return 2 * (e[i + 1] * h1 + e[i - 1] * h2 - e[i] * (h1 + h2)) / (h1 * h2 * (h1 + h2))
        return None
    logs = []
    for j, f in ((ov - 1, np.argmax), (ov, np.argmin)):
        i = int(f(A[:, j]))
        c_ref, c_eht = curvature(A[:, j], i), curvature(B[:, j], i)
        if c_ref is not None and abs(c_ref) > 1e-6:
            logs.append(float(np.clip(np.log(c_eht / c_ref), -2, 2)) if c_eht is not None and c_ref * c_eht > 0 else 2.0)
    rmass = float(np.sqrt(np.mean(np.square(logs)))) if logs else 0.0
    terms = {
        "valence": s.w_valence * rv ** 2,
        "conduction": s.w_conduction * rc ** 2,
        "frontier": s.w_frontier * rf ** 2,
        "gap_indirect": s.w_gap_indirect * dgap ** 2,
        "gap_direct": s.w_gap_direct * dgap_dir ** 2,
        "vbm": s.w_vbm * (vb[0] ** 2 + vb[1] ** 2 + vb[2] ** 2),
        "cbm": s.w_cbm * (cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2),
        "high_symmetry": s.w_high_symmetry * rhs ** 2,
        "extrema": s.w_extrema * rext ** 2 + s.w_position * rpos ** 2,
        "bands": s.w_bands * rbands ** 2,
        "slope": s.w_slope * rslope ** 2,
        "mass": s.w_mass * rmass ** 2,
        "split": s.w_split * _split_rms(A, B, inside, G) ** 2,
    }
    window_rms = float(np.sqrt(np.mean(D[inside] ** 2)))
    near = inside & (np.abs(A - e_mid) <= 2.0)
    near_rms = float(np.sqrt(np.mean(D[near] ** 2))) if near.any() else 0.0
    gw = np.exp(-np.abs(A - e_mid)) * inside
    return dict(total=float(np.sqrt(sum(terms.values()))), window_rms_eV=window_rms,
                rms_within_2eV_of_EF_eV=near_rms, fermi_weighted_rms_eV=wrms(D, gw),
                rms_valence_eV=rv, rms_conduction_eV=rc, rms_frontier_eV=rf,
                gap_ref_eV=gap_ref, gap_eht_eV=gap_eht, gap_error_eV=dgap, direct_gap_error_eV=dgap_dir,
                vbm_energy_error_eV=vb[0], cbm_energy_error_eV=cb[0],
                vbm_position_error=vb[1], cbm_position_error=cb[1],
                rms_high_symmetry_eV=rhs, rms_extrema_eV=rext,
                n_valence_compared=int(nv), n_conduction_compared=int(nc),
                rms_named_bands_eV=rbands, rms_slope_eV=rslope, mass_log_error=rmass,
                rms_split_eV=_split_rms(A, B, inside, G),
                rms_v1v3_c1c3_eV=float(np.sqrt(np.mean(D[:, max(0, ov - 3):ov + 3] ** 2))),
                **per_band,
                **{f"term_{k}": float(v) for k, v in terms.items()})


def metal_error(ref, eht, x, nodes_x, s: WindowSettings):
    """Objective 'metal': no gap, so no VBM/CBM alignment.

    ref (nk, nb_dft): DFT energies relative to the Fermi level; eht (nk, norb): absolute
    YAeHMOP energies on the same k points. Bands are paired from the bottom (DFT band j
    <-> EHT band j). EHT has no absolute energy scale, so it is shifted by the constant
    c that minimizes the g-weighted RMS (closed form: the weighted mean of ref - eht);
    E_F of the model is then -c on the EHT scale. The window [window_min, window_max]
    is relative to E_F; g(E) = far_weight + (1-far_weight) exp(-|E|/energy_scale).
    Terms: weighted RMS, frontier (sharper decay, frontier_scale), high-symmetry points,
    local extrema (energy and position), named bands b1, b2... via band_weights
    ('v1:5' = lowest band; for metals v<j> counts from the bottom) and slope.
    """
    nb = min(ref.shape[1], eht.shape[1])
    A = np.asarray(ref[:, :nb], float)
    B0 = np.asarray(eht[:, :nb], float)
    inside = (A >= s.window_min) & (A <= s.window_max)
    if not inside.any():
        raise ValueError("no DFT band inside the metal window")
    G = (s.far_weight + (1 - s.far_weight) * np.exp(-np.abs(A) / max(s.energy_scale, 1e-6))) * inside
    G = G * _segment_factor(x, nodes_x, s.segment_weights)[:, None]
    c = float(np.sum(G * (A - B0)) / np.sum(G))
    B = B0 + c
    D = B - A

    def wrms(d, w):
        return float(np.sqrt(np.sum(w * d ** 2) / np.sum(w))) if np.sum(w) > 0 else 0.0
    rw = wrms(D, G)
    wf = np.exp(-np.abs(A) / max(s.frontier_scale, 1e-6)) * inside
    rf = wrms(D, wf)
    xs = np.asarray(x, float)
    segs = _segments(xs, nodes_x)
    hs_idx = sorted({int(np.argmin(np.abs(xs - t))) for t in nodes_x})
    hs = [(D[i, j], G[i, j]) for i in hs_idx for j in range(nb) if inside[i, j]]
    rhs = wrms(np.array([h[0] for h in hs]), np.array([h[1] for h in hs])) if hs else 0.0
    ext_e, ext_p, ext_w = [], [], []
    for j in range(nb):
        cand = _local_extrema(B[:, j], segs)
        for i, kind in _local_extrema(A[:, j], segs):
            if not inside[i, j]:
                continue
            seg = next((sg for sg in segs if sg[0] <= i <= sg[1]), (0, len(xs) - 1))
            same = [ii for ii, kk in cand if kk == kind and seg[0] <= ii <= seg[1]]
            if same:
                ii = min(same, key=lambda q: abs(xs[q] - xs[i]))
                ext_e.append(B[ii, j] - A[i, j])
                ext_p.append(abs(xs[ii] - xs[i]) / max(xs[seg[1]] - xs[seg[0]], 1e-9))
            else:
                ext_e.append(D[i, j]); ext_p.append(0.5)
            ext_w.append(G[i, j])
    rext = wrms(np.array(ext_e), np.array(ext_w)) if ext_e else 0.0
    rpos = wrms(np.array(ext_p), np.array(ext_w)) if ext_p else 0.0
    col_w = np.ones(nb); per_band = {}; num = den = 0.0
    for (kind, k), w in parse_band_weights(s.band_weights).items():
        j = k - 1
        if 0 <= j < nb:
            col_w[j] = max(col_w[j], w)
            m = inside[:, j]
            r = float(np.sqrt(np.mean(D[m, j] ** 2))) if m.any() else 0.0
            per_band[f"rms_band{k}_eV"] = r
            num += w * r ** 2; den += w
    rbands = float(np.sqrt(num / den)) if den else 0.0
    same = np.diff(xs) > 1e-12
    Wd = 0.5 * (G[1:] + G[:-1]) * col_w[None, :] * same[:, None]
    rslope = wrms(np.diff(B, axis=0) - np.diff(A, axis=0), Wd)
    terms = {
        "weighted": s.w_valence * rw ** 2,
        "frontier": s.w_frontier * rf ** 2,
        "high_symmetry": s.w_high_symmetry * rhs ** 2,
        "extrema": s.w_extrema * rext ** 2 + s.w_position * rpos ** 2,
        "bands": s.w_bands * rbands ** 2,
        "slope": s.w_slope * rslope ** 2,
        "split": s.w_split * _split_rms(A, B, inside, G) ** 2,
    }
    near = inside & (np.abs(A) <= 2.0)
    near1 = inside & (np.abs(A) <= 1.0)
    return dict(total=float(np.sqrt(sum(terms.values()))), shift_eV=c, eht_fermi_eV=-c,
                window_rms_eV=float(np.sqrt(np.mean(D[inside] ** 2))),
                rms_within_2eV_of_EF_eV=float(np.sqrt(np.mean(D[near] ** 2))) if near.any() else 0.0,
                rms_within_1eV_of_EF_eV=float(np.sqrt(np.mean(D[near1] ** 2))) if near1.any() else 0.0,
                max_abs_in_window_eV=float(np.abs(D[inside]).max()),
                fermi_weighted_rms_eV=rf, g_weighted_rms_eV=rw,
                rms_high_symmetry_eV=rhs, rms_extrema_eV=rext, rms_named_bands_eV=rbands,
                rms_slope_eV=rslope, rms_split_eV=_split_rms(A, B, inside, G), n_bands_compared=int(nb),
                n_bands_in_window=int(inside.any(axis=0).sum()), **per_band,
                **{f"term_{k}": float(v) for k, v in terms.items()})


# --------------------------------------------------------------------------
# Larger basis: shells YAeHMOP accepts but the stock table lacks
# --------------------------------------------------------------------------
_EXTRA_SHELLS = {
    # filled 3d shell of Zn (the stock table has s, p only but counts 12 electrons):
    # double-zeta starting guess, refined by the fit
    ("Zn", "d"): dict(nzeta=2, n=3, Hii=-17.30, zeta1=5.95, zeta2=2.30, c1=0.5933, c2=0.5744),
}


def extra_shell(el, orb, spec):
    """Starting parameters of an added shell (fitted afterwards).

    Polarization d on s/p elements: n of the p shell (3d for 2p elements), Hii a
    few eV above the p level, zeta 0.8 x zeta(p). Polarization p on H/alkalis:
    n = n(s)+1 (2p for H), Hii 3 eV above s, zeta 0.8 x zeta(s).
    """
    if (el, orb) in _EXTRA_SHELLS:
        return dict(_EXTRA_SHELLS[(el, orb)])
    if orb == "d":
        if "p" not in spec:
            raise ValueError(f"Cannot add d to {el}: no p shell to start from")
        p = spec["p"]
        n = p["n"] + 1 if p["n"] == 2 else p["n"]
        return dict(nzeta=1, n=n, Hii=p["Hii"] + 4.0, zeta1=max(0.8, 0.8 * p["zeta1"]), zeta2=0.0, c1=1.0, c2=0.0)
    if orb == "p":
        s = spec["s"]
        return dict(nzeta=1, n=s["n"] + 1, Hii=s["Hii"] + 3.0, zeta1=max(0.8, 0.8 * s["zeta1"]), zeta2=0.0, c1=1.0, c2=0.0)
    if orb == "f":
        if "d" not in spec:
            raise ValueError(f"Cannot add f to {el}: no d shell to start from")
        d = spec["d"]
        return dict(nzeta=1, n=d["n"] + 1, Hii=d["Hii"] + 6.0, zeta1=max(0.8, 0.6 * d["zeta1"]), zeta2=0.0, c1=1.0, c2=0.0)
    raise ValueError(f"Cannot add a {orb} shell")


def parse_basis_add(text):
    """'S:d, Zn:d H:p' or {'S': ['d']} -> {'S': ['d'], 'Zn': ['d'], 'H': ['p']}; 'auto' is kept."""
    if not text:
        return {}
    if isinstance(text, str) and text.strip().lower() == "auto":
        return "auto"
    if isinstance(text, dict):
        return {k.capitalize(): [o.lower() for o in (v if isinstance(v, (list, tuple)) else [v])] for k, v in text.items()}
    out = {}
    for item in str(text).replace(",", " ").split():
        el, _, orbs = item.partition(":")
        if not orbs:
            raise ValueError(f"basis entry {item!r}: use Element:orbitals, e.g. S:d or H:p")
        for o in orbs.lower():
            if o not in "spdf":
                raise ValueError(f"basis entry {item!r}: orbital must be s, p, d or f")
            out.setdefault(el.capitalize(), []).append(o)
    return out


def extend_basis(params, add):
    """Add shells to a YAeHMOP parameter dict (returns a copy); existing shells are kept."""
    import copy
    p = copy.deepcopy(params)
    add = parse_basis_add(add)
    if add == "auto":   # p where missing (H, alkalis); d where missing on elements with p (s/p elements, Zn)
        add = {}
        for el, spec in p.items():
            have = {o for o in "spdf" if o in spec and float(spec[o].get("zeta1", 0)) > 0}
            if "p" not in have:
                add.setdefault(el, []).append("p")
            elif "d" not in have:
                add.setdefault(el, []).append("d")
    for el, orbs in add.items():
        if el not in p:
            continue
        for orb in orbs:
            if orb not in p[el] or float(p[el][orb].get("zeta1", 0)) <= 0:
                p[el][orb] = extra_shell(el, orb, p[el])
    return p


def make_double_zeta(params, spec, zeta2_factor=0.5, ratio=0.3):
    """Turn single-zeta shells into contracted double-zeta ones (returns a copy).

    spec: "S:p Te:p" or {"S": ["p"]}. The existing exponent stays as the inner (contracted)
    function; a diffuse tail zeta2 = zeta2_factor * zeta1 with weight c2/c1 = ratio is added,
    and both are fitted. p shells need a YAeHMOP with double-zeta p (ac-dias/yaehmop fork);
    d and f shells work with the original program."""
    import copy
    p = copy.deepcopy(params)
    for el, orbs in parse_basis_add(spec).items() if spec else []:
        if el not in p:
            continue
        for orb in orbs:
            if orb not in p[el] or float(p[el][orb].get("zeta1", 0)) <= 0:
                raise ValueError(f"basis_dz {el}:{orb}: {el} has no {orb} shell")
            o = p[el][orb]
            if int(o["nzeta"]) == 2:
                continue
            o.update(nzeta=2, zeta2=max(0.5, zeta2_factor * float(o["zeta1"])), c1=1.0, c2=float(ratio))
    return p

