"""Numerical DFT band structures as fitting references (VASP, QE, SIESTA, FHI-aims).

``read_dft_bands(folder, code)`` reads the raw eigenvalues of a band-structure
calculation, the k points in fractional reciprocal coordinates, the crystal
structure, the number of occupied bands and the high-symmetry labels.  No
image or digitization is involved: the fit uses the DFT numbers themselves.

``eht_reference(ref, points_per_line, periodic_dim)`` turns them into a
YAeHMOP band path (one line per straight DFT segment; discontinuities such as
``X|U`` become an extra line that is excluded from the error) and the DFT
energies interpolated at exactly the YAeHMOP k points.

Files read in the folder (``dft_file`` pins one when several are present):

* VASP      vasprun.xml, or EIGENVAL + POSCAR/CONTCAR (+ OUTCAR, INCAR*);
            labels from a line-mode KPOINTS / KPOINTS-bnd.
* QE        <prefix>.save/data-file-schema.xml (or <prefix>.xml) of a
            'bands' run, or the pw.x text output of that run; labels from the
            pw.x input (K_POINTS crystal_b/tpiba_b with "! label" comments).
* SIESTA    <SystemLabel>.bands + the .fdf with %block BandLines;
            structure from .STRUCT_OUT, .XV or the .fdf.
* FHI-aims  band1001.out ... + control.in ("output band") + geometry.in
            (+ aims.out for the Fermi level).

Spinless references only: collinear spin-polarized data are accepted when both
channels coincide; spin-orbit data are refused (FHI-aims *.no_soc band files
are used instead when present).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np

HARTREE = 27.211386245988
BOHR = 0.529177210903
DFT_CODES = ("vasp", "qe", "siesta", "aims", "wannier90")
_CODE_ALIASES = {"quantum-espresso": "qe", "espresso": "qe", "pw": "qe",
                 "fhi-aims": "aims", "fhiaims": "aims", "w90": "wannier90", "wannier": "wannier90"}


@dataclass
class DFTBands:
    code: str
    atoms: object                 # ase.Atoms of the band calculation
    kfrac: np.ndarray             # (nk, 3) fractional reciprocal coordinates
    energies: np.ndarray          # (nk, nbands) eV
    nocc: int                     # occupied (doubly occupied) bands
    fermi: float | None = None    # eV, same zero as energies
    segments: list | None = None  # list of index arrays (straight, continuous)
    labels: list = field(default_factory=list)  # [(frac(3), label)]
    files: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    nocc_source: str = ""
    spin_channel: str | None = None
    metallic: bool = False        # no gap: bands are aligned at the Fermi level


def normalize_code(code):
    key = str(code).strip().lower()
    key = _CODE_ALIASES.get(key, key)
    if key not in DFT_CODES:
        raise ValueError(f"Unknown DFT code {code!r}; use one of: {', '.join(DFT_CODES)}")
    return key


def pretty_label(label):
    s = str(label).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1].strip()
    s = s.lstrip("\\")
    if s.lower() in ("g", "gamma", "gm", "gam", "γ"):
        return "Γ"
    return s


def _floats(text):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?",
                                         text.replace("D", "E").replace("d", "e"))]


def _pick(files, what, folder, pinned=None):
    files = sorted(set(Path(f) for f in files))
    if pinned is not None:
        p = Path(pinned)
        p = p if p.is_absolute() else Path(folder) / p
        if not p.is_file():
            raise FileNotFoundError(f"{what}: {p} not found")
        return p.resolve()
    if not files:
        return None
    if len(files) > 1:
        raise ValueError(f"Several {what} in {folder}: " + ", ".join(str(f.relative_to(folder)) for f in files)
                         + " ; choose one with --dft-file")
    return files[0].resolve()


# --------------------------------------------------------------------------
# occupied bands
# --------------------------------------------------------------------------
def gap_after(E, nocc):
    if not 0 < nocc < E.shape[1]:
        return -np.inf
    return float(E[:, nocc].min() - E[:, nocc - 1].max())


def nocc_from_fermi(E, fermi, tol=0.05, reach=0.3):
    """Bands entirely below the Fermi level (+tol); refuses metals.

    A Fermi level taken from a different k set (e.g. the SCF mesh) can sit slightly below
    the band maximum on the path; then the gap closest to it, within `reach` eV, is used.
    """
    occ = int(np.sum(E.max(axis=0) <= fermi + tol))
    crossing = [j for j in range(E.shape[1]) if E[:, j].min() < fermi - tol < fermi + tol < E[:, j].max()]
    if not crossing and gap_after(E, occ) > 0:
        return occ
    best = None
    for n in range(1, E.shape[1]):
        if gap_after(E, n) <= 0.02:
            continue
        lo, hi = E[:, n - 1].max(), E[:, n].min()
        dist = 0.0 if lo <= fermi <= hi else min(abs(fermi - lo), abs(fermi - hi))
        if best is None or dist < best[0]:
            best = (dist, n)
    if best is not None and best[0] <= reach:
        return best[1]
    raise ValueError(f"Bands cross the Fermi level ({fermi:.3f} eV): metallic reference; "
                     "set valence_bands in [reference] if this is wrong")


def nocc_from_electrons(nelect, spinor=False):
    n = float(nelect)
    if abs(n - round(n)) > 1e-4:
        raise ValueError(f"Non-integer number of electrons {n}")
    n = int(round(n))
    if spinor:
        return n
    if n % 2:
        raise ValueError(f"Odd number of electrons ({n}): not a spinless closed-shell reference")
    return n // 2


# --------------------------------------------------------------------------
# VASP
# --------------------------------------------------------------------------
def _vasp_kpoints_labels(folder):
    """Labels and points/segment from a line-mode KPOINTS file (reciprocal)."""
    for name in ("KPOINTS", "KPOINTS-bnd", "KPOINTS_bnd", "KPOINTS.bands", "KPOINTS-band"):
        p = Path(folder) / name
        if not p.is_file():
            continue
        lines = p.read_text(errors="replace").splitlines()
        if len(lines) < 5 or not lines[2].strip().lower().startswith("l"):
            continue
        if not lines[3].strip().lower().startswith("r"):
            return [], None, p  # cartesian line mode: labels not mapped
        try:
            nper = int(lines[1].split()[0])
        except (ValueError, IndexError):
            nper = None
        labels = []
        for ln in lines[4:]:
            raw = ln.split("!")
            nums = _floats(raw[0])
            if len(nums) < 3:
                continue
            lab = raw[1].strip() if len(raw) > 1 else " ".join(raw[0].split()[3:])
            if lab:
                labels.append((np.array(nums[:3]), pretty_label(lab.split()[0])))
        return labels, nper, p
    return [], None, None


def _vasp_flags(folder):
    """ISPIN / SOC from OUTCAR or INCAR files."""
    flags = {}
    for name in ("OUTCAR", "INCAR-bnd", "INCAR", "INCAR-band"):
        p = Path(folder) / name
        if not p.is_file():
            continue
        text = p.read_text(errors="replace") if p.stat().st_size < 50e6 else ""
        for key in ("LSORBIT", "LNONCOLLINEAR"):
            m = re.search(rf"^\s*{key}\s*=\s*\.?([TtFf])", text, re.M)
            if m and key not in flags:
                flags[key] = m.group(1).upper() == "T"
        m = re.search(r"E-fermi\s*:\s*([-\d.]+)", text)
        if m and "efermi" not in flags:
            flags["efermi"] = float(m.group(1))
        if name == "OUTCAR":
            flags["_outcar"] = True
    return flags


def _read_vasprun(path):
    from ase.io import read
    root = ET.parse(path).getroot()
    params = {i.get("name"): (i.text or "").strip() for i in root.iter("i")}
    kp = root.find("kpoints")
    kfrac = np.array([_floats(v.text) for v in kp.find("varray[@name='kpointlist']")])
    w = kp.find("varray[@name='weights']")
    weights = np.array([float(v.text) for v in w]) if w is not None else np.ones(len(kfrac))
    eig = root.find("calculation/eigenvalues/array/set")
    if eig is None:
        raise ValueError(f"{path}: no eigenvalues (incomplete vasprun.xml?)")
    spins = []
    for sset in eig.findall("set"):
        spins.append(np.array([[float(r.text.split()[0]) for r in kset.findall("r")]
                               for kset in sset.findall("set")]))
    efermi = None
    for e in root.iter("i"):
        if e.get("name") == "efermi":
            efermi = float(e.text)
    atoms = read(path, index=-1, format="vasp-xml")
    nelect = float(params.get("NELECT", "nan"))
    soc = params.get("LSORBIT", "F").upper().startswith("T") or params.get("LNONCOLLINEAR", "F").upper().startswith("T")
    return atoms, kfrac, weights, spins, efermi, nelect, soc


def _read_eigenval(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    ispin = int(lines[0].split()[-1])
    nelect, nk, nb = (int(round(float(x))) for x in lines[5].split()[:3])
    if len(lines) < 6 + nk * (nb + 2):
        raise ValueError(f"{path}: incomplete EIGENVAL ({len(lines)} lines for {nk} k points x {nb} bands); "
                         "the VASP run did not finish")
    kfrac, weights = [], []
    spins = [np.zeros((nk, nb)) for _ in range(ispin)]
    i = 6
    for k in range(nk):
        while not lines[i].strip():
            i += 1
        v = _floats(lines[i])
        kfrac.append(v[:3]); weights.append(v[3] if len(v) > 3 else 1.0)
        i += 1
        for b in range(nb):
            f = lines[i].split(); i += 1
            for s in range(ispin):
                spins[s][k, b] = float(f[1 + s])
    return np.array(kfrac), np.array(weights), spins, nelect


def read_vasp(folder, dft_file=None, soc=False):
    from ase.io import read
    want_soc = soc
    folder = Path(folder)
    notes, files = [], {}
    vr = _pick([p for p in folder.glob("vasprun*.xml")], "vasprun.xml files", folder,
               dft_file if dft_file and "vasprun" in Path(dft_file).name else None)
    flags = _vasp_flags(folder)
    atoms = None
    if vr is not None and (dft_file is None or "vasprun" in Path(dft_file).name):
        try:
            atoms, kfrac, weights, spins, efermi, nelect, soc = _read_vasprun(vr)
            files["eigenvalues"] = str(vr)
        except (ET.ParseError, AttributeError, ValueError) as exc:
            notes.append(f"vasprun.xml unreadable ({exc}); using EIGENVAL")
            atoms = None
    if atoms is None:
        ev = _pick([folder / "EIGENVAL"] if (folder / "EIGENVAL").is_file() else [], "EIGENVAL", folder,
                   dft_file)
        if ev is None:
            raise FileNotFoundError(f"{folder}: no vasprun.xml or EIGENVAL")
        kfrac, weights, spins, nelect = _read_eigenval(ev)
        files["eigenvalues"] = str(ev)
        for name in ("POSCAR", "CONTCAR"):
            p = folder / name
            if p.is_file() and p.stat().st_size > 0:
                atoms = read(p, format="vasp"); files["structure"] = str(p); break
        if atoms is None:
            raise FileNotFoundError(f"{folder}: EIGENVAL needs POSCAR or CONTCAR for the cell")
        efermi = flags.get("efermi")
        soc = flags.get("LSORBIT", False) or flags.get("LNONCOLLINEAR", False)
        if "LSORBIT" not in flags and "LNONCOLLINEAR" not in flags:
            notes.append("no OUTCAR/INCAR: spin-orbit coupling assumed absent")
    if soc and not want_soc:
        raise ValueError("VASP reference with spin-orbit coupling (LSORBIT/LNONCOLLINEAR): "
                         "EHT here is spinless; use a calculation without SOC (or soc_fit.py)")
    if want_soc and not soc:
        raise ValueError("Not a spin-orbit VASP calculation (LSORBIT/LNONCOLLINEAR not set)")
    E = _merge_spins(spins, notes)
    # Hybrid-functional band runs: SCF mesh (weight > 0) + path (weight 0).
    if np.any(weights == 0) and np.any(weights > 0):
        keep = weights == 0
        kfrac, E = kfrac[keep], E[keep]
        notes.append(f"dropped {int((~keep).sum())} weighted SCF k points (zero-weight band path kept)")
    nocc = 0 if _SPIN_CHANNEL else nocc_from_electrons(nelect, spinor=want_soc)   # spin channels: from E_F
    labels, nper, kpfile = _vasp_kpoints_labels(folder)
    if kpfile is not None:
        files["kpoints"] = str(kpfile)
    segments = None
    if nper and len(kfrac) % nper == 0 and nper >= 2:
        segments = [np.arange(s, s + nper) for s in range(0, len(kfrac), nper)]
    return DFTBands("vasp", atoms, kfrac, E, nocc, efermi, segments, labels, files, notes,
                    nocc_source=f"NELECT={nelect:g}")


_SPIN_CHANNEL = None   # set by read_dft_bands(spin_channel=...) for collinear spin-polarized runs


def _merge_spins(spins, notes):
    if _SPIN_CHANNEL is not None:
        if len(spins) != 2:
            raise ValueError("spin_channel requested but the calculation is not spin-polarized")
        notes.append(f"spin channel {'up' if _SPIN_CHANNEL == 1 else 'down'} of a spin-polarized run")
        return spins[_SPIN_CHANNEL - 1]
    if len(spins) == 1:
        return spins[0]
    if np.max(np.abs(spins[0] - spins[1])) < 1e-3:
        notes.append("spin-polarized run with identical channels: spin 1 used")
        return spins[0]
    raise ValueError("Spin-polarized reference with different up/down bands: not supported "
                     "by spinless EHT")


# --------------------------------------------------------------------------
# Quantum ESPRESSO
# --------------------------------------------------------------------------
def _qe_input_labels(folder, cell):
    """Labelled nodes from pw.x inputs with K_POINTS crystal_b / tpiba_b."""
    out = []
    for p in sorted(Path(folder).iterdir()):
        if not p.is_file() or p.stat().st_size > 2e6:
            continue
        try:
            text = p.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        m = re.search(r"K_POINTS\s*[{(]?\s*(crystal_b|tpiba_b)", text, re.I)
        if not m or not re.search(r"calculation\s*=\s*['\"]bands", text, re.I):
            continue
        kind = m.group(1).lower()
        lines = text[m.end():].splitlines()[1:]
        try:
            n = int(lines[0].split()[0])
        except (ValueError, IndexError):
            continue
        alat = None
        if kind == "tpiba_b":
            mm = re.search(r"celldm\(1\)\s*=\s*([\d.eEdD+-]+)", text)
            alat = float(mm.group(1).replace("d", "e").replace("D", "e")) * BOHR if mm else np.linalg.norm(cell[0])
        for ln in lines[1:1 + n]:
            raw = ln.split("!")
            nums = _floats(raw[0])
            if len(nums) < 3 or len(raw) < 2 or not raw[1].strip():
                continue
            k = np.array(nums[:3])
            if kind == "tpiba_b":
                k = (k * 2 * np.pi / alat) @ np.asarray(cell).T / (2 * np.pi)
            out.append((k, pretty_label(raw[1].split()[0])))
        if out:
            return out, p
    return [], None


def _read_qe_xml(path, soc=False):
    from ase import Atoms
    root = ET.parse(path).getroot()
    out = root.find("output")
    if out is None:
        raise ValueError(f"{path}: not a QE output XML")
    calc = root.findtext("input/control_variables/calculation", "").strip()
    bs = out.find("band_structure")
    lsda = bs.findtext("lsda", "false").strip().lower() == "true"
    noncolin = bs.findtext("noncolin", "false").strip().lower() == "true"
    is_soc = noncolin or bs.findtext("spinorbit", "false").strip().lower() == "true"
    if is_soc and not soc:
        raise ValueError("QE reference is noncollinear / spin-orbit: not supported by spinless EHT")
    if soc and not is_soc:
        raise ValueError("Not a noncollinear / spin-orbit QE calculation")
    nelec = float(bs.findtext("nelec"))
    fermi = None
    for tag in ("fermi_energy", "highestOccupiedLevel"):
        if bs.find(tag) is not None:
            fermi = float(bs.findtext(tag)) * HARTREE
            break
    ast = out.find("atomic_structure")
    alat = float(ast.get("alat"))
    cell = np.array([_floats(ast.findtext(f"cell/a{i}")) for i in (1, 2, 3)]) * BOHR
    names = [a.get("name") for a in ast.findall("atomic_positions/atom")]
    pos = np.array([_floats(a.text) for a in ast.findall("atomic_positions/atom")]) * BOHR
    from ase.io.espresso import label_to_symbol
    atoms = Atoms(symbols=[label_to_symbol(n) for n in names], positions=pos, cell=cell, pbc=True)
    kc, E = [], []
    for ks in bs.findall("ks_energies"):
        kc.append(_floats(ks.findtext("k_point")))
        E.append(_floats(ks.findtext("eigenvalues")))
    kc = np.array(kc) * 2 * np.pi / (alat * BOHR)  # 1/Angstrom
    kfrac = kc @ cell.T / (2 * np.pi)
    E = np.array(E) * HARTREE
    spins = [E]
    if lsda:
        nb = E.shape[1] // 2
        spins = [E[:, :nb], E[:, nb:]]
    return atoms, kfrac, spins, nelec, fermi, calc


def alat_from(path):
    m = re.search(r"lattice parameter \(alat\)\s*=\s*([\d.]+)", Path(path).read_text(errors="replace"))
    return float(m.group(1)) * BOHR if m else None


def _read_pw_output(path, soc=False):
    from ase import Atoms
    text = Path(path).read_text(errors="replace")
    is_soc = "Noncollinear" in text or "with spin-orbit" in text.lower()
    if is_soc and not soc:
        raise ValueError("QE reference is noncollinear / spin-orbit: not supported by spinless EHT")
    if soc and not is_soc:
        raise ValueError("Not a noncollinear / spin-orbit pw.x calculation")
    alat = float(re.search(r"lattice parameter \(alat\)\s*=\s*([\d.]+)", text).group(1)) * BOHR
    axes = [re.search(rf"a\({i}\) = \(\s*([^)]+)\)", text).group(1) for i in (1, 2, 3)]
    cell = np.array([_floats(a)[:3] for a in axes]) * alat
    syms, pos = [], []
    m = re.search(r"site n\.\s+atom\s+positions \(alat units\)\n(.*?)\n\s*\n", text, re.S)
    for ln in m.group(1).splitlines():
        mm = re.search(r"\d+\s+(\S+)\s+tau\(\s*\d+\)\s*=\s*\(\s*([^)]+)\)", ln)
        if mm:
            from ase.io.espresso import label_to_symbol
            syms.append(label_to_symbol(mm.group(1))); pos.append(_floats(mm.group(2))[:3])
    atoms = Atoms(symbols=syms, positions=np.array(pos) * alat, cell=cell, pbc=True)
    nelec = float(re.search(r"number of electrons\s*=\s*([\d.]+)", text).group(1))
    fermi = None
    for pat in (r"the Fermi energy is\s+([-\d.]+)", r"highest occupied, lowest unoccupied level \(ev\):\s+([-\d.]+)",
                r"highest occupied level \(ev\):\s+([-\d.]+)"):
        mm = re.findall(pat, text)
        if mm:
            fermi = float(mm[-1]); break
    start = max(text.rfind("End of band structure calculation"), text.rfind("End of self-consistent calculation"))
    if start < 0:
        raise ValueError(f"{path}: no band-structure block")
    block = text[start:]
    if "bands (ev)" not in block:
        return atoms, None, None, nelec, fermi
    spin_blocks = re.split(r"-+\s*SPIN DOWN\s*-+", block)
    spins, kc = [], None
    for sb in spin_blocks:
        ks, ens = [], []
        for chunk in re.split(r"\n\s*k\s*=", sb)[1:]:
            head, _, body = chunk.partition("\n")
            ks.append(_floats(head.split("(")[0])[:3])
            vals = []
            for ln in body.splitlines():
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("occupation") or not re.match(r"^[-\d.\s]+$", s):
                    if vals:
                        break
                    continue
                vals += _floats(s)
            ens.append(vals)
        nb = min(len(e) for e in ens)
        spins.append(np.array([e[:nb] for e in ens]))
        kc = np.array(ks) if kc is None else kc
    # Precise k list (cart. 2pi/alat) printed before the run, when it matches.
    mm = re.search(r"number of k points=\s*(\d+).*?cart\. coord\. in units 2pi/alat\)\n(.*?)\n\s*\n", text, re.S)
    if mm:
        kk = np.array([_floats(l.split("=")[1].split("),")[0])[:3] for l in mm.group(2).splitlines() if "k(" in l])
        if len(kk) == len(kc) and np.allclose(kk, kc, atol=2e-4):
            kc = kk
    kfrac = (kc * 2 * np.pi / alat) @ cell.T / (2 * np.pi)
    return atoms, kfrac, spins, nelec, fermi


def _read_filband(path):
    """bands.x 'filband' file: k (cart. 2pi/alat) and eigenvalues (eV)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    m = re.search(r"nbnd\s*=\s*(\d+)\s*,\s*nks\s*=\s*(\d+)", lines[0])
    nb, nk = int(m.group(1)), int(m.group(2))
    data = _floats(" ".join(lines[1:]))
    if len(data) < nk * (3 + nb):
        raise ValueError(f"{path}: truncated bands.x file")
    arr = np.array(data[:nk * (3 + nb)]).reshape(nk, 3 + nb)
    return arr[:, :3], arr[:, 3:]


def _filband_files(folder):
    out = []
    for p in Path(folder).iterdir():
        if p.is_file() and p.stat().st_size < 200e6 and not p.name.endswith((".gnu", ".rap")):
            try:
                with open(p, errors="replace") as fh:
                    if fh.readline().lstrip().lower().startswith("&plot"):
                        out.append(p)
            except OSError:
                pass
    return out


def read_qe(folder, dft_file=None, soc=False):
    folder = Path(folder)
    notes, files = [], {}
    cand = list(folder.glob("*.save/data-file-schema.xml"))
    cand += [p for p in folder.glob("*.xml") if p.stat().st_size < 500e6 and "<qes:espresso" in p.read_text(errors="replace")[:2000]]
    pinned = Path(dft_file) if dft_file else None
    xml_files = []
    for p in cand:
        try:
            calc = ET.parse(p).getroot().findtext("input/control_variables/calculation", "").strip()
        except ET.ParseError:
            continue
        xml_files.append((p, calc))
    bands_xml = [p for p, c in xml_files if c == "bands"]
    outputs = []
    for p in folder.iterdir():
        if p.is_file() and p.stat().st_size < 500e6 and p.suffix.lower() not in (".xml", ".upf", ".dat"):
            try:
                head = p.read_text(errors="replace")
            except OSError:
                continue
            if "Program PWSCF" in head and "End of band structure calculation" in head:
                outputs.append(p)
    if pinned is not None:
        p = pinned if pinned.is_absolute() else folder / pinned
        use_xml = p.suffix.lower() == ".xml"
        src = p.resolve()
    elif bands_xml:
        # <prefix>.xml duplicates <prefix>.save/data-file-schema.xml: prefer the latter
        saved = [p for p in bands_xml if p.name == "data-file-schema.xml"]
        src, use_xml = _pick(saved or bands_xml, "QE 'bands' XML files", folder), True
    elif outputs:
        src, use_xml = _pick(outputs, "pw.x band-structure outputs", folder), False
    else:
        raise FileNotFoundError(f"{folder}: no QE 'bands' XML (*.save/data-file-schema.xml) or pw.x bands output")
    if use_xml:
        atoms, kfrac, spins, nelec, fermi, calc = _read_qe_xml(src, soc)
        if calc != "bands":
            notes.append(f"XML is from a '{calc}' run, not 'bands'")
    else:
        atoms, kfrac, spins, nelec, fermi = _read_pw_output(src, soc)
        files["structure"] = str(src)
        if spins is None:   # pw.x did not print the bands (>100 k, low verbosity)
            fb = _pick(_filband_files(folder), "bands.x band files", folder)
            if fb is None:
                raise ValueError(f"{src.name} does not print the eigenvalues (set verbosity='high') "
                                 "and no bands.x file (filband) was found")
            kc, E1 = _read_filband(fb)
            cell = np.asarray(atoms.cell)
            alat = float(np.linalg.norm(cell[0])) if alat_from(src) is None else alat_from(src)
            kfrac = (kc * 2 * np.pi / alat) @ cell.T / (2 * np.pi)
            spins = [E1]
            src = fb
            notes.append("eigenvalues from the bands.x file; cell and electrons from the pw.x output")
    files["eigenvalues"] = str(src)
    E = _merge_spins(spins, notes)
    labels, inp = _qe_input_labels(folder, np.asarray(atoms.cell))
    if inp is not None:
        files["labels"] = str(inp)
    return DFTBands("qe", atoms, kfrac, E, 0 if _SPIN_CHANNEL else nocc_from_electrons(nelec, spinor=soc),
                    fermi, None, labels, files, notes,
                    nocc_source=f"nelec={nelec:g}")


# --------------------------------------------------------------------------
# SIESTA
# --------------------------------------------------------------------------
def _read_siesta_bands_file(path):
    text = Path(path).read_text(errors="replace").splitlines()
    fermi = float(text[0].split()[0])
    nb, nspin, nk = (int(x) for x in text[3].split()[:3])
    data = _floats(" ".join(text[4:]))
    need = nk * (1 + nb * nspin)
    if len(data) < need:
        raise ValueError(f"{path}: truncated .bands file")
    arr = np.array(data[:need]).reshape(nk, 1 + nb * nspin)
    kpath = arr[:, 0]
    E = arr[:, 1:].reshape(nk, nspin, nb)
    labels = []
    rest = text[4:]
    m = re.findall(r"^\s*([-\d.Ee+]+)\s+'([^']*)'", "\n".join(rest), re.M)
    for x, lab in m:
        labels.append((float(x), pretty_label(lab)))
    return fermi, kpath, [E[:, s, :] for s in range(nspin)], labels


def read_siesta(folder, dft_file=None, soc=False):
    from structure_io import _parse_fdf, _labelize, read_siesta_fdf, _length
    from ase.io import read
    folder = Path(folder)
    notes, files = [], {}
    bands = _pick([p for p in folder.glob("*.bands")], "SIESTA .bands files", folder,
                  dft_file if dft_file and str(dft_file).endswith(".bands") else None)
    if bands is None:
        raise FileNotFoundError(f"{folder}: no SIESTA .bands file")
    fermi, kpath, spins, path_labels = _read_siesta_bands_file(bands)
    files["eigenvalues"] = str(bands)
    nk = len(kpath)
    label = bands.stem
    chosen = None
    for fdf in sorted(folder.glob("*.fdf")):
        try:
            scal, blocks = _parse_fdf(fdf)
        except (ValueError, OSError):
            continue
        if "bandlines" not in blocks:
            continue
        rows = blocks["bandlines"]
        total = sum(int(r[0]) for r in rows)
        sl = scal.get("systemlabel", [None])[0]
        score = (total == nk) * 2 + (sl == label)
        if total == nk and (chosen is None or score > chosen[0]):
            chosen = (score, fdf, scal, blocks)
    if chosen is None:
        raise ValueError(f"{folder}: no .fdf with a %block BandLines giving {nk} k points "
                         "(BandPoints and path-only .bands files are not supported)")
    _, fdf, scal, blocks = chosen
    files["fdf"] = str(fdf)
    spin = " ".join(scal.get("spin", [])).lower()
    so = scal.get("spinorbit", ["F"])[0].lower().lstrip(".")
    is_soc = so.startswith("t") or spin in ("so", "spin-orbit", "spinorbit", "nc", "non-colinear", "noncolinear", "non-collinear")
    if is_soc and not soc:
        raise ValueError("SIESTA reference with spin-orbit / noncollinear spin: not supported by spinless EHT")
    if soc and not is_soc:
        raise ValueError("Not a spin-orbit SIESTA calculation")
    atoms = None
    for ext in (".STRUCT_OUT", ".XV"):
        p = folder / (label + ext)
        if p.is_file():
            atoms = read(p, format="struct_out" if ext == ".STRUCT_OUT" else "siesta-xv")
            files["structure"] = str(p); break
    if atoms is None:
        atoms = read_siesta_fdf(fdf); files["structure"] = str(fdf)
    cell = np.asarray(atoms.cell)
    scale = _labelize(scal.get("bandlinesscale", ["pi/a"])[0])
    nodes, counts, names = [], [], []
    for r in blocks["bandlines"]:
        counts.append(int(r[0])); nodes.append(np.array([float(x) for x in r[1:4]]))
        names.append(pretty_label(r[4]) if len(r) > 4 else "")
    nodes = np.array(nodes)
    if scale in ("pi/a", "pia"):
        alat = _length(scal["latticeconstant"], "LatticeConstant", "bohr")
        nodes = (nodes * np.pi / alat) @ cell.T / (2 * np.pi)
    elif scale != "reciprocallatticevectors":
        raise ValueError(f"Unsupported BandLinesScale {scal['bandlinesscale'][0]!r}")
    kfrac, segments = [], []
    for i, (n, node) in enumerate(zip(counts, nodes)):
        if i == 0 or n == 1:
            kfrac.append(node)            # path (re)starts here: discontinuity if i > 0
            continue
        a = nodes[i - 1]
        seg0 = len(kfrac) - 1
        kfrac += [a + (node - a) * (j / n) for j in range(1, n + 1)]
        segments.append(np.arange(seg0, len(kfrac)))
    kfrac = np.array(kfrac)
    # Consistency check: SIESTA path coordinate (1/Bohr) against our k points
    rec = 2 * np.pi * np.linalg.inv(cell).T
    cart = kfrac @ rec * BOHR
    steps = np.linalg.norm(np.diff(cart, axis=0), axis=1)
    mine = np.diff(kpath)
    ok = np.isclose(steps, mine, rtol=2e-3, atol=2e-5) | np.isclose(mine, 0, atol=1e-7)
    if not ok.all():
        raise ValueError(f"{bands.name}: k-path lengths do not match BandLines of {fdf.name} "
                         f"with the cell of {Path(files['structure']).name}")
    labels = [(nodes[i], names[i]) for i in range(len(nodes)) if names[i]]
    E = _merge_spins(spins, notes)
    return DFTBands("siesta", atoms, kfrac, E, 0 if _SPIN_CHANNEL else nocc_from_fermi(E, fermi), fermi, segments, labels,
                    files, notes, nocc_source=f"Fermi level {fermi:.4f} eV")


# --------------------------------------------------------------------------
# FHI-aims
# --------------------------------------------------------------------------
def _aims_band_file(path):
    rows = [ _floats(l) for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
    arr = np.array(rows)
    k = arr[:, 1:4]
    occ = arr[:, 4::2]
    e = arr[:, 5::2]
    return k, occ, e


def read_aims(folder, dft_file=None, soc=False):
    from ase.io import read
    import warnings
    folder = Path(folder)
    notes, files = [], {}
    ctrl = folder / "control.in"
    if not ctrl.is_file():
        raise FileNotFoundError(f"{folder}: control.in not found")
    bands_def = []
    for ln in ctrl.read_text(errors="replace").splitlines():
        s = ln.split("#")[0].split()
        if len(s) >= 9 and s[0] == "output" and s[1] == "band":
            a = np.array([float(x) for x in s[2:5]]); b = np.array([float(x) for x in s[5:8]])
            bands_def.append((a, b, int(s[8]), [pretty_label(x) for x in s[9:11]]))
    if not bands_def:
        raise ValueError(f"{ctrl}: no 'output band' lines")
    text = ctrl.read_text(errors="replace")
    spin_on = re.search(r"^\s*spin\s+collinear", text, re.M) is not None
    soc_wanted = soc
    soc = re.search(r"^\s*include_spin_orbit", text, re.M) is not None
    suffix = ""
    want_soc = soc_wanted
    if want_soc and not soc:
        raise ValueError("Not a spin-orbit FHI-aims calculation (no include_spin_orbit)")
    if soc and not want_soc:
        if all((folder / f"band1{i + 1:03d}.out.no_soc").is_file() for i in range(len(bands_def))):
            suffix = ".no_soc"
            notes.append("spin-orbit run: the *.no_soc band files (scalar-relativistic) are used")
        else:
            raise ValueError("FHI-aims reference with spin-orbit coupling and no *.no_soc band files")
    ks, occs, es, segments, n0 = [], [], [], [], 0
    for i in range(len(bands_def)):
        p = folder / f"band1{i + 1:03d}.out{suffix}"
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing ({len(bands_def)} 'output band' lines in control.in)")
        k, occ, e = _aims_band_file(p)
        if spin_on:
            p2 = folder / f"band2{i + 1:03d}.out{suffix}"
            if p2.is_file():
                e2 = _aims_band_file(p2)[2]
                e = _merge_spins([e, e2], notes)
        ks.append(k); occs.append(occ); es.append(e)
        segments.append(np.arange(n0, n0 + len(k))); n0 += len(k)
    files["eigenvalues"] = str(folder / f"band1001.out{suffix}") + f" ... ({len(bands_def)} files)"
    kfrac = np.vstack(ks); occ = np.vstack(occs); E = np.vstack(es)
    full = 1.0 if (spin_on or want_soc) else 2.0
    nk_occ = np.sum(occ > 0.5 * full, axis=1)
    if _SPIN_CHANNEL:
        nk_occ = np.zeros(1, int)                    # spin channels: occupied bands from E_F
    if nk_occ.min() != nk_occ.max():
        raise ValueError("Occupations change along the path: metallic FHI-aims reference")
    nocc = int(nk_occ[0])
    fermi = None
    for out in sorted(folder.glob("*.out")):
        if out.name.startswith(("band", "KS_")) or out.stat().st_size > 500e6:
            continue
        t = out.read_text(errors="replace")
        if "Invoking FHI-aims" not in t:
            continue
        m = re.findall(r"Chemical potential \(Fermi level\):\s*([-\d.Ee+]+)\s*eV", t) or \
            re.findall(r"Electron chemical potential:\s*([-\d.Ee+]+)\s*eV", t)
        if m:
            fermi = float(m[-1]); files["output"] = str(out)
        break
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*FHI-aims IO is moving.*")
        atoms = read(folder / "geometry.in", format="aims")
    files["structure"] = str(folder / "geometry.in")
    labels = []
    for a, b, _, names in bands_def:
        if len(names) == 2:
            labels += [(a, names[0]), (b, names[1])]
    return DFTBands("aims", atoms, kfrac, E, nocc, fermi, segments, labels, files, notes,
                    nocc_source="occupations in band files")


# --------------------------------------------------------------------------
# Wannier90 (interpolated tight-binding bands)
# --------------------------------------------------------------------------
def _w90_block(text, name):
    m = re.search(rf"^\s*begin\s+{name}\s*$(.*?)^\s*end\s+{name}\s*$", text, re.M | re.S | re.I)
    if not m:
        return None
    return [l.split("!")[0].split("#")[0].strip() for l in m.group(1).splitlines()
            if l.split("!")[0].split("#")[0].strip()]


def _w90_keyword(text, key):
    m = re.search(rf"^\s*{key}\s*[=:]?\s*([^\s!#]+)", text, re.M | re.I)
    return m.group(1) if m else None


def _fermi_nearby(folder):
    """Fermi level of the DFT run next to the wannierization (VASP OUTCAR, QE outputs)."""
    p = folder / "OUTCAR"
    if p.is_file():
        m = re.findall(r"E-fermi\s*:\s*([-\d.]+)", p.read_text(errors="replace"))
        if m:
            return float(m[-1]), str(p)
    for q in sorted(folder.iterdir()):
        if q.is_file() and q.stat().st_size < 50e6 and q.suffix.lower() in (".out", ".pwo", ".log"):
            txt = q.read_text(errors="replace")
            m = re.findall(r"the Fermi energy is\s+([-\d.]+)", txt) or \
                re.findall(r"highest occupied, lowest unoccupied level \(ev\):\s+([-\d.]+)", txt) or \
                re.findall(r"highest occupied level \(ev\):\s+([-\d.]+)", txt)
            if m:
                return float(m[-1]), str(q)
    return None, None


def read_wannier90(folder, dft_file=None, soc=False):
    """Bands interpolated by Wannier90 (seedname_band.dat/.kpt) as the reference.

    Cell from seedname.nnkp (real_lattice) or seedname.win (unit_cell_cart), atoms from
    the .win, labels from seedname_band.labelinfo.dat; the Fermi level from fermi_energy
    in the .win, else from the DFT run in the same folder (OUTCAR, pw.x output); the
    occupied Wannier bands are those below it.
    """
    from ase import Atoms
    folder = Path(folder)
    notes, files = [], {}
    wins = [p for p in folder.glob("*.win")]
    win = _pick(wins, "Wannier90 .win files", folder, dft_file if dft_file and str(dft_file).endswith(".win") else None)
    if win is None:
        raise FileNotFoundError(f"{folder}: no Wannier90 .win file")
    seed = win.stem
    text = win.read_text(errors="replace")
    files["win"] = str(win)
    spinors = (_w90_keyword(text, "spinors") or "f").lower().lstrip(".").startswith("t")
    if spinors and not soc:
        raise ValueError("Wannier90 run with spinors = true (spin-orbit): not supported by spinless EHT")
    if soc and not spinors:
        raise ValueError("Not a spinor (spin-orbit) Wannier90 run")
    # cell
    nnkp = folder / f"{seed}.nnkp"
    cell = None
    if nnkp.is_file():
        rl = _w90_block(nnkp.read_text(errors="replace"), "real_lattice")
        if rl:
            cell = np.array([[float(x) for x in r.split()[:3]] for r in rl[:3]]); files["nnkp"] = str(nnkp)
    if cell is None:
        uc = _w90_block(text, "unit_cell_cart")
        if not uc:
            raise ValueError(f"{win}: no unit_cell_cart (and no .nnkp)")
        scale = 1.0
        if uc[0].lower() in ("bohr", "ang"):
            scale = BOHR if uc[0].lower() == "bohr" else 1.0; uc = uc[1:]
        cell = np.array([[float(x) for x in r.split()[:3]] for r in uc[:3]]) * scale
    # atoms
    syms, pos, frac = [], [], None
    blk = _w90_block(text, "atoms_frac")
    if blk:
        frac = True
    else:
        blk = _w90_block(text, "atoms_cart"); frac = False
    if not blk:
        raise ValueError(f"{win}: no atoms_frac / atoms_cart block")
    scale = 1.0
    if not frac and blk[0].lower() in ("bohr", "ang"):
        scale = BOHR if blk[0].lower() == "bohr" else 1.0; blk = blk[1:]
    for r in blk:
        f = r.split(); syms.append(re.sub(r"[^A-Za-z]", "", f[0]).capitalize()); pos.append([float(x) for x in f[1:4]])
    pos = np.array(pos)
    atoms = Atoms(symbols=syms, scaled_positions=pos, cell=cell, pbc=True) if frac else \
        Atoms(symbols=syms, positions=pos * scale, cell=cell, pbc=True)
    files["structure"] = str(win)
    # bands
    kpt = folder / f"{seed}_band.kpt"; dat = folder / f"{seed}_band.dat"
    if not (kpt.is_file() and dat.is_file()):
        raise FileNotFoundError(f"{folder}: {seed}_band.kpt / {seed}_band.dat not found (bands_plot = true)")
    kl = kpt.read_text().split("\n")
    nk = int(kl[0].split()[0])
    kfrac = np.array([[float(x) for x in l.split()[:3]] for l in kl[1:1 + nk]])
    vals = _floats(dat.read_text())
    nw = len(vals) // (2 * nk)
    if nw * 2 * nk != len(vals):
        raise ValueError(f"{dat}: {len(vals)} numbers is not a whole number of bands of {nk} k points")
    E = np.array(vals).reshape(nw, nk, 2)[:, :, 1].T
    files["eigenvalues"] = str(dat)
    labels = []
    li = folder / f"{seed}_band.labelinfo.dat"
    if li.is_file():
        for l in li.read_text().splitlines():
            f = l.split()
            if len(f) >= 6:
                labels.append((np.array([float(x) for x in f[3:6]]), pretty_label(f[0])))
        files["labels"] = str(li)
    fermi = _w90_keyword(text, "fermi_energy")
    if fermi is not None:
        fermi = float(fermi.replace("d", "e").replace("D", "e")); src = "fermi_energy in the .win"
    else:
        fermi, src = _fermi_nearby(folder)
    if fermi is None:
        raise ValueError(f"{folder}: no Fermi level (fermi_energy in the .win, OUTCAR or pw.x output); "
                         "give valence_bands instead")
    notes.append(f"Fermi level {fermi:.4f} eV from {src}")
    return DFTBands("wannier90", atoms, kfrac, E, nocc_from_fermi(E, fermi), fermi, None, labels, files, notes,
                    nocc_source=f"Fermi level ({src})")


def _read_metal(folder, code, dft_file, soc):
    """Run a reader that may refuse a metal while counting occupied bands: the count is
    irrelevant here, so a gap/occupation refusal is retried with a dummy count."""
    global nocc_from_fermi
    orig = nocc_from_fermi
    nocc_from_fermi = lambda E, fermi, tol=0.05, reach=0.3: max(1, int(np.sum(np.asarray(E).max(axis=0) <= fermi)))
    try:
        return READERS[normalize_code(code)](folder, dft_file, soc=soc)
    finally:
        nocc_from_fermi = orig


READERS = {"vasp": read_vasp, "qe": read_qe, "siesta": read_siesta, "aims": read_aims, "wannier90": read_wannier90}


def read_dft_bands(folder, code, dft_file=None, valence_bands=None, soc=False, spin_channel=None,
                   metallic=False, fermi=None):
    """Read a DFT band-structure folder; checks for a positive gap.

    metallic=True accepts a metal (no gap check): the bands are fitted relative to the
    Fermi level (objective 'metal'); `fermi` (eV) replaces the Fermi level of the
    files, e.g. by the one of the self-consistent run.

    soc=True reads a spin-orbit (noncollinear) calculation instead: every band is a
    spinor holding one electron, so nocc counts electrons, not pairs.
    spin_channel='up'/'down' (or 1/2) reads one channel of a collinear spin-polarized
    run; its occupied bands are those below the common Fermi level.
    """
    global _SPIN_CHANNEL
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"DFT folder not found: {folder}")
    ch = {None: None, "up": 1, "down": 2, "1": 1, "2": 2, 1: 1, 2: 2}.get(
        spin_channel if not isinstance(spin_channel, str) else spin_channel.lower())
    if spin_channel is not None and ch is None:
        raise ValueError("spin_channel must be up or down")
    _SPIN_CHANNEL = ch
    try:
        if metallic:
            ref = _read_metal(folder, code, dft_file, soc)
        else:
            ref = READERS[normalize_code(code)](folder, dft_file, soc=soc)
    finally:
        _SPIN_CHANNEL = None
    if fermi is not None:
        ref.notes.append(f"Fermi level set by the user: {float(fermi):.4f} eV (file: {ref.fermi})")
        ref.fermi = float(fermi)
    if metallic:
        if ref.fermi is None:
            raise ValueError("a metallic reference needs the Fermi level (set fermi in [reference])")
        E = np.sort(np.asarray(ref.energies, float), axis=1)
        ref.energies = E
        ref.nocc = int(np.sum(E.max(axis=0) <= ref.fermi))   # bands entirely below E_F (information only)
        ref.nocc_source = f"metal: {ref.nocc} bands entirely below E_F = {ref.fermi:.4f} eV"
        ref.metallic = True
        if ch is not None:
            ref.spin_channel = "up" if ch == 1 else "down"
        if ref.energies.ndim != 2 or len(ref.kfrac) != len(ref.energies) or len(ref.kfrac) < 2:
            raise ValueError("Inconsistent k points / eigenvalues")
        return ref
    if ch is not None:
        if ref.fermi is None:
            raise ValueError("a spin channel needs the Fermi level of the run (not found)")
        if valence_bands is None:
            ref.nocc = nocc_from_fermi(np.sort(np.asarray(ref.energies, float), axis=1), ref.fermi)
            ref.nocc_source = f"Fermi level {ref.fermi:.4f} eV (spin channel {'up' if ch == 1 else 'down'})"
        ref.spin_channel = "up" if ch == 1 else "down"
    # Energies in ascending order at every k: bands.x (QE) and other band tools reorder the
    # bands to follow them through crossings, which breaks the band count from the gap.
    E = np.asarray(ref.energies, float)
    if np.any(np.diff(E, axis=1) < -1e-9):
        ref.notes.append("eigenvalues sorted in energy at every k (file had bands reordered across crossings)")
        ref.energies = np.sort(E, axis=1)
    if valence_bands is not None:
        ref.nocc = int(valence_bands); ref.nocc_source = "user (valence_bands)"
    if ref.energies.ndim != 2 or len(ref.kfrac) != len(ref.energies) or len(ref.kfrac) < 2:
        raise ValueError("Inconsistent k points / eigenvalues")
    if not np.isfinite(ref.energies).all():
        raise ValueError("Non-finite eigenvalues")
    g = gap_after(ref.energies, ref.nocc)
    if g <= 0.02:
        raise ValueError(f"No positive gap above band {ref.nocc} ({ref.nocc_source}): gap {g:.3f} eV. "
                         "The fitter needs a gapped spinless reference; set valence_bands if the "
                         "occupied-band count is wrong")
    return ref


# --------------------------------------------------------------------------
# DFT path -> YAeHMOP path and target
# --------------------------------------------------------------------------
def auto_segments(kfrac, cell):
    """Split a k list into straight continuous segments (index arrays)."""
    rec = 2 * np.pi * np.linalg.inv(np.asarray(cell)).T
    cart = kfrac @ rec
    d = np.diff(cart, axis=0)
    step = np.linalg.norm(d, axis=1)
    pos = step[step > 1e-9]
    med = np.median(pos) if len(pos) else 1.0
    segs, cur = [], [0]
    for i in range(len(step)):
        if step[i] <= 1e-9:                     # repeated node: continuous break
            if len(cur) > 1:
                segs.append(np.array(cur))
            cur = [i + 1]
            continue
        if step[i] > 4 * med and len(cur) >= 1: # discontinuous jump
            if len(cur) > 1:
                segs.append(np.array(cur))
            cur = [i + 1]
            continue
        if len(cur) >= 2:
            prev = cart[cur[-1]] - cart[cur[-2]]
            cosang = prev @ d[i] / (np.linalg.norm(prev) * step[i])
            if cosang < 1 - 1e-5:               # direction change: shared node
                segs.append(np.array(cur))
                cur = [cur[-1]]
        cur.append(i + 1)
    if len(cur) > 1:
        segs.append(np.array(cur))
    return segs


def _label_for(k, labels, tol=2e-3):
    for kk, lab in labels:
        if np.allclose(k, kk, atol=tol):
            return lab
    if np.allclose(k, 0, atol=1e-6):
        return "Γ"
    return None


def bravais_labels(cell, periodic_dim=3):
    """Special points of the Bravais lattice (ASE), for paths without labels."""
    try:
        from ase.cell import Cell
        c = Cell(np.asarray(cell))
        sp = c.bandpath(npoints=0, pbc=[True] * periodic_dim + [False] * (3 - periodic_dim)).special_points
    except Exception:
        return []
    out = []
    for name, k in sp.items():
        out.append((np.asarray(k, float), pretty_label(name)))
    return out


def lattice_point_group(cell, tol=1e-4):
    """Integer matrices W (entries -1..1) with W^T G W = G: the lattice holohedry."""
    import itertools
    cell = np.asarray(cell, float)
    G = cell @ cell.T
    vals = np.array(list(itertools.product((-1, 0, 1), repeat=9))).reshape(-1, 3, 3)
    vals = vals[np.abs(np.linalg.det(vals)).round() == 1]
    scale = max(1.0, np.abs(G).max())
    ok = np.all(np.abs(np.einsum("nji,jk,nkl->nil", vals, G, vals) - G) < tol * scale, axis=(1, 2))
    return vals[ok]


def _match_mod_g(k, labels, ops=None, tol=2e-3):
    ks = [np.asarray(k, float)]
    if ops is not None:
        ks = [w.T @ ks[0] for w in ops]
        ks += [-x for x in ks]
    for kk, lab in labels:
        for q in ks:
            d = q - kk
            if np.allclose(d, np.round(d), atol=tol):
                return lab
    return None


@dataclass
class EHTReference:
    nodes: np.ndarray        # (L+1, 3) YAeHMOP band-path nodes
    labels: list             # node labels (YAeHMOP-safe ASCII)
    plot_labels: list        # node labels for figures
    n: int                   # points per line
    mask: np.ndarray         # (npts,) True where a DFT value exists
    x: np.ndarray            # path coordinate of masked points, 0..1, jumps collapsed
    energies: np.ndarray     # (nmask, nbands) DFT eV at masked points
    ticks: list              # [(x, label)]
    real_lines: list         # bool per line


def eht_reference(ref, n=20, periodic_dim=3):
    """YAeHMOP path through the DFT segments and DFT energies at its k points."""
    cell = np.asarray(ref.atoms.cell)
    segs = ref.segments if ref.segments else auto_segments(ref.kfrac, cell)
    segs = [np.asarray(s) for s in segs if len(s) >= 2]
    if not segs:
        raise ValueError("The DFT k list has no straight segment of two or more points")
    n = int(n)
    if n < 2:
        raise ValueError("points per line must be >= 2")
    kf = np.array(ref.kfrac, float)
    E = np.asarray(ref.energies, float)
    nodes, real, seg_of_line = [kf[segs[0][0]]], [], []
    for s in segs:
        a, b = kf[s[0]], kf[s[-1]]
        if not np.allclose(a, nodes[-1], atol=1e-6):
            nodes.append(a); real.append(False); seg_of_line.append(None)   # jump
        nodes.append(b); real.append(True); seg_of_line.append(s)
    nodes = np.array(nodes)
    L = len(nodes) - 1
    rec = 2 * np.pi * np.linalg.inv(cell).T
    rec[periodic_dim:] = 0.0                   # inactive directions carry no k

    def interp(line, t):
        s, a, b = seg_of_line[line], nodes[line], nodes[line + 1]
        tt = (kf[s] - a) @ (b - a) / max((b - a) @ (b - a), 1e-30)
        o = np.argsort(tt)
        return np.array([np.interp(t, tt[o], E[s][o, j]) for j in range(E.shape[1])])

    mask, values, xs, node_x = [], [], [], []
    x, prev = 0.0, None
    for line in range(L):
        a, b = nodes[line], nodes[line + 1]
        for i in range(n + (1 if line == L - 1 else 0)):
            t = i / n
            if real[line]:
                val = interp(line, t)
            elif i == 0:                       # end point of the previous segment
                val = interp(line - 1, 1.0)
            else:
                mask.append(False); continue
            c = (a + (b - a) * t) @ rec
            if prev is not None:
                x += float(np.linalg.norm(c - prev))
            prev = c
            if i == 0:
                node_x.append(x)
            mask.append(True); values.append(val); xs.append(x)
        if not real[line]:
            prev = nodes[line + 1] @ rec        # the jump itself has zero length
    node_x.append(x)
    xs = np.array(xs)
    if xs[-1] <= 0:
        raise ValueError("Zero-length k path")
    xs /= xs[-1]
    node_x = np.array(node_x) / x
    auto = [] if ref.labels else bravais_labels(cell, periodic_dim)
    ops = lattice_point_group(cell) if auto else None
    names = [_label_for(k, ref.labels) or _match_mod_g(k, auto) or _match_mod_g(k, auto, ops)
             or f"K{j + 1}" for j, k in enumerate(nodes)]
    ticks = []
    for j in range(L + 1):
        if j < L and not real[j]:
            continue                           # merged into "A|B" at the next node
        if j > 0 and not real[j - 1]:
            ticks.append((float(node_x[j - 1]), names[j - 1] + "|" + names[j]))
        else:
            ticks.append((float(node_x[j]), names[j]))
    eht_nodes = nodes.copy()
    eht_nodes[:, periodic_dim:] = 0.0
    safe = [re.sub(r"[^A-Za-z0-9]", "", "G" if nm == "Γ" else nm)[:3] or "K" for nm in names]
    return EHTReference(eht_nodes, safe, names, n, np.array(mask), xs, np.array(values), ticks, real)


def save_reference(ref, path):
    """Write the raw DFT data (eV, fractional k) to an .npz the fitters read."""
    from ase.io import write
    lab_k = np.array([k for k, _ in ref.labels]) if ref.labels else np.zeros((0, 3))
    lab_n = np.array([n for _, n in ref.labels], dtype=object) if ref.labels else np.array([], dtype=object)
    seg_id = np.full(len(ref.kfrac), -1)
    seg_start = []
    if ref.segments:
        seg_start = [int(s[0]) for s in ref.segments]
        seg_len = [len(s) for s in ref.segments]
    np.savez_compressed(path, kfrac=ref.kfrac, energies=ref.energies, nocc=ref.nocc,
                        fermi=np.nan if ref.fermi is None else ref.fermi,
                        cell=np.asarray(ref.atoms.cell),
                        seg_start=np.array(seg_start, int),
                        seg_len=np.array(seg_len if ref.segments else [], int),
                        label_k=lab_k, label_name=np.array([str(x) for x in lab_n]),
                        code=ref.code)
    return Path(path)


def load_reference(path, atoms=None):
    """Inverse of save_reference (atoms: the structure used for the fit)."""
    d = np.load(path, allow_pickle=False)
    from ase import Atoms
    if atoms is None:
        atoms = Atoms(cell=d["cell"], pbc=True)
    elif not np.allclose(np.asarray(atoms.cell), d["cell"], atol=1e-4):
        raise ValueError("Structure cell differs from the cell of the DFT band calculation; "
                         "the k points would be wrong")
    segs = [np.arange(s, s + n) for s, n in zip(d["seg_start"], d["seg_len"])] or None
    labels = [(k, str(n)) for k, n in zip(d["label_k"], d["label_name"])]
    fermi = float(d["fermi"])
    return DFTBands(str(d["code"]), atoms, d["kfrac"], d["energies"], int(d["nocc"]),
                    None if np.isnan(fermi) else fermi, segs, labels)


def vacuum_gaps(atoms):
    """Largest empty slab (Angstrom) along each lattice vector, periodic images included."""
    cell = np.asarray(atoms.cell, float)
    vol = abs(np.linalg.det(cell))
    frac = atoms.get_scaled_positions(wrap=True)
    gaps = []
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        height = vol / np.linalg.norm(np.cross(cell[j], cell[k]))   # interplanar spacing
        f = np.sort(frac[:, i])
        d = np.diff(np.r_[f, f[0] + 1.0])
        gaps.append(float(d.max() * height))
    return gaps


def detect_periodic_dim(atoms, vacuum=6.0):
    """1, 2 or 3 from vacuum along the LAST lattice vectors (YAeHMOP convention)."""
    vac = [g >= vacuum for g in vacuum_gaps(atoms)]
    dim = 3 - sum(vac)
    if dim == 0:
        raise ValueError("Vacuum along all lattice vectors: not a periodic system")
    if any(vac[:dim]):
        raise ValueError(f"Vacuum along lattice vector(s) {[i + 1 for i, v in enumerate(vac) if v]}: the periodic "
                         f"directions must be the first ones (a for 1D, a and b for 2D); reorder the cell or set "
                         "the dimension explicitly")
    return dim
