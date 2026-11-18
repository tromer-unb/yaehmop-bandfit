"""Crystal-structure input for the fitters: CIF, VASP, Quantum ESPRESSO, SIESTA, FHI-aims.

Every fitter only needs the periodic cell, the atomic positions and the
chemical symbols.  ``read_structure`` returns them as an ``ase.Atoms`` from
any of the supported files; the format is detected from the file name and,
when the name is not conclusive, from the file content.  It can always be
forced with ``structure_format`` (``--structure-format`` on the command
line, ``structure_format:`` in YAML manifests).

Supported files (``structure_format`` value in brackets):

* CIF                                   [cif]
* VASP POSCAR/CONTCAR, *.vasp           [vasp]
* VASP OUTCAR, vasprun.xml (last step)  [vasp-out], [vasp-xml]
* Quantum ESPRESSO pw.x input, any ibrav [qe]       (own lattice code, below)
* Quantum ESPRESSO pw.x output (last)   [qe-out]
* SIESTA input *.fdf                    [siesta]   (own reader, below)
* SIESTA *.XV, *.STRUCT_OUT             [siesta-xv], [siesta-struct-out]
* FHI-aims geometry.in                  [aims]
* FHI-aims output (last step)           [aims-out]

ASE does the parsing except for the SIESTA .fdf (not read by ASE) and the
pw.x lattice for ibrav != 0 (ASE refuses it); both are implemented here.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import re
import warnings
import numpy as np

# User-facing name -> internal reader name (ASE format, or "siesta-fdf").
STRUCTURE_FORMATS = {
    "cif": "cif",
    "vasp": "vasp",
    "vasp-out": "vasp-out",
    "vasp-xml": "vasp-xml",
    "qe": "qe-in",
    "qe-out": "espresso-out",
    "siesta": "siesta-fdf",
    "siesta-xv": "siesta-xv",
    "siesta-struct-out": "struct_out",
    "aims": "aims",
    "aims-out": "aims-output",
}
_ALIASES = {
    "poscar": "vasp", "contcar": "vasp", "outcar": "vasp-out", "vasprun": "vasp-xml",
    "espresso": "qe", "espresso-in": "qe", "quantum-espresso": "qe", "pwi": "qe",
    "espresso-out": "qe-out", "pwo": "qe-out",
    "fdf": "siesta", "siesta-fdf": "siesta", "xv": "siesta-xv",
    "struct_out": "siesta-struct-out", "struct-out": "siesta-struct-out",
    "fhi-aims": "aims", "geometry.in": "aims", "aims-output": "aims-out",
}
STRUCTURE_FORMAT_CHOICES = ("auto",) + tuple(STRUCTURE_FORMATS)
STRUCTURE_FORMAT_HELP = ("crystal-structure file format (default: auto-detected from the "
                         "file name and content): " + ", ".join(STRUCTURE_FORMATS))


def normalize_structure_format(fmt):
    """Return the user-facing format name, or None for auto-detection."""
    if fmt is None:
        return None
    key = str(fmt).strip().lower()
    if key in ("", "auto"):
        return None
    key = _ALIASES.get(key, key)
    if key not in STRUCTURE_FORMATS:
        raise ValueError(f"Unknown structure format {fmt!r}; use one of: "
                         + ", ".join(STRUCTURE_FORMAT_CHOICES))
    return key


def _by_name(path):
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return "cif"
    if suffix == ".fdf":
        return "siesta"
    if suffix == ".xv":
        return "siesta-xv"
    if suffix == ".struct_out":
        return "siesta-struct-out"
    if suffix == ".pwi":
        return "qe"
    if suffix == ".pwo":
        return "qe-out"
    if name.startswith("vasprun") and suffix == ".xml":
        return "vasp-xml"
    if "outcar" in name:
        return "vasp-out"
    if "poscar" in name or "contcar" in name or suffix in (".vasp", ".poscar"):
        return "vasp"
    if name == "geometry.in" or name.startswith("geometry.in.") or suffix == ".aims":
        return "aims"
    return None


def _floats(tokens, n):
    if len(tokens) < n:
        return False
    try:
        [float(t) for t in tokens[:n]]
    except ValueError:
        return False
    return True


def _by_content(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(400_000)
    except OSError:
        return None
    low = head.lower()
    if "program pwscf" in low:
        return "qe-out"
    if "invoking fhi-aims" in low or "fhi-aims version" in low:
        return "aims-out"
    if "<modeling>" in low:
        return "vasp-xml"
    if re.search(r"&system\b", low) and "atomic_positions" in low:
        return "qe"
    if "_cell_length_a" in low:
        return "cif"
    labels = {_labelize(m) for m in re.findall(r"%block\s+(\S+)", low)}
    if labels & {"atomiccoordinatesandatomicspecies", "chemicalspecieslabel",
                 "latticevectors", "latticeparameters"}:
        return "siesta"
    first = [ln.split()[0] for ln in low.splitlines() if ln.split()]
    if any(t in ("lattice_vector", "atom", "atom_frac") for t in first):
        return "aims"
    lines = head.splitlines()
    if len(lines) >= 7:
        scale = lines[1].split()
        if (_floats(scale, 1) and len(scale) in (1, 3)
                and all(_floats(lines[i].split(), 3) for i in (2, 3, 4))):
            return "vasp"
    return None


def detect_structure_format(path):
    """User-facing format name of a structure file (raises if unknown)."""
    path = Path(path)
    fmt = _by_name(path) or _by_content(path)
    if fmt is None:
        raise ValueError(
            f"Cannot recognize the structure format of {path}. Use a conventional file "
            "name (*.cif, POSCAR/CONTCAR/*.vasp, *.pwi/*.pwo, *.fdf/*.XV/*.STRUCT_OUT, "
            "geometry.in) or force it with --structure-format / structure_format: "
            + ", ".join(STRUCTURE_FORMATS))
    return fmt


# --------------------------------------------------------------------------
# SIESTA .fdf reader (ASE only reads .XV and .STRUCT_OUT)
# --------------------------------------------------------------------------
_BOHR = 0.529177210903  # Angstrom
_LENGTH_UNITS = {"ang": 1.0, "angstrom": 1.0, "bohr": _BOHR, "au": _BOHR,
                 "nm": 10.0, "m": 1e10, "cm": 1e8}


def _labelize(label):
    """FDF labels ignore case, '-', '_' and '.'."""
    return re.sub(r"[-_.]", "", str(label)).lower()


def _strip_comment(line):
    for mark in ("#", "!", ";"):
        if mark in line:
            line = line.split(mark, 1)[0]
    return line.strip()


def _fdf_lines(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Recursive %include of {path}")
    seen = seen | {path}
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        tok = line.split()
        if tok[0].lower() == "%include":
            if len(tok) < 2:
                raise ValueError(f"{path}: %include without a file name")
            out += _fdf_lines(path.parent / tok[1], seen)
            continue
        if tok[0].lower() == "%block" and len(tok) >= 4 and tok[2] == "<":
            # %block Name < file  : block body read from another file
            out.append(f"%block {tok[1]}")
            out += _fdf_lines(path.parent / tok[3], seen)
            out.append(f"%endblock {tok[1]}")
            continue
        out.append(line)
    return out


def _parse_fdf(path):
    scalars, blocks = {}, {}
    lines = _fdf_lines(path)
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if tok[0].lower() == "%block":
            if len(tok) < 2:
                raise ValueError(f"{path}: %block without a name")
            name = _labelize(tok[1])
            body = []
            i += 1
            while i < len(lines) and lines[i].split()[0].lower() != "%endblock":
                body.append(lines[i].split())
                i += 1
            if i == len(lines):
                raise ValueError(f"{path}: %block {tok[1]} has no %endblock")
            blocks.setdefault(name, body)  # the first definition wins, as in fdf
        else:
            scalars.setdefault(_labelize(tok[0]), tok[1:])
        i += 1
    return scalars, blocks


def _length(tokens, label, default_unit):
    if not tokens:
        raise ValueError(f"{label} has no value")
    value = float(tokens[0])
    unit = tokens[1].lower() if len(tokens) > 1 else default_unit
    if len(tokens) == 1:
        warnings.warn(f"{label} has no unit; assuming {default_unit} (SIESTA default)")
    if unit not in _LENGTH_UNITS:
        raise ValueError(f"{label}: unsupported length unit {tokens[1]!r}")
    return value * _LENGTH_UNITS[unit]


def read_siesta_fdf(path):
    """ase.Atoms from a SIESTA input .fdf (periodic cell required)."""
    from ase import Atoms
    from ase.geometry import cellpar_to_cell
    path = Path(path)
    scalars, blocks = _parse_fdf(path)
    if "latticeconstant" not in scalars:
        raise ValueError(f"{path}: LatticeConstant is required (the fitters need an explicit periodic cell)")
    alat = _length(scalars["latticeconstant"], "LatticeConstant", "bohr")
    if "latticevectors" in blocks:
        rows = blocks["latticevectors"]
        if len(rows) != 3 or not all(_floats(r, 3) for r in rows):
            raise ValueError(f"{path}: LatticeVectors must have three rows of three numbers")
        cell = alat * np.array([[float(x) for x in r[:3]] for r in rows])
    elif "latticeparameters" in blocks:
        rows = blocks["latticeparameters"]
        if len(rows) != 1 or not _floats(rows[0], 6):
            raise ValueError(f"{path}: LatticeParameters must be one row: a b c alpha beta gamma")
        a, b, c, al, be, ga = (float(x) for x in rows[0][:6])
        cell = cellpar_to_cell([a * alat, b * alat, c * alat, al, be, ga])
    else:
        cell = alat * np.eye(3)  # SIESTA default: cubic cell of side LatticeConstant

    if "chemicalspecieslabel" not in blocks:
        raise ValueError(f"{path}: missing %block ChemicalSpeciesLabel")
    species = {}
    for r in blocks["chemicalspecieslabel"]:
        if len(r) < 2:
            raise ValueError(f"{path}: bad ChemicalSpeciesLabel line {' '.join(r)!r}")
        idx, z = int(r[0]), int(r[1])
        if not 1 <= z <= 118:
            raise ValueError(f"{path}: species {idx} has Z={z}; ghost/synthetic atoms are not supported")
        species[idx] = z

    coords = blocks.get("atomiccoordinatesandatomicspecies")
    if not coords:
        if "zmatrix" in blocks:
            raise ValueError(f"{path}: Zmatrix geometries are not supported; use AtomicCoordinatesAndAtomicSpecies")
        raise ValueError(f"{path}: missing %block AtomicCoordinatesAndAtomicSpecies")
    xyz, numbers = [], []
    for r in coords:
        if not _floats(r, 3) or len(r) < 4:
            raise ValueError(f"{path}: bad coordinate line {' '.join(r)!r}")
        idx = int(r[3])
        if idx not in species:
            raise ValueError(f"{path}: species index {idx} is not in ChemicalSpeciesLabel")
        xyz.append([float(x) for x in r[:3]])
        numbers.append(species[idx])
    xyz = np.array(xyz)
    if "numberofatoms" in scalars and int(scalars["numberofatoms"][0]) != len(xyz):
        raise ValueError(f"{path}: NumberOfAtoms={scalars['numberofatoms'][0]} but "
                         f"{len(xyz)} coordinates were given")

    if "atomiccoordinatesorigin" in blocks:
        r = blocks["atomiccoordinatesorigin"][0]
        if not _floats(r, 3):
            raise ValueError(f"{path}: only a numeric AtomicCoordinatesOrigin is supported")
        xyz = xyz + np.array([float(x) for x in r[:3]])

    fmt = _labelize(scalars.get("atomiccoordinatesformat", ["NotScaledCartesianBohr"])[0])
    if fmt in ("fractional", "scaledbylatticevectors"):
        return Atoms(numbers=numbers, scaled_positions=xyz, cell=cell, pbc=True)
    if fmt in ("bohr", "notscaledcartesianbohr"):
        pos = xyz * _BOHR
    elif fmt in ("ang", "notscaledcartesianang"):
        pos = xyz
    elif fmt == "scaledcartesian":
        pos = xyz * alat
    else:
        raise ValueError(f"{path}: unsupported AtomicCoordinatesFormat "
                         f"{scalars['atomiccoordinatesformat'][0]!r}")
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)


# --------------------------------------------------------------------------
# Quantum ESPRESSO pw.x input (ASE only reads ibrav = 0)
# --------------------------------------------------------------------------
def _qe_celldm(system):
    """celldm(1..6) from &SYSTEM (celldm in Bohr, or A, B, C, cosAB, cosAC, cosBC in Angstrom)."""
    ibrav = int(system["ibrav"])
    cd = [float(system.get(f"celldm({i})", 0.0)) for i in range(1, 7)]
    if cd[0] == 0.0 and "a" in system:
        a = float(system["a"])
        cd = [a / _BOHR, float(system.get("b", 0.0)) / a, float(system.get("c", 0.0)) / a, 0.0, 0.0, 0.0]
        cosab, cosac, cosbc = (float(system.get(k, 0.0)) for k in ("cosab", "cosac", "cosbc"))
        if ibrav == 14:
            cd[3:] = [cosbc, cosac, cosab]
        elif ibrav in (-12, -13):
            cd[4] = cosac
        else:
            cd[3] = cosab
    elif cd[0] != 0.0 and "a" in system:
        raise ValueError("&SYSTEM gives both celldm and A; QE forbids this")
    if cd[0] <= 0.0:
        raise ValueError("&SYSTEM needs celldm(1) or A")
    return cd


def qe_ibrav_cell(ibrav, celldm):
    """Lattice vectors (Angstrom, rows) of pw.x for a Bravais index, as in QE latgen.f90."""
    a = celldm[0] * _BOHR
    ba, ca = celldm[1], celldm[2]
    if ibrav == 1:
        v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif ibrav == 2:
        v = [[-.5, 0, .5], [0, .5, .5], [-.5, .5, 0]]
    elif ibrav == 3:
        v = [[.5, .5, .5], [-.5, .5, .5], [-.5, -.5, .5]]
    elif ibrav == -3:
        v = [[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]]
    elif ibrav == 4:
        v = [[1, 0, 0], [-.5, np.sqrt(3) / 2, 0], [0, 0, ca]]
    elif ibrav in (5, -5):
        c = celldm[3]
        if not -0.5 < c < 1.0:
            raise ValueError("ibrav=5/-5 needs -0.5 < celldm(4) = cos(alpha) < 1")
        tx, ty, tz = np.sqrt((1 - c) / 2), np.sqrt((1 - c) / 6), np.sqrt((1 + 2 * c) / 3)
        if ibrav == 5:
            v = [[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]]
        else:
            u, w = tz - 2 * np.sqrt(2) * ty, tz + np.sqrt(2) * ty
            v = (np.array([[u, w, w], [w, u, w], [w, w, u]]) / np.sqrt(3)).tolist()
    elif ibrav == 6:
        v = [[1, 0, 0], [0, 1, 0], [0, 0, ca]]
    elif ibrav == 7:
        v = [[.5, -.5, ca / 2], [.5, .5, ca / 2], [-.5, -.5, ca / 2]]
    elif ibrav == 8:
        v = [[1, 0, 0], [0, ba, 0], [0, 0, ca]]
    elif ibrav == 9:
        v = [[.5, ba / 2, 0], [-.5, ba / 2, 0], [0, 0, ca]]
    elif ibrav == -9:
        v = [[.5, -ba / 2, 0], [.5, ba / 2, 0], [0, 0, ca]]
    elif ibrav == 91:
        v = [[1, 0, 0], [0, ba / 2, -ca / 2], [0, ba / 2, ca / 2]]
    elif ibrav == 10:
        v = [[.5, 0, ca / 2], [.5, ba / 2, 0], [0, ba / 2, ca / 2]]
    elif ibrav == 11:
        v = [[.5, ba / 2, ca / 2], [-.5, ba / 2, ca / 2], [-.5, -ba / 2, ca / 2]]
    elif ibrav in (12, 13):
        g = celldm[3]
        sg = np.sqrt(1 - g * g)
        if ibrav == 12:
            v = [[1, 0, 0], [ba * g, ba * sg, 0], [0, 0, ca]]
        else:
            v = [[.5, 0, -ca / 2], [ba * g, ba * sg, 0], [.5, 0, ca / 2]]
    elif ibrav in (-12, -13):
        b = celldm[4]
        sb = np.sqrt(1 - b * b)
        if ibrav == -12:
            v = [[1, 0, 0], [0, ba, 0], [ca * b, 0, ca * sb]]
        else:
            v = [[.5, ba / 2, 0], [-.5, ba / 2, 0], [ca * b, 0, ca * sb]]
    elif ibrav == 14:
        cal, cbe, cga = celldm[3], celldm[4], celldm[5]
        sga = np.sqrt(1 - cga * cga)
        term = 1 + 2 * cal * cbe * cga - cal ** 2 - cbe ** 2 - cga ** 2
        if term <= 0:
            raise ValueError("ibrav=14: celldm(4:6) do not form a valid cell")
        v = [[1, 0, 0], [ba * cga, ba * sga, 0],
             [ca * cbe, ca * (cal - cbe * cga) / sga, ca * np.sqrt(term) / sga]]
    else:
        raise ValueError(f"ibrav={ibrav} is not supported")
    return a * np.array(v, float)


def read_qe_input(path):
    """ase.Atoms from a pw.x input, including ibrav != 0 (celldm or A/B/C/cos*)."""
    from ase import Atoms
    from ase.io.espresso import (read_fortran_namelist, get_cell_parameters,
                                 get_atomic_positions, label_to_symbol)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data, cards = read_fortran_namelist(fh)
    system = data.get("system")
    if not system:
        raise ValueError("no &SYSTEM namelist")
    if "ibrav" not in system or "nat" not in system:
        raise ValueError("&SYSTEM needs ibrav and nat")
    ibrav = int(system["ibrav"])
    if ibrav == 0:
        if "celldm(1)" in system:
            alat = float(system["celldm(1)"]) * _BOHR
        elif "a" in system:
            alat = float(system["a"])
        else:
            alat = None
        cell, _ = get_cell_parameters(cards, alat=alat)
        if cell is None:
            raise ValueError("ibrav=0 needs a CELL_PARAMETERS card")
    else:
        cd = _qe_celldm(system)
        alat = cd[0] * _BOHR
        cell = qe_ibrav_cell(ibrav, cd)
    upper = [c.strip().upper() for c in cards]
    if any(c.startswith("ATOMIC_POSITIONS") and "CRYSTAL_SG" in c for c in upper):
        raise ValueError("ATOMIC_POSITIONS crystal_sg (space-group input) is not supported")
    positions = get_atomic_positions(cards, n_atoms=int(system["nat"]), cell=cell, alat=alat)
    if positions is None:
        raise ValueError("no ATOMIC_POSITIONS card")
    symbols = [label_to_symbol(p[0]) for p in positions]
    return Atoms(symbols=symbols, positions=[p[1] for p in positions], cell=cell, pbc=True)


# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _read_cached(path_str, mtime_ns, size, fmt):
    path = Path(path_str)
    reader = STRUCTURE_FORMATS[fmt]
    if reader == "siesta-fdf":
        atoms = read_siesta_fdf(path)
    elif reader == "qe-in":
        atoms = read_qe_input(path)
    else:
        from ase.io import read
        with warnings.catch_warnings():
            # ASE announces that FHI-aims IO will move to a plugin; not actionable here.
            warnings.filterwarnings("ignore", message=".*FHI-aims IO is moving.*")
            atoms = read(path, format=reader)  # last image for outputs
    if len(atoms) == 0:
        raise ValueError(f"{path}: no atoms read")
    cell = np.asarray(atoms.cell)
    if atoms.cell.rank != 3 or abs(np.linalg.det(cell)) < 1e-6:
        raise ValueError(
            f"{path}: no full 3x3 periodic cell (molecule?). YAeHMOP band fitting needs "
            "a periodic cell; for 2D/1D materials give the vacuum lattice vectors too.")
    return atoms


def read_structure(path, structure_format=None):
    """Read a crystal structure from any supported file into ase.Atoms."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Structure file not found: {path}")
    fmt = normalize_structure_format(structure_format) or detect_structure_format(path)
    st = path.stat()
    try:
        atoms = _read_cached(str(path), st.st_mtime_ns, st.st_size, fmt)
    except (ValueError, KeyError, IndexError, StopIteration) as exc:
        raise ValueError(f"Could not read {path} as {fmt!r}: {exc}") from exc
    return atoms.copy()


def resolved_structure_format(path, structure_format=None):
    """The format actually used for ``path`` (user-facing name)."""
    return normalize_structure_format(structure_format) or detect_structure_format(path)


_GENERIC_STEMS = {"structure", "poscar", "contcar", "outcar", "vasprun", "geometry",
                  "pw", "scf", "nscf", "relax", "vc-relax", "input", "siesta", "aims"}


def structure_label(path):
    """Short material name for titles: file stem, or the folder for generic names."""
    path = Path(path)
    stem = path.stem.lower()
    if stem in _GENERIC_STEMS or path.name.lower() in ("poscar", "contcar", "geometry.in"):
        return path.parent.name or path.stem
    return path.stem
