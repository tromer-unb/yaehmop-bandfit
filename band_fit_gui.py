#!/usr/bin/env python3
"""BandFit GUI: desktop interface (Qt) for every command of yaehmop-bandfit.

    conda activate yaehmop-bandfit
    python band_fit_gui.py

Tabs:
  Fit to image         structure file + picture (or CSV) of the bands -> EHT fit
  Fit to DFT           folder of a DFT band calculation (VASP, QE, SIESTA, FHI-aims,
                       Wannier90); gapped, spin-polarized or metallic references
  Spin-orbit           soc_fit.py: lambda L.S on top of a spinless fit
  Spin combine         spin_combine.py: join spin-up and spin-down fits
  Frozen transfer      frozen_transfer.py
  Family fit           family_fit.py
  Compare strategies   validation/compare_strategies.py
  Image tools          prepare_band_png, inspect_png, digitize_bands, strategy_advisor

Every command-line option of each command is in the form of its tab: the forms are
built from the commands' own argparse definitions (read in a separate process at
start-up), so new options appear here automatically. Fits run as separate processes,
several at a time; the Runs panel shows the log, the live loss, the figures and the
metrics of each run.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import runpy
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TOOLS = {
    "bands_fit": "USER_WORKFLOW/bands_fit.py",
    "soc_fit": "USER_WORKFLOW/soc_fit.py",
    "spin_combine": "USER_WORKFLOW/spin_combine.py",
    "frozen_transfer": "USER_WORKFLOW/frozen_transfer.py",
    "family_fit": "USER_WORKFLOW/family_fit.py",
    "compare_strategies": "validation/compare_strategies.py",
    "prepare_band_png": "USER_WORKFLOW/prepare_band_png.py",
    "inspect_png": "USER_WORKFLOW/inspect_png.py",
    "digitize_bands": "USER_WORKFLOW/digitize_bands.py",
    "strategy_advisor": "USER_WORKFLOW/strategy_advisor.py",
}


# ============================================================================
# 1. Option specs, read from the argparse parsers in a separate process
#    (no Qt here: the scripts select the Agg backend of matplotlib)
# ============================================================================
class _Captured(Exception):
    pass


def _capture_parser(script):
    orig = argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args

    def grab(self, *a, **k):
        raise _Captured(self)
    argparse.ArgumentParser.parse_args = grab
    argparse.ArgumentParser.parse_known_args = grab
    old_argv, old_path = sys.argv, list(sys.path)
    sys.argv = [str(script)]
    sys.path[:0] = [str(Path(script).parent), str(ROOT / "parameter_fitter"), str(ROOT / "USER_WORKFLOW")]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except _Captured as c:
        return c.args[0]
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = orig
        sys.argv, sys.path[:] = old_argv, old_path
    raise RuntimeError(f"{script}: no argument parser found")


def _jsonable(v):
    if v is None or v is argparse.SUPPRESS:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v)


def _action_spec(a, group):
    if isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction)):
        kind = "flag"
    elif isinstance(a, argparse._AppendAction):
        kind = "append"
    else:
        kind = "value"
    helptext = a.help or ""
    try:
        helptext = helptext % dict(default=a.default, prog="")
    except (TypeError, ValueError, KeyError):
        pass
    t = a.type
    return dict(dest=a.dest, opts=list(a.option_strings), help=helptext,
                choices=[str(c) for c in a.choices] if a.choices else None,
                type=getattr(t, "__name__", None) if t else None,
                default=_jsonable(a.default), nargs=a.nargs if isinstance(a.nargs, (str, type(None))) else str(a.nargs),
                required=bool(a.required or not a.option_strings), kind=kind, group=group,
                metavar=_jsonable(a.metavar))


def dump_specs():
    specs = {}
    for key, rel in TOOLS.items():
        try:
            p = _capture_parser(ROOT / rel)
        except Exception as exc:          # a broken tool must not stop the GUI
            specs[key] = dict(error=str(exc), actions=[])
            continue
        acts = []
        for g in p._action_groups:
            title = g.title or ""
            for a in g._group_actions:
                if isinstance(a, argparse._HelpAction):
                    continue
                acts.append(_action_spec(a, title))
        specs[key] = dict(description=(p.description or "").strip().split("\n")[0], actions=acts)
    sys.path[:0] = [str(ROOT / "parameter_fitter")]
    try:
        from dataclasses import fields
        from band_objective import WindowSettings
        specs["_objective"] = [dict(name=f.name, default=_jsonable(f.default)) for f in fields(WindowSettings)]
    except Exception as exc:
        specs["_objective"] = []
        specs["_objective_error"] = str(exc)
    print(json.dumps(specs))


# ============================================================================
# 2. INI schema of bands_fit (sections and keys read by the command)
# ============================================================================
CODES = ["vasp", "qe", "siesta", "aims", "wannier90"]
INI_SCHEMA = {
    "reference": [
        ("image", "file", "band picture (PNG/JPEG/TIFF/WebP) or numeric CSV"),
        ("mode", ["auto", "layered", "colors", "mono", "csv"], "how the picture is read"),
        ("emin", "float", "energy at the bottom of the plot / of the full comparison figure (eV)"),
        ("emax", "float", "energy at the top of the plot / of the full comparison figure (eV)"),
        ("trace_nx", "int", "columns traced in the picture"),
        ("min_coverage", "float", "minimum fraction of columns where every band is found"),
        ("bbox", "str", "plot box in pixels: L,T,R,B"),
        ("colors", "str", "RGB colors of the bands, e.g. #1f77b4,#d62728"),
        ("color_tolerance", "int", "color distance accepted for a band"),
        ("dft_dir", "dir", "folder of the DFT band-structure calculation"),
        ("dft_code", CODES, "DFT package of dft_dir"),
        ("dft_file", "str", "main band file inside dft_dir when there are several"),
        ("valence_bands", "int", "occupied DFT bands, if the automatic count is wrong"),
        ("spin_channel", ["up", "down", "average"],
         "one channel of a collinear spin-polarized run; average = mean of both (spurious tiny magnetization)"),
        ("semicore", "str", "lowest DFT bands dropped (semicore states absent from the EHT basis): auto, an integer or 0"),
        ("fermi", "float", "Fermi level (eV) replacing the one of the band files (metals: use the SCF one)"),
    ],
    "band_path": [
        ("dimension", ["1", "2", "3"], "periodic dimension"),
        ("points", "str", "k points (fractional), ';' between points: 0,0; 0.5,0; 0.3333,0.3333; 0,0"),
        ("labels", "str", "labels of the k points: G,M,K,G"),
        ("points_per_segment", "int", "YAeHMOP points per straight segment"),
    ],
    "fit": [
        ("name", "str", "material label"),
        ("objective", ["window", "frontier", "metal"],
         "window: bands in an energy window (DFT default); frontier: n_pairs bands; metal: no gap, aligned at E_F"),
        ("n_pairs", ["1", "2", "3", "4"], "valence and conduction bands each (figure; frontier objective)"),
        ("budget", "int", "YAeHMOP objective calls"),
        ("optimizer", "OPTIMIZERS", "optimization method"),
        ("polish", ["nelder-mead", "powell"], "local refinement after a global optimizer"),
        ("basis_add", "str", "extra EHT shells, e.g. 'Al:d' or 'S:d H:p' or auto"),
        ("basis_dz", "str", "shells made double-zeta (inner exponent + fitted diffuse tail), e.g. 'S:p Te:p'; "
                            "s/p need the bind of github.com/ac-dias/yaehmop"),
        ("start_from", "dir", "fitting/ folder of an earlier fit to start from"),
        ("overlap_radius", "str", "lattice images in each periodic direction (1..8), or auto: the smallest radius "
                                  "whose bands agree with overlap_radius_max within overlap_radius_tol"),
        ("overlap_radius_max", "int", "auto radius: reference (largest) radius; default 5"),
        ("overlap_radius_tol", "float", "auto radius: largest band difference accepted (eV); default 1e-3"),
        ("parallel", "str", "candidates evaluated at the same time by DE/CMA-ES/GA/Bayesian/alternating; "
                            "auto = CPUs / systems"),
        ("fast_budget", "int", "optimizer alternating: evaluations of the fast Hii/K stages (cached S(k), numpy); "
                               "default 10 x budget"),
        ("alternating_rounds", "int", "optimizer alternating: rounds of (all parameters with YAeHMOP, then Hii/K "
                                      "only); default 3"),
        ("overlap_min_eig", "float", "smallest eigenvalue of S(k) accepted (H, S exportable); default 1e-3"),
        ("gap_weight", "float", "frontier objective: weight of the gap error"),
        ("reg_weight", "float", "weight of the distance to the starting parameters"),
        ("structure_format", "FORMATS", "format of the structure file (default: detected)"),
    ],
}
# Values offered in the drop-down of free-text fields (the field stays editable)
SUGGEST = {
    "basis_add": ["auto", "Al:d", "S:d", "Se:d", "Si:d", "C:d", "Zn:d", "S:d H:p"],
    "budget": ["300", "1000", "2000", "3000", "5000", "10000"],
    "emin": ["-6", "-4", "-3", "-2", "-10"], "emax": ["6", "4", "3", "2", "8"],
    "trace_nx": ["151", "201", "301"], "min_coverage": ["0.75", "0.80", "0.90"],
    "colors": ["#1f77b4,#d62728", "#0000ff,#ff0000", "#1f77b4,#ff7f0e", "#000000"],
    "color_tolerance": ["20", "40", "60"],
    "points_per_segment": ["10", "15", "20", "30"],
    "overlap_radius": ["auto", "3", "4", "5", "6"],
    "overlap_radius_max": ["5", "6"], "overlap_radius_tol": ["1e-3", "1e-4", "5e-3"],
    "parallel": ["auto", "1", "4", "8", "12", "24"], "fast_budget": ["2000", "5000", "10000", "20000"],
    "alternating_rounds": ["3", "2", "4", "5"], "workers": ["1", "2", "4", "8"],
    "basis_dz": ["S:p", "Se:p", "Te:p", "S:p Te:p", "O:p", "Ti:d"],
    "semicore": ["auto", "0", "1", "2", "4", "6"],
    "segment_weights": ["0:3", "0:3 3:3", "0:4 3:4", "1:2"],
    "overlap_min_eig": ["1e-3", "1e-4", "0"],
    "gap_weight": ["0.35", "0", "1"], "reg_weight": ["0.04", "0", "0.1"],
    "window_min": ["-6", "-4", "-2.6", "-9"], "window_max": ["4", "3", "6", "12"],
    "n_cond": ["1", "2", "3", "4", "6", "8"],
    "energy_scale": ["2", "1", "1.5", "4"], "far_weight": ["0.05", "0.15", "0.3"],
    "frontier_scale": ["1", "0.5", "0.7"],
    "band_weights": ["v1:10 c1:10", "v1:10 c1:10 c2:10 c3:10", "v1:10 v2:6 v3:3 c1:10 c2:6 c3:3",
                     "v1:15 c1:15 v2:5 c2:5"],
    "w_bands": ["0", "6", "8"], "w_slope": ["0", "1", "2", "4"], "w_mass": ["0", "1"],
    "soc_shells": ["Mo:d S:p", "W:d Se:p", "I:p", "In:p S:p"],
    "w_split": ["100", "20", "400"], "lambda_max": ["1.5", "1.0", "0.5"],
    "split_pairs": ["1", "2", "3"], "maxiter": ["60", "100", "200"],
    "k": ["1.75", "2.0"], "seed": ["20261014", "1", "2", "3"], "fermi": [],
    "dft_file": ["vasprun.xml", "EIGENVAL"],
    "name": [],
}
# k paths offered in the "k path" tab (fractional coordinates of the reciprocal cell, ASE conventions)
KPATHS = {
    "2D hexagonal  Γ-M-K-Γ": ("2", "0,0; 0.5,0; 0.3333333333,0.3333333333; 0,0", "G,M,K,G"),
    "2D square  Γ-X-M-Γ": ("2", "0,0; 0.5,0; 0.5,0.5; 0,0", "G,X,M,G"),
    "2D rectangular  Γ-X-S-Y-Γ": ("2", "0,0; 0.5,0; 0.5,0.5; 0,0.5; 0,0", "G,X,S,Y,G"),
    "3D fcc  Γ-X-W-K-Γ-L": ("3", "0,0,0; 0.5,0,0.5; 0.5,0.25,0.75; 0.375,0.375,0.75; 0,0,0; 0.5,0.5,0.5",
                           "G,X,W,K,G,L"),
    "3D bcc  Γ-H-N-Γ-P-H": ("3", "0,0,0; 0.5,-0.5,0.5; 0,0,0.5; 0,0,0; 0.25,0.25,0.25; 0.5,-0.5,0.5",
                           "G,H,N,G,P,H"),
    "3D simple cubic  Γ-X-M-Γ-R-X": ("3", "0,0,0; 0,0.5,0; 0.5,0.5,0; 0,0,0; 0.5,0.5,0.5; 0,0.5,0",
                                    "G,X,M,G,R,X"),
    "3D hexagonal  Γ-M-K-Γ-A": ("3", "0,0,0; 0.5,0,0; 0.3333333333,0.3333333333,0; 0,0,0; 0,0,0.5", "G,M,K,G,A"),
    "1D  Γ-X": ("1", "0; 0.5", "G,X"),
}
# validation examples for compare_strategies --only
SUGGEST["only"] = sorted(f"{q.parent.name}/{q.name}" for q in (ROOT / "validation").glob("*/*")
                         if q.is_dir() and (q / "fitting").is_dir())
LOGO = ROOT / "assets/logo/bandfit_logo_dark.png"
ICON = ROOT / "assets/logo/bandfit_icon_256.png"
IMAGE_REF_KEYS = ["image", "mode", "emin", "emax", "trace_nx", "min_coverage", "bbox", "colors", "color_tolerance"]
DFT_REF_KEYS = ["dft_dir", "dft_code", "dft_file", "valence_bands", "spin_channel", "semicore", "fermi", "emin", "emax"]


# ============================================================================
# 3. Family manifest (family_fit.py) <-> editor rows, without Qt (tested in tests/test_gui_specs.py)
# ============================================================================
FAMILY_COLS = ["name", "role", "dft_dir", "dft_code", "structure", "settings", "reference", "weight",
               "spin_channel", "metallic", "semicore", "overlap_radius", "objective_settings"]
FAMILY_GLOBAL = [  # key, kind, help
    ("n_pairs", ["1", "2", "3", "4"], "valence and conduction bands each (figures; frontier objective)"),
    ("objective", ["window", "metal", "frontier"],
     "default: metal if a training member is metallic, window for DFT members, frontier for pictures"),
    ("overlap_radius", "str", "lattice images per periodic direction (1..8) or auto (per system: column)"),
    ("basis_add", "str", "extra EHT shells, e.g. 'S:d' or auto"),
    ("basis_dz", "str", "double-zeta shells, e.g. 'S:p' (s/p: bind of github.com/ac-dias/yaehmop)"),
    ("start_from", "dir", "fitting/ or fit/ folder (or a parameter table) to start from"),
    ("objective_settings", "str", "window/metal weights for every member, e.g.  w_split: 100, n_cond: 4"),
    ("parallel", "str", "candidates evaluated at the same time (auto = CPUs / systems)"),
    ("fast_budget", "int", "optimizer alternating: fast Hii/K evaluations (default 10 x budget)"),
    ("alternating_rounds", "int", "optimizer alternating: rounds (default 3)"),
]
FAMILY_DEFAULTS = [  # keys of `defaults:` (apply to every DFT member without its own value)
    ("dft_code", CODES, "DFT package of the members"),
    ("spin_channel", ["up", "down", "average"], "spin channel of spin-polarized members"),
    ("semicore", "str", "semicore bands dropped: auto (default), an integer or 0"),
    ("points_per_segment", "int", "YAeHMOP points per straight segment of the DFT path (default 20)"),
]


def _settings_text(d):
    """{'w_split': 100, 'segment_weights': '0:3'} -> "w_split: 100, segment_weights: '0:3'"."""
    if not d:
        return ""
    import yaml
    return yaml.safe_dump(dict(d), default_flow_style=True, sort_keys=False).strip().strip("{}").strip()


def _parse_settings(text):
    """Inverse of _settings_text; '' -> None. Raises ValueError on a malformed text."""
    text = (text or "").strip()
    if not text:
        return None
    import yaml
    try:
        d = yaml.safe_load("{" + text.strip("{}") + "}")
    except yaml.YAMLError as exc:
        raise ValueError(f"objective settings '{text}': {exc}") from None
    if not isinstance(d, dict):
        raise ValueError(f"objective settings '{text}' must be key: value pairs")
    return d


def family_manifest(glob, defaults, rows):
    """Manifest dict for family_fit.py from the editor values (strings; empty = not set)."""
    m = {}
    for key, _, _ in FAMILY_GLOBAL:
        v = (glob.get(key) or "").strip()
        if not v:
            continue
        if key == "objective_settings":
            m[key] = _parse_settings(v)
        elif key in ("n_pairs", "fast_budget", "alternating_rounds"):
            m[key] = int(v)
        elif key == "overlap_radius":
            m[key] = v if v.lower() == "auto" else int(v)
        else:
            m[key] = v
    d = {k: (int(v) if k == "points_per_segment" else v) for k, v in
         ((k, (defaults.get(k) or "").strip()) for k, _, _ in FAMILY_DEFAULTS) if v}
    if d:
        m["defaults"] = d
    systems = []
    for i, row in enumerate(rows, 1):
        r = {k: str(row.get(k) or "").strip() for k in FAMILY_COLS}
        if not any(r.values()):
            continue
        if not r["name"]:
            raise ValueError(f"system {i}: name is empty")
        s = {"name": r["name"]}
        if r["role"] and r["role"] != "train":
            s["role"] = r["role"]
        for k in ("dft_dir", "dft_code", "structure", "settings", "reference", "spin_channel", "semicore"):
            if r[k]:
                s[k] = r[k]
        if r["weight"]:
            s["weight"] = float(r["weight"])
        if r["metallic"] in ("true", "false"):
            s["metallic"] = r["metallic"] == "true"
        if r["overlap_radius"]:
            s["overlap_radius"] = r["overlap_radius"] if r["overlap_radius"].lower() == "auto" else int(r["overlap_radius"])
        if r["objective_settings"]:
            s["objective_settings"] = _parse_settings(r["objective_settings"])
        if not s.get("dft_dir") and not (s.get("structure") and s.get("settings")):
            raise ValueError(f"system {r['name']}: give dft_dir (DFT member) or structure + settings (picture member)")
        systems.append(s)
    if not systems:
        raise ValueError("add at least one system")
    m["systems"] = systems
    return m


def family_rows(manifest, base):
    """(globals, defaults, rows) editor values of a manifest; relative paths made absolute from `base`."""
    base = Path(base)

    def absolute(v):
        return str(v) if Path(str(v)).expanduser().is_absolute() else str((base / str(v)).resolve())
    glob = {}
    for key, _, _ in FAMILY_GLOBAL:
        v = manifest.get(key)
        if v is None:
            continue
        glob[key] = _settings_text(v) if key == "objective_settings" else (absolute(v) if key == "start_from" else str(v))
    defaults = {k: str(v) for k, v in (manifest.get("defaults") or {}).items() if k in dict((a, 1) for a, _, _ in FAMILY_DEFAULTS)}
    rows = []
    for s in manifest.get("systems") or []:
        r = {k: "" for k in FAMILY_COLS}
        for k in FAMILY_COLS:
            v = s.get(k)
            if v is None:
                continue
            if k in ("dft_dir", "structure", "settings", "reference"):
                v = absolute(v)
            elif k == "objective_settings":
                v = _settings_text(v)
            elif k == "metallic":
                v = "true" if v else "false"
            r[k] = str(v)
        r["role"] = r["role"] or "train"
        rows.append(r)
    return glob, defaults, rows


def detect_dft_code(folder):
    """Guess the DFT package of a band-structure folder from its files."""
    f = Path(folder)
    if not f.is_dir():
        return None
    names = [p.name for p in f.iterdir()]
    low = [n.lower() for n in names]
    if any(n.endswith("_band.dat") for n in low):
        return "wannier90"
    if any(n.startswith("vasprun") for n in low) or "eigenval" in low:
        return "vasp"
    if any(n.endswith((".bands", ".fdf", ".struct_out", ".xv")) for n in low):
        return "siesta"
    if "control.in" in low or any(n.startswith("band1") and n.endswith(".out") for n in low):
        return "aims"
    if any(n.endswith((".save", ".pwo", ".pwi")) for n in low) or any((f / n).is_dir() and n.endswith(".save") for n in names):
        return "qe"
    for n in names:                      # pw.x output with another extension
        p = f / n
        if p.is_file() and p.stat().st_size < 50e6:
            try:
                head = p.read_text(errors="ignore")[:4000]
            except OSError:
                continue
            if "Program PWSCF" in head:
                return "qe"
    return None


def vasp_is_spin_polarized(folder):
    for name in ("vasprun.xml", "INCAR", "OUTCAR"):
        p = Path(folder) / name
        if p.is_file():
            txt = p.read_text(errors="ignore")[:400000]
            if 'name="ISPIN">      2' in txt or 'name="ISPIN">    2' in txt or "ISPIN = 2" in txt.replace("ISPIN=2", "ISPIN = 2") \
                    or "ISPIN  =      2" in txt:
                return True
    return False


# ============================================================================
# 3. Qt interface
# ============================================================================
def run_gui():
    # matplotlib (and its FreeType) before Qt: the PySide6 wheels bring an older libfreetype
    # that breaks the conda matplotlib if it is loaded first
    import matplotlib
    from matplotlib import ft2font  # noqa: F401
    matplotlib.use("QtAgg")
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    settings = QtCore.QSettings("yaehmop-bandfit", "band_fit_gui")

    # ------------------------------------------------------------- fields
    class Field(QtCore.QObject):
        """One input widget with get()/set(); kind: str, int, float, file, dir, flag, or a list of choices."""
        changed = QtCore.Signal()

        def __init__(self, kind, *, default=None, help="", placeholder=None, suggest=None):
            super().__init__()
            self.kind = kind
            if suggest and kind in ("str", "int", "float"):
                # free text, with typical values one click away
                w = QtWidgets.QComboBox()
                w.setEditable(True)
                w.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
                w.addItem("")
                w.addItems([str(s) for s in suggest])
                e = w.lineEdit()
                if placeholder is not None:
                    e.setPlaceholderText(placeholder)
                elif default not in (None, "", False):
                    e.setPlaceholderText(f"default: {default}  (▾ typical values)")
                else:
                    e.setPlaceholderText("▾ typical values, or type")
                e.textChanged.connect(lambda *_: self.changed.emit())
                self.widget, self._edit = w, e
            elif isinstance(kind, list):
                w = QtWidgets.QComboBox()
                w.addItem("" if default in (None, "") else "")
                w.setItemText(0, f"(default{': ' + str(default) if default not in (None, '') else ''})")
                for c in kind:
                    w.addItem(str(c))
                w.currentIndexChanged.connect(lambda *_: self.changed.emit())
                self.widget, self._edit = w, None
            elif kind == "flag":
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(default))
                w.toggled.connect(lambda *_: self.changed.emit())
                self.widget, self._edit = w, None
            else:
                e = QtWidgets.QLineEdit()
                if placeholder is not None:
                    e.setPlaceholderText(placeholder)
                elif default not in (None, "", False):
                    e.setPlaceholderText(f"default: {default}")
                e.textChanged.connect(lambda *_: self.changed.emit())
                self._edit = e
                if kind in ("file", "dir", "save"):
                    box = QtWidgets.QWidget()
                    lay = QtWidgets.QHBoxLayout(box)
                    lay.setContentsMargins(0, 0, 0, 0)
                    lay.addWidget(e, 1)
                    b = QtWidgets.QToolButton()
                    b.setText("…")
                    b.setToolTip("Browse")
                    b.clicked.connect(self._browse)
                    lay.addWidget(b)
                    self.widget = box
                else:
                    self.widget = e
            if help:
                self.widget.setToolTip(help)
                if self._edit is not None:
                    self._edit.setToolTip(help)

        def _browse(self):
            start = self.get() or settings.value("last_dir", str(Path.home()))
            if self.kind == "dir":
                p = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "Choose a folder", start)
            elif self.kind == "save":
                p, _ = QtWidgets.QFileDialog.getSaveFileName(self.widget, "Choose a file", start)
            else:
                p, _ = QtWidgets.QFileDialog.getOpenFileName(self.widget, "Choose a file", start)
            if p:
                self.set(p)
                settings.setValue("last_dir", str(Path(p).parent if self.kind != "dir" else Path(p)))

        def get(self):
            if isinstance(self.kind, list):
                i = self.widget.currentIndex()
                return "" if i <= 0 else self.widget.currentText()
            if self.kind == "flag":
                return self.widget.isChecked()
            return self._edit.text().strip()

        def set_suggestions(self, items):
            """Replace the drop-down values of a suggestion field (keeps the text)."""
            if isinstance(self.widget, QtWidgets.QComboBox) and self.widget.isEditable():
                txt = self._edit.text()
                self.widget.blockSignals(True)
                self.widget.clear()
                self.widget.addItem("")
                self.widget.addItems([str(s) for s in items])
                self.widget.blockSignals(False)
                self._edit.setText(txt)

        def set(self, value):
            if isinstance(self.kind, list):
                v = "" if value is None else str(value)
                i = self.widget.findText(v) if v else 0
                self.widget.setCurrentIndex(max(i, 0))
            elif self.kind == "flag":
                self.widget.setChecked(bool(value) and str(value).lower() not in ("0", "false", "no"))
            else:
                self._edit.setText("" if value is None else str(value))

    def path_kind(spec):
        d = spec["dest"]
        if spec["kind"] == "flag":
            return "flag"
        if spec["choices"]:
            return spec["choices"]
        if spec["type"] == "int":
            return "int"
        if spec["type"] == "float":
            return "float"
        dirs = ("dir", "outdir", "work", "from_fitting", "fitting", "up", "down", "start_from")
        if d in dirs or d.endswith("_dir"):
            return "dir"
        files = ("cif", "input_file", "png", "image", "from_fit", "from_parameters", "parameters", "manifest",
                 "source_run",
                 "reference", "settings", "annotations", "template", "convert", "structure")
        if spec["type"] == "Path" or d in files:
            return "file"
        return "str"

    def collapsible(title, inner, expanded=False):
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 4, 0, 0)
        btn = QtWidgets.QToolButton()
        btn.setText(title)
        btn.setCheckable(True)
        btn.setChecked(expanded)
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        btn.setStyleSheet("QToolButton{border:none;font-weight:bold}")
        inner.setVisible(expanded)

        def toggle(on):
            inner.setVisible(on)
            btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        btn.toggled.connect(toggle)
        lay.addWidget(btn)
        lay.addWidget(inner)
        return box

    # ------------------------------------------------------------- argparse form
    class ArgForm(QtWidgets.QWidget):
        """Every option of one command, built from its argparse spec."""

        def __init__(self, spec, *, primary=(), hide=(), advanced_title="All command-line options",
                     expanded=False):
            super().__init__()
            self.spec = spec
            self.fields = {}
            self.actions = {a["dest"]: a for a in spec.get("actions", [])}
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            if spec.get("error"):
                lay.addWidget(QtWidgets.QLabel(f"<i>Options not available: {spec['error']}</i>"))
            prim = [d for d in primary if d in self.actions and d not in hide]
            if prim:
                f = QtWidgets.QFormLayout()
                for d in prim:
                    self._add(f, self.actions[d])
                lay.addLayout(f)
            rest = [a for a in spec.get("actions", []) if a["dest"] not in hide and a["dest"] not in prim]
            if rest:
                inner = QtWidgets.QWidget()
                il = QtWidgets.QVBoxLayout(inner)
                il.setContentsMargins(12, 0, 0, 0)
                groups = {}
                for a in rest:
                    groups.setdefault(a["group"], []).append(a)
                for g, acts in groups.items():
                    if len(groups) > 1 and g not in ("options", "optional arguments"):
                        lab = QtWidgets.QLabel(f"<b>{g}</b>")
                        il.addWidget(lab)
                    f = QtWidgets.QFormLayout()
                    for a in acts:
                        self._add(f, a)
                    il.addLayout(f)
                lay.addWidget(collapsible(advanced_title + f" ({len(rest)})", inner, expanded))

        def _add(self, form, a):
            kind = path_kind(a)
            default = a["default"] if a["kind"] != "flag" else False
            fld = Field(kind, default=default, help=a["help"], suggest=SUGGEST.get(a["dest"]))
            name = a["opts"][-1] if a["opts"] else a["dest"]
            label = QtWidgets.QLabel(name + (" *" if a["required"] and not a["opts"] else ""))
            label.setToolTip(a["help"])
            form.addRow(label, fld.widget)
            self.fields[a["dest"]] = fld

        def set(self, dest, value):
            if dest in self.fields:
                self.fields[dest].set(value)

        def get(self, dest):
            return self.fields[dest].get() if dest in self.fields else ""

        def tokens(self, positional_values=None, skip=()):
            """Command-line tokens of the filled fields (positionals first, in parser order)."""
            positional_values = positional_values or {}
            pos, opts = [], []
            for d, a in self.actions.items():
                if d in skip:
                    continue
                if not a["opts"]:
                    v = positional_values.get(d, self.get(d))
                    if v:
                        pos.append(str(v))
                    elif a["nargs"] not in ("?", "*"):
                        raise ValueError(f"'{d}' is required")
                    continue
                if d not in self.fields:
                    continue
                v = self.fields[d].get()
                opt = a["opts"][-1]
                if a["kind"] == "flag":
                    if v:
                        opts.append(opt)
                elif v not in ("", None):
                    if a["type"] == "int":
                        int(v)
                    elif a["type"] == "float":
                        float(v)
                    if a["nargs"] in ("+", "*") or (a["nargs"] and str(a["nargs"]).isdigit()):
                        opts += [opt] + shlex.split(str(v))
                    elif a["kind"] == "append":
                        for item in shlex.split(str(v)):
                            opts += [opt, item]
                    else:
                        opts += [opt, str(v)]
            return pos + opts

    # ------------------------------------------------------------- INI form
    class IniForm(QtWidgets.QTabWidget):
        """The INI file of bands_fit: [reference], [band_path], [fit], [objective]."""

        def __init__(self, keys, objective_fields, choices):
            super().__init__()
            self.fields = {}
            titles = {"reference": "Reference", "band_path": "k path", "fit": "Fit", "objective": "Objective weights"}
            for sec, wanted in keys.items():
                w = QtWidgets.QWidget()
                f = QtWidgets.QFormLayout(w)
                if sec == "objective":
                    f.addRow(QtWidgets.QLabel("<i>Weights of the window / metal objectives; empty = default. "
                                              "Named bands: band_weights = v1:10 c1:10 c2:5; segments of the k path: "
                                              "segment_weights = 0:3 3:3; neighbouring-band splittings: w_split</i>"))
                    for o in objective_fields:
                        kind = ("str" if o["name"] in ("band_weights", "segment_weights")
                                else ("int" if o["name"] == "n_cond" else "float"))
                        fld = Field(kind, default=o["default"], suggest=SUGGEST.get(o["name"]))
                        f.addRow(o["name"], fld.widget)
                        self.fields[(sec, o["name"])] = fld
                else:
                    schema = {k: (t, h) for k, t, h in INI_SCHEMA[sec]}
                    if sec == "band_path" and "points" in wanted:
                        preset = QtWidgets.QComboBox()
                        preset.addItem("(choose a standard path to fill dimension, points and labels)")
                        preset.addItems(list(KPATHS))
                        preset.setToolTip("Standard k paths in fractional coordinates of the reciprocal cell; "
                                          "check that they match the cell of your structure and of the picture")
                        preset.currentTextChanged.connect(self._kpath)
                        f.addRow("standard path", preset)
                    for k in wanted:
                        t, h = schema[k]
                        if isinstance(t, str) and t in choices:
                            t = choices[t]
                        sug = SUGGEST.get(k)
                        fld = Field(t, help=h, placeholder=(h if t in ("str",) and not sug else None), suggest=sug)
                        lab = QtWidgets.QLabel(k)
                        lab.setToolTip(h)
                        f.addRow(lab, fld.widget)
                        self.fields[(sec, k)] = fld
                sc = QtWidgets.QScrollArea()
                sc.setWidgetResizable(True)
                sc.setWidget(w)
                self.addTab(sc, titles[sec])

        def _kpath(self, name):
            if name in KPATHS:
                dim, pts, labs = KPATHS[name]
                self.set("band_path", "dimension", dim)
                self.set("band_path", "points", pts)
                self.set("band_path", "labels", labs)

        def field(self, sec, key):
            return self.fields.get((sec, key))

        def set(self, sec, key, value):
            if (sec, key) in self.fields:
                self.fields[(sec, key)].set(value)

        def get(self, sec, key):
            f = self.fields.get((sec, key))
            return f.get() if f else ""

        def config(self):
            c = configparser.ConfigParser(interpolation=None)
            for s in ("reference", "band_path", "fit"):
                c.add_section(s)
            for (s, k), f in self.fields.items():
                v = f.get()
                if v not in ("", None, False):
                    if not c.has_section(s):
                        c.add_section(s)
                    c.set(s, k, str(v))
            return c

        def text(self):
            import io
            buf = io.StringIO()
            self.config().write(buf)
            return buf.getvalue()

        def save(self, path):
            Path(path).write_text("# written by band_fit_gui.py\n" + self.text())

        def load(self, path):
            c = configparser.ConfigParser(interpolation=None)
            c.read(path, encoding="utf-8")
            base = Path(path).resolve().parent
            unknown = []
            for (s, k), f in self.fields.items():
                f.set("")
            for s in c.sections():
                for k, v in c[s].items():
                    if (s, k) in self.fields:
                        if self.fields[(s, k)].kind in ("file", "dir") and v and not Path(v).expanduser().is_absolute():
                            v = str((base / v).resolve())
                        self.fields[(s, k)].set(v)
                    else:
                        unknown.append(f"[{s}] {k}")
            return unknown

    # ------------------------------------------------------------- image viewer
    class ImageView(QtWidgets.QScrollArea):
        def __init__(self, fit_height=False):
            super().__init__()
            self.fit_height = fit_height
            self.setWidgetResizable(True)
            self.label = QtWidgets.QLabel("No figure")
            self.label.setAlignment(Qt.AlignCenter)
            self.setWidget(self.label)
            self.pix = None

        def show_file(self, path):
            self.pix = QtGui.QPixmap(str(path)) if path else None
            self._rescale()

        def _rescale(self):
            if self.pix and not self.pix.isNull():
                w = max(120, self.viewport().width() - 8)
                if self.fit_height:
                    h = max(80, self.viewport().height() - 8)
                    self.label.setPixmap(self.pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    self.label.setPixmap(self.pix.scaledToWidth(w, Qt.SmoothTransformation))
            else:
                self.label.setPixmap(QtGui.QPixmap())
                self.label.setText("No figure")

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self._rescale()

    # ------------------------------------------------------------- jobs
    class Job(QtCore.QObject):
        finished = QtCore.Signal(object)

        def __init__(self, name, tool, cmd, outdir=None, fitting_dir=None, kind="run"):
            super().__init__()
            self.name, self.tool, self.cmd, self.kind = name, tool, [str(c) for c in cmd], kind
            self.outdir = Path(outdir) if outdir else None
            self.fitting_dir = Path(fitting_dir) if fitting_dir else None
            self.log = []
            self.status = "queued"
            self.t0 = self.t1 = None
            self.proc = QtCore.QProcess()
            self.proc.setWorkingDirectory(str(ROOT))
            env = QtCore.QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONUNBUFFERED", "1")
            env.insert("MPLBACKEND", "Agg")
            for k in ("YAEHMOP_BIND", "YAEHMOP_PARAM_FILE"):
                v = settings.value(k, "")
                if v:
                    env.insert(k, v)
            self.proc.setProcessEnvironment(env)
            self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self.proc.readyReadStandardOutput.connect(self._read)
            self.proc.finished.connect(self._done)
            self.proc.errorOccurred.connect(self._error)
            self.on_done = []

        def start(self):
            self.t0 = time.time()
            self.status = "running"
            self.log.append("$ " + " ".join(shlex.quote(c) for c in self.cmd) + "\n")
            self.proc.start(self.cmd[0], self.cmd[1:])

        def _read(self):
            self.log.append(bytes(self.proc.readAllStandardOutput()).decode(errors="replace"))

        def _done(self, code, status):
            self.t1 = time.time()
            if self.status != "stopped":
                self.status = "done" if code == 0 else f"failed ({code})"
            self.log.append(f"\n[finished: {self.status}]\n")
            for fn in self.on_done:
                fn(self)
            self.finished.emit(self)

        def _error(self, err):
            if err == QtCore.QProcess.FailedToStart:
                self.status = "failed to start"
                self.log.append("\n[could not start the process]\n")
                self.finished.emit(self)

        def stop(self):
            if self.proc.state() != QtCore.QProcess.NotRunning:
                self.status = "stopped"
                self.proc.kill()

        def elapsed(self):
            if not self.t0:
                return ""
            s = int((self.t1 or time.time()) - self.t0)
            return f"{s // 60}:{s % 60:02d}"

        def progress(self):
            """(calls, loss, best) arrays from the newest progress.csv under outdir."""
            if self.kind == "preview" or not self.outdir or not self.outdir.is_dir():
                return None
            files = sorted(self.outdir.glob("fit*/progress.csv"), key=lambda p: p.stat().st_mtime)
            if not files:
                return None
            calls, loss, best = [], [], []
            try:
                with open(files[-1]) as fh:
                    next(fh, None)
                    for line in fh:
                        p = line.strip().split(",")
                        if len(p) == 3:
                            calls.append(int(p[0])); loss.append(float(p[1])); best.append(float(p[2]))
            except (OSError, ValueError):
                return None
            return calls, loss, best

        def text(self):
            return "".join(self.log)

    class ComparisonDialog(QtWidgets.QDialog):
        """Opens when a fit ends: the fitted EHT bands against the original data."""
        FIGS = [("bands_on_original.png", "On the original picture"),
                ("bands_full_comparison.png", "All bands vs reference"),
                ("bands_comparison.png", "Fitted bands"),
                ("soc_comparison.png", "Spin-orbit")]

        def __init__(self, parent, job):
            super().__init__(parent)
            self.setWindowTitle(f"Fit finished — {job.name}")
            self.resize(1000, 820)
            lay = QtWidgets.QVBoxLayout(self)
            dirs = [d for d in (job.fitting_dir, job.outdir) if d and d.is_dir()]
            if job.tool == "spin_combine" and job.fitting_dir:
                dirs = [job.fitting_dir.parent / "spin_up", job.fitting_dir.parent / "spin_down"]
            tabs = QtWidgets.QTabWidget()
            seen = set()
            family = job.outdir / "family_report.json" if job.outdir else None
            if family is not None and family.is_file():
                # family fit: the overview, then the full comparison of every member
                dirs = [job.outdir]
                if (job.outdir / "family_overview.png").is_file():
                    v = ImageView(fit_height=True)
                    v.show_file(str(job.outdir / "family_overview.png"))
                    tabs.addTab(v, "Family overview")
                for sd in sorted((job.outdir / "systems").glob("*")):
                    for name in ("bands_full_comparison.png", "bands_comparison.png", "bands_on_original.png"):
                        if (sd / name).is_file() and tabs.count() < 16:
                            v = ImageView(fit_height=True)
                            v.show_file(str(sd / name))
                            tabs.addTab(v, sd.name)
                            break
                seen.add("*")
            for d in ([] if "*" in seen else dirs):
                pref = f"{d.name}: " if job.tool == "spin_combine" else ""
                for name, title in self.FIGS:
                    f = d / name
                    if f.is_file() and f.name + pref not in seen:
                        seen.add(f.name + pref)
                        v = ImageView(fit_height=True)
                        v.show_file(str(f))
                        tabs.addTab(v, pref + title)
                for f in sorted(d.glob("*.png")):          # other figures of soc_fit etc.
                    if f.name + pref not in seen and f.name not in dict(self.FIGS) and len(seen) < 8:
                        seen.add(f.name + pref)
                        v = ImageView(fit_height=True)
                        v.show_file(str(f))
                        tabs.addTab(v, pref + f.stem.replace("_", " "))
            self.ok = tabs.count() > 0
            head = QtWidgets.QLabel(self._summary(dirs))
            head.setWordWrap(True)
            lay.addWidget(head)
            lay.addWidget(tabs, 1)
            row = QtWidgets.QHBoxLayout()
            again = QtWidgets.QCheckBox("Show this window when a fit ends")
            again.setChecked(True)
            again.toggled.connect(lambda on: settings.setValue("show_comparison", "true" if on else "false"))
            row.addWidget(again)
            row.addStretch(1)
            b_open = QtWidgets.QPushButton("Open folder")
            b_open.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(dirs[0]))))
            b_close = QtWidgets.QPushButton("Close")
            b_close.clicked.connect(self.close)
            row.addWidget(b_open)
            row.addWidget(b_close)
            lay.addLayout(row)

        @staticmethod
        def _summary(dirs):
            parts = []
            for d in dirs:
                if (d / "family_report.json").is_file():
                    try:
                        rows = RunsPanel.family_rows(json.loads((d / "family_report.json").read_text()))
                    except ValueError:
                        rows = []
                    return "<br>".join(f"<b>{k}</b>: {v:.4g}" if isinstance(v, float) else f"<b>{k}</b>: {v}"
                                       for k, v in rows) or "Family fit finished."
                f = d / "fit_summary.json"
                if not f.is_file():
                    continue
                try:
                    s = json.loads(f.read_text())
                except ValueError:
                    continue
                w = s.get("window_metrics") or {}
                tb = s.get("tight_binding_files") or {}
                bits = []
                for lab, v in (("RMSE", s.get("selected_rmse_eV")), ("gap error", s.get("gap_error_eV")),
                               ("RMS ≤2 eV of E_F", w.get("rms_within_2eV_of_EF_eV")),
                               ("RMS ±1 eV of E_F", w.get("rms_within_1eV_of_EF_eV"))):
                    if isinstance(v, (int, float)) and not (lab == "gap error" and s.get("metallic")):
                        bits.append(f"{lab} {v:+.3f} eV" if lab == "gap error" else f"{lab} {v:.3f} eV")
                if isinstance(tb.get("overlap_min_eigenvalue"), float):
                    bits.append(f"min eig S {tb['overlap_min_eigenvalue']:.3g}")
                if tb.get("warning"):
                    bits.append("<span style='color:#b00020'>" + tb["warning"] + "</span>")
                parts.append(f"<b>{d.name}</b>: " + ", ".join(bits))
            return "<br>".join(parts) or "Fit finished."

    class RunsPanel(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.jobs = []
            lay = QtWidgets.QVBoxLayout(self)
            top = QtWidgets.QHBoxLayout()
            top.addWidget(QtWidgets.QLabel("<b>Runs</b>"))
            top.addStretch(1)
            self.b_stop = QtWidgets.QPushButton("Stop")
            self.b_open = QtWidgets.QPushButton("Open folder")
            self.b_clear = QtWidgets.QPushButton("Remove finished")
            self.b_load = QtWidgets.QPushButton("Open results…")
            self.b_load.setToolTip("Show the figures and metrics of an existing fitting/ or output folder")
            for b in (self.b_stop, self.b_open, self.b_clear, self.b_load):
                top.addWidget(b)
            lay.addLayout(top)
            self.table = QtWidgets.QTableWidget(0, 6)
            self.table.setHorizontalHeaderLabels(["Name", "Command", "Status", "Time", "Calls", "Best loss"])
            self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.verticalHeader().setVisible(False)
            self.table.setMaximumHeight(190)
            lay.addWidget(self.table)
            self.tabs = QtWidgets.QTabWidget()
            # log
            self.logview = QtWidgets.QPlainTextEdit()
            self.logview.setReadOnly(True)
            self.logview.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            self.tabs.addTab(self.logview, "Log")
            # progress
            self.fig = Figure(figsize=(5, 3), tight_layout=True)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.tabs.addTab(self.canvas, "Progress")
            # results
            res = QtWidgets.QWidget()
            rl = QtWidgets.QVBoxLayout(res)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Figure:"))
            self.figsel = QtWidgets.QComboBox()
            row.addWidget(self.figsel, 1)
            self.b_ext = QtWidgets.QPushButton("Open externally")
            row.addWidget(self.b_ext)
            rl.addLayout(row)
            split = QtWidgets.QSplitter(Qt.Vertical)
            self.image = ImageView()
            split.addWidget(self.image)
            self.metrics = QtWidgets.QTableWidget(0, 2)
            self.metrics.setHorizontalHeaderLabels(["Metric", "Value"])
            self.metrics.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
            self.metrics.horizontalHeader().setStretchLastSection(True)
            self.metrics.verticalHeader().setVisible(False)
            split.addWidget(self.metrics)
            split.setSizes([500, 180])
            rl.addWidget(split, 1)
            self.tabs.addTab(res, "Results")
            lay.addWidget(self.tabs, 1)
            self.table.itemSelectionChanged.connect(self.refresh_selected)
            self.figsel.currentIndexChanged.connect(self._show_fig)
            self.b_stop.clicked.connect(self._stop)
            self.b_open.clicked.connect(self._open_folder)
            self.b_clear.clicked.connect(self._clear)
            self.b_load.clicked.connect(self._load_results)
            self.b_ext.clicked.connect(self._open_fig)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(1000)
            self._shown = None
            self._figs_for = None

        # --- jobs
        def add(self, job, start=True):
            self.jobs.append(job)
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c in range(6):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(""))
            job.finished.connect(lambda j: self._finished(j))
            if start:
                job.start()
            self.table.selectRow(r)
            self.tick()
            return job

        def _finished(self, job):
            self.tick()
            if job.status == "done" and job.kind == "run" and settings.value("show_comparison", "true") == "true":
                dlg = ComparisonDialog(self, job)
                if dlg.ok:
                    self._dialogs = [d for d in getattr(self, "_dialogs", []) if d.isVisible()] + [dlg]
                    dlg.show()
            if self.current() is job:
                self._figs_for = None
                self.refresh_selected()
                if job.status == "done":
                    self.tabs.setCurrentIndex(2)

        def current(self):
            rows = self.table.selectionModel().selectedRows()
            return self.jobs[rows[0].row()] if rows else None

        def tick(self):
            for r, j in enumerate(self.jobs):
                vals = [j.name, j.tool, j.status, j.elapsed(), "", ""]
                pr = j.progress() if j.status in ("running",) or j._last_progress is None else j._last_progress
                if pr is not None:
                    j._last_progress = pr
                if pr and pr[0]:
                    vals[4] = str(pr[0][-1])
                    vals[5] = f"{pr[2][-1]:.4g}"
                for c, v in enumerate(vals):
                    it = self.table.item(r, c)
                    if it and it.text() != v:
                        it.setText(v)
                st = self.table.item(r, 2)
                if st:
                    color = {"running": "#1f5fbf", "done": "#2e7d32"}.get(j.status, "#b00020" if "fail" in j.status
                                                                          else "#666")
                    st.setForeground(QtGui.QColor(color))
            j = self.current()
            if j is not None:
                self._update_log(j)
                if j.status == "running":
                    self._plot(j)

        def _update_log(self, j):
            txt = j.text()
            if self._shown is not j or len(txt) != len(self.logview.toPlainText()):
                bar = self.logview.verticalScrollBar()
                at_end = bar.value() >= bar.maximum() - 4
                self.logview.setPlainText(txt)
                if at_end or self._shown is not j:
                    bar.setValue(bar.maximum())
                self._shown = j

        def _plot(self, j):
            pr = j.progress()
            self.ax.clear()
            if pr and pr[0]:
                calls, loss, best = pr
                ok = [(c, l) for c, l in zip(calls, loss) if l < 1e3]
                if ok:
                    self.ax.plot(*zip(*ok), ".", ms=2, color="0.65", label="trial")
                self.ax.plot(calls, best, "-", color="#d62728", lw=1.6, label="best")
                self.ax.set_yscale("log")
                self.ax.set_xlabel("evaluation")
                self.ax.set_ylabel("loss")
                self.ax.legend(frameon=False, fontsize=8)
                self.ax.set_title(f"{j.name}: best {best[-1]:.4g} after {calls[-1]} calls", fontsize=9)
            else:
                self.ax.text(.5, .5, "no progress data (yet)", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()

        def refresh_selected(self):
            j = self.current()
            if j is None:
                return
            self._update_log(j)
            self._plot(j)
            if self._figs_for is not j or j.status == "running":
                self._fill_results(j, keep=(self._figs_for is j and j.status == "running"))
                self._figs_for = j

        # --- results
        def _result_dirs(self, j):
            out = []
            for d in (j.fitting_dir, j.outdir):
                if d and d.is_dir() and d not in out:
                    out.append(d)
            return out

        def _fill_results(self, j, keep=False):
            figs = []
            for d in self._result_dirs(j):
                pats = ["*.png", "*/*.png", "*/*/*.png", "systems/*/fitting/*.png"]
                for pat in pats:
                    for p in sorted(d.glob(pat)):
                        if "/work/" in str(p) or p in figs:
                            continue
                        figs.append(p)
            prefer = ["family_overview.png", "bands_on_original.png", "bands_full_comparison.png", "bands_comparison.png",
                      "extraction_preview.png", "dft_bands_preview.png"]
            figs.sort(key=lambda p: (prefer.index(p.name) if p.name in prefer else 9, len(p.parts)))
            cur = self.figsel.currentData() if keep else None
            self.figsel.blockSignals(True)
            self.figsel.clear()
            base = j.outdir or (j.fitting_dir.parent if j.fitting_dir else None)
            for p in figs:
                try:
                    lab = str(p.relative_to(base)) if base else p.name
                except ValueError:
                    lab = str(p)
                self.figsel.addItem(lab, str(p))
            self.figsel.blockSignals(False)
            i = self.figsel.findData(cur) if cur else -1
            self.figsel.setCurrentIndex(i if i >= 0 else (0 if figs else -1))
            self._show_fig()
            self._fill_metrics(j)

        def _show_fig(self):
            p = self.figsel.currentData()
            self.image.show_file(p if p else None)

        @staticmethod
        def family_rows(report):
            """Metric rows of a family_report.json: shared values, then one row per member."""
            rows = [(k, v) for k, v in (
                ("mode", report.get("mode")), ("K_EHT", report.get("K_EHT")),
                ("optimizer", report.get("optimizer")), ("budget", report.get("budget")),
                ("evaluations", report.get("objective_optimizer_evaluations")),
                ("mean objective without regularization", report.get("training_objective_without_regularization")),
                ("basis added", report.get("basis_added")), ("double-zeta shells", report.get("basis_double_zeta")))
                if v not in (None, "", {})]
            for name, m in (report.get("systems") or {}).items():
                bits = []
                for lab, key in (("window RMS", "window_rms_eV"), ("RMSE", "selected_rmse_eV"),
                                 ("±1 eV of E_F", "rms_within_1eV_of_EF_eV"), ("gap err", "gap_error_eV")):
                    v = m.get(key)
                    if isinstance(v, (int, float)):
                        bits.append(f"{lab} {v:+.3f}" if key == "gap_error_eV" else f"{lab} {v:.3f}")
                rows.append((f"{name} [{m.get('role', '')}]", ", ".join(bits) + " eV" if bits else ""))
            return rows

        def _fill_metrics(self, j):
            rows = []
            summ = None
            src = None
            for d in self._result_dirs(j):
                if (d / "family_report.json").is_file():
                    try:
                        rep = json.loads((d / "family_report.json").read_text())
                    except ValueError:
                        break
                    rows = self.family_rows(rep) + [("summary file", str(d / "family_report.json"))]
                    self.metrics.setRowCount(len(rows))
                    for r, (k, v) in enumerate(rows):
                        self.metrics.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
                        self.metrics.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{v:.4g}" if isinstance(v, float) else str(v)))
                    return
            for d in self._result_dirs(j):
                for name in ("fit_summary.json", "run_summary.json", "evaluation_summary.json", "dft_reference/dft_reference.json"):
                    if (d / name).is_file():
                        try:
                            summ = json.loads((d / name).read_text())
                        except ValueError:
                            continue
                        src = str(d / name)
                        break
                if summ:
                    break
            if summ:
                w = summ.get("window_metrics") or (summ.get("metrics") or {}).get("window") or {}
                tb = summ.get("tight_binding_files") or {}
                dr = summ.get("dft_reference") or {}
                pick = [
                    ("RMSE of the fitted bands (eV)", summ.get("selected_rmse_eV") or (summ.get("metrics") or {}).get("selected_rmse_eV")),
                    ("gap error (eV)", summ.get("gap_error_eV")),
                    ("RMS in the window (eV)", w.get("window_rms_eV")),
                    ("RMS within 2 eV of E_F (eV)", w.get("rms_within_2eV_of_EF_eV")),
                    ("RMS within 1 eV of E_F (eV)", w.get("rms_within_1eV_of_EF_eV")),
                    ("K_EHT", summ.get("K_EHT")),
                    ("objective", summ.get("objective")),
                    ("optimizer", summ.get("optimizer_choice")),
                    ("YAeHMOP calls", summ.get("optimizer_evaluations")),
                    ("basis added", summ.get("basis_added")),
                    ("spin channel", dr.get("spin_channel")),
                    ("metal: EHT electrons in the channel", (tb.get("metal") or {}).get("states_below_EF_per_cell_mesh")),
                    ("smallest eigenvalue of S(k)", tb.get("overlap_min_eigenvalue")),
                    ("warning", tb.get("warning")),
                    ("DFT gap on the path (eV)", summ.get("gap_on_path_eV")),
                    ("occupied DFT bands", summ.get("occupied_bands")),
                    ("DFT Fermi level (eV)", summ.get("fermi_eV")),
                ]
                for k, v in w.items():
                    if k.startswith("rms_") and k[4:6] in ("v1", "v2", "v3", "c1", "c2", "c3", "ba"):
                        pick.append((k, v))
                for k, v in pick:
                    if v not in (None, "", {}):
                        rows.append((k, f"{v:.4g}" if isinstance(v, float) else str(v)))
            if src:
                rows.append(("summary file", src))
            self.metrics.setRowCount(len(rows))
            for r, (k, v) in enumerate(rows):
                self.metrics.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
                self.metrics.setItem(r, 1, QtWidgets.QTableWidgetItem(v))

        # --- buttons
        def _stop(self):
            j = self.current()
            if j:
                j.stop()

        def _open_folder(self):
            j = self.current()
            if j:
                for d in (j.fitting_dir, j.outdir):
                    if d and d.is_dir():
                        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(d)))
                        return

        def _open_fig(self):
            p = self.figsel.currentData()
            if p:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p))

        def _clear(self):
            keep = [j for j in self.jobs if j.status in ("running", "queued")]
            self.jobs = keep
            self.table.setRowCount(0)
            for j in keep:
                r = self.table.rowCount()
                self.table.insertRow(r)
                for c in range(6):
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(""))
            self.tick()

        def _load_results(self):
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Fitting or output folder",
                                                           settings.value("last_dir", str(Path.home())))
            if not d:
                return
            j = Job(Path(d).name, "results", [sys.executable], outdir=d, fitting_dir=d)
            j.status = "loaded"
            self.add(j, start=False)
            self._figs_for = None
            self.refresh_selected()
            self.tabs.setCurrentIndex(2)

    Job._last_progress = None

    # ------------------------------------------------------------- base tab
    class ToolTab(QtWidgets.QWidget):
        def __init__(self, win):
            super().__init__()
            self.win = win

        def scroll(self, inner):
            sc = QtWidgets.QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(inner)
            sc.setFrameShape(QtWidgets.QFrame.NoFrame)
            return sc

        def script(self, key):
            return str(ROOT / TOOLS[key])

        def error(self, msg):
            QtWidgets.QMessageBox.warning(self, "BandFit", str(msg))

    class GenericTab(ToolTab):
        """Any command: its whole argparse form and a Run button."""

        def __init__(self, win, key, intro, primary=()):
            super().__init__(win)
            self.key = key
            spec = win.specs.get(key, {})
            lay = QtWidgets.QVBoxLayout(self)
            inner = QtWidgets.QWidget()
            il = QtWidgets.QVBoxLayout(inner)
            lab = QtWidgets.QLabel(intro + (f"<br><span style='color:gray'>{spec.get('description', '')}</span>"
                                            if spec.get("description") else ""))
            lab.setWordWrap(True)
            il.addWidget(lab)
            self.form = ArgForm(spec, primary=primary or [a["dest"] for a in spec.get("actions", [])],
                                advanced_title="Other options", expanded=True)
            il.addWidget(self.form)
            il.addStretch(1)
            lay.addWidget(self.scroll(inner), 1)
            row = QtWidgets.QHBoxLayout()
            self.cmdview = QtWidgets.QLineEdit()
            self.cmdview.setReadOnly(True)
            self.cmdview.setPlaceholderText("command line")
            b_show = QtWidgets.QPushButton("Show command")
            b_run = QtWidgets.QPushButton("Run")
            b_run.setDefault(True)
            row.addWidget(self.cmdview, 1)
            row.addWidget(b_show)
            row.addWidget(b_run)
            lay.addLayout(row)
            b_show.clicked.connect(lambda: self._cmd(show=True))
            b_run.clicked.connect(self.run)

        def _cmd(self, show=False):
            try:
                cmd = [sys.executable, "-u", self.script(self.key)] + self.form.tokens()
            except ValueError as exc:
                self.error(exc)
                return None
            self.cmdview.setText(" ".join(shlex.quote(c) for c in cmd[2:]))
            return cmd

        def run(self):
            cmd = self._cmd()
            if not cmd:
                return
            out = self.form.get("outdir") or self.form.get("work") or None
            fit = self.form.get("fitting_dir") or None
            name = Path(out).name if out else self.key
            self.win.runs.add(Job(name, self.key, cmd, outdir=out, fitting_dir=fit))

    # ------------------------------------------------------------- bands_fit tabs
    class FitTab(ToolTab):
        """Common part of the image and DFT fit tabs (bands_fit.py)."""
        mode = "image"

        def __init__(self, win):
            super().__init__(win)
            self.spec = win.specs.get("bands_fit", {})
            acts = {a["dest"]: a for a in self.spec.get("actions", [])}
            self.choices = dict(OPTIMIZERS=(acts.get("optimizer") or {}).get("choices") or [],
                                FORMATS=[c for c in ((acts.get("structure_format") or {}).get("choices") or []) if c != "auto"])
            lay = QtWidgets.QVBoxLayout(self)
            inner = QtWidgets.QWidget()
            self.il = QtWidgets.QVBoxLayout(inner)
            intro = QtWidgets.QLabel(self.intro)
            intro.setWordWrap(True)
            self.il.addWidget(intro)
            g = QtWidgets.QGroupBox("Run")
            gf = QtWidgets.QFormLayout(g)
            self.f_name = Field("str", placeholder="e.g. MoS2_test (folder and figure titles)")
            self.f_out = Field("dir", placeholder="parent folder of the runs")
            self.f_out.set(settings.value("runs_dir", str(Path.home() / "bandfit_runs")))
            gf.addRow("Run name *", self.f_name.widget)
            gf.addRow("Output folder", self.f_out.widget)
            self.il.addWidget(g)
            self.build_inputs()
            self.ini = IniForm(self.ini_keys(), win.specs.get("_objective", []), self.choices)
            self.ini.setMinimumHeight(330)
            gi = QtWidgets.QGroupBox("Settings file (INI)")
            gl = QtWidgets.QVBoxLayout(gi)
            gl.addWidget(self.ini)
            br = QtWidgets.QHBoxLayout()
            for text, fn in (("Load INI…", self.load_ini), ("Save INI…", self.save_ini), ("Show INI", self.show_ini)):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(fn)
                br.addWidget(b)
            br.addStretch(1)
            gl.addLayout(br)
            self.il.addWidget(gi)
            self.args = ArgForm(self.spec, primary=(), hide=self.hidden_args(),
                                advanced_title="All command-line options of bands_fit.py (override the INI)")
            self.il.addWidget(self.args)
            self.il.addStretch(1)
            lay.addWidget(self.scroll(inner), 1)
            row = QtWidgets.QHBoxLayout()
            self.cmdview = QtWidgets.QLineEdit()
            self.cmdview.setReadOnly(True)
            self.cmdview.setPlaceholderText("command line")
            row.addWidget(self.cmdview, 1)
            for text, fn, tip in self.buttons():
                b = QtWidgets.QPushButton(text)
                b.setToolTip(tip)
                b.clicked.connect(fn)
                row.addWidget(b)
            lay.addLayout(row)
            self.defaults()

        # hooks
        def defaults(self):
            pass

        def hidden_args(self):
            return {"cif", "input_file", "outdir", "name"}

        def buttons(self):
            return [("Show command", lambda: self.command(show_only=True), "build the command without running it"),
                    ("Preview", self.preview, "read the reference and make the QA figure, no fit (--prepare-only)"),
                    ("Run fit", self.fit, "start the fit (several runs can go at the same time)")]

        # INI helpers
        def load_ini(self):
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load INI", settings.value("last_dir", str(ROOT)),
                                                         "INI (*.ini *.txt);;All files (*)")
            if p:
                unknown = self.ini.load(p)
                settings.setValue("last_dir", str(Path(p).parent))
                self.after_load()
                if unknown:
                    QtWidgets.QMessageBox.information(self, "BandFit", "Keys not shown in this tab (ignored):\n" +
                                                      "\n".join(unknown))

        def after_load(self):
            pass

        def save_ini(self):
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save INI", settings.value("last_dir", str(ROOT)),
                                                         "INI (*.ini)")
            if p:
                self.ini.save(p)

        def show_ini(self):
            d = QtWidgets.QDialog(self)
            d.setWindowTitle("INI file")
            d.resize(560, 520)
            l = QtWidgets.QVBoxLayout(d)
            t = QtWidgets.QPlainTextEdit(self.ini.text())
            t.setReadOnly(True)
            t.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            l.addWidget(t)
            d.exec()

        # command
        def run_dir(self):
            name = self.f_name.get()
            if not name or any(c in name for c in "/\\") or ".." in name:
                raise ValueError("Give a run name (no '/' or '..')")
            parent = Path(self.f_out.get() or Path.home() / "bandfit_runs").expanduser()
            settings.setValue("runs_dir", str(parent))
            return name, parent / name

        def command(self, show_only=False, extra=(), outdir=None, ini_override=None, suffix=""):
            try:
                name, out = self.run_dir()
                if outdir:
                    out = outdir
                self.validate()
                tokens = self.args.tokens(positional_values={"cif": self.structure_value(), "input_file": "INI"})
            except ValueError as exc:
                self.error(exc)
                return None
            ini_path = out / "settings.ini"
            cmd = [sys.executable, "-u", self.script("bands_fit")]
            cmd += [str(ini_path) if t == "INI" else t for t in tokens]
            cmd += ["--outdir", str(out), "--name", name + suffix] + list(extra)
            self.cmdview.setText(" ".join(shlex.quote(c) for c in cmd[2:]))
            if show_only:
                return None
            out.mkdir(parents=True, exist_ok=True)
            cfg = self.ini.config()
            for s, k, v in (ini_override or []):
                if not cfg.has_section(s):
                    cfg.add_section(s)
                if v is None:
                    cfg.remove_option(s, k)
                else:
                    cfg.set(s, k, str(v))
            with open(ini_path, "w") as fh:
                fh.write("# written by band_fit_gui.py\n")
                cfg.write(fh)
            return cmd, name, out

        def fitting_dir(self, out):
            v = self.args.get("fitting_dir")
            return Path(v) if v else out / "fitting"

        def preview(self):
            r = self.command(extra=["--prepare-only"])
            if r:
                cmd, name, out = r
                job = Job(name + " (preview)", "bands_fit", cmd, outdir=out, kind="preview")
                job.on_done.append(self.preview_done)
                self.win.runs.add(job)

        def preview_done(self, job):
            pass

        def fit(self):
            r = self.command()
            if r:
                cmd, name, out = r
                self.win.runs.add(Job(name, "bands_fit", cmd, outdir=out, fitting_dir=self.fitting_dir(out)))

    class ImageTab(FitTab):
        intro = ("<b>Fit to a picture of the bands.</b> Give the crystal structure and a picture (or CSV) of the "
                 "reference bands, the energy range of the plot and the k path; <i>Preview</i> shows how the "
                 "bands were read before fitting.")

        def build_inputs(self):
            g = QtWidgets.QGroupBox("Structure and band picture")
            f = QtWidgets.QFormLayout(g)
            self.f_struct = Field("file", help="CIF, POSCAR/CONTCAR, QE, SIESTA (.fdf/.XV/.STRUCT_OUT) or FHI-aims file")
            self.f_image = Field("file", help="PNG/JPEG/TIFF/WebP picture of the bands, or numeric CSV")
            f.addRow("Crystal structure *", self.f_struct.widget)
            f.addRow("Band picture / CSV *", self.f_image.widget)
            self.thumb = ImageView(fit_height=True)
            self.thumb.setMinimumHeight(170)
            self.thumb.setMaximumHeight(240)
            f.addRow("", self.thumb)
            self.f_image.changed.connect(lambda: self.thumb.show_file(self.f_image.get() if Path(self.f_image.get()).is_file() else None))
            self.il.addWidget(g)

        def ini_keys(self):
            return {"reference": IMAGE_REF_KEYS, "band_path": ["dimension", "points", "labels", "points_per_segment"],
                    "fit": [k for k, _, _ in INI_SCHEMA["fit"] if k != "objective"]}

        def hidden_args(self):
            return {"cif", "input_file", "outdir", "name", "png", "dft_dir", "dft_code", "dft_file",
                    "valence_bands", "spin_channel", "semicore", "dimension"}

        def defaults(self):
            for s, k, v in (("reference", "mode", "auto"), ("reference", "emin", "-6"), ("reference", "emax", "6"),
                            ("band_path", "dimension", "2"), ("band_path", "points", "0,0; 0.5,0; 0.3333333333,0.3333333333; 0,0"),
                            ("band_path", "labels", "G,M,K,G"), ("band_path", "points_per_segment", "20"),
                            ("fit", "n_pairs", "1"), ("fit", "budget", "400")):
                self.ini.set(s, k, v)

        def after_load(self):
            img = self.ini.get("reference", "image")
            if img:
                self.f_image.set(img)

        def structure_value(self):
            return self.f_struct.get()

        def validate(self):
            if not Path(self.f_struct.get()).is_file():
                raise ValueError("Choose the crystal structure file")
            if not Path(self.f_image.get()).is_file():
                raise ValueError("Choose the band picture (or CSV)")
            self.ini.set("reference", "image", self.f_image.get())

    class DftTab(FitTab):
        intro = ("<b>Fit to a DFT band calculation.</b> Choose the folder of the band run; the package is detected. "
                 "Gapped references use the <i>window</i> objective; metals use <i>metal</i> (bands aligned at "
                 "E<sub>F</sub>; give the Fermi level of the SCF run). Spin-polarized runs: fit each channel, or "
                 "<i>Fit both spins</i> to fit up and down at the same time and join them (huckel-SP.dat).")

        def build_inputs(self):
            g = QtWidgets.QGroupBox("DFT calculation")
            f = QtWidgets.QFormLayout(g)
            self.f_dir = Field("dir", help="folder of the band-structure run")
            self.f_struct = Field("file", help="optional: structure file with the same cell (default: the DFT one)")
            self.info = QtWidgets.QLabel("")
            self.info.setWordWrap(True)
            f.addRow("DFT folder *", self.f_dir.widget)
            f.addRow("Structure (optional)", self.f_struct.widget)
            f.addRow("", self.info)
            self.f_dir.changed.connect(self.detect)
            self.il.addWidget(g)

        def ini_keys(self):
            return {"reference": DFT_REF_KEYS, "band_path": ["dimension", "points_per_segment"],
                    "fit": [k for k, _, _ in INI_SCHEMA["fit"]], "objective": None}

        def hidden_args(self):
            return {"cif", "input_file", "outdir", "name", "dft_dir", "png", "mode", "bbox", "colors",
                    "allow_ambiguous"}

        def buttons(self):
            b = super().buttons()
            b.append(("Fit both spins", self.fit_spins,
                      "fit spin up and spin down in parallel, then join them with spin_combine.py"))
            return b

        def defaults(self):
            for s, k, v in (("reference", "emin", "-6"), ("reference", "emax", "6"),
                            ("band_path", "points_per_segment", "20"), ("fit", "objective", "window"),
                            ("fit", "n_pairs", "1"), ("fit", "budget", "2000"), ("fit", "optimizer", "cmaes"),
                            ("fit", "polish", "nelder-mead")):
                self.ini.set(s, k, v)

        def after_load(self):
            d = self.ini.get("reference", "dft_dir")
            if d:
                self.f_dir.set(d)

        def detect(self):
            d = self.f_dir.get()
            if not Path(d).is_dir():
                self.info.setText("")
                return
            code = detect_dft_code(d)
            msg = [f"package: <b>{code}</b>" if code else "<span style='color:#b00020'>package not recognized: "
                   "set dft_code in the Reference tab</span>"]
            if code:
                self.ini.set("reference", "dft_code", code)
            fld = self.ini.field("reference", "dft_file")
            if fld:
                pats = ("vasprun", "eigenval", ".bands", "band1", "_band.dat", ".pwo", ".out", ".xml", ".xv", "struct_out")
                files = sorted(p.name for p in Path(d).iterdir() if p.is_file() and any(s in p.name.lower() for s in pats))
                fld.set_suggestions(files)
            if code == "vasp" and vasp_is_spin_polarized(d):
                msg.append("spin-polarized run (ISPIN = 2): set spin_channel or use <i>Fit both spins</i>")
            if not self.f_name.get():
                self.f_name.set(Path(d).name)
            self.info.setText("; ".join(msg))

        def structure_value(self):
            return self.f_struct.get()

        def validate(self):
            if not Path(self.f_dir.get()).is_dir():
                raise ValueError("Choose the DFT folder")
            self.ini.set("reference", "dft_dir", self.f_dir.get())
            if not self.ini.get("reference", "dft_code"):
                raise ValueError("Set dft_code (Reference tab): the package was not detected")

        def preview_done(self, job):
            txt = job.text()
            if job.status != "done" and ("metallic" in txt or "No positive gap" in txt or "cross the Fermi" in txt):
                b = QtWidgets.QMessageBox.question(
                    self, "Metallic reference",
                    "The DFT bands cross the Fermi level (metal) or have no gap.\n\n"
                    "Use the 'metal' objective (bands aligned at E_F)? Give the Fermi level of the SCF run in "
                    "Reference → fermi if the band files may be off.")
                if b == QtWidgets.QMessageBox.Yes:
                    self.ini.set("fit", "objective", "metal")
                    for k, v in (("window_min", "-9"), ("window_max", "3"), ("far_weight", "0.15")):
                        if not self.ini.get("objective", k):
                            self.ini.set("objective", k, v)

        def fit_spins(self):
            try:
                name, out = self.run_dir()
            except ValueError as exc:
                self.error(exc)
                return
            fdir = Path(self.args.get("fitting_dir") or out / "fitting")
            jobs = {}
            for ch in ("up", "down"):
                r = self.command(outdir=out / f"spin_{ch}", suffix=f"_{ch}",
                                 extra=["--fitting-dir", str(fdir / f"spin_{ch}")] if not self.args.get("fitting_dir") else [],
                                 ini_override=[("reference", "spin_channel", ch)])
                if not r:
                    return
                cmd, _, o = r
                if self.args.get("fitting_dir"):        # user path: one folder per channel
                    i = cmd.index("--fitting-dir")
                    cmd[i + 1] = str(fdir / f"spin_{ch}")
                if "--spin-channel" in cmd:
                    i = cmd.index("--spin-channel")
                    cmd[i + 1] = ch
                jobs[ch] = Job(f"{name} spin {ch}", "bands_fit", cmd, outdir=o, fitting_dir=fdir / f"spin_{ch}",
                               kind="spin-channel")
            state = {}

            def done(job):
                state[job.name] = job.status
                if len(state) == 2:
                    if all(s == "done" for s in state.values()):
                        cmd = [sys.executable, "-u", self.script("spin_combine"), "--up", str(fdir / "spin_up"),
                               "--down", str(fdir / "spin_down"), "--outdir", str(fdir / "spin_polarized")]
                        self.win.runs.add(Job(f"{name} spin combine", "spin_combine", cmd,
                                              outdir=fdir / "spin_polarized", fitting_dir=fdir / "spin_polarized"))
                    else:
                        self.error(f"{name}: a spin channel failed; spin_combine was not run")
            for ch in ("up", "down"):
                jobs[ch].on_done.append(done)
                self.win.runs.add(jobs[ch])

    # ------------------------------------------------------------- family_fit tab
    class FamilyTab(ToolTab):
        """family_fit.py: manifest editor (members, roles, per-system settings) + every option."""
        MODES = [("Fit", ["--fit"]), ("Validate only (no data read)", ["--validate-only"]),
                 ("Prepare only (references, no fit)", ["--prepare-only"]),
                 ("Evaluate an earlier fit (no optimization)", ["--evaluate"])]
        COMBOS = {"role": ["train", "test"], "dft_code": [""] + CODES, "spin_channel": ["", "up", "down", "average"],
                  "metallic": ["", "auto", "true", "false"]}
        TIPS = {"name": "label of the member (folder name by default)", "role": "train: fitted; test: scored "
                "after the fit with the frozen table", "dft_dir": "DFT band folder (DFT member)",
                "dft_code": "package (empty: defaults / detected when added)",
                "structure": "structure file: required for picture members, optional for DFT ones",
                "settings": "INI of a picture member (axis, k path), as in Fit to image",
                "reference": "band picture or CSV of a picture member (default: [reference] image of its INI)",
                "weight": "weight in the shared objective (default 1)", "spin_channel": "up, down or average",
                "metallic": "true/false; empty = detected", "semicore": "auto, integer or 0",
                "overlap_radius": "1..8 or auto for this member (default: the global one)",
                "objective_settings": "objective weights of this member only, e.g.  segment_weights: '0:3'"}

        def __init__(self, win):
            super().__init__(win)
            self.spec = win.specs.get("family_fit", {})
            lay = QtWidgets.QVBoxLayout(self)
            inner = QtWidgets.QWidget()
            il = QtWidgets.QVBoxLayout(inner)
            lab = QtWidgets.QLabel(
                "<b>Family fit.</b> One shared EHT table (and one K) for several materials: DFT band folders "
                "(any package; gapped, metallic or spin-polarized) and band pictures with their INI, mixed freely. "
                "<i>test</i> members are opened only after the fit and scored with the frozen table. Each member "
                "gets its figures and fitting/ export in OUTDIR/systems/&lt;name&gt;/.")
            lab.setWordWrap(True)
            il.addWidget(lab)
            # manifest file
            g = QtWidgets.QGroupBox("Manifest (YAML)")
            f = QtWidgets.QHBoxLayout(g)
            self.f_manifest = Field("save", help="family YAML: written here before each run "
                                                  "(empty: OUTDIR/family_manifest.yaml)")
            b_load, b_save, b_view = (QtWidgets.QPushButton(x) for x in ("Load…", "Save", "Show YAML"))
            b_load.clicked.connect(self.load)
            b_save.clicked.connect(self.save)
            b_view.clicked.connect(self.show_yaml)
            f.addWidget(self.f_manifest.widget, 1)
            for b in (b_load, b_save, b_view):
                f.addWidget(b)
            il.addWidget(g)
            # global settings + defaults
            g = QtWidgets.QGroupBox("Shared settings")
            grid = QtWidgets.QHBoxLayout(g)
            fl, fr = QtWidgets.QFormLayout(), QtWidgets.QFormLayout()
            self.glob, self.defs = {}, {}
            for key, kind, h in FAMILY_GLOBAL:
                fld = Field(kind, help=h, suggest=SUGGEST.get(key) if key != "objective_settings"
                            else ["w_split: 100", "n_cond: 4", "window_min: -4, window_max: 3"])
                fl.addRow(key, fld.widget)
                self.glob[key] = fld
            fr.addRow(QtWidgets.QLabel("<i>defaults: of every DFT member</i>"))
            for key, kind, h in FAMILY_DEFAULTS:
                fld = Field(kind, help=h, suggest=SUGGEST.get(key))
                fr.addRow(key, fld.widget)
                self.defs[key] = fld
            grid.addLayout(fl, 3)
            grid.addLayout(fr, 2)
            il.addWidget(g)
            # members
            g = QtWidgets.QGroupBox("Members")
            gl = QtWidgets.QVBoxLayout(g)
            self.table = QtWidgets.QTableWidget(0, len(FAMILY_COLS))
            self.table.setHorizontalHeaderLabels(FAMILY_COLS)
            for c, k in enumerate(FAMILY_COLS):
                self.table.horizontalHeaderItem(c).setToolTip(self.TIPS.get(k, ""))
            self.table.setMinimumHeight(190)
            self.table.horizontalHeader().setStretchLastSection(True)
            gl.addWidget(self.table)
            row = QtWidgets.QHBoxLayout()
            for text, fn, tip in (("Add DFT folders…", self.add_dft, "one member per folder; package detected"),
                                  ("Add picture member…", self.add_image, "structure + INI (+ picture)"),
                                  ("Duplicate", self.duplicate, "copy the selected member"),
                                  ("Remove", self.remove, "remove the selected members")):
                b = QtWidgets.QPushButton(text)
                b.setToolTip(tip)
                b.clicked.connect(fn)
                row.addWidget(b)
            row.addStretch(1)
            gl.addLayout(row)
            il.addWidget(g)
            # mode + options
            g = QtWidgets.QGroupBox("Run")
            f = QtWidgets.QFormLayout(g)
            self.mode = QtWidgets.QComboBox()
            self.mode.addItems([m for m, _ in self.MODES])
            self.f_eval = Field("file", help="fit.json of the earlier fit (param_best_native.dat next to it)")
            f.addRow("mode", self.mode)
            f.addRow("fit.json to evaluate", self.f_eval.widget)
            self.f_eval.widget.setEnabled(False)
            self.mode.currentIndexChanged.connect(lambda i: self.f_eval.widget.setEnabled(i == 3))
            il.addWidget(g)
            self.form = ArgForm(self.spec, primary=["outdir", "optimizer", "budget", "polish", "seed", "parallel",
                                                    "fast_budget", "workers"],
                                hide=["manifest", "validate_only", "prepare_only", "fit", "evaluate"],
                                advanced_title="Other options")
            il.addWidget(self.form)
            il.addStretch(1)
            lay.addWidget(self.scroll(inner), 1)
            row = QtWidgets.QHBoxLayout()
            self.cmdview = QtWidgets.QLineEdit()
            self.cmdview.setReadOnly(True)
            self.cmdview.setPlaceholderText("command line")
            b_show = QtWidgets.QPushButton("Show command")
            b_run = QtWidgets.QPushButton("Run")
            b_run.setDefault(True)
            row.addWidget(self.cmdview, 1)
            row.addWidget(b_show)
            row.addWidget(b_run)
            lay.addLayout(row)
            b_show.clicked.connect(lambda: self.command(show=True))
            b_run.clicked.connect(self.run)
            self.glob["n_pairs"].set("1")

        # --- table
        def _set_row(self, r, values):
            for c, k in enumerate(FAMILY_COLS):
                v = str(values.get(k, "") or "")
                if k in self.COMBOS:
                    w = QtWidgets.QComboBox()
                    w.addItems(self.COMBOS[k])
                    if v and w.findText(v) < 0:
                        w.addItem(v)
                    w.setCurrentText(v or self.COMBOS[k][0])
                    w.setToolTip(self.TIPS.get(k, ""))
                    self.table.setCellWidget(r, c, w)
                else:
                    it = QtWidgets.QTableWidgetItem(v)
                    it.setToolTip(v or self.TIPS.get(k, ""))
                    self.table.setItem(r, c, it)

        def add_row(self, values):
            r = self.table.rowCount()
            self.table.insertRow(r)
            self._set_row(r, values)
            self.table.resizeColumnsToContents()
            return r

        def row_values(self, r):
            out = {}
            for c, k in enumerate(FAMILY_COLS):
                w = self.table.cellWidget(r, c)
                if w is not None:
                    out[k] = w.currentText()
                else:
                    it = self.table.item(r, c)
                    out[k] = it.text() if it else ""
            if out.get("metallic") == "auto":
                out["metallic"] = ""
            return out

        def rows(self):
            return [self.row_values(r) for r in range(self.table.rowCount())]

        def add_dft(self):
            dlg = QtWidgets.QFileDialog(self, "DFT band folders (select one or more)",
                                        settings.value("last_dir", str(Path.home())))
            dlg.setFileMode(QtWidgets.QFileDialog.Directory)
            dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
            for view in dlg.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
                view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            if not dlg.exec():
                return
            names = {v["name"] for v in self.rows()}
            for d in dlg.selectedFiles():
                if not Path(d).is_dir():
                    continue
                name = Path(d).name
                while name in names:
                    name += "_2"
                names.add(name)
                vals = {"name": name, "role": "train", "dft_dir": d, "dft_code": detect_dft_code(d) or ""}
                if vals["dft_code"] == "vasp" and vasp_is_spin_polarized(d) and not self.defs["spin_channel"].get():
                    vals["spin_channel"] = "average"
                self.add_row(vals)
                settings.setValue("last_dir", str(Path(d).parent))

        def add_image(self):
            start = settings.value("last_dir", str(Path.home()))
            s, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Structure file of the member", start)
            if not s:
                return
            ini, _ = QtWidgets.QFileDialog.getOpenFileName(self, "INI of the member (axis, k path)", str(Path(s).parent),
                                                           "INI (*.ini *.txt);;All files (*)")
            if not ini:
                return
            img, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Band picture or CSV (Cancel: [reference] image of the INI)",
                                                           str(Path(s).parent))
            self.add_row({"name": Path(s).stem, "role": "train", "structure": s, "settings": ini, "reference": img or ""})

        def duplicate(self):
            for r in sorted({i.row() for i in self.table.selectedIndexes()}):
                v = self.row_values(r)
                v["name"] += "_copy"
                self.add_row(v)

        def remove(self):
            for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
                self.table.removeRow(r)

        # --- manifest
        def manifest(self):
            return family_manifest({k: f.get() for k, f in self.glob.items()},
                                   {k: f.get() for k, f in self.defs.items()}, self.rows())

        def yaml_text(self):
            import yaml
            return "# written by band_fit_gui.py (Family fit)\n" + yaml.safe_dump(self.manifest(), sort_keys=False)

        def manifest_path(self):
            p = self.f_manifest.get()
            if p:
                return Path(p)
            out = self.form.get("outdir") or "outputs/family_fit"
            out = Path(out) if Path(out).is_absolute() else ROOT / out
            return out / "family_manifest.yaml"

        def save(self):
            try:
                text = self.yaml_text()
            except ValueError as exc:
                self.error(exc)
                return None
            p = self.manifest_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            self.f_manifest.set(str(p))
            return p

        def show_yaml(self):
            try:
                text = self.yaml_text()
            except ValueError as exc:
                self.error(exc)
                return
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Family manifest")
            dlg.resize(640, 520)
            v = QtWidgets.QVBoxLayout(dlg)
            e = QtWidgets.QPlainTextEdit(text)
            e.setReadOnly(True)
            e.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            v.addWidget(e)
            dlg.show()
            self._dlg = dlg

        def load(self, path=None):
            import yaml
            if not path:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Family manifest", settings.value("last_dir", str(Path.home())),
                                                                "YAML (*.yaml *.yml);;All files (*)")
            if not path:
                return
            try:
                m = yaml.safe_load(Path(path).read_text())
                glob, defs, rows = family_rows(m, Path(path).resolve().parent)
            except Exception as exc:
                self.error(f"{path}: {exc}")
                return
            for k, f in self.glob.items():
                f.set(glob.get(k, ""))
            for k, f in self.defs.items():
                f.set(defs.get(k, ""))
            self.table.setRowCount(0)
            for r in rows:
                self.add_row(r)
            self.f_manifest.set(str(Path(path).resolve()))

        # --- run
        def command(self, show=False, manifest=None):
            mode = self.MODES[self.mode.currentIndex()][1]
            if mode == ["--evaluate"]:
                if not Path(self.f_eval.get()).is_file():
                    self.error("Choose the fit.json to evaluate")
                    return None
                mode = ["--evaluate", self.f_eval.get()]
            try:
                m = str(manifest or self.manifest_path())
                cmd = [sys.executable, "-u", self.script("family_fit")] + self.form.tokens({"manifest": m})
            except ValueError as exc:
                self.error(exc)
                return None
            cmd[3:3] = mode
            self.cmdview.setText(" ".join(shlex.quote(c) for c in cmd[2:]))
            return cmd

        def run(self):
            p = self.save()
            if p is None:
                return
            cmd = self.command(manifest=p)
            if not cmd:
                return
            out = self.form.get("outdir") or "outputs/family_fit"
            out = Path(out) if Path(out).is_absolute() else ROOT / out
            if "--outdir" not in cmd:
                cmd += ["--outdir", str(out)]
            kind = "run" if self.mode.currentIndex() in (0, 3) else "preview"
            self.win.runs.add(Job(out.name, "family_fit", cmd, outdir=out, kind=kind))

    # ------------------------------------------------------------- tools tab
    class ToolsTab(QtWidgets.QTabWidget):
        def __init__(self, win):
            super().__init__()
            for key, intro in (
                    ("prepare_band_png", "<b>Read a band picture</b> into the internal layered format, with a QA preview."),
                    ("inspect_png", "<b>Inspect a picture</b>: colors and suggested plot box."),
                    ("digitize_bands", "<b>Digitize bands</b> by hand-made annotations or a template."),
                    ("strategy_advisor", "<b>Strategy advisor</b>: rule-based recommendation (no calculation).")):
                self.addTab(GenericTab(win, key, intro), key)

    # ------------------------------------------------------------- side navigation
    class SideTabs(QtWidgets.QWidget):
        """A list on the left and the pages on the right (same calls as QTabWidget)."""

        def __init__(self):
            super().__init__()
            lay = QtWidgets.QHBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            side = QtWidgets.QWidget()
            side.setFixedWidth(170)
            sl = QtWidgets.QVBoxLayout(side)
            sl.setContentsMargins(0, 0, 0, 0)
            sl.setSpacing(0)
            if LOGO.is_file():
                logo = QtWidgets.QLabel()
                logo.setPixmap(QtGui.QPixmap(str(LOGO)).scaledToWidth(166, Qt.SmoothTransformation))
                logo.setAlignment(Qt.AlignCenter)
                logo.setStyleSheet("background:#000;padding:6px 2px")
                logo.setToolTip("YAeHMOP BandFit")
                sl.addWidget(logo)
            self.list = QtWidgets.QListWidget()
            self.list.setStyleSheet("QListWidget{font-size:13px} QListWidget::item{padding:7px 6px}")
            self.stack = QtWidgets.QStackedWidget()
            sl.addWidget(self.list, 1)
            lay.addWidget(side)
            lay.addWidget(self.stack, 1)
            self.list.currentRowChanged.connect(self.stack.setCurrentIndex)

        def addTab(self, w, title):
            self.stack.addWidget(w)
            self.list.addItem(title)
            if self.list.count() == 1:
                self.list.setCurrentRow(0)

        def count(self):
            return self.stack.count()

        def widget(self, i):
            return self.stack.widget(i)

        def tabText(self, i):
            return self.list.item(i).text()

        def setCurrentIndex(self, i):
            self.list.setCurrentRow(i)

    # ------------------------------------------------------------- main window
    class Main(QtWidgets.QMainWindow):
        def __init__(self, specs):
            super().__init__()
            self.specs = specs
            self.setWindowTitle("BandFit — EHT (YAeHMOP) band fitting")
            if ICON.is_file():
                self.setWindowIcon(QtGui.QIcon(str(ICON)))
            split = QtWidgets.QSplitter(Qt.Horizontal)
            self.tabs = SideTabs()
            self.runs = RunsPanel()
            self.tabs.addTab(ImageTab(self), "Fit to image")
            self.tabs.addTab(DftTab(self), "Fit to DFT")
            self.tabs.addTab(GenericTab(self, "soc_fit",
                                        "<b>Spin-orbit coupling.</b> On-site λ L·S on top of a spinless fit "
                                        "(<i>--from-fitting</i>), fitted to a DFT run with SOC; writes huckel-SOC.dat.",
                                        primary=["dft_dir", "dft_code", "from_fitting", "outdir", "fitting_dir",
                                                 "soc_shells", "w_split"]), "Spin-orbit")
            self.tabs.addTab(GenericTab(self, "spin_combine",
                                        "<b>Spin-polarized Hamiltonian.</b> Join the fitting/ folders of a spin-up and "
                                        "a spin-down fit into huckel-SP.dat (WanTiBEXOS)."), "Spin combine")
            self.tabs.addTab(GenericTab(self, "frozen_transfer",
                                        "<b>Frozen transfer.</b> Apply fitted parameters to another structure "
                                        "without refitting (evaluation only)."), "Frozen transfer")
            self.tabs.addTab(FamilyTab(self), "Family fit")
            self.tabs.addTab(GenericTab(self, "compare_strategies",
                                        "<b>Compare strategies</b> on the validation examples (several protocols)."),
                             "Compare strategies")
            self.tabs.addTab(ToolsTab(self), "Image tools")
            split.addWidget(self.tabs)
            split.addWidget(self.runs)
            split.setSizes([780, 660])
            self.setCentralWidget(split)
            self.resize(1440, 900)
            self._menus()
            self._check_env()

        def _menus(self):
            m = self.menuBar().addMenu("&File")
            a = m.addAction("Open results…")
            a.triggered.connect(self.runs._load_results)
            a = m.addAction("YAeHMOP executable and parameters…")
            a.triggered.connect(self._yaehmop_dialog)
            m.addSeparator()
            a = m.addAction("Quit")
            a.triggered.connect(self.close)
            h = self.menuBar().addMenu("&Help")
            a = h.addAction("Manual (PDF)")
            a.triggered.connect(lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(ROOT / "manual/manual.pdf"))))
            a = h.addAction("About")
            a.triggered.connect(lambda: QtWidgets.QMessageBox.about(
                self, "BandFit", "BandFit GUI — extended-Hückel (YAeHMOP) parameters fitted to band structures.\n"
                "Each run is a call of the command-line tools; see the manual."))

        def _bind(self):
            return settings.value("YAEHMOP_BIND", "") or os.environ.get("YAEHMOP_BIND", "")

        def _check_env(self):
            b = self._bind()
            p = settings.value("YAEHMOP_PARAM_FILE", "") or os.environ.get("YAEHMOP_PARAM_FILE", "")
            ok = b and Path(b).is_file() and p and Path(p).is_file()
            self.statusBar().showMessage(
                f"YAeHMOP: {b}" if ok else "YAeHMOP bind / parameter file not set: File → YAeHMOP executable and "
                "parameters (or activate the conda environment yaehmop-bandfit)")

        def _yaehmop_dialog(self):
            d = QtWidgets.QDialog(self)
            d.setWindowTitle("YAeHMOP")
            f = QtWidgets.QFormLayout(d)
            fb = Field("file")
            fp = Field("file")
            fb.set(self._bind())
            fp.set(settings.value("YAEHMOP_PARAM_FILE", "") or os.environ.get("YAEHMOP_PARAM_FILE", ""))
            f.addRow("bind executable", fb.widget)
            f.addRow("parameter table (eht_parms.dat)", fp.widget)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(d.accept)
            bb.rejected.connect(d.reject)
            f.addRow(bb)
            d.resize(620, 120)
            if d.exec():
                settings.setValue("YAEHMOP_BIND", fb.get())
                settings.setValue("YAEHMOP_PARAM_FILE", fp.get())
                self._check_env()

        def closeEvent(self, e):
            running = [j for j in self.runs.jobs if j.status == "running"]
            if running:
                b = QtWidgets.QMessageBox.question(self, "BandFit", f"{len(running)} run(s) still going. Stop them and quit?")
                if b != QtWidgets.QMessageBox.Yes:
                    e.ignore()
                    return
                for j in running:
                    j.stop()
            e.accept()

    # ------------------------------------------------------------- start
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("BandFit")
    if ICON.is_file():
        app.setWindowIcon(QtGui.QIcon(str(ICON)))
    splash = QtWidgets.QLabel()
    if LOGO.is_file():
        splash.setPixmap(QtGui.QPixmap(str(LOGO)).scaledToWidth(420, Qt.SmoothTransformation))
    else:
        splash.setText("BandFit: reading the command-line options…")
    splash.setWindowFlags(Qt.SplashScreen)
    splash.setStyleSheet("padding:24px;font-size:14px;background:#000;color:#eee")
    splash.show()
    app.processEvents()
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--dump-specs"], capture_output=True,
                       text=True, cwd=str(ROOT), env=dict(os.environ, MPLBACKEND="Agg"))
    splash.close()
    try:
        specs = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        QtWidgets.QMessageBox.critical(None, "BandFit", "Could not read the command-line options:\n" + r.stderr[-2000:])
        return 1
    w = Main(specs)
    w.show()
    shots = os.environ.get("BANDFIT_GUI_SELFTEST")
    if shots:                       # developer check: screenshots of every tab + one short real fit
        _selftest(app, w, Path(shots), QtCore)
    return app.exec()


def _selftest(app, w, out, QtCore):
    out.mkdir(parents=True, exist_ok=True)
    steps = []

    def shot(name):
        app.processEvents()
        w.grab().save(str(out / f"{name}.png"))

    def tabs():
        for i in range(w.tabs.count()):
            w.tabs.setCurrentIndex(i)
            shot(f"tab{i}_{w.tabs.tabText(i).replace(' ', '_')}")
        w.tabs.setCurrentIndex(1)
        t = w.tabs.widget(1)
        # the settings of the best validated Al2O3 bilayer fit (window objective, Al d shell, 8000 calls)
        t.ini.load(str(ROOT / "validation/refine_v1c3/cmaes-basis-8k.ini"))
        t.f_dir.set(str(ROOT / "validation/siesta/Al2O2_bilayer_test"))
        t.f_name.set("selftest_Al2O2")
        t.f_out.set(str(out / "runs"))
        t.ini.set("fit", "overlap_radius", "auto")
        shot("dft_filled")
        t.fit()
        steps.append(time.time())
        QtCore.QTimer.singleShot(1500, wait)

    def wait():
        j = w.runs.jobs[-1]
        if j.status == "running" and time.time() - steps[0] < 1200:
            pr = j.progress()
            if not (out / "progress.png").exists() and pr and len(pr[0]) > 400:
                w.runs.tabs.setCurrentIndex(1)
                w.runs.refresh_selected()
                shot("progress")
            QtCore.QTimer.singleShot(1000, wait)
            return
        w.runs.refresh_selected()
        w.runs.tabs.setCurrentIndex(2)
        shot("results")
        w.runs.tabs.setCurrentIndex(0)
        shot("log")
        print("SELFTEST dft", j.status, flush=True)
        image()

    def image():
        w.tabs.setCurrentIndex(0)
        t = w.tabs.widget(0)
        # MoS2 monolayer: picture of the PBE top valence / bottom conduction band (examples/gui_mos2)
        t.ini.load(str(ROOT / "examples/gui_mos2/MoS2_2bands.ini"))
        t.after_load()
        t.f_struct.set(str(ROOT / "examples/gui_mos2/MoS2_monolayer.vasp"))
        t.f_image.set(str(ROOT / "examples/gui_mos2/MoS2_PBE_2bands.png"))
        t.f_name.set("selftest_MoS2")
        t.f_out.set(str(out / "runs"))
        shot("image_filled")
        t.preview()
        QtCore.QTimer.singleShot(1000, lambda: wait_image("preview"))

    def wait_image(stage):
        j = w.runs.jobs[-1]
        if j.status == "running":
            QtCore.QTimer.singleShot(1000, lambda: wait_image(stage))
            return
        w.runs.refresh_selected()
        w.runs.tabs.setCurrentIndex(2 if j.status == "done" else 0)
        shot(f"image_{stage}")
        for k, d in enumerate(getattr(w.runs, "_dialogs", [])):
            app.processEvents()
            d.grab().save(str(out / f"dialog_{stage}_{k}.png"))
        print("SELFTEST image", stage, j.status, flush=True)
        if stage == "preview" and j.status == "done":
            w.tabs.widget(0).fit()
            QtCore.QTimer.singleShot(1000, lambda: wait_image("fit"))
        else:
            family()

    def family():
        # Family fit tab: one table for two silicon polymorphs (QE, LDA), automatic radius, Si d shell
        i = [w.tabs.tabText(k) for k in range(w.tabs.count())].index("Family fit")
        w.tabs.setCurrentIndex(i)
        t = w.tabs.widget(i)
        t.table.setRowCount(0)
        t.add_row({"name": "Si_diamond", "role": "train", "dft_dir": str(ROOT / "validation/qe/Si_diamond_lda"),
                   "dft_code": "qe"})
        t.add_row({"name": "Si_wurtzite", "role": "train", "dft_dir": str(ROOT / "validation/qe/Si_wurtzite_lda"),
                   "dft_code": "qe"})
        t.glob["overlap_radius"].set("auto")
        t.glob["basis_add"].set("auto")
        t.form.set("outdir", str(out / "runs/selftest_family"))
        t.form.set("optimizer", "cmaes")
        t.form.set("polish", "nelder-mead")
        t.form.set("budget", "3000")
        t.f_manifest.set(str(out / "runs/selftest_family.yaml"))
        shot("family_filled")
        t.run()
        steps.append(time.time())
        QtCore.QTimer.singleShot(1500, wait_family)

    def wait_family():
        j = w.runs.jobs[-1]
        if j.status == "running" and time.time() - steps[-1] < 1800:
            QtCore.QTimer.singleShot(1000, wait_family)
            return
        w.runs.refresh_selected()
        w.runs.tabs.setCurrentIndex(2 if j.status == "done" else 0)
        shot("family_results")
        for k, d in enumerate(getattr(w.runs, "_dialogs", [])):
            if d.isVisible() and "family" in d.windowTitle():
                app.processEvents()
                d.grab().save(str(out / f"dialog_family_{k}.png"))
        print("SELFTEST family", j.status, flush=True)
        app.quit()
    QtCore.QTimer.singleShot(500, tabs)


def main():
    if "--dump-specs" in sys.argv:
        dump_specs()
        return 0
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print("band_fit_gui.py needs PySide6:  pip install PySide6-Essentials  (or conda install -c conda-forge pyside6)")
        return 1
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
