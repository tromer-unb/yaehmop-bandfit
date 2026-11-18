# Crystal-structure input formats

Wherever the tools ask for a **CIF**, they also accept the structure files of
**VASP**, **Quantum ESPRESSO**, **SIESTA** and **FHI-aims**. The fitters only use
the periodic cell, the atomic positions and the chemical symbols, so the same
crystal gives the **same YAeHMOP input and the same bands** in any format.

```bash
python USER_WORKFLOW/bands_fit.py inputs/MoS2/POSCAR      inputs/MoS2/file.ini --outdir outputs/MoS2
python USER_WORKFLOW/bands_fit.py inputs/MoS2/scf.in      inputs/MoS2/file.ini --outdir outputs/MoS2
python USER_WORKFLOW/bands_fit.py inputs/MoS2/MoS2.fdf    inputs/MoS2/file.ini --outdir outputs/MoS2
python USER_WORKFLOW/bands_fit.py inputs/MoS2/geometry.in inputs/MoS2/file.ini --outdir outputs/MoS2
```

The detected format is printed (`STRUCTURE ... format vasp`) and stored as
`structure_format` in `run_summary.json`, `fit.json`, `transfer_report.json`
and the advisor JSON.

## Supported files

| Package | File | `structure_format` | Detected from |
|---|---|---|---|
| — | `*.cif` | `cif` | name, or `_cell_length_a` in the file |
| VASP | `POSCAR`, `CONTCAR`, `*.vasp`, `*.poscar` (VASP 5 format with element line) | `vasp` | name (contains POSCAR/CONTCAR), or the POSCAR layout |
| VASP | `OUTCAR` (last ionic step) | `vasp-out` | name |
| VASP | `vasprun*.xml` (last ionic step) | `vasp-xml` | name, or `<modeling>` |
| Quantum ESPRESSO | pw.x input `*.pwi`, `*.in`, ... — **any `ibrav`** (celldm or A/B/C/cosAB...), positions in `alat`, `bohr`, `angstrom` or `crystal` | `qe` | `.pwi`, or `&SYSTEM` + `ATOMIC_POSITIONS` in the file |
| Quantum ESPRESSO | pw.x output `*.pwo`, `*.out` (last step) | `qe-out` | `.pwo`, or `Program PWSCF` in the file |
| SIESTA | input `*.fdf` | `siesta` | `.fdf`, or SIESTA `%block`s in the file |
| SIESTA | `*.XV` | `siesta-xv` | name only |
| SIESTA | `*.STRUCT_OUT` | `siesta-struct-out` | name only |
| FHI-aims | `geometry.in` (`lattice_vector`, `atom` or `atom_frac`) | `aims` | name, or `lattice_vector`/`atom` lines |
| FHI-aims | output (last step) | `aims-out` | `Invoking FHI-aims` in the file |

Parsing is done by ASE, except for two cases ASE does not handle, which have
their own code in `parameter_fitter/structure_io.py`:

* **SIESTA `.fdf`**: `LatticeConstant` (Ang, Bohr or nm), `LatticeVectors` or
  `LatticeParameters` (cubic cell if neither), `ChemicalSpeciesLabel`,
  `AtomicCoordinatesFormat` (`Bohr`/`NotScaledCartesianBohr` — the default —,
  `Ang`/`NotScaledCartesianAng`, `ScaledCartesian`, `Fractional`/`ScaledByLatticeVectors`),
  `AtomicCoordinatesAndAtomicSpecies`, numeric `AtomicCoordinatesOrigin`,
  `%include file` and `%block Name < file`. Labels ignore case, `-`, `_` and `.`,
  as in fdf. Not supported: Z-matrix geometries and ghost/synthetic species (Z ≤ 0).
* **Quantum ESPRESSO with `ibrav ≠ 0`**: the lattice vectors follow pw.x
  `latgen` for ibrav 1–14 and the negative/91 variants (current QE convention
  for 13/−13). `ATOMIC_POSITIONS crystal_sg` is not supported.

## Forcing the format

If the name is not conventional and the content is not recognized (for example an
`.XV` renamed to `relaxed.dat`), force it:

```bash
python USER_WORKFLOW/bands_fit.py relaxed.dat file.ini --structure-format siesta-xv ...
```

`--structure-format` exists in `bands_fit.py`, `frozen_transfer.py`,
`strategy_advisor.py` and `parameter_fitter/fit_from_image.py`. In YAML manifests
(`family_fit.py`, `fit_shared.py`, `fit_sparse_window.py`) add
`structure_format: <name>` to a system, or once under `shared:`. The aliases
`poscar`, `contcar`, `espresso-in`, `pwi`, `fdf`, `xv`, `fhi-aims` are accepted too.

## Things to check

* **The k path refers to the lattice vectors of the file you give.** The same
  crystal written with other primitive vectors (e.g. QE `ibrav=2` vs. a CIF cell)
  needs the `[band_path]` points in *those* reciprocal vectors; usually that is
  automatic, because the reference bands come from the same DFT calculation.
* **2D/1D materials** need the full 3×3 cell including the vacuum vector, as in
  the DFT calculation; the periodic dimension still comes from `[band_path] dimension`.
  Files without a periodic cell (e.g. a molecular `geometry.in`) are rejected.
* Output files (OUTCAR, vasprun.xml, pw.x and FHI-aims outputs) give the **last**
  geometry, i.e. the relaxed one.
* `--from-fit` and frozen transfer check the **file** SHA-256: refitting
  needs the same file, and a target saved in a different format than the
  source training file is **not** detected as the same crystal.
