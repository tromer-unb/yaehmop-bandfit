# Fitting to raw DFT band data (VASP, Quantum ESPRESSO, SIESTA, FHI-aims)

Instead of an image or a CSV, the reference can be the **folder of a DFT
band-structure calculation**. The program reads the eigenvalues themselves,
the k points, the crystal structure and the number of occupied bands, and fits
the extended Hückel parameters to those numbers. No digitization is involved.

```bash
python USER_WORKFLOW/bands_fit.py settings.ini --dft-dir path/to/bands --dft-code vasp --outdir outputs/MoS2
```

* `--dft-code`: `vasp`, `qe`, `siesta` or `aims` (also `[reference] dft_code`).
* `--dft-dir` (also `[reference] dft_dir`, relative to the INI file).
* The structure positional argument is **optional**: the structure of the DFT
  calculation is used (written to `OUTDIR/dft_reference/POSCAR`). If you give one,
  it must have the same cell.
* The INI file only needs `[fit]`; template: `examples/configs/dft_template.ini`.
* `--prepare-only` reads the folder, writes `dft_reference/dft_bands_preview.png`
  (bands to fit in blue) and `dft_reference/dft_reference.json`, without fitting.
* `--from-fit fit.json` or `--from-parameters table --k K` with `--dft-dir`
  evaluates existing parameters against the DFT bands (one YAeHMOP run, figures).

## What is read

| Package | Eigenvalues and k points | Structure | Occupied bands | Labels |
|---|---|---|---|---|
| VASP | `vasprun.xml`, or `EIGENVAL` | vasprun / `POSCAR` / `CONTCAR` | `NELECT` | line-mode `KPOINTS` / `KPOINTS-bnd` |
| QE | `<prefix>.save/data-file-schema.xml` (or `<prefix>.xml`) of the `bands` run; else the pw.x output of that run; else the `bands.x` file (`&plot nbnd=...`) | XML / pw.x output | `nelec` | pw.x input with `K_POINTS crystal_b`/`tpiba_b` and `! label` |
| SIESTA | `<SystemLabel>.bands` + the `.fdf` with `%block BandLines` | `.STRUCT_OUT`, `.XV` or `.fdf` | Fermi level of `.bands` | `BandLines` |
| FHI-aims | `band1001.out`, … + `control.in` (`output band`) | `geometry.in` | occupations in the band files | `output band` |

When the files carry no labels, the high-symmetry points are named from the
Bravais lattice (ASE), including symmetry-equivalent positions; otherwise
`K1, K2, …`. If several candidate files exist, choose one with `--dft-file`.
If the occupied-band count is wrong, set `--valence-bands N`.

**Refused** (EHT here is spinless): spin–orbit / noncollinear calculations
(VASP `LSORBIT`/`LNONCOLLINEAR`, QE `noncolin`, SIESTA `SpinOrbit`/`Spin SO`),
and spin-polarized runs whose two channels differ. FHI-aims runs with
`include_spin_orbit` use the `*.no_soc` band files when they exist. Metals
(bands crossing the Fermi level, or no gap above the occupied bands) are refused,
because the fitter aligns and scores the bands around the gap. VASP hybrid runs
with the SCF mesh in the same file keep only the zero-weight path points.

## How the DFT path becomes the YAeHMOP path

The DFT k list is split into straight segments (from the file structure —
VASP line mode, BandLines, `output band` — or by direction changes). YAeHMOP
computes one band line per segment with `points_per_segment` points
(`[band_path]`, default 20), and the DFT eigenvalues are interpolated linearly
**at exactly those k points** along each segment. A discontinuity such as `X|U`
becomes an extra YAeHMOP line whose interior points are computed but excluded
from the error and drawn as a single tick `X|U`.

The periodic dimension comes from `--dimension` / `[band_path] dimension`, or is
detected from the vacuum (≥ 6 Å along the last lattice vectors: 2D materials
need the vacuum along c, 1D along b and c). Components of k along vacuum
directions are ignored by YAeHMOP.

The fit uses `n_pairs` valence and conduction bands around the DFT gap, each
set aligned to its own midgap, with the same objective and optimizers as the
image workflow (`--optimizer`, `--budget`, `--polish`, …).

## Standard outputs of every fit: `OUTDIR/fitting/`

Every fit of `bands_fit.py` (DFT, image or CSV reference) writes, by default to
`OUTDIR/fitting` (or `--fitting-dir`):

| File | Content |
|---|---|
| `parameters_huckel.dat` | fitted parameters, native YAeHMOP table |
| `parameters_huckel.txt` | `K_EHT` and every fitted variable |
| `bands_comparison.png/pdf/csv` | fitted bands: reference **blue solid**, EHT **red dashed**; RMSE, MAE, max \|ΔE\| and gaps/gap error in the figure |
| `bands_full_comparison.png/pdf/csv` | (DFT) all DFT and EHT bands in the `[reference] emin/emax` window |
| `hamiltonian_overlap.npz` | YAeHMOP **H(k)** (eV) and **S(k)**, complex `(nk, norb, norb)`, at the k points of the figure; `kpoints_frac`, `orbital_labels` (e.g. `Mo1:dz2`), `eigenvalues_eV`, `K_EHT` |
| `yaehmop_matrices.bind` | YAeHMOP input that produced H and S (`K points` + `Dump Hamil/Overlap`) |
| `fit_summary.json`, `README.txt` | metrics, settings, sources; file description |

H and S come from one extra YAeHMOP run with the final parameters; the
generalized eigenvalues of (H, S) reproduce the fitted bands (checked to 10⁻⁶ eV).
Orbital order is YAeHMOP's: s; px, py, pz; dx²−y², dz², dxy, dxz, dyz.

```python
import numpy as np
d = np.load("fitting/hamiltonian_overlap.npz")
H, S, k = d["H_k"], d["S_k"], d["kpoints_frac"]
```

## Validation examples

`validation/<package>/<example>/` holds real DFT band calculations, each with its
`fitting/` folder; `validation/run_validation.sh [bands_fit options]` refits all
of them. See `validation/README.md`.
