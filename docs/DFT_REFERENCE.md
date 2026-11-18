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
| `huckel-NP.dat` | real-space **H(R)** and **S(R)** of YAeHMOP, plain text in the **WanTiBEXOS** non-orthogonal format (same as `siesta2wtb.py`; use `DFT= S`, `PARAMS_FILE= "huckel-NP.dat"`) |
| `basis_set-NP` | WanTiBEXOS basis file: index, species, x y z (Å), l, m, spin |
| `orbitals.dat` | orbital index, atom, orbital name, l, m |
| `hamiltonian_overlap_k.dat` | H(k), S(k) (raw YAeHMOP values, eV) at the high-symmetry points. Header as in WanTiBEXOS: `NP`, scissors, Fermi level (EHT midgap, absolute), 3 lattice vectors (Å), number of basis orbitals, number of hopping vectors (cells of `huckel-NP.dat`), number of k points; then `k1 k2 k3 i j ReH ImH ReS ImS` |
| `POSCAR` | the structure used (VASP format; exact cell and atom order of the matrices) |
| `yaehmop_matrices.bind` | YAeHMOP input of the matrix run (`K points` + `Dump Hamil/Overlap`) |
| `fit_summary.json`, `README.txt` | metrics, settings, sources, checks; file description |

All files are plain text except the figures.

**How H(R), S(R) are obtained.** YAeHMOP only writes H(k), S(k) (binary dumps of a
`K points` run). Its lattice sums stop at the overlap radius r, so H(R) and S(R)
for all cells −r…r are recovered **exactly** by a Fourier transform of H(k), S(k)
on the full Γ-centred mesh of (2r+1)^dim points. Convention:
H(k) = Σ_R e^{i2πk·R} H(R), H_ij(R) = ⟨φ_i(0)|H|φ_j(R)⟩ (checked against
interatomic distances). `fit_summary.json` reports how well H(R), S(R) rebuild
YAeHMOP's H(k), S(k) on the band path (≈10⁻⁷ eV; cells whose elements are all
below 10⁻⁷ are omitted).

**Energy zero of `huckel-NP.dat`.** `wtb.x` diagonalizes (H(k) − E_F·I, S(k)), which
is not a rigid shift in a non-orthogonal basis. The file therefore holds
H(R) − E_mid·S(R) with E_F = 0, where E_mid is the EHT midgap (`eht_midgap_eV`
in `fit_summary.json`): `wtb.x` reproduces the YAeHMOP bands relative to midgap
(checked for MoS₂: 10⁻⁴ eV near the gap, single-precision level). Add E_mid for
absolute YAeHMOP energies; `hamiltonian_overlap_k.dat` has the unshifted values.
Orbital order is YAeHMOP's: s; px, py, pz; dx²−y², dz², dxy, dxz, dyz
(l, m as in sisl: px m=1, py m=−1, pz m=0; dxy −2, dyz −1, dz² 0, dxz 1, dx²−y² 2).

## Objective, larger basis and strategies

For DFT references the default objective is the **band window** (`--objective window`,
`parameter_fitter/band_objective.py`, ported from dftb_sk_gen): every DFT band from −6 to
+4 eV around the VBM against the EHT band at the same place counted from the gap, both
VBM-aligned, with the Fermi level in the middle of the gap and every state weighted by
g(E) = 0.05 + 0.95·exp(−|E−E_F|/2 eV). Terms: valence and conduction RMS, a sharper
frontier term, indirect and direct gap, VBM/CBM (energy, position, edge states),
high-symmetry points and the local maxima/minima of every band (energy and position).
Weights go in an `[objective]` INI section (`w_extrema = 6`, `energy_scale = 1.5`, ...);
`--objective frontier` keeps the earlier n_pairs objective.

`--basis-add "S:d H:p"` or `auto` adds shells YAeHMOP accepts but the stock table lacks
(p on H, d on s/p elements, the 3d of Zn); they start from estimates and are fitted.

`validation/compare_strategies.py` refits all examples with several protocols
(`frontier-de`, `window-de`, `window-ga`, `window-cmaes`, `window-ga-basis`) into
`validation/<code>/<example>/fitting/<protocol>/`; see the manual for the table. The window
objective wins everywhere; genetic + Nelder-Mead and CMA-ES + Nelder-Mead alternate as
best; the extended basis is needed for Si, SiC, the nanoribbon and ZnI₂.

## Spin–orbit coupling (`USER_WORKFLOW/soc_fit.py`)

YAeHMOP has no SOC. `soc_fit.py` adds on-site λ L·S per element and shell on top of a
spinless fit (EHT parameters fixed) and fits the λ's against a DFT calculation with SOC
(band window + spin-orbit splittings of the pairs near the gap). The spinor basis has
the n orbitals with spin up first, then the same n with spin down. Outputs in
`fitting-soc/`: `soc_parameters.txt`, `huckel-SOC.dat` + `basis_set-SOC` (WanTiBEXOS SOC
format), figures, `fit_summary.json`.

```bash
python USER_WORKFLOW/soc_fit.py --dft-dir path/to/bands-soc --dft-code vasp \
    --from-fitting outputs/material/fitting --outdir outputs/material_soc
```

## Validation examples

`validation/<package>/<example>/` holds real DFT band calculations, each with its
`fitting/` folder; `validation/run_validation.sh [bands_fit options]` refits all
of them. See `validation/README.md`.
