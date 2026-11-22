# Fitting to raw DFT band data (VASP, Quantum ESPRESSO, SIESTA, FHI-aims)

Instead of an image or a CSV, the reference can be the **folder of a DFT
band-structure calculation**. The program reads the eigenvalues themselves,
the k points, the crystal structure and the number of occupied bands, and fits
the extended Hückel parameters to those numbers. No digitization is involved.

```bash
python USER_WORKFLOW/bands_fit.py settings.ini --dft-dir path/to/bands --dft-code vasp --outdir outputs/MoS2
```

* `--dft-code`: `vasp`, `qe`, `siesta`, `aims` or `wannier90` (also `[reference] dft_code`).
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
| Wannier90 | `seedname_band.dat` + `seedname_band.kpt` (interpolated TB bands, `bands_plot = true`) | `seedname.nnkp` (`real_lattice`) or `.win` (`unit_cell_cart`) + atoms of the `.win` | bands below the Fermi level: `fermi_energy` in the `.win`, else the OUTCAR / pw.x output in the folder | `seedname_band.labelinfo.dat` |

Spin-polarized runs whose channels differ only through a spurious small magnetization can be fitted
to the mean of both channels with `--spin-channel average` (`[reference] spin_channel = average`).
Semicore states of the DFT potential that the EHT basis lacks (Ti 3s3p of `Ti_sv`, Ga 3d, ...) are dropped
by default: `--semicore auto` (`[reference] semicore`) removes the (NELECT − EHT electrons)/2 lowest bands
when the DFT electron count is known (VASP, QE); give an integer, or 0 to keep them. This matters for the
metal objective, which pairs bands from the bottom.

When the files carry no labels, the high-symmetry points are named from the
Bravais lattice (ASE), including symmetry-equivalent positions; otherwise
`K1, K2, …`. If several candidate files exist, choose one with `--dft-file`.
If the occupied-band count is wrong, set `--valence-bands N`.

**Refused** (EHT here is spinless): spin–orbit / noncollinear calculations
(VASP `LSORBIT`/`LNONCOLLINEAR`, QE `noncolin`, SIESTA `SpinOrbit`/`Spin SO`),
and spin-polarized runs whose two channels differ. FHI-aims runs with
`include_spin_orbit` use the `*.no_soc` band files when they exist. Metals
(bands crossing the Fermi level, or no gap above the occupied bands) are refused by the
gap-based objectives (`window`, `frontier`); fit them with `objective = metal` (below).
VASP hybrid runs
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
For a metal (`objective = metal`) E_mid is replaced by the EHT energy of the DFT Fermi level
(`eht_midgap_eV` = `eht_fermi_eV`), so the zero of `huckel-NP.dat` is E_F.

**Exportability.** H(R), S(R) are only useful if S(k) is positive definite. Very diffuse
shells (small ζ, often added ones) with the lattice sums cut at r cells can break this.
The fitter therefore accepts a new best candidate only if the smallest eigenvalue of S(k)
at every k point of the export (Γ mesh + band path) is at least `overlap_min_eig`
(`[fit]`, default 10⁻³; 0 disables the check); `fit.json` → `overlap_check` counts the
checks and rejected candidates. The export never stops: `fit_summary.json` →
`tight_binding_files.overlap_min_eigenvalue` (and `warning` if it is not positive).
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

Targeted refinement: `band_weights = v1:10 v2:6 c1:10 ...` with `w_bands` (named bands counted
from the gap), `w_slope` (dispersion: ΔE between consecutive k points) and `w_mass` (curvature
at the band edges) in `[objective]`; `w_split` (`[objective]`, also in a family manifest's `objective_settings`) compares the splittings,
i.e. the energy differences between neighbouring bands at every k point, DFT against EHT (g-weighted,
inside the window). It targets the splitting of the N copies of a band in an N-layer slab, which the
band RMS barely penalizes when all copies stay close to the DFT average. `segment_weights = 0:3 3:3`
multiplies the state weights of path segments 0 and 3 (counted along the path, as in the figure ticks)
by 3, to stress one region of the Brillouin zone. Both work with the window and metal objectives; the
metric `rms_split_eV` is reported.

`--start-from FITTINGDIR` starts from an earlier fit (also of
another material) with the bounds centred there. Examples: `validation/refine/*.ini`.

`--basis-add "S:d H:p"` or `auto` adds shells YAeHMOP accepts but the stock table lacks
(p on H, d on s/p elements, the 3d of Zn); they start from estimates and are fitted.

`--basis-dz "S:p"` (`[fit] basis_dz`, `basis_dz:` in a family manifest) makes a shell double
zeta: its exponent becomes the inner function and a diffuse tail (ζ2 = ζ1/2, c2/c1 = 0.3 at the
start) is added; ζ1, ζ2 and c2/c1 are fitted. Use it when one exponent cannot give both the bonds
inside a layer and the coupling across a van der Waals gap (splitting of the bands of the
different layers of a slab). d and f shells work with the original YAeHMOP; s/p shells need a
`bind` with double-zeta p, [github.com/ac-dias/yaehmop](https://github.com/ac-dias/yaehmop)
(commit 742719c or later), set through `YAEHMOP_BIND`. The original program reads the line but
uses only the first exponent, so every YAeHMOP run checks the `.out` (`3P DZ:`) and stops if
the second exponent was ignored. The exported `huckel-NP.dat` and the WanTiBEXOS interface of
the fork work with double-zeta p.

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

## Spin-polarized references (`--spin-channel`, `USER_WORKFLOW/spin_combine.py`)

A collinear spin-polarized run is fitted channel by channel (`--spin-channel up`, then `down`;
occupied bands below the common Fermi level, each channel needs a gap). `spin_combine.py
--up FIT_UP --down FIT_DOWN --outdir OUT` joins both in one spinor basis (n spin-up orbitals,
then the same n spin-down), with zero up-down blocks and each channel shifted by
(DFT midgap of the channel − E_F) so that the exchange splitting of the DFT is kept. Output:
`huckel-SP.dat` + `basis_set-SP` (WanTiBEXOS `SP` format).

## Metals (`objective = metal`)

No gap, so no midgap or VBM to align at. The DFT bands are taken relative to the Fermi level
and paired from the bottom with the EHT bands (DFT band j ↔ EHT band j); the EHT bands are
shifted by the constant c that minimizes the g-weighted RMS (closed form), and −c is the Fermi
level of the model. Settings in `[objective]` (same names as `window`): `window_min`,
`window_max` (relative to E_F), `energy_scale`, `far_weight`, `w_frontier`, `frontier_scale`,
`w_high_symmetry`, `w_extrema`, `w_position`, `w_slope`, `band_weights` (`v1` = lowest band).
`fermi = …` in `[reference]` replaces the Fermi level of the band files (use the SCF one).
`fit_summary.json` also counts the EHT states below the model E_F on the export mesh
(electrons per cell in that channel). For a spin-polarized metal fit each channel with
`--spin-channel`; both are already at the common E_F, so `spin_combine.py` adds no shift.
Example: `validation/vasp/Fe_bcc_pbe_spin` with `validation/refine_fe/`:

```ini
[reference]
dft_code = vasp
fermi = 5.86385846
[band_path]
points_per_segment = 20
dimension = 3
[objective]
window_min = -9
window_max = 3
energy_scale = 2
far_weight = 0.15
[fit]
objective = metal
budget = 10000
optimizer = cmaes
polish = nelder-mead
```

The tool exports the Hamiltonian to WanTiBEXOS in all three cases: `huckel-NP.dat`
(spinless), `huckel-SOC.dat` (spin–orbit) and `huckel-SP.dat` (spin-polarized).

## Validation examples

`validation/<package>/<example>/` holds real DFT band calculations, each with its
`fitting/` folder; `validation/run_validation.sh [bands_fit options]` refits all
of them. See `validation/README.md`.
