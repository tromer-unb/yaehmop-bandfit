# Joint (family) fitting: one parameter table for several systems

**This is a physical multi-system YAeHMOP workflow.** It optimizes **one** table of shared elemental onsite/radial parameters and **one** common Wolfsberg–Helmholtz K using the multi-system backend (`parameter_fitter/fit_sparse_window.py`). It does not average separately optimized tables. Typical uses: one table per chemical composition valid for several polymorphs and thicknesses (monolayer to bulk), or a table shared by a chemical family, with optional held-out test systems.

## Every reference of bands_fit.py, mixed freely

Each member of `systems:` is either

* a **DFT band folder** (`dft_dir`, `dft_code`: vasp, qe, siesta, aims, wannier90), read exactly as by `bands_fit.py --dft-dir`. Gapped members use the band-window objective, metallic members the metal objective (bands aligned at E_F), in the same fit. Per member, or once for all in `defaults:`, or in the `[reference]` section of a member's `settings:` INI:

  | key | meaning |
  |---|---|
  | `metallic` | `auto` (default: metal when the reader finds no gap), `true`, `false` |
  | `spin_channel` | `up`, `down`, or `average` (mean of both channels, for a spurious small magnetization) |
  | `semicore` | `auto` (default): drop (NELECT_DFT − n_EHT)/2 lowest bands, e.g. Ti 3s3p of a `Ti_sv` PAW potential; an integer; or `0` |
  | `fermi`, `valence_bands`, `dft_file` | as in `bands_fit.py` |
  | `dimension`, `points_per_segment` | periodic dimension (default: vacuum detection) and YAeHMOP points per line |
  | `emin`, `emax` | energy range of the full comparison figure |
  | `structure`, `structure_format` | optional; default the structure of the DFT run |
  | `weight` | weight of the member in the shared objective |

* or a **structure + band image/CSV + INI** (`structure`, `settings`, `reference`, `mode`, `bbox`, `colors`), as before (gapped, 2/4/6/8 frontier bands).

Global keys: `n_pairs`, `objective` (default: `metal` if a training member is metallic, `window` for DFT, `frontier` for images), `objective_settings` (the `[objective]` keys of the window/metal objectives), `basis_add`, `basis_dz` (double-zeta shells such as `S:p`, see [DFT_REFERENCE.md](DFT_REFERENCE.md); p needs the `bind` of github.com/ac-dias/yaehmop), `start_from` (+ `start_K`), `overlap_radius`, `gap_weight`, `reg_weight`, `overlap_min_eig`, `workers`.

Example: one table for TiS₂ in seven structures (VASP, PBE, spin-polarized runs with ~0 magnetization, Ti_sv):

~~~yaml
n_pairs: 1
defaults:
  dft_code: vasp
  spin_channel: average
  semicore: auto
  metallic: auto
systems:
  - {name: 1L_1T_TiS2, dft_dir: ../1L_PBE_1T_TiS2_Bands}
  - {name: 2L_1T_TiS2, dft_dir: ../2L_PBE_1T_TiS2_Bands}
  - {name: bulk_anatase_TiS2, dft_dir: ../bulk_PBE_anatase_TiS2_Bands}
  - {name: bulk_rutilo_TiS2, dft_dir: ../bulk_PBE_rutilo_TiS2_Bands, role: test}
~~~

~~~bash
python USER_WORKFLOW/family_fit.py family.yaml --prepare-only --outdir out/TiS2     # read, previews, no solver
python USER_WORKFLOW/family_fit.py family.yaml --fit --optimizer cmaes --polish nelder-mead \
       --budget 3000 --outdir out/TiS2
python USER_WORKFLOW/family_fit.py family.yaml --evaluate out/TiS2/fit/fit.json --outdir out/TiS2_eval
~~~

Options: every optimizer of `bands_fit.py` (`--optimizer`, `--polish`, `--budget`, `--seed`), `--objective`, `--basis-add`, `--basis-dz`, `--start-from`, `--workers` (members are computed in parallel), `--no-export`, `--evaluate FIT_JSON` (no fit: score every member with an earlier table).

Outputs: `fit/` (fit.json, param_best_native.dat, history), `parameters/parameters_huckel.dat` + `.txt` (the shared table and K), `systems/<name>/` with the comparison figures of every member (DFT blue, EHT red dashed) and, for DFT members, the standard `fitting/` export (H(R), S(R) in WanTiBEXOS format, POSCAR), `heldout/<name>/` for test members, `family_overview.png` (all members in one figure) and `family_report.json` (per-member errors, roles, hashes).

At least one training member is required; test members are optional (without them every reported error is a training error). Test DFT folders are read only after the shared vector is frozen.

Slab references computed with a bulk k path (Γ–A, L–M, K–H along the vacuum direction) are handled: segments with zero length in the periodic directions are dropped.

## Image/CSV members

## Quick preparation

Copy the commented [family_template.yaml](../examples/transfer/family_template.yaml) to a folder containing your own materials.

A typical structure:
~~~text
my_family/
  family.yaml
  train_MoS2/structure.cif
  train_MoS2/bands.png
  train_MoS2/file.ini
  train_WSe2/structure.cif
  train_WSe2/bands.png
  train_WSe2/file.ini
  test_WS2/structure.cif
  test_WS2/bands.png
  test_WS2/file.ini
~~~

**The compound names are illustrative only**; they are not bundled PBE files or converged training data. Do not try to use a PBE image from a different crystal or mismatched k path.

Example family.yaml:
~~~yaml
n_pairs: 1
gap_weight: 0.35
reg_weight: 0.04
systems:
  - name: train_MoS2
    role: train
    structure: train_MoS2/structure.cif
    settings: train_MoS2/file.ini
    reference: train_MoS2/bands.png
    mode: auto
    weight: 1.0
  - name: train_WSe2
    role: train
    structure: train_WSe2/structure.cif
    settings: train_WSe2/file.ini
    reference: train_WSe2/bands.png
    mode: auto
    weight: 1.0
  - name: test_WS2
    role: test
    structure: test_WS2/structure.cif
    settings: test_WS2/file.ini
    reference: test_WS2/bands.png
    mode: auto
~~~

**Important physical conditions:** the held-out material must be entirely excluded from fitting, but its elements must already be represented in the training compounds' atomic tables. The example train MoS2 + WSe2 covers Mo, W, S and Se, whereas training only MoS2 then testing WSe2 lacks W and Se parameters and is rejected rather than guessed.

Any `structure:` may also be a VASP, Quantum ESPRESSO, SIESTA or FHI-aims file (optional per-system `structure_format:`; see [STRUCTURE_FORMATS.md](STRUCTURE_FORMATS.md)).

Each file.ini uses the same standalone structure as the direct fit:
~~~ini
[reference]
image = bands.png
mode = auto
emin = -6
emax = 6
trace_nx = 301

[band_path]
dimension = 2
points = 0,0; 0.5,0; 0.3333333333,0.3333333333; 0,0
labels = G,M,K,G
points_per_segment = 20

[fit]
n_pairs = 1
budget = 400
optimizer = nelder-mead
~~~

These values are placeholders. Replace the k points, labels, energy range and input source with the actual material's bands. Source band images must have the matched number of selected curves. A generic PNG may require --mode colors or explicit bbox/colors in its manifest entries; numerical CSV is preferred when degeneracies obscure band identity.

## Step 1: validate the manifest (free)

~~~bash
python USER_WORKFLOW/family_fit.py my_family/family.yaml --validate-only
~~~

Checks source paths, complete train/test roles, distinct CIF hashes and elemental coverage. The target/reference PNG is **not digitized**, and YAeHMOP is not invoked. File existence and CIF structure are checked.

## Step 2: prepare TRAIN reference bands only (free)

~~~bash
python USER_WORKFLOW/family_fit.py my_family/family.yaml \
  --prepare-only --outdir outputs/my_family
~~~

This does not open the held-out PNG and does not call the native solver. It converts training source PNG/CSV into exactly N=2,4,6,8 band-bitplane images. Open every PREVIEW before fitting:
~~~text
outputs/my_family/prepared/train_MoS2/extraction_preview.png
outputs/my_family/prepared/train_WSe2/extraction_preview.png
outputs/my_family/prepared/<name>/extraction_qa.json
outputs/my_family/frozen_train_only.yaml
~~~

Failed extraction or incorrect axes must be corrected in the original input files. Successful bitplane round-trip does not prove source curve identity.

## Step 3: fit ONE common vector and evaluate the held-out material

Requires correctly installed native YAeHMOP bind and original atomic table, both supplied by the user.

~~~bash
export YAEHMOP_BIND=/absolute/path/to/yaehmop/bind
export YAEHMOP_PARAM_FILE=/absolute/path/to/eht_parms.dat

python USER_WORKFLOW/family_fit.py my_family/family.yaml \
  --fit --budget 400 --seed 20261014 --outdir outputs/my_family
~~~

What happens:
1. Only the **train** references are prepared and sent to the original selected-frontier-band YAeHMOP optimizer.
2. The optimizer uses **one** global vector shared between all training crystals. The per-compound objective contributions can be weighted, but this is not a validated universality guarantee.
3. After freezing parameters, source-fit data are saved as outputs/my_family/fit/fit.json, fit/param_best_native.dat and shared_frozen_source.json, which stores K and training hashes.
4. Only **after** the shared vector is frozen are the test-band images digitized and scored by USER_WORKFLOW/frozen_transfer.py (no optimizer).
5. A concise family_report.json records the numerical transfer error and the source/target hashes.

Output per held-out system:
~~~text
outputs/my_family/heldout/test_WS2/
  bands_comparison.png
  bands_comparison.pdf
  bands_comparison.csv
  frozen_bands.npz
  bands_full.npz
  transfer_report.json
  reference_prepared/extraction_preview.png
  reference_prepared/extraction_qa.json
~~~

n_pairs: 1 means 1V+1C; n_pairs: 2 means 2V+2C; 3 and 4 likewise. All training systems must use the **same** matched-window size; each must contain enough physical EHT orbitals.

## Exact transfer semantics

- Direct fit: the evaluated material's reference can enter the optimization.
- Joint family fit: only designated **training** materials' references enter optimization.
- Frozen test: the held-out reference is opened **after** the global vector is saved. Parameter vector and K remain exactly unchanged.
- Retrospective evaluation: if the researcher has inspected/used the test crystal in earlier design, the test is *not* prospectively blind even when the code enforces the current exclusion.
- Re-adaptation: fitting the test target afterward is another **direct calibration**, not the same frozen transfer measurement.

**Limitations:** image/CSV members must be gapped spinless references. Metallic and spin-polarized references are supported only as DFT members (metal objective, one spin channel or the channel mean); SOC is not part of the family fit (fit SOC afterwards with `soc_fit.py`).