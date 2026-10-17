# Joint fitting and frozen whole-material transfer — 2/4/6/8 frontier bands

**This is a physical multi-system YAeHMOP workflow.** It optimizes **one** table of shared elemental onsite/radial parameters and **one** common Wolfsberg–Helmholtz K using the existing multi-system sparse-window backend. It does not average separately optimized tables or copy fit parameters from each new target. It requires real gapped nonmagnetic DFT reference bands and an installed native YAeHMOP engine.

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

**Limitations:** this public workflow is for positively gapped spinless semiconductors. It is not a general-purpose 1T-metal Fermi surface fitter, not an SOC/magnetic band tracker, and it does not select a model family automatically. For those studies, the separate P27/P28 research protocols are documented scientifically but not made into unsupported stock-YAeHMOP flags.