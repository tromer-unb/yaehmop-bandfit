# Start here — fit one material or test transfer (CIF + bands)

This guide contains **real commands**. The CIF never creates a DFT reference; you must supply a physically documented PNG or numeric band CSV.

## 1. Install once

~~~bash
git clone https://github.com/tromer-unb/yaehmop-bandfit.git
cd yaehmop-bandfit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python quick.py                 # saved figures, no physics calculation
~~~

For a physical fit or transfer, separately install [upstream YAeHMOP](https://github.com/greglandrum/yaehmop) and its licensed atomic parameter table; neither is bundled.

~~~bash
export YAEHMOP_BIND=/absolute/path/to/yaehmop/bind
export YAEHMOP_PARAM_FILE=/absolute/path/to/eht_parms.dat
test -x "$YAEHMOP_BIND" && test -f "$YAEHMOP_PARAM_FILE"
~~~

These environment variables address the common YAeHMOP bind-not-found error.

## 2. Which strategy should I try?

~~~bash
python USER_WORKFLOW/strategy_advisor.py \
  --cif path/to/structure.cif --reference path/to/bands.png \
  --goal direct --electronic semiconductor --phase 1H \
  --bands auto --outdir outputs/advisor
~~~

The result is outputs/advisor/recommendation.json. This is a *rules-based recommendation* and does not automatically infer the band gap/phase from the CIF, extract the PNG or run a solver.

## 3. OPTION A: direct fitting for one crystal

~~~bash
mkdir -p inputs/source
cp /your/files/structure.cif inputs/source/structure.cif
cp /your/files/bands.png inputs/source/bands.png
cp examples/configs/material_2bands_template.ini inputs/source/file.ini
# EDIT file.ini: real emin/emax, reference image, k-path nodes and labels.

python USER_WORKFLOW/inspect_png.py inputs/source/bands.png --outdir outputs/png_check
python USER_WORKFLOW/bands_fit.py inputs/source/structure.cif inputs/source/file.ini \
  --prepare-only --outdir outputs/source
~~~

Inspect outputs/source/reference_prepared/extraction_preview.png and extraction_qa.json. Make sure crossings and visible curves match the original **before** fitting. Bad occlusions require a digitized or source numerical CSV, not a fabricated curve.

~~~bash
python USER_WORKFLOW/bands_fit.py inputs/source/structure.cif inputs/source/file.ini \
  --pairs 1 --budget 400 --outdir outputs/source

python USER_WORKFLOW/bands_fit.py inputs/source/structure.cif inputs/source/file.ini \
  --from-fit outputs/source/fit/fit.json --outdir outputs/source_full
~~~

The direct fit outputs bands_comparison.png/pdf/csv, fit/fit.json, fit/param_best_native.dat and run_summary.json. The recomputation outputs bands_full.npz/csv/png/pdf containing **all** physical orbital bands, not just the selected fit window. The optimizer is not guaranteed to reach a global minimum.

**Pairs**: --pairs 1 = 1V+1C; --pairs 2 = 2V+2C; --pairs 3 = 3V+3C; --pairs 4 = 4V+4C. Never increase the number without a real multi-band reference.

## 4. OPTION B: frozen transfer, no refit

Supply a **different** target CIF, its **own** reference image and a target-specific INI file (real energy bounds/path). Required elements must exist in the frozen parameter table and the orbital basis must support the requested bands.

~~~bash
mkdir -p inputs/target
cp /your/other/structure.cif inputs/target/structure.cif
cp /your/other/bands.png inputs/target/bands.png
cp examples/configs/material_2bands_template.ini inputs/target/file.ini
# EDIT target file.ini to match target's real reciprocal path and energy bounds.

python USER_WORKFLOW/frozen_transfer.py \
  inputs/target/structure.cif inputs/target/file.ini \
  --source-run outputs/source/run_summary.json \
  --pairs 1 --outdir outputs/target_frozen
~~~

It reads the frozen source table and K, physically calculates **one** target EHT band spectrum and scores it with **ZERO optimizer calls**. It refuses exactly identical source and target CIF files. It produces bands_comparison.png/.pdf/.csv, bands_full.npz, frozen_bands.npz, the reference-preparation QA and transfer_report.json.

Alternatively, if you already have shared parameters from elsewhere:

~~~bash
python USER_WORKFLOW/frozen_transfer.py TARGET.cif TARGET.ini \
  --parameters /path/to/shared_native_parameters.dat \
  --k YOUR_ACTUAL_K --pairs 1 --outdir outputs/frozen
~~~

This mode is *frozen evaluation*, not necessarily a prospectively blind experiment. Document whether the target was previously inspected.

## 5. OPTION C: joint shared fit, then held-out transfer

The new family-facing wrapper uses the tested multi-system selected-band objective. Prepare family.yaml with at least **two training crystals and one separate test crystal**, each with their own CIF, PNG/CSV and INI. See [FAMILY_FIT.md](FAMILY_FIT.md) and [template](../examples/transfer/family_template.yaml).

~~~bash
python USER_WORKFLOW/family_fit.py family.yaml --validate-only

python USER_WORKFLOW/family_fit.py family.yaml \
  --prepare-only --outdir outputs/family

# Inspect every outputs/family/prepared/NAME/extraction_preview.png BEFORE fitting.

python USER_WORKFLOW/family_fit.py family.yaml \
  --fit --budget 400 --outdir outputs/family
~~~

The optimizer receives **training references only**. After the family parameter vector is frozen, target images are opened and compared. Expect outputs/family/fit/fit.json, shared_frozen_source.json, family_report.json and per-target heldout/NAME/bands_comparison.png and transfer_report.json.

The separate older expert fit_shared.py traces full orbital images; it is **not** a convenient ordinary two-band PNG interface. Use family_fit.py for the 2/4/6/8 selected-frontier-band contract.

## 6. The PNG format explained

The internal bands_N_layers.png is a lossless 8-bit bitplane encoding of 2, 4, 6 or 8 distinct bands, one bit per band. Crossing bands can share one pixel. The software **generates** it from a source raster/CSV; users do not need to draw it. This preserves *encoded* crossings, not physically hidden or completely occluded source curves. Axis limits in eV, plot rectangle and actual reciprocal path are mandatory for meaningful work.

Use a numerical CSV when screenshot extraction is ambiguous, or supply separate known RGB colors and plot bounds. See [image guide](IMAGE_INPUT.md) and [PNG standard](PNG_STANDARD.md).

## 7. Scope and limitations

These public fitting and transfer entrypoints support **spinless, nonmagnetic, positively gapped systems with an eligible EHT orbital basis**. Metallic 1T and ideal Dirac graphene require specialized objectives, not a disguised semiconductor fit. P27/P28 d/p coupling research variants are not arbitrary-element stock-YAeHMOP flags; better direct path RMSE need not imply better transferable Fermi physics.

No new DFT calculation is performed by these scripts. Third-party DFT arrays and upstream YAeHMOP binary/atomic tables are not redistributed.