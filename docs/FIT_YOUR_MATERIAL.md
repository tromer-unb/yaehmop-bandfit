# Fit your own material: CIF + band image → EHT parameters → all bands

**Start here if you have a crystal structure and a band-structure plot.** This is the tested individual-material workflow for **2, 4, 6, or 8 selected bands**. The *reference* may be a conventional PNG/JPEG/TIFF/WebP or a numeric CSV.

What you provide:

- **structure.cif** — your material's structure (MoS2, WS2, GeSe, etc.).
- **bands.png** — its reference DFT band plot, with known energy-axis limits and reciprocal path. Prefer a numeric CSV if available.
- **file.ini** — a small configuration specifying the energy axis, k path, and number of selected bands.

What the program produces: fitted native YAeHMOP orbital table and K, numerical RMSE, a DFT-versus-EHT PNG/PDF/CSV plot, and a way to **recalculate all physical EHT eigenbands without refitting**.

> **Scope:** the present image-to-fit frontend assumes a nonmagnetic, spinless, **gapped** semiconductor. It rejects near-zero/negative frontier gaps. Graphene with a Dirac crossing, spin-polarized systems, SOC-split bands, and other metallic references need specialized objectives. "Any material" means any material with supported YAeHMOP orbitals and compatible reference bands, **not** universal physics for all substances.

## 0. Install once

From a clean clone of this repository (all following commands assume its root):

~~~bash
git clone https://github.com/tromer-unb/yaehmop-bandfit.git
cd yaehmop-bandfit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Obtain/build original YAeHMOP separately, then point to YOUR installation.
export YAEHMOP_BIND=/absolute/path/to/yaehmop/bind
export YAEHMOP_PARAM_FILE=/absolute/path/to/yaehmop/eht_parms.dat
test -x "$YAEHMOP_BIND" && test -f "$YAEHMOP_PARAM_FILE"
~~~

No third-party YAeHMOP binaries, its atomic parameter table, or external DFT numerical sources are redistributed here. **Only steps 3–4 actually run the physical solver.** Converting PNG or reviewing archived examples does not.

## 1. Place your MoS2 (or other) files in one folder

~~~bash
mkdir -p inputs/my_material

cp /your/path/structure.cif inputs/my_material/structure.cif
cp /your/path/bands.png inputs/my_material/bands.png

cp examples/configs/material_2bands_template.ini inputs/my_material/file.ini
~~~

**Edit inputs/my_material/file.ini** before continuing. The template shows a Γ–M–K–Γ path and energies −6 to +6 eV as *examples only*; these must be changed to the actual reference image and your CIF reciprocal lattice.

~~~ini
[reference]
image = bands.png
mode = auto
emin = -6
emax = 6
trace_nx = 301
min_coverage = 0.80

[band_path]
dimension = 2
points = 0,0; 0.5,0; 0.3333333333,0.3333333333; 0,0
labels = G,M,K,G
points_per_segment = 20

[fit]
n_pairs = 1
budget = 400
optimizer = nelder-mead
gap_weight = 0.35
reg_weight = 0.04
~~~

- **n_pairs=1:** 1 valence + 1 conduction; change to 2, 3 or 4 *only when the reference actually contains 4, 6 or 8 identifiable bands.*
- **emin/emax:** numerical energies at the bottom/top of the *plot frame*, not estimated band extrema.
- **points/labels:** fractional reciprocal nodes as used by **that** CIF and **that** DFT path. Set dimension=3 for genuine 3D crystals and provide 3 coordinates per node.
- The image path is relative to the INI file. You can override it with **--image** or **--png** (both are valid aliases in this GitHub edition).

## 2. Inspect the PNG and prepare the band reference (fast: NO solver)

~~~bash
python USER_WORKFLOW/inspect_png.py inputs/my_material/bands.png \
  --outdir outputs/inspect_my_material

python USER_WORKFLOW/bands_fit.py \
  inputs/my_material/structure.cif inputs/my_material/file.ini \
  --validate-only --outdir outputs/my_material

python USER_WORKFLOW/bands_fit.py \
  inputs/my_material/structure.cif inputs/my_material/file.ini \
  --prepare-only --outdir outputs/my_material
~~~

Open **outputs/my_material/reference_prepared/extraction_preview.png** and **extraction_qa.json**. Verify the reference bands actually follow the source image throughout the k path. If the plot rectangle is wrong, rerun with **--bbox 110,80,920,650** using the *actual* plot coordinates; for two explicitly colored lines use **--mode colors --colors '#c82d37,#1c7ad5'** with the actual color hexes.

Preparing the input produces **bands_2_layers.png** and **extracted_curves.csv**. The layered PNG encodes each selected band in a different binary bit so genuine recorded crossings can share a pixel. It **cannot reconstruct physically hidden or illegible curves** from a compressed screenshot.

### Standalone PNG-to-standard-format conversion

~~~bash
python USER_WORKFLOW/prepare_band_png.py inputs/my_material/bands.png \
  --mode colors --n-pairs 1 --emin -6 --emax 6 \
  --bbox 110,80,920,650 --colors '#c82d37,#1c7ad5' \
  --outdir outputs/converted
~~~

This yields **bands_2_layers.png**, extracted curves, QA JSON and an extraction preview. For four, six or eight bands use **--n-pairs 2, 3 or 4**, with that many true reference curves and distinct colors (if in colors mode). See [Image guide](IMAGE_INPUT.md).

## 3. Fit your material (physical YAeHMOP solver required)

~~~bash
python USER_WORKFLOW/bands_fit.py \
  inputs/my_material/structure.cif inputs/my_material/file.ini \
  --pairs 1 --budget 400 --optimizer nelder-mead \
  --outdir outputs/my_material
~~~

The optimizer uses exactly the selected N=2×pairs frontier reference curves and exactly N YAeHMOP eigenvalues. It does *not* optimize all other orbital bands. A 400-call budget does not guarantee a global minimum; repeat with seeds/higher budget if evaluating convergence.

Expected outputs:

| Path | Contents |
|---|---|
| outputs/my_material/bands_comparison.png | Reference versus selected physical EHT bands |
| outputs/my_material/bands_comparison.pdf | Vector-like printable plot |
| outputs/my_material/bands_comparison.csv | Numerical selected DFT/EHT energies per k |
| outputs/my_material/fit/param_best_native.dat | Native fitted YAeHMOP orbital table |
| outputs/my_material/fit/fit.json | Parameters, K, fit metrics and optimizer state |
| outputs/my_material/run_summary.json | RMSE, K, source hashes and run settings |

## 4. Reproduce **all** physical EHT bands without refitting

~~~bash
python USER_WORKFLOW/bands_fit.py \
  inputs/my_material/structure.cif inputs/my_material/file.ini \
  --from-fit outputs/my_material/fit/fit.json \
  --outdir outputs/my_material_full
~~~

Check **outputs/my_material_full/bands_full.png**, **bands_full.csv** and **bands_full.npz**. The CSV/NPZ store **all** YAeHMOP orbital eigenvalues; the PNG displays only the chosen energy interval. The original *same* CIF is SHA-256-checked against the fitted reference.

If you already have a native parameter table **and know its K**, you can skip fitting:

~~~bash
python USER_WORKFLOW/bands_fit.py \
  inputs/my_material/structure.cif inputs/my_material/file.ini \
  --from-parameters /absolute/path/to/your_parameters.dat \
  --k 1.75 --outdir outputs/with_existing_parameters
~~~

**1.75 is illustrative, not the fitted K.** Read the correct K from run_summary.json or fit.json.

## When the PNG is bad: use numerical CSV instead

A numerical CSV is the **preferred scientific reference**, especially for crossings/overlap. For 2 bands its header must be exactly **x,band_1,band_2**, energies ascending in each row, with ≥15 samples, x strictly increasing from 0 to 1. Save it as **inputs/my_material/bands.csv**, then:

~~~bash
python USER_WORKFLOW/bands_fit.py \
  inputs/my_material/structure.cif inputs/my_material/file.ini \
  --image inputs/my_material/bands.csv --mode csv \
  --prepare-only --outdir outputs/my_material_csv

python USER_WORKFLOW/bands_fit.py \
  inputs/my_material/structure.cif inputs/my_material/file.ini \
  --image inputs/my_material/bands.csv --mode csv \
  --pairs 1 --budget 400 --outdir outputs/my_material_csv
~~~

If your only source is an unreadable image, [manual band digitization](IMAGE_INPUT.md#manual-digitization) is possible, but it requires visual annotation: missing curves are never invented.

## Essential CLI flags

| Flag | Example | Purpose |
|---|---|---|
| --image / --png | --png bands.png | Override reference source (same option) |
| --pairs | --pairs 2 | 2 V + 2 C = 4 selected bands |
| --budget | --budget 400 | Requested local physical YAeHMOP evaluations |
| --optimizer | --optimizer nelder-mead | Staged differential evolution + local refinement |
| --mode | --mode colors | Decode auto, colors, mono, layered or csv |
| --bbox | --bbox 100,80,920,650 | Pixel rectangle inside figure axes |
| --colors | --colors '#c82d37,#1c7ad5' | True per-band curve colors |
| --validate-only | (none) | Check files/format without fitting |
| --prepare-only | (none) | Extract/preview curves without fitting |
| --from-fit | --from-fit path/fit.json | Re-run solver from saved fit and K |
| --from-parameters | --from-parameters params.dat | Run solver using supplied native table |
| --k | --k 1.75 | Required with --from-parameters |
| --overlap-radius | --overlap-radius 5 | Neighbor-image radius in *recomputation* |
| --seed | --seed 42 | Deterministic optimizer initialization |
| --outdir | --outdir outputs/test | Result directory |

For the complete flags see [CLI reference](CLI_REFERENCE.md). To fit **several materials with shared parameters**, read [Multi-material transfer guide](SHARED_FITTING.md); this is a separate advanced interface and has stricter reference-band requirements than the individual 2/4/6/8 frontend.

**No DFT or physical parameter fitting is run by GitHub Actions or by these documentation examples unless you explicitly execute step 3 locally.**
