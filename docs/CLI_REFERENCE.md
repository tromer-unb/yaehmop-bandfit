# Complete command-line reference

All commands assume the current directory is the cloned repository root and a Python virtual environment has `requirements.txt` installed. Check `--help` for authoritative flags. The commands below are separated into **pure input/plotting work** and **solver work**.

## Main frontend: `USER_WORKFLOW/bands_fit.py`

```text
python USER_WORKFLOW/bands_fit.py STRUCTURE.cif SETTINGS.ini [options]
```

| Flag | Meaning and runtime |
|---|---|
| `cif` (positional) | Crystal structure in CIF format (required). |
| `input_file` (positional) | INI settings file with `[reference]`, `[band_path]`, `[fit]` (required). |
| `--image PATH`, `--png PATH` | Supply raster or CSV bands instead of INI `[reference] image`; `--png` is the backward-compatible alias. |
| `--name NAME` | Label for output and fitted material; no path separators. |
| `--outdir DIR` | Where to write the extracted reference, figures, fit and metadata. |
| `--pairs {1,2,3,4}` | **Number on EACH side**: 1V+1C, 2V+2C, 3V+3C, 4V+4C. |
| `--mode {auto,layered,colors,mono,csv}` | Choose input decoding/tracing approach. CSV recommended for exact data. |
| `--bbox L,T,R,B` | Plot interior rectangle **in pixels**, `left,top,right,bottom`. |
| `--colors '#RRGGBB,#RRGGBB,...'` | Expected RGB line color **in energy order**, one per selected band. |
| `--allow-ambiguous` | Overrides certain conservative tracing rejection checks; potentially unsafe scientifically. Avoid for publications. |
| `--validate-only` | **Fast**: check CIF/config and decode the image or CSV, record raster QA; **no band extraction and no YAeHMOP**. |
| `--prepare-only` | **Fast**: decode, extract, encode PNG bitplanes, preview and QA; **no YAeHMOP**. |
| `--budget N` | Requested physical solver objective evaluations. Only used when actually fitting. Staged fitting requires at least 80; SA refinements require >=160. |
| `--optimizer {nelder-mead,simulated-annealing,sa-nelder-mead,hooke-jeeves}` | Select staged base (physical original YAeHMOP) and optional refinement. See note below. |
| `--seed INTEGER` | Random seed for stochastic optimizers, default `20261014`. |
| `--from-fit path/to/fit.json` | **Runs solver**: load a saved fit's native parameter file and stored K, verify CIF SHA, regenerate all physical orbitals and plot. |
| `--from-parameters PATH` | **Runs solver**: compute full original YAeHMOP bands with YOUR table; skips image fitting. |
| `--k FLOAT` | Wolfsberg–Helmholtz constant, required with `--from-parameters`. |
| `--overlap-radius INT` | Number of images for **each active lattice direction** in recomputation, default `4`, allowed 1–8. Not a training-stage convergence guarantee. |

**Optimizer naming:** `nelder-mead` chooses the original staged YAeHMOP differential-evolution initializer followed by bounded Nelder–Mead. Other options run that stage plus a refinement; the total is split and `--budget` is a requested number of objective calls, not a wall-time limit. Results may vary by system. **Never run fitting during CI**.

```bash
# Check the command without running physics:
python USER_WORKFLOW/bands_fit.py examples/structures/graphene.cif \
  examples/configs/synthetic_2bands.ini --validate-only

# Prepare numerical toy input only:
python USER_WORKFLOW/bands_fit.py examples/structures/graphene.cif \
  examples/configs/synthetic_4bands.ini --mode csv \
  --prepare-only --outdir outputs/prepared_four

# Local physical fit, requires separately installed YAeHMOP:
python USER_WORKFLOW/bands_fit.py material.cif config.ini \
  --image target.csv --mode csv --pairs 2 --budget 300 \
  --optimizer nelder-mead --outdir outputs/physical_fit
```

## Image converter: `USER_WORKFLOW/prepare_band_png.py`

```text
python USER_WORKFLOW/prepare_band_png.py INPUT --outdir DIR
  --n-pairs {1,2,3,4} --emin EMIN --emax EMAX [options]
```

| Flag | Explanation |
|---|---|
| `INPUT` | Raster image or numeric CSV of target curves. |
| `--outdir DIR` | Directory for the PNG output, preview, QA and numeric curve extraction. |
| `--n-pairs 1/2/3/4` | 2/4/6/8 independently selected curves (N=2*n_pairs). |
| `--emin EMIN`, `--emax EMAX` | Physical energy values corresponding to bottom/top of plot, eV. |
| `--mode {auto,layered,colors,mono,csv}` | Source representation; see [image guide](IMAGE_INPUT.md). |
| `--bbox L,T,R,B` | Exact plot interior coordinates, before conversion but after EXIF handling. |
| `--colors '#hex,...'` | One line color per physically identifiable band. |
| `--color-tolerance INT` | Maximum pixel RGB distance allowed in curve extraction; default 36. |
| `--trace-nx INT` | Number of output x samples for trace/interpolation; default 301. |
| `--min-coverage FRACTION` | Required per-band coverage, default 0.75. |
| `--allow-ambiguous` | Relaxes some unsafe ambiguity checks. Do not enable without manual QA. |

It writes `bands_N_layers.png`, `extraction_preview.png`, `extraction_qa.json` and `extracted_curves.csv`. For ordinary raster inputs it additionally writes `normalized_source.png` and `raster_ingest_qa.json`. This is **not a solver call**.

## Region inspector: `USER_WORKFLOW/inspect_png.py`

`python USER_WORKFLOW/inspect_png.py --help` shows `IMAGE`, `--outdir`, optional `--bbox` and palette-size options. It normalizes raster and suggests a candidate `L,T,R,B` and visible hex colors. Suggestions require visual confirmation before fitting.

## Manual digitizer: `USER_WORKFLOW/digitize_bands.py`

`python USER_WORKFLOW/digitize_bands.py --help` shows explicit flags for `--template`, `--interactive`, `--annotations` and `--convert`. Interactive mode needs a graphical environment. It never invents hidden band identities; source and canonical SHA-256 hashes protect annotations against silent substitution.

## Plot archived physical eigenvalues: `scripts/plot_saved_bands.py`

```bash
python scripts/plot_saved_bands.py examples/precomputed/graphene/eht_full_bands.csv \
  --orbitals frontier --out outputs/graphene.pdf
python scripts/plot_saved_bands.py examples/precomputed/graphite_AB/eht_full_bands.csv \
  --orbitals 1,2,3,4 --emin -25 --emax 10 --out outputs/graphite.png
```

Options: `csv` (required), `--out` (required), `--orbitals frontier|all|1,2,...` (1-based), `--emin FLOAT`, `--emax FLOAT`. **Zero YAeHMOP calls**, using only CSV numeric values.

## Local crystal recomputation: `scripts/recompute_carbon.py`

`--system graphene|diamond|graphite`, `--parameters PATH` (required), `--k FLOAT` (default 1.75), `--overlap-radius INT` (default 4), `--outdir DIR`. Requires the user's external `YAEHMOP_BIND` and `YAEHMOP_PARAM_FILE` and explicitly **runs** physical band calculation. Does not optimize a fit.

## INI settings reference

```ini
[reference]
image = ../synthetic/toy_2bands.csv
mode = csv
emin = -3.5
emax = 3.5
trace_nx = 151
min_coverage = 0.75

[band_path]
dimension = 2
points = 0,0; 0.5,0; 0.3333333333,0.3333333333; 0,0
labels = G,M,K,G
points_per_segment = 20

[fit]
n_pairs = 1
budget = 200
optimizer = nelder-mead
gap_weight = 0.35
reg_weight = 0.04
```

Relative paths in `image =` resolve relative to **the INI file's directory**, not the shell's current directory. For `dimension=2`, k nodes contain 2 coordinates each; for `dimension=3`, use 3. The reciprocal paths must match the reference data's ordering and units. Do not copy an arbitrary graphene path onto a diamond figure and assume physical correctness.


## New user entrypoints: recommendation, frozen transfer, joint family fit

**Advisor (no solver; not an automatic electronic classifier):**

~~~bash
python USER_WORKFLOW/strategy_advisor.py --cif material.cif --reference bands.png   --goal direct --electronic semiconductor --phase 1H --bands auto   --outdir outputs/advisor
~~~

Options: --cif required; --reference optional; --goal direct|transfer; --electronic semiconductor|metal|unknown; --phase unknown|1H|1T|bulk; --bands auto|2|4|6|8; --outdir. The user declares the physical labels, and the advisor explains suitable workflow commands. It does not fit or guarantee best parameters.

**Frozen target evaluation (one native band calculation; ZERO optimizer calls):**

~~~bash
python USER_WORKFLOW/frozen_transfer.py TARGET.cif TARGET.ini   --source-run outputs/source/run_summary.json   --pairs 1 --outdir outputs/target_frozen
~~~

Alternative: --parameters PATH together with --k FLOAT. Mutually exclusive source modes. Optional --png/--image for target reference, --mode, --bbox, --colors, --allow-ambiguous (unsafe), --overlap-radius 1..8, --pairs 1..4 and --outdir. This command prepares target reference, reruns the physical solver once, computes the selected-window RMSE without parameter changes, writes transfer_report.json and full eigenbands. The source summary captures exact table and K and prevents evaluating the identical source CIF as a new target.

**One shared vector trained on ≥2 CIFs with ≥1 reserved complete heldout material:**

~~~bash
python USER_WORKFLOW/family_fit.py family.yaml --validate-only
python USER_WORKFLOW/family_fit.py family.yaml --prepare-only --outdir outputs/family
python USER_WORKFLOW/family_fit.py family.yaml --fit --budget 400   --seed 20261014 --outdir outputs/family
~~~

Options: manifest YAML, --validate-only|--prepare-only|--fit (exactly one), --budget, --seed, --outdir. The manifest [template](../examples/transfer/family_template.yaml) has n_pairs and train/test entries with name, role, structure, settings, reference, mode, weight. Fit uses native selected-window multi-system backend. Test PNG/CSV is opened only after training coefficients are frozen. This implementation covers selected-band spinless, gapped systems, not metallic full-BZ fitting. See [family guide](FAMILY_FIT.md).