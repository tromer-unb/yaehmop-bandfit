# Numerical bands instead of PNG: exact CSV protocol

**Prefer numerical band energies when available.** A raster plot cannot reveal an occluded band, a physical level crossing with identical colors, hidden spin branches, or the original DFT source precision. The CSV input is not digitized; it is read and checked directly.

## Input CSV contract (fitted target)

The input file must be UTF-8 CSV with exactly the following headers, for N=2/4/6/8 respectively:

```csv
x,band_1,band_2
x,band_1,band_2,band_3,band_4
x,band_1,band_2,band_3,band_4,band_5,band_6
x,band_1,band_2,band_3,band_4,band_5,band_6,band_7,band_8
```

- `x`: monotonically strictly increasing **normalized reciprocal path coordinate**, starts at `0` and ends at `1` (not physical k in inverse angstroms).
- `band_1 … band_N`: **energies in eV**, ordered ascending at every sample k; no missing, infinite, NaN or repeated-x values. Must have at least 15 rows.
- Energy levels must be inside the INI `[reference]` values `emin` and `emax`, which define the PNG encoder's vertical axis.
- For gapped semiconductor targets, `N/2` columns correspond to the selected valence states and `N/2` to the selected conduction states. The original fitter requires a positive frontier gap (>~0.02 eV).
- The reference CSV can include actual ab initio PBE energies **if licensed and scientifically validated**, not just synthetically generated toy data.

Example (toy data, not DFT):

```bash
head -4 examples/synthetic/toy_2bands.csv
python USER_WORKFLOW/prepare_band_png.py examples/synthetic/toy_2bands.csv \
    --mode csv --n-pairs 1 --emin -3.5 --emax 3.5 \
    --outdir outputs/toy_numeric
```

The resulting `bands_2_layers.png` is the same canonical decoder contract as if colors had been traced from an image; it is not the original DFT file.

## Two different CSV formats

**Do not confuse** the fitted-target CSV above with the archived physical full-orbital EHT CSVs under `examples/precomputed/<system>/eht_full_bands.csv`, which use `x,kx,ky,kz,orbital_1,...`. Those files intentionally contain the **entire original YAeHMOP EHT spectrum**, including unconstrained orbitals, and are only used with `scripts/plot_saved_bands.py`. The `kx,ky,kz` columns are reduced reciprocal coordinates; `x` is normalized path distance.

The archived EHT numeric examples are precomputed, **not independent DFT reference data**. Using a full-orbital EHT CSV as a PBE reference would incorrectly make an EHT→EHT task, not a first-principles validation.

## Source checks for scientific use

Document the DFT functional, pseudopotential, orbital/plane-wave basis, k-point sampling, spin/SOC, units, absolute energy offset and band-index conventions for your original target. An eigenvalue CSV plus a mismatched CIF or k path is not reproducible. For graphene gapless Dirac physics, the gapped P5 CSV fitting protocol alone is not an adequate independent test; use the appropriate carbon-specific objective described in the scientific research, not invented finite-gapped curves.
