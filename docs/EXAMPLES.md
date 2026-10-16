# Reproducible examples — no cloud calculation

This repository includes three different categories of demonstration. They serve different scientific purposes; do not present them as equivalent.

| Example | Input / structure | What is provided | Solver needed? |
|---|---|---|---|
| Synthetic bands | `examples/synthetic/toy_2bands.{csv,png}` and `toy_4bands` | **Original toy equations**, input parser/bitplane QA only; NOT DFT. | No |
| Graphene (2D) | `examples/structures/graphene.cif` | Saved original-YAeHMOP **full 8-orbital** eigenvalues on 37 k nodes, from an earlier 1V+1C fitted calculation; not external PBE source bands. | No for viewing |
| Diamond (3D) | `examples/structures/diamond.cif` | Saved original-YAeHMOP **full 8-orbital** eigenvalues on 25 k nodes from a 1V+1C direct fit. | No for viewing |
| AB graphite (3D) | `examples/structures/graphite_AB.cif` | Saved original-YAeHMOP **full 16-orbital** eigenvalues on 49 k nodes from a 1V+1C direct fit. | No for viewing |
| New physical calculation | Any of the three carbon CIFs | Recomputed full orbital dispersion using **your** separately installed native YAeHMOP and parameter table. | **Yes** |

## Display archived individual carbon bands immediately

```bash
python scripts/plot_saved_bands.py examples/precomputed/graphene/eht_full_bands.csv \
  --orbitals frontier --out outputs/graphene.pdf

python scripts/plot_saved_bands.py examples/precomputed/diamond/eht_full_bands.csv \
  --orbitals all --out outputs/diamond.pdf

python scripts/plot_saved_bands.py examples/precomputed/graphite_AB/eht_full_bands.csv \
  --orbitals 8,9 --out outputs/graphite.pdf
```

The `--orbitals` indices are **1-based**. `frontier` uses the `nocc_spatial` in the adjacent `eht_full_bands_metadata.json`, not an assumption that the Hamiltonian is exactly half filled. Original spectra are **physical EHT** computed with the complete orbital basis; in these fixtures only two frontier bands were fitted to a separate PBE source. The PBE reference is NOT shipped. Files include reduced fractional k coordinates and normalized x; inspect metadata before comparing any external system.

`examples/precomputed/<system>/frontier.png` supplies a small static preview for GitHub visitors. These images were generated locally once and are not recalculated by CI.

## Build the actual physical 3D bands yourself

```bash
export YAEHMOP_BIND=/path/to/bind
export YAEHMOP_PARAM_FILE=/path/to/eht_parms.dat

python scripts/recompute_carbon.py \
  --system diamond --parameters /path/to/C_params.dat --k 1.75 \
  --overlap-radius 4 --outdir outputs/diamond
```

Select `graphene` or `graphite` for the other systems. Their example INI paths are representative paths, not necessarily precisely the same as the source paper grids. Crucially, 3D diamond/graphite use three actual periodic translation vectors; no fake 2D lattice is used. `--k` and the provided table determine new EHT results; **the archived tuned fit cannot be reproduced numerically without its specific fitted parameters and original settings**. These new runs are optional, local and can be computationally nontrivial.

## Convert numerical band targets or PNG plots

```bash
# No raster involved; exact toy numeric target:
python USER_WORKFLOW/bands_fit.py examples/structures/graphene.cif \
  examples/configs/synthetic_2bands.ini --prepare-only --outdir outputs/toy

# Input is a normal color PNG (from the same original toy equations):
python USER_WORKFLOW/prepare_band_png.py examples/synthetic/toy_2bands.png \
  --outdir outputs/toy_image --n-pairs 1 --emin -3.5 --emax 3.5 \
  --mode colors --bbox 115,105,1210,820 \
  --colors '#c82d37,#1c7ad5'
```

The example graphene crystal has **no physical relation** to the artificially gapped toy band target. It is used only to demonstrate a CIF parser interface; no scientific fitting should be performed against these toy bands.

More details: [IMAGE_INPUT.md](IMAGE_INPUT.md), [NUMERIC_BANDS.md](NUMERIC_BANDS.md), [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md).
