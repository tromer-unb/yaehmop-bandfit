# Installation and runtime environments

## Supported local environment

Recommended: Python 3.10+ on Linux or macOS. Native YAeHMOP is a separate C/Fortran executable and may require compiler, BLAS/LAPACK and platform-specific build dependencies. The Python input formatter and plotting utilities do **not** need native YAeHMOP.

```bash
git clone https://github.com/tromer-unb/yaehmop-bandfit.git
cd yaehmop-bandfit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python quick.py                 # See all six Ge/Sn materials + SnS₂ Fig. 4 FIRST
python quick.py sns2            # Just SnS₂ paper Figure 4
python USER_WORKFLOW/bands_fit.py --help
```

For Windows PowerShell, use `py -m venv .venv` followed by `.venv\Scripts\Activate.ps1`; YAeHMOP native executable compatibility may differ by environment. All examples use working-directory-relative paths from the cloned repository root.

## No native solver needed for these operations

- `python USER_WORKFLOW/inspect_png.py examples/synthetic/toy_2bands.png --outdir outputs/inspect`
- `python USER_WORKFLOW/bands_fit.py examples/structures/graphene.cif examples/configs/synthetic_2bands.ini --prepare-only --outdir outputs/preparation`
- `python scripts/plot_saved_bands.py examples/precomputed/diamond/eht_full_bands.csv --orbitals frontier --out outputs/diamond.png`
- `python -m unittest discover -s tests -v`

The synthetic sample depicts a semiconductor-like **toy** spectrum, not actual graphene DFT. The CSV files of carbon bands are saved YAeHMOP eigenvalues from earlier research calculations.

## Physical YAeHMOP installation (only when requested)

Obtain YAeHMOP from its upstream project, following its **own installation and license**:
- https://github.com/greglandrum/yaehmop
- https://github.com/greglandrum/yaehmop/blob/master/docs/bind_manual.pdf

A Conda distribution may also be available at https://anaconda.org/conda-forge/yaehmop; check that the `bind` executable and compatible atomic table are present. This repository intentionally **does not bundle the solver, compiled binary or original atomic parameters**.

Set shell variables to absolute paths:

```bash
export YAEHMOP_BIND=/absolute/path/to/bind
export YAEHMOP_PARAM_FILE=/absolute/path/to/eht_parms.dat
test -x "$YAEHMOP_BIND"
test -f "$YAEHMOP_PARAM_FILE"
```

Only do this if the installed YAeHMOP version is compatible with the expected `bind input_file parameter_file` command-line form. The driver captures native stdout and saves detailed logs alongside its outputs. The Python requirements are separate from the YAeHMOP build.

Optional independent manual check, with your own authorized input and parameters:

```bash
"$YAEHMOP_BIND" path/to/your_input.bind "$YAEHMOP_PARAM_FILE"
```

## Running saved data vs computing new data

The command below is **fast** and does not invoke the solver:

```bash
python scripts/plot_saved_bands.py \
  examples/precomputed/graphene/eht_full_bands.csv \
  --orbitals frontier --out outputs/graphene.pdf
```

By contrast, these commands **do** run a local physical YAeHMOP solver; only call them deliberately and supply a valid parameter table with C orbitals:

```bash
python scripts/recompute_carbon.py --system graphene \
  --parameters /absolute/path/to/your_C_parameters.dat --k 1.75 \
  --outdir outputs/graphene_physical

python scripts/recompute_carbon.py --system diamond \
  --parameters /absolute/path/to/your_C_parameters.dat --k 1.75 \
  --overlap-radius 4 --outdir outputs/diamond_physical

python scripts/recompute_carbon.py --system graphite \
  --parameters /absolute/path/to/your_C_parameters.dat --k 1.75 \
  --overlap-radius 4 --outdir outputs/graphite_physical
```

The `--from-parameters` interface uses a dimensionality-aware YAeHMOP `Lattice` block. For diamond and graphite it writes **3** lattice vectors with nonzero overlap images in all three active directions. Changing K, exponents, basis or orbital parameters changes the spectra; these are **not guaranteed to reproduce the saved paper fits**, whose best-fit parameters are not distributed here. The graph path in the INI sample is illustrative, not necessarily the one used to generate the archived CSV.

## Testing and CI

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The GitHub Actions workflow is intentionally **lightweight**: syntax/import, numeric CSV, raster conversion, source-dimensionality and rendering tests. It installs Python dependencies but does not download/run DFT, optimize EHT parameters, or install/compile YAeHMOP.
