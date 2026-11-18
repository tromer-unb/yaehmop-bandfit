# Validation examples: real DFT band structures

One folder per DFT package and one per example. Each example holds only the
files the reader needs (no wavefunctions, charge densities, pseudopotentials or
POTCAR). `fitting/` inside each example has the standard fit outputs
(parameters, comparison figures, YAeHMOP H(R)/S(R) in the WanTiBEXOS text format `tb-NP.dat` + `basis_set-NP`, and H(k)/S(k) at the high-symmetry points), produced by

```bash
./run_validation.sh                       # settings in fit_settings.ini
./run_validation.sh --optimizer cmaes --budget 600   # any bands_fit.py option
```

These examples are meant for comparing fitting strategies (optimizers, budgets,
number of bands, objectives) on the same references.

| Package | Example | System | Dim | Notes |
|---|---|---|---|---|
| VASP | `TiBr2_monolayer_pbe` | 1T-TiBr₂ monolayer, PBE | 2D | `vasprun.xml` + `EIGENVAL` |
| VASP | `ZnI2_monolayer_pbe` | ZnI₂ monolayer, PBE | 2D | `EIGENVAL`; the stock EHT table has no Zn 3d orbitals (12 valence electrons on s/p), so YAeHMOP cannot be fitted until Zn d parameters are added |
| VASP | `In2S2_monolayer_pbe` | In₂S₂ monolayer, PBE, no SOC | 2D | `EIGENVAL`; labels K1… as in the KPOINTS file |
| QE | `Si_diamond_lda` | Si, diamond, LDA | 3D | pw.x output (text) |
| QE | `Si_wurtzite_lda` | Si, wurtzite, LDA | 3D | eigenvalues from the `bands.x` file (pw.x did not print them) |
| QE | `SiC_zincblende_pbe` | 3C-SiC, PBE | 3D | `sic.save/data-file-schema.xml` |
| SIESTA | `Al2O2_bilayer_test` | Al₂O₂ bilayer (test) | 2D | `.bands` + `BandLines`, structure from `.STRUCT_OUT` |
| SIESTA | `CNR_nanoribbon_1d_pbe` | carbon nanoribbon C₅₂H₁₆, PBE | 1D | 224 EHT orbitals |
| SIESTA | `MoS2_monolayer_soc_refused` | MoS₂ with spin–orbit | 2D | **refused on purpose** (spin–orbit): test of the refusal |
| FHI-aims | `MoS2_monolayer_pbe` | MoS₂ monolayer, PBE | 2D | all-electron eigenvalues |
| FHI-aims | `MoS2_monolayer_hse06_soc` | MoS₂ monolayer, HSE06 | 2D | SOC run: the `*.no_soc` band files are used |
