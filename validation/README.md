# Validation examples: real DFT band structures

One folder per DFT package and one per example. Each example holds only the
files the reader needs (no wavefunctions, charge densities, pseudopotentials or
POTCAR). `fitting/<protocol>/` inside each example has the standard outputs of one fit
protocol (`frontier-de-300` is the first run; `frontier-de`, `window-de`, `window-ga`,
`window-cmaes`, `window-ga-basis` come from `compare_strategies.py`; `soc-from-<protocol>`
are spin-orbit fits made by `USER_WORKFLOW/soc_fit.py` on top of that protocol). Each has
(parameters, comparison figures, YAeHMOP H(R)/S(R) in the WanTiBEXOS text format `huckel-NP.dat` + `basis_set-NP`, and H(k)/S(k) at the high-symmetry points), produced by

```bash
python compare_strategies.py --work /tmp/strategies --collect   # all protocols, results into fitting/<protocol>
PROTOCOL=my-test ./run_validation.sh --optimizer cmaes --budget 600   # one protocol, any bands_fit.py option
```
Examples marked with a `SOC_ONLY` file are spin-orbit references, used by `soc_fit.py` only.

Only the **two best protocols** of each example (plus the spin-orbit fits) are in the
repository; a `fitting/.gitignore` per example keeps the other protocol folders local.
Matrix files above 10 MB are stored compressed (`huckel-NP.dat.gz`, `gunzip` before use
in WanTiBEXOS). Targeted refinements: `refine/`, `refine2/`, `refine3/`, `refine_w90/`
(settings `*.ini`, each starting from an earlier fit with `start_from`).

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
| SIESTA | `MoS2_monolayer_pbe_soc` | MoS₂ monolayer, PBE + SOC | 2D | `SOC_ONLY` (refused by spinless fits); SOC fit on top of the EHT of `aims/MoS2_monolayer_pbe/fitting/window-cmaes` (same cell) |
| FHI-aims | `MoS2_monolayer_pbe` | MoS₂ monolayer, PBE | 2D | all-electron eigenvalues |
| Wannier90 | `Si_diamond_vasp_pbe` | Si, VASP → Wannier90 (64 functions) | 3D | interpolated bands; `.nnkp` generated here with `wannier90.x -pp`; Fermi level from the OUTCAR |
| Wannier90 | `TiBr2_monolayer_vasp_pbe` | TiBr₂ monolayer, VASP → Wannier90 (32) | 2D | Fermi level (OUTCAR) 0.12 eV below the path VBM: the nearest gap is used |
| Wannier90 | `GaN_monolayer_qe_pbe` | GaN monolayer, QE → Wannier90 (16) | 2D | `fermi_energy` in the `.win` |
| FHI-aims | `MoS2_monolayer_hse06_soc` | MoS₂ monolayer, HSE06 | 2D | SOC run: the `*.no_soc` band files are used |
| VASP | `ZnI2_monolayer_pbe_soc` | ZnI₂ monolayer, PBE + SOC | 2D | `SOC_ONLY`; SOC fit on top of `ZnI2_monolayer_pbe/fitting/window-ga-basis` |
| VASP | `In2S2_monolayer_pbe_soc` | In₂S₂ monolayer, PBE + SOC | 2D | `SOC_ONLY`; SOC fit on top of `In2S2_monolayer_pbe/fitting/window-cmaes` |

The MoS₂ HSE06 example (FHI-aims) holds the SOC bands too: its spinless fits use the
`*.no_soc` files and `fitting/soc-from-window-ga` uses the bands with SOC.
