> **Carbon extension only.** The original paper's main dataset comprises
> **GeS, GeSe, GeTe, SnS₂, SnSe and SnTe** with matched **2/4/6/8-band**
> calibration and full chemical zero-shot validation. **Start with
> [the original six-material benchmark and SnS₂ Figure 4](CHALCOGENIDES.md).**
> This page describes the later graphene/phagraphene/diamond/graphite work.

---

# Best results first — real saved EHT bands, instant examples

The command `python3 quick.py carbon` or `python3 quick.py graphene` explores
carbon; `python3 quick.py` also shows the original Ge/Sn results **first** and
then generates the **carbon extension** entirely offline, using archived numerical YAeHMOP predictions and previously audited summary errors. It is intended for **readers of the paper who want to see the best working examples immediately**, not to train a new Hamiltonian.

## One-command gallery

```bash
python3 quick.py best       # both P5 six-material outcomes and separate carbon
python3 quick.py graphene   # graphene C2DB p-DZ7 fit, archived EHT
python3 quick.py transfer   # graphene / phagraphene family
```

Carbon-extension outputs in `outputs/quick/`:

| Figure/file | Scientific content |
|---|---|
| `carbon_best_results.png` | Three clearly separated PBE benchmark groups, with RMSE in eV. |
| `graphene_best_pdz7.png` | **Best graphene model against external C2DB PBE:** native genuine 2p-double-zeta p-DZ7, saved 361-node **EHT** valence and conduction predictions. |
| `transfer_graphene_phagraphene.png` | Same **frozen shared sp5** carbon parameters: graphene training bands and independently tested phagraphene holdout predictions. |
| `carbon_graphene_diamond_graphite.png` | Saved full-Hamiltonian YAeHMOP frontier orbitals from the earlier separately fitted graphene/diamond/graphite experiments. |
| `image_input/bands_2_layers.png` | Demonstration of an ordinary RGB toy band plot converted to a two-band native bitplane PNG, with quality checks. |

Other commands: `python3 quick.py best`, `python3 quick.py graphene`, `python3 quick.py transfer`, `python3 quick.py carbon`, `python3 quick.py image`.

## Which fits are really the best?

### 1. External C2DB graphene PBE: best specific model

| Model | Two-band RMSE |
|---|---:|
| **Genuine native p-DZ7 (best)** | **0.32230993 eV** |
| Stock single-zeta sp5 (control) | 0.58227733 eV |

Training used 37 k nodes. The 361-node comparison evaluates YAeHMOP predictions against interpolation of the **same original external C2DB PBE** curve (no 361 independent DFT computations). The winning genuine p-DZ7 model **requires the separately documented experimental native patch**; passing a two-zeta 2p table to **stock** YAeHMOP does **not** activate a second 2p radial Slater function.

The actual numerical best-fit parameters are in `examples/best_results/metrics.json`: `Es=-22.55867858` eV, `z_s=2.83601207`, `Ep=-19.53673544` eV, `z_p1=1.67929374`, `z_p2=2.74023084`, `c2/c1=0.78123265`, `K=2.00168664`. The saved model-predicted valence/conduction energies are in `graphene_c2db_pdz7.csv`.

The fast command displays **saved EHT bands** without redistributing the original third-party PBE curve. The RMSE comes from the archived verified original comparison, not a new calculation during quick mode.

### 2. Different PySCF/GTH-SZV PBE source: graphene + carbon-family transfer

| Task | Model | Two-band RMSE |
|---|---|---:|
| Graphene, **individual fit** | stock sp5, fitted only to graphene | **0.16569896 eV** |
| Graphene, **shared training** | stock sp5, trained jointly on graphene, T-graphene, net-C | 0.16831147 eV |
| T-graphene, training | same frozen shared sp5 | 0.16011356 eV |
| net-C, training | same frozen shared sp5 | 0.15704140 eV |
| Phagraphene, **genuine holdout** | same frozen shared sp5, **not retrained on phagraphene** | **0.15643801 eV** |
| Phagraphene, alternative nested p-DZ7 | double-zeta initialized from sp5; second radial practically inactive | 0.15628056 eV |

The numerically smallest phagraphene pilot number is 0.15628056 eV, but its fitted second 2p coefficient is nearly zero and it is not convincing evidence of a physical double-zeta transfer benefit. For an easy, interpretable showcase, **shared sp5 is the recommended transfer example** (0.15644 eV). Its 5 shared EHT parameters are also in `metrics.json`: `Es=-13.19019898` eV, `z_s=2.63406311`, `Ep=-9.43181231` eV, `z_p=2.03848864`, `K=3.22401644`.

**Scientific caution:** the phagraphene reference follows a nine-node path extracted from **Gamma-only SCF**, not a fully converged 2D Brillouin zone. This cannot prove a fundamental band gap, Dirac point or converged slope. Graphene and T-graphene and net-C numbers are **training errors**, whereas phagraphene is the held-out test.

**Do not rank 0.322 vs 0.166 as if they came from a single DFT reference.** The C2DB source and PySCF source differ in basis/functionals/settings; they are distinct experiments, shown in separate dashboard groups on purpose.

## Numerical files: models, not third-party references

All `examples/best_results/*.csv` contain archived **EHT predicted bands only** with header `x,EHT_valence_eV,EHT_conduction_eV`. They are intentionally **not** the input band CSV convention `x,band_1,band_2` of the gapped fitter, because they are **not independent PBE reference values**. They cannot be used as a physical DFT benchmark on their own. Each source, target, training status and fitted parameter vector are identified in `metrics.json`.

The earlier `examples/precomputed/{graphene,diamond,graphite_AB}/` files contain **saved full-orbital original YAeHMOP** spectra with metadata for independently tuned P6 direct fit trials; they do **not** use the same shared graphene-to-3D parameter table. These are accessible instantly via `python3 quick.py carbon`.

## Recomputing rather than viewing

`python3 quick.py` never loads YAeHMOP, DFT/PySCF, an optimizer or the network. To compute new bands using YOUR parameters and solver, read [INSTALLATION.md](INSTALLATION.md), `scripts/recompute_carbon.py` and [the native p-DZ7 patch](../experimental/README.md) (for that specific experimental model). Recomputing the published p-DZ7 graphene result from stock binary is **not** supported. Heavy parameter fitting is always explicit and local, never on GitHub.
