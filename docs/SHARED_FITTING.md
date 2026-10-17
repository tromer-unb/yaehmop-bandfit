# Transferability: one parameter vector, several materials

> **2026-10-17 user-facing upgrade:** To fit a shared vector from multiple CIF+PNG/CSV training materials with exactly 2/4/6/8 matched bands and score whole held-out crystals **after freezing** use [FAMILY_FIT.md](FAMILY_FIT.md) and [family_fit.py](../USER_WORKFLOW/family_fit.py). For applying an existing source table to another target without optimization, use [frozen_transfer.py](../USER_WORKFLOW/frozen_transfer.py) and [START_HERE.md](START_HERE.md). The explanation of the older full-orbital fit_shared.py below remains accurate for that separate expert prototype.


The single-material workflow is [CIF + PNG → EHT parameters → all bands](FIT_YOUR_MATERIAL.md). Transferability asks a *different* question: **can parameters fitted on some complete crystal structures predict the bands of another structure that was not used for fitting?**

## See the actual archived demonstrations immediately (NO solver)

~~~bash
python3 quick.py chemical-transfer  # Ge/Sn study: leave-one-compound-out
python3 quick.py transfer           # Carbon sp2: shared graphene/T-graphene/net-C model → withheld phagraphene
python3 quick.py graphene           # Best graphene-only 2p double-zeta model (separate reference)
~~~

These commands plot saved, physically calculated EHT results without downloading third-party original DFT files. They do **not** retrain any model.

**Do not confuse direct fitting with transfer**:
- Direct fit: model is optimized on the material being evaluated.
- Shared fit: a single vector is optimized on several training structures.
- Leave-one-material-out: all bands of an entire test crystal are excluded until the vector is frozen.
- Target-adapted refitting is *not* zero-shot transfer.

## What works for a new material today?

The **individual** CIF+PNG+INI interface documented in [Fit your material](FIT_YOUR_MATERIAL.md) is the primary tested selected-band entrypoint. The current release also includes [family_fit.py](../USER_WORKFLOW/family_fit.py) for **joint 2/4/6/8-band shared optimization with test references withheld until the training vector is frozen**, and [frozen_transfer.py](../USER_WORKFLOW/frozen_transfer.py) to evaluate any existing supported native parameter table on another material with zero refitting. Both methods require real gapped spinless source bands and the user's authorized YAeHMOP executable.

There is a separate expert prototype, **parameter_fitter/fit_shared.py**, that accepts a YAML file with a **systems** list, each entry giving a name, CIF, image, energy range and k path plus **role: train** or **role: test**. It uses original native YAeHMOP. Its actual syntax is:

~~~bash
python3 parameter_fitter/fit_shared.py /path/to/config.yaml \
  --validate-only

python3 parameter_fitter/fit_shared.py /path/to/config.yaml \
  --tag shared --seed 42 --outdir outputs/shared
~~~

**Important current limitation:** the expert prototype traces the reference using **the complete number of YAeHMOP orbitals**, not the N=2/4/6/8 selected frontier curves used by **USER_WORKFLOW/bands_fit.py**. Consequently, **a normal two-band MoS2 PNG is NOT a plug-and-play input to fit_shared.py**, and the layered 8-bit public converter cannot represent more than eight curves. Do not claim the generic multi-material case is production-ready merely because this command accepts YAML. First implement a unified N-band, source-verified multi-system objective and matching input contract. Running **--validate-only** only checks schema; it does not validate the numerical physical reference.

The specialized source-code experiment for carbon's true 2p double-zeta and independent pilot PySCF/GTH-SZV PBE references lives in the separate research branch/provenance, and is **not** an automatic arbitrary-element option in the public solver. In particular, a true carbon double-zeta p basis is an isolated patched YAeHMOP executable, distinct from the unmodified upstream solver.

## How to design a valid transfer experiment

1. Choose **at least two distinct training lattices**, plus an independent test lattice. For MoS2, compare *different actual structures/strains with consistent physical references*; a new element absent from every training CIF has no learned element-specific orbital parameters.
2. Use compatible DFT method, pseudopotential, band selection, treatment of SOC/spin, energy zero and reciprocal sampling across all structures. Band curves or PNG axis limits must be physically identifiable.
3. Use **one global element parameter table and one global K**, with (if necessary) one energy-origin offset *only for numerical band comparison*. Never tune hopping/onsite parameters after opening test bands.
4. Equalize system contributions, not just total numbers of k points/atoms. Report both macro-average and worst held-out error.
5. Hold out an **entire material**, rather than only randomly held-out k points from the same fit target.
6. Compare models on **the same independent reference type**, and separately label model-to-model TB surrogates versus first-principles PBE/HSE bands.
7. Be explicit about which models are merely fitted, which are frozen, and whether the physical method and k meshes are converged.

## Published-archive benchmark scope

For the original six Ge/Sn chalcogenides, direct selected-band RMSE averages are 0.229/0.248/0.273/0.321 eV for 2/4/6/8 bands. The corresponding separate chemical held-out tests yield 0.807/0.909/0.704/3.416 eV with different eligibility counts. See [Chalcogenides](CHALCOGENIDES.md).

In a later carbon-only **pilot**, one shared five-variable carbon sp vector is fitted to PySCF/GTH-SZV PBE bands of graphene, T-graphene and net-C. Held-out phagraphene gives 0.1564 eV for **two frontier bands over just nine symmetry-path points**, with a *Gamma-only SCF reference*. It is **not** a converged 2D Dirac-cone validation, not universal carbon parameters, and not evidence for accurate 2D→3D diamond/graphite transfer. See [Best Results](BEST_RESULTS.md).

## Remaining development

The new family_fit.py wrapper now uses the selected-band multi-system physical backend and performs post-freeze withheld evaluation. It is not a universal environment-conditioned parameterizer and has not yet been independently benchmarked on new prospective materials beyond existing research cohorts. Remaining goals include a shared-data standards benchmark, metal-sensitive fitting with reliable BZ references, and automatic structure/PDF/PNG classification with quality uncertainties. [Start here](START_HERE.md).