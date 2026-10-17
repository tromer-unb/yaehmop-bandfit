# What the P5–P30 research shows: select a task, not just a smaller RMSE

These are **historical archived results** from separate studies, not a new universal benchmark of this public release. Some use different DFT bases or electronic reference approximations. Numerical comparisons must be made **within** their documented protocol, not pooled across all systems.

## Direct fitting versus excluded-material transfer

The original six-material GeS/GeSe/GeTe/SnS2/SnSe/SnTe 2-band study, with matching 1V+1C, found:

| Fit type | 2-band mean RMSE |
|---|---:|
| Direct to each material | 0.229 eV |
| One vector, held-out whole material | 0.807 eV |

The 4/6/8-band results use 2V+2C / 3V+3C / 4V+4C objectives with separate eligibility counts. See [six-compound benchmark](CHALCOGENIDES.md) for denominators. A direct fit and held-out whole-material transfer do **not** answer the same question.

## Carbon is a distinct chemistry and basis problem

An audited physical carbon double-radial 2p treatment improved an external C2DB/PBE (no-SOC) graphene **361-node interpolated** two-band check from 0.5823 to 0.3223 eV. This genuine double-zeta experiment uses a separate **native YAeHMOP source patch**, NOT the stock solver in this repository.

A shared 2D planar-carbon pilot trained on graphene, T-graphene and net-C gave a frozen phagraphene result of 0.1564 eV for two frontier bands **on a nine-point path generated from a Γ-only SCF**. This is a limited pilot, not converged graphene-family Dirac physics. The same graphite/diamond 3D coordination/dimensionality cannot be treated as a universal extension of graphene parameters; several frozen 2D-to-3D errors exceeded 2 eV under their own independent PBE reference protocol.

## Chalcogenide 1H versus 1T: a more demanding property test

Earlier joint family work produced a frozen 33-parameter EHT vector (P16). In a later retrospective 1T WS2 study, adding three grouped metal d onsite corrections (P27 CF3) reduced its two-band path error from 0.1326 eV to 0.1020 eV and reduced retrospective two-band path errors in WSe2/MoS2/MoSe2, but degraded the WS2 numerical finite-grid Fermi crossing mask's IoU from **0.515 to 0.273**.

An extended P28 d/p coupling correction improved the WS2 path further to **0.0915 eV**, and Gaussian DOS(EF) to **1.952** compared with finite PBE **1.972** states/(eV cell), but the Fermi-cell IoU was still only **0.385**, worse than frozen P16 **0.515**. Transfer to MoSe2 worsened from 0.2698 eV with P16 to 0.2803 eV with P28 F3PD. The **predeclared model-promotion gate failed**, so P16 remains the appropriate family baseline and P28 F3PD is experimental.

Numerical values for the four 1T materials are in [processed P27/P28 table](../examples/research/p27_p28_retrospective_1T_metrics.json). It contains no raw third-party PBE eigenvalues.

## How this maps to this public tool

- **DIRECT**: fitting is legitimate when you want one material to be reproduced and its DFT reference is available. [Practical guide](START_HERE.md).
- **FROZEN TRANSFER**: reuse the exact same parameters/K on another CIF and measure the error without any new optimizer calls. [Executable evaluator](../USER_WORKFLOW/frozen_transfer.py).
- **JOINT FAMILY**: fit one parameter vector across two or more training structures with consistent N=2/4/6/8 windows, and then evaluate reserved test structures without refitting. [Executable wrapper](FAMILY_FIT.md).
- **METALS**: the present user-facing PNG-based objective assumes a positive spinless gap; do **not** run it on a 1T metal and pretend a one-dimensional screenshot establishes Fermi topology. Metal/d/CF3/F3PD work remains a distinct research protocol; it is not exposed as a generic YAeHMOP flag.

**Paper status:** P30 is an author-review Computational Materials Science manuscript draft; it is **not yet a peer-reviewed publication** and the full research archive/third-party electronic-reference data are not bundled in this public repository. Public release of the P30 Overleaf package itself should follow the authors' approval, bibliography review and data/figure-rights audit.

The separate rule-based [strategy advisor](../USER_WORKFLOW/strategy_advisor.py) produces transparent suggestions, **not** an empirical guarantee that it can infer the universally best Hamiltonian from a CIF and PNG.