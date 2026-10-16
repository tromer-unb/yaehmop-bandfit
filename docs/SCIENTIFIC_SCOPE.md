# Scientific scope, accuracy and the source of reference curves

## What the code solves

During physical fitting, the Python frontend generates native YAeHMOP periodic input and compares selected valence/conduction eigenvalues against a reference band dataset. **Original YAeHMOP** diagonalizes the full orbital Hamiltonian, with physically meaningful nonorthogonal overlaps and orbital-based eigenvalues. The P5 fit objective scores exactly N=2,4,6,8 selected bands (N/2 occupied-side and N/2 conduction-side), not N=8 bands regardless of fit configuration. A constant energy alignment is handled in the objective. The image encoder is only a reference-input bridge, not a new Hamiltonian.

A fit against the same material is a *calibration*, not predictive generalization. Transfer requires previously fitted orbital parameters held fixed while evaluating an **excluded** material.

## Distinct published benchmark streams (do not pool)

1. **P5 Ge/Sn chalcogenides**: independently fitted 2,4,6,8 selected PBE reference bands in 6/6/6/5 eligible direct fits; respective mean direct RMSE about 0.229,0.248,0.273,0.321 eV. A separate leave-one-chemical-compound-out study yielded 0.807,0.909,0.704,3.416 eV (6/6/6/4 eligible folds). These are not all eight-band prediction errors.
2. **External C2DB graphene PBE**: an independently fitted patched carbon 2p double-zeta EHT potential gave a 0.32231 eV two-band RMSE on a 361-node EHT grid against interpolation of the same original C2DB 400-node PBE curve (single-zeta control: 0.58228 eV). **361 is not the number of independent DFT computations.**
3. **Independent PySCF/GTH-SZV PBE carbon 2D pilot**: a shared five-variable s+p EHT parameter table trained on graphene, T-graphene and net-C gave 0.16831,0.16011,0.15704 eV two-band errors on those **training materials**. Blind phagraphene 9-node path RMSE 0.15644 eV, but its SCF was Gamma-only and the full 2D Brillouin-zone topology is not converged or validated.
4. **Diamond and AB graphite, 3D**: direct EHT calibration with their own PBE/DZVP target spectra is **not** the same thing as frozen graphene-to-3D transfer. Frozen external-C2DB-graphene p-DZ7 parameters gave 2.22247 eV diamond and 2.18000 eV AB graphite two-band RMSE under another PBE numerical source, not universal transfer.

These numerical statements document research scope; **this GitHub repository does not rerun** any of those DFT/PBE optimizations. It ships small *saved EHT outputs* for plotting and local reproduction demonstrations without redistributing numerical third-party PBE input files.

## Important implementation distinctions

- Original stock YAeHMOP does not actually use second 2p Slater radial exponents/coefficients for carbon; implementing physical 2p double-zeta requires a separate audited native patch. The public frontend here uses the original YAeHMOP solver and **does not silently patch it**.
- The Shi one-pz carbon tight-binding model is a **model surrogate calibrated to HSE06**, not independently calculated HSE06 band energies.
- The P5 raster-to-band objective imposes a positive-frontier-gap gate (>about 0.02 eV). This implementation is not a ready-to-use Dirac crossing fitter for ideal graphene.
- Carbon 3D recalculation must use three actual periodic vectors and three active periodic overlap directions. It is explicitly implemented in the period-aware writer in this repository, but that does not establish a particular PBE reference or publication-grade convergence.
- Source plots and interpolated images are not themselves validation of orbital character, degeneracy, SOC, Berry curvature, topology or numerical DFT convergence.

## Reproducibility and provenance

Each `examples/precomputed/<system>` directory contains saved full-orbital EHT eigenvalues, original result provenance, source CIF name, number of sampled k points, number of orbitals and occupied spatial orbital index. These are archived training outcomes from the P6 calculations, not newly recalculated by the public package.

For a journal submission, record the YAeHMOP source version/build, Slater exponents, on-site energies, K, image radius, pseudopotential, functional, mesh, basis, k path, exact train/test split and licenses. The supplied illustrative parameters and plots should not be called a validated universal potential.

## Attributions

- YAeHMOP: G. A. Landrum and W. V. Glassey; see the [original YAeHMOP](https://github.com/greglandrum/yaehmop) and its manual.
- ASE: https://wiki.fysik.dtu.dk/ase/
- SciPy: https://scipy.org/
- PySCF (underlying research reference, not a required GitHub dependency): https://pyscf.org/

See the manuscript supplied through the project's research record when its bibliographic metadata and DOI are finalized; do not invent a journal issue or acceptance date.
