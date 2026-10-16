# Optional research extension: genuine carbon 2p double-zeta

**Research-only C source patch for an external YAeHMOP checkout. It is NOT part of the standard installation or the original YAeHMOP stock binary.** The original `bind` source/binary and atomic tables are NOT redistributed here.

### What it changes

Stock YAeHMOP's 2p radial table does **not** use a second Slater exponent/coefficient for the carbon 2p orbital. The archived research patch extends three original `tightbind` source files (`bind.h`, `fileio.c`, `mov.c`) to support normalized contracted 2p double-zeta overlap, using all four radial contributions. It was verified on the original project's archived source snapshot and tested for exact single-zeta recovery. The patched binary is separate from stock YAeHMOP and is not built by GitHub CI.

### Rebuild with authorization

1. Obtain a legal original YAeHMOP checkout, its upstream license and the native development toolchain. Tested source snapshot in the research record: `yaehmop-4ded45bb2bb7a2d38d2527c991debce9226a9066` (the patch may **NOT** apply to more recent releases).
2. Make your **own copy** of the YAeHMOP `tightbind` source directory. Do not apply this to the stock solver used for direct baseline comparisons.
3. Verify the copy matches the archived source revision and read the entire patch before applying it.
4. In that specific `tightbind` directory, apply:
   ```bash
   patch --dry-run -p1 < /path/to/yaehmop-bandfit/experimental/carbon_2p_double_zeta.patch
   patch -p1 < /path/to/yaehmop-bandfit/experimental/carbon_2p_double_zeta.patch
   ```
5. Compile using the **appropriate original YAeHMOP makefile and native BLAS/LAPACK setup**. Never assume a successful compilation proves scientific equivalence; verify single-zeta limiting bands, orbital overlaps, lattice convergence and double-zeta coefficient sensitivity before results are published.

### Benchmark context

The C2DB external graphene PBE two-band benchmark gave **0.32231 eV** dense 361-node RMSE for a fitted native p-DZ7 model, versus **0.58228 eV** for the matched single-zeta sp5 control. The 361 PBE target points were interpolated from the original external C2DB PBE curve, **not new DFT calculations**. Separate transfer across carbon allotropes does not demonstrate a universal set of radial parameters, especially diamond/graphite 3D.

The base package **does not invoke the patch**. A physical fit with the public standard frontend uses the user's separately installed YAeHMOP solver. Refer to [scientific scope](../docs/SCIENTIFIC_SCOPE.md) for limitations and proper attribution.
