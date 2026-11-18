Spin-orbit outputs (EHT + on-site lambda L.S)
=============================================
soc_parameters.txt     fitted lambda (eV) per element and shell
parameters_huckel.dat  the EHT table used (fixed, from the spinless fit)
huckel-SOC.dat         spinor H(R), S(R) in the WanTiBEXOS SOC format: SOC / scissors /
                       Fermi level (0) / 3 lattice vectors [A] / 2n orbitals / n cells /
                       R_x R_y R_z [A]  i  j  ReH  ImH  S. Orbitals 1..n are spin up,
                       n+1..2n the same orbitals spin down. ReH + i ImH is
                       H(R) - E_mid*S(R) (x) I2 plus lambda L.S on R = 0 (E_mid in
                       fit_summary.json), so energies are relative to the EHT midgap.
basis_set-SOC          WanTiBEXOS basis file (spin column 1 / -1)
orbitals.dat           spinless orbital list (index, atom, orbital, l, m)
bands_comparison.*     2 VB + 2 CB with SOC: DFT blue, EHT+SOC red dashed, errors
bands_full_comparison.* all bands in the window
POSCAR                 structure used
fit_summary.json       lambdas, errors with and without SOC, settings
