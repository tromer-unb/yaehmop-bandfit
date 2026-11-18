Outputs of one EHT (YAeHMOP) fit (all plain text except the figures)
==================================================================
parameters_huckel.dat      fitted parameters in the native YAeHMOP table format
                           (use with: bind input.bind parameters_huckel.dat)
parameters_huckel.txt      K_EHT and every fitted variable (name value)
fit_summary.json           errors, gaps, optimizer, reference, structure, checks
bands_comparison.*         fitted bands: reference blue (solid), EHT red (dashed),
                           with RMSE / MAE / max |dE| / gap error in the figure
                           (.png, .pdf; .csv with the curves)
bands_full_comparison.*    (DFT references) all DFT and EHT bands in a window
huckel-NP.dat              real-space Hamiltonian H(R) and overlap S(R) in the
                           WanTiBEXOS format (as siesta2wtb.py; DFT= S):
                           NP / scissors / Fermi level (0) / 3 lattice vectors [A] /
                           n orbitals / n hopping vectors (cells) / then for every
                           cell and orbital pair: R_x R_y R_z [A]  i  j  ReH  ImH  S.
                           ReH is H(R) - E_mid*S(R) (E_mid = EHT midgap, in
                           fit_summary.json), so energies are relative to midgap and
                           wtb.x reproduces YAeHMOP's bands (wtb.x uses H - E_F*I,
                           which is not a shift when S != I).
basis_set-NP               WanTiBEXOS basis file: index species x y z [A] l m spin
orbitals.dat               orbital index, atom, orbital, l, m (YAeHMOP order)
hamiltonian_overlap_k.dat  H(k), S(k) at the high-symmetry points of the figure.
                           Header as huckel-NP.dat (Fermi level = EHT midgap,
                           absolute, since these H(k) are not shifted) plus the
                           number of k points; then k1 k2 k3 i j ReH ImH ReS ImS,
                           with H(k) = sum_R exp(i 2pi k.R) H(R)
yaehmop_matrices.bind      YAeHMOP input of the matrix run (K points + Dump)
POSCAR                     the structure used (VASP format, exact cell of huckel-NP.dat)
README.txt                 this description

H(R), S(R) are recovered exactly from YAeHMOP's H(k), S(k) on the full
(2r+1)^dim mesh (r = overlap radius); fit_summary.json reports how well they
rebuild YAeHMOP's H(k), S(k) on the band path (round-off level).
