Outputs of one EHT (YAeHMOP) fit
================================
parameters_huckel.dat    fitted parameters in the native YAeHMOP table format
                         (use with: bind input.bind parameters_huckel.dat)
parameters_huckel.txt    K_EHT and every fitted variable (name value)
fit_summary.json         errors, gaps, optimizer, reference and structure used
bands_comparison.*       fitted bands: reference blue (solid), EHT red (dashed),
                         with RMSE / MAE / max |dE| / gap error in the figure
bands_full_comparison.*  (DFT references) all DFT and EHT bands in a window
hamiltonian_overlap.npz  YAeHMOP Hamiltonian H(k) [eV] and overlap S(k), complex
                         arrays (nk, norb, norb) at the k points of the figure,
                         with kpoints_frac, orbital_labels, eigenvalues_eV
                         (generalized eigenvalues of H, S) and K_EHT.
                         Python: d=numpy.load('hamiltonian_overlap.npz'); d['H_k'][0]
yaehmop_matrices.bind    YAeHMOP input that produced the matrices (K points block)
