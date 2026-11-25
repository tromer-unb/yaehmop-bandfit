"""Position matrix of the Slater basis (huckel-*_r.dat, WanTiBEXOS BSE_CENTER_FILE) and the frame of the
YAeHMOP run. The checks against bind need YAeHMOP (skipped otherwise)."""
from pathlib import Path
import sys, tempfile, unittest
import numpy as np
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "parameter_fitter"))
from position_matrix import slater_functions, position_matrix, write_rmatrix, read_rmatrix
SI = BASE / "validation/qe/Si_diamond_lda/fitting/window-ga-basis"


def bind_ok():
    try:
        from yaehmop_generic import resolve_yaehmop_assets
        resolve_yaehmop_assets()
        return True
    except Exception:
        return False


class SlaterFunctions(unittest.TestCase):
    def test_coefficients_as_bind(self):
        # double zeta: normalized with the overlap of the two exponents (fileio.c)
        sh = dict(n=4, nzeta=2, zeta1=4.869, zeta2=1.644, c1=1.0, c2=1.258135)
        (c1, z1), (c2, z2) = slater_functions(sh, "d")
        s12 = (4 * z1 * z2 / (z1 + z2) ** 2) ** 4.5
        self.assertAlmostEqual(c1 * c1 + c2 * c2 + 2 * c1 * c2 * s12, 1.0, places=12)
        self.assertAlmostEqual(c1, 0.5511, places=3)
        # s and single-zeta p: c = 1 whatever the table says; single-zeta d: c1 as given
        self.assertEqual(slater_functions(dict(n=3, nzeta=1, zeta1=1.7, c1=0.5), "p"), [(1.0, 1.7)])
        self.assertEqual(slater_functions(dict(n=3, nzeta=1, zeta1=1.0, c1=0.9), "d"), [(0.9, 1.0)])

    def test_one_atom(self):
        """An isolated atom: S = 1 on the diagonal, X_ii = position, on-site s-p dipole along the p axis."""
        table = {"Si": {"s": dict(n=3, nzeta=1, zeta1=1.6, c1=1.0), "p": dict(n=3, nzeta=1, zeta1=1.4, c1=1.0)}}
        pos = np.array([[0.3, -0.2, 0.5]])
        X, S = position_matrix(pos, ["Si"], table, np.eye(3) * 30, [(0, 0, 0)], [np.ones((4, 4))], nproc=1)
        self.assertTrue(np.allclose(S[0], np.eye(4), atol=1e-6))
        self.assertTrue(np.allclose([X[0][i, i] for i in range(4)], np.repeat(pos, 4, axis=0), atol=1e-6))
        d = X[0][0, 1:] - pos[0] * S[0][0, 1:, None]          # <s|r - A|p_x,y,z>
        self.assertGreater(abs(d[0, 0]), 0.1)                  # <s|x|p_x> != 0
        self.assertAlmostEqual(d[0, 0], d[1, 1], places=6)
        self.assertAlmostEqual(d[0, 1], 0.0, places=8)

    def test_file_round_trip(self):
        rng = np.random.default_rng(0)
        Rs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0)]
        X = rng.normal(size=(3, 2, 2, 3))
        with tempfile.TemporaryDirectory() as t:
            write_rmatrix(Path(t) / "x_r.dat", Rs, [X, X], "test")
            nb, D = read_rmatrix(Path(t) / "x_r.dat")
        self.assertEqual(nb, 4)
        self.assertTrue(np.allclose(D[(1, 0, 0)][:2, :2], X[1], atol=1e-9))
        self.assertTrue(np.allclose(D[(1, 0, 0)][:2, 2:], 0))   # up-down blocks zero
        self.assertTrue(np.allclose(D[(-1, 0, 0)][2:, 2:], X[2], atol=1e-9))


@unittest.skipUnless(bind_ok() and SI.is_dir(), "YAeHMOP not configured or Si example missing")
class AgainstBind(unittest.TestCase):
    def test_si_overlap_frame_and_symmetry(self):
        """Si diamond (fcc primitive cell, whose POSCAR frame differs from bind's): the overlap of the
        quadrature equals bind's S(R) only in the frame of the YAeHMOP run (yaehmop_frame)."""
        from yaehmop_generic import (resolve_yaehmop_assets, parse_parameter_table, subset_for_elements,
                                     write_parameter_file, write_periodic_input_generic, run_bind, gamma_mesh,
                                     read_matrix_dump, real_space_matrices, contiguous_along_vacuum, yaehmop_frame)
        from structure_io import read_structure
        at = contiguous_along_vacuum(read_structure(SI / "POSCAR", "vasp"), 3)
        tab = subset_for_elements(parse_parameter_table(SI / "parameters_huckel.dat"), ["Si"])
        bind, _ = resolve_yaehmop_assets()
        r = 1
        mesh = gamma_mesh(3, r)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_parameter_file(t / "p", tab)
            inp = write_periodic_input_generic(SI / "POSCAR", t / "m.bind", None, 1.98, True, 8, overlap_images=(r, r, r),
                                               periodic_dim=3, structure_format="vasp", kpoints=mesh)
            run_bind(bind, inp, t / "p", t / "l")
            cells = real_space_matrices(read_matrix_dump(str(inp) + ".HAM"), read_matrix_dump(str(inp) + ".OV"),
                                        mesh, 3, r)
        Rs = [tuple(int(v) for v in R) for R, _, _ in cells]
        Sref = [S.real for _, _, S in cells]
        ay = yaehmop_frame(at)
        X, Sq = position_matrix(ay.positions, ay.get_chemical_symbols(), tab, ay.cell, Rs, Sref, nproc=2)
        self.assertLess(max(np.abs(Sq[i] - Sref[i]).max() for i in range(len(Rs))), 1e-6)
        _, Sp = position_matrix(at.positions, at.get_chemical_symbols(), tab, at.cell, Rs, Sref, nproc=2)
        self.assertGreater(max(np.abs(Sp[i] - Sref[i]).max() for i in range(len(Rs))), 0.1)   # POSCAR frame: wrong
        i0 = Rs.index((0, 0, 0))
        self.assertTrue(np.allclose([X[i0][k, k] for k in range(18)], np.repeat(ay.positions, 9, axis=0), atol=1e-6))
        cell = np.asarray(ay.cell)
        for i, R in enumerate(Rs):
            j = Rs.index(tuple(-v for v in R))
            T = np.array(R, float) @ cell
            self.assertTrue(np.allclose(X[i], np.transpose(X[j], (1, 0, 2)) + Sq[j].T[:, :, None] * T, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
