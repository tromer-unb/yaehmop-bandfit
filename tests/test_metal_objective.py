import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parameter_fitter"))
from band_objective import WindowSettings, metal_error


class MetalObjective(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 1, 41)
        self.x = x
        self.ref = np.column_stack([-6 + 4 * np.cos(np.pi * x), -1 + np.sin(2 * np.pi * x), 0.5 + x, 8 + x])
        self.s = WindowSettings(window_min=-9, window_max=3)

    def test_rigid_shift_is_free(self):
        # EHT = DFT - 11.3 eV on an absolute scale: zero error, shift recovered, E_F of the model = -shift
        eht = np.column_stack([self.ref, self.ref[:, -1:] + 5]) - 11.3
        m = metal_error(self.ref, eht, self.x, [0, 0.5, 1], self.s)
        self.assertAlmostEqual(m["shift_eV"], 11.3, places=9)
        self.assertAlmostEqual(m["eht_fermi_eV"], -11.3, places=9)
        self.assertLess(m["total"], 1e-9)
        self.assertEqual(m["n_bands_compared"], 4)
        self.assertEqual(m["n_bands_in_window"], 3)          # the band at +8 eV is outside

    def test_error_near_EF_weighs_more(self):
        far = self.ref.copy(); far[:, 0] += 0.3                # error in the deep band
        near = self.ref.copy(); near[:, 2] += 0.3              # error in the band at E_F
        mf = metal_error(self.ref, far, self.x, [0, 1], self.s)
        mn = metal_error(self.ref, near, self.x, [0, 1], self.s)
        self.assertGreater(mn["fermi_weighted_rms_eV"], mf["fermi_weighted_rms_eV"])


if __name__ == "__main__":
    unittest.main()
