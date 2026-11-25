"""Joint fit of two spin channels with a common Fermi level (spin_joint.py, spin_joint_fit.py).
The end-to-end run needs the bind of the fork github.com/ac-dias/yaehmop and the Fe example."""
from pathlib import Path
import json, subprocess, sys, tempfile, unittest
import numpy as np
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "parameter_fitter"))
from spin_joint import occupation, common_fermi
from band_objective import WindowSettings, metal_error
FE = BASE / "validation/vasp/Fe_bcc_pbe_spin"
FE_FIT = FE / "fitting/metal-cmaes-10k"


def features():
    try:
        from yaehmop_generic import resolve_yaehmop_assets, bind_features
        return bind_features(resolve_yaehmop_assets()[0])
    except Exception:
        return set()


class CommonFermi(unittest.TestCase):
    def test_occupations(self):
        for kind in ("mp1", "gaussian", "fermi"):
            self.assertAlmostEqual(float(occupation(0.0, 0.0, 0.1, kind)), 0.5, places=12)
            self.assertAlmostEqual(float(occupation(-5.0, 0.0, 0.1, kind)), 1.0, places=9)
            self.assertAlmostEqual(float(occupation(5.0, 0.0, 0.1, kind)), 0.0, places=9)

    def test_rigid_exchange_splitting(self):
        """Two identical flat-ish bands split by 2*delta: the common E_F fills N electrons and the moment
        follows from the splitting, independent of the absolute energy scale."""
        rng = np.random.default_rng(1)
        band = np.sort(rng.uniform(-3, 3, size=(400, 5)), axis=1)
        delta = 0.4
        for offset in (0.0, -12.0):
            mu, nu, nd, m = common_fermi(band - delta + offset, band + delta + offset, 4.0, 0.05, "gaussian")
            self.assertAlmostEqual(nu + nd, 4.0, places=8)
            self.assertGreater(m, 0.0)
            if offset == 0.0:
                ref = (mu, m)
            else:
                self.assertAlmostEqual(mu - offset, ref[0], places=6)
                self.assertAlmostEqual(m, ref[1], places=8)

    def test_metal_error_fixed_shift(self):
        """metal_error with shift = the optimal constant gives the free result; another shift is worse."""
        x = np.linspace(0, 1, 30)
        ref = np.column_stack([np.cos(3 * x) - 1.0, np.sin(2 * x)])
        eht = ref - 7.3
        s = WindowSettings.from_dict({"window_min": -3, "window_max": 3})
        free = metal_error(ref, eht, x, [0.0, 1.0], s)
        self.assertAlmostEqual(free["shift_eV"], 7.3, places=10)
        fixed = metal_error(ref, eht, x, [0.0, 1.0], s, shift=7.3)
        self.assertAlmostEqual(fixed["total"], free["total"], places=10)
        off = metal_error(ref, eht, x, [0.0, 1.0], s, shift=7.0)
        self.assertGreater(off["total"], free["total"] + 0.1)
        self.assertAlmostEqual(off["eht_fermi_eV"], -7.0, places=12)


class JointFit(unittest.TestCase):
    @unittest.skipUnless({"spin", "common_fermi"} <= features() and FE_FIT.is_dir(),
                         "bind without --parm-up/--common-fermi or Fe example missing")
    def test_fe_joint_fit_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, str(BASE / "USER_WORKFLOW/spin_joint_fit.py"), "--dft-dir", str(FE),
                                "--dft-code", "vasp", "--fermi", "5.86385846",
                                "--start-up", str(FE_FIT / "spin_up"), "--start-down", str(FE_FIT / "spin_down"),
                                "--optimizer", "nelder-mead", "--budget", "12", "--mesh-radius", "3",
                                "--outdir", tmp], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            st = json.loads((Path(tmp) / "fit/fit.json").read_text())
            j = st["joint"]
            self.assertAlmostEqual(j["electrons_up"] + j["electrons_down"], 8.0, places=6)
            self.assertLessEqual(st["best_cost"], st["start_cost"] + 1e-12)
            self.assertGreater(j["moment"], 1.5)
            # both channel folders use the same energy zero (the common E_F of the model)
            zs = [json.loads((Path(tmp) / f"fitting/spin_{ch}/fit_summary.json").read_text())
                  ["tight_binding_files"]["eht_midgap_eV"] for ch in ("up", "down")]
            self.assertAlmostEqual(zs[0], zs[1], places=10)
            self.assertAlmostEqual(zs[0], j["eht_fermi_eV"], places=10)
            sp = json.loads((Path(tmp) / "fitting/spin_polarized/fit_summary.json").read_text())
            self.assertTrue((Path(tmp) / "fitting/spin_polarized/huckel-SP.dat").is_file())
            self.assertEqual(sp["shift_up_eV"], 0.0)
            self.assertEqual(sp["shift_down_eV"], 0.0)


if __name__ == "__main__":
    unittest.main()
