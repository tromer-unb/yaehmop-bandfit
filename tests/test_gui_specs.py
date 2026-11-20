"""band_fit_gui.py reads the options of every command from its argparse parser (no Qt needed)."""
import json, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GuiSpecs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r = subprocess.run([sys.executable, str(ROOT / "band_fit_gui.py"), "--dump-specs"], capture_output=True,
                           text=True, cwd=ROOT, env={"MPLBACKEND": "Agg", "PATH": ""}, timeout=120)
        cls.specs = json.loads(r.stdout.strip().splitlines()[-1])

    def test_every_tool_has_its_options(self):
        sys.path.insert(0, str(ROOT))
        from band_fit_gui import TOOLS
        for key in TOOLS:
            self.assertIn(key, self.specs)
            self.assertNotIn("error", self.specs[key], key)
            self.assertTrue(self.specs[key]["actions"], key)

    def test_bands_fit_options(self):
        acts = {a["dest"]: a for a in self.specs["bands_fit"]["actions"]}
        for d in ("cif", "input_file", "dft_dir", "spin_channel", "optimizer", "basis_add", "fitting_dir"):
            self.assertIn(d, acts)
        self.assertIn("metal", acts["objective"]["choices"])
        self.assertEqual(acts["prepare_only"]["kind"], "flag")

    def test_objective_settings(self):
        names = {o["name"] for o in self.specs["_objective"]}
        self.assertTrue({"window_min", "window_max", "band_weights", "w_frontier"} <= names)


if __name__ == "__main__":
    unittest.main()
