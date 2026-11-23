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

    def test_speed_and_family_options(self):
        acts = {a["dest"]: a for a in self.specs["bands_fit"]["actions"]}
        for d in ("parallel", "fast_budget", "basis_dz", "semicore", "start_from"):
            self.assertIn(d, acts)
        self.assertIn("alternating", acts["optimizer"]["choices"])
        fam = {a["dest"]: a for a in self.specs["family_fit"]["actions"]}
        for d in ("evaluate", "parallel", "fast_budget", "basis_dz", "start_from", "workers"):
            self.assertIn(d, fam)
        self.assertIn("segment_weights", {o["name"] for o in self.specs["_objective"]})

    def test_ini_schema_has_the_new_keys(self):
        sys.path.insert(0, str(ROOT))
        from band_fit_gui import INI_SCHEMA, DFT_REF_KEYS, SUGGEST
        fit = {k for k, _, _ in INI_SCHEMA["fit"]}
        self.assertTrue({"basis_dz", "parallel", "fast_budget", "alternating_rounds", "overlap_radius_tol"} <= fit)
        ref = dict((k, t) for k, t, _ in INI_SCHEMA["reference"])
        self.assertIn("average", ref["spin_channel"])
        self.assertIn("semicore", DFT_REF_KEYS)
        self.assertIn("auto", SUGGEST["overlap_radius"])
        self.assertIn("segment_weights", SUGGEST)


class FamilyManifestEditor(unittest.TestCase):
    """The Family fit tab writes manifests that family_fit.py accepts, and reads them back."""

    def test_round_trip_and_family_fit_accepts_it(self):
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "USER_WORKFLOW"))
        sys.path.insert(0, str(ROOT / "parameter_fitter"))
        import tempfile, yaml
        from band_fit_gui import family_manifest, family_rows, FAMILY_COLS
        d1, d2 = ROOT / "validation/qe/Si_diamond_lda", ROOT / "validation/qe/Si_wurtzite_lda"
        if not (d1.is_dir() and d2.is_dir()):
            self.skipTest("validation examples missing")
        rows = [dict.fromkeys(FAMILY_COLS, "") for _ in range(3)]
        rows[0].update(name="Si_d", role="train", dft_dir=str(d1), dft_code="qe", weight="2",
                       objective_settings="segment_weights: '0:3'", overlap_radius="auto")
        rows[1].update(name="Si_w", role="test", dft_dir=str(d2), dft_code="qe", metallic="false", semicore="0")
        m = family_manifest({"n_pairs": "1", "overlap_radius": "4", "objective_settings": "w_split: 100",
                             "fast_budget": "500"}, {"points_per_segment": "12"}, rows)
        self.assertEqual(len(m["systems"]), 2)                  # the empty row is dropped
        self.assertEqual(m["systems"][0]["objective_settings"], {"segment_weights": "0:3"})
        self.assertEqual(m["systems"][0]["overlap_radius"], "auto")
        self.assertEqual(m["objective_settings"], {"w_split": 100})
        self.assertEqual(m["defaults"], {"points_per_segment": 12})
        self.assertIs(m["systems"][1]["metallic"], False)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "fam.yaml"
            f.write_text(yaml.safe_dump(m))
            import family_fit
            _, _, train, test = family_fit.load_manifest(f)
            self.assertEqual([s["name"] for s in train], ["Si_d"])
            self.assertEqual([s["name"] for s in test], ["Si_w"])
            self.assertEqual(train[0]["overlap_radius"], "auto")
            glob, defs, back = family_rows(yaml.safe_load(f.read_text()), tmp)
        self.assertEqual(glob["overlap_radius"], "4")
        self.assertEqual(defs["points_per_segment"], "12")
        self.assertEqual(back[0]["dft_dir"], str(d1))
        self.assertEqual(family_manifest(glob, defs, back), m)
        with self.assertRaises(ValueError):
            family_manifest({}, {}, [dict(rows[0], name="")])
        with self.assertRaises(ValueError):
            family_manifest({}, {}, [dict.fromkeys(FAMILY_COLS, "") | {"name": "x"}])


if __name__ == "__main__":
    unittest.main()
