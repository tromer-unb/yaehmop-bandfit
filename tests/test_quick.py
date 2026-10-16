"""Instant best-results showcase: archived EHT only, no solver/DFT calls."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
import numpy as np

R=Path(__file__).resolve().parents[1]
class QuickShowcaseTests(unittest.TestCase):
    def test_numbers_and_sources(self):
        m=json.loads((R/"examples/best_results/metrics.json").read_text())
        self.assertAlmostEqual(m["graphene_c2db"]["best_rmse_eV"],.3223099313513552,places=10)
        self.assertAlmostEqual(m["graphene_c2db"]["comparison_rmse_eV"],.5822773337396134,places=10)
        self.assertAlmostEqual(m["graphene_pyscf"]["system_specific_sp5_rmse_eV"],.16569896052970817,places=10)
        self.assertAlmostEqual(m["graphene_pyscf"]["shared_sp5_rmse_eV"],.16831147428456159,places=10)
        self.assertAlmostEqual(m["family_pyscf"]["heldout_phagraphene_shared_sp5_rmse_eV"],.15643801168366284,places=10)
        self.assertNotEqual(m["graphene_c2db"]["target"],m["graphene_pyscf"]["target"])
        self.assertIn("Gamma-only",m["family_pyscf"]["reference"])
        self.assertIn("cannot be pooled",m["source_notice"])
    def test_archived_prediction_csv(self):
        for name,count in [
          ("graphene_c2db_pdz7",361),
          ("graphene_pyscf_sp5_shared",37),
          ("phagraphene_pyscf_sp5_holdout",9),
          ("phagraphene_pyscf_pdz7_nested_holdout",9),
        ]:
            import csv
            with (R/"examples/best_results"/(name+".csv")).open() as f:
                reader=csv.DictReader(f)
                self.assertEqual(reader.fieldnames,["x","EHT_valence_eV","EHT_conduction_eV"])
                arr=np.asarray([[float(t) for t in row.values()] for row in reader])
            self.assertEqual(arr.shape,(count,3))
            self.assertTrue(np.isfinite(arr).all())
            self.assertAlmostEqual(arr[0,0],0)
            self.assertAlmostEqual(arr[-1,0],1)
            self.assertTrue(np.all(arr[:,1]<=arr[:,2]))
    def test_one_command_without_solver(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/"demo"
            result=subprocess.run([sys.executable,str(R/"quick.py"),"--outdir",str(dest)],
                  cwd=R,capture_output=True,text=True,timeout=35,check=True)
            for p in [
              "overall_2_4_6_8.png","six_materials_direct.png",
              "SnS2_paper_Fig4_PBE_vs_EHT.png","SnS2_2_4_6_8_eht.png",
              "six_materials_zero_shot.png","carbon_best_results.png",
              "graphene_best_pdz7.png",
              "transfer_graphene_phagraphene.png",
              "carbon_graphene_diamond_graphite.png",
              "image_input/bands_2_layers.png","image_input/extraction_qa.json",
            ]:
                self.assertGreater((dest/p).stat().st_size,1000,p)
            self.assertIn("SnS₂ 1V+1C: 0.1043 eV",result.stdout)
            self.assertIn("Best 4-band direct material: SnTe",result.stdout)
            self.assertIn("0.3223",result.stdout)
            self.assertIn("0.1657",result.stdout)
            self.assertIn("0.1564",result.stdout)
            self.assertIn("NO DFT / NO FIT / NO SOLVER",result.stdout)
    def test_first_screen_readme_has_quick(self):
        readme=(R/"README.md").read_text()
        prefix=readme[:1800]
        self.assertIn("python3 quick.py",prefix)
        self.assertIn("paper Fig. 4",prefix)
        self.assertIn("docs/BEST_RESULTS.md",prefix)
        self.assertIn("different",readme.lower())
if __name__=="__main__":unittest.main()
