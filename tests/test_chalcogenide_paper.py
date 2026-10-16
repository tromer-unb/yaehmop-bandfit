"""P5 six-material matched-window archive checks. Lightweight, no YAeHMOP."""
from pathlib import Path
import csv,json,subprocess,sys,tempfile,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"examples/chalcogenides"
MAT=("GeS","GeSe","GeTe","SnS2","SnSe","SnTe")
N=(2,4,6,8)
class P5PaperArchiveChecks(unittest.TestCase):
 def test_all_six_compounds_and_nband_eligibility(self):
  rows=list(csv.DictReader((DATA/"direct_and_transfer.csv").open(newline="")))
  self.assertEqual(len(rows),24)
  got={(r["material"],int(r["bands"])):r for r in rows}
  self.assertEqual(set(got),{(m,n) for m in MAT for n in N})
  for (m,n),r in got.items():
   path=ROOT/r["saved_EHT_only"] if r["saved_EHT_only"] else None
   if (m,n)==("GeTe",8):
    self.assertEqual(r["direct_rmse_eV"],"")
    self.assertEqual(r["saved_EHT_only"],"")
   else:
    self.assertTrue(path.is_file())
    with path.open(newline="") as f:
     d=list(csv.DictReader(f))
    self.assertGreaterEqual(len(d),60)
    self.assertEqual(set(d[0]),{"x"}|{f"EHT_band_{j}_eV" for j in range(1,n+1)})
    x=np.asarray([float(v["x"]) for v in d])
    self.assertAlmostEqual(x[0],0)
    self.assertAlmostEqual(x[-1],1)
    self.assertTrue(np.all(np.diff(x)>0))
 def test_exact_p5_summary_and_benchmark_task_boundaries(self):
  doc=json.loads((DATA/"benchmark_summary.json").read_text())
  expected=[(2,.22927770599,.80693477473,6,6),
            (4,.24818507803,.90882508664,6,6),
            (6,.27325835333,.70382406458,6,6),
            (8,.32058175428,3.41591345091,5,4)]
  for n,di,zo,nd,nz in expected:
   a=doc["summary"]["direct"][str(n)]
   b=doc["summary"]["zero-shot shared"][str(n)]
   self.assertAlmostEqual(a["mean_rmse_eV"],di,places=8)
   self.assertAlmostEqual(b["mean_rmse_eV"],zo,places=8)
   self.assertEqual(a["n_materials"],nd)
   self.assertEqual(b["n_materials"],nz)
  self.assertIn("different",doc["direct_vs_holdout"].lower()) if "different" in doc["direct_vs_holdout"].lower() else self.assertIn("excluded",doc["direct_vs_holdout"].lower())
 def test_SnS2_paper_fig4_and_best_fit(self):
  self.assertGreater((ROOT/"assets/paper_fig04_SnS2.png").stat().st_size,100000)
  rows=list(csv.DictReader((DATA/"direct_and_transfer.csv").open(newline="")))
  got={(r["material"],int(r["bands"])):r for r in rows}
  for n,val in [(2,.10425838854498277),(4,.27250241742564973),
                (6,.25606821650922046),(8,.3332556396740769)]:
   self.assertAlmostEqual(float(got["SnS2",n]["direct_rmse_eV"]),val,places=8)
  winners={n:min((float(r["direct_rmse_eV"]),r["material"])
                for r in rows if int(r["bands"])==n and r["direct_rmse_eV"])
           for n in N}
  self.assertEqual(winners[2][1],"SnS2")
  self.assertEqual([winners[n][1] for n in (4,6,8)],["SnTe"]*3)
 def test_P5_short_commands_and_cli_help(self):
  x=subprocess.run([sys.executable,str(ROOT/"quick.py"),"--help"],
                  capture_output=True,text=True,check=True)
  for item in ("sns2","materials","chemical-transfer","GeS","SnTe","graphene"):
   self.assertIn(item,x.stdout)
  with tempfile.TemporaryDirectory() as tmp:
   o=Path(tmp)
   for name,expect in [("sns2","SnS2_paper_Fig4_PBE_vs_EHT.png"),
                        ("materials","six_materials_direct.png"),
                        ("chemical-transfer","six_materials_zero_shot.png"),
                        ("SnTe","SnTe_2_4_6_8_eht.png")]:
    p=subprocess.run([sys.executable,str(ROOT/"quick.py"),name,
                      "--outdir",str(o/name)],
                     capture_output=True,text=True,check=True,timeout=30)
    self.assertGreater((o/name/expect).stat().st_size,1000)
 def test_scientific_source_and_no_expensive_CI(self):
  readme=(ROOT/"README.md").read_text()
  pos_sns2=readme.index("Featured example: SnS₂")
  pos_carbon=readme.index("Later extension: carbon")
  self.assertLess(pos_sns2,pos_carbon)
  self.assertIn("23 individual",readme)
  self.assertIn("not redistributed",readme.lower())
  ci=(ROOT/".github/workflows/tests.yml").read_text()
  self.assertNotIn("quick.py all",ci)
  self.assertNotIn("recompute_carbon",ci)
  self.assertNotIn("pyscf",ci.lower().split("# deliberately")[0])
if __name__=="__main__":unittest.main()
