"""Quick repository release checks: no YAeHMOP, DFT, fitting or network."""
from pathlib import Path
import csv,json,subprocess,sys,tempfile,unittest
import numpy as np
from PIL import Image
from ase.io import read
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"USER_WORKFLOW"),str(BASE/"parameter_fitter"),str(BASE/"source"),str(BASE/"scripts")]
from prepare_band_png import prepare,csv_curves
from raster_ingest import normalize_raster
from yaehmop_generic import BandPath,write_periodic_input_generic
from plot_saved_bands import visualize
class PublishedInputTests(unittest.TestCase):
 def test_two_and_four_band_synthetic_numerical_curves(self):
  for n in (2,4):
   with self.subTest(n=n):
    src=BASE/"examples/synthetic"/f"toy_{n}bands.csv"
    x,arr,meta=csv_curves(src,n,-3.5,3.5)
    self.assertEqual(arr.shape,(151,n))
    self.assertAlmostEqual(float(x[0]),0)
    self.assertAlmostEqual(float(x[-1]),1)
    self.assertTrue(np.all(np.diff(arr,axis=1)>0))
    self.assertFalse(meta["estimated"])
 def test_csv_to_bitplanes_without_solver(self):
  with tempfile.TemporaryDirectory() as d:
   out,qa=prepare(BASE/"examples/synthetic/toy_4bands.csv",
      Path(d),2,-3.5,3.5,mode="csv",nx=151)
   self.assertTrue(out.is_file())
   with Image.open(out) as image:
    self.assertEqual(image.mode,"L")
   self.assertEqual(qa["n_bands"],4)
   self.assertLess(qa["encode_roundtrip_rmse_eV"],0.015)
   self.assertTrue((Path(d)/"extraction_preview.png").exists())
 def test_visible_png_traced_into_bitplanes(self):
  with tempfile.TemporaryDirectory() as d:
   out,qa=prepare(BASE/"examples/synthetic/toy_2bands.png",
      Path(d),1,-3.5,3.5,mode="colors",
      bbox=(115,105,1210,820),colors="#c82d37,#1c7ad5",nx=151)
   self.assertTrue(out.is_file())
   self.assertEqual(qa["n_bands"],2)
   self.assertGreater(qa["gap_eV"],1.0)
   self.assertEqual(qa["raster_ingest"]["canonical_format"],"PNG")
 def test_content_detected_not_extension(self):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/"misleading.jpg"
   src.write_bytes((BASE/"examples/synthetic/toy_2bands.png").read_bytes())
   path,qa=normalize_raster(src,Path(d)/"canonical.png",n_bands=2)
   self.assertEqual(qa["detected_format"],"PNG")
   with Image.open(path) as image:
    self.assertEqual(image.format,"PNG")
 def test_3d_input_is_true_three_dimensional(self):
  for material,dim in (("graphene",2),("diamond",3),("graphite_AB",3)):
   with self.subTest(material=material):
    cif=BASE/"examples/structures"/f"{material}.cif"
    atoms=read(cif)
    v=np.zeros((2,3));v[1,0]=0.5
    path=BandPath(("G","X"),v,3)
    with tempfile.TemporaryDirectory() as d:
     out=Path(d)/"test.bind"
     images=(4,4,4) if dim==3 else (4,4,0)
     write_periodic_input_generic(cif,out,path,
        electrons=4*len(atoms),overlap_images=images,periodic_dim=dim)
     txt=out.read_text()
     self.assertIn("Lattice\n"+str(dim)+"\n"+" ".join(map(str,images)),txt)
     lines=txt.splitlines()
     i=lines.index("Geometry Crystallographic")
     self.assertEqual(int(lines[i+1]),len(atoms)+dim)
     i=lines.index("Lattice")
     self.assertEqual(lines[i+3:i+3+dim],
                      [f"1 {len(atoms)+1+j}" for j in range(dim)])
 def test_wrong_3d_image_radius_rejected(self):
  cif=BASE/"examples/structures/diamond.cif"
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,"inactive|Inactive"):
    write_periodic_input_generic(cif,Path(d)/"invalid.bind",
      BandPath(("G","X"),np.zeros((2,3)),3),
      overlap_images=(4,4,4),periodic_dim=2)
 def test_saved_physical_eht_band_csv_and_metadata(self):
  expectations={"graphene":(37,8,4),"diamond":(25,8,4),
                "graphite_AB":(49,16,8)}
  for name,(rows,orbs,filled) in expectations.items():
   p=BASE/"examples/precomputed"/name
   j=json.loads((p/"eht_full_bands_metadata.json").read_text())
   with (p/"eht_full_bands.csv").open(newline="") as f:
    d=list(csv.DictReader(f))
   self.assertEqual(len(d),rows)
   self.assertEqual(j["nocc_spatial"],filled)
   self.assertEqual(j["orbitals"],orbs)
   self.assertEqual(len([k for k in d[0] if k.startswith("orbital_")]),orbs)
   self.assertTrue((p/"frontier.png").is_file())
 def test_saved_band_plotter_offline(self):
  with tempfile.TemporaryDirectory() as d:
   src=BASE/"examples/precomputed/graphene/eht_full_bands.csv"
   dest=Path(d)/"individual_orbitals.pdf"
   visualize(src,dest,orbitals="4,5")
   self.assertGreater(dest.stat().st_size,2000)
 def test_config_validation_does_not_require_solver(self):
  with tempfile.TemporaryDirectory() as d:
   cmd=[sys.executable,str(BASE/"USER_WORKFLOW/bands_fit.py"),
        str(BASE/"examples/structures/graphene.cif"),
        str(BASE/"examples/configs/synthetic_2bands.ini"),
        "--validate-only","--outdir",str(Path(d)/"out")]
   p=subprocess.run(cmd,capture_output=True,text=True,check=True)
   j=json.loads(p.stdout)
   self.assertEqual(j["band_count"],2)
   self.assertIsNone(j["raster_ingest"])
 def test_docs_and_license(self):
  for f in ("LICENSE","README.md","CONTRIBUTING.md","SECURITY.md",
             "CITATION.md","docs/IMAGE_INPUT.md","docs/CLI_REFERENCE.md",
             "docs/NUMERIC_BANDS.md","docs/SCIENTIFIC_SCOPE.md"):
   self.assertTrue((BASE/f).is_file(),f)
if __name__=="__main__":unittest.main()
