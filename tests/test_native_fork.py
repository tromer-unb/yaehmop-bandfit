"""soc_fit.py and spin_combine.py with the native SOC / spin polarization of the bind of the
fork github.com/ac-dias/yaehmop (skipped with the original bind or without the validation data)."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"parameter_fitter"))
SOC=BASE/"validation/aims/MoS2_monolayer_hse06_soc";SOC_FIT=BASE/"validation/aims/MoS2_monolayer_hse06_soc/fitting/window-ga"
FE=BASE/"validation/vasp/Fe_bcc_pbe_spin/fitting/metal-cmaes-10k"

def features():
    try:
        from yaehmop_generic import resolve_yaehmop_assets,bind_features
        return bind_features(resolve_yaehmop_assets()[0])
    except Exception:
        return set()

class NativeFork(unittest.TestCase):
    @unittest.skipUnless("soc" in features() and SOC_FIT.is_dir(),"bind without --soc or MoS2 SOC example missing")
    def test_soc_fit_matches_native_bind(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=subprocess.run([sys.executable,str(BASE/"USER_WORKFLOW/soc_fit.py"),"--dft-dir",str(SOC),"--dft-code","aims",
                              "--from-fitting",str(SOC_FIT),"--outdir",tmp,"--maxiter","4"],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout[-2000:]+r.stderr[-2000:])
            s=json.loads((Path(tmp)/"fitting-soc/fit_summary.json").read_text())
            self.assertLess(s["native_soc"]["max_difference_eV"],1e-5)
            self.assertTrue((Path(tmp)/"fitting-soc/yaehmop_soc/soc.dat").is_file())
            self.assertAlmostEqual(s["lambdas_eV"]["Mo-d"],0.063,delta=0.01)

    @unittest.skipUnless("spin" in features() and FE.is_dir(),"bind without --parm-up or Fe example missing")
    def test_spin_combine_native_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=subprocess.run([sys.executable,str(BASE/"USER_WORKFLOW/spin_combine.py"),"--up",str(FE/"spin_up"),
                              "--down",str(FE/"spin_down"),"--outdir",tmp,"--mesh-radius","2"],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout[-2000:]+r.stderr[-2000:])
            ns=json.loads((Path(tmp)/"fit_summary.json").read_text())["native_spin"]
            for ch in ("up","down"):
                self.assertLess(ns["bands_vs_huckel_SP"][ch]["max_deviation_eV"],2e-3)
            self.assertAlmostEqual(ns["electrons_up"]+ns["electrons_down"],8.0,places=3)
            self.assertGreater(ns["huckel_SP_electrons_below_EF"]["moment"],1.0)   # bcc Fe is ferromagnetic
            self.assertTrue((Path(tmp)/"yaehmop_spin/spin.bind").is_file())

if __name__=="__main__":unittest.main()
