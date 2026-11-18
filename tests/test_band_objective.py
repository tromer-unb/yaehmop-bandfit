"""Band-window objective and basis extension. No YAeHMOP."""
from pathlib import Path
import sys,unittest
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"parameter_fitter")]
from band_objective import WindowSettings,window_error,extend_basis,parse_basis_add

def bands(nk=41,shift=0.0,gap=1.0,extra_cb=0):
    x=np.linspace(0,1,nk);c=np.cos(2*np.pi*x)
    vb=[-6+.3*c,-4+.5*c,-1.0+.4*c-.4]          # VBM at x=0 (c=1): -1
    cb=[gap-1.0+.4-.4*c+.4,3+.3*c]             # CBM at x=0
    cb+= [5+.1*c]*extra_cb
    return x,np.column_stack(vb+cb)+shift

class WindowObjectiveTests(unittest.TestCase):
    def test_identical_bands_give_zero(self):
        x,A=bands()
        m=window_error(A,3,A+7.0,3,x,[0,.5,1],WindowSettings())   # rigid shift: aligned away
        self.assertAlmostEqual(m["total"],0.0,places=12)
        self.assertEqual((m["n_valence_compared"],m["n_conduction_compared"]),(3,2))

    def test_band_count_and_terms(self):
        x,A=bands(extra_cb=3)
        _,B=bands(gap=1.5)
        B=np.hstack([np.full((len(x),2),-30.0),B])      # two extra deep EHT bands
        m=window_error(A,3,B,5,x,[0,.5,1],WindowSettings(n_cond=6))
        self.assertEqual(m["n_conduction_compared"],2)  # limited by the EHT conduction bands
        self.assertAlmostEqual(m["gap_error_eV"],0.5,places=10)
        self.assertGreater(m["term_gap_indirect"],0)
        self.assertAlmostEqual(m["term_valence"],0.0,places=12)

    def test_weights_change_total(self):
        x,A=bands();_,B=bands(gap=1.3)
        a=window_error(A,3,B,3,x,[0,1],WindowSettings())["total"]
        b=window_error(A,3,B,3,x,[0,1],WindowSettings(w_gap_indirect=50))["total"]
        self.assertGreater(b,a)
        with self.assertRaises(ValueError):WindowSettings.from_dict({"w_nonsense":1})

class BasisTests(unittest.TestCase):
    P={"S":{"atnum":16,"nval":6,"s":{"nzeta":1,"n":3,"Hii":-20,"zeta1":2.1,"zeta2":0,"c1":1,"c2":0},
                                    "p":{"nzeta":1,"n":3,"Hii":-11,"zeta1":1.8,"zeta2":0,"c1":1,"c2":0}},
       "H":{"atnum":1,"nval":1,"s":{"nzeta":1,"n":1,"Hii":-13.6,"zeta1":1.3,"zeta2":0,"c1":1,"c2":0}},
       "Zn":{"atnum":30,"nval":12,"s":{"nzeta":1,"n":4,"Hii":-12.4,"zeta1":2.0,"zeta2":0,"c1":1,"c2":0},
                                      "p":{"nzeta":1,"n":4,"Hii":-6.5,"zeta1":1.7,"zeta2":0,"c1":1,"c2":0}}}
    def test_parse(self):
        self.assertEqual(parse_basis_add("S:d, H:p zn:d"),{"S":["d"],"H":["p"],"Zn":["d"]})
        self.assertEqual(parse_basis_add("auto"),"auto")
        with self.assertRaises(ValueError):parse_basis_add("S")
    def test_extend(self):
        p=extend_basis(self.P,"S:d H:p")
        self.assertEqual(p["S"]["d"]["n"],3);self.assertGreater(p["S"]["d"]["Hii"],p["S"]["p"]["Hii"])
        self.assertEqual(p["H"]["p"]["n"],2)
        self.assertNotIn("d",self.P["S"])                 # input untouched
    def test_auto(self):
        p=extend_basis(self.P,"auto")
        self.assertIn("d",p["S"]);self.assertIn("p",p["H"]);self.assertNotIn("d",p["H"])
        self.assertEqual(p["Zn"]["d"]["nzeta"],2)          # double-zeta 3d of Zn

if __name__=="__main__":unittest.main()
