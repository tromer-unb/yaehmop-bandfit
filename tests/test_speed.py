"""Speed features of fit_sparse_window with the real YAeHMOP bind: H(k) = F(Hii,K) o S(k) from a
cached S(k) (numpy) against bind, automatic overlap radius, parallel candidates = sequential ones.

Skipped when bind or the validation example is missing."""
from pathlib import Path
import os,shutil,sys,tempfile,unittest
import numpy as np,yaml
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"parameter_fitter"));sys.path.insert(0,str(BASE/"USER_WORKFLOW"))
EX=BASE/"validation/qe/SiC_zincblende_pbe"

def bind_ok():
    try:
        from yaehmop_generic import resolve_yaehmop_assets
        resolve_yaehmop_assets();return True
    except Exception:
        return False

@unittest.skipUnless(bind_ok() and EX.is_dir(),"YAeHMOP bind or validation/qe/SiC_zincblende_pbe missing")
class SpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bands_fit import load_dft_reference
        cls.tmp=Path(tempfile.mkdtemp())
        L=load_dft_reference(EX,"qe",cls.tmp/"ref",name="SiC",step=8,pairs=1)
        cls.entry=dict(L["entry"])
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp,ignore_errors=True)
    def fitter(self,name,**cfg):
        from fit_sparse_window import Fitter
        c={"n_pairs":1,"points_per_line":8,"objective":"window","overlap_min_eig":1e-3,
           "systems":[dict(self.entry)],**cfg}
        return Fitter(c,self.tmp,self.tmp/name)
    def test_numpy_hamiltonian_matches_bind(self):
        from fit_sparse_window import apply_x
        f=self.fitter("fast",overlap_radius=3)
        self.assertTrue(f.prepare_fast(f.stock))
        self.assertLess(f.fast_check_error,2e-3)
        x=f.stock.copy()
        for i,s in enumerate(f.specs):
            if s[2]=="Hii":x[i]+=.6*(-1)**i
            if s[2]=="K":x[i]=2.1
        p,K=apply_x(f.stockparams,f.specs,x)
        s=f.train[0]
        fast=f.run(s,p,K,"t");exact=f.run(s,p,K,"t",exact=True)
        self.assertLess(float(np.abs(fast-exact).max()),2e-3)
        self.assertGreater(f.fast_evals,0)
        # other exponents: not on the cache, bind is used
        x[[i for i,s in enumerate(f.specs) if s[2]=="zeta1"][0]]*=.95
        p2,_=apply_x(f.stockparams,f.specs,x)
        n=f.fast_evals;f.run(s,p2,K,"u");self.assertEqual(f.fast_evals,n)
    def test_auto_radius(self):
        f=self.fitter("auto",overlap_radius="auto",overlap_radius_tol=1e-3)
        rep=f.radius_report["SiC"]
        self.assertLess(rep["radius"],5)
        self.assertLess(rep["errors_eV"][str(rep["radius"])],1e-3)
        self.assertEqual(max(f.train[0]["overlap_images"]),rep["radius"])
    def test_parallel_candidates_equal_sequential(self):
        f=self.fitter("par",overlap_radius=3,parallel=4)
        rng=np.random.default_rng(3)
        xs=[f.stock+rng.normal(scale=.02,size=len(f.stock)) for _ in range(5)]
        seq=[f.evaluate(x,100+i)[0] for i,x in enumerate(xs)]
        par=[v for v,_ in f.objective_many(xs,None,"t")]
        self.assertEqual(seq,par)
        f.prepare_fast(f.stock)
        xf=[f.stock.copy() for _ in range(4)]
        for i,x in enumerate(xf):x[-1]=1.6+.1*i
        seqf=[f.evaluate(x,200+i)[0] for i,x in enumerate(xf)]
        parf=[v for v,_ in f.objective_many(xf,None,"t")]   # forked processes on the cache
        f.close_fast_pool()
        np.testing.assert_allclose(seqf,parf,rtol=0,atol=1e-12)

if __name__=="__main__":unittest.main()
