"""Optimizers ported from dftb_sk_gen: strict budget, bounds, progress.

Analytic objectives only; no YAeHMOP, DFT or network.
"""
from pathlib import Path
import importlib.util,sys,unittest
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"parameter_fitter"))
from global_optimizers import GLOBAL_METHODS,POLISH_METHODS,run_global,polish

HAS_SK=importlib.util.find_spec("sklearn") is not None
HAS_CMA=importlib.util.find_spec("cma") is not None
BOUNDS=[(-3.,5.),(0.5,4.),(-10.,-2.)]
TARGET=np.array([1.2,2.7,-6.1])
X0=np.array([4.,1.,-3.])

def make_objective(fail_every=None):
    calls=[]
    def f(x,eid,stage):
        calls.append((np.asarray(x,float).copy(),eid,stage))
        if fail_every and eid%fail_every==0:
            raise RuntimeError("bind failed")
        z=(np.asarray(x)-TARGET)/np.array([8.,3.5,8.])
        return float(np.sum(z**2)+.1),{"rmse":float(np.sqrt(np.mean(z**2)))}
    return f,calls

def available(method):
    return not ((method=="bayesian" and not HAS_SK) or (method=="cmaes" and not HAS_CMA))

class GlobalOptimizerTests(unittest.TestCase):
    def check(self,method,budget,polish_method=None):
        f,calls=make_objective()
        start=f(X0,0,"x0")[0]; calls.clear()
        r=run_global(method,f,X0,BOUNDS,budget,seed=7,polish_method=polish_method)
        self.assertLessEqual(len(calls),budget)
        self.assertEqual(r["evaluations"],len(calls))
        self.assertEqual(len(r["history"]),len(calls))
        self.assertEqual([c[1] for c in calls],list(range(1,len(calls)+1)))  # unique call ids
        lo=np.array([a for a,_ in BOUNDS]); hi=np.array([b for _,b in BOUNDS])
        for x,_,_ in calls:
            self.assertTrue(np.all(x>=lo-1e-12) and np.all(x<=hi+1e-12))
        self.assertLess(r["best_loss"],start)
        self.assertAlmostEqual(r["best_loss"],min(h["loss"] for h in r["history"]))
        self.assertTrue(all("x" not in h for h in r["history"]))
        return r

    def test_every_method_respects_budget_and_improves(self):
        for method in GLOBAL_METHODS:
            if not available(method):continue
            with self.subTest(method=method):
                budget=40 if method=="bayesian" else 90
                r=self.check(method,budget)
                self.assertLess(r["best_loss"],.35)

    def test_polish_inside_the_same_budget(self):
        for method in ("differential-evolution","genetic"):
            for pm in POLISH_METHODS:
                with self.subTest(method=method,polish=pm):
                    r=self.check(method,80,pm)
                    self.assertIn("polish",r)
                    self.assertEqual(r["optimizer"],f"{method}+{pm}")
                    self.assertTrue(any(h["stage"].startswith("polish-") for h in r["history"]))

    def test_failed_physical_calls_do_not_crash(self):
        f,calls=make_objective(fail_every=3)
        r=run_global("genetic",f,X0,BOUNDS,60,seed=3)
        self.assertEqual(len(calls),60)
        self.assertTrue(any("error" in h for h in r["history"]))
        self.assertLess(r["best_loss"],1e4)

    def test_x0_is_evaluated_first(self):
        for method in ("differential-evolution","genetic","bayesian","cmaes","powell"):
            if not available(method):continue
            with self.subTest(method=method):
                f,calls=make_objective()
                run_global(method,f,X0,BOUNDS,30,seed=1)
                first=[c[0] for c in calls[:max(30,1)]]
                self.assertTrue(any(np.allclose(x,X0) for x in first))

    def test_invalid_names_rejected(self):
        f,_=make_objective()
        with self.assertRaises(ValueError):run_global("random-walk",f,X0,BOUNDS,10)
        with self.assertRaises(ValueError):run_global("genetic",f,X0,BOUNDS,10,polish_method="bfgs")
        with self.assertRaises(ValueError):polish(f,[],BOUNDS,10,"bfgs")

if __name__=="__main__":unittest.main()
