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
            if not available(method) or method=="alternating":continue  # needs a fast stage: own test
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


def make_many(f):
    """objective_many built from the sequential objective (as the Fitter does in threads)."""
    batches=[]
    def many(xs,eids,stage):
        batches.append(len(xs))
        out=[]
        for x,eid in zip(xs,eids):
            try:out.append(f(x,eid,stage))
            except Exception as e:out.append((1e4,{"error":repr(e)}))
        return out
    return many,batches


class ParallelPopulationTests(unittest.TestCase):
    def test_batches_reproduce_the_sequential_search(self):
        # CMA-ES and GA follow exactly the same path when a generation is evaluated at once
        for method in ("cmaes","genetic"):
            if not available(method):continue
            with self.subTest(method=method):
                f1,c1=make_objective();r1=run_global(method,f1,X0,BOUNDS,70,seed=5)
                f2,c2=make_objective();many,batches=make_many(f2)
                r2=run_global(method,f2,X0,BOUNDS,70,seed=5,objective_many=many,parallel=8)
                self.assertTrue(max(batches)>1)
                self.assertEqual(r1["best_loss"],r2["best_loss"])
                self.assertEqual([h["loss"] for h in r1["history"]],[h["loss"] for h in r2["history"]])
                self.assertEqual(r2["evaluations"],70)

    def test_parallel_de_and_bayesian_respect_the_budget(self):
        for method in ("differential-evolution","bayesian"):
            if not available(method):continue
            with self.subTest(method=method):
                f,calls=make_objective(fail_every=7);many,batches=make_many(f)
                budget=30 if method=="bayesian" else 60
                r=run_global(method,f,X0,BOUNDS,budget,seed=2,objective_many=many,parallel=4)
                self.assertLessEqual(len(calls),budget)
                self.assertEqual(r["evaluations"],len(calls))
                self.assertTrue(max(batches)>1)
                self.assertLess(r["best_loss"],1e4)


@unittest.skipUnless(HAS_CMA,"cma not installed")
class AlternatingTests(unittest.TestCase):
    def test_fast_stages_and_slow_budget(self):
        f,calls=make_objective()
        prepared=[]
        def prepare(x):
            prepared.append(np.asarray(x,float).copy());return True
        r=run_global("alternating",f,X0,BOUNDS,60,seed=4,
                     alternating=dict(fast_idx=[0,2],prepare=prepare,fast_budget=120,rounds=2))
        self.assertEqual(r["slow_evaluations"],60)
        self.assertEqual(r["fast_evaluations"],120)
        self.assertEqual(len(prepared),3)            # before each fast stage
        fast=[c for c,h in zip(calls,r["history"]) if h["stage"].startswith("fast")]
        self.assertEqual(len(fast),120)
        # a fast stage moves only the fast parameters (index 1 stays at the stage's start)
        stage1=[c[0] for c,h in zip(calls,r["history"]) if h["stage"].startswith("fast1:")]
        self.assertTrue(all(np.isclose(x[1],X0[1]) for x in stage1))
        self.assertLess(r["best_loss"],.2)
        self.assertAlmostEqual(r["best_loss"],min(h["loss"] for h in r["history"]))

    def test_needs_the_fast_settings(self):
        f,_=make_objective()
        with self.assertRaises(ValueError):run_global("alternating",f,X0,BOUNDS,30)


class ParallelPolishTests(unittest.TestCase):
    """The local searches of the polish starts run in threads: same result and history as one by one."""

    def history(self):
        rng=np.random.default_rng(3)
        rows=[]
        for _ in range(40):
            x=np.array([rng.uniform(a,b) for a,b in BOUNDS])
            z=(x-TARGET)/np.array([8.,3.5,8.])
            rows.append({"loss":float(np.sum(z**2)+.1),"x":list(x)})
        return rows

    def test_same_result_as_sequential(self):
        for method in POLISH_METHODS:
            with self.subTest(method=method):
                hist=self.history()
                f1,_=make_objective();f2,_=make_objective()
                seq=polish(f1,hist,BOUNDS,80,method,n_starts=4,prefix=10)
                par=polish(f2,hist,BOUNDS,80,method,n_starts=4,prefix=10,parallel=4)
                self.assertEqual(par["parallel"],4)
                self.assertEqual(seq["evaluations"],par["evaluations"])
                self.assertAlmostEqual(seq["best_loss"],par["best_loss"],places=12)
                self.assertTrue(np.allclose(seq["best_x"],par["best_x"]))
                self.assertEqual([h["eval"] for h in seq["history"]],[h["eval"] for h in par["history"]])
                self.assertEqual([h["stage"] for h in seq["history"]],[h["stage"] for h in par["history"]])

    def test_faster_with_a_slow_objective(self):
        import time
        f,_=make_objective()
        def slow(x,eid,stage):
            time.sleep(0.01)
            return f(x,eid,stage)
        hist=self.history()
        t=time.time();polish(slow,hist,BOUNDS,40,"nelder-mead",n_starts=4);t_seq=time.time()-t
        t=time.time();polish(slow,hist,BOUNDS,40,"nelder-mead",n_starts=4,parallel=4);t_par=time.time()-t
        self.assertLess(t_par,0.6*t_seq)

    def test_run_global_reports_parallel_polish(self):
        if not available("differential-evolution"):self.skipTest("optimizer unavailable")
        f,_=make_objective();many,_=make_many(f)
        r=run_global("differential-evolution",f,X0,BOUNDS,60,seed=7,polish_method="nelder-mead",
                     objective_many=many,parallel=4)
        self.assertEqual(r["polish"]["parallel"],4)
        self.assertEqual(r["evaluations"],60)
