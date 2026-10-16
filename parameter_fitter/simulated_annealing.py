"""Bounded simulated annealing; YAeHMOP remains the electronic Hamiltonian.

Inspired by the annealing concept in Ribeiro et al., Solid State
Communications 186 (2014) 50-55. Their exact cooling schedule and total
computational budget are not published; this is NOT their implementation.
"""
import math
import numpy as np


def simulated_annealing(objective, x0, bounds, budget=150, seed=20261014,
                        step_fraction=.14, initial_temp_ratio=.12,
                        final_temp_ratio=.004):
    """Metropolis annealing with bounded proposals, exact evaluation budget.

    Objective: (x, call_id, stage) -> (loss float, diagnostics dict).
    Temperature units inherit the scalar training loss.
    Cooling is geometric with four reheated cycles. Proposals use single,
    small-block or whole-parameter perturbations, chosen by seed.
    """
    if budget<1:raise ValueError("budget must be >=1")
    if not 0 < final_temp_ratio < initial_temp_ratio:
        raise ValueError("0 < end temperature < start temperature required")
    lo=np.asarray([a for a,b in bounds],float)
    hi=np.asarray([b for a,b in bounds],float)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (hi<=lo).any():
        raise ValueError("Parameter bounds must be finite/increasing")
    x=np.asarray(x0,float)
    if x.shape!=lo.shape or not np.isfinite(x).all():
        raise ValueError("Initial vector shape/values invalid")
    x=np.clip(x,lo,hi)
    rng=np.random.default_rng(int(seed))
    span=hi-lo
    history=[]
    best=[float("inf"),None]
    failures=0
    def evaluate(point,stage,temp):
        nonlocal failures
        n=len(history)+1
        try:
            loss,diagnostics=objective(point.copy(),n,stage)
            loss=float(loss)
            if not math.isfinite(loss):raise ValueError("Nonfinite loss")
            if "error" in diagnostics:raise ValueError(str(diagnostics["error"]))
            entry={"eval":n,"stage":stage,"loss":loss,"temperature":float(temp),
                   **dict(diagnostics)}
        except Exception as e:
            loss=1e6;failures+=1
            entry={"eval":n,"stage":stage,"loss":loss,"temperature":float(temp),
                   "error":str(e)}
        history.append(entry)
        if loss<best[0]:
            best[:]=[loss,point.copy()]
        return loss,entry
    loss,_=evaluate(x,"sa-initial",0.)
    unit=max(0.05,abs(loss) if loss<1e5 else 1.)
    high=initial_temp_ratio*unit
    low=final_temp_ratio*unit
    accepted=0
    accepted_uphill=0
    for idx in range(1,budget):
        fraction=(idx-1)/max(1,budget-2)
        phase=(4*fraction)%1.
        temp=high*(low/high)**phase
        mode=rng.random()
        if mode<.65:
            indices=rng.choice(len(x),1,replace=False)
        elif mode<.92:
            indices=rng.choice(len(x),min(len(x),3),replace=False)
        else:
            indices=np.arange(len(x))
        candidate=x.copy()
        size=step_fraction*(.12+.88*(temp/high)**.45)
        candidate[indices]=np.clip(x[indices]+rng.normal(0.,size,len(indices))*span[indices],
                                   lo[indices],hi[indices])
        if np.array_equal(candidate,x):
            j=int(rng.integers(len(x)))
            candidate[j]=np.clip(candidate[j]+rng.choice([-1.,1.])*.01*span[j],lo[j],hi[j])
        value,entry=evaluate(candidate,"sa-proposal",temp)
        delta=value-loss
        accept=(delta<=0 or rng.random()<math.exp(-min(700.,delta/max(temp,1e-12))))
        entry["accepted"]=bool(accept)
        if accept:
            accepted+=1
            accepted_uphill+=int(delta>0)
            x=candidate;loss=value
    return {"best_x":best[1],"best_loss":best[0],
            "evaluations":len(history),"sweeps":None,"history":history,
            "seed":int(seed),"acceptance_rate":accepted/max(1,budget-1),
            "accepted_uphill":accepted_uphill,"physical_failures":failures}
