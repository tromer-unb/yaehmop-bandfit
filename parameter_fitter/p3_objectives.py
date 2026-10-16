"""Objective-aware band fitting with the original YAeHMOP eigenvalues.

Both digitized image bands and numerical reference bands can use these
pure numerical loss functions. No ML electronic-structure surrogate.
"""
from __future__ import annotations
import numpy as np

MODES = ("global", "shape-gap", "frontier", "k-ml-weights",
         "dispersion", "pareto", "strict-pareto", "valence2", "valence2-global",
         "edge4", "edge4-global")


def diagnostics(model, target, nocc, weights, frontier_window=(-2.0, 2.0)):
    model = np.asarray(model, float)
    target = np.asarray(target, float)
    weights = np.asarray(weights, float)
    if model.shape != target.shape or model.shape != weights.shape:
        raise ValueError("Model, target and weights must have the same 2D shape")
    if model.ndim != 2 or not (0 < nocc < model.shape[1]):
        raise ValueError("Invalid electron count or orbital manifold")
    if not np.isfinite(model).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite band energies")
    t_vbm = float(np.max(target[:, nocc-1]))
    t_cbm = float(np.min(target[:, nocc]))
    m_vbm = float(np.max(model[:, nocc-1]))
    m_cbm = float(np.min(model[:, nocc]))
    shift = (t_vbm+t_cbm-m_vbm-m_cbm)/2.
    delta = model + shift - target
    point_err = np.mean(delta**2, axis=1)
    mask = (target >= frontier_window[0]) & (target <= frontier_window[1])
    frontier_rmse = float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None
    edge = delta[:, [nocc-1, nocc]]
    valence2 = delta[:, :min(nocc+2, model.shape[1])]
    edge4 = delta[:, list(range(max(0,nocc-2),nocc)) +
                   list(range(nocc,min(model.shape[1],nocc+2)))]
    take = max(1, len(point_err)//5)
    score = {
        "full_rmse_eV": float(np.sqrt(np.mean(delta**2))),
        "valence2_rmse_eV": float(np.sqrt(np.mean(valence2**2))),
        "valence2_band_count": int(valence2.shape[1]),
        "edge4_rmse_eV": float(np.sqrt(np.mean(edge4**2))),
        "edge4_band_count": int(edge4.shape[1]),
        "weighted_rmse_eV": float(np.sqrt(np.sum(weights*delta**2)/np.sum(weights))),
        "frontier_rmse_eV": frontier_rmse,
        "edge_rmse_eV": float(np.sqrt(np.mean(edge**2))),
        "top20_k_rmse_eV": float(np.sqrt(np.mean(np.sort(point_err)[-take:]))),
        "slope_residual_rmse_eV": float(np.sqrt(np.mean(np.diff(delta,axis=0)**2))),
        "gap_error_eV": float((m_cbm-m_vbm)-(t_cbm-t_vbm)),
        "model_gap_eV": m_cbm-m_vbm,
        "reference_gap_eV": t_cbm-t_vbm,
        "shift_eV": shift,
        "frontier_points": int(mask.sum()),
    }
    return score, delta, point_err


def learned_k_weights(kpath, initial_point_err, seed=20261014):
    """Train RF only on the fixed STARTING residual; freeze throughout HJ."""
    from sklearn.ensemble import RandomForestRegressor
    x = np.asarray(kpath, float).reshape(-1,1)
    y = np.sqrt(np.asarray(initial_point_err, float))
    forest = RandomForestRegressor(n_estimators=48, min_samples_leaf=5,
                                   random_state=seed, n_jobs=1)
    forest.fit(x, y)
    predicted = forest.predict(x)
    weights = np.clip(predicted/max(float(np.mean(predicted)),1.e-7), .55, 2.5)
    return weights / np.mean(weights)


def loss(mode, score, delta, point_err, k_weights, regularization,
         initial_full, gap_weight=2.0, reg_weight=0.04):
    """Seven P3 criteria; scalar gap penalty coefficients in energy units."""
    if mode not in MODES:
        raise ValueError(f"Unknown loss profile: {mode}")
    r=score["full_rmse_eV"]
    rw=score["weighted_rmse_eV"]
    f=score["frontier_rmse_eV"]
    edge=score["edge_rmse_eV"]
    g=abs(score["gap_error_eV"])
    sl=score["slope_residual_rmse_eV"]
    kml=float(np.sqrt(np.dot(k_weights,point_err)/np.sum(k_weights)))
    r20=score["top20_k_rmse_eV"]
    # Reproduce coefficients of the P3 scientific loss study. For the
    # global/legacy loss, allow customizable weight parameters.
    hinge=max(0.,g-.05)
    if mode=="global":
        return .4*r+.6*rw+gap_weight*g+reg_weight*regularization
    if mode=="valence2":
        # Same band count selection as Ribeiro et al. (2014), but no
        # externally corrected experimental-gap conduction target.
        return score["valence2_rmse_eV"]+reg_weight*regularization
    if mode=="valence2-global":
        # Our NEW conservative variant, not in Ribeiro et al.
        # Keep unselected conduction from diverging catastrophically.
        return (.65*score["valence2_rmse_eV"]+.35*r+
                4.*max(0.,r-initial_full-.05)+reg_weight*regularization)
    if mode in ("edge4","edge4-global"):
        if score["edge4_band_count"] != 4:
            raise ValueError("edge4 fitting requires two occupied and two unoccupied bands")
        selected=score["edge4_rmse_eV"]
        if mode=="edge4":
            return selected+0.4*g+reg_weight*regularization
        # Proposed protective profile, not from Ribeiro et al. (2014).
        return (.65*selected+.35*r+.4*g+
                4.*max(0.,r-initial_full-.05)+reg_weight*regularization)
    if mode=="shape-gap":
        return .85*r+.15*rw+5*hinge+.2*g+.02*regularization
    if mode in ("frontier","pareto","strict-pareto") and f is None:
        raise ValueError("No target states in frontier window for this loss mode")
    if mode=="frontier":
        return .20*r+.50*f+.30*edge+5*hinge+.2*g+.02*regularization
    if mode=="k-ml-weights":
        return .40*r+.40*kml+.20*r20+5*hinge+.2*g+.02*regularization
    if mode=="dispersion":
        return .40*r+.35*rw+sl+5*hinge+.2*g+.02*regularization
    if mode=="pareto":
        return .55*r+.45*f+6*max(0.,r-initial_full-.02)+5*hinge+.2*g+.03*regularization
    return (.55*r+.45*f+6*max(0.,r-initial_full-.02)
            +12*max(0.,g-.005)+.8*g+.03*regularization)


def hooke_jeeves(objective, x0, bounds, budget=300, step_fraction=.055,
                 shrink=.55, grow=1.13, step_tol=5e-5):
    """Bounded sequential Hooke-Jeeves search with strict call budget."""
    if budget < 1:
        raise ValueError("Budget must be positive")
    lo=np.asarray([a for a,b in bounds],float)
    hi=np.asarray([b for a,b in bounds],float)
    width=hi-lo
    if (width<=0).any(): raise ValueError("Every parameter bound must have positive width")
    x=np.clip(np.asarray(x0,float),lo,hi)
    step=np.maximum(step_fraction*width,1e-8)
    used=0
    history=[]
    best=[float("inf"),None]
    def evaluate(z,stage):
        nonlocal used
        if used>=budget: return float("inf")
        used+=1
        try:
            value,metrics=objective(np.asarray(z,float),used,stage)
            if not np.isfinite(value):
                raise ValueError("Nonfinite objective")
            entry={"eval":used,"stage":stage,"loss":float(value), **metrics}
        except Exception as e:
            value=100.
            entry={"eval":used,"stage":stage,"loss":100.,"error":repr(e)}
        history.append(entry)
        if value < best[0]:
            best[:]=[float(value),np.asarray(z,float).copy()]
        return float(value)
    fx=evaluate(x,"hj-start")
    sweeps=0
    while used<budget and float(np.max(step/width))>step_tol:
        sweeps+=1
        incumbent=fx
        candidate=x.copy()
        # Coordinate-wise poll; unlike prior all-direction batch,
        # stops exactly at budget, no artificial >budget calls.
        for j in range(len(x)):
            for sign in (+1,-1):
                if used>=budget:break
                y=np.clip(x+np.eye(1,len(x),j).ravel()*sign*step[j],lo,hi)
                if np.array_equal(y,x):continue
                f=evaluate(y,f"hj-poll-{sweeps}")
                if f < incumbent-1e-12:
                    incumbent=f;candidate=y.copy()
            if used>=budget:break
        if incumbent<fx-1e-12:
            old=x.copy()
            x=candidate;fx=incumbent
            if used<budget:
                pattern=np.clip(x+(x-old),lo,hi)
                if not np.array_equal(pattern,x):
                    f=evaluate(pattern,f"hj-pattern-{sweeps}")
                    if f<fx-1e-12:
                        x=pattern;fx=f
            step=np.minimum(step*grow,width)
        else:
            step*=shrink
    return {"best_x":best[1],"best_loss":best[0],"evaluations":used,
            "sweeps":sweeps,"history":history}


def bounded_nelder_mead(objective, x0, bounds, budget=150):
    """Nelder-Mead control with the same physical call cap as Hooke-Jeeves.

    Uses scipy.optimize.minimize with maxfev, same starting vector and
    chemical bounds, and records the actually evaluated best candidate.
    This is a legacy-style NM continuation, not a second full DE fit.
    """
    from scipy.optimize import minimize
    if budget<1:
        raise ValueError("Budget must be positive")
    lo=np.asarray([a for a,b in bounds],float)
    hi=np.asarray([b for a,b in bounds],float)
    start=np.clip(np.asarray(x0,float),lo,hi)
    history=[]
    best=[float("inf"),None]
    def evaluate(x):
        eid=len(history)+1
        try:
            value,metrics=objective(np.asarray(x,float),eid,"nm")
            if not np.isfinite(value):raise ValueError("Nonfinite objective")
            item={"eval":eid,"stage":"nm","loss":float(value),**metrics}
        except Exception as e:
            value=100.
            item={"eval":eid,"stage":"nm","loss":100.,"error":str(e)}
        history.append(item)
        if value<best[0]:
            best[:]=[float(value),np.asarray(x,float).copy()]
        return float(value)
    minimize(evaluate,start,method="Nelder-Mead",bounds=bounds,
             options={"maxfev":int(budget),"maxiter":int(budget),
                      "adaptive":True,"xatol":1e-8,"fatol":1e-8})
    return {"best_x":best[1],"best_loss":best[0],
            "evaluations":len(history),"sweeps":None,"history":history}
