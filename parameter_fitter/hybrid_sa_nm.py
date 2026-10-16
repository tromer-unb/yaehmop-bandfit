"""Two-stage SA -> bounded Nelder-Mead refinement with exact total budget.

The SAME physical YAeHMOP objective is used in both stages. Every trial
is recorded; the second-stage starting point is the SA *best training
loss*, never an oracle heldout result. No electronic surrogate.
"""
from __future__ import annotations
import numpy as np
from simulated_annealing import simulated_annealing
from p3_objectives import bounded_nelder_mead

def hybrid_sa_nelder_mead(objective,x0,bounds,budget=150,seed=20261014,
                          sa_fraction=.6,step_fraction=.14,
                          initial_temp_ratio=.12,final_temp_ratio=.004):
    if budget < 4:
        raise ValueError("Hybrid needs at least four objective calls")
    if not 0 < sa_fraction < 1:
        raise ValueError("SA fraction must be between zero and one")
    sa_calls=max(2,min(budget-2,int(round(budget*sa_fraction))))
    nm_calls=budget-sa_calls
    history=[]
    global_calls=0
    def wrap(x,eid,stage):
        nonlocal global_calls
        global_calls+=1
        # Preserve per-stage provenance and a GLOBAL call ID for YAeHMOP.
        value,metrics=objective(np.asarray(x,float),global_calls,stage)
        return value,metrics
    sa=simulated_annealing(wrap,x0,bounds,budget=sa_calls,seed=seed,
                           step_fraction=step_fraction,
                           initial_temp_ratio=initial_temp_ratio,
                           final_temp_ratio=final_temp_ratio)
    for row in sa["history"]:
        history.append(dict(row,stage="hybrid-sa-"+row["stage"]))
    if sa["best_x"] is None:
        raise RuntimeError("SA stage found no finite physical candidate")
    nm=bounded_nelder_mead(wrap,sa["best_x"],bounds,budget=nm_calls)
    for row in nm["history"]:
        history.append(dict(row,eval=row["eval"]+sa_calls,
                            stage="hybrid-nm-"+row["stage"]))
    assert len(history)==sa["evaluations"]+nm["evaluations"]
    # Very important: return the objective minimum achieved in EITHER
    # stage, not only the last NM vector (NM may be worse).
    if nm["best_x"] is not None and nm["best_loss"]<sa["best_loss"]:
        best_x=nm["best_x"];best_loss=nm["best_loss"]
    else:
        best_x=sa["best_x"];best_loss=sa["best_loss"]
    return {"best_x":best_x,"best_loss":best_loss,
            "evaluations":len(history),"sweeps":None,"history":history,
            "seed":seed,"sa_stage_evaluations":sa["evaluations"],
            "nm_stage_evaluations":nm["evaluations"],
            "sa_acceptance_rate":sa["acceptance_rate"],
            "stage_transition_best_loss":sa["best_loss"]}
