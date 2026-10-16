#!/usr/bin/env python3
"""Variable 1+1, 2+2, 3+3, or 4+4 CIF+PNG YAeHMOP direct/shared fitter (P5).

Train from a YAML containing 1+ CIF/PNG training systems and optional
withheld systems. Each PNG contains N highest VB and N lowest CB, N in (1,2,3,4).
NO absent deep or upper DFT bands are fabricated; YAeHMOP still solves
the FULL original Hamiltonian for each real parameter candidate.

Staged initializer: original YAeHMOP DE -> bounded Nelder-Mead.
Optional equal-budget refinements: NM, SA, SA+NM, Hooke-Jeeves.
"""
from pathlib import Path
import argparse,csv,json,hashlib,sys
import numpy as np,yaml
from ase.io import read
from scipy.optimize import differential_evolution
from fit_from_image import build_specs,apply_x,physical_ok,parse_points
from fit_shared import valid_basis
from image_target import load_band_image,trace_bands
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from generate_mc2d_multiband import read_colored_png
from yaehmop_generic import (resolve_yaehmop_assets,parse_parameter_table,
 subset_for_elements,orbital_count,valence_electrons,
 BandPath,normalized_path_coordinate,write_parameter_file,
 write_periodic_input_generic,run_bind,parse_band)
from p3_objectives import bounded_nelder_mead,hooke_jeeves
from simulated_annealing import simulated_annealing
from hybrid_sa_nm import hybrid_sa_nelder_mead

def parse_sample(e,root,defaults,shared):
    cif=(root/e["structure"]).resolve()
    png=(root/e["reference"]).resolve()
    atoms=read(cif);syms=atoms.get_chemical_symbols()
    params={k:defaults[k] for k in set(syms)}
    norb=orbital_count(params,syms)
    nelec=valence_electrons(params,syms)
    if nelec%2:raise ValueError("spinless YAeHMOP needs even nelec")
    nocc=nelec//2
    n_pairs=int(e.get("n_pairs",shared.get("n_pairs",2)))
    if n_pairs not in (1,2,3,4):
        raise ValueError("P5 N must be 1,2,3,4")
    if min(nocc,norb-nocc)<n_pairs:
        raise ValueError(f"{e['name']} requires {n_pairs} conduction/valence orbitals; EHT only has {norb-nocc} conduction")

    dim=int(e.get("point_dim",2))
    nodes=parse_points([x for node in e["symmetry_points"] for x in node],dim)
    labels=tuple(e["labels"])
    n=int(e.get("points_per_line",shared.get("points_per_line",10)))
    path=BandPath(labels,nodes,n)
    kgrid=[]
    for a,b in zip(nodes[:-1],nodes[1:]):
        kgrid.extend(a+(b-a)*(i/n) for i in range(n))
    kgrid.append(nodes[-1])
    kx=normalized_path_coordinate(np.asarray(kgrid),atoms.cell)
    mode=e.get("image_mode",shared.get("image_mode","auto"))
    emin,emax=float(e["emin"]),float(e["emax"])
    trace_nx=int(e.get("trace_nx",shared.get("trace_nx",201)))
    if mode=="colored":
        rx,bands=read_colored_png(png,n_pairs,emin,emax,nx=trace_nx)
    elif mode=="monochrome":
        image=load_band_image(png,emin,emax)
        rx,bands,counts,seed=trace_bands(image,2*n_pairs,nx=trace_nx)
    else:
        try:
            rx,bands=read_colored_png(png,n_pairs,emin,emax,nx=trace_nx)
        except (ValueError,RuntimeError):
            image=load_band_image(png,emin,emax)
            rx,bands,counts,seed=trace_bands(image,2*n_pairs,nx=trace_nx)
    target=np.column_stack([np.interp(kx,rx,bands[:,j]) for j in range(2*n_pairs)])
    tv=float(target[:,n_pairs-1].max());tc=float(target[:,n_pairs].min())
    if tc-tv<.02:raise ValueError(e["name"]+" invalid band gap in image")
    target-=.5*(tv+tc)
    return dict(name=e["name"],cif=cif,png=png,symbols=syms,
      elements=list(dict.fromkeys(syms)),norb=norb,nelec=nelec,nocc=nocc,
      path=path,kx=kx,target=target,n_pairs=n_pairs,gap=tc-tv,weight=float(e.get("weight",1.)),
      png_hash=hashlib.sha256(png.read_bytes()).hexdigest(),
      cif_hash=hashlib.sha256(cif.read_bytes()).hexdigest())

def scores(bands,s):
    n=s["nocc"]
    if bands.shape!=(len(s["kx"]),s["norb"]):
        raise ValueError("physical EHT spectrum shape mismatch")
    gap=float(bands[:,n].min()-bands[:,n-1].max())
    mid=.5*(float(bands[:,n].min())+float(bands[:,n-1].max()))
    p=s["n_pairs"]
    selected=bands[:,n-p:n+p]-mid
    delta=selected-s["target"]
    two_delta=delta[:,p-1:p+1]
    four_delta=(delta[:,p-2:p+2] if p>=2 else None)
    return dict(
       selected_rmse_eV=float(np.sqrt(np.mean(delta**2))),
       two_rmse_eV=float(np.sqrt(np.mean(two_delta**2))),
       four_rmse_eV=(float(np.sqrt(np.mean(four_delta**2))) if four_delta is not None else None),
       valence_rmse_eV=float(np.sqrt(np.mean(delta[:,:p]**2))),
       conduction_rmse_eV=float(np.sqrt(np.mean(delta[:,p:]**2))),
       frontier_valence_rmse_eV=float(np.sqrt(np.mean(two_delta[:,:1]**2))),
       frontier_conduction_rmse_eV=float(np.sqrt(np.mean(two_delta[:,1:]**2))),
       valence2_rmse_eV=(float(np.sqrt(np.mean(four_delta[:,:2]**2))) if four_delta is not None else None),
       conduction2_rmse_eV=(float(np.sqrt(np.mean(four_delta[:,2:]**2))) if four_delta is not None else None),
       gap_error_eV=gap-s["gap"],ref_gap_eV=s["gap"],
       model_gap_eV=gap,nocc=n,norb=s["norb"],n_pairs=p),selected

class Fitter:
    def __init__(self,config,base,out):
        self.conf=config;self.base=base;self.out=out
        self.work=out/"work";self.work.mkdir(exist_ok=True,parents=True)
        self.train_entries=[e for e in config["systems"] if e.get("role","train")=="train"]
        self.test_entries=[e for e in config["systems"] if e.get("role")=="test"]
        if not self.train_entries:raise ValueError("Need at least one training system")
        elements=[]
        for e in self.train_entries:
            for el in read((base/e["structure"]).resolve()).get_chemical_symbols():
                if el not in elements:elements.append(el)
        self.elements=elements
        self.bind,table=resolve_yaehmop_assets()
        self.stockparams=valid_basis(subset_for_elements(parse_parameter_table(table),elements))
        self.train=[parse_sample(e,base,self.stockparams,config) for e in self.train_entries]
        self.specs,self.stock,self.bounds,self.scales=build_specs(self.stockparams,elements)
        self.bounds=list(self.bounds)
        for i,spec in enumerate(self.specs):
            if spec[2]=="Hii":
                self.bounds[i]=(self.stock[i]-5.,self.stock[i]+5.)
        self.names=[f"{el}_{orb}_{key}".strip("_") for el,orb,key in self.specs]
        self.pairs=int(config.get("n_pairs",2))
        if any(s["n_pairs"]!=self.pairs for s in self.train):
            raise ValueError("All train systems must use same n_pairs")
        self.gap_w=float(config.get("gap_weight",.35))
        self.reg_w=float(config.get("reg_weight",.04))
        self.calls=0
    def run(self,s,params,k,tag):
        w=self.work/s["name"];w.mkdir(exist_ok=True)
        native=w/(tag+".parms");inp=w/(tag+".bind");log=w/(tag+".log")
        write_parameter_file(native,{a:params[a] for a in s["elements"]})
        write_periodic_input_generic(s["cif"],inp,s["path"],k,True,s["nelec"])
        run_bind(self.bind,inp,native,log)
        _,bands=parse_band(str(inp)+".band")
        for tmp in w.glob(tag+"*"):
            if tmp.is_file():tmp.unlink()
        return bands
    def objective(self,x,eid,stage):
        self.calls+=1
        try:
            params,K=apply_x(self.stockparams,self.specs,np.asarray(x,float))
            if not physical_ok(params):raise ValueError("Invalid double-zeta parameter")
            values=[];weights=[]
            line=dict(K=float(K))
            for s in self.train:
                m,_=scores(self.run(s,params,K,f"trial_{self.calls:07d}"),s)
                val=m["selected_rmse_eV"]+self.gap_w*abs(m["gap_error_eV"])
                values.append(val);weights.append(s["weight"])
                line[s["name"]+"_selected_rmse_eV"]=m["selected_rmse_eV"]
                line[s["name"]+"_two_rmse_eV"]=m["two_rmse_eV"]
                line[s["name"]+"_four_rmse_eV"]=m["four_rmse_eV"]
                line[s["name"]+"_gap_error_eV"]=m["gap_error_eV"]
            reg=float(np.sqrt(np.mean(((np.asarray(x)-self.stock)/self.scales)**2)))
            cost=float(np.average(values,weights=weights)+self.reg_w*reg)
            line["regularization"]=reg
            return cost,line
        except Exception as exc:
            return 1e4,{"error":str(exc)}
    def characterize(self,x,withheld):
        params,K=apply_x(self.stockparams,self.specs,np.asarray(x,float))
        specs=list(self.train)
        if withheld:
            # Heldout CIF and PNG are opened ONLY after the optimizer freezes x.
            for e in self.test_entries:
                ats=read((self.base/e["structure"]).resolve())
                if not set(ats.get_chemical_symbols()).issubset(self.elements):
                    raise ValueError("Unseen test elements cannot be transferred")
                specs.append(parse_sample(e,self.base,self.stockparams,self.conf))
        result={}
        for s in specs:
            met,b=scores(self.run(s,params,K,"final_eigen"),s)
            met["role"]="train" if s["name"] in [t["name"] for t in self.train] else "test"
            result[s["name"]]=met
            np.savez_compressed(self.out/(s["name"]+"_bands.npz"),k=s["kx"],
                                reference=s["target"],YAeHMOP=b)
        return result

def base_fit(f,budget,seed):
    if budget<80:raise ValueError("Minimum base budget 80")
    hist=[];best=[float("inf"),f.stock.copy()]
    def ev(x,stage):
        val,meta=f.objective(x,len(hist)+1,stage)
        hist.append({"eval":len(hist)+1,"stage":stage,"loss":float(val),**meta})
        if val<best[0]:
            best[:]=[float(val),np.asarray(x,float).copy()]
        return val
    ev(f.stock,"stock")
    dim=len(f.names); pop=2
    iterations=max(1,min(3,(budget-40)//(dim*pop)-1))
    differential_evolution(lambda x:ev(x,"DE"),f.bounds,
            seed=seed,maxiter=iterations,popsize=pop,polish=False,
            updating="immediate",workers=1)
    budget_left=budget-len(hist)
    if budget_left<1:raise RuntimeError("Global pass overran base call budget")
    prefix=len(hist)
    def local(x,eid,stage):
        return f.objective(x,prefix+eid,"base-NM")
    res=bounded_nelder_mead(local,best[1],f.bounds,budget=budget_left)
    for row in res["history"]:
        hist.append({**row,"eval":prefix+row["eval"],"stage":"base-NM"})
    if res["best_x"] is not None and res["best_loss"]<best[0]:
        best[:]=[res["best_loss"],res["best_x"].copy()]
    return dict(best_x=best[1],best_loss=best[0],
                evaluations=len(hist),history=hist,optimizer="DE+NM",
                de_iters=iterations)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--outdir",required=True)
    p.add_argument("--initialization",choices=("staged","existing","transferred"),default="staged")
    p.add_argument("--start-json")
    p.add_argument("--optimizer",choices=("nelder-mead","simulated-annealing",
                   "sa-nelder-mead","hooke-jeeves"),default="nelder-mead")
    p.add_argument("--budget",type=int,default=300)
    p.add_argument("--n-pairs",type=int,choices=(1,2,3,4),default=None,
                   help="Override YAML; number of VB and CB bands each")
    p.add_argument("--seed",type=int,default=20261014)
    p.add_argument("--skip-test-eval",action="store_true")
    p.add_argument("--validate-only",action="store_true")
    args=p.parse_args()
    cfgfile=Path(args.config).resolve()
    config=yaml.safe_load(cfgfile.read_text())
    if args.n_pairs is not None:
        config["n_pairs"]=args.n_pairs
    if int(config.get("n_pairs",2)) not in (1,2,3,4):
        p.error("n_pairs must be 1, 2, 3 or 4")
    names=[e["name"] for e in config["systems"] if e.get("role","train")=="train"]
    tests=[e["name"] for e in config["systems"] if e.get("role")=="test"]
    if args.validate_only:
        print(json.dumps(dict(train=names,test=tests,test_files_opened=False)))
        return
    out=Path(args.outdir).resolve();out.mkdir(exist_ok=True,parents=True)
    f=Fitter(config,cfgfile.parent,out)
    if args.initialization=="staged":
        result=base_fit(f,args.budget,args.seed)
        x=result["best_x"]
        start_meta=None
    else:
        if not args.start_json:p.error("--start-json is mandatory for an existing or transferred start")
        prior=json.loads(Path(args.start_json).read_text())
        if args.initialization=="existing":
            if prior["train_systems"]!=names:
                raise ValueError("Previous individual training systems mismatch")
            if set(prior["best_x"])!=set(f.names):
                raise ValueError("Element/spec names mismatch")
        else:
            # True adaptation: previous SHARED model was trained without
            # the current new material. Project elemental parameters
            # into the smaller per-material subspace.
            if prior.get("test_systems")!=names:
                raise ValueError("Transfer warm start must have withheld exactly this material")
            if any(n in prior["train_systems"] for n in names):
                raise ValueError("Transferred vector has target-data leakage")
            if not set(f.names).issubset(prior["best_x"]):
                raise ValueError("Transfer input misses required elemental parameters")
        if prior.get("loss_profile")!=f"sparse_{2*f.pairs}":
            raise ValueError("Start JSON has different spectral-window size")
        x0=np.array([prior["best_x"][n] for n in f.names],float)
        # For adaptation we intentionally use this TARGET reference;
        # the resulting fit is NOT zero-shot transfer.
        baseline_loss=None
        if args.initialization=="transferred":
            baseline_loss=float(f.objective(x0,0,"adaptation-before-refit")[0])
        options={"nelder-mead":bounded_nelder_mead,
                 "simulated-annealing":simulated_annealing,
                 "sa-nelder-mead":hybrid_sa_nelder_mead,
                 "hooke-jeeves":hooke_jeeves}
        kw={"budget":args.budget}
        if args.optimizer in ("simulated-annealing","sa-nelder-mead"):
            kw["seed"]=args.seed
        result=options[args.optimizer](f.objective,x0,f.bounds,**kw)
        result["optimizer"]=args.optimizer
        # Freeze at global best TRAINING objective found in either stage;
        # do not use the heldout reference to choose this vector.
        compare_loss=(baseline_loss if baseline_loss is not None else
                      prior["best_training_loss"])
        x=(result["best_x"] if result["best_x"] is not None and
           result["best_loss"]<compare_loss else x0)
        start_meta=str(Path(args.start_json).resolve())
    met=f.characterize(x,not args.skip_test_eval)
    state=dict(loss_profile=f"sparse_{2*f.pairs}",n_pairs=f.pairs,
      image_mode=config.get("image_mode","auto"),initializer=args.initialization,
      adaptation_prior_material_was_heldout=(
           args.initialization=="transferred"),
      optimizer=result["optimizer"],train_systems=names,test_systems=tests,
      test_evaluated=not args.skip_test_eval,budget=args.budget,
      objective_evaluations=result["evaluations"],seed=args.seed,
      best_training_loss=float(result["best_loss"] if args.initialization=="staged"
                        else min(result["best_loss"],baseline_loss
                            if args.initialization=="transferred"
                            else prior["best_training_loss"])),
      optimizer_best_loss=float(result["best_loss"]),
      transfer_baseline_loss=(baseline_loss if args.initialization=="transferred" else None),
      transferred_initial_extra_calls=(1 if args.initialization=="transferred" else 0),
      best_x={n:float(v) for n,v in zip(f.names,x)},
      stock_x={n:float(v) for n,v in zip(f.names,f.stock)},
      train_reference_hashes={s["name"]:{"png":s["png_hash"],"cif":s["cif_hash"]}
                        for s in f.train},
      training_only_selection=True,source_base_json=start_meta,
      hybrid_stage_calls=[result.get("sa_stage_evaluations"),
                          result.get("nm_stage_evaluations")] if args.optimizer=="sa-nelder-mead" and
                            args.initialization=="existing" else None,
      objective_weights={"rmse_selected":1.,"gapMAE":f.gap_w,"param_reg":f.reg_w},
      metrics=met,n_params=len(f.names))
    (out/"fit.json").write_text(json.dumps(state,indent=2)+"\n")
    # Export directly loadable atomic parameters for original YAeHMOP.
    parameters,K=apply_x(f.stockparams,f.specs,x)
    write_parameter_file(out/"param_best_native.dat",parameters)
    (out/"param_best.txt").write_text(
        "K_EHT "+format(K,".12g")+"\n"+
        "".join(name+" "+format(float(v),".12g")+"\n"
                for name,v in zip(f.names,x)))
    history=result["history"]
    keys=list(dict.fromkeys(k for r in history for k in r))
    with (out/"history.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(history)
    print("FIT_DONE",out,"evals",result["evaluations"],"best train",
           round(state["best_training_loss"],5),"scores",[
           (k,round(v["selected_rmse_eV"],4),round(v["two_rmse_eV"],4),
            round(abs(v["gap_error_eV"]),4),
            v["role"]) for k,v in met.items()],flush=True)

if __name__=="__main__":
    main()
