#!/usr/bin/env python3
"""User-facing original-YAeHMOP CIF + PNG fitter, figure writer and band runner.

python3 bands_fit.py MoS2.cif file.txt --png MoS2.png --outdir runs/mos2
python3 bands_fit.py MoS2.cif file.txt --from-fit runs/mos2/fit/fit.json
python3 bands_fit.py MoS2.cif file.txt --from-parameters parameters.dat --k 1.75

file.txt is ordinary INI with [reference], [band_path], [fit] sections.
This is a standalone frontend, NOT a new approximate Hamiltonian.
"""
from pathlib import Path
import argparse,configparser,csv,hashlib,json,os,subprocess,sys
import numpy as np
import yaml
from ase.io import read
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"parameter_fitter"),str(ROOT/"source"),str(HERE)]
from prepare_band_png import prepare,parse_bbox
from yaehmop_generic import (resolve_yaehmop_assets,parse_parameter_table,
   subset_for_elements,valence_electrons,orbital_count,BandPath,
   write_periodic_input_generic,run_bind,parse_band,normalized_path_coordinate)
from fit_from_image import parse_points

def csv_path(x,base):
    p=Path(os.path.expandvars(os.path.expanduser(x)))
    return (p if p.is_absolute() else (base/p)).resolve()

def conf(path):
    c=configparser.ConfigParser(interpolation=None)
    if not Path(path).is_file() or not c.read(path,encoding="utf-8"):
        raise FileNotFoundError("Missing input file.txt: "+str(path))
    for s in ("reference","band_path","fit"):
        if s not in c:raise ValueError(f"Missing [{s}] in input file")
    return c

def read_nodes(c):
    cfg=c["band_path"]
    dim=int(cfg.get("dimension","2"))
    if dim not in (1,2,3):raise ValueError("dimension must be 1, 2 or 3")
    raw=cfg.get("points","")
    nodes=[[float(t.strip()) for t in row.split(",")] for row in raw.split(";") if row.strip()]
    labels=[x.strip() for x in cfg.get("labels","").split(",") if x.strip()]
    if len(nodes)<2 or len(nodes)!=len(labels) or any(len(row)!=dim for row in nodes):
        raise ValueError("point coordinates and label count inconsistent: use ';' among k points")
    count=int(cfg.get("points_per_segment","20"))
    if count<2:raise ValueError("points_per_segment must be >= 2")
    return dim,nodes,labels,count

def write_bands_plot(npz,out_dir,n,*,title,also_csv=True):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    d=np.load(npz)
    k=np.asarray(d["k"],float)
    ref=np.asarray(d["reference"],float)
    E=np.asarray(d["YAeHMOP"],float)
    if E.shape!=ref.shape or E.shape!=(len(k),n):
        raise ValueError("Stored fit/reference arrays do not have the expected N-band shape")
    fig,ax=plt.subplots(figsize=(9.7,5.5))
    for j in range(n):
        ax.plot(k,ref[:,j],color="#1F2833",lw=1.4,alpha=.74,
                label="DFT extracted PNG" if j==0 else None)
        ax.plot(k,E[:,j],color="#D25B12",lw=1.3,ls="--",
                label="YAeHMOP physical" if j==0 else None)
    ax.set_title(title)
    ax.set_xlabel("Normalized k-path length")
    ax.set_ylabel("Energy relative to midgap (eV)")
    ax.grid(alpha=.16);ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir/"bands_comparison.png",dpi=220)
    fig.savefig(out_dir/"bands_comparison.pdf")
    plt.close(fig)
    if also_csv:
        with (out_dir/"bands_comparison.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["k"]+[f"DFT_band_{j+1}" for j in range(n)]+
                       [f"EHT_band_{j+1}" for j in range(n)])
            w.writerows(np.column_stack((k,ref,E)))
    return float(np.sqrt(np.mean((ref-E)**2)))

def plot_full(k,E,out,*,name,nocc,emin=-6,emax=6):
    # The full original EHT orbital basis may contain high-energy states
    # not constrained by fitting 1V+1C; DO NOT let those untrained
    # orbitals squash the visually important near-gap region.
    mid=.5*(float(E[:,nocc].min())+float(E[:,nocc-1].max()))
    aligned=E-mid
    fig,ax=plt.subplots(figsize=(9.7,5.5))
    for j in range(E.shape[1]):
        style=dict(color="#A1ABB8",lw=.8,alpha=.65)
        if j in (nocc-1,nocc):
            style=dict(color="#1E5E95",lw=1.9,alpha=1.)
        ax.plot(k,aligned[:,j],**style)
    ax.set_title("Physical YAeHMOP band structure (gap region): "+name)
    ax.set_xlabel("Normalized k-path length")
    ax.set_ylabel("Energy relative to EHT midgap (eV)")
    ax.set_ylim(emin,emax)
    ax.grid(alpha=.16)
    fig.tight_layout();fig.savefig(out/"bands_full.png",dpi=220)
    fig.savefig(out/"bands_full.pdf");plt.close(fig)
    # Numeric CSV/NPZ keep ABSOLUTELY ALL physical eigenvalues, including
    # untrained high states outside plot limits. Nothing is discarded.
    with (out/"bands_full.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["k"]+[f"orbital_{i+1}" for i in range(E.shape[1])])
        w.writerows(np.column_stack([k,E]))
    meta={"N_occupied_spatial":int(nocc),"N_orbitals":int(E.shape[1]),
          "absolute_eigenvalue_midgap_eV":mid,"plot_ymin_eV":emin,
          "plot_ymax_eV":emax,
          "note":"PNG/PDF only show the near-gap region; CSV/NPZ preserve all physical eigenvalues."}
    (out/"bands_full_plot_info.json").write_text(json.dumps(meta,indent=2)+"\n")

def recompute(cif,config,parameters,K,outdir,overlap_radius=4):
    """Recompute physical YAeHMOP eigenbands WITHOUT using any target pixels."""
    bind,_=resolve_yaehmop_assets()
    p=parse_parameter_table(parameters)
    atoms=read(cif)
    symbols=atoms.get_chemical_symbols()
    params=subset_for_elements(p,list(dict.fromkeys(symbols)))
    nelec=valence_electrons(params,symbols)
    norb=orbital_count(params,symbols)
    if nelec%2:raise ValueError("Spinless YAeHMOP requires an even electron count")
    nocc=nelec//2
    dim,nodes,labels,step=read_nodes(config)
    xyz=parse_points([y for pt in nodes for y in pt],dim)
    bandpath=BandPath(tuple(labels),xyz,step)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    inp=outdir/"recompute.bind"
    logfile=outdir/"recompute.log"
    overlap=tuple(int(overlap_radius) if j<dim else 0 for j in range(3))
    write_periodic_input_generic(cif,inp,bandpath,float(K),True,nelec,
                                 overlap_images=overlap,periodic_dim=dim)
    run_bind(bind,inp,parameters,logfile)
    _,E=parse_band(str(inp)+".band")
    coords=[]
    for a,b in zip(xyz[:-1],xyz[1:]):
        coords.extend(a+(b-a)*(j/step) for j in range(step))
    coords.append(xyz[-1])
    k=normalized_path_coordinate(np.asarray(coords),atoms.cell)
    if E.shape!=(len(k),norb):raise ValueError("YAeHMOP band table dimension mismatch")
    ymin=config["reference"].getfloat("emin",fallback=-6.)
    ymax=config["reference"].getfloat("emax",fallback=6.)
    plot_full(k,E,outdir,name=(cif.parent.name if cif.stem.lower()=="structure"
                                  else cif.stem),nocc=nocc,emin=ymin,emax=ymax)
    np.savez_compressed(outdir/"bands_full.npz",k=k,energies=E,nocc=nocc)
    print("RECOMPUTED",len(k),"k points,",norb,
          "physical YAeHMOP eigenvalues per k from",parameters)
    print("FIGURE",outdir/"bands_full.png")
    return k,E

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif",type=Path,help="CIF crystal structure")
    parser.add_argument("input_file",type=Path,help="Plain INI file.txt (axis, path, optimizer settings)")
    parser.add_argument("--image","--png",dest="png",type=Path,
                        help="Any Pillow-decodable static raster or CSV overriding [reference] image; --png is a legacy alias")
    parser.add_argument("--name",help="Material label in saved files and plot titles")
    parser.add_argument("--outdir",type=Path,default=Path("bands_fit_output"))
    parser.add_argument("--pairs",type=int,choices=(1,2,3,4),help="VB/CB bands EACH")
    parser.add_argument("--budget",type=int,help="YAeHMOP objective calls")
    parser.add_argument("--mode",choices=("auto","layered","colors","mono","csv"))
    parser.add_argument("--bbox",help="Override plot interior: L,T,R,B")
    parser.add_argument("--colors",help="Override distinct RGB colors, one per band")
    parser.add_argument("--allow-ambiguous",action="store_true",
                        help="Unreliable single-color traced image accepted at user's risk")
    parser.add_argument("--prepare-only",action="store_true",
                        help="Create encoded input/QA preview but do not fit")
    parser.add_argument("--validate-only",action="store_true",
                        help="Decode/validate raster format and config without band extraction or solver (use --prepare-only for full band QA)")
    parser.add_argument("--from-fit",type=Path,
                        help="Load saved fit.json and RE-RUN physical eigenbands with saved K")
    parser.add_argument("--from-parameters",type=Path,
                        help="Recompute full physical bands from original-YAeHMOP parameter table")
    parser.add_argument("--k",type=float,help="Required with --from-parameters")
    parser.add_argument("--overlap-radius",type=int,default=4,
                        help="Number of overlap images in each active lattice direction when recomputing physical bands (default: 4)")
    parser.add_argument("--optimizer",choices=("nelder-mead","simulated-annealing",
                        "sa-nelder-mead","hooke-jeeves"),default=None)
    parser.add_argument("--seed",type=int,default=20261014)
    args=parser.parse_args()
    cif=args.cif.resolve()
    inputs=args.input_file.resolve()
    if not cif.is_file():parser.error("CIF not found: "+str(cif))
    config=conf(inputs)
    dim,nodes,labels,step=read_nodes(config)
    fit=config["fit"];ref=config["reference"]
    pairs=int(args.pairs or fit.get("n_pairs","1"))
    if pairs not in (1,2,3,4):parser.error("n_pairs must be 1,2,3,4")
    outdir=args.outdir.resolve();outdir.mkdir(parents=True,exist_ok=True)
    if args.overlap_radius not in range(1,9):
        parser.error("--overlap-radius must be in [1,8]")
    if args.from_fit:
        source=args.from_fit.resolve()
        status=json.loads(source.read_text())
        saved_K=float(status["best_x"]["GLOBAL__K"])
        native=source.parent/"param_best_native.dat"
        if not native.is_file():raise FileNotFoundError(native)
        # Ensure the user supplied the same CIF as the original fit.
        files=status.get("train_reference_hashes",{})
        if len(files)==1:
            record=next(iter(files.values()))
            if record["cif"]!=hashlib.sha256(cif.read_bytes()).hexdigest():
                raise ValueError("CIF SHA256 differs from original fit; refusing silent substitution")
        recompute(cif,config,native,saved_K,outdir,args.overlap_radius)
        return
    if args.from_parameters:
        if args.k is None:parser.error("--k is required with --from-parameters")
        recompute(cif,config,args.from_parameters.resolve(),args.k,outdir,args.overlap_radius)
        return
    image=args.png
    if image is None:
        entry=ref.get("image","").strip()
        if not entry:parser.error("Supply --png or [reference] image")
        image=csv_path(entry,inputs.parent)
    image=image.resolve()
    if not image.is_file():parser.error("Raster/CSV input not found: "+str(image))
    emin=ref.getfloat("emin")
    emax=ref.getfloat("emax")
    mode=args.mode or ref.get("mode","auto")
    bbox=parse_bbox(args.bbox or ref.get("bbox",fallback=None))
    colors=args.colors or ref.get("colors",fallback=None)
    tolerance=ref.getint("color_tolerance",fallback=36)
    nx=ref.getint("trace_nx",fallback=301)
    coverage=ref.getfloat("min_coverage",fallback=.75)
    budget=int(args.budget or fit.get("budget","200"))
    optimizer=args.optimizer or fit.get("optimizer","nelder-mead")
    if budget<80 and not args.validate_only and not args.prepare_only:
        parser.error("Staged fit requires at least 80 YAeHMOP objective evaluations")
    if args.validate_only:
        rasterqa=None
        if mode!="csv" and not (mode=="auto" and image.suffix.lower()==".csv"):
            from raster_ingest import normalize_and_write_qa
            _,rasterqa=normalize_and_write_qa(image,outdir/"input_validation",
                                    n_bands=2*pairs,expect_layered=(mode=="layered"))
        else:
            from prepare_band_png import csv_curves
            csv_curves(image,2*pairs,emin,emax)
        print(json.dumps(dict(cif=str(cif),image=str(image),band_count=2*pairs,
            high_symmetry_labels=labels,points=nodes,energy_limits=[emin,emax],
            optimizer=optimizer,budget=budget,image_mode=mode,
            raster_ingest=rasterqa,
            note="Format and configuration verified only. For N-band tracing, coverage and gap validation use --prepare-only. No YAeHMOP solver used."),indent=2))
        return
    prepared,qa=prepare(image,outdir/"reference_prepared",pairs,emin,emax,
            mode=mode,bbox=bbox,colors=colors,tolerance=tolerance,nx=nx,
            min_coverage=coverage,allow_ambiguous=args.allow_ambiguous)
    if args.prepare_only:
        print("PREPARED_ONLY: review",qa["preview"],"before fitting.")
        return
    # Name identifies system; source paths absolute for reproducibility
    name=(args.name or fit.get("name",fallback=None) or
          (cif.parent.name if cif.stem.lower()=="structure" else cif.stem))
    if not name or any(c in name for c in ("/","\\","..")):
        raise ValueError("Material name must not contain slash or '..'")
    yml={"n_pairs":pairs,"points_per_line":step,"trace_nx":nx,
         "image_mode":"colored",
         "gap_weight":float(fit.get("gap_weight",".35")),
         "reg_weight":float(fit.get("reg_weight",".04")),
         "systems":[{
             "name":name,"role":"train","weight":1.0,
             "structure":str(cif),"reference":str(prepared),
             "emin":float(emin),"emax":float(emax),
             "point_dim":dim,"symmetry_points":nodes,
             "labels":labels,"n_pairs":pairs,
             "image_mode":"colored","points_per_line":step,"trace_nx":nx}]}
    target=outdir/"fit_input.yaml"
    target.write_text(yaml.safe_dump(yml,sort_keys=False))
    core=[sys.executable,str(ROOT/"parameter_fitter/fit_sparse_window.py"),str(target)]
    if optimizer=="nelder-mead":
        # P5 staged mode is original YAeHMOP DE followed by bounded NM.
        base_calls=budget
        refinement_calls=0
        cmd=core+["--outdir",str(outdir/"fit"),"--initialization","staged",
                  "--budget",str(budget),"--seed",str(args.seed)]
        subprocess.run(cmd,check=True,text=True)
    else:
        if budget<160:
            parser.error("For SA/HJ refinement set total --budget >=160 (base >=80)")
        base_calls=max(80,int(round(.6*budget)))
        refinement_calls=budget-base_calls
        # Base and refinement share the SAME prepared input and CIF;
        # no cold-start extra calls hidden inside the refinement.
        subprocess.run(core+["--outdir",str(outdir/"fit_base"),
                      "--initialization","staged","--budget",str(base_calls),
                      "--seed",str(args.seed)],check=True,text=True)
        subprocess.run(core+["--outdir",str(outdir/"fit"),
                      "--initialization","existing",
                      "--start-json",str(outdir/"fit_base/fit.json"),
                      "--optimizer",str(optimizer),
                      "--budget",str(refinement_calls),"--seed",str(args.seed)],
                      check=True,text=True)
    state=json.loads((outdir/"fit/fit.json").read_text())
    met=state["metrics"][name]
    objective_rmse=write_bands_plot(outdir/"fit"/f"{name}_bands.npz",outdir,2*pairs,
                title=f"{name} — {pairs}VB + {pairs}CB (fit with {2*pairs} bands)")
    if abs(objective_rmse-float(met["selected_rmse_eV"]))>1.e-8:
        raise ValueError("Physical DFT/EHT figure disagrees with optimizer metric")
    status={
        "cif":str(cif),"cif_sha256":hashlib.sha256(cif.read_bytes()).hexdigest(),
        "reference_input":str(image),"input_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
        "extraction":qa,"n_pairs":pairs,"n_scored_bands":2*pairs,
        "optimizer_evaluations":state["objective_evaluations"],
        "initialization_calls":base_calls,"refinement_calls":refinement_calls,
        "requested_total_budget":budget,"optimizer_choice":optimizer,
        "selected_rmse_eV":met["selected_rmse_eV"],"gap_error_eV":met["gap_error_eV"],
        "K_EHT":state["best_x"]["GLOBAL__K"],"native_parameters":str(outdir/"fit/param_best_native.dat"),
        "figure":str(outdir/"bands_comparison.png"),
        "note":"Physical YAeHMOP calculations; trained only N=2*n_pairs frontier bands"}
    (outdir/"run_summary.json").write_text(json.dumps(status,indent=2)+"\n")
    print("USER_FIT_DONE",name,"bands",2*pairs,
          "RMSE_eV",round(objective_rmse,5))
    print("BEST_PARAMS",outdir/"fit/param_best_native.dat")
    print("BANDS_PLOT",outdir/"bands_comparison.png")
    print("BANDS_CSV",outdir/"bands_comparison.csv")

if __name__=="__main__":
    main()
