#!/usr/bin/env python3
"""Variable 1+1, 2+2, 3+3, or 4+4 CIF+PNG YAeHMOP direct/shared fitter (P5).

Train from a YAML containing 1+ CIF/PNG training systems and optional
withheld systems. Each PNG contains N highest VB and N lowest CB, N in (1,2,3,4).
NO absent deep or upper DFT bands are fabricated; YAeHMOP still solves
the FULL original Hamiltonian for each real parameter candidate.

Staged initializer: original YAeHMOP DE -> bounded Nelder-Mead.
Optional equal-budget refinements: NM, SA, SA+NM, Hooke-Jeeves.
Global searches from the stock parameters (ported from dftb_sk_gen):
full DE, Bayesian optimization, CMA-ES, genetic algorithm, Powell, each
with an optional local polish (Nelder-Mead or Powell) inside the budget.
"""
from pathlib import Path
import argparse,csv,json,hashlib,os,sys,threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np,yaml
from structure_io import read_structure,resolved_structure_format
from dft_bands import load_reference,eht_reference
from band_objective import WindowSettings,window_error,metal_error,extend_basis,parse_basis_add,make_double_zeta,OBJECTIVES
from scipy.optimize import differential_evolution
from fit_from_image import build_specs,apply_x,physical_ok,parse_points
from fit_shared import valid_basis
from image_target import load_band_image,trace_bands
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from generate_mc2d_multiband import read_colored_png
from yaehmop_generic import (resolve_yaehmop_assets,parse_parameter_table,
 subset_for_elements,orbital_count,valence_electrons,
 BandPath,normalized_path_coordinate,write_parameter_file,
 write_periodic_input_generic,run_bind,parse_band,
 DEFAULT_OVERLAP_RADIUS,overlap_images_for,gamma_mesh,read_matrix_dump)
from p3_objectives import bounded_nelder_mead,hooke_jeeves
from simulated_annealing import simulated_annealing
from hybrid_sa_nm import hybrid_sa_nelder_mead
from global_optimizers import run_global,GLOBAL_METHODS,POLISH_METHODS

# YAeHMOP diagonal of the periodic images of an orbital: H_ii(k) = Hii (1 - Kd + Kd S_ii(k)),
# Kd = THE CONST DIAG (not written in our inputs, so the default 1.75)
THE_CONST_DIAG=1.75
FAST_KEYS=("Hii","K")

_FAST_FITTER=None   # the Fitter seen by the forked workers of the fast (cached S(k)) stage

def _fast_init():
    _FAST_FITTER.workers=1   # one process per candidate, its systems in sequence

def _fast_child(a):
    try:return _FAST_FITTER.evaluate(a[0],a[1])
    except Exception as exc:return RuntimeError(str(exc))

def cpu_count():
    try:return len(os.sched_getaffinity(0))
    except AttributeError:return os.cpu_count() or 1

def radius_spec(e,shared):
    """overlap_radius of a system: an integer, or 'auto' (chosen by convergence in the Fitter)."""
    r=e.get("overlap_radius",shared.get("overlap_radius",DEFAULT_OVERLAP_RADIUS))
    if str(r).strip().lower()=="auto":return "auto"
    return int(r)

def _radius(e,shared):
    r=radius_spec(e,shared)
    return int(shared.get("overlap_radius_max",DEFAULT_OVERLAP_RADIUS)) if r=="auto" else r

def parse_sample(e,root,defaults,shared):
    cif=(root/e["structure"]).resolve()
    png=(root/e["reference"]).resolve()
    sfmt=e.get("structure_format",shared.get("structure_format"))
    atoms=read_structure(cif,sfmt);syms=atoms.get_chemical_symbols()
    if e.get("reference_format",shared.get("reference_format","image"))=="dft":
        return parse_dft_sample(e,root,defaults,shared,cif,png,sfmt,atoms)
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
    # The k-point dimension IS the lattice periodicity: a 3D crystal must
    # be written as Lattice 3, not as a one-cell-thick 2D slab.
    radius=_radius(e,shared)
    images=overlap_images_for(dim,radius)
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
      periodic_dim=dim,overlap_images=images,structure_format=sfmt,
      png_hash=hashlib.sha256(png.read_bytes()).hexdigest(),
      cif_hash=hashlib.sha256(cif.read_bytes()).hexdigest())

def parse_dft_sample(e,root,defaults,shared,cif,ref_path,sfmt,atoms):
    """Numerical DFT reference (.npz from dft_bands.save_reference).

    YAeHMOP runs on a path through the DFT segments; the DFT eigenvalues are
    interpolated at exactly those k points. Points on the connecting line of a
    discontinuity (X|U) are computed but excluded through s["kmask"].
    """
    syms=atoms.get_chemical_symbols()
    params={k:defaults[k] for k in set(syms)}
    norb=orbital_count(params,syms)
    nelec=valence_electrons(params,syms)
    if e.get("metallic",shared.get("metallic")):
        return parse_metal_sample(e,defaults,shared,cif,ref_path,sfmt,atoms,syms,norb,nelec)
    if nelec%2:raise ValueError("spinless YAeHMOP needs even nelec")
    nocc=nelec//2
    p=int(e.get("n_pairs",shared.get("n_pairs",2)))
    if p not in (1,2,3,4):raise ValueError("P5 N must be 1,2,3,4")
    if min(nocc,norb-nocc)<p:
        raise ValueError(f"{e['name']} requires {p} conduction/valence orbitals; EHT only has {norb-nocc} conduction")
    dim=int(e.get("point_dim",3))
    radius=_radius(e,shared)
    images=overlap_images_for(dim,radius)
    n=int(e.get("points_per_line",shared.get("points_per_line",20)))
    ref=load_reference(ref_path,atoms)
    er=eht_reference(ref,n,dim)
    dn=ref.nocc;T=er.energies
    if dn<p or T.shape[1]-dn<p:
        raise ValueError(f"{e['name']}: the DFT data has {dn} occupied and {T.shape[1]-dn} empty bands; "
                         f"{p}+{p} requested")
    target=T[:,dn-p:dn+p].copy()
    tv=float(target[:,p-1].max());tc=float(target[:,p].min())
    if tc-tv<.02:raise ValueError(e["name"]+": no positive DFT gap on the path")
    mid=.5*(tv+tc)
    return dict(name=e["name"],cif=cif,png=ref_path,symbols=syms,
      elements=list(dict.fromkeys(syms)),norb=norb,nelec=nelec,nocc=nocc,
      path=BandPath(tuple(er.labels),er.nodes,n),kx=er.x,kmask=er.mask,
      target=target-mid,target_all=T-mid,dft_nocc=dn,ticks=er.ticks,
      n_pairs=p,gap=tc-tv,weight=float(e.get("weight",1.)),
      periodic_dim=dim,overlap_images=images,structure_format=sfmt,
      png_hash=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
      cif_hash=hashlib.sha256(cif.read_bytes()).hexdigest())

def parse_metal_sample(e,defaults,shared,cif,ref_path,sfmt,atoms,syms,norb,nelec):
    """Metallic DFT reference: all bands relative to the DFT Fermi level, paired from the
    bottom with the EHT bands (objective 'metal')."""
    dim=int(e.get("point_dim",3))
    radius=_radius(e,shared)
    n=int(e.get("points_per_line",shared.get("points_per_line",20)))
    ref=load_reference(ref_path,atoms)
    if ref.fermi is None:raise ValueError(e["name"]+": metallic reference without Fermi level")
    er=eht_reference(ref,n,dim)
    T=er.energies-ref.fermi
    nb=min(T.shape[1],norb)
    return dict(name=e["name"],cif=cif,png=ref_path,symbols=syms,
      elements=list(dict.fromkeys(syms)),norb=norb,nelec=nelec,nocc=max(1,nelec//2),metal=True,
      path=BandPath(tuple(er.labels),er.nodes,n),kx=er.x,kmask=er.mask,
      target=T[:,:nb],target_all=T,dft_nocc=ref.nocc,dft_fermi=float(ref.fermi),ticks=er.ticks,
      n_pairs=int(e.get("n_pairs",shared.get("n_pairs",1))),gap=0.0,weight=float(e.get("weight",1.)),
      periodic_dim=dim,overlap_images=overlap_images_for(dim,radius),structure_format=sfmt,
      png_hash=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
      cif_hash=hashlib.sha256(cif.read_bytes()).hexdigest())

def metal_scores(bands,s,wset):
    """Metrics of the 'metal' objective and the shifted EHT bands (relative to E_F)."""
    E=bands[s["kmask"]]
    if E.shape!=(len(s["kx"]),s["norb"]):raise ValueError("physical EHT spectrum shape mismatch")
    m=metal_error(s["target_all"],E,s["kx"],[x for x,_ in s["ticks"]],wset)
    nb=s["target"].shape[1];sel=E[:,:nb]+m["shift_eV"]
    d=sel-s["target"];inside=(s["target"]>=wset.window_min)&(s["target"]<=wset.window_max)
    m.update(selected_rmse_eV=float(np.sqrt(np.mean(d[inside]**2))),gap_error_eV=0.0,
             two_rmse_eV=m["rms_within_1eV_of_EF_eV"],four_rmse_eV=m["rms_within_2eV_of_EF_eV"],
             nocc=s["nocc"],norb=s["norb"],metallic=True)
    return m,sel,E+m["shift_eV"]

def write_sample_input(s,inp,k):
    """Native periodic input for one sample, honouring its dimensionality."""
    return write_periodic_input_generic(s["cif"],inp,s["path"],k,True,s["nelec"],
        overlap_images=s["overlap_images"],periodic_dim=s["periodic_dim"],
        structure_format=s.get("structure_format"))

def scores(bands,s):
    n=s["nocc"]
    if s.get("kmask") is not None:
        if bands.shape!=(len(s["kmask"]),s["norb"]):
            raise ValueError("physical EHT spectrum shape mismatch")
        bands=bands[s["kmask"]]
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
            for el in read_structure((base/e["structure"]).resolve(),e.get("structure_format",config.get("structure_format"))).get_chemical_symbols():
                if el not in elements:elements.append(el)
        self.elements=elements
        self.bind,table=resolve_yaehmop_assets()
        self.stockparams=valid_basis(subset_for_elements(parse_parameter_table(table),elements))
        self.basis_add=parse_basis_add(config.get("basis_add"))
        if self.basis_add:
            before={el:set(v) for el,v in self.stockparams.items()}
            self.stockparams=extend_basis(self.stockparams,self.basis_add)
            self.basis_add={el:sorted(set(v)-before[el]-{"atnum","nval"}) for el,v in self.stockparams.items()
                            if set(v)-before[el]}
        self.start_from=config.get("start_table")
        if self.start_from:
            # Start (and centre the bounds and the regularization) at an earlier fit:
            # its shells replace the stock ones; shells it has and the stock lacks are added.
            st=parse_parameter_table(self.start_from)
            for el in self.stockparams:
                if el in st:
                    for orb in ("s","p","d","f"):
                        if orb in st[el]:
                            self.stockparams[el][orb]=dict(st[el][orb])
        self.basis_dz=parse_basis_add(config.get("basis_dz")) or {}
        if self.basis_dz:
            if self.basis_dz=="auto":raise ValueError("basis_dz needs explicit shells, e.g. S:p")
            # refining an earlier fit: start the diffuse tail with the smallest weight allowed by the
            # bounds (c2/c1 = 0.05), so that the first evaluation is close to the starting table
            self.stockparams=make_double_zeta(self.stockparams,self.basis_dz,
                                              ratio=0.05 if self.start_from else 0.3,
                                              refine=bool(self.start_from))
        self.objective_name=str(config.get("objective","frontier"))
        if self.objective_name not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        self.wset=WindowSettings.from_dict(config.get("objective_settings"))
        self.train=[parse_sample(e,base,self.stockparams,config) for e in self.train_entries]
        for s,e in zip(self.train,self.train_entries):
            s["wset"]=self.system_wset(e)
        self.specs,self.stock,self.bounds,self.scales=build_specs(self.stockparams,elements)
        if self.start_from and config.get("start_K") is not None:
            iK=[i for i,s in enumerate(self.specs) if s[2]=="K"][0]
            K0=float(config["start_K"]);self.stock[iK]=K0
            self.bounds=list(self.bounds);self.bounds[iK]=(max(1.0,K0-0.6),min(2.8,K0+0.6))
        self.bounds=list(self.bounds)
        for i,spec in enumerate(self.specs):
            if spec[2]=="Hii":
                self.bounds[i]=(self.stock[i]-5.,self.stock[i]+5.)
        self.names=[f"{el}_{orb}_{key}".strip("_") for el,orb,key in self.specs]
        self.pairs=int(config.get("n_pairs",2))
        # Metallic references always use the 'metal' objective; gapped DFT references in the
        # same (family) fit use 'window' when the objective is 'window' or 'metal'.
        if any(s.get("metal") for s in self.train) and self.objective_name=="frontier":
            raise ValueError("a metallic reference needs objective 'metal' (or 'window' for mixed families)")
        # systems are evaluated in parallel (each one is an independent YAeHMOP run) and, in the
        # population optimizers, `parallel` candidates at a time: workers x parallel <= CPUs
        ncpu=cpu_count()
        self.workers=max(1,min(len(self.train),int(config.get("workers",os.environ.get("BANDFIT_WORKERS",min(len(self.train),ncpu))))))
        par=config.get("parallel",os.environ.get("BANDFIT_PARALLEL","auto"))
        self.parallel=max(1,ncpu//self.workers) if str(par).strip().lower()=="auto" else max(1,int(par))
        self.lock=threading.Lock()
        self.fast={};self.fast_enabled=bool(config.get("fast_hamiltonian",True));self.fast_checked=False
        self.fast_evals=0;self.overlap_cache={};self.fast_pool=None
        if any(s["n_pairs"]!=self.pairs for s in self.train):
            raise ValueError("All train systems must use same n_pairs")
        # H(R), S(R) must stay exportable: a new best candidate is accepted only if S(k) is
        # positive definite (smallest eigenvalue >= overlap_min_eig) on the export mesh
        self.overlap_min_eig=float(config.get("overlap_min_eig",1e-3))
        self.best_checked=float("inf");self.overlap_rejected=0;self.overlap_checks=0
        self.gap_w=float(config.get("gap_weight",.35))
        self.reg_w=float(config.get("reg_weight",.04))
        self.calls=0
        self.radius_report={}
        self.k_eht0=float(self.stock[[i for i,s in enumerate(self.specs) if s[2]=="K"][0]])
        autos=[s for s,e in zip(self.train,self.train_entries) if radius_spec(e,config)=="auto"]
        if autos and not config.get("_probe"):  # a probe only counts the parameters
            with ThreadPoolExecutor(self.workers) as pool:list(pool.map(self.auto_radius,autos))
        if any(int(s[o].get("nzeta",1))==2 for s in self.stockparams.values() for o in ("s","p") if o in s):
            self.check_double_zeta_sp()
        # live progress for monitors (band_fit_gui.py): call, loss, best loss so far
        self.best_loss=float("inf")
        self.progress=open(out/"progress.csv","w",buffering=1)
        self.progress.write("call,loss,best\n")
    def check_double_zeta_sp(self):
        """The original YAeHMOP reads only the first exponent of an s/p shell: refuse to fit
        double-zeta s/p parameters with a binary that would silently ignore them."""
        import copy
        s=self.train[0];sz=copy.deepcopy(self.stockparams)
        for el in sz.values():
            for o in ("s","p"):
                if o in el and int(el[o].get("nzeta",1))==2:el[o]["c2"]=0.0
        dz=copy.deepcopy(self.stockparams)
        for el in dz.values():
            for o in ("s","p"):
                if o in el and int(el[o].get("nzeta",1))==2:el[o].update(zeta2=max(.5,.5*el[o]["zeta1"]),c2=.5)
        a=self.run(s,sz,1.75,"dzcheck_sz");b=self.run(s,dz,1.75,"dzcheck_dz")
        if np.allclose(a,b,atol=1e-6):
            raise RuntimeError(f"The YAeHMOP binary {self.bind} ignores the second exponent of s/p shells: "
                               "double-zeta p (basis_dz) needs a bind with double-zeta p support "
                               "(github.com/ac-dias/yaehmop, commit 742719c or later); set YAEHMOP_BIND to it")
    def system_wset(self,e):
        """Objective settings of one system: the global ones updated by its own objective_settings
        (e.g. segment_weights only for the slabs, whose k path differs from the bulk one)."""
        own=e.get("objective_settings") or {}
        if not own:return self.wset
        return WindowSettings.from_dict({**(self.conf.get("objective_settings") or {}),**own})
    def log_progress(self,cost,call=None):
        with self.lock:
            self.best_loss=min(self.best_loss,cost)
            self.progress.write(f"{self.calls if call is None else call},{cost:.6g},{self.best_loss:.6g}\n")
    def run(self,s,params,k,tag,exact=False,kpoints=False):
        """EHT bands of one system on its path. With S(k) cached for these exponents (fast stage of the
        alternating optimizer) the bands come from numpy (H = F(Hii,K) o S), else from bind."""
        if not exact and not kpoints and self.fast_enabled:
            c=self.fast.get(s["name"])
            if c is not None and c["key"]==self.slow_key(params):
                with self.lock:self.fast_evals+=1
                return self.fast_bands(c,params,k)
        w=self.work/s["name"];w.mkdir(exist_ok=True)
        native=w/(tag+".parms");inp=w/(tag+".bind");log=w/(tag+".log")
        write_parameter_file(native,{a:params[a] for a in s["elements"]})
        write_sample_input(s,inp,k)
        try:
            run_bind(self.bind,inp,native,log)
            kp,bands=parse_band(str(inp)+".band")
        finally:
            for tmp in w.glob(tag+"*"):
                if tmp.is_file():tmp.unlink()
        return (kp,bands) if kpoints else bands
    # ---------------- cached S(k): H(k) = F(Hii,K) o S(k) when only Hii and K change ----------------
    @staticmethod
    def slow_key(params):
        """Everything that changes S(k): exponents and contraction coefficients of every shell."""
        return tuple((el,o,int(sp[o].get("nzeta",1)),float(sp[o]["zeta1"]),float(sp[o].get("zeta2",0)),
                      float(sp[o].get("c1",1)),float(sp[o].get("c2",0)))
                     for el,sp in sorted(params.items()) for o in ("s","p","d","f") if o in sp)
    @staticmethod
    def orbital_hii(s,params):
        deg={"s":1,"p":3,"d":5,"f":7}
        return np.array([params[el][o]["Hii"] for el in s["symbols"] for o in ("s","p","d","f")
                         if o in params[el] for _ in range(deg[o])],float)
    @staticmethod
    def hamiltonian_k(S,hii,K):
        """YAeHMOP H(k): weighted Wolfsberg-Helmholz off the diagonal (THE CONST = K), times S_ij(k);
        diagonal Hii (1 - Kd + Kd S_ii(k)) (R_hamil.c / K_hamil.c)."""
        a=hii[:,None];b=hii[None,:];t=a+b
        with np.errstate(divide="ignore",invalid="ignore"):d2=np.where(t!=0,((a-b)/t)**2,0.)
        F=.5*(K+d2+d2*d2*(1-K))*t
        np.fill_diagonal(F,THE_CONST_DIAG*hii)
        H=F[None]*S
        idx=np.arange(len(hii));H[:,idx,idx]+=(1-THE_CONST_DIAG)*hii
        return H
    def fast_bands(self,c,params,K):
        H=self.hamiltonian_k(c["S"],self.orbital_hii(c["s"],params),float(K))
        A=c["Linv"]@H@np.conj(np.swapaxes(c["Linv"],1,2))
        return np.linalg.eigvalsh(A)
    def fast_process_pool(self,xs):
        """Process pool for a batch that lies entirely on the cached S(k), else None."""
        if not self.fast or self.parallel<2 or len(xs)<2:return None
        key=next(iter(self.fast.values()))["key"]
        try:
            if any(self.slow_key(apply_x(self.stockparams,self.specs,np.asarray(x,float))[0])!=key for x in xs):return None
        except Exception:
            return None
        if self.fast_pool is not None and self.fast_pool[0]==key:return self.fast_pool[1]
        self.close_fast_pool()
        import multiprocessing as mp
        try:ctx=mp.get_context("fork")
        except ValueError:return None
        global _FAST_FITTER
        _FAST_FITTER=self
        self.fast_pool=(key,ctx.Pool(max(1,min(self.parallel*self.workers,cpu_count())),initializer=_fast_init))
        return self.fast_pool[1]
    def close_fast_pool(self):
        if self.fast_pool is not None:
            self.fast_pool[1].terminate();self.fast_pool[1].join();self.fast_pool=None
    def prepare_fast(self,x):
        """Cache S(k) (and its Cholesky inverse) of every training system at the exponents of x.
        The first time, the numpy bands are checked against bind (also with Hii and K moved);
        if they disagree (e.g. a bind with THE CONST hardwired) the fast path is switched off."""
        if not self.fast_enabled:return False
        params,K=apply_x(self.stockparams,self.specs,np.asarray(x,float))
        key=self.slow_key(params)
        if all(self.fast.get(s["name"],{}).get("key")==key for s in self.train):return True
        def one(s):
            tag=f"fast_{self.calls:07d}"
            kp,ref=self.run(s,params,K,tag+"_band",kpoints=True)
            w=self.work/s["name"];native=w/(tag+".parms");inp=w/(tag+".bind")
            write_parameter_file(native,{a:params[a] for a in s["elements"]})
            write_periodic_input_generic(s["cif"],inp,None,K,True,s["nelec"],overlap_images=s["overlap_images"],
                periodic_dim=s["periodic_dim"],structure_format=s.get("structure_format"),kpoints=kp)
            try:
                run_bind(self.bind,inp,native,w/(tag+".log"))
                S=read_matrix_dump(str(inp)+".OV")
            finally:
                for tmp in w.glob(tag+"*"):
                    if tmp.is_file():tmp.unlink()
            L=np.linalg.cholesky(S)
            c=dict(key=key,s=s,S=S,Linv=np.linalg.inv(L),kpts=kp)
            return c,float(np.abs(self.fast_bands(c,params,K)-ref).max())
        try:
            with ThreadPoolExecutor(self.workers) as pool:out=list(pool.map(one,self.train))
        except (np.linalg.LinAlgError,RuntimeError,ValueError) as exc:
            print("FAST_HAMILTONIAN unavailable for these exponents:",exc,flush=True)
            return False
        err=max(e for _,e in out)
        if not self.fast_checked:
            # independent check with other Hii and K (same exponents) against bind
            p2,K2=apply_x(self.stockparams,self.specs,np.asarray(x,float))
            for el in p2:
                for o in ("s","p","d","f"):
                    if o in p2[el]:p2[el][o]["Hii"]=float(p2[el][o]["Hii"])-.37*(1+("spdf".index(o)))
            K2=float(K)+.11
            for (c,_),s in zip(out,self.train):
                b=self.run(s,p2,K2,f"fastchk_{self.calls:07d}",exact=True)
                err=max(err,float(np.abs(self.fast_bands(c,p2,K2)-b).max()))
            self.fast_checked=True
            self.fast_check_error=err
            if err>2e-3:
                print(f"FAST_HAMILTONIAN disabled: numpy bands differ from bind by {err:.2e} eV",flush=True)
                self.fast_enabled=False;self.fast={}
                return False
            print(f"FAST_HAMILTONIAN on: H = F(Hii,K) o S(k), max |numpy - bind| = {err:.1e} eV",flush=True)
        for c,_ in out:self.fast[c["s"]["name"]]=c
        return True
    # ---------------- automatic overlap radius ----------------
    def check_final_radius(self,s,params,K,full):
        """The fitted table may have more diffuse orbitals than those tested before the fit:
        compare its bands with those at overlap_radius_max and report (warn above 10 x tol)."""
        rep=self.radius_report[s["name"]];r=rep["radius"];rmax=rep["max_radius"]
        if r>=rmax:rep["final_check_error_eV"]=0.0;return
        orig=s["overlap_images"]
        try:
            s["overlap_images"]=overlap_images_for(s["periodic_dim"],rmax)
            ref=self.run(s,params,K,"final_radius_check",exact=True)
        finally:
            s["overlap_images"]=orig
        n=max(1,min(s["nocc"],ref.shape[1]));keep=ref.min(0)<=ref[:,n-1].max()+8.
        err=float(np.abs(full[:,keep]-ref[:,keep]).max());rep["final_check_error_eV"]=err
        if err>10*rep["tolerance_eV"]:
            print(f"WARNING {s['name']}: with the fitted table, radius {r} differs from radius {rmax} by {err:.2e} eV "
                  f"(more diffuse orbitals than tested); the exported H(R), S(R) use radius {r} consistently, "
                  f"but refit with overlap_radius {r+1} or more for converged lattice sums",flush=True)
    def auto_radius(self,s):
        """Smallest overlap radius whose bands (start exponents and 20% more diffuse ones) agree with
        those of overlap_radius_max within overlap_radius_tol eV, for the bands up to 8 eV above the
        highest occupied one (the fitted window lies well inside)."""
        import copy
        rmax=int(self.conf.get("overlap_radius_max",DEFAULT_OVERLAP_RADIUS))
        tol=float(self.conf.get("overlap_radius_tol",1e-3))
        diffuse=copy.deepcopy(self.stockparams)
        for sp in diffuse.values():
            for o in ("s","p","d","f"):
                if o in sp:
                    sp[o]["zeta1"]=.8*float(sp[o]["zeta1"])
                    if int(sp[o].get("nzeta",1))==2:sp[o]["zeta2"]=.8*float(sp[o]["zeta2"])
        dim=s["periodic_dim"];orig=s["overlap_images"];errs={}
        def bands(params,r):
            s["overlap_images"]=overlap_images_for(dim,r)
            return self.run(s,params,self.k_eht0,f"radius{r}",exact=True)
        refs=[bands(p,rmax) for p in (self.stockparams,diffuse)]
        chosen=rmax
        for r in range(1,rmax):
            err=0.
            for p,ref in zip((self.stockparams,diffuse),refs):
                n=max(1,min(s["nocc"],ref.shape[1]))
                keep=ref.min(0)<=ref[:,n-1].max()+8.
                err=max(err,float(np.abs(bands(p,r)[:,keep]-ref[:,keep]).max()))
            errs[r]=err
            if err<tol:chosen=r;break
        s["overlap_images"]=overlap_images_for(dim,chosen)
        self.radius_report[s["name"]]=dict(radius=chosen,max_radius=rmax,tolerance_eV=tol,
                                           errors_eV={str(k):v for k,v in errs.items()})
        print(f"OVERLAP_RADIUS {s['name']}: auto -> {chosen} (max {rmax}, tol {tol:g} eV, errors "
              +", ".join(f"r={k}: {v:.1e}" for k,v in errs.items())+")",flush=True)
        return chosen
    @staticmethod
    def overlap_kpoints(s,radius):
        """Every k point where the export evaluates S(k): the Gamma mesh and the band path."""
        nodes=np.asarray(s["path"].points,float);n=int(s["path"].points_per_line);ks=[]
        for a,b in zip(nodes[:-1],nodes[1:]):
            ks+=[a+(b-a)*i/n for i in range(n)]
        ks.append(nodes[-1])
        return np.vstack([gamma_mesh(s["periodic_dim"],radius),np.array(ks)])
    def overlap_min(self,s,params,k):
        """Smallest eigenvalue of S(k) at the k points of the H(R),S(R) export (Gamma mesh + band path)."""
        w=self.work/s["name"];w.mkdir(exist_ok=True)
        tag=f"ovl_{self.calls:07d}_{threading.get_ident()}";native=w/(tag+".parms");inp=w/(tag+".bind")
        write_parameter_file(native,{a:params[a] for a in s["elements"]})
        radius=int(max(s["overlap_images"]))
        write_periodic_input_generic(s["cif"],inp,None,k,True,s["nelec"],overlap_images=s["overlap_images"],
            periodic_dim=s["periodic_dim"],structure_format=s.get("structure_format"),
            kpoints=self.overlap_kpoints(s,radius))
        try:
            run_bind(self.bind,inp,native,w/(tag+".log"))
            S=read_matrix_dump(str(inp)+".OV")
            return float(min(np.linalg.eigvalsh(Sk).min() for Sk in S))
        finally:
            for tmp in w.glob(tag+"*"):
                if tmp.is_file():tmp.unlink()
    def next_call(self):
        with self.lock:
            self.calls+=1
            return self.calls
    def objective(self,x,eid,stage):
        call=self.next_call()
        try:
            cost,line,params,K=self.evaluate(x,call)
            return self.accept(cost,line,params,K)
        except Exception as exc:
            self.log_progress(1e4,call)
            return 1e4,{"error":str(exc)}
    def objective_many(self,xs,eids,stage):
        """Several candidates at once (a population): the YAeHMOP runs go in parallel, then the
        candidates are accepted in order, exactly as one by one (the overlap check is sequential)."""
        calls=[self.next_call() for _ in xs]
        def ev(a):
            try:return self.evaluate(a[0],a[1])
            except Exception as exc:return exc
        pool=self.fast_process_pool(xs)
        if pool is not None:
            # every candidate is on the cached S(k): numpy only, in forked processes (threads are
            # limited by the GIL in the scoring code)
            outs=pool.map(_fast_child,list(zip(xs,calls)))
            with self.lock:self.fast_evals+=len(xs)*len(self.train)
        else:
            with ThreadPoolExecutor(max(1,min(self.parallel,len(xs)))) as tp:outs=list(tp.map(ev,zip(xs,calls)))
        res=[]
        for o,call in zip(outs,calls):
            if isinstance(o,Exception):
                self.log_progress(1e4,call);res.append((1e4,{"error":str(o)}));continue
            try:res.append(self.accept(*o,call=call))
            except Exception as exc:
                self.log_progress(1e4,call);res.append((1e4,{"error":str(exc)}))
        return res
    def evaluate(self,x,call):
        """Objective of one candidate without the overlap check: (cost, diagnostics, params, K)."""
        params,K=apply_x(self.stockparams,self.specs,np.asarray(x,float))
        if not physical_ok(params):raise ValueError("Invalid double-zeta parameter")
        values=[];weights=[]
        line=dict(K=float(K))
        tag=f"trial_{call:07d}"
        if self.workers>1 and len(self.train)>1:
            with ThreadPoolExecutor(min(self.workers,len(self.train))) as pool:
                fulls=list(pool.map(lambda s:self.run(s,params,K,tag),self.train))
        else:
            fulls=[self.run(s,params,K,tag) for s in self.train]
        for s,full in zip(self.train,fulls):
            if s.get("metal"):
                wm,_,_=metal_scores(full,s,s.get("wset",self.wset))
                values.append(wm["total"]);weights.append(s["weight"])
                line[s["name"]+"_window_rms_eV"]=wm["window_rms_eV"]
                line[s["name"]+"_rms_2eV_EF"]=wm["rms_within_2eV_of_EF_eV"]
                continue
            m,_=scores(full,s)
            if self.objective_name in ("window","metal") and s.get("target_all") is not None:
                wm=self.window_metrics(full,s)
                val=wm["total"]
                line[s["name"]+"_window_rms_eV"]=wm["window_rms_eV"]
            else:
                val=m["selected_rmse_eV"]+self.gap_w*abs(m["gap_error_eV"])
            values.append(val);weights.append(s["weight"])
            line[s["name"]+"_selected_rmse_eV"]=m["selected_rmse_eV"]
            line[s["name"]+"_two_rmse_eV"]=m["two_rmse_eV"]
            line[s["name"]+"_four_rmse_eV"]=m["four_rmse_eV"]
            line[s["name"]+"_gap_error_eV"]=m["gap_error_eV"]
        reg=float(np.sqrt(np.mean(((np.asarray(x)-self.stock)/self.scales)**2)))
        cost=float(np.average(values,weights=weights)+self.reg_w*reg)
        line["regularization"]=reg
        return cost,line,params,K
    def accept(self,cost,line,params,K,call=None):
        """Overlap check of a would-be best candidate (S(k) positive definite on the export mesh);
        S depends only on the exponents, so the result is cached per exponent set."""
        call=self.calls if call is None else call
        if self.overlap_min_eig>0 and cost<self.best_checked:
            key=self.slow_key(params)
            smin=self.overlap_cache.get(key)
            if smin is None:
                self.overlap_checks+=1
                if self.workers>1 and len(self.train)>1:
                    with ThreadPoolExecutor(min(self.workers,len(self.train))) as pool:
                        smin=min(pool.map(lambda s:self.overlap_min(s,params,K),self.train))
                else:
                    smin=min(self.overlap_min(s,params,K) for s in self.train)
                self.overlap_cache[key]=smin
            line["overlap_min_eig"]=smin
            if smin<self.overlap_min_eig:
                self.overlap_rejected+=1
                self.log_progress(1e4,call)
                return 1e4,{"error":f"overlap S(k) not positive definite enough (min eigenvalue {smin:.2e}): "
                                     "H(R), S(R) could not be exported"}
            self.best_checked=cost
        self.log_progress(cost,call)
        return cost,line
    def window_metrics(self,full,s):
        if s.get("target_all") is None:
            raise ValueError("objective 'window' needs a numerical DFT reference")
        E=full[s["kmask"]]
        return window_error(s["target_all"],s["dft_nocc"],E,s["nocc"],s["kx"],
                            [x for x,_ in s["ticks"]],s.get("wset",self.wset))
    def characterize(self,x,withheld):
        params,K=apply_x(self.stockparams,self.specs,np.asarray(x,float))
        specs=list(self.train)
        if withheld:
            # Heldout CIF and PNG are opened ONLY after the optimizer freezes x.
            for e in self.test_entries:
                ats=read_structure((self.base/e["structure"]).resolve(),e.get("structure_format",self.conf.get("structure_format")))
                if not set(ats.get_chemical_symbols()).issubset(self.elements):
                    raise ValueError("Unseen test elements cannot be transferred")
                specs.append(parse_sample(e,self.base,self.stockparams,self.conf))
                specs[-1]["wset"]=self.system_wset(e)
                if radius_spec(e,self.conf)=="auto":self.auto_radius(specs[-1])
        result={}
        for s in specs:
            full=self.run(s,params,K,"final_eigen",exact=True)
            if s["name"] in self.radius_report:self.check_final_radius(s,params,K,full)
            if s.get("metal"):
                met,b,Eall=metal_scores(full,s,s.get("wset",self.wset))
                met["window"]=dict(met)
                met["role"]="train" if s["name"] in [t["name"] for t in self.train] else "test"
                result[s["name"]]=met
                np.savez_compressed(self.out/(s["name"]+"_bands.npz"),k=s["kx"],reference=s["target"],
                    YAeHMOP=b,reference_all=s["target_all"],YAeHMOP_all=Eall,
                    tick_x=np.array([x for x,_ in s["ticks"]],float),
                    tick_labels=np.array([l for _,l in s["ticks"]]),metallic=True,
                    dft_fermi=s["dft_fermi"],eht_fermi=met["eht_fermi_eV"],
                    window=np.array([s.get("wset",self.wset).window_min,s.get("wset",self.wset).window_max]))
                continue
            met,b=scores(full,s)
            if s.get("target_all") is not None:
                met["window"]=self.window_metrics(full,s)
            met["role"]="train" if s["name"] in [t["name"] for t in self.train] else "test"
            result[s["name"]]=met
            extra={}
            if s.get("target_all") is not None:
                # Whole spectra for the full comparison figure, each relative to its midgap
                E=full[s["kmask"]];n=s["nocc"]
                emid=.5*(float(E[:,n].min())+float(E[:,n-1].max()))
                extra=dict(reference_all=s["target_all"],YAeHMOP_all=E-emid,
                           tick_x=np.array([x for x,_ in s["ticks"]],float),
                           tick_labels=np.array([l for _,l in s["ticks"]]),
                           dft_nocc=s["dft_nocc"],eht_nocc=n,n_pairs=s["n_pairs"])
            np.savez_compressed(self.out/(s["name"]+"_bands.npz"),k=s["kx"],
                                reference=s["target"],YAeHMOP=b,**extra)
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
    par=dict(updating="immediate",workers=1)
    if f.parallel>1:
        # the DE population in parallel (scipy needs 'deferred' updating for that)
        def many(func,xs):
            xs=[np.asarray(v,float) for v in xs]
            outs=f.objective_many(xs,None,"DE")
            for x,(val,meta) in zip(xs,outs):
                hist.append({"eval":len(hist)+1,"stage":"DE","loss":float(val),**meta})
                if val<best[0]:best[:]=[float(val),x.copy()]
            return [v for v,_ in outs]
        par=dict(updating="deferred",workers=many)
    differential_evolution(lambda x:ev(x,"DE"),f.bounds,
            seed=seed,maxiter=iterations,popsize=pop,polish=False,**par)
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

def global_kwargs(f,config):
    """Parallel population and alternating-stage settings for run_global."""
    return dict(objective_many=f.objective_many if f.parallel>1 else None,parallel=f.parallel,
                alternating=dict(fast_idx=[i for i,s in enumerate(f.specs) if s[2] in FAST_KEYS],
                                 prepare=f.prepare_fast,fast_budget=config.get("fast_budget"),
                                 rounds=int(config.get("alternating_rounds",3))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--outdir",required=True)
    p.add_argument("--initialization",choices=("staged","existing","transferred"),default="staged")
    p.add_argument("--start-json")
    p.add_argument("--optimizer",choices=("nelder-mead","simulated-annealing",
                   "sa-nelder-mead","hooke-jeeves")+GLOBAL_METHODS,default="nelder-mead",
                   help="staged: nelder-mead = short DE + NM; a global method runs from the stock "
                        "parameters with the whole budget. existing/transferred: refinement method")
    p.add_argument("--polish",choices=POLISH_METHODS,default=None,
                   help="local refinement after a global method, inside the same budget")
    p.add_argument("--polish-fraction",type=float,default=.25,
                   help="share of the budget given to --polish (default 0.25)")
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
    if args.polish and args.optimizer not in GLOBAL_METHODS:
        p.error("--polish is only used with "+", ".join(GLOBAL_METHODS))
    if not 0<args.polish_fraction<1:
        p.error("--polish-fraction must be between 0 and 1")
    if args.validate_only:
        print(json.dumps(dict(train=names,test=tests,test_files_opened=False)))
        return
    out=Path(args.outdir).resolve();out.mkdir(exist_ok=True,parents=True)
    f=Fitter(config,cfgfile.parent,out)
    if args.initialization=="staged" and args.optimizer in GLOBAL_METHODS:
        result=run_global(args.optimizer,f.objective,f.stock,f.bounds,args.budget,
                          seed=args.seed,polish_method=args.polish,
                          polish_fraction=args.polish_fraction,**global_kwargs(f,config))
        x=result["best_x"] if result["best_x"] is not None else f.stock
        start_meta=None
    elif args.initialization=="staged":
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
        if args.optimizer in GLOBAL_METHODS:
            result=run_global(args.optimizer,f.objective,x0,f.bounds,args.budget,
                              seed=args.seed,polish_method=args.polish,
                              polish_fraction=args.polish_fraction,**global_kwargs(f,config))
        else:
            result=options[args.optimizer](f.objective,x0,f.bounds,**kw)
            result["optimizer"]=args.optimizer
        # Freeze at global best TRAINING objective found in either stage;
        # do not use the heldout reference to choose this vector.
        compare_loss=(baseline_loss if baseline_loss is not None else
                      prior["best_training_loss"])
        x=(result["best_x"] if result["best_x"] is not None and
           result["best_loss"]<compare_loss else x0)
        start_meta=str(Path(args.start_json).resolve())
    f.close_fast_pool()
    met=f.characterize(x,not args.skip_test_eval)
    state=dict(loss_profile=f"sparse_{2*f.pairs}",n_pairs=f.pairs,
      image_mode=config.get("image_mode","auto"),initializer=args.initialization,
      adaptation_prior_material_was_heldout=(
           args.initialization=="transferred"),
      optimizer=result["optimizer"],optimizer_message=result.get("message"),
      polish=result.get("polish"),train_systems=names,test_systems=tests,
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
      metrics=met,n_params=len(f.names),
      overlap_radius=(lambda r:r if r=="auto" else int(r))(radius_spec({},config)),
      overlap_radius_by_system={s["name"]:int(max(s["overlap_images"])) for s in f.train},
      overlap_radius_auto=f.radius_report or None,
      parallel=dict(systems=f.workers,candidates=f.parallel),
      fast_hamiltonian=dict(enabled=f.fast_enabled,evaluations=f.fast_evals,
                            check_error_eV=getattr(f,"fast_check_error",None)) if f.fast_checked else None,
      lattice={s["name"]:dict(periodic_dim=s["periodic_dim"],
                              overlap_images=list(s["overlap_images"]))
               for s in f.train},
      structure_formats={s["name"]:resolved_structure_format(s["cif"],s.get("structure_format"))
                         for s in f.train},
      objective=f.objective_name,objective_settings=f.wset.to_dict() if f.objective_name in ("window","metal") else None,
      basis_added=f.basis_add or None,basis_double_zeta=f.basis_dz or None,start_from=str(f.start_from) if f.start_from else None,
      overlap_check=dict(min_eigenvalue_required=f.overlap_min_eig,checks=f.overlap_checks,
                         candidates_rejected=f.overlap_rejected))
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
