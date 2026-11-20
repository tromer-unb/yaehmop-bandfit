#!/usr/bin/env python3
"""User-facing original-YAeHMOP structure + PNG fitter, figure writer and band runner.

python3 bands_fit.py MoS2.cif file.txt --png MoS2.png --outdir runs/mos2
python3 bands_fit.py MoS2.cif file.txt --from-fit runs/mos2/fit/fit.json
python3 bands_fit.py MoS2.cif file.txt --from-parameters parameters.dat --k 1.75

file.txt is ordinary INI with [reference], [band_path], [fit] sections.
The structure may be a CIF or a VASP (POSCAR/CONTCAR/OUTCAR/vasprun.xml),
Quantum ESPRESSO (pw.x input/output), SIESTA (.fdf/.XV/.STRUCT_OUT) or
FHI-aims (geometry.in/output) file; see --structure-format.
This is a standalone frontend, NOT a new approximate Hamiltonian.
"""
from pathlib import Path
import argparse,configparser,csv,hashlib,json,os,subprocess,sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"parameter_fitter"),str(ROOT/"source"),str(HERE)]
from prepare_band_png import prepare,parse_bbox
from yaehmop_generic import (resolve_yaehmop_assets,parse_parameter_table,
   subset_for_elements,valence_electrons,orbital_count,BandPath,
   write_periodic_input_generic,run_bind,parse_band,normalized_path_coordinate,
   DEFAULT_OVERLAP_RADIUS,overlap_images_for,write_parameter_file,orbital_labels,
   read_matrix_dump)
from fit_from_image import parse_points
from global_optimizers import GLOBAL_METHODS,POLISH_METHODS
from structure_io import (read_structure,resolved_structure_format,structure_label,
   STRUCTURE_FORMAT_CHOICES,STRUCTURE_FORMAT_HELP)
from dft_bands import (DFT_CODES,read_dft_bands,save_reference,load_reference,
   eht_reference,detect_periodic_dim,normalize_code)

def csv_path(x,base):
    p=Path(os.path.expandvars(os.path.expanduser(x)))
    return (p if p.is_absolute() else (base/p)).resolve()

def conf(path,required=("reference","band_path","fit")):
    c=configparser.ConfigParser(interpolation=None)
    if not Path(path).is_file() or not c.read(path,encoding="utf-8"):
        raise FileNotFoundError("Missing input file.txt: "+str(path))
    for s in required:
        if s not in c:raise ValueError(f"Missing [{s}] in input file")
    for s in ("reference","band_path","fit"):
        if s not in c:c.add_section(s)
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

REF_COLOR="#1f5fbf"   # reference (DFT / image): blue, solid
EHT_COLOR="#d62728"   # extended Hueckel (YAeHMOP): red, dashed

def _ticks(ax,d):
    if "tick_x" in d and len(d["tick_x"]):
        xs=[float(x) for x in d["tick_x"]];labs=[str(l) for l in d["tick_labels"]]
        ax.set_xticks(xs);ax.set_xticklabels(labs)
        for x in xs:ax.axvline(x,color="0.75",lw=.6,zorder=0)
        ax.set_xlabel("Wave vector")
    else:
        ax.set_xlabel("Normalized k-path length")
    ax.set_xlim(0,1)

def fit_errors(ref,E):
    """RMSE, MAE, max |dE| and the gaps of the 2p fitted bands (both midgap-aligned)."""
    n=ref.shape[1];p=n//2;d=E-ref
    gr=float(ref[:,p].min()-ref[:,p-1].max());ge=float(E[:,p].min()-E[:,p-1].max())
    return dict(rmse=float(np.sqrt(np.mean(d**2))),mae=float(np.mean(np.abs(d))),
                maxabs=float(np.max(np.abs(d))),gap_ref=gr,gap_eht=ge,gap_err=ge-gr)

def _footer(fig,ax,m,n,ref_name,note=None):
    """Legend and fit errors in a band below the axes (never over the curves)."""
    h,l=ax.get_legend_handles_labels();keep=dict(zip(l,h))
    fig.tight_layout(rect=(0,.15,1,1))
    ax.legend(keep.values(),keep.keys(),frameon=False,loc="upper left",
              bbox_to_anchor=(0,-.11),ncol=1,fontsize=9.5)
    txt=(f"Fit error ({n} bands):  RMSE = {m['rmse']:.3f} eV   MAE = {m['mae']:.3f} eV   "
         f"max |ΔE| = {m['maxabs']:.3f} eV\n"
         f"Gap: {ref_name} {m['gap_ref']:.3f} eV,  EHT {m['gap_eht']:.3f} eV,  "
         f"error {m['gap_err']:+.3f} eV")
    if note:txt+="\n"+note
    fig.text(.985,.015,txt,ha="right",va="bottom",fontsize=9,
             bbox=dict(boxstyle="round,pad=.45",fc="white",ec="0.6"))

def write_bands_plot(npz,out_dir,n,*,title,also_csv=True,ref_name=None):
    """Fitted bands: reference blue (solid), EHT red (dashed), errors in the figure."""
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    d=np.load(npz)
    k=np.asarray(d["k"],float)
    ref=np.asarray(d["reference"],float)
    E=np.asarray(d["YAeHMOP"],float)
    if E.shape!=ref.shape or E.shape!=(len(k),n):
        raise ValueError("Stored fit/reference arrays do not have the expected N-band shape")
    if ref_name is None:
        ref_name="DFT" if "reference_all" in d else "Reference"
    m=fit_errors(ref,E)
    fig,ax=plt.subplots(figsize=(9.7,6.4))
    for j in range(n):
        ax.plot(k,ref[:,j],color=REF_COLOR,lw=1.6,
                label=f"{ref_name} (reference)" if j==0 else None)
        ax.plot(k,E[:,j],color=EHT_COLOR,lw=1.5,ls="--",
                label="EHT (YAeHMOP)" if j==0 else None)
    ax.axhline(0,color="0.6",lw=.5,ls=":")
    ax.set_title(title)
    _ticks(ax,d)
    ax.set_ylabel("Energy relative to midgap (eV)")
    ax.grid(alpha=.16)
    _footer(fig,ax,m,n,ref_name)
    fig.savefig(out_dir/"bands_comparison.png",dpi=220)
    fig.savefig(out_dir/"bands_comparison.pdf")
    plt.close(fig)
    if also_csv:
        with (out_dir/"bands_comparison.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["k"]+[f"REF_band_{j+1}" for j in range(n)]+
                       [f"EHT_band_{j+1}" for j in range(n)])
            w.writerows(np.column_stack((k,ref,E)))
    return m["rmse"]

def write_full_comparison(npz,out_dir,*,title,emin=None,emax=None,window_metrics=None):
    """All DFT bands (blue) and all EHT bands (red, dashed) in an energy window."""
    out_dir=Path(out_dir)
    d=np.load(npz)
    if "reference_all" not in d:return None
    k=np.asarray(d["k"],float);R=np.asarray(d["reference_all"],float);E=np.asarray(d["YAeHMOP_all"],float)
    ref=np.asarray(d["reference"],float);sel=np.asarray(d["YAeHMOP"],float)
    n=ref.shape[1];m=fit_errors(ref,sel)
    lo=-6. if emin is None else float(emin);hi=6. if emax is None else float(emax)
    fig,ax=plt.subplots(figsize=(9.7,7.0))
    for j in range(R.shape[1]):
        if R[:,j].max()<lo-1 or R[:,j].min()>hi+1:continue
        ax.plot(k,R[:,j],color=REF_COLOR,lw=1.3,label="DFT (reference)" if not ax.lines else None)
    first=True
    for j in range(E.shape[1]):
        if E[:,j].max()<lo-1 or E[:,j].min()>hi+1:continue
        ax.plot(k,E[:,j],color=EHT_COLOR,lw=1.2,ls="--",label="EHT (YAeHMOP)" if first else None);first=False
    ax.axhline(0,color="0.6",lw=.5,ls=":")
    ax.set_ylim(lo,hi);_ticks(ax,d)
    ax.set_ylabel("Energy relative to midgap (eV)")
    ax.set_title(title)
    note=f"bands of the figure above: {n//2} VB + {n//2} CB"
    if window_metrics:
        w=window_metrics
        note=(f"window objective: {w['n_valence_compared']} VB + {w['n_conduction_compared']} CB compared, "
              f"RMS in window = {w['window_rms_eV']:.3f} eV (VBM-aligned)")
    _footer(fig,ax,m,n,"DFT",note=note)
    fig.savefig(out_dir/"bands_full_comparison.png",dpi=220)
    fig.savefig(out_dir/"bands_full_comparison.pdf");plt.close(fig)
    with (out_dir/"bands_full_comparison.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["k"]+[f"DFT_band_{j+1}" for j in range(R.shape[1])]+[f"EHT_orbital_{j+1}" for j in range(E.shape[1])])
        w.writerows(np.column_stack((k,R,E)))
    return out_dir/"bands_full_comparison.png"

def write_metal_plots(npz,out_dir,*,title,emin=None,emax=None):
    """Metallic reference: bands relative to E_F (DFT blue, EHT red dashed), two figures:
    bands_comparison (the fitted window) and bands_full_comparison (wider range)."""
    out_dir=Path(out_dir);d=np.load(npz)
    k=np.asarray(d["k"],float);R=np.asarray(d["reference_all"],float);E=np.asarray(d["YAeHMOP_all"],float)
    ref=np.asarray(d["reference"],float);sel=np.asarray(d["YAeHMOP"],float)
    wlo,whi=[float(v) for v in d["window"]]
    inside=(ref>=wlo)&(ref<=whi);D=(sel-ref)[inside]
    near=inside&(np.abs(ref)<=1.0)
    m=dict(rmse=float(np.sqrt(np.mean(D**2))),mae=float(np.mean(np.abs(D))),maxabs=float(np.abs(D).max()),
           near=float(np.sqrt(np.mean(((sel-ref)[near])**2))) if near.any() else 0.0)
    for tag,(lo,hi) in (("bands_comparison",(wlo-.5,whi+.5)),
                        ("bands_full_comparison",(min(wlo,-6.) if emin is None else emin,whi+3 if emax is None else emax))):
        fig,ax=plt.subplots(figsize=(9.7,7.0 if tag.endswith("full_comparison") else 6.4))
        for j in range(R.shape[1]):
            if R[:,j].max()<lo-1 or R[:,j].min()>hi+1:continue
            ax.plot(k,R[:,j],color=REF_COLOR,lw=1.5,label="DFT (reference)" if j==0 else None)
        for j in range(E.shape[1]):
            if E[:,j].max()<lo-1 or E[:,j].min()>hi+1:continue
            ax.plot(k,E[:,j],color=EHT_COLOR,lw=1.4,ls="--",label="EHT (YAeHMOP)" if j==0 else None)
        ax.axhline(0,color="0.45",lw=.7,ls=":");ax.text(1.005,0,"E$_F$",transform=ax.get_yaxis_transform(),va="center",fontsize=9)
        if tag=="bands_full_comparison":
            ax.axhspan(wlo,whi,color="0.5",alpha=.06,lw=0)
        ax.set_ylim(lo,hi);_ticks(ax,d);ax.grid(alpha=.16)
        ax.set_ylabel("Energy relative to the Fermi level (eV)");ax.set_title(title)
        h,l=ax.get_legend_handles_labels();keep=dict(zip(l,h))
        fig.tight_layout(rect=(0,.13,1,1))
        ax.legend(keep.values(),keep.keys(),frameon=False,loc="upper left",bbox_to_anchor=(0,-.10),fontsize=9.5)
        txt=(f"Metal: bands aligned at E$_F$ (EHT shifted by the optimal constant)\n"
             f"Fit window [{wlo:+.1f}, {whi:+.1f}] eV:  RMSE = {m['rmse']:.3f} eV   MAE = {m['mae']:.3f} eV   "
             f"max |ΔE| = {m['maxabs']:.3f} eV\nwithin ±1 eV of E$_F$: RMSE = {m['near']:.3f} eV")
        fig.text(.985,.015,txt,ha="right",va="bottom",fontsize=9,bbox=dict(boxstyle="round,pad=.45",fc="white",ec="0.6"))
        fig.savefig(out_dir/f"{tag}.png",dpi=220);fig.savefig(out_dir/f"{tag}.pdf");plt.close(fig)
        with (out_dir/f"{tag}.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["k"]+[f"DFT_band_{j+1}" for j in range(R.shape[1])]+[f"EHT_orbital_{j+1}" for j in range(E.shape[1])])
            w.writerows(np.column_stack((k,R,E)))
    return m

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

def recompute(cif,config,parameters,K,outdir,overlap_radius=DEFAULT_OVERLAP_RADIUS,
              structure_format=None):
    """Recompute physical YAeHMOP eigenbands WITHOUT using any target pixels."""
    bind,_=resolve_yaehmop_assets()
    p=parse_parameter_table(parameters)
    atoms=read_structure(cif,structure_format)
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
    overlap=overlap_images_for(dim,overlap_radius)
    write_periodic_input_generic(cif,inp,bandpath,float(K),True,nelec,
                                 overlap_images=overlap,periodic_dim=dim,
                                 structure_format=structure_format)
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
    plot_full(k,E,outdir,name=structure_label(cif),nocc=nocc,emin=ymin,emax=ymax)
    np.savez_compressed(outdir/"bands_full.npz",k=k,energies=E,nocc=nocc)
    print("RECOMPUTED",len(k),"k points,",norb,
          "physical YAeHMOP eigenvalues per k from",parameters)
    print("FIGURE",outdir/"bands_full.png")
    return k,E

def node_positions(n_nodes,n,mask=None):
    """Index of each band-path node among the (masked) path k points."""
    idx=[j*n for j in range(n_nodes)]
    if mask is None:return idx
    pos=np.cumsum(mask)-1
    return [int(pos[i]) for i in idx if mask[i]]

def path_kpoints(nodes,n):
    """The k points YAeHMOP uses for a Band block (n per line, last node added)."""
    nodes=np.asarray(nodes,float);ks=[]
    for a,b in zip(nodes[:-1],nodes[1:]):
        ks += [a+(b-a)*i/n for i in range(n)]
    ks.append(nodes[-1])
    return np.array(ks)

def write_tb_header(fermi,cell,nbasis,nvec):
    """WanTiBEXOS header: NP, scissors, Fermi level, 3 lattice vectors, basis size, hopping vectors."""
    s=f"NP\n0.000000\n{fermi:.6f}\n"
    s+="".join(f"{v[0]:.8f}  {v[1]:.8f}  {v[2]:.8f}\n" for v in np.asarray(cell))
    return s+f"{nbasis}\n{nvec}\n"

def export_matrices(structure,sfmt,parameters,K,kpoints,dim,radius,fdir,node_index=None,energy_zero=None):
    """Text files with the YAeHMOP Hamiltonian and overlap of the final parameters.

    Two extra YAeHMOP runs ('K points' + Dump Hamil/Overlap): the full
    Gamma-centred mesh of (2r+1)^dim points, from which H(R) and S(R) of all
    cells within the overlap radius r are recovered exactly, and the k points
    of the figure (to check that H(R), S(R) reproduce YAeHMOP's H(k), S(k)
    and its bands). Written for WanTiBEXOS (huckel-NP.dat, basis_set-NP) and as
    plain tables of H(k), S(k) at the high-symmetry points.
    """
    import shutil,tempfile,scipy.linalg as sl
    from yaehmop_generic import real_space_matrices,gamma_mesh,ORBITAL_LM
    bind,_=resolve_yaehmop_assets()
    atoms=read_structure(structure,sfmt);syms=atoms.get_chemical_symbols()
    params=subset_for_elements(parse_parameter_table(parameters),list(dict.fromkeys(syms)))
    nelec=valence_electrons(params,syms);nocc=nelec//2
    images=overlap_images_for(dim,radius)
    mesh=gamma_mesh(dim,radius);kpoints=np.asarray(kpoints,float)
    def dump(ks,tmp,tag):
        write_parameter_file(tmp/"p.parms",params)
        inp=write_periodic_input_generic(structure,tmp/f"{tag}.bind",None,float(K),True,nelec,
            overlap_images=images,periodic_dim=dim,structure_format=sfmt,kpoints=ks)
        run_bind(bind,inp,tmp/"p.parms",tmp/f"{tag}.log")
        return read_matrix_dump(str(inp)+".HAM"),read_matrix_dump(str(inp)+".OV"),inp
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        Hm,Sm,_=dump(mesh,tmp,"mesh")
        Hp,Sp,inp=dump(kpoints,tmp,"path")
        shutil.copy(inp,Path(fdir)/"yaehmop_matrices.bind")
    cells=real_space_matrices(Hm,Sm,mesh,dim,radius)
    imag=max(max(np.abs(H.imag).max(),np.abs(S.imag).max()) for _,H,S in cells)
    if imag>1e-6:raise ValueError(f"H(R)/S(R) not real (max imaginary part {imag:.2e})")
    kept=[(R,H.real,S.real) for R,H,S in cells if max(np.abs(H).max(),np.abs(S).max())>1e-7]
    # Check: H(R), S(R) rebuild YAeHMOP's H(k), S(k) on the figure path
    def rebuild(q,which):
        return sum(np.exp(2j*np.pi*(q@R))*(H if which=="H" else S) for R,H,S in kept)
    errH=max(np.abs(rebuild(q,"H")-Hp[i]).max() for i,q in enumerate(kpoints))
    errS=max(np.abs(rebuild(q,"S")-Sp[i]).max() for i,q in enumerate(kpoints))
    # The matrices are written whatever happens; the eigenvalue checks need S(k) positive definite.
    smin=float(min(np.linalg.eigvalsh(S).min() for S in list(Sp)+list(Sm)))
    warning=None
    if smin<=1e-8:
        warning=(f"overlap S(k) is not positive definite (smallest eigenvalue {smin:.3e}): the basis is nearly "
                 "linearly dependent; H(R), S(R) were written but the generalized eigenproblem is ill-posed")
        print("WARNING",warning,flush=True)
        # eigenvalues in the subspace of S above the threshold (canonical orthogonalization)
        def geig(H,S):
            w,U=np.linalg.eigh(S);keep=w>1e-8*max(w.max(),1.)
            X=U[:,keep]/np.sqrt(w[keep]);e=np.linalg.eigvalsh(X.conj().T@H@X)
            return np.concatenate([e,np.full(H.shape[0]-len(e),np.inf)])
    else:
        geig=lambda H,S:sl.eigh(H,S,eigvals_only=True)
    eig=np.array([geig(Hp[i],Sp[i]) for i in range(len(kpoints))])
    eigm=np.array([geig(Hm[i],Sm[i]) for i in range(len(mesh))])
    allE=np.vstack([eig,eigm])
    metal=None
    if energy_zero is None:
        vbm=float(allE[:,nocc-1].max());cbm=float(allE[:,nocc].min());mid=.5*(vbm+cbm)
    else:
        # Metal: the zero is the EHT energy that matches the DFT Fermi level. States below it
        # on the (2r+1)^dim mesh = electrons per cell in this (spin) channel (one per state).
        vbm=cbm=None;mid=float(energy_zero)
        metal=dict(eht_fermi_eV=mid,states_below_EF_per_cell_mesh=float(np.sum(eigm<=mid)/len(mesh)),
                   mesh_note="Gamma-centred (2r+1)^dim mesh, equal weights, no smearing: a rough count")
    labels=orbital_labels(params,syms);n=len(labels)
    cell=np.asarray(atoms.cell)
    fdir=Path(fdir)
    # WanTiBEXOS non-orthogonal TB file (same layout as siesta2wtb.py, spinless 'NP').
    # wtb.x diagonalizes (H(k) - E_F*I, S(k)), which is not a rigid shift when S != I;
    # so H(R) is written as H(R) - E_mid*S(R) with E_F = 0: wtb.x then gives exactly
    # the YAeHMOP eigenvalues minus the EHT midgap (bands below 0 = occupied).
    with open(fdir/"huckel-NP.dat","w") as f:
        f.write(write_tb_header(0.0,cell,n,len(kept))+"#rcell x   rcell y   rcell z   i   j   ReH   ImH   S\n")
        for R,H,S in kept:
            r=R@cell;Hs=H-mid*S
            for i in range(n):
                for j in range(n):
                    f.write(f"{r[0]:.8f}   {r[1]:.8f}   {r[2]:.8f}   {i+1}   {j+1}   "
                            f"{Hs[i,j]:.10f}   0.0000000000   {S[i,j]:.10f}\n")
    pos=atoms.positions
    with open(fdir/"basis_set-NP","w") as f:
        f.write("bindex aspecie ax ay az l m spin\n")
        for b,lab in enumerate(labels):
            atom,orb=lab.split(":");ia=int("".join(c for c in atom if c.isdigit()))-1
            l,m=ORBITAL_LM[orb]
            f.write(f"{b+1}   {syms[ia]}   {pos[ia,0]:.8f}   {pos[ia,1]:.8f}   {pos[ia,2]:.8f}   {l}   {m}   0\n")
    with open(fdir/"orbitals.dat","w") as f:
        f.write("# index  atom  orbital  l  m   (YAeHMOP order)\n")
        for b,lab in enumerate(labels):
            atom,orb=lab.split(":");l,m=ORBITAL_LM[orb]
            f.write(f"{b+1:6d}  {atom:8s} {orb:12s} {l} {m:2d}\n")
    nodes=sorted(set(node_index or [0,len(kpoints)-1]))
    with open(fdir/"hamiltonian_overlap_k.dat","w") as f:
        # Header as in the WanTiBEXOS files (type, scissors, Fermi level, lattice
        # vectors [A], basis size, hopping vectors of huckel-NP.dat), then k points.
        # H(k) here is the raw YAeHMOP one, so the Fermi level is the EHT midgap.
        f.write(write_tb_header(mid,cell,n,len(kept))+f"{len(nodes)}\n"
                "#k1 k2 k3   i   j   ReH   ImH   ReS   ImS\n")
        for q in nodes:
            k=kpoints[q]
            for i in range(n):
                for j in range(n):
                    f.write(f"{k[0]:.8f} {k[1]:.8f} {k[2]:.8f}   {i+1}   {j+1}   {Hp[q][i,j].real:.10f}   "
                            f"{Hp[q][i,j].imag:.10f}   {Sp[q][i,j].real:.10f}   {Sp[q][i,j].imag:.10f}\n")
    return dict(n_orbitals=n,n_cells=len(kept),overlap_radius=int(radius),mesh_points=len(mesh),
                max_error_rebuilt_Hk_eV=float(errH),max_error_rebuilt_Sk=float(errS),
                eht_vbm_eV=vbm,eht_cbm_eV=cbm,eht_midgap_eV=mid,metal=metal,
                overlap_min_eigenvalue=smin,warning=warning,
                huckel_NP_energy_zero=("H(R) in huckel-NP.dat is H(R) - eht_midgap_eV*S(R); its Fermi level is 0"
                    if metal is None else "metal: H(R) - E0*S(R), E0 = eht_midgap_eV = EHT energy of the DFT Fermi level; Fermi level 0"),
                high_symmetry_k_in_hamiltonian_overlap_k=len(nodes))

FITTING_README="""Outputs of one EHT (YAeHMOP) fit (all plain text except the figures)
==================================================================
parameters_huckel.dat      fitted parameters in the native YAeHMOP table format
                           (use with: bind input.bind parameters_huckel.dat)
parameters_huckel.txt      K_EHT and every fitted variable (name value)
fit_summary.json           errors, gaps, optimizer, reference, structure, checks
bands_comparison.*         fitted bands: reference blue (solid), EHT red (dashed),
                           with RMSE / MAE / max |dE| / gap error in the figure
                           (.png, .pdf; .csv with the curves)
bands_full_comparison.*    (DFT references) all DFT and EHT bands in a window
huckel-NP.dat              real-space Hamiltonian H(R) and overlap S(R) in the
                           WanTiBEXOS format (as siesta2wtb.py; DFT= S):
                           NP / scissors / Fermi level (0) / 3 lattice vectors [A] /
                           n orbitals / n hopping vectors (cells) / then for every
                           cell and orbital pair: R_x R_y R_z [A]  i  j  ReH  ImH  S.
                           ReH is H(R) - E_mid*S(R) (E_mid = EHT midgap, in
                           fit_summary.json), so energies are relative to midgap and
                           wtb.x reproduces YAeHMOP's bands (wtb.x uses H - E_F*I,
                           which is not a shift when S != I).
basis_set-NP               WanTiBEXOS basis file: index species x y z [A] l m spin
orbitals.dat               orbital index, atom, orbital, l, m (YAeHMOP order)
hamiltonian_overlap_k.dat  H(k), S(k) at the high-symmetry points of the figure.
                           Header as huckel-NP.dat (Fermi level = EHT midgap,
                           absolute, since these H(k) are not shifted) plus the
                           number of k points; then k1 k2 k3 i j ReH ImH ReS ImS,
                           with H(k) = sum_R exp(i 2pi k.R) H(R)
yaehmop_matrices.bind      YAeHMOP input of the matrix run (K points + Dump)
POSCAR                     the structure used (VASP format, exact cell of huckel-NP.dat)
README.txt                 this description

H(R), S(R) are recovered exactly from YAeHMOP's H(k), S(k) on the full
(2r+1)^dim mesh (r = overlap radius); fit_summary.json reports how well they
rebuild YAeHMOP's H(k), S(k) on the band path (round-off level).
"""

def portable_paths(obj,fdir):
    """Absolute paths -> relative to fdir inside the repository, else file name only."""
    import os
    root=ROOT.resolve();fdir=Path(fdir).resolve()
    if isinstance(obj,dict):return {k:portable_paths(v,fdir) for k,v in obj.items()}
    if isinstance(obj,list):return [portable_paths(v,fdir) for v in obj]
    if isinstance(obj,str) and obj.startswith("/"):
        tail=""
        if " ..." in obj:obj,tail=obj.split(" ...",1);tail=" ..."+tail
        q=Path(obj)
        try:
            q.resolve().relative_to(root)
            return os.path.relpath(q.resolve(),fdir)+tail
        except ValueError:
            return q.name+tail
    return obj

def write_fitting_dir(fdir,outdir,*,structure,sfmt,native,param_txt,K,kpoints,dim,radius,summary,node_index=None,energy_zero=None):
    """Standard output folder of every fit: parameters, figures, H(k) and S(k)."""
    import shutil
    fdir=Path(fdir);fdir.mkdir(parents=True,exist_ok=True)
    shutil.copy(native,fdir/"parameters_huckel.dat")
    if param_txt is not None and Path(param_txt).is_file():
        shutil.copy(param_txt,fdir/"parameters_huckel.txt")
    else:
        (fdir/"parameters_huckel.txt").write_text(f"K_EHT {float(K):.12g}\n")
    for f in ("bands_comparison.png","bands_comparison.pdf","bands_comparison.csv",
              "bands_full_comparison.png","bands_full_comparison.pdf","bands_full_comparison.csv"):
        if (Path(outdir)/f).is_file() and Path(outdir).resolve()!=fdir.resolve():
            shutil.copy(Path(outdir)/f,fdir/f)
    for old_file in ("hamiltonian_overlap.npz","tb-NP.dat"):
        if (fdir/old_file).exists():(fdir/old_file).unlink()
    # The structure used (exact cell of huckel-NP.dat), so that fitting/ is self-contained
    from ase.io import write as ase_write
    ase_write(fdir/"POSCAR",read_structure(structure,sfmt),format="vasp",direct=True)
    info=export_matrices(structure,sfmt,native,K,kpoints,dim,radius,fdir,node_index,energy_zero)
    summary=dict(summary,structure_in_fitting="POSCAR",
                 tight_binding_files=dict(info,files=["huckel-NP.dat","basis_set-NP","orbitals.dat",
                                                              "hamiltonian_overlap_k.dat"]))
    summary=portable_paths(json.loads(json.dumps(summary,default=str)),fdir)
    (fdir/"fit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (fdir/"README.txt").write_text(FITTING_README)
    print("FITTING_DIR",fdir,flush=True)
    return fdir

def run_optimizer(target,outdir,optimizer,budget,polish,seed,parser):
    """Run fit_sparse_window on fit_input.yaml; returns (initialization, refinement) calls."""
    core=[sys.executable,str(ROOT/"parameter_fitter/fit_sparse_window.py"),str(target)]
    if optimizer in GLOBAL_METHODS:
        # One global search from the stock parameters with the WHOLE budget.
        base_calls=budget
        refinement_calls=0
        cmd=core+["--outdir",str(outdir/"fit"),"--initialization","staged",
                  "--optimizer",optimizer,"--budget",str(budget),"--seed",str(seed)]
        if polish:cmd+=["--polish",polish]
        subprocess.run(cmd,check=True,text=True)
    elif optimizer=="nelder-mead":
        # P5 staged mode is original YAeHMOP DE followed by bounded NM.
        base_calls=budget
        refinement_calls=0
        cmd=core+["--outdir",str(outdir/"fit"),"--initialization","staged",
                  "--budget",str(budget),"--seed",str(seed)]
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
                      "--seed",str(seed)],check=True,text=True)
        subprocess.run(core+["--outdir",str(outdir/"fit"),
                      "--initialization","existing",
                      "--start-json",str(outdir/"fit_base/fit.json"),
                      "--optimizer",str(optimizer),
                      "--budget",str(refinement_calls),"--seed",str(seed)],
                      check=True,text=True)
    return base_calls,refinement_calls

def optimizer_settings(args,fit,parser,check_budget=True):
    budget=int(args.budget or fit.get("budget","200"))
    optimizer=args.optimizer or fit.get("optimizer","nelder-mead")
    polish=args.polish or fit.get("polish",fallback=None) or None
    valid=("nelder-mead","simulated-annealing","sa-nelder-mead","hooke-jeeves")+GLOBAL_METHODS
    if optimizer not in valid:
        parser.error("optimizer must be one of: "+", ".join(valid))
    if polish and (optimizer not in GLOBAL_METHODS or polish not in POLISH_METHODS):
        parser.error("polish (nelder-mead or powell) only follows "+", ".join(GLOBAL_METHODS))
    if check_budget and budget<80:
        parser.error("Staged fit requires at least 80 YAeHMOP objective evaluations")
    return budget,optimizer,polish

def plot_dft_preview(ref,er,pairs,out,title):
    """QA figure of the DFT bands read from the folder, fitted ones highlighted."""
    T=er.energies;dn=ref.nocc
    if getattr(ref,"metallic",False):
        fig,ax=plt.subplots(figsize=(9.7,5.5))
        for j in range(T.shape[1]):ax.plot(er.x,T[:,j]-ref.fermi,color=REF_COLOR,lw=1.2)
        _ticks(ax,dict(tick_x=[x for x,_ in er.ticks],tick_labels=[l for _,l in er.ticks]))
        ax.set_ylim(-10,8);ax.axhline(0,color="0.6",lw=.5,ls=":")
        ax.set_ylabel("DFT energy relative to E$_F$ (eV)");ax.set_title(title+" (metal)")
        fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig);return
    mid=.5*(T[:,dn].min()+T[:,dn-1].max())
    fig,ax=plt.subplots(figsize=(9.7,5.5))
    for j in range(T.shape[1]):
        sel=dn-pairs<=j<dn+pairs
        ax.plot(er.x,T[:,j]-mid,color=REF_COLOR if sel else "0.7",lw=1.7 if sel else .8)
    _ticks(ax,dict(tick_x=[x for x,_ in er.ticks],tick_labels=[l for _,l in er.ticks]))
    ax.set_ylim(-6,6);ax.axhline(0,color="0.6",lw=.5,ls=":")
    ax.set_ylabel("DFT energy relative to midgap (eV)")
    ax.set_title(title+f" — bands to fit: {pairs} VB + {pairs} CB (blue)")
    fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)

def evaluate_dft(entry,parameters,K,outdir,name,pairs,window):
    """One YAeHMOP run with given parameters along the DFT path, with figures."""
    from fit_sparse_window import parse_sample,scores,write_sample_input
    from yaehmop_generic import write_parameter_file
    bind,_=resolve_yaehmop_assets()
    table=parse_parameter_table(parameters)
    syms=read_structure(entry["structure"],entry.get("structure_format")).get_chemical_symbols()
    params=subset_for_elements(table,list(dict.fromkeys(syms)))
    s=parse_sample(entry,outdir,params,{})
    work=outdir/"work";work.mkdir(parents=True,exist_ok=True)
    write_parameter_file(work/"eval.parms",params)
    inp=write_sample_input(s,work/"eval.bind",float(K))
    run_bind(bind,inp,work/"eval.parms",work/"eval.log")
    _,full=parse_band(str(inp)+".band")
    met,b=scores(full,s)
    E=full[s["kmask"]];n=s["nocc"]
    emid=.5*(float(E[:,n].min())+float(E[:,n-1].max()))
    npz=outdir/f"{name}_bands.npz"
    np.savez_compressed(npz,k=s["kx"],reference=s["target"],YAeHMOP=b,
        reference_all=s["target_all"],YAeHMOP_all=E-emid,
        tick_x=np.array([x for x,_ in s["ticks"]],float),tick_labels=np.array([l for _,l in s["ticks"]]),
        dft_nocc=s["dft_nocc"],eht_nocc=n,n_pairs=pairs)
    write_bands_plot(npz,outdir,2*pairs,title=f"{name} — {pairs}VB + {pairs}CB, EHT vs DFT")
    write_full_comparison(npz,outdir,title=f"{name} — all bands, EHT vs DFT",emin=window[0],emax=window[1])
    return met

def evaluate_metal(entry,parameters,K,outdir,name,obj_set,window):
    """One YAeHMOP run of given parameters against a metallic DFT reference, with figures."""
    from fit_sparse_window import parse_sample,metal_scores,write_sample_input
    from band_objective import WindowSettings
    bind,_=resolve_yaehmop_assets()
    table=parse_parameter_table(parameters)
    syms=read_structure(entry["structure"],entry.get("structure_format")).get_chemical_symbols()
    params=subset_for_elements(table,list(dict.fromkeys(syms)))
    s=parse_sample(entry,outdir,params,{})
    wset=WindowSettings.from_dict(obj_set)
    work=outdir/"work";work.mkdir(parents=True,exist_ok=True)
    write_parameter_file(work/"eval.parms",params)
    inp=write_sample_input(s,work/"eval.bind",float(K))
    run_bind(bind,inp,work/"eval.parms",work/"eval.log")
    _,full=parse_band(str(inp)+".band")
    met,b,Eall=metal_scores(full,s,wset)
    npz=outdir/f"{name}_bands.npz"
    np.savez_compressed(npz,k=s["kx"],reference=s["target"],YAeHMOP=b,reference_all=s["target_all"],YAeHMOP_all=Eall,
        tick_x=np.array([x for x,_ in s["ticks"]],float),tick_labels=np.array([l for _,l in s["ticks"]]),
        metallic=True,window=np.array([wset.window_min,wset.window_max]))
    write_metal_plots(npz,outdir,title=f"{name} — EHT vs DFT (metal)")
    return met

def metal_finish(args,outdir,name,met,state,bnpz,info,cif,sfmt,npz,obj,start,budget,optimizer,polish,
                 base_calls,refinement_calls,dim,radius,er,step,window):
    """Figures, run_summary.json and fitting/ of a fit to a metallic reference."""
    ch=info.get("spin_channel")
    m=write_metal_plots(bnpz,outdir,title=f"{name} — EHT fit vs DFT"+(f" (spin {ch})" if ch else "")+", metal")
    status={"reference_type":"dft","metallic":True,"dft_reference":info,"cif":str(cif),"structure_format":sfmt,
        "objective":obj,"objective_settings":state.get("objective_settings"),"basis_added":state.get("basis_added"),
        "start_from":str(start) if start else None,"window_metrics":met.get("window"),
        "cif_sha256":hashlib.sha256(Path(cif).read_bytes()).hexdigest(),
        "reference_input":str(npz),"input_sha256":hashlib.sha256(npz.read_bytes()).hexdigest(),
        "optimizer_evaluations":state["objective_evaluations"],
        "initialization_calls":base_calls,"refinement_calls":refinement_calls,
        "requested_total_budget":budget,"optimizer_choice":optimizer,"polish":polish,
        "selected_rmse_eV":met["selected_rmse_eV"],"rms_within_1eV_of_EF_eV":m["near"],
        "eht_fermi_eV":met["eht_fermi_eV"],
        "K_EHT":state["best_x"]["GLOBAL__K"],"periodic_dim":dim,"overlap_radius":int(radius),
        "native_parameters":str(outdir/"fit/param_best_native.dat"),
        "figure":str(outdir/"bands_comparison.png"),"full_figure":str(outdir/"bands_full_comparison.png"),
        "note":"Metal: DFT bands relative to E_F, paired from the bottom with the EHT bands; EHT shifted by the "
               "constant minimizing the weighted RMS; selected_rmse_eV = RMSE inside the fit window"}
    (outdir/"run_summary.json").write_text(json.dumps(status,indent=2)+"\n")
    write_fitting_dir(args.fitting_dir or outdir/"fitting",outdir,structure=cif,sfmt=sfmt,
        native=outdir/"fit/param_best_native.dat",param_txt=outdir/"fit/param_best.txt",
        K=state["best_x"]["GLOBAL__K"],kpoints=path_kpoints(er.nodes,step)[er.mask],dim=dim,
        radius=int(radius),summary=dict(name=name,metrics=met,**status),
        node_index=node_positions(len(er.nodes),step,er.mask),energy_zero=met["eht_fermi_eV"])
    print("USER_FIT_DONE",name,"metal window RMSE_eV",round(m["rmse"],5),"within 1 eV of E_F",round(m["near"],5))
    print("BEST_PARAMS",outdir/"fit/param_best_native.dat")
    print("BANDS_PLOT",outdir/"bands_comparison.png")
    print("FULL_PLOT",outdir/"bands_full_comparison.png")

def objective_config(args,config,default):
    """objective, its settings ([objective] section) and the extra basis shells."""
    from band_objective import WindowSettings,parse_basis_add
    fit=config["fit"]
    name=args.objective or fit.get("objective",fallback=default)
    sett={}
    if config.has_section("objective"):
        sett={k:v for k,v in config["objective"].items()}
    WindowSettings.from_dict(sett)             # validate early
    add=args.basis_add or fit.get("basis_add",fallback="")
    return name,sett,parse_basis_add(add)

def dft_main(args,parser,config,inputs,dft_dir):
    """Fit (or evaluate) EHT parameters against raw DFT eigenvalues of a folder."""
    ref_cfg=config["reference"];fit=config["fit"];bp=config["band_path"]
    code=args.dft_code or ref_cfg.get("dft_code")
    if not code:parser.error("--dft-code (vasp, qe, siesta, aims) is required with a DFT folder")
    try:code=normalize_code(code)
    except ValueError as exc:parser.error(str(exc))
    dft_file=args.dft_file or ref_cfg.get("dft_file",fallback=None)
    vb=args.valence_bands or (ref_cfg.getint("valence_bands") if ref_cfg.get("valence_bands") else None)
    pairs=int(args.pairs or fit.get("n_pairs","1"))
    if pairs not in (1,2,3,4):parser.error("n_pairs must be 1,2,3,4")
    outdir=args.outdir.resolve();outdir.mkdir(parents=True,exist_ok=True)
    refdir=outdir/"dft_reference";refdir.mkdir(exist_ok=True)
    spin_ch=args.spin_channel or ref_cfg.get("spin_channel",fallback=None)
    metallic=(args.objective or fit.get("objective",fallback="window"))=="metal"
    fermi_user=ref_cfg.getfloat("fermi",fallback=None)
    try:
        ref=read_dft_bands(dft_dir,code,dft_file,vb,spin_channel=spin_ch,metallic=metallic,fermi=fermi_user)
    except (ValueError,FileNotFoundError) as exc:
        parser.error(f"DFT folder {dft_dir}: {exc}")
    # Structure: the DFT one, or the user's file if it has the same cell
    if args.cif is not None:
        cif=args.cif.resolve()
        sfmt=resolved_structure_format(cif,args.structure_format)
        atoms=read_structure(cif,sfmt)
        if not np.allclose(np.asarray(atoms.cell),np.asarray(ref.atoms.cell),atol=1e-3):
            parser.error("The structure file has a different cell from the DFT band calculation; "
                         "omit it to use the DFT structure")
    else:
        from ase.io import write
        cif=refdir/"POSCAR";write(cif,ref.atoms,format="vasp",direct=True);sfmt="vasp"
    dim=args.dimension or (bp.getint("dimension") if bp.get("dimension") else None)
    dim_source="user"
    if dim is None:
        dim=detect_periodic_dim(ref.atoms);dim_source="vacuum detection"
    step=bp.getint("points_per_segment",fallback=20)
    name=args.name or fit.get("name",fallback=None) or Path(dft_dir).name
    if any(c in name for c in ("/","\\","..")):parser.error("name must not contain slash or '..'")
    npz=save_reference(ref,refdir/"reference_bands.npz")
    er=eht_reference(ref,step,dim)
    plot_dft_preview(ref,er,pairs,refdir/"dft_bands_preview.png",name)
    info=dict(dft_dir=str(Path(dft_dir).resolve()),code=code,files=ref.files,notes=ref.notes,
              formula=ref.atoms.get_chemical_formula(),n_kpoints=len(ref.kfrac),
              n_bands=int(ref.energies.shape[1]),occupied_bands=ref.nocc,occupied_from=ref.nocc_source,
              fermi_eV=ref.fermi,
              gap_on_path_eV=None if metallic else float(ref.energies[:,ref.nocc].min()-ref.energies[:,ref.nocc-1].max()),
              midgap_eV=None if metallic else float(.5*(ref.energies[:,ref.nocc].min()+ref.energies[:,ref.nocc-1].max())),
              metallic=metallic,energy_zero_eV=ref.fermi if metallic else None,
              spin_channel=ref.spin_channel,
              periodic_dim=dim,periodic_dim_from=dim_source,yaehmop_path=[l for _,l in er.ticks],
              yaehmop_points_per_line=step,structure=str(cif))
    (refdir/"dft_reference.json").write_text(json.dumps(info,indent=2)+"\n")
    print("DFT_REFERENCE",json.dumps(dict((k,info[k]) for k in ("code","formula","n_kpoints","n_bands",
          "occupied_bands","gap_on_path_eV","periodic_dim","yaehmop_path"))),flush=True)
    for note in ref.notes:print("NOTE",note,flush=True)
    radius=args.overlap_radius or fit.getint("overlap_radius",fallback=DEFAULT_OVERLAP_RADIUS)
    if radius not in range(1,9):parser.error("--overlap-radius must be in [1,8]")
    entry={"name":name,"role":"train","weight":1.0,"structure":str(cif),"structure_format":sfmt,
           "reference":str(npz),"reference_format":"dft","point_dim":dim,"n_pairs":pairs,
           "points_per_line":step}
    if metallic:entry["metallic"]=True
    window=(ref_cfg.getfloat("emin",fallback=-6.),ref_cfg.getfloat("emax",fallback=6.))
    if args.validate_only or args.prepare_only:
        print("PREPARED_ONLY: review",refdir/"dft_bands_preview.png","before fitting.")
        return
    if args.from_fit or args.from_parameters:
        if args.from_fit:
            st=json.loads(args.from_fit.resolve().read_text())
            K=float(st["best_x"]["GLOBAL__K"]);table=args.from_fit.resolve().parent/"param_best_native.dat"
        else:
            if args.k is None:parser.error("--k is required with --from-parameters")
            K=args.k;table=args.from_parameters.resolve()
        entry["overlap_radius"]=radius
        if metallic:
            try:
                obj,obj_set,_=objective_config(args,config,"metal")
            except ValueError as exc:
                parser.error(str(exc))
            met=evaluate_metal(entry,table,K,outdir,name,obj_set,window)
        else:
            met=evaluate_dft(entry,table,K,outdir,name,pairs,window)
        (outdir/"evaluation_summary.json").write_text(json.dumps(dict(parameters=str(table),K_EHT=K,
            dft_reference=info,metrics=met),indent=2)+"\n")
        ptxt=table.parent/"param_best.txt" if args.from_fit else None
        write_fitting_dir(args.fitting_dir or outdir/"fitting",outdir,structure=cif,sfmt=sfmt,
            native=table,param_txt=ptxt,K=K,kpoints=path_kpoints(er.nodes,step)[er.mask],dim=dim,
            radius=int(radius),summary=dict(name=name,mode="evaluation (no refit)",parameters=str(table),
            K_EHT=K,metrics=met,dft_reference=info),node_index=node_positions(len(er.nodes),step,er.mask),
            energy_zero=met.get("eht_fermi_eV"))
        print("DFT_EVALUATION_DONE",name,"RMSE_eV",round(met["selected_rmse_eV"],5),
              "gap_error_eV",round(met["gap_error_eV"],4))
        print("BANDS_PLOT",outdir/"bands_comparison.png");print("FULL_PLOT",outdir/"bands_full_comparison.png")
        return
    budget,optimizer,polish=optimizer_settings(args,fit,parser)
    try:
        obj,obj_set,basis_add=objective_config(args,config,"window")
    except ValueError as exc:
        parser.error(str(exc))
    yml={"n_pairs":pairs,"points_per_line":step,"overlap_radius":int(radius),
         "gap_weight":float(fit.get("gap_weight",".35")),"reg_weight":float(fit.get("reg_weight",".04")),
         "objective":obj,"objective_settings":obj_set,"basis_add":basis_add,
         "systems":[entry]}
    start=args.start_from or (fit.get("start_from") and csv_path(fit["start_from"],inputs.parent))
    if start:
        start=Path(start).resolve()
        table=start/"parameters_huckel.dat"
        if not table.is_file():parser.error(f"--start-from: {table} not found")
        yml["start_table"]=str(table)
        sj=start/"fit_summary.json"
        if sj.is_file():
            yml["start_K"]=float(json.loads(sj.read_text())["K_EHT"])
        print("START_FROM",start,flush=True)
    target=outdir/"fit_input.yaml"
    target.write_text(yaml.safe_dump(yml,sort_keys=False))
    base_calls,refinement_calls=run_optimizer(target,outdir,optimizer,budget,polish,args.seed,parser)
    state=json.loads((outdir/"fit/fit.json").read_text())
    met=state["metrics"][name]
    bnpz=outdir/"fit"/f"{name}_bands.npz"
    if metallic:
        return metal_finish(args,outdir,name,met,state,bnpz,info,cif,sfmt,npz,obj,start,budget,optimizer,polish,
                            base_calls,refinement_calls,dim,radius,er,step,window)
    rmse=write_bands_plot(bnpz,outdir,2*pairs,title=f"{name} — {pairs}VB + {pairs}CB, EHT fit to DFT")
    if abs(rmse-float(met["selected_rmse_eV"]))>1e-8:
        raise ValueError("Figure disagrees with optimizer metric")
    write_full_comparison(bnpz,outdir,title=f"{name} — all bands, EHT fit vs DFT",emin=window[0],emax=window[1],
                          window_metrics=met.get("window"))
    status={"reference_type":"dft","dft_reference":info,"cif":str(cif),"structure_format":sfmt,
        "objective":obj,"objective_settings":state.get("objective_settings"),"basis_added":state.get("basis_added"),
        "start_from":str(start) if start else None,
        "window_metrics":met.get("window"),
        "cif_sha256":hashlib.sha256(Path(cif).read_bytes()).hexdigest(),
        "reference_input":str(npz),"input_sha256":hashlib.sha256(npz.read_bytes()).hexdigest(),
        "n_pairs":pairs,"n_scored_bands":2*pairs,
        "optimizer_evaluations":state["objective_evaluations"],
        "initialization_calls":base_calls,"refinement_calls":refinement_calls,
        "requested_total_budget":budget,"optimizer_choice":optimizer,"polish":polish,
        "selected_rmse_eV":met["selected_rmse_eV"],"gap_error_eV":met["gap_error_eV"],
        "K_EHT":state["best_x"]["GLOBAL__K"],"periodic_dim":dim,"overlap_radius":int(radius),
        "native_parameters":str(outdir/"fit/param_best_native.dat"),
        "figure":str(outdir/"bands_comparison.png"),"full_figure":str(outdir/"bands_full_comparison.png"),
        "note":"Raw DFT eigenvalues interpolated at the YAeHMOP k points; trained only 2*n_pairs frontier bands"}
    (outdir/"run_summary.json").write_text(json.dumps(status,indent=2)+"\n")
    write_fitting_dir(args.fitting_dir or outdir/"fitting",outdir,structure=cif,sfmt=sfmt,
        native=outdir/"fit/param_best_native.dat",param_txt=outdir/"fit/param_best.txt",
        K=state["best_x"]["GLOBAL__K"],kpoints=path_kpoints(er.nodes,step)[er.mask],dim=dim,
        radius=int(radius),summary=dict(name=name,metrics=met,**status),
        node_index=node_positions(len(er.nodes),step,er.mask))
    print("USER_FIT_DONE",name,"bands",2*pairs,"RMSE_eV",round(rmse,5))
    print("BEST_PARAMS",outdir/"fit/param_best_native.dat")
    print("BANDS_PLOT",outdir/"bands_comparison.png")
    print("FULL_PLOT",outdir/"bands_full_comparison.png")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif",type=Path,nargs="?",
                        help="crystal structure: CIF, VASP, Quantum ESPRESSO, SIESTA or FHI-aims file "
                             "(optional with --dft-dir: the structure of the DFT calculation is used)")
    parser.add_argument("--structure-format",choices=STRUCTURE_FORMAT_CHOICES,default="auto",
                        help=STRUCTURE_FORMAT_HELP)
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
    parser.add_argument("--overlap-radius",type=int,default=None,
                        help="Overlap images in each active lattice direction, used by BOTH fitting and recomputation "
                             f"(default: [fit] overlap_radius, else {DEFAULT_OVERLAP_RADIUS}; --from-fit reuses the fit's value)")
    parser.add_argument("--optimizer",choices=("nelder-mead","simulated-annealing",
                        "sa-nelder-mead","hooke-jeeves")+GLOBAL_METHODS,default=None,
                        help="nelder-mead (default) = short DE + NM; SA/HJ = 60%% DE+NM then refinement; "
                             "differential-evolution/bayesian/cmaes/genetic/powell = one global search "
                             "from the stock parameters with the whole budget")
    parser.add_argument("--polish",choices=POLISH_METHODS,default=None,
                        help="local refinement after a global optimizer (25%% of the budget)")
    parser.add_argument("--seed",type=int,default=20261014)
    parser.add_argument("--objective",choices=("frontier","window","metal"),default=None,
                        help="frontier: n_pairs bands around the gap (image/CSV default); window: every DFT band "
                             "in an energy window with weights on gaps, band edges, Fermi-level states, critical "
                             "points (DFT default; settings in [objective])")
    parser.add_argument("--basis-add",default=None,
                        help="extra EHT shells, e.g. 'S:d Zn:d H:p' ([fit] basis_add); started from "
                             "estimates and fitted")
    parser.add_argument("--start-from",type=Path,default=None,
                        help="fitting/ folder of an earlier fit: start from its parameters (and basis), with "
                             "the search bounds centred there ([fit] start_from)")
    parser.add_argument("--fitting-dir",type=Path,default=None,
                        help="folder for the standard fit outputs (parameters, figures, H(k), S(k)); "
                             "default OUTDIR/fitting")
    dft=parser.add_argument_group("numerical DFT reference (instead of an image/CSV)")
    dft.add_argument("--dft-dir",type=Path,help="folder of a DFT band-structure calculation "
                     "([reference] dft_dir in the INI)")
    dft.add_argument("--dft-code",choices=DFT_CODES,help="DFT package of --dft-dir: vasp, qe, siesta, aims "
                     "([reference] dft_code)")
    dft.add_argument("--dft-file",help="main band file inside --dft-dir when several exist "
                     "([reference] dft_file)")
    dft.add_argument("--valence-bands",type=int,help="occupied DFT bands, if the automatic count is wrong "
                     "([reference] valence_bands)")
    dft.add_argument("--spin-channel",choices=("up","down"),help="one channel of a collinear spin-polarized run "
                     "([reference] spin_channel); fit each channel, then join them with spin_combine.py")
    dft.add_argument("--dimension",type=int,choices=(1,2,3),help="periodic dimension for a DFT reference "
                     "(default: [band_path] dimension, else detected from the vacuum)")
    args=parser.parse_args()
    inputs=args.input_file.resolve()
    if not inputs.is_file():parser.error("INI settings file not found: "+str(inputs))
    probe=conf(inputs,required=())
    dft_dir=args.dft_dir or (probe["reference"].get("dft_dir") and csv_path(probe["reference"]["dft_dir"],inputs.parent))
    if dft_dir:
        return dft_main(args,parser,probe,inputs,Path(dft_dir))
    if args.cif is None:parser.error("a structure file is required (or use --dft-dir)")
    cif=args.cif.resolve()
    if not cif.is_file():parser.error("Structure file not found: "+str(cif))
    try:
        sfmt=resolved_structure_format(cif,args.structure_format)
        read_structure(cif,sfmt)
    except ValueError as exc:
        parser.error(str(exc))
    print("STRUCTURE",cif,"format",sfmt,file=sys.stderr,flush=True)
    config=conf(inputs)
    dim,nodes,labels,step=read_nodes(config)
    fit=config["fit"];ref=config["reference"]
    pairs=int(args.pairs or fit.get("n_pairs","1"))
    if pairs not in (1,2,3,4):parser.error("n_pairs must be 1,2,3,4")
    outdir=args.outdir.resolve();outdir.mkdir(parents=True,exist_ok=True)
    radius=args.overlap_radius
    if radius is None and not args.from_fit:
        radius=fit.getint("overlap_radius",fallback=DEFAULT_OVERLAP_RADIUS)
    if radius is not None and radius not in range(1,9):
        parser.error("--overlap-radius must be in [1,8]")
    if args.from_fit:
        source=args.from_fit.resolve()
        status=json.loads(source.read_text())
        if "lattice" not in status:
            print("WARNING: fit.json predates the periodicity fix; it was optimized with a 2D "
                  "Lattice (5,5,0) regardless of [band_path] dimension."+
                  (" This 3D recomputation will NOT reproduce that fit; refit."
                   if dim==3 else ""),file=sys.stderr)
        if radius is None:
            radius=int(status.get("overlap_radius",DEFAULT_OVERLAP_RADIUS))
        saved_K=float(status["best_x"]["GLOBAL__K"])
        native=source.parent/"param_best_native.dat"
        if not native.is_file():raise FileNotFoundError(native)
        # Ensure the user supplied the same CIF as the original fit.
        files=status.get("train_reference_hashes",{})
        if len(files)==1:
            record=next(iter(files.values()))
            if record["cif"]!=hashlib.sha256(cif.read_bytes()).hexdigest():
                raise ValueError("Structure-file SHA256 differs from original fit; refusing silent substitution")
        recompute(cif,config,native,saved_K,outdir,radius,sfmt)
        return
    if args.from_parameters:
        if args.k is None:parser.error("--k is required with --from-parameters")
        recompute(cif,config,args.from_parameters.resolve(),args.k,outdir,radius,sfmt)
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
    budget,optimizer,polish=optimizer_settings(args,fit,parser,
        check_budget=not (args.validate_only or args.prepare_only))
    if args.validate_only:
        rasterqa=None
        if mode!="csv" and not (mode=="auto" and image.suffix.lower()==".csv"):
            from raster_ingest import normalize_and_write_qa
            _,rasterqa=normalize_and_write_qa(image,outdir/"input_validation",
                                    n_bands=2*pairs,expect_layered=(mode=="layered"))
        else:
            from prepare_band_png import csv_curves
            csv_curves(image,2*pairs,emin,emax)
        print(json.dumps(dict(cif=str(cif),structure_format=sfmt,image=str(image),band_count=2*pairs,
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
          structure_label(cif))
    if not name or any(c in name for c in ("/","\\","..")):
        raise ValueError("Material name must not contain slash or '..'")
    try:
        _obj,_set,basis_add=objective_config(args,config,"frontier")
    except ValueError as exc:
        parser.error(str(exc))
    if _obj!="frontier":
        parser.error("objective 'window' needs a numerical DFT reference (--dft-dir)")
    yml={"n_pairs":pairs,"points_per_line":step,"trace_nx":nx,
         "overlap_radius":int(radius),"basis_add":basis_add,
         "image_mode":"colored",
         "gap_weight":float(fit.get("gap_weight",".35")),
         "reg_weight":float(fit.get("reg_weight",".04")),
         "systems":[{
             "name":name,"role":"train","weight":1.0,
             "structure":str(cif),"structure_format":sfmt,"reference":str(prepared),
             "emin":float(emin),"emax":float(emax),
             "point_dim":dim,"symmetry_points":nodes,
             "labels":labels,"n_pairs":pairs,
             "image_mode":"colored","points_per_line":step,"trace_nx":nx}]}
    target=outdir/"fit_input.yaml"
    target.write_text(yaml.safe_dump(yml,sort_keys=False))
    base_calls,refinement_calls=run_optimizer(target,outdir,optimizer,budget,polish,args.seed,parser)
    state=json.loads((outdir/"fit/fit.json").read_text())
    met=state["metrics"][name]
    objective_rmse=write_bands_plot(outdir/"fit"/f"{name}_bands.npz",outdir,2*pairs,
                title=f"{name} — {pairs}VB + {pairs}CB (fit with {2*pairs} bands)")
    if abs(objective_rmse-float(met["selected_rmse_eV"]))>1.e-8:
        raise ValueError("Physical DFT/EHT figure disagrees with optimizer metric")
    status={
        "cif":str(cif),"structure_format":sfmt,
        "cif_sha256":hashlib.sha256(cif.read_bytes()).hexdigest(),
        "reference_input":str(image),"input_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
        "extraction":qa,"n_pairs":pairs,"n_scored_bands":2*pairs,
        "optimizer_evaluations":state["objective_evaluations"],
        "initialization_calls":base_calls,"refinement_calls":refinement_calls,
        "requested_total_budget":budget,"optimizer_choice":optimizer,"polish":polish,
        "selected_rmse_eV":met["selected_rmse_eV"],"gap_error_eV":met["gap_error_eV"],
        "K_EHT":state["best_x"]["GLOBAL__K"],
        "periodic_dim":dim,"overlap_radius":int(radius),"native_parameters":str(outdir/"fit/param_best_native.dat"),
        "figure":str(outdir/"bands_comparison.png"),
        "note":"Physical YAeHMOP calculations; trained only N=2*n_pairs frontier bands"}
    (outdir/"run_summary.json").write_text(json.dumps(status,indent=2)+"\n")
    xyz=parse_points([y for pt in nodes for y in pt],dim)
    write_fitting_dir(args.fitting_dir or outdir/"fitting",outdir,structure=cif,sfmt=sfmt,
        native=outdir/"fit/param_best_native.dat",param_txt=outdir/"fit/param_best.txt",
        K=state["best_x"]["GLOBAL__K"],kpoints=path_kpoints(xyz,step),dim=dim,radius=int(radius),
        summary=dict(status,name=name),node_index=node_positions(len(xyz),step))
    print("USER_FIT_DONE",name,"bands",2*pairs,
          "RMSE_eV",round(objective_rmse,5))
    print("BEST_PARAMS",outdir/"fit/param_best_native.dat")
    print("BANDS_PLOT",outdir/"bands_comparison.png")
    print("BANDS_CSV",outdir/"bands_comparison.csv")

if __name__=="__main__":
    main()
