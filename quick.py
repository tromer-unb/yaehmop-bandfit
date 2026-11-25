#!/usr/bin/env python3
"""YAeHMOP BandFit quick tour: saved results only, no solver, no DFT, no fit (a few seconds).

Usage:
  python quick.py                 # tour of what the code does now, from the saved fits of validation/:
                                  #   DFT fits of 5 codes, metal + spin (joint fit), spin-orbit, WanTiBEXOS files
  python quick.py dft             # gapped DFT fits (VASP, QE, SIESTA, FHI-aims, Wannier90)
  python quick.py spin            # bcc Fe: metal, two spin channels, joint fit with one Fermi level
  python quick.py soc             # spin-orbit fits (on-site lambda L.S)
  python quick.py wtb             # the WanTiBEXOS files every fit writes (huckel-*.dat, basis_set-*)

  python quick.py paper           # the original study: six Ge/Sn chalcogenides FIRST, Fig. 4 SnS2, carbon
  python quick.py sns2            # Figure 4: SnS2 2/4/6/8 fitted bands
  python quick.py materials       # All six Ge/Sn systems across 2/4/6/8
  python quick.py chemical-transfer # Leave-one-compound-out, NOT a direct fit
  python quick.py best            # Full matched-window study + separate carbon scores
  python quick.py graphene        # Best external C2DB-fitted graphene pDZ7
  python quick.py transfer        # Carbon shared sp5 + blind phagraphene
  python quick.py carbon          # Saved full EHT bands graphene/diamond/graphite
  python quick.py image           # Synthetic RGB -> bitplane PNG, no solver

To fit your own material: band_fit_gui.py (graphical interface) or USER_WORKFLOW/bands_fit.py
(command line; manual, section "Fitting from the command line").
Only stdlib+NumPy+matplotlib+Pillow and dependencies in requirements.txt.
"""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent
C="#17679A";G="#178567";O="#C47A39";D="#253344"
def style():
    plt.rcParams.update({
      "font.family":"DejaVu Sans","font.weight":"bold",
      "font.size":11,"axes.titlesize":14,"axes.labelsize":11,
      "axes.titleweight":"bold","axes.labelweight":"bold",
      "xtick.labelsize":10,"ytick.labelsize":10,
      "axes.linewidth":1.2,"savefig.facecolor":"white"
    })
def save(fig,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=185,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print("  →",p)
def metrics():
    return json.loads((ROOT/"examples/best_results/metrics.json").read_text())
def results(out):
    """Shows comparisons only WITHIN their own PBE references."""
    m=metrics()
    a=m["graphene_c2db"];b=m["graphene_pyscf"];t=m["family_pyscf"]
    fig,axs=plt.subplots(1,3,figsize=(15.7,6.1))
    fig.subplots_adjust(left=.035,right=.975,bottom=.25,top=.77,wspace=.12)
    fig.suptitle("YAeHMOP BandFit | Best saved research results",
                 fontsize=18.2,fontweight="bold",y=.955,color=D)
    panels=[
      (axs[0],"01  GRAPHENE · C2DB PBE",
       ["Stock sp5","Best p-DZ7"],
       [a["comparison_rmse_eV"],a["best_rmse_eV"]],[O,C],
       "Same external PBE reference"),
      (axs[1],"02  GRAPHENE · PySCF PBE",
       ["Specific sp5","Shared sp5"],
       [b["system_specific_sp5_rmse_eV"],b["shared_sp5_rmse_eV"]],
       [C,G],"Different PBE dataset than C2DB"),
      (axs[2],"03  2D FAMILY · PySCF PBE",
       ["Graphene · train","T-graphene · train","net-C · train",
        "Phag · held out"],
       [t["trained"]["graphene"],t["trained"]["t_graphene"],t["trained"]["net_c"],
        t["heldout_phagraphene_shared_sp5_rmse_eV"]],
       ["#86B2A7","#86B2A7","#86B2A7",G],
       "Frozen shared sp5; phagraphene is blind"),
    ]
    for ax,title,labels,values,colors,sub in panels:
        y=np.arange(len(labels))
        bars=ax.barh(y,values,color=colors,height=.52)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.24)
        ax.set_title(title,loc="left",fontsize=12.0,pad=14,color=D)
        ax.set_xlabel("Two-band RMSE (eV)",fontweight="bold")
        ax.grid(axis="x",color="#CCD8E0",linewidth=.7,alpha=.9)
        ax.set_axisbelow(True)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        for bar,val,label in zip(bars,values,labels):
            yc=bar.get_y()+bar.get_height()/2
            ax.text(max(values)*.022,yc,label,ha="left",va="center",
                fontsize=10.4,fontweight="bold",color="white")
            ax.text(val+max(values)*.028,yc,
                f"{val:.3f}",va="center",fontsize=11.5,fontweight="bold",color=D)
        ax.text(0,-.18,sub,transform=ax.transAxes,ha="left",
                fontsize=9.3,fontweight="bold",color="#556675")
    fig.text(.5,.055,
       "C2DB and PySCF: DIFFERENT PBE references — only compare within panels. "
       "Phagraphene: Γ-only SCF, 9-node pilot.",
       ha="center",fontsize=10.5,fontweight="bold",color="#526171")
    save(fig,out/"carbon_best_results.png")
    print("BEST GRAPHENE (C2DB): p-DZ7, RMSE 0.3223 eV; stock sp5 0.5823 eV.")
    print("BEST GRAPHENE (PySCF): local sp5, RMSE 0.1657 eV; shared sp5 0.1683 eV.")
    print("BLIND PHAGRAPHENE: shared sp5, RMSE 0.1564 eV (Γ-only 9-k pilot).")
    print("NOTE: smallest tested phagraphene error 0.15628 eV used nested p-DZ7")
    print("      with nearly inactive second radial term; simpler shared sp5 preferred.")
    return out/"best_results.png"
def saved_pair(path):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    vals=np.asarray([[float(r[c]) for c in ("x","EHT_valence_eV","EHT_conduction_eV")]
                     for r in rows])
    if len(vals)<2 or np.any(np.diff(vals[:,0])<=0) or not np.isfinite(vals).all():
        raise ValueError("Malformed archived EHT frontier CSV: "+str(path))
    return vals
def bandline(ax,model,title,xlabels):
    d=saved_pair(ROOT/"examples/best_results"/(model+".csv"))
    ax.plot(d[:,0],d[:,1],color=C,lw=2.4,label="EHT valence")
    ax.plot(d[:,0],d[:,2],color=O,lw=2.4,label="EHT conduction")
    ax.set_title(title,loc="left",weight="bold",pad=14)
    ax.set_xlabel("Normalized k-path",weight="bold")
    ax.set_ylabel("Aligned EHT energy (eV)",weight="bold")
    ax.set_xlim(0,1)
    ax.set_xticks(np.linspace(0,1,len(xlabels)),xlabels)
    ax.grid(axis="y",alpha=.24)
    ax.legend(frameon=False,fontsize=10)
def graphene(out):
    fig,ax=plt.subplots(figsize=(10.3,5.0))
    bandline(ax,"graphene_c2db_pdz7",
       "Graphene · best C2DB fit · native p-DZ7 (RMSE 0.3223 eV)",
       [r"$\Gamma$","M","K",r"$\Gamma$"])
    fig.text(.50,-.04,
        "SAVED EHT predictions (361 k-nodes), not redistributed C2DB PBE bands. Patched solver required for recompute.",
        ha="center",weight="bold",fontsize=9.5)
    save(fig,out/"graphene_best_pdz7.png")
def transfer(out):
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.8),layout="constrained")
    bandline(axes[0],"graphene_pyscf_sp5_shared",
      "Graphene · shared sp5 · TRAIN (0.1683 eV)",
      [r"$\Gamma$","M","K",r"$\Gamma$"])
    bandline(axes[1],"phagraphene_pyscf_sp5_holdout",
      "Phagraphene · same frozen sp5 · HOLDOUT (0.1564 eV)",
      [r"$\Gamma$","X","S","Y",r"$\Gamma$"])
    fig.suptitle("Same shared 2D-carbon parameters · different structures",
       fontsize=16,weight="bold",y=1.055)
    fig.text(.50,-.075,
       "Archived EHT bands only; Phagraphene reference was a 9-node Γ-only PBE pilot, not Dirac-converged.",
       ha="center",weight="bold",fontsize=10.3)
    save(fig,out/"transfer_graphene_phagraphene.png")
def carbon(out):
    import json
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.5),layout="constrained")
    for ax,model,title in zip(axs,
          ("graphene","diamond","graphite_AB"),
          ("Graphene · 2D","Diamond · 3D","AB graphite · 3D")):
        p=ROOT/"examples/precomputed"/model
        with (p/"eht_full_bands.csv").open(newline="") as f:
            d=list(csv.DictReader(f))
        j=json.loads((p/"eht_full_bands_metadata.json").read_text())
        nb=int(j["nocc_spatial"])
        x=np.array([float(r["x"]) for r in d])
        for offset,color,name in ((nb,C,"Valence"),(nb+1,O,"Conduction")):
            e=np.array([float(r[f"orbital_{offset}"]) for r in d])
            ax.plot(x,e,color=color,lw=2.3,label=name)
        ax.set_title(title,weight="bold")
        ax.set_xlabel("Normalized k-path",weight="bold")
        ax.set_ylabel("Full EHT energy (eV)",weight="bold")
        ax.grid(axis="y",alpha=.18);ax.legend(frameon=False)
    fig.suptitle("Saved physical YAeHMOP orbital bands · no new computation",
                 fontsize=16.2,weight="bold",y=1.055)
    save(fig,out/"carbon_graphene_diamond_graphite.png")
def image(out):
    sys.path.insert(0,str(ROOT/"USER_WORKFLOW"))
    from prepare_band_png import prepare
    f=ROOT/"examples/synthetic/toy_2bands.png"
    dest=out/"image_input"
    png,qa=prepare(f,dest,1,-3.5,3.5,mode="colors",
                   bbox=(115,105,1210,820),
                   colors="#c82d37,#1c7ad5",nx=151)
    print("  →",png,"(two original synthetic curves encoded in separate bits)")
    print("  →",dest/"extraction_preview.png")
    print(f"  Round-trip encoding error: {qa['encode_roundtrip_rmse_eV']:.4f} eV")

# ============================================================================
# Tour of the current features: the saved best fits of validation/
# ============================================================================
VAL=ROOT/"validation"
# (label, DFT code, folder of the fit): the best fit of each material (manual, "Validation: best fits")
DFT_FITS=[
    ("MoS2 monolayer, PBE","FHI-aims","aims/MoS2_monolayer_pbe/fitting/refine-bands"),
    ("Si diamond, LDA","Quantum ESPRESSO","qe/Si_diamond_lda/fitting/window-ga-basis"),
    ("Si wurtzite, LDA","Quantum ESPRESSO","qe/Si_wurtzite_lda/fitting/window-ga-basis"),
    ("SiC zincblende, PBE","Quantum ESPRESSO","qe/SiC_zincblende_pbe/fitting/window-ga-basis"),
    ("Al2O2 bilayer","SIESTA","siesta/Al2O2_bilayer_test/fitting/v1c3-cmaes-basis-8k"),
    ("C nanoribbon (1D), PBE","SIESTA","siesta/CNR_nanoribbon_1d_pbe/fitting/window-ga-basis"),
    ("In2S2 monolayer, PBE","VASP","vasp/In2S2_monolayer_pbe/fitting/refine-bands3"),
    ("TiBr2 monolayer, PBE","VASP","vasp/TiBr2_monolayer_pbe/fitting/window-ga"),
    ("ZnI2 monolayer, PBE","VASP","vasp/ZnI2_monolayer_pbe/fitting/refine-bands"),
    ("GaN monolayer, PBE","Wannier90","wannier90/GaN_monolayer_qe_pbe/fitting/refine-bands"),
    ("Si diamond, PBE","Wannier90","wannier90/Si_diamond_vasp_pbe/fitting/refine-bands"),
    ("TiBr2 monolayer, PBE","Wannier90","wannier90/TiBr2_monolayer_vasp_pbe/fitting/window-cmaes"),
]
SOC_FITS=[
    ("MoS2 monolayer, HSE06+SOC","FHI-aims","aims/MoS2_monolayer_hse06_soc/fitting/soc-from-window-ga"),
    ("MoS2 monolayer, PBE+SOC","SIESTA","siesta/MoS2_monolayer_pbe_soc/fitting/soc-from-aims-window-cmaes"),
    ("In2S2 monolayer, PBE+SOC","VASP","vasp/In2S2_monolayer_pbe_soc/fitting/soc-from-window-cmaes"),
    ("ZnI2 monolayer, PBE+SOC","VASP","vasp/ZnI2_monolayer_pbe_soc/fitting/soc-from-window-ga-basis"),
]
FE=VAL/"vasp/Fe_bcc_pbe_spin/fitting"

def _summary(d):
    return json.loads((d/"fit_summary.json").read_text())

def _copy_figure(src,dest):
    if not src.is_file():
        return None
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return dest

def _gallery(figs,out,title,ncol=4):
    """Thumbnails of saved figures in one overview picture."""
    from PIL import Image
    figs=[(t,f) for t,f in figs if f is not None]
    if not figs:
        return
    nrow=(len(figs)+ncol-1)//ncol
    fig,axs=plt.subplots(nrow,ncol,figsize=(4.2*ncol,3.2*nrow+0.6))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax,(t,f) in zip(np.atleast_1d(axs).ravel(),figs):
        im=Image.open(f);im.thumbnail((900,700))
        ax.imshow(im);ax.set_title(t,fontsize=9.5)
    fig.suptitle(title,fontsize=15,weight="bold")
    fig.tight_layout()
    save(fig,out)

def tour_dft(out):
    print("\n[DFT] Fits to raw DFT bands of five codes (window objective; blue DFT, red dashed EHT)")
    print(f"  {'material':26s} {'code':17s} {'RMS window':>10s} {'RMS ±2 eV':>10s} {'gap error':>10s}   folder")
    figs=[]
    for label,code,rel in DFT_FITS:
        d=VAL/rel
        s=_summary(d);w=s.get("window_metrics") or {}
        rw,r2,g=w.get("window_rms_eV"),w.get("rms_within_2eV_of_EF_eV"),w.get("gap_error_eV",s.get("gap_error_eV"))
        f=lambda v,fmt="{:.3f}":fmt.format(v) if v is not None else "-"
        print(f"  {label:26s} {code:17s} {f(rw):>10s} {f(r2):>10s} {f(g,'{:+.3f}'):>10s}   validation/{rel}")
        name=rel.split("/")[1]+"_"+rel.split("/")[0]
        figs.append((f"{label} ({code})",_copy_figure(d/"bands_full_comparison.png",out/"dft"/f"{name}.png")))
    print("  (eV; RMS of EHT - DFT in the fitted window and within 2 eV of the gap; every folder has the")
    print("   parameters, figures and H(R), S(R) files; the INIs of these fits are in validation/refine*/)")
    _gallery(figs,out/"dft_overview.png","EHT fits to DFT bands of five codes · saved results")

def tour_spin(out):
    print("\n[SPIN] bcc Fe (VASP PBE, ferromagnetic metal): two spin channels, metal objective")
    j=json.loads((FE/"metal-joint/joint_fit.json").read_text())
    j1=json.loads((FE/"metal-joint/joint_fit_stage1.json").read_text())
    sep={ch:_summary(FE/f"metal-cmaes-10k/spin_{ch}") for ch in ("up","down")}
    w=lambda s:(s.get("window_metrics") or s.get("metrics") or {}).get("window_rms_eV")
    print(f"  {'fit':42s} {'RMS window up / down':>22s} {'moment (μB)':>12s}")
    m0=j1["start_metrics"]["moment"]       # the separate tables with the common Fermi level of the model
    print(f"  {'separate channels (metal-cmaes-10k)':42s} {w(sep['up']):10.3f} / {w(sep['down']):.3f} {m0:12.3f}"
          "  (free constant per channel)")
    m1=j1["metrics"];m=j["metrics"]
    print(f"  {'joint fit, common E_F (stage 1)':42s} {m1['up']['window_rms_eV']:10.3f} / {m1['down']['window_rms_eV']:.3f} "
          f"{j1['joint']['moment']:12.3f}")
    print(f"  {'joint fit + weight on the s band (stage 2)':42s} {m['up']['window_rms_eV']:10.3f} / {m['down']['window_rms_eV']:.3f} "
          f"{j['joint']['moment']:12.3f}")
    print(f"  PBE moment: {j['joint']['dft_moment']} μB. The joint fit (USER_WORKFLOW/spin_joint_fit.py) runs both channels")
    print("  with the native spin-polarized bind of the fork and one common Fermi level: no alignment afterwards.")
    figs=[(f"Fe, joint fit, spin {ch}",_copy_figure(FE/f"metal-joint/spin_{ch}/bands_full_comparison.png",out/"spin"/f"Fe_joint_{ch}.png"))
          for ch in ("up","down")]
    print("  → huckel-SP.dat for WanTiBEXOS: validation/vasp/Fe_bcc_pbe_spin/fitting/metal-joint/spin_polarized/")
    _gallery(figs,out/"spin_overview.png","bcc Fe · joint fit of both spin channels · saved results",ncol=2)

def tour_soc(out):
    print("\n[SOC] Spin-orbit coupling on top of a spinless fit (on-site λ L·S; USER_WORKFLOW/soc_fit.py)")
    print(f"  {'system':28s} {'code':9s} {'λ (eV)':34s} {'RMS ±2 eV':>9s} {'gap error':>10s}")
    figs=[]
    for label,code,rel in SOC_FITS:
        d=VAL/rel;s=_summary(d);m=s.get("metrics_with_soc") or {}
        lam=", ".join(f"{k} {v:.3f}" for k,v in (s.get("lambdas_eV") or {}).items())
        r2=m.get("rms_within_2eV_of_EF_eV");g=m.get("gap_error_eV")
        print(f"  {label:28s} {code:9s} {lam:34s} {r2 if r2 is None else f'{r2:9.3f}'} {g if g is None else f'{g:+10.3f}'}")
        figs.append((f"{label} ({code})",_copy_figure(d/"bands_full_comparison.png",out/"soc"/(rel.split("/")[1]+".png"))))
    _gallery(figs,out/"soc_overview.png","Spin-orbit fits · saved results",ncol=2)

def _tb_header(path):
    import gzip
    p=Path(path)
    if not p.is_file() and Path(str(p)+".gz").is_file():
        p=Path(str(p)+".gz")
    opener=gzip.open if p.suffix==".gz" else open
    with opener(p,"rt") as f:
        head=[next(f).strip() for _ in range(8)]
    return p,head

def tour_wtb(out):
    print("\n[WTB] Every fit writes its Hamiltonian for WanTiBEXOS (read with DFT= S):")
    for label,rel,tb,basis in (("Si diamond (spinless)","qe/Si_diamond_lda/fitting/window-ga-basis","huckel-NP.dat","basis_set-NP"),
                               ("MoS2 HSE06 + SOC","aims/MoS2_monolayer_hse06_soc/fitting/soc-from-window-ga","huckel-SOC.dat","basis_set-SOC"),
                               ("Fe, spin polarized","vasp/Fe_bcc_pbe_spin/fitting/metal-joint/spin_polarized","huckel-SP.dat","basis_set-SP")):
        d=VAL/rel
        try:
            p,h=_tb_header(d/tb)
        except (OSError,StopIteration):
            continue
        print(f"  {label:22s} {p.relative_to(ROOT)}: type {h[0]}, {h[6]} basis functions, {h[7]} cells"
              + (f" (+ {basis})" if (d/basis).is_file() else ""))
    print("  Layout (as siesta2wtb.py): type, scissors, Fermi level (0), 3 lattice vectors [Å], basis size,")
    print("  number of cells, then 'Rx Ry Rz i j ReH ImH S' per element; H(R) - E_F*S(R), so E_F = 0.")
    print("  New exports also write huckel-*_r.dat, the position matrix <m,0|r|n,R> of the Slater basis")
    print("  (BSE_CENTER_FILE, for excitons and optics), in the frame of the YAeHMOP run.")

def tour(out,which):
    print("\nYAeHMOP BAND FIT | QUICK TOUR · SAVED RESULTS OF validation/")
    print("NO DFT / NO FIT / NO SOLVER / NO INTERNET")
    if which in ("tour","dft"):tour_dft(out)
    if which in ("tour","spin"):tour_spin(out)
    if which in ("tour","soc"):tour_soc(out)
    if which in ("tour","wtb"):tour_wtb(out)
    print("\nNext: fit your own material with the graphical interface (python band_fit_gui.py) or from the")
    print("command line (python USER_WORKFLOW/bands_fit.py settings.ini --dft-dir FOLDER --dft-code vasp --outdir runs/X;")
    print("manual/manual.pdf, section 'Fitting from the command line'). The original study: python quick.py paper")

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("example",nargs="?",default="tour",
      choices=("tour","dft","spin","soc","wtb",
               "paper","all","best","sns2","SnS2","materials","chemical-transfer",
               "GeS","GeSe","GeTe","SnSe","SnTe",
               "graphene","transfer","carbon-transfer","carbon","image"),
      help="default tour: the current features from the saved fits of validation/; paper (or all): the "
           "original study (P5 Ge/Sn matched-window fits, Fig. 4 SnS2, P6/P7 carbon, PNG demo); the other "
           "names show one part")
    p.add_argument("--outdir",type=Path,default=Path("outputs/quick"),
                   help="Output folder (default: outputs/quick)")
    a=p.parse_args(argv)
    style();out=a.outdir
    if a.example in ("tour","dft","spin","soc","wtb"):
        tour(out,a.example)
        print("\nDONE. Open PNG figures in",out.resolve())
        return 0
    paper(out,"all" if a.example=="paper" else a.example)
    print("\nDONE. Open PNG figures in",out.resolve())
    return 0

def paper(out,example):
    """The original study, as the earlier quick.py: saved results of the paper and the carbon extension."""
    sys.path.insert(0,str(ROOT/"scripts"))
    from chalcogenide_showcase import (
        make_global,make_heatmap,individual,winners,MATERIALS)
    print("\nYAeHMOP BAND FIT | COMPLETE PAPER · INSTANT SAVED EXAMPLES")
    print("P5 original Ge/Sn chalcogenides (2/4/6/8 bands) + P6/P7 carbon extension")
    print("NO DFT / NO FIT / NO SOLVER / NO INTERNET")
    if example in ("all","best","materials"):
        print("\n[1] MAIN STUDY · six Ge/Sn chalcogenides, matched windows")
        make_global(out)
        make_heatmap(out,"direct")
        winners()
    if example in ("all","sns2","SnS2"):
        print("\n[2] PAPER FIGURE 4 · SnS2, matched 2/4/6/8 bands")
        individual(out,"SnS2")
    elif example in MATERIALS:
        print("\n[2] INDIVIDUAL MATERIAL ·",example)
        individual(out,example)
    if example in ("all","chemical-transfer"):
        print("\n[3] CHEMICAL TRANSFER · hold out one of six Ge/Sn compounds")
        make_heatmap(out,"zero-shot")
    if example in ("all","best"):
        print("\n[4] SEPARATE CARBON EXTENSION (different PBE references)")
        results(out)
    if example in ("all","graphene"):graphene(out)
    if example in ("all","transfer","carbon-transfer"):transfer(out)
    if example in ("all","carbon"):carbon(out)
    if example in ("all","image"):image(out)
if __name__=="__main__":
    raise SystemExit(main())
