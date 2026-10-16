#!/usr/bin/env python3
"""One-command, research-faithful showcase: saved data ONLY, no physics runs.

Usage:
  python quick.py                 # P5 six chalcogenides FIRST, Fig. 4 SnS2 + carbon
  python quick.py sns2            # Figure 4: SnS2 2/4/6/8 fitted bands
  python quick.py materials       # All six Ge/Sn systems across 2/4/6/8
  python quick.py chemical-transfer # Leave-one-compound-out, NOT a direct fit
  python quick.py best            # Full matched-window study + separate carbon scores
  python quick.py graphene        # Best external C2DB-fitted graphene pDZ7
  python quick.py transfer        # Carbon shared sp5 + blind phagraphene
  python quick.py carbon          # Saved full EHT bands graphene/diamond/graphite
  python quick.py image           # Synthetic RGB -> bitplane PNG, no solver

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
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("example",nargs="?",default="all",
      choices=("all","best","sns2","SnS2","materials","chemical-transfer",
               "GeS","GeSe","GeTe","SnSe","SnTe",
               "graphene","transfer","carbon-transfer","carbon","image"),
      help="default all: P5 Ge/Sn matched-window fits FIRST, paper Fig.4 SnS2,"
           "then P6/P7 carbon extension and a tiny PNG input demo")
    p.add_argument("--outdir",type=Path,default=Path("outputs/quick"),
                   help="Output folder (default: outputs/quick)")
    a=p.parse_args(argv)
    style();out=a.outdir
    sys.path.insert(0,str(ROOT/"scripts"))
    from chalcogenide_showcase import (
        make_global,make_heatmap,individual,winners,MATERIALS)
    print("\nYAeHMOP BAND FIT | COMPLETE PAPER · INSTANT SAVED EXAMPLES")
    print("P5 original Ge/Sn chalcogenides (2/4/6/8 bands) + P6/P7 carbon extension")
    print("NO DFT / NO FIT / NO SOLVER / NO INTERNET")
    if a.example in ("all","best","materials"):
        print("\n[1] MAIN STUDY · six Ge/Sn chalcogenides, matched windows")
        make_global(out)
        make_heatmap(out,"direct")
        winners()
    if a.example in ("all","sns2","SnS2"):
        print("\n[2] PAPER FIGURE 4 · SnS2, matched 2/4/6/8 bands")
        individual(out,"SnS2")
    elif a.example in MATERIALS:
        print("\n[2] INDIVIDUAL MATERIAL ·",a.example)
        individual(out,a.example)
    if a.example in ("all","chemical-transfer"):
        print("\n[3] CHEMICAL TRANSFER · hold out one of six Ge/Sn compounds")
        make_heatmap(out,"zero-shot")
    if a.example in ("all","best"):
        print("\n[4] SEPARATE CARBON EXTENSION (different PBE references)")
        results(out)
    if a.example in ("all","graphene"):graphene(out)
    if a.example in ("all","transfer","carbon-transfer"):transfer(out)
    if a.example in ("all","carbon"):carbon(out)
    if a.example in ("all","image"):image(out)
    print("\nDONE. Open PNG figures in",out.resolve())
    return 0
if __name__=="__main__":
    raise SystemExit(main())
