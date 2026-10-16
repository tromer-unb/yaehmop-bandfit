"""Fast P5 six-material showcase, built exclusively from SAVED EHT spectra.

Neither native YAeHMOP nor DFT is imported/run. There is no optimization.
The numerical direct-fit and excluded-chemical-system metrics were audited
against the original P5 archived targets during one-time offline export.
External PBE arrays are NOT distributed; original paper Fig. 4 image included.
"""
from __future__ import annotations
from pathlib import Path
import csv,json,shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE/"examples/chalcogenides"
MATERIALS=("GeS","GeSe","GeTe","SnS2","SnSe","SnTe")
WINDOWS=(2,4,6,8)
DARK="#263449";BLUE="#17679A";ORANGE="#C67532";GREEN="#18836A"
def numbers():
    rows=list(csv.DictReader((ROOT/"direct_and_transfer.csv").open(newline="")))
    out={(r["material"],int(r["bands"])):r for r in rows}
    assert len(out)==24
    return out
def summary():
    return json.loads((ROOT/"benchmark_summary.json").read_text())
def sci(c):
    if not c:return float("nan")
    return float(c)
def make_heatmap(out,mode="direct"):
    data=numbers()
    direct=mode=="direct"
    key="direct_rmse_eV" if direct else "zero_shot_rmse_eV"
    mat=np.asarray([[sci(data[(m,n)][key]) for n in WINDOWS] for m in MATERIALS])
    fig,ax=plt.subplots(figsize=(9.0,5.2))
    fig.subplots_adjust(left=.125,right=.89,bottom=.16,top=.78)
    cm=plt.get_cmap("YlGnBu").copy()
    cm.set_bad("#D3D9DE")
    vmax=.55 if direct else max(1.4,float(np.nanpercentile(mat,80)))
    im=ax.imshow(np.ma.masked_invalid(mat),cmap=cm,vmin=0,vmax=vmax,aspect="auto")
    ax.set_xticks(range(4),[f"{n} bands" for n in WINDOWS],fontsize=12,fontweight="bold")
    ax.set_yticks(range(6),MATERIALS,fontsize=12,fontweight="bold")
    ax.xaxis.set_ticks_position("bottom")
    for i,m in enumerate(MATERIALS):
        for j,n in enumerate(WINDOWS):
            v=mat[i,j]
            if np.isfinite(v):
                norm=min(v/vmax,1)
                color="white" if norm>.45 else DARK
                ax.text(j,i,f"{v:.3f}",ha="center",va="center",
                    color=color,weight="bold",fontsize=12.4)
            else:
                ax.text(j,i,"N/A",ha="center",va="center",
                    color="#6F7D87",weight="bold",fontsize=11)
    title=("Six Ge/Sn chalcogenides · direct 2/4/6/8-band fits"
           if direct else "Six Ge/Sn chalcogenides · zero-shot chemical transfer")
    fig.suptitle(title,fontsize=16.5,fontweight="bold",y=.971,color=DARK)
    fig.text(.5,.900,"Exactly N/2 valence + N/2 conduction states fitted and scored",
             ha="center",fontsize=10.4,fontweight="bold",color="#526274")
    fig.text(.5,.04,("Same-material fitting · mean RMSE 2/4/6/8: 0.229 / 0.248 / 0.273 / 0.321 eV"
        if direct else "Whole-compound held-out transfer · 8-band mean strongly affected by poor folds"),
        ha="center",fontsize=11.3,fontweight="bold",color="#526274")
    cbar=fig.colorbar(im,ax=ax,pad=.025,shrink=.8)
    cbar.set_label("Fitted-window RMSE (eV)",fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    for tick in cbar.ax.get_yticklabels():tick.set_weight("bold")
    out.mkdir(parents=True,exist_ok=True)
    filename=out/("six_materials_direct.png" if direct else "six_materials_zero_shot.png")
    fig.savefig(filename,dpi=190,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print("  →",filename)
    return filename
def make_global(out):
    j=summary();d=j["summary"]["direct"];z=j["summary"]["zero-shot shared"]
    x=np.arange(4)
    fitted=np.array([d[str(n)]["mean_rmse_eV"] for n in WINDOWS])
    unseen=np.array([z[str(n)]["mean_rmse_eV"] for n in WINDOWS])
    fig,ax=plt.subplots(figsize=(9.7,5.4))
    fig.subplots_adjust(left=.085,right=.985,bottom=.23,top=.77)
    wd=.32
    ax.bar(x-wd/2,fitted,wd,label="Direct fit (same material)",color=BLUE)
    ax.bar(x+wd/2,unseen,wd,label="Chemical holdout (zero-shot)",color=ORANGE)
    for xv,y in zip(x-wd/2,fitted):
        ax.text(xv,y+.08,f"{y:.3f}",ha="center",weight="bold",color=DARK,fontsize=11)
    for xv,y in zip(x+wd/2,unseen):
        ax.text(xv,y+.08,f"{y:.3f}",ha="center",weight="bold",color=DARK,fontsize=11)
    ax.set_xticks(x,[f"{n} bands" for n in WINDOWS],fontweight="bold")
    ax.set_ylabel("Mean RMSE (eV)",weight="bold")
    ax.set_ylim(0,4.0)
    ax.set_title("Original study: direct calibration versus real chemical transfer",
       fontsize=17,fontweight="bold",color=DARK,pad=32)
    ax.legend(loc="upper left",frameon=False,ncol=2)
    ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    fig.text(.5,.83,"GeS · GeSe · GeTe · SnS₂ · SnSe · SnTe",
        ha="center",fontweight="bold",fontsize=12.5,color=DARK)
    fig.text(.5,.045,
       "Direct-fit eligible systems: 6/6/6/5; zero-shot folds: 6/6/6/4. "
       "Not equivalent tasks.",
       ha="center",fontweight="bold",fontsize=10.6,color="#526274")
    out.mkdir(parents=True,exist_ok=True);path=out/"overall_2_4_6_8.png"
    fig.savefig(path,dpi=190,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print("  →",path)
    return path
def saved_bands(m,n):
    p=ROOT/m/f"{n}bands_eht.csv"
    if not p.exists():return None
    with p.open(newline="") as f:
        rr=list(csv.DictReader(f))
    arr=np.asarray([[float(x) for x in r.values()] for r in rr])
    assert arr.shape==(len(rr),n+1)
    assert np.all(np.diff(arr[:,0])>0)
    return arr
def individual(out,material):
    if material not in MATERIALS:raise ValueError(f"Unknown material {material}")
    data=numbers();out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(12.5,7.7),sharex=True)
    fig.subplots_adjust(left=.085,right=.965,bottom=.13,top=.84,wspace=.19,hspace=.39)
    nmade=0
    for n,ax in zip(WINDOWS,axs.flat):
        a=saved_bands(material,n)
        if a is None:
            ax.text(.5,.5,"Not eligible in the original benchmark",
                 transform=ax.transAxes,ha="center",weight="bold")
            ax.set_xticks([]);ax.set_yticks([]);continue
        for j in range(n):
            c=BLUE if j<n//2 else ORANGE
            ax.plot(a[:,0],a[:,j+1],c=c,lw=1.7,alpha=.82)
        rmse=float(data[(material,n)]["direct_rmse_eV"])
        ax.set_title(f"{n//2}V + {n//2}C · direct fit RMSE = {rmse:.3f} eV",
                     fontweight="bold",fontsize=12.0)
        ax.grid(axis="y",alpha=.16)
        ax.set_xlim(0,1)
        ax.set_xlabel("Normalized k-path",weight="bold")
        ax.set_ylabel("YAeHMOP EHT energy (eV)",weight="bold")
        nmade+=1
    fig.suptitle(f"{material} · individual EHT bands fitted with matched windows",
                 fontweight="bold",fontsize=17.1,y=.967,color=DARK)
    fig.text(.5,.047,
       "SAVED PHYSICAL EHT eigenvalues only; the paper's external PBE bands are not redistributed.",
       ha="center",fontweight="bold",fontsize=10.7,color="#4E6274")
    p=out/f"{material}_2_4_6_8_eht.png"
    fig.savefig(p,dpi=190,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print("  →",p,f"({nmade} fitted windows)")
    if material=="SnS2":
        paper=HERE/"assets/paper_fig04_SnS2.png"
        target=out/"SnS2_paper_Fig4_PBE_vs_EHT.png"
        shutil.copy2(paper,target)
        print("  →",target,"(ORIGINAL paper Fig. 4 — PBE versus EHT overlays)")
        print("  SnS₂ 1V+1C: 0.1043 eV; 2V+2C: 0.2725 eV; "
              "3V+3C: 0.2561 eV; 4V+4C: 0.3333 eV")
    return p
def winners():
    m=numbers()
    for n in WINDOWS:
        eligible=[(float(r["direct_rmse_eV"]),name) for (name,w),r in m.items()
                  if w==n and r["direct_rmse_eV"]]
        err,name=min(eligible)
        print(f"  Best {n}-band direct material: {name:5s} = {err:.4f} eV "
              f"({n//2}V+{n//2}C)")
