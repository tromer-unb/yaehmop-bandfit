#!/usr/bin/env python3
"""Plot saved individual EHT orbital bands from paper examples — zero solver calls.

CSV units are eV. x is normalized reciprocal path length. Orbital
indices are 1-based indices from the saved full YAeHMOP Hamiltonian.
The plots are illustrations of archived EHT results, not independent DFT.
"""
from pathlib import Path
import argparse,csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def visualize(source,output,orbitals="frontier",emin=None,emax=None):
    source=Path(source);output=Path(output)
    with source.open(newline="") as f:
        reader=csv.DictReader(f)
        records=list(reader)
        fields=reader.fieldnames
    if not records or "x" not in fields:
        raise ValueError("Saved band CSV must contain x and orbital columns")
    names=[v for v in fields if v.startswith("orbital_")]
    if len(names)<2:raise ValueError("Requires at least 2 physical orbital bands")
    x=np.asarray([float(row["x"]) for row in records])
    energies=np.asarray([[float(row[k]) for k in names] for row in records])
    if np.any(np.diff(x)<=0):raise ValueError("k-path must be strictly increasing")
    if not np.isfinite(energies).all():raise ValueError("Nonfinite orbital energies")
    if orbitals=="all":selected=list(range(len(names)))
    elif orbitals=="frontier":
        n=len(names)//2
        # Electron occupation is system-specific; use paper fixture orbital map
        # from metadata rather than assuming half filling of EHT orbitals.
        import json
        manifest=source.parent/(source.stem+"_metadata.json")
        if not manifest.is_file():
            raise ValueError(f"frontier requires metadata file: {manifest}")
        data=json.loads(manifest.read_text())
        occupied=int(data["nocc_spatial"])
        selected=[occupied-1,occupied]
    else:
        selected=[int(t.strip())-1 for t in orbitals.split(",")]
        if any(j<0 or j>=len(names) for j in selected):
            raise ValueError(f"Requested orbital is outside 1..{len(names)}")
    fig,ax=plt.subplots(figsize=(9,5))
    colors=plt.get_cmap("tab10")
    for ix,j in enumerate(selected):
        ax.plot(x,energies[:,j],color=colors(ix%10),
                lw=2.1,label=f"Orbital {j+1}")
    ax.set_xlabel("Normalized reciprocal-path coordinate",weight="bold")
    ax.set_ylabel("YAeHMOP eigenvalue (eV)",weight="bold")
    ax.set_title(source.parent.name.replace("_"," ").title()+" — saved EHT bands",
                 weight="bold")
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    if emin is not None or emax is not None:
        ax.set_ylim(bottom=emin,top=emax)
    fig.tight_layout()
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=220,bbox_inches="tight")
    plt.close(fig)
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv",type=Path,help="Archived example EHT CSV")
    p.add_argument("--out",type=Path,required=True,help="Figure .png, .pdf or .svg")
    p.add_argument("--orbitals",default="frontier",
        help="'frontier' (from fixture metadata), 'all', or 1-based list e.g. '4,5'")
    p.add_argument("--emin",type=float,default=None)
    p.add_argument("--emax",type=float,default=None)
    a=p.parse_args()
    print("SAVED_EHT_FIGURE",visualize(a.csv,a.out,a.orbitals,a.emin,a.emax))
