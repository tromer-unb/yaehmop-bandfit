#!/usr/bin/env python3
"""Generate *synthetic* (NOT DFT) two/four-band CSV and raster examples.

All data use explicit toy formulas; safe for documentation and licensing.
No YAeHMOP, DFT, internet, or optimization.
"""
from pathlib import Path
import argparse,csv,math
import numpy as np
from PIL import Image,ImageDraw
COLORS=["#c82d37","#1c7ad5","#1a9d5f","#ae46af"]
def make(outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    x=np.linspace(0,1,151)
    v0=-.92-.28*np.sin(np.pi*x)**2
    c0=1.03+.35*np.cos(2*np.pi*x)**2
    vdeep=-2.35-.24*np.cos(3*np.pi*x)**2
    chigh=2.34+.22*np.sin(2*np.pi*x)**2
    for n,yy in [(2,np.column_stack([v0,c0])),
                 (4,np.column_stack([vdeep,v0,c0,chigh]))]:
        assert np.all(np.diff(yy,axis=1)>0)
        path=outdir/f"toy_{n}bands.csv"
        with path.open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["x"]+[f"band_{j}" for j in range(1,n+1)])
            w.writerows(np.column_stack([x,yy]))
        # Real visible colored curve image, NOT the machine bitplane format.
        width,height=1320,960;box=(115,105,1210,820)
        pic=Image.new("RGB",(width,height),"white")
        d=ImageDraw.Draw(pic)
        L,T,R,B=box
        d.rectangle(box,outline="#1d2730",width=3)
        for band in range(n):
            pts=[(round(L+float(q)*(R-L)),
                 round(T+(3.5-float(e))/7*(B-T))) for q,e in zip(x,yy[:,band])]
            d.line(pts,fill=COLORS[band],width=4)
        pic.save(outdir/f"toy_{n}bands.png",format="PNG")
        print("GENERATED",path.name,f"toy_{n}bands.png","colors",",".join(COLORS[:n]))
    (outdir/"README.md").write_text(
       "# Synthetic test curves — NOT DFT or experimental spectra\n\n"
       "Generated deterministically by `scripts/make_synthetic_example.py`.\n"
       "Source: invented toy smooth equations, not copied from any article.\n"
       "The synthetic CSV and RGB plot are both supplied for illustration.\n"
       "Plot bounding box: `115,105,1210,820`, vertical energy limits "
       "`-3.5..3.5 eV`. Colors in ascending band order: "
       "`#c82d37,#1c7ad5,#1a9d5f,#ae46af`.\n"
       "These curves are demonstrative software fixtures, not valid physical references.\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir",type=Path,default=Path("examples/synthetic"))
    a=p.parse_args();make(a.outdir)
