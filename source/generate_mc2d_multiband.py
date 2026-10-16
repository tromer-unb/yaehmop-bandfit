#!/usr/bin/env python3
"""Physically faithful, uniquely colored DFT frontier PNGs for 1+1/2+2/3+3/4+4.

Every curve comes from ORIGINAL matching MC2D PBE XMGrace eigenvalues;
its RGB identity is retained through crossings. No artificial bands.
Output PNGs are independent visual input; downstream fitting reads only
PNG/CIF, not DFT AGR/NPZ. Regenerate qc and reject bad digitization.
"""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from ase.io import read
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/"source"))
from inspect_agr import parse
sys.path.insert(0,str(P/"parameter_fitter"))
from yaehmop_generic import (resolve_yaehmop_assets,parse_parameter_table,
    subset_for_elements,orbital_count,valence_electrons)
from fit_shared import valid_basis
# All colors DISTINCT with one exact color per energy band; no antialiasing
COLORS=[(200,45,55),(28,122,213),(26,157,95),(174,70,175),
        (239,139,44),(40,170,165),(120,95,45),(195,55,131)]
SIZE=(3200,3400); FRAME=(160,100,3000,3260)
BAND_PX=2
def read_colored_png(path,n,emin,emax,nx=301):
    """Decode 8-bit OR bitplanes of exact reference band geometry.

    Degenerate bands occupy DIFFERENT bits at the SAME physical pixel;
    no synthetic energy shifts or phantom bands are introduced. This
    benchmark PNG is machine-encoded, not an ordinary journal plot.
    """
    img=np.asarray(Image.open(path))
    if img.ndim!=2:raise ValueError("requires 8-bit curve-layer PNG")
    left,top,right,bottom=FRAME
    w,h=right-left,bottom-top
    x=np.arange(nx,dtype=float)/(nx-1)
    xpix=np.round(left+x*w).astype(int)
    traces=np.full((nx,2*n),np.nan)
    for band in range(2*n):
        pix=(img&(1<<band))!=0
        for i,xx in enumerate(xpix):
            lo=max(0,xx-2);hi=min(img.shape[1],xx+3)
            ys=np.where(pix[top:bottom+1,lo:hi].any(axis=1))[0]
            if len(ys):traces[i,band]=emax-(ys.mean()/h)*(emax-emin)
        ok=np.isfinite(traces[:,band])
        if ok.mean()<.91:
            raise RuntimeError(f"Band {band} encoded coverage insufficient: {ok.mean():.3f}")
        traces[:,band]=np.interp(x,x[ok],traces[ok,band])
    return x,traces

def main():
    root=P/"data"/"mc2d"
    root.mkdir(exist_ok=True)
    from os import environ
    _,tab=resolve_yaehmop_assets()
    stock=parse_parameter_table(tab)
    records=[]
    for uid in ("GeS","GeSe","GeTe","SnS2","SnSe","SnTe"):
        source=(P/"data"/"mc2d"/uid if
            (P/"data"/"mc2d"/uid/"reference.agr").is_file() else
            P/"data"/"mc2d_original_p4"/uid)
        out=root/uid
        out.mkdir(exist_ok=True)
        for fn in ("reference.agr","structure.cif","frontier4_qc.json"):
            if (source/fn).resolve()!=(out/fn).resolve():
                shutil.copyfile(source/fn,out/fn)
        x,E,labels=parse(out/"reference.agr")
        model=read(out/"structure.cif")
        sym=model.get_chemical_symbols()
        basis=valid_basis(subset_for_elements(stock,list(dict.fromkeys(sym))))
        norb=orbital_count(basis,sym)
        noccE=valence_electrons(basis,sym)//2
        old=json.loads((out/"frontier4_qc.json").read_text())
        noccDFT=old["DFT_nocc_with_semicore"]
        gap=float(E[:,noccDFT].min()-E[:,noccDFT-1].max())
        assert abs(gap-old["metadata_gap_eV"])<.00005
        shift=float((E[:,noccDFT].min()+E[:,noccDFT-1].max())/2)
        xs=x/x[-1]
        for n in (1,2,3,4):
            if min(noccE,norb-noccE,noccDFT,E.shape[1]-noccDFT)<n:
                records.append(dict(material=uid,n=n,eligible=False,
                    reason="insufficient actual DFT/EHT occupied/unoccupied orbitals",
                    noccE=noccE,norb=norb))
                continue
            target=E[:,noccDFT-n:noccDFT+n]-shift
            bounds=max(6.,float(np.ceil(max(abs(target.min()),abs(target.max()))+1)))
            emin,emax=-bounds,bounds
            png=out/f"bands_{2*n}.png"
            image=Image.new("RGB",SIZE,(255,255,255))
            draw=ImageDraw.Draw(image)
            left,top,right,bottom=FRAME
            for j in range(2*n):
                seq=[(round(left+float(q)*(right-left)),
                      round(top+(emax-float(e))/(emax-emin)*(bottom-top)))
                     for q,e in zip(xs,target[:,j])]
                draw.line(seq,fill=COLORS[j],width=BAND_PX)
            draw.rectangle(FRAME,outline=(0,0,0),width=1)
            image.save(png,optimize=True)
            # Generate a second encoded PNG: bitplanes preserve every
            # original eigenvalue at exact overlap/degeneracy.
            mask=np.zeros((SIZE[1],SIZE[0]),dtype=np.uint8)
            for j in range(2*n):
                layer=Image.new("1",SIZE,0)
                painter=ImageDraw.Draw(layer)
                seq=[(round(left+float(q)*(right-left)),
                      round(top+(emax-float(e))/(emax-emin)*(bottom-top)))
                     for q,e in zip(xs,target[:,j])]
                painter.line(seq,fill=1,width=BAND_PX)
                mask |= np.asarray(layer,dtype=np.uint8)*(1<<j)
            encoded=out/f"bands_{2*n}_layers.png"
            Image.fromarray(mask,mode="L").save(encoded,optimize=True)
            try:
                rx,obs=read_colored_png(encoded,n,emin,emax,nx=301)
                ref=np.column_stack([np.interp(rx,xs,target[:,j])
                                     for j in range(2*n)])
                rmse=float(np.sqrt(np.mean((obs-ref)**2)))
                worst=float(np.max(np.abs(obs-ref)))
                cover=1.0
            except Exception as exc:
                rmse=float("inf");worst=float("inf");cover=0
                print("FAILED",uid,2*n,str(exc),flush=True)
            rec=dict(material=uid,n=n,eligible=bool(np.isfinite(rmse) and rmse<.03),
                     dft_indices=list(range(noccDFT-n,noccDFT+n)),
                     noccE=noccE,norb=norb,noccDFT=noccDFT,
                     dft_gap_eV=gap,emin=emin,emax=emax,
                     png_rmse_eV=rmse,png_max_error_eV=worst,
                     png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                     input_png_sha256=hashlib.sha256(encoded.read_bytes()).hexdigest(),
                     input_image=f"bands_{2*n}_layers.png",
                     image_type="bitplane-coded PNG for lossless degenerate-band tracing",
                     cif_sha256=hashlib.sha256((out/"structure.cif").read_bytes()).hexdigest(),
                     agr_sha256=hashlib.sha256((out/"reference.agr").read_bytes()).hexdigest(),
                     kpoint_count=len(x),energy_align_eV=shift,
                     dft_arc_k_ticks=labels,band_colors=[list(y) for y in COLORS[:2*n]])
            records.append(rec)
            print("IMAGE_QC",uid,2*n,"orbitals",norb,
                  "rmse",round(rmse,5),"worst",round(worst,4),
                  "eligible",rec["eligible"],flush=True)
    (P/"manifests/image_fidelity.json").write_text(json.dumps(records,indent=2)+"\n")
    print("VERIFIED",sum(r["eligible"] for r in records),"of",len(records),
          "potential image cases; invalid cases WILL NOT be fitted")
if __name__=="__main__":main()
