#!/usr/bin/env python3
"""Calibrate static raster/CSV curves and encode N fitted DFT bands as P5 PNG bitplanes.

Supported modes:
- layered: already P5 exact 8-bit per-band bitplanes (lossless)
- colors: trace one user-specified RGB color per original band
- mono: generic single-color image, energy-sort N traces (heuristic)
- csv: explicit numeric band curves, best alternative for degeneracies.

NEVER claim that an ordinary PNG uniquely recovers coincident/hidden curves.
Every output includes diagnostic quality metrics and a visual QA overlay.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/"source"
FITTER=ROOT.parent/"parameter_fitter"
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(FITTER))
from generate_mc2d_multiband import (FRAME,SIZE,read_colored_png)
from image_target import load_band_image,trace_bands,detect_axes_bbox
from image_trace_advanced import (
    sparse_bitplane_signature,palette_from_rgb,trace_rgb_image,mono_quality_gate)
from plot_region import infer_plot_bbox
from raster_ingest import normalize_and_write_qa

def parse_bbox(txt):
    if txt is None:return None
    bits=[int(t.strip()) for t in txt.split(",")]
    if len(bits)!=4 or bits[2]-bits[0]<20 or bits[3]-bits[1]<20:
        raise ValueError("bbox must be left,top,right,bottom with positive size")
    return tuple(bits)

def parse_colors(s,n):
    colors=[]
    for token in s.split(","):
        token=token.strip().lstrip("#")
        if len(token)!=6:raise ValueError("Expected RGB colors such as #c82d37,#1c7ad5")
        colors.append(tuple(bytes.fromhex(token)))
    if len(colors)!=n:raise ValueError(f"Need precisely {n} colors, one for each band")
    if len(set(colors))!=n:raise ValueError("RGB colors must be distinct")
    return colors

def rgb_curves(png,n,bbox,colors,tolerance,nx,emin,emax,coverage):
    return trace_rgb_image(png,n,bbox,colors,emin,emax,nx,
                           tolerance=tolerance,min_coverage=coverage)

def mono_curves(png,n,bbox,nx,emin,emax,coverage,allow_ambiguous):
    target=load_band_image(png,emin,emax,bbox=bbox)
    x,data,counts,seed=trace_bands(target,n,nx=nx)
    if not allow_ambiguous:
        qa=mono_quality_gate(counts,n,coverage=coverage)
    else:
        qa=dict(visible_fraction=float(np.mean(counts>=n)),
            largest_ambiguous_fraction=float("nan"),
            warning="UNSAFE APPROXIMATE mono tracing was explicitly enabled")
    return x,data,dict(**qa,minimum_visible=int(counts.min()),
        maximum_visible=int(counts.max()),seeded_column=int(seed),
        estimated=True)

def csv_curves(path,n,emin,emax):
    with path.open(newline="") as f:
        r=csv.DictReader(f)
        if not r.fieldnames:raise ValueError("CSV must have a header")
        columns=["x"]+[f"band_{j}" for j in range(1,n+1)]
        if r.fieldnames!=columns:
            raise ValueError("CSV header must be exactly: "+",".join(columns))
        data=np.array([[float(row[k]) for k in columns] for row in r],float)
    if len(data)<15 or not np.all(np.isfinite(data)):
        raise ValueError("CSV needs >=15 finite rows")
    if np.any(np.diff(data[:,0])<=0) or abs(data[0,0])>1e-6 or abs(data[-1,0]-1)>1e-6:
        raise ValueError("CSV x coordinate must be strictly increasing 0..1")
    if np.any(np.diff(data[:,1:],axis=1)<-1e-5):
        raise ValueError("At each x, band_1..band_N must be sorted in energy ascending")
    if np.any(data[:,1:]<emin) or np.any(data[:,1:]>emax):
        raise ValueError("CSV energy extends beyond emin/emax")
    return data[:,0],data[:,1:],dict(estimated=False,warning="CSV input, not PNG extraction")

def encode_layered(x,energies,out,emin,emax):
    left,top,right,bottom=FRAME
    mask=np.zeros((SIZE[1],SIZE[0]),dtype=np.uint8)
    for j in range(energies.shape[1]):
        layer=Image.new("1",SIZE,0)
        pix=[(int(round(left+q*(right-left))),
              int(round(top+(emax-e)/(emax-emin)*(bottom-top))))
             for q,e in zip(x,energies[:,j])]
        ImageDraw.Draw(layer).line(pix,fill=1,width=2)
        mask|=(np.asarray(layer,dtype=np.uint8)*(1<<j))
    Image.fromarray(mask).save(out,optimize=True)

def prepare(src,outdir,n_pairs,emin,emax,mode="auto",bbox=None,colors=None,
            tolerance=36,nx=301,min_coverage=.75,allow_ambiguous=False):
    src=Path(src).resolve();outdir=Path(outdir).resolve()
    outdir.mkdir(parents=True,exist_ok=True)
    if not src.is_file():raise FileNotFoundError(src)
    n=2*int(n_pairs)
    if n not in (2,4,6,8):raise ValueError("Only 2,4,6,8 bands supported")
    if emin>=emax:raise ValueError("emin must be less than emax")
    # Single normalized PNG ingress; raw CSV is the only bypass.
    if mode=="auto" and src.suffix.lower()==".csv":
        mode="csv"
    raster_src=None;ingest=None
    if mode!="csv":
        raster_src,ingest=normalize_and_write_qa(
            src,outdir,n_bands=n,expect_layered=(mode=="layered"))
    if mode=="auto":
        arr=np.asarray(Image.open(raster_src))
        if sparse_bitplane_signature(arr,n,FRAME,SIZE):
            mode="layered"
        elif colors:
            mode="colors"
        else:
            try:
                raw=np.asarray(Image.open(raster_src).convert("RGB"))
                region=tuple(bbox or infer_plot_bbox(raw)[0])
                guess=palette_from_rgb(raw,region,n)
            except ValueError as err:
                raw=np.asarray(Image.open(raster_src).convert("RGB"))
                region=tuple(bbox or infer_plot_bbox(raw)[0])
                L,T,R,B=region
                core=raw[T:B+1,L:R+1]
                chroma=np.max(core.astype(np.int16),axis=2)-np.min(core.astype(np.int16),axis=2)
                has_many_colored_pixels=int(np.sum(chroma>40))>max(50,n*35)
                if has_many_colored_pixels:
                    raise ValueError("Color image present but N reliable curve colors "
                                     "were not automatically found. Specify --colors "
                                     "or use manual digitization; do not silently "
                                     "downgrade to monochrome tracing.") from err
                mode="mono"
            else:
                mode="colors"
                colors=",".join("#"+"".join(f"{c:02x}" for c in item) for item in guess)
    filename=outdir/f"bands_{n}_layers.png"
    meta={"source":str(src),"mode":mode,"n_pairs":n_pairs,"n_bands":n,
          "emin":emin,"emax":emax}
    if ingest is not None:
        meta["raster_ingest"]=ingest
        meta["internal_raster_png"]=str(raster_src)
        meta["internal_final_contract"]="P5_8bit_N_band_bitplanes_PNG"
    if mode=="layered":
        if not sparse_bitplane_signature(np.asarray(Image.open(raster_src)),n,FRAME,SIZE):
            raise ValueError("Not a verified P5 bitplane raster; grayscale images are NOT native layers")
        x,energies=read_colored_png(raster_src,n_pairs,emin,emax,nx=nx)
        # Preserve original bytes for legacy P5 PNG; other supported formats
        # are normalized to the exact same native layer pixels.
        import shutil
        if ingest["detected_format"]=="PNG" and ingest["canonical_mode"]=="L_BITPLANES" and ingest["exif_orientation"] in (None,1):
            shutil.copy2(src,filename)
        else:
            shutil.copy2(raster_src,filename)
        meta.update({"estimated":False,"warning":"Verified native P5 bitplanes, no curve inference"})
    elif mode=="csv":
        x,energies,details=csv_curves(src,n,emin,emax)
        meta.update(details)
        encode_layered(x,energies,filename,emin,emax)
    else:
        image=np.asarray(Image.open(raster_src).convert("RGB"))
        bbox=tuple(bbox or infer_plot_bbox(image)[0])
        if mode=="colors":
            if not colors:
                guess=palette_from_rgb(image,bbox,n)
                colors=",".join("#"+"".join(f"{v:02x}" for v in item) for item in guess)
            x,energies,details=rgb_curves(raster_src,n,bbox,parse_colors(colors,n),
                                         tolerance,nx,emin,emax,min_coverage)
            details["used_hex_colors"]=colors.split(",")
        elif mode=="mono":
            x,energies,details=mono_curves(raster_src,n,bbox,nx,emin,emax,
                                            min_coverage,allow_ambiguous)
        else:raise ValueError("Unknown input mode "+mode)
        meta.update(details);meta["bbox"]=bbox
        meta["estimated"]=True
        encode_layered(x,energies,filename,emin,emax)
    xr,reconstructed=read_colored_png(filename,n_pairs,emin,emax,nx=nx)
    expected=np.column_stack([np.interp(xr,x,energies[:,j]) for j in range(n)])
    err=float(np.sqrt(np.mean((reconstructed-expected)**2)))
    if not np.isfinite(err) or err>.08:
        raise ValueError(f"Bitplane output validation RMSE {err:.3f} eV too large")
    meta["encode_roundtrip_rmse_eV"]=err
    meta["output"]=str(filename)
    mid=(float(reconstructed[:,n_pairs-1].max())+
         float(reconstructed[:,n_pairs].min()))*.5
    gap=float(reconstructed[:,n_pairs].min()-
              reconstructed[:,n_pairs-1].max())
    meta["gap_eV"]=gap
    meta["human_review_required"]=mode not in ("layered","csv")
    meta["qc_note"]="Good encoder roundtrip measures only the re-encoding step, NOT DFT correctness."
    if gap<=.02:raise ValueError(f"No positive >=0.02 eV gap around selected frontier ({gap:.3f} eV). "
                                "Check colors / energy order / plot limits / spinless-semiconductor eligibility.")
    preview=outdir/"extraction_preview.png"
    fig,ax=plt.subplots(figsize=(9,5.5))
    if mode not in ("csv","layered"):
        L,T,R,B=meta["bbox"]
        crop=Image.open(raster_src).convert("RGB").crop((L,T,R+1,B+1))
        ax.imshow(crop,extent=[0,1,emin-mid,emax-mid],
                  origin="upper",alpha=.56,aspect="auto")
    for j in range(n):
        label=f"VB {j+1}" if j<n_pairs else f"CB {j-n_pairs+1}"
        ax.plot(x,energies[:,j]-mid,lw=1.6,
                label=label,alpha=.96)
    ax.set_xlim(0,1)
    ax.set_ylim(emin-mid,emax-mid)
    ax.set_xlabel("Normalized reciprocal path")
    ax.set_ylabel("Energy relative to extracted midgap (eV)")
    ax.set_title(f"Original image and extracted {n_pairs}V + {n_pairs}C bands")
    ax.legend(ncol=min(n,4),fontsize=9)
    fig.tight_layout()
    fig.savefig(preview,dpi=180)
    plt.close(fig)
    meta["preview"]=str(preview)
    (outdir/"extraction_qa.json").write_text(json.dumps(meta,indent=2)+"\n")
    # Human-readable numeric curves also facilitate third-party checks.
    with (outdir/"extracted_curves.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["x"]+[f"band_{j+1}" for j in range(n)])
        w.writerows(zip(x,*[energies[:,j] for j in range(n)]))
    return filename,meta

def main():
    p=argparse.ArgumentParser(description="Prepare any Pillow-decodable STATIC raster (PNG/JPEG/TIFF/BMP/WEBP/GIF/PPM etc.) or CSV; canonicalize to PNG and validate N P5 band bitplanes")
    p.add_argument("image",type=Path)
    p.add_argument("--outdir",type=Path,required=True)
    p.add_argument("--n-pairs",type=int,choices=(1,2,3,4),required=True)
    p.add_argument("--emin",type=float,required=True)
    p.add_argument("--emax",type=float,required=True)
    p.add_argument("--mode",choices=("auto","layered","colors","mono","csv"),default="auto")
    p.add_argument("--bbox",help="left,top,right,bottom pixel coordinate (plot interior)")
    p.add_argument("--colors",help="comma-separated hex RGB colors, one per band")
    p.add_argument("--color-tolerance",type=int,default=36)
    p.add_argument("--trace-nx",type=int,default=301)
    p.add_argument("--min-coverage",type=float,default=.75)
    p.add_argument("--allow-ambiguous",action="store_true")
    q=p.parse_args()
    filename,details=prepare(q.image,q.outdir,q.n_pairs,q.emin,q.emax,mode=q.mode,
                             bbox=parse_bbox(q.bbox),colors=q.colors,
                             tolerance=q.color_tolerance,nx=q.trace_nx,
                             min_coverage=q.min_coverage,allow_ambiguous=q.allow_ambiguous)
    print("PREPARED",filename,"gap",round(details["gap_eV"],4),"eV",
          "roundtrip",round(details["encode_roundtrip_rmse_eV"],5),"eV")
    print("REVIEW_PREVIEW",details["preview"])
if __name__=="__main__":
    main()
