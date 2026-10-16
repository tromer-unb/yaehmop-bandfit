#!/usr/bin/env python3
"""Show normalized static raster image axis bounds and suggest distinct visible curve RGB colors.

The tool NEVER asserts that colors correspond to actual band identities.
Use suggested hex values as a starting point and check the QA overlay.
"""
from pathlib import Path
import argparse,json
from PIL import Image,ImageDraw
from raster_ingest import normalize_and_write_qa
import numpy as np

def inspect(src,outdir,bbox=None,palette_size=12):
    src=Path(src).resolve();outdir=Path(outdir).resolve()
    outdir.mkdir(parents=True,exist_ok=True)
    normalized,source_qa=normalize_and_write_qa(src,outdir)
    img=np.asarray(Image.open(normalized).convert("RGB"))
    h,w=img.shape[:2]
    if bbox is None:
        from sys import path as syspath
        from pathlib import Path as _Path
        syspath.insert(0,str(_Path(__file__).resolve().parent.parent/"parameter_fitter"))
        from plot_region import infer_plot_bbox
        bbox,source_of_box=infer_plot_bbox(img)
    if "source_of_box" not in locals():source_of_box="user-specified"
    L,T,R,B=bbox
    if min(L,T)<0 or R>=w or B>=h or not (L<R and T<B):
        raise ValueError("bbox outside PNG or invalid")
    pixels=img[T+2:B-2:2,L+2:R-2:2]
    # Filter white page, black axes/text, and grey grid lines.
    mx=pixels.max(axis=2).astype(np.int16)
    mn=pixels.min(axis=2).astype(np.int16)
    eligible=(mx-mn>=42)&(mn<215)&(mx>70)
    colors=pixels[eligible]
    if len(colors):
        packed=(colors[:,0].astype(np.uint32)<<16) | (colors[:,1].astype(np.uint32)<<8) | colors[:,2].astype(np.uint32)
        vals,count=np.unique(packed,return_counts=True)
        chosen=np.argsort(count)[::-1][:palette_size]
        palette=[{"rgb":"#"+format(int(vals[i]),"06x"),"pixels_sampled":int(count[i])}
                 for i in chosen]
    else:palette=[]
    original=Image.open(normalized).convert("RGB")
    pen=ImageDraw.Draw(original)
    pen.rectangle((L,T,R,B),outline=(255,0,255),width=max(2,w//700))
    im_out=outdir/"inspect_bbox.png";original.save(im_out)
    data={"image":str(src),"normalized_image":str(normalized),
          "raster_ingest":source_qa,"width":w,"height":h,"bbox":[L,T,R,B],"bbox_method":source_of_box,
          "suggested_visible_colors":palette,
          "warning":"Colors are frequency candidates, NOT correctly ordered bands. "
          "Degeneracies / covered curves cannot always be recovered from a raster PNG."}
    (outdir/"inspect.json").write_text(json.dumps(data,indent=2)+"\n")
    print("IMAGE",src.name,"DIM",w,"x",h,"BBOX",data["bbox"])
    print("COLOR_SUGGESTIONS"," ".join(p["rgb"] for p in palette))
    print("ANNOTATED",im_out)
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--outdir",default="inspect_result")
    p.add_argument("--bbox",help="L,T,R,B (optional, auto-detects dark plot frame)")
    p.add_argument("--top",type=int,default=12)
    a=p.parse_args()
    bbox=None
    if a.bbox:
        bbox=tuple(int(x.strip()) for x in a.bbox.split(","))
        if len(bbox)!=4:raise ValueError("bbox must have four pixel coordinates")
    inspect(a.image,a.outdir,bbox,palette_size=a.top)
if __name__=="__main__":main()
