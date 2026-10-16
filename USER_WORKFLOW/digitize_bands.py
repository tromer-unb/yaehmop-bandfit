#!/usr/bin/env python3
"""Correct difficult band PNGs using user-provided visible pixel anchors.

Template workflow:
  python3 digitize_bands.py journal.png --bbox L,T,R,B --emin -6 --emax 6 \
    --n-pairs 2 --outdir digitized --template
  # Fill annotation.json manually, or run with --interactive for mouse clicks:
  python3 digitize_bands.py journal.png --bbox ... --emin ... --emax ... \
    --n-pairs 2 --outdir digitized --interactive
  python3 digitize_bands.py journal.png --annotations digitized/annotation.json \
    --outdir digitized --convert

No inference of invisible bands: curve anchors are provided by the user.
"""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from PIL import Image
import matplotlib
if not __import__("os").environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from image_trace_advanced import validate_bbox
from prepare_band_png import prepare
from raster_ingest import normalize_and_write_qa

def template(png,outdir,bbox,emin,emax,n):
    path=Path(png).resolve()
    outdir=Path(outdir).resolve();outdir.mkdir(exist_ok=True,parents=True)
    normalized,ingest=normalize_and_write_qa(path,outdir)
    raw=np.asarray(Image.open(normalized).convert("RGB"))
    L,T,R,B=validate_bbox(raw,bbox)
    if emax<=emin:raise ValueError("emax must exceed emin")
    annotation=outdir/"annotation.json"
    data=dict(image=str(normalized),original_image=str(path),
              source_sha256=ingest["original_sha256"],
              normalized_sha256=ingest["canonical_sha256"],
              raster_ingest=ingest,
              bbox=[L,T,R,B],emin=float(emin),emax=float(emax),
              n_bands=int(n),coordinate_space="canonical_oriented_image_pixels",
              comment="Mark the exact visible centerline pixels for EACH band; "
                      "include both path endpoints. Never invent missing band anchors.",
              anchors=[[] for _ in range(n)])
    if annotation.exists():
        prev=json.loads(annotation.read_text())
        previous_hash=prev.get("source_sha256")
        # Preserve user clicks, but never reuse anchors for a different raster.
        if previous_hash is not None and previous_hash!=ingest["original_sha256"]:
            raise ValueError("Existing manual annotations belong to a different raster SHA256")
        if previous_hash is None and Path(prev["image"]).resolve()!=path:
            raise ValueError("Existing legacy manual annotations reference a different source image")
    else:
        annotation.write_text(json.dumps(data,indent=2)+"\n")
    preview=Image.open(normalized).convert("RGB")
    from PIL import ImageDraw
    ImageDraw.Draw(preview).rectangle((L,T,R,B),outline="#e000dd",width=max(2,preview.width//800))
    preview.save(outdir/"annotation_bbox.png")
    return annotation

def annotate_interactive(path):
    info=json.loads(Path(path).read_text())
    norm_source,conversion=normalize_and_write_qa(
        info["image"],Path(path).resolve().parent/"interactive_normalized"
        if "path" in locals() else Path(annotation).resolve().parent/"annotation_normalized")
    rgb=np.asarray(Image.open(norm_source).convert("RGB"))
    L,T,R,B=validate_bbox(rgb,info["bbox"])
    n=info["n_bands"]
    fig,ax=plt.subplots(figsize=(11,7))
    ax.imshow(rgb)
    ax.set_xlim(L,R);ax.set_ylim(B,T)
    points=[]
    for j in range(n):
        ax.set_title(f"Band {j+1}/{n}: click along curve LEFT-to-RIGHT; ENTER when complete")
        plt.draw()
        clicks=plt.ginput(n=-1,timeout=0,show_clicks=True)
        if len(clicks)<3:
            plt.close(fig)
            raise ValueError(f"Need at least three manually clicked anchors on band {j+1}")
        points.append([[float(x),float(y)] for x,y in clicks])
        arr=np.asarray(clicks)
        ax.plot(arr[:,0],arr[:,1],lw=1.3,marker="o",ms=2.2)
    plt.close(fig)
    info["anchors"]=points
    Path(path).write_text(json.dumps(info,indent=2)+"\n")
    return info

def extract_annotations(annotation,outdir,nx=301,max_anchor_spacing=.22):
    info=json.loads(Path(annotation).read_text())
    if "source_sha256" in info:
        orig=Path(info["original_image"])
        if not orig.is_file() or hashlib.sha256(orig.read_bytes()).hexdigest()!=info["source_sha256"]:
            raise ValueError("Annotated source raster modified since anchors were created")
        normalized=Path(info["image"])
        if not normalized.is_file() or hashlib.sha256(normalized.read_bytes()).hexdigest()!=info["normalized_sha256"]:
            raise ValueError("Canonical annotated raster modified since anchors were created")
    norm_source,conversion=normalize_and_write_qa(
        info["image"],Path(outdir).resolve()/"annotation_normalized")
    rgb=np.asarray(Image.open(norm_source).convert("RGB"))
    L,T,R,B=validate_bbox(rgb,info["bbox"])
    n=int(info["n_bands"])
    if n not in (2,4,6,8) or len(info["anchors"])!=n:
        raise ValueError("Annotation requires exactly N=2,4,6,8 band lists")
    emin=float(info["emin"]);emax=float(info["emax"])
    if not emin<emax:raise ValueError("Invalid energy calibration")
    x=np.linspace(0,1,nx)
    E=np.empty((len(x),n),float)
    coverage=[]
    for j,points in enumerate(info["anchors"]):
        a=np.asarray(points,float)
        if a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all():
            raise ValueError(f"Band {j+1} needs >=3 finite manual (x,y) anchors")
        if np.any((a[:,0]<L)|(a[:,0]>R)|(a[:,1]<T)|(a[:,1]>B)):
            raise ValueError(f"Band {j+1} has clicks outside annotated plot rectangle")
        norm=(a[:,0]-L)/(R-L)
        order=np.argsort(norm)
        norm=norm[order];pixels=a[order,1]
        if np.any(np.diff(norm)<=.001):
            raise ValueError(f"Duplicate or nonmonotonic x anchors on band {j+1}")
        if norm[0]>.025 or norm[-1]<.975:
            raise ValueError(f"Band {j+1} has no anchor at both path endpoints")
        largest=float(np.max(np.diff(norm)))
        if largest>max_anchor_spacing:
            raise ValueError(f"Band {j+1} max unannotated horizontal interval {largest:.0%} "
                             f"> {max_anchor_spacing:.0%}; click more points")
        E[:,j]=emax-(np.interp(x,norm,pixels)-T)/(B-T)*(emax-emin)
        coverage.append(dict(band=j+1,n_anchors=len(points),largest_gap_fraction=largest))
    # Eigenvalue ordering at each k, including crossed character-tracked
    # trajectories. User annotations remain separate in annotation.json.
    E.sort(axis=1)
    if not np.isfinite(E).all():raise ValueError("Nonfinite clicked bands")
    outdir=Path(outdir).resolve();outdir.mkdir(exist_ok=True,parents=True)
    csvpath=outdir/"manually_digitized_bands.csv"
    with csvpath.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["x"]+[f"band_{j+1}" for j in range(n)])
        w.writerows(np.column_stack([x,E]))
    qa=dict(reference_image=info["image"],calibration=info["bbox"],
        n_bands=n,n_pairs=n//2,nk=len(x),energy_bounds=[emin,emax],
        method="USER_SUPPLIED_MANUAL_ANCHORS",band_anchor_coverage=coverage,
        warning="Manual curves interpreted from visible pixels. Hidden crossings "
                "remain the annotator's responsibility, not automatically inferred.")
    (outdir/"manual_qa.json").write_text(json.dumps(qa,indent=2)+"\n")
    figure,ax=plt.subplots(figsize=(10,6))
    ax.imshow(rgb[T:B+1,L:R+1],extent=[0,1,emin,emax],origin="upper",alpha=.62,aspect="auto")
    for j in range(n):ax.plot(x,E[:,j],lw=1.6,label=f"Manual band {j+1}")
    ax.set_xlim(0,1);ax.set_ylim(emin,emax)
    ax.set_xlabel("Normalized k path");ax.set_ylabel("Energy (eV)")
    ax.legend(ncol=min(n,4),fontsize=8)
    ax.set_title(f"Check {n} manually anchored bands against ORIGINAL pixels")
    figure.tight_layout()
    figure.savefig(outdir/"manual_overlay.png",dpi=170);plt.close(figure)
    return csvpath,qa

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("image",type=Path)
    p.add_argument("--outdir",type=Path,required=True)
    p.add_argument("--bbox",help="Plot rectangle L,T,R,B in original image pixels")
    p.add_argument("--emin",type=float)
    p.add_argument("--emax",type=float)
    p.add_argument("--n-pairs",type=int,choices=(1,2,3,4))
    p.add_argument("--annotations",type=Path)
    ops=p.add_mutually_exclusive_group()
    ops.add_argument("--interactive",action="store_true")
    ops.add_argument("--template",action="store_true")
    ops.add_argument("--convert",action="store_true")
    p.add_argument("--trace-nx",type=int,default=301)
    q=p.parse_args()
    if q.annotations:
        ann=q.annotations.resolve()
    else:
        if not(q.bbox and q.n_pairs and q.emin is not None and q.emax is not None):
            p.error("Supply --annotations or all of --bbox, --n-pairs, --emin, --emax")
        bbox=tuple(int(v.strip()) for v in q.bbox.split(","))
        ann=template(q.image,q.outdir,bbox,q.emin,q.emax,2*q.n_pairs)
    if q.interactive:
        if matplotlib.get_backend().lower()=="agg":
            p.error("--interactive requires a desktop graphical display; edit annotation.json manually instead")
        annotate_interactive(ann)
    if q.interactive or q.convert:
        csvpath,meta=extract_annotations(ann,q.outdir,nx=q.trace_nx)
        path,details=prepare(csvpath,q.outdir/"prepared_for_fit",
                     meta["n_pairs"],meta["energy_bounds"][0],
                     meta["energy_bounds"][1],mode="csv",nx=q.trace_nx)
        print("CURVES_CSV",csvpath)
        print("P5_REFERENCE_PNG",path)
        print("QA_OVERLAY",Path(q.outdir)/"manual_overlay.png")
    else:
        print("ANNOTATION_TEMPLATE",ann)
        print("Fill all N anchor lists or run with --interactive, then --convert")

if __name__=="__main__":main()
