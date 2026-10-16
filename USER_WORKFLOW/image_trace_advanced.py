#!/usr/bin/env python3
"""Conservative, inspectable curve tracing from *visible* pixels of ordinary PNGs.

No hidden curve is ever presented as directly observed. Every interpolated
pixel interval and uncertain selection is reported in QA.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

def validate_bbox(rgb,bbox):
    if bbox is None: raise ValueError("PNG plot bounds are unknown: specify --bbox")
    left,top,right,bottom=map(int,bbox)
    h,w=rgb.shape[:2]
    if (left<0 or top<0 or right>=w or bottom>=h or
        right-left<20 or bottom-top<20):
        raise ValueError("Invalid or out-of-range plot box (L,T,R,B)")
    return left,top,right,bottom

def sparse_bitplane_signature(image,n,frame,size):
    """Compatibility check, NOT inferred from grayscale shape alone."""
    if image.ndim!=2 or image.shape!=(size[1],size[0]):
        return False
    if not np.issubdtype(image.dtype,np.integer):
        return False
    if np.max(image)>((1<<n)-1):
        return False
    L,T,R,B=frame
    # Source P5 bitplanes have only curve pixels within these coordinates.
    outside=np.ones(image.shape,dtype=bool)
    outside[T:B+1,L:R+1]=False
    if np.any(image[outside]):return False
    inside=image[T:B+1,L:R+1]
    frac=np.mean(inside!=0)
    if not (.00001<frac<.14):return False
    # Require EACH layer present and spread over at least half the k range.
    for j in range(n):
        cols=np.any((inside&(1<<j))!=0,axis=0)
        if np.mean(cols)<.75:return False
    return True

def palette_from_rgb(rgb,bbox,n,*,min_fraction=.72):
    """Return N exact dominant stable chromatic palette colors or reject."""
    L,T,R,B=validate_bbox(rgb,bbox)
    crop=rgb[T+2:B-1:2,L+2:R-1:2].astype(np.uint8)
    hi=crop.max(2).astype(int);lo=crop.min(2).astype(int)
    visible=(hi-lo>=40)&(lo<=220)&(hi>65)
    points=crop[visible]
    if len(points)<n*50:
        raise ValueError("Too few distinct saturated pixels for automatic color detection")
    packed=(points[:,0].astype(np.uint32)<<16) | (points[:,1].astype(np.uint32)<<8) | points[:,2].astype(np.uint32)
    unique,count=np.unique(packed,return_counts=True)
    order=np.argsort(count)[::-1]
    chosen=[]
    for j in order[:500]:
        color=np.array([(unique[j]>>16)&255,(unique[j]>>8)&255,unique[j]&255],int)
        if any(np.max(np.abs(color-np.asarray(c,int)))<35 for c in chosen):continue
        chosen.append(tuple(int(x) for x in color))
        if len(chosen)>=n:break
    if len(chosen)!=n:raise ValueError("Cannot reliably identify N distinct curve colors")
    # Validate columns with a matching-color pixel.
    block=rgb[T:B+1,L:R+1].astype(np.int16)
    for col in chosen:
        dist=np.max(abs(block-np.asarray(col,np.int16)),axis=2)
        fraction=float(np.mean(np.any(dist<=25,axis=0)))
        if fraction<min_fraction:
            raise ValueError(f"Auto palette found color {col} with only {fraction:.0%} horizontal coverage")
    return chosen

def _runs(boolean):
    v=np.asarray(boolean,dtype=bool)
    if len(v)==0:return []
    start=np.where(v & np.r_[True,~v[:-1]])[0]
    end=np.where(v & np.r_[~v[1:],True])[0]
    return list(zip(start,end+1))

def _groups(values,maximum_gap=3):
    if not len(values):return np.array([],float)
    ys=np.unique(np.asarray(values,int))
    separations=np.where(np.diff(ys)>maximum_gap)[0]+1
    return np.array([float(np.mean(a)) for a in np.split(ys,separations)],float)

def anti_aliased_color_mask(crop,color,tolerance=30):
    """Find a colored line even if it is blended with WHITE background.

    Pixels along the RGB direction from white to the original line
    color are accepted; black labels and grayscale grid lines are not.
    """
    delta=(255.-np.asarray(color,dtype=np.float32))
    pix=crop.astype(np.float32)
    residue=255.-pix
    alpha=np.clip(np.sum(residue*delta,axis=2)/max(float(np.dot(delta,delta)),1.),
                  0,1.2)
    expected=255.-alpha[:,:,None]*delta
    mismatch=np.max(abs(pix-expected),axis=2)
    return (alpha>=.25)&(alpha<=1.08)&(mismatch<=min(9.,max(6.,tolerance)))

def trace_rgb_image(source,n,bbox,colors,emin,emax,nx,
                    *,tolerance=30,min_coverage=.75,max_gap_fraction=.065,
                    max_multi_fraction=.25):
    """Trace N color-resolved curves; refuse extensive hidden fragments.

    A nearest-path continuity tracker chooses one visible component per
    sampled x, not the global median of all pixels of the same hue.
    """
    rgb=np.asarray(Image.open(source).convert("RGB"))
    L,T,R,B=validate_bbox(rgb,bbox)
    crop=rgb[T:B+1,L:R+1].astype(np.int16)
    h,w=crop.shape[:2]
    x=np.linspace(0,1,int(nx))
    traces=np.full((len(x),n),np.nan)
    quality=[]
    masks=[]
    for j,color in enumerate(colors):
        diff=np.max(np.abs(crop-np.asarray(color,dtype=np.int16)),axis=2)
        mask=diff<=tolerance
        anti_alias_used=False
        if float(np.mean(np.any(mask,axis=0)))<min_coverage:
            blend=anti_aliased_color_mask(crop,color,tolerance)
            if float(np.mean(np.any(blend,axis=0)))>float(np.mean(np.any(mask,axis=0)))+.08:
                mask=blend
                anti_alias_used=True
        masks.append(mask)
        # Only columns within the crop; median across ALL y is forbidden
        # because plot keys, text, grid lines may be the same color.
        comps=[]
        for q in x:
            col=int(round(q*(w-1)))
            radius=max(1,int(w/(len(x)-1)/2.8))
            lo=max(0,col-radius);hi=min(w,col+radius+1)
            ys=np.where(np.any(mask[:,lo:hi],axis=1))[0]
            comps.append(_groups(ys))
        count=np.array([len(v) for v in comps],int)
        observed=np.zeros(len(x),bool)
        track=np.full(len(x),np.nan)
        # Start from a high-support unambiguous point near the middle.
        good=np.flatnonzero(count==1)
        if not len(good):
            raise ValueError(f"Band {j+1} coverage 0 percent: no unambiguous visible starting position")
        start=int(good[np.argmin(abs(good-(len(x)-1)/2))])
        track[start]=comps[start][0];observed[start]=True
        for direction in (1,-1):
            idx=start+direction
            last=start
            older=None
            while 0<=idx<len(x):
                choices=comps[idx]
                if len(choices):
                    if older is not None and abs(last-older)<8:
                        pred=track[last]+(track[last]-track[older])*(idx-last)/(last-older)
                    else:pred=track[last]
                    v=float(choices[int(np.argmin(np.abs(choices-pred)))])
                    # Large abrupt jumps signal legend/stray pixels or lost
                    # tracks; do not attach them to the physical curve.
                    tolerance_px=max(9.,h*.022+(idx-last)*h*.006)
                    if abs(v-pred)<=tolerance_px:
                        track[idx]=v;observed[idx]=True
                        older,last=last,idx
                idx+=direction
        observed_count=int(observed.sum())
        if observed_count==0:raise ValueError(f"No visible pixels for band {j+1}")
        fraction=observed_count/len(x)
        missing=_runs(~observed)
        longest=max((stop-begin for begin,stop in missing),default=0)/len(x)
        multi=float(np.mean(count>1))
        if fraction<min_coverage or longest>max_gap_fraction or multi>max_multi_fraction:
            raise ValueError(
                f"Band {j+1} unreliable: observed={fraction:.1%}, "
                f"longest-hidden={longest:.1%}, multiple-components={multi:.1%}. "
                f"Try more accurate plot bounds/color or provide digitized CSV.")
        track=np.interp(x,x[observed],track[observed])
        traces[:,j]=emax-track/(h-1)*(emax-emin)
        quality.append(dict(band=j+1,observed_fraction=fraction,
            longest_missing_fraction=longest,multiple_component_fraction=multi,
            interpolated_points=int((~observed).sum()),observed_points=observed_count,
            observed_mask=observed.tolist(),
            antialias_white_blend_used=anti_alias_used))
    # Ambiguous pixels claimed by multiple colors can arise if chosen
    # colors are too similar or tolerance too liberal.
    if len(masks)>1:
        overlap=np.sum(np.stack(masks,axis=0),axis=0)
        occupied=np.count_nonzero(overlap>0)
        ambiguity=float(np.count_nonzero(overlap>1)/max(occupied,1))
        if ambiguity>.025:
            raise ValueError(f"RGB color masks overlap at {ambiguity:.1%} of band pixels; "
                             "reduce color tolerance or provide separate curves.")
    else:ambiguity=0.0
    return x,np.sort(traces,axis=1),dict(
        method="rgb_continuity_v2",bands=quality,
        minimum_observed_fraction=min(i["observed_fraction"] for i in quality),
        largest_missing_fraction=max(i["longest_missing_fraction"] for i in quality),
        max_multiple_component_fraction=max(i["multiple_component_fraction"] for i in quality),
        pixel_color_overlap_fraction=ambiguity,
        warning="Visible pixels were traced; any hidden fragments are interpolated, not observed.")

def mono_quality_gate(image,n,*,coverage=.75,max_gap_fraction=.065):
    """After legacy mono tracing, audit independently visible curve counts."""
    counts=np.asarray(image,int)
    reliable=(counts>=n)
    missing=_runs(~reliable)
    largest=max((b-a for a,b in missing),default=0)/max(len(counts),1)
    fraction=float(np.mean(reliable))
    if fraction<coverage or largest>max_gap_fraction:
        raise ValueError(f"Monochrome image underdetermined: N visible traces in only "
                         f"{fraction:.1%} of columns; largest ambiguous stretch "
                         f"{largest:.1%}. Use CSV or distinct colors.")
    return dict(visible_fraction=fraction,largest_ambiguous_fraction=largest,
                warning="N monochrome traces visible; crossings may still be intrinsically ambiguous.")
