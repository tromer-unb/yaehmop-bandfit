#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

def detect_axes_bbox(rgb):
    # Find long near-black frame lines; generated/reference plots use a full rectangular frame.
    dark=(rgb[:,:,0]<70)&(rgb[:,:,1]<70)&(rgb[:,:,2]<70)
    rs=dark.sum(1); cs=dark.sum(0); h,w=dark.shape
    rows=np.where(rs>0.45*w)[0]; cols=np.where(cs>0.45*h)[0]
    if len(rows)<2 or len(cols)<2:
        raise ValueError('Could not auto-detect plot frame; pass explicit bbox metadata.')
    return int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max())

def load_band_image(path,emin,emax,bbox=None):
    rgb=np.asarray(Image.open(path).convert('RGB'))
    if bbox is None: bbox=detect_axes_bbox(rgb)
    left,top,right,bottom=map(int,bbox)
    crop=rgb[top:bottom+1,left:right+1]
    r,g,b=[crop[:,:,i].astype(int) for i in range(3)]
    # Preferred colored-band mask: saturated blue pixels. Fallback handled below.
    mask=(b-r>55)&(b-g>25)&(b>100)
    if mask.sum()<100:
        # generic dark-line fallback; exclude outer frame by one pixel
        mask=(r<90)&(g<90)&(b<90)
        mask[[0,-1],:]=False; mask[:,[0,-1]]=False
    h,w=mask.shape
    def px_to_energy(py): return emax-(py/(h-1))*(emax-emin)
    return {'rgb':rgb,'mask':mask,'bbox':(left,top,right,bottom),'emin':float(emin),'emax':float(emax),
            'width':w,'height':h,'px_to_energy':px_to_energy}

def energies_at_x(target,xnorm,half_window=1,cluster_gap_px=3):
    mask=target['mask']; w=target['width']
    col=int(round(float(xnorm)*(w-1))); a=max(0,col-half_window); z=min(w,col+half_window+1)
    ys=np.where(mask[:,a:z])[0]
    if len(ys)==0: return np.empty(0)
    ys=np.unique(ys); groups=[[ys[0]]]
    for y in ys[1:]:
        if y-groups[-1][-1] <= cluster_gap_px: groups[-1].append(y)
        else: groups.append([y])
    py=np.array([np.mean(g) for g in groups])
    return np.sort(np.asarray([target['px_to_energy'](v) for v in py],float))

def image_gap(target,nx=600,fermi=0.0):
    vmax=-np.inf; cmin=np.inf
    for x in np.linspace(0,1,nx):
        e=energies_at_x(target,x,half_window=1)
        neg=e[e<=fermi]; pos=e[e>fermi]
        if len(neg): vmax=max(vmax,float(neg.max()))
        if len(pos): cmin=min(cmin,float(pos.min()))
    return vmax,cmin,cmin-vmax

def _expand_candidates_monotone(pred,cands,n):
    pred=np.asarray(pred,float); c=np.asarray(cands,float); m=len(c)
    if m==0: return pred.copy()
    c=np.sort(c)
    if m>n:
        # Rare raster fragmentation: monotone nearest-neighbour subset via DP.
        dp=np.full((n+1,m+1),np.inf); take=np.zeros((n+1,m+1),bool); dp[0,:]=0.0
        for i in range(1,n+1):
            for j in range(1,m+1):
                skip=dp[i,j-1]
                use=dp[i-1,j-1]+(pred[i-1]-c[j-1])**2
                if use<skip: dp[i,j]=use; take[i,j]=True
                else: dp[i,j]=skip
        out=[]; i=n; j=m
        while i>0 and j>0:
            if take[i,j]: out.append(c[j-1]); i-=1; j-=1
            else: j-=1
        return np.asarray(out[::-1],float)
    # Partition n ordered tracks into m non-empty consecutive groups; each group maps to one observed curve.
    cs=[]
    for val in c:
        d=(pred-val)**2; cs.append(np.r_[0.0,np.cumsum(d)])
    dp=np.full((m+1,n+1),np.inf); prev=np.full((m+1,n+1),-1,int); dp[0,0]=0.0
    for j in range(1,m+1):
        min_i=j; max_i=n-(m-j)
        for i in range(min_i,max_i+1):
            for q in range(j-1,i):
                cost=dp[j-1,q]+(cs[j-1][i]-cs[j-1][q])
                if cost<dp[j,i]: dp[j,i]=cost; prev[j,i]=q
    groups=[]; i=n
    for j in range(m,0,-1):
        q=prev[j,i]
        if q<0: return np.interp(np.arange(n),np.linspace(0,n-1,m),c)
        groups.append((q,i,c[j-1])); i=q
    out=np.empty(n,float)
    for q,i,val in groups[::-1]: out[q:i]=val
    return out

def trace_bands(target,nbands,nx=401,half_window=1,cluster_gap_px=3):
    xs=np.linspace(0.0,1.0,int(nx)); candidates=[energies_at_x(target,x,half_window,cluster_gap_px) for x in xs]
    counts=np.array([len(v) for v in candidates]); exact=np.where(counts==nbands)[0]
    if not len(exact):
        seed=int(np.argmax(counts))
        if counts[seed]==0: raise ValueError('No band pixels found')
        base=np.interp(np.arange(nbands),np.linspace(0,nbands-1,len(candidates[seed])),candidates[seed])
    else:
        seed=int(exact[np.argmin(np.abs(exact-(len(xs)-1)/2))]); base=np.asarray(candidates[seed],float)
    tracks=np.full((len(xs),nbands),np.nan); tracks[seed]=base
    # forward/backward, with one-step linear prediction to preserve multiplicities through crossings/degeneracies
    for direction in (1,-1):
        i=seed+direction; prev1=tracks[seed].copy(); prev2=None
        while 0<=i<len(xs):
            pred=prev1 if prev2 is None else prev1+(prev1-prev2)
            cand=candidates[i]
            cur=_expand_candidates_monotone(pred,cand,nbands) if len(cand) else prev1.copy()
            tracks[i]=np.sort(cur); prev2,prev1=prev1,tracks[i].copy(); i+=direction
    return xs,tracks,counts,seed
