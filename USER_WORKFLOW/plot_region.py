#!/usr/bin/env python3
"""Heuristic bounding-box recovery for rectangular or L-axis plots.

Detection is merely a pixel-coordinate suggestion. Every automatic
bounding box must be visualized in inspect_bbox.png before fitting.
"""
import numpy as np
def detect_full_frame(rgb):
    dark=np.max(np.asarray(rgb).astype(np.int16),axis=2)<70
    h,w=dark.shape
    rows=np.flatnonzero(dark.sum(axis=1)>.45*w)
    cols=np.flatnonzero(dark.sum(axis=0)>.45*h)
    if len(rows)<2 or len(cols)<2:
        raise ValueError("No rectangular frame detected")
    bounds=(int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max()))
    if bounds[2]-bounds[0]<=30 or bounds[3]-bounds[1]<=30:
        raise ValueError("Incomplete L-axis not a rectangle")
    return bounds

def longest_run(condition):
    arr=np.asarray(condition,bool)
    changes=np.diff(np.r_[False,arr,False].astype(np.int8))
    starts=np.flatnonzero(changes==1)
    ends=np.flatnonzero(changes==-1)
    if not len(starts):return (0,0,0)
    idx=int(np.argmax(ends-starts))
    return int(starts[idx]),int(ends[idx]),int(ends[idx]-starts[idx])

def infer_plot_bbox(rgb):
    try:
        box=detect_full_frame(rgb)
        # Two adjacent pixels of an L-shaped axis are NOT a full frame.
        if box[2]-box[0]>30 and box[3]-box[1]>30:
            return box,"full-frame"
    except ValueError:
        pass
    a=np.asarray(rgb)
    if a.ndim!=3 or a.shape[2]!=3:raise ValueError("Requires RGB bitmap")
    h,w=a.shape[:2]
    black=np.max(a,axis=2)<105
    candidates_h=[]
    for row in range(max(0,int(h*.18)),int(h*.90)):
        start,end,length=longest_run(black[row])
        if length>w*.32:
            visible=np.flatnonzero(black[row])
            candidates_h.append((row,int(visible[0]),int(visible[-1])+1,length))
    candidates_v=[]
    for col in range(max(0,int(w*.05)),int(w*.70)):
        start,end,length=longest_run(black[:,col])
        if length>h*.32:
            visible=np.flatnonzero(black[:,col])
            candidates_v.append((col,int(visible[0]),int(visible[-1])+1,length))
    if not candidates_h or not candidates_v:
        raise ValueError("No reliable rectangular/L-shaped axes detected; "
                         "use --bbox L,T,R,B from the image pixel coordinates.")
    # Favor the bottom horizontal and left vertical plot-axis pair,
    # requiring their intersection to lie within 3px.
    pairs=[]
    for row,x0,x1,wide in candidates_h:
        for col,y0,y1,tall in candidates_v:
            if abs(col-x0)>5 or abs(row-y1)>5:
                continue
            if not (col+20<x1 and y0+20<row):
                continue
            score=(wide/w)+(tall/h)+.05*(row/h)
            pairs.append((score,(col,y0,x1-1,row)))
    if not pairs:
        raise ValueError("Detected horizontal/vertical line segments do not "
                         "form a credible plot axis intersection; provide --bbox.")
    _,bounds=max(pairs,key=lambda item:item[0])
    return tuple(int(v) for v in bounds), "left-bottom-axes"
