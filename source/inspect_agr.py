#!/usr/bin/env python3
from pathlib import Path
import numpy as np,re
from ase.io import read
from ase.formula import Formula
R=Path(__file__).resolve().parents[1]
def parse(path):
 txt=Path(path).read_text().splitlines()
 pointline=next(l for l in txt if re.match(r'^#\s+(\d+)\s+(\d+)',l))
 points,nband=map(int,re.findall(r'\d+',pointline)[:2])
 labels=[]
 for l in txt:
  m=re.match(r'^# \t([A-Z][A-Z0-9]*)\t([\d.]+)',l)
  if m:labels.append((m.group(1),float(m.group(2))))
 datasets=[];current=[];started=False
 for raw in txt:
  line=raw.strip()
  if line.startswith('@target'):
   if current:datasets.append(np.asarray(current,float))
   current=[];started=True
  elif started:
   if line=='&':
    if current:datasets.append(np.asarray(current,float));current=[];started=False
   elif line and not line.startswith(('@','#')):
    s=line.split()
    if len(s)>=2:
     try:current.append([float(s[0]),float(s[1])])
     except ValueError:pass
 if current:datasets.append(np.asarray(current,float))
 assert len(datasets)==nband,(len(datasets),nband)
 assert all(len(v)==points for v in datasets),(points,[len(v) for v in datasets])
 x=datasets[0][:,0]
 E=np.stack([v[:,1] for v in datasets],axis=1)
 assert all(np.allclose(d[:,0],x,atol=1e-7) for d in datasets)
 assert np.all(np.diff(E,axis=1)>=-1e-5),'not sorted'
 return x,E,labels
if __name__=="__main__":
 for p in (R/'external_references').glob('*/reference.agr'):
  x,E,L=parse(p)
  neg=(E<0).sum(1)
  c=read(p.parent/'structure.cif')
  print(p.parent.name,'npoints',len(x),'nbands',E.shape[1],'counts below0 range',min(neg),max(neg),
        'negative max',E[E<0].max() if (E<0).any() else None,
        'positive min',E[E>=0].min() if (E>=0).any() else None,'Energy range',np.round([E.min(),E.max()],2),
        'labels',L,'atoms',len(c))
