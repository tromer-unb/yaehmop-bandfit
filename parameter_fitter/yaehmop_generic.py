#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import copy,re,subprocess,os
import numpy as np

HERE=Path(__file__).resolve().parent

@dataclass(frozen=True)
class BandPath:
    labels: tuple[str, ...]
    points: np.ndarray
    points_per_line: int = 100

def resolve_yaehmop_assets():
    env_bind=os.environ.get("YAEHMOP_BIND")
    env_parms=os.environ.get("YAEHMOP_PARAM_FILE")
    bind_candidates=[Path(env_bind).expanduser()] if env_bind else [HERE/"yaehmop"/"bind", HERE.parent/"reference"/"yaehmop"/"tightbind"/"bind"]
    parm_candidates=[Path(env_parms).expanduser()] if env_parms else [HERE/"yaehmop"/"eht_parms.dat", HERE.parent/"reference"/"yaehmop"/"tightbind"/"eht_parms.dat"]
    bind=next((q.resolve() for q in bind_candidates if q.is_file()),None)
    parms=next((q.resolve() for q in parm_candidates if q.is_file()),None)
    if bind is None:
        raise FileNotFoundError("YAeHMOP bind not found. Put it in ./yaehmop/bind or set YAEHMOP_BIND.")
    if parms is None:
        raise FileNotFoundError("YAeHMOP parameter table not found. Put it in ./yaehmop/eht_parms.dat or set YAEHMOP_PARAM_FILE.")
    return bind,parms

def run_bind(bind, input_file, param_file, log_file=None):
    bind,input_file,param_file=map(Path,(bind,input_file,param_file))
    proc=subprocess.run([str(bind),str(input_file),str(param_file)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if log_file is not None: Path(log_file).write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"YAeHMOP bind failed ({proc.returncode}); see {log_file}.\n{proc.stdout[-4000:]}")
    return proc

def parse_band(path):
    path=Path(path); lines=path.read_text().splitlines(); norb=None
    for line in lines:
        m=re.match(r"\s*(\d+) orbitals in the unit cell",line)
        if m: norb=int(m.group(1)); break
    if norb is None: raise ValueError(f"Could not find orbital count in {path}")
    kpts=[]; energies=[]; i=0
    while i<len(lines):
        if lines[i].startswith("; K point:"):
            toks=lines[i].split(":",1)[1].split(); kpts.append([float(x) for x in toks[:3]])
            energies.append([float(lines[i+1+j].strip()) for j in range(norb)]); i+=norb+1
        else: i+=1
    if not energies: raise ValueError(f"No band energies found in {path}")
    return np.asarray(kpts),np.asarray(energies)

def normalized_path_coordinate(kpts_frac,cell):
    recip=np.asarray(cell.reciprocal()); cart=np.asarray(kpts_frac)@recip
    ds=np.linalg.norm(np.diff(cart,axis=0),axis=1); x=np.r_[0.0,np.cumsum(ds)]
    if x[-1] == 0: raise ValueError("Band path has zero total length")
    return x/x[-1]

def parse_parameter_table(path):
    table={}
    for raw in Path(path).read_text().splitlines():
        s=raw.strip()
        if not s or s.startswith(';') or s.upper()=='END': continue
        f=s.split()
        if len(f)<11: continue
        try:
            el=f[0].capitalize(); atnum=int(f[1]); nval=int(f[2]); nzeta=int(f[3]); n=int(f[4]); orb=f[5].lower()
            vals=list(map(float,f[6:11]))
        except Exception: continue
        table.setdefault(el,{'atnum':atnum,'nval':nval})[orb]={
            'nzeta':nzeta,'n':n,'Hii':vals[0],'zeta1':vals[1],'zeta2':vals[2],'c1':vals[3],'c2':vals[4]}
    return table

def subset_for_elements(table,elements):
    out={}
    for el in elements:
        key=el.capitalize()
        if key not in table: raise KeyError(f'No YAeHMOP parameters for {el}')
        out[key]=copy.deepcopy(table[key])
    return out

def orbital_count(params, symbols):
    degeneracy={'s':1,'p':3,'d':5,'f':7}
    return sum(sum(degeneracy[o] for o in params[s.capitalize()] if o in degeneracy) for s in symbols)

def valence_electrons(params,symbols):
    return sum(int(params[s.capitalize()]['nval']) for s in symbols)

def write_parameter_file(path,params):
    lines=['; Generic YAeHMOP parameter file']
    for el,spec in params.items():
        for orb in ('s','p','d','f'):
            if orb not in spec: continue
            o=spec[orb]
            lines.append(f"{el.upper():<2s} {spec['atnum']:5d} {spec['nval']:5d} {o['nzeta']:5d} {o['n']:5d} {orb:>2s} "
                         f"{o['Hii']:10.6f} {o['zeta1']:10.6f} {o['zeta2']:10.6f} {o['c1']:10.6f} {o['c2']:10.6f}")
    lines.append('END')
    Path(path).write_text('\n'.join(lines)+'\n')
    return Path(path)

def write_periodic_input_generic(cif_path,out_path,band_path,k_const=1.75,weighted_hij=True,electrons=None,overlap_images=(5,5,0),periodic_dim=2):
    from ase.io import read
    cif_path,out_path=Path(cif_path),Path(out_path)
    atoms=read(cif_path); frac=atoms.get_scaled_positions(wrap=False); syms=atoms.get_chemical_symbols(); nphys=len(atoms)
    if periodic_dim not in (1,2,3):
        raise ValueError("periodic_dim must be 1, 2, or 3")
    if len(overlap_images)!=3 or any(int(v)<0 for v in overlap_images):
        raise ValueError("overlap_images must contain three nonnegative integers")
    if any(int(v)!=0 for v in overlap_images[periodic_dim:]):
        raise ValueError("Inactive lattice directions must have zero overlap images")
    if min(int(v) for v in overlap_images[:periodic_dim])<1:
        raise ValueError("Active lattice directions require >=1 periodic image")
    # Explicit 1D/2D/3D periodicity; do not silently write Lattice 2 for diamond.
    anchor=frac[0]
    helpers=[anchor+np.eye(3)[j] for j in range(periodic_dim)]
    lengths=atoms.cell.lengths(); angles=atoms.cell.angles()
    lines=['; Generic YAeHMOP periodicity-aware input',
           f'Generic EHT: {atoms.get_chemical_formula()}', '',
           'Geometry Crystallographic',str(nphys+periodic_dim)]
    for i,(sym,xyz) in enumerate(zip(syms,frac),1):
        lines.append(f"{i:3d} {sym:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    for j,xyz in enumerate(helpers,nphys+1):
        lines.append(f"{j:3d} {syms[0]:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    lines += ['', 'Lattice',str(periodic_dim),
              ' '.join(str(int(v)) for v in overlap_images)]
    lines.extend(f'1 {nphys+1+j}' for j in range(periodic_dim))
    lines += ['', 'Crystal Spec',
              f'{lengths[0]:.12f} {lengths[1]:.12f} {lengths[2]:.12f}',
              f'{angles[0]:.12f} {angles[1]:.12f} {angles[2]:.12f}',
              '', f'THE CONST {k_const:.12f}']
    if not weighted_hij: lines.append('NONWEIGHTED')
    lines += ['','Band',str(band_path.points_per_line),str(len(band_path.labels))]
    for label,k in zip(band_path.labels,band_path.points): lines.append(f"{label:3s} {k[0]: .12f} {k[1]: .12f} {k[2]: .12f}")
    if electrons is not None: lines += ['','electrons',str(int(electrons))]
    lines.append(''); out_path.write_text('\n'.join(lines)); return out_path
