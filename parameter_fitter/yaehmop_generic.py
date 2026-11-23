#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import copy,re,subprocess,os
import numpy as np

HERE=Path(__file__).resolve().parent

# Periodic images per ACTIVE lattice direction, shared by fitting and
# recomputation so that the fitted and the recomputed Hamiltonians match.
# 5 converges carbon test bands to ~1e-5 eV (4 leaves ~1e-3 eV errors).
DEFAULT_OVERLAP_RADIUS=5

def overlap_images_for(periodic_dim,radius=DEFAULT_OVERLAP_RADIUS):
    """Overlap images (a,b,c): `radius` along active directions, 0 elsewhere."""
    periodic_dim=int(periodic_dim); radius=int(radius)
    if periodic_dim not in (1,2,3):
        raise ValueError("periodic_dim must be 1, 2, or 3")
    if not 1<=radius<=8:
        raise ValueError("overlap radius must be in 1..8")
    return tuple(radius if j<periodic_dim else 0 for j in range(3))

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
    # one BLAS/LAPACK thread per bind run: the fitters already run several systems in parallel,
    # and a threaded MKL/OpenBLAS build would oversubscribe the CPUs (override via the environment)
    env=dict(os.environ)
    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):env.setdefault(var,"1")
    proc=subprocess.run([str(bind),str(input_file),str(param_file)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    if log_file is not None: Path(log_file).write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"YAeHMOP bind failed ({proc.returncode}); see {log_file}.\n{proc.stdout[-4000:]}")
    check_double_zeta_sp(bind,input_file,param_file)
    return proc

_FEATURES={}

def bind_features(bind):
    """Options of a bind binary: {'soc', 'spin', 'common_fermi'} for the fork github.com/ac-dias/yaehmop
    (bind --help lists --soc / --parm-up / --common-fermi), empty for the original program."""
    key=str(bind)
    if key not in _FEATURES:
        try:
            out=subprocess.run([key,"--help"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20).stdout
        except (OSError,subprocess.SubprocessError):
            out=""
        _FEATURES[key]={f for f,flag in (("soc","--soc"),("spin","--parm-up"),("common_fermi","--common-fermi"))
                        if flag in out}
    return _FEATURES[key]

def run_bind_args(bind, args, input_file, param_file=None, log_file=None):
    """bind with extra options of the fork (e.g. --soc FILE, --parm-up/--parm-down); one BLAS thread."""
    env=dict(os.environ)
    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):env.setdefault(var,"1")
    cmd=[str(bind)]+[str(a) for a in args]+[str(input_file)]+([str(param_file)] if param_file else [])
    proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    if log_file is not None: Path(log_file).write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"YAeHMOP bind failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-3000:]}")
    return proc

def _double_zeta_sp_shells(param_file):
    """(element, shell) of the s/p lines with Nzeta = 2 and a nonzero second coefficient."""
    out=[]
    for raw in Path(param_file).read_text().splitlines():
        f=raw.split()
        if len(f)>=11 and not raw.lstrip().startswith(';') and f[5].lower() in ('s','p'):
            try:
                if int(f[3])==2 and abs(float(f[10]))>0:out.append((f[0],f[5].lower()))
            except ValueError:
                pass
    return out

def check_double_zeta_sp(bind,input_file,param_file):
    """The original YAeHMOP reads double-zeta s/p lines but uses only the first exponent,
    silently. A bind with double-zeta p (github.com/ac-dias/yaehmop) lists such a shell as
    '3P DZ:' in the .out; refuse the result when that line is missing."""
    dz=_double_zeta_sp_shells(param_file)
    if not dz:return
    out=Path(str(input_file)+'.out')
    if not out.is_file():return
    text=out.read_text(errors='replace')
    if 'P DZ:' not in text and 'S DZ:' not in text:
        raise RuntimeError(f"{bind} ignores the second exponent of the double-zeta shells {dz}: "
                           "use a bind with double-zeta p support (github.com/ac-dias/yaehmop, commit "
                           "742719c or later) through YAEHMOP_BIND")

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

def contiguous_along_vacuum(atoms,periodic_dim):
    """Copy of atoms with the slab (2D) or wire (1D) made contiguous along the
    non-periodic lattice directions.

    A DFT code is periodic in 3D, so an atom of a slab may sit at z = 0.9999 instead of
    z = -0.0001 (wrapped across the vacuum). YAeHMOP has no images along the inactive
    directions, so such an atom would be isolated from its layer. Along each inactive axis
    the fractional coordinates are cut at the largest empty interval (the vacuum)."""
    a=atoms.copy()
    if periodic_dim>=3 or len(a)<2:
        return a
    frac=a.get_scaled_positions(wrap=False)
    for ax in range(periodic_dim,3):
        z=np.mod(frac[:,ax],1.0);s=np.sort(z)
        gaps=np.append(np.diff(s),s[0]+1.0-s[-1])
        i=int(np.argmax(gaps));start=s[(i+1)%len(s)]
        new=np.mod(z-start,1.0)+start
        # whole-cell translation back near the original coordinates (no physical change)
        frac[:,ax]=new+np.round(np.mean(frac[:,ax])-np.mean(new))
    a.set_scaled_positions(frac)
    return a

def write_periodic_input_generic(cif_path,out_path,band_path,k_const=1.75,weighted_hij=True,electrons=None,overlap_images=(5,5,0),periodic_dim=2,structure_format=None,kpoints=None,average_properties=False):
    """cif_path: any structure file read by structure_io (CIF, VASP, QE, SIESTA, FHI-aims)."""
    from structure_io import read_structure
    cif_path,out_path=Path(cif_path),Path(out_path)
    if periodic_dim not in (1,2,3):
        raise ValueError("periodic_dim must be 1, 2, or 3")
    atoms=contiguous_along_vacuum(read_structure(cif_path,structure_format),periodic_dim)
    frac=atoms.get_scaled_positions(wrap=False); syms=atoms.get_chemical_symbols(); nphys=len(atoms)
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
    if kpoints is None:
        lines += ['','Band',str(band_path.points_per_line),str(len(band_path.labels))]
        for label,k in zip(band_path.labels,band_path.points): lines.append(f"{label:3s} {k[0]: .12f} {k[1]: .12f} {k[2]: .12f}")
    else:
        # Explicit k list: YAeHMOP writes H(k) and S(k) (Dump Hamil / Dump Overlap)
        kpoints=np.atleast_2d(np.asarray(kpoints,float))
        lines += ['','Dump Hamil','Dump Overlap','','K points',str(len(kpoints))]
        lines.extend(f"{k[0]: .12f} {k[1]: .12f} {k[2]: .12f} 1.0" for k in kpoints)
    if electrons is not None: lines += ['','electrons',str(int(electrons))]
    if average_properties: lines += ['','Average Properties']
    lines.append(''); out_path.write_text('\n'.join(lines)); return out_path


def orbital_labels(params,symbols):
    """Atom/orbital label of every basis function, in YAeHMOP order (s, p, d, f)."""
    shapes={"s":["s"],"p":["px","py","pz"],"d":["dx2-y2","dz2","dxy","dxz","dyz"],
            "f":["fz3","fxz2","fyz2","fxyz","fz(x2-y2)","fx(x2-3y2)","fy(3x2-y2)"]}
    out=[]
    for i,s in enumerate(symbols,1):
        for shell in ("s","p","d","f"):
            if shell in params[s]:
                out += [f"{s}{i}:{o}" for o in shapes[shell]]
    return out

def read_matrix_dump(path):
    """YAeHMOP .HAM/.OV binary dump -> complex array (nk, n, n).

    File: int nk, int n, then per k an n*n double array (row major) holding the
    real part of element (l,m), l<=m, at [l,m] and an imaginary part at [m,l].
    Returned in the usual convention H(k) = sum_R exp(+i k.R) H(R), with
    H_ij(R) = <phi_i(0)|H|phi_j(R)> (checked against interatomic distances).
    """
    raw=Path(path).read_bytes()
    nk,n=np.frombuffer(raw[:8],dtype=np.int32)
    data=np.frombuffer(raw[8:8+8*nk*n*n],dtype=np.float64).reshape(nk,n,n)
    iu=np.triu_indices(n,1)
    out=np.zeros((nk,n,n),complex)
    for q in range(nk):
        m=data[q]
        up=m[iu]-1j*m.T[iu]
        H=np.diag(np.diag(m)).astype(complex)
        H[iu]=up
        H[(iu[1],iu[0])]=np.conj(up)
        out[q]=H
    return out


ORBITAL_LM={"s":(0,0),"px":(1,1),"py":(1,-1),"pz":(1,0),
    "dx2-y2":(2,2),"dz2":(2,0),"dxy":(2,-2),"dxz":(2,1),"dyz":(2,-1),
    "fz3":(3,0),"fxz2":(3,1),"fyz2":(3,-1),"fxyz":(3,-2),"fz(x2-y2)":(3,2),
    "fx(x2-3y2)":(3,3),"fy(3x2-y2)":(3,-3)}

def real_space_matrices(Hk,Sk,mesh,dim,radius):
    """H(R), S(R) for the cells -radius..radius from H(k), S(k) on the full
    Gamma-centred mesh of (2*radius+1)^dim points (exact for YAeHMOP, whose
    lattice sums stop at `radius` cells)."""
    import itertools
    rng=[range(-radius,radius+1) if i<dim else (0,) for i in range(3)]
    cells=[np.array(R) for R in itertools.product(*rng)]
    out=[]
    for R in cells:
        ph=np.exp(-2j*np.pi*(mesh@R))[:,None,None]
        out.append((R,(ph*Hk).sum(0)/len(mesh),(ph*Sk).sum(0)/len(mesh)))
    return out

def gamma_mesh(dim,radius):
    import itertools
    n=2*radius+1
    axes=[np.arange(n)/n if i<dim else np.zeros(1) for i in range(3)]
    return np.array(list(itertools.product(*axes)))
