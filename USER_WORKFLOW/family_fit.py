#!/usr/bin/env python3
"""One shared EHT parameter table (and one K) fitted to several systems at once.

Every reference that bands_fit.py accepts can be a member of the family, and the
members may be mixed freely:

* raw DFT band folders (VASP, Quantum ESPRESSO, SIESTA, FHI-aims, Wannier90):
  gapped references use the band-window objective, metallic ones the metal
  objective; spin-polarized runs (one channel or the mean of both), semicore
  bands dropped automatically, user Fermi level, explicit valence count;
* band images (PNG/JPEG/TIFF/WebP) or numerical CSV with a structure file and an
  INI with the energy axis and k path (gapped, 2/4/6/8 frontier bands).

Each system's "structure" may be a CIF, VASP, Quantum ESPRESSO, SIESTA or
FHI-aims file (optional per-system "structure_format"); DFT systems may omit
it (the structure of the DFT run is used).

Roles: "train" systems enter the optimization; "test" systems (optional) are
opened only after the shared vector is frozen and scored without refitting.
After the fit every system gets its comparison figures (DFT/reference blue,
EHT red dashed) and, for DFT systems, the standard fitting/ export (parameters,
H(R), S(R)) in OUTDIR/systems/<name>/.

python family_fit.py family.yaml --validate-only
python family_fit.py family.yaml --prepare-only --outdir out/family
python family_fit.py family.yaml --fit --budget 2000 --optimizer cmaes --polish nelder-mead --outdir out/family
python family_fit.py family.yaml --evaluate out/family/fit/fit.json --outdir out/eval
"""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
import yaml
from bands_fit import (conf,read_nodes,csv_path,load_dft_reference,run_optimizer,write_bands_plot,
    write_full_comparison,write_metal_plots,write_fitting_dir,path_kpoints,node_positions,
    evaluate_dft,evaluate_metal)
from prepare_band_png import prepare,parse_bbox
from frozen_transfer import evaluate,sha
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'parameter_fitter'))
from yaehmop_generic import DEFAULT_OVERLAP_RADIUS
from structure_io import read_structure,resolved_structure_format
from dft_bands import normalize_code
from global_optimizers import GLOBAL_METHODS,POLISH_METHODS
from band_objective import WindowSettings,OBJECTIVES

# keys of a DFT system (in the manifest row, in `defaults:` or in the INI [reference]/[band_path])
DFT_KEYS=('dft_dir','dft_code','dft_file','valence_bands','spin_channel','fermi','metallic','semicore',
          'dimension','points_per_segment','emin','emax')
OPTIMIZERS=('nelder-mead','simulated-annealing','sa-nelder-mead','hooke-jeeves')+tuple(GLOBAL_METHODS)

def _structure_candidates(folder,code):
    """Structure files of a DFT folder (only to know its elements before the fit;
    the band data themselves are not read)."""
    pats={'vasp':('CONTCAR','POSCAR','vasprun.xml'),'qe':('*.in','*.pwi','*.out'),
          'siesta':('*.STRUCT_OUT','*.XV','*.fdf'),'aims':('geometry.in',),'wannier90':()}
    for pat in pats.get(code,()):
        for f in sorted(folder.glob(pat)):
            if f.is_file() and f.stat().st_size>0:yield f

def _dft_symbols(folder,code,structure,sfmt):
    if structure is not None:
        return read_structure(structure,sfmt).get_chemical_symbols(),Path(structure)
    for f in _structure_candidates(folder,code):
        try:return read_structure(f,None).get_chemical_symbols(),f
        except Exception:continue
    return None,None

def _row_value(row,defaults,cfg,key):
    if key in row and row[key] is not None:return row[key]
    if cfg is not None:
        for sec in ('reference','band_path'):
            if cfg.has_section(sec) and cfg[sec].get(key) not in (None,''):return cfg[sec].get(key)
    return defaults.get(key)

def load_manifest(path):
    """Validate the manifest; returns (path, manifest, train rows, test rows).
    No band data (DFT eigenvalues or images) are read here."""
    path=Path(path).resolve()
    a=yaml.safe_load(path.read_text())
    if not isinstance(a,dict) or not isinstance(a.get('systems'),list):
        raise ValueError('Manifest requires systems: list')
    if int(a.get('n_pairs',1)) not in (1,2,3,4):raise ValueError('n_pairs must be 1..4')
    defaults=dict(a.get('defaults') or {})
    unknown=set(defaults)-set(DFT_KEYS)-{'structure_format','mode','weight'}
    if unknown:raise ValueError(f'unknown keys in defaults: {sorted(unknown)}')
    if defaults.get('dft_dir'):raise ValueError('dft_dir cannot be a default (one folder per system)')
    obj=a.get('objective')
    if obj is not None and obj not in OBJECTIVES:raise ValueError(f'objective must be one of {OBJECTIVES}')
    WindowSettings.from_dict(a.get('objective_settings'))
    names=set();train=[];test=[]
    for row in a['systems']:
        if not isinstance(row,dict):raise ValueError('Each system must be a mapping')
        if not row.get('name'):raise ValueError('System missing name')
        name=str(row['name'])
        if name in names or '/' in name or '\\' in name or '..' in name:
            raise ValueError('Repeated or invalid system name '+name)
        names.add(name)
        role=str(row.get('role','train')).lower()
        if role not in ('train','test'):raise ValueError('role must be train or test')
        ini=(path.parent/str(row['settings'])).resolve() if row.get('settings') else None
        if ini is not None and not ini.is_file():raise FileNotFoundError('INI missing for '+name+': '+str(ini))
        cfg=conf(ini,required=()) if ini is not None else None
        base_ini=ini.parent if ini is not None else path.parent
        weight=float(row.get('weight',defaults.get('weight',1.0)))
        if weight<=0:raise ValueError('Weight must be positive for '+name)
        sfmt_req=row.get('structure_format',defaults.get('structure_format'))
        cif=(path.parent/str(row['structure'])).resolve() if row.get('structure') else None
        if cif is not None and not cif.is_file():raise FileNotFoundError('Structure missing for '+name+': '+str(cif))
        dft_dir=row.get('dft_dir')
        if dft_dir:dft_dir=csv_path(str(dft_dir),path.parent)
        elif cfg is not None and cfg['reference'].get('dft_dir'):dft_dir=csv_path(cfg['reference']['dft_dir'],base_ini)
        if dft_dir:
            if not Path(dft_dir).is_dir():raise FileNotFoundError(f'DFT folder missing for {name}: {dft_dir}')
            code=_row_value(row,defaults,cfg,'dft_code')
            if not code:raise ValueError(f'{name}: dft_code (vasp, qe, siesta, aims, wannier90) is required')
            code=normalize_code(str(code))
            vals={k:_row_value(row,defaults,cfg,k) for k in DFT_KEYS if k not in ('dft_dir','dft_code')}
            sfmt=resolved_structure_format(cif,sfmt_req) if cif is not None else None
            symbols,sfile=_dft_symbols(Path(dft_dir),code,cif,sfmt)
            entry={'name':name,'role':role,'kind':'dft','dft_dir':Path(dft_dir),'dft_code':code,
                   'cif':cif,'structure_format':sfmt,'structure_file':sfile,'ini':ini,'cfg':cfg,
                   'symbols':symbols,'weight':weight,**vals}
            if entry['spin_channel'] is not None and str(entry['spin_channel']).lower() not in ('up','down','average','mean'):
                raise ValueError(f'{name}: spin_channel must be up, down or average')
        else:
            for key in ('structure','settings'):
                if not row.get(key):raise ValueError(f'System {name} missing {key} (or give dft_dir)')
            read_nodes(cfg)
            img=(path.parent/str(row['reference'])).resolve() if row.get('reference') else csv_path(cfg['reference'].get('image',''),ini.parent)
            if not img.is_file():raise FileNotFoundError('Reference missing for '+name+': '+str(img))
            sfmt=resolved_structure_format(cif,sfmt_req)
            ref=cfg['reference']
            entry={'name':name,'role':role,'kind':'image','cif':cif,'structure_format':sfmt,'structure_file':cif,
                   'ini':ini,'cfg':cfg,'image':img,'symbols':read_structure(cif,sfmt).get_chemical_symbols(),
                   'weight':weight,'image_mode':str(row.get('mode',ref.get('mode',defaults.get('mode','auto')))),
                   'bbox':row.get('bbox',ref.get('bbox',fallback=None)),
                   'colors':row.get('colors',ref.get('colors',fallback=None))}
        (train if role=='train' else test).append(entry)
    if not train:raise ValueError('At least one training system is required')
    trained={e for row in train if row['symbols'] for e in row['symbols']}
    for s in test:
        if s['symbols'] is None:continue           # checked after the fit, when its structure is read
        unseen=set(s['symbols'])-trained
        if unseen:raise ValueError(f"Test {s['name']} contains elements absent from training: {sorted(unseen)}")
    thashes={sha(s['structure_file']) for s in train if s.get('structure_file')}
    tdirs={str(s['dft_dir'].resolve()) for s in train if s['kind']=='dft'}
    for s in test:
        if s.get('structure_file') and sha(s['structure_file']) in thashes:
            raise ValueError('Heldout structure file is identical to a training one: '+s['name'])
        if s['kind']=='dft' and str(s['dft_dir'].resolve()) in tdirs:
            raise ValueError('Heldout DFT folder is also a training folder: '+s['name'])
    return path,a,train,test

def _int(v):
    return None if v in (None,'') else int(v)

def _float(v):
    return None if v in (None,'') else float(v)

def load_dft(row,refdir,pairs):
    """Read and prepare one DFT member (bands_fit.load_dft_reference)."""
    return load_dft_reference(row['dft_dir'],row['dft_code'],refdir,name=row['name'],
        dft_file=row.get('dft_file'),valence_bands=_int(row.get('valence_bands')),
        spin_channel=row.get('spin_channel'),metallic=row.get('metallic') if row.get('metallic') is not None else 'auto',
        fermi=_float(row.get('fermi')),semicore=row.get('semicore') if row.get('semicore') is not None else 'auto',
        structure=row.get('cif'),structure_format=row.get('structure_format'),
        dimension=_int(row.get('dimension')),step=_int(row.get('points_per_segment')) or 20,pairs=pairs)

def preparation(row,out,pairs):
    """Training entry for the multi-system backend (prepares the reference)."""
    if row.get('kind')=='dft':
        L=load_dft(row,out/'prepared'/row['name'],pairs)
        row['loaded']=L
        return dict(L['entry'],weight=row['weight'])
    ref=row['cfg']['reference']
    prepared,qa=prepare(row['image'],out/'prepared'/row['name'],pairs,
        float(ref['emin']),float(ref['emax']),mode=row['image_mode'],
        bbox=parse_bbox(row['bbox']),colors=row['colors'],
        tolerance=ref.getint('color_tolerance',fallback=36),
        nx=ref.getint('trace_nx',fallback=301),
        min_coverage=ref.getfloat('min_coverage',fallback=.75))
    dim,nodes,labels,step=read_nodes(row['cfg'])
    return {'name':row['name'],'role':'train','weight':row['weight'],
       'structure':str(row['cif']),'structure_format':row['structure_format'],'reference':str(prepared),
       'emin':float(ref['emin']),'emax':float(ref['emax']),
       'point_dim':dim,'symmetry_points':nodes,'labels':labels,
       'n_pairs':pairs,'image_mode':'colored','points_per_line':step,
       'trace_nx':ref.getint('trace_nx',fallback=301)}

def minimum_staged_budget(n_active):
    """The staged native DE initializer needs two populations of 2*d calls,
    plus the stock point, and at least one local call. Check before launching."""
    return max(80,4*int(n_active)+2)

def _window(row):
    lo=_float(row.get('emin'));hi=_float(row.get('emax'))
    return (-6. if lo is None else lo,6. if hi is None else hi)

def _system_outputs(row,npz,sysdir,title,export,native,param_txt,K,radius,pairs):
    """Figures (+ fitting/ export) of one system evaluated with the frozen shared table."""
    sysdir.mkdir(parents=True,exist_ok=True)
    if row.get('kind')=='dft':
        L=row['loaded']
        if L['metallic']:
            m=write_metal_plots(npz,sysdir,title=title+' (metal)',emin=_float(row.get('emin')),emax=_float(row.get('emax')))
            metrics=dict(window_rmse_eV=m['rmse'],rms_within_1eV_of_EF_eV=m['near'],mae_eV=m['mae'],maxabs_eV=m['maxabs'])
        else:
            r=write_bands_plot(npz,sysdir,2*pairs,title=title+f' — {pairs}VB + {pairs}CB')
            write_full_comparison(npz,sysdir,title=title+' — all bands',emin=_window(row)[0],emax=_window(row)[1])
            metrics=dict(selected_rmse_eV=r)
        if export:
            er=L['er'];step=er.n
            d=np.load(npz)
            ez=float(d['eht_fermi']) if L['metallic'] and 'eht_fermi' in d else None
            write_fitting_dir(sysdir/'fitting',sysdir,structure=L['cif'],sfmt=L['sfmt'],native=native,
                param_txt=param_txt,K=K,kpoints=path_kpoints(er.nodes,step)[er.mask],dim=L['dim'],radius=radius,
                summary=dict(name=row['name'],role=row['role'],mode='shared family table',K_EHT=K,
                             dft_reference=L['info'],metrics=metrics),
                node_index=node_positions(len(er.nodes),step,er.mask),energy_zero=ez)
        return metrics
    r=write_bands_plot(npz,sysdir,2*pairs,title=title+f' — {pairs}VB + {pairs}CB')
    return dict(selected_rmse_eV=r)

def overview_figure(panels,out,title):
    """All systems in one figure: reference blue, EHT red dashed (relative to E_F or midgap)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=len(panels)
    if not n:return None
    nc=min(4,n);nr=(n+nc-1)//nc
    fig,axes=plt.subplots(nr,nc,figsize=(4.3*nc,3.6*nr),squeeze=False)
    for ax in axes.flat[n:]:ax.axis('off')
    for ax,(name,npz,role,metric) in zip(axes.flat,panels):
        d=np.load(npz);k=d['k']
        R=d['reference_all'] if 'reference_all' in d else d['reference']
        E=d['YAeHMOP_all'] if 'YAeHMOP_all' in d else d['YAeHMOP']
        metal='metallic' in d
        lo,hi=((float(d['window'][0])-.5,float(d['window'][1])+.5) if metal else (-6,6))
        for j in range(R.shape[1]):
            if R[:,j].max()<lo-1 or R[:,j].min()>hi+1:continue
            ax.plot(k,R[:,j],color='#1f5fbf',lw=.9)
        for j in range(E.shape[1]):
            if E[:,j].max()<lo-1 or E[:,j].min()>hi+1:continue
            ax.plot(k,E[:,j],color='#d62728',lw=.85,ls='--')
        if 'tick_x' in d:
            for x in d['tick_x']:ax.axvline(float(x),color='0.8',lw=.5,zorder=0)
            ax.set_xticks([float(x) for x in d['tick_x']]);ax.set_xticklabels([str(l) for l in d['tick_labels']],fontsize=7)
        ax.axhline(0,color='0.5',lw=.5,ls=':');ax.set_xlim(0,1);ax.set_ylim(lo,hi)
        ax.set_title(f"{name} [{role}]\n{metric}",fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.suptitle(title+'   (DFT/reference blue, EHT red dashed; 0 = E$_F$ or midgap)',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out,dpi=150);plt.close(fig)
    return out

def build_training_config(m,train_entries,args,manifest_dir,radius):
    any_metal=any(e.get('metallic') for e in train_entries)
    any_dft=any(e.get('reference_format')=='dft' for e in train_entries)
    obj=args.objective or m.get('objective') or ('metal' if any_metal else 'window' if any_dft else 'frontier')
    if obj=='frontier' and any_metal:
        raise ValueError("metallic training references need objective 'metal' or 'window' (not 'frontier')")
    if obj in ('window','metal') and not any_dft:
        raise ValueError(f"objective '{obj}' needs at least one numerical DFT reference")
    cfg={'n_pairs':int(m.get('n_pairs',1)),'overlap_radius':radius,
        'gap_weight':float(m.get('gap_weight',.35)),'reg_weight':float(m.get('reg_weight',.04)),
        'overlap_min_eig':float(m.get('overlap_min_eig',1e-3)),
        'objective':obj,'objective_settings':dict(m.get('objective_settings') or {}),
        'basis_add':args.basis_add if args.basis_add is not None else (m.get('basis_add') or ''),
        'basis_dz':args.basis_dz if args.basis_dz is not None else (m.get('basis_dz') or ''),
        'systems':train_entries}
    if args.workers or m.get('workers'):cfg['workers']=int(args.workers or m.get('workers'))
    start=args.start_from or m.get('start_from')
    if start:
        start=Path(start) if Path(str(start)).is_absolute() else csv_path(str(start),manifest_dir)
        table=start
        if start.is_dir():
            table=start/'parameters_huckel.dat'
            if not table.is_file():table=start/'param_best_native.dat'
        if not table.is_file():raise FileNotFoundError(f'start_from: no parameter table in {start}')
        cfg['start_table']=str(table)
        K0=m.get('start_K')
        for js in ([start/'fit_summary.json',start/'fit.json'] if start.is_dir() else []):
            if K0 is None and js.is_file():
                j=json.loads(js.read_text());K0=j.get('K_EHT',(j.get('best_x') or {}).get('GLOBAL__K'))
        if K0 is not None:cfg['start_K']=float(K0)
    return cfg

def _metric_text(met):
    for key,lab in (('window_rmse_eV','window RMSE'),('window_rms_eV','window RMS'),('selected_rmse_eV','RMSE')):
        if met.get(key) is not None:
            t=f"{lab} {met[key]:.3f} eV"
            if met.get('rms_within_1eV_of_EF_eV') is not None:t+=f", ±1 eV: {met['rms_within_1eV_of_EF_eV']:.3f}"
            return t
    return ''

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('manifest',type=Path,help='family YAML (systems: DFT folders and/or structure + image/CSV + INI)')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true',
       help='schema/structure/role checks, no reference data read, no solver')
    mode.add_argument('--prepare-only',action='store_true',
       help='prepare TRAIN references only (previews, QA); NO solver')
    mode.add_argument('--fit',action='store_true',help='native family optimization, then frozen evaluation of every system')
    mode.add_argument('--evaluate',type=Path,metavar='FIT_JSON',
       help='no fit: evaluate every system (train and test) with the table and K of an earlier fit.json '
            '(its param_best_native.dat next to it)')
    p.add_argument('--budget',type=int,default=400)
    p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--optimizer',default='nelder-mead',choices=OPTIMIZERS,
       help='nelder-mead = short DE + NM (default); a global method runs from the stock (or start_from) '
            'parameters with the whole budget; SA/HJ refine a staged base fit')
    p.add_argument('--polish',choices=POLISH_METHODS,
       help='local refinement after a global optimizer, inside the budget')
    p.add_argument('--objective',choices=OBJECTIVES,default=None,
       help='overrides the manifest; default: metal if a training reference is metallic, window for '
            'DFT references, frontier for images (metallic members always use metal, gapped DFT members window)')
    p.add_argument('--basis-add',default=None,help='extra shells, e.g. "S:d" or auto (overrides the manifest)')
    p.add_argument('--basis-dz',default=None,
       help='shells made double-zeta (inner exponent + fitted diffuse tail), e.g. "S:p"; s/p need a bind '
            'with double-zeta p (github.com/ac-dias/yaehmop) (overrides the manifest basis_dz)')
    p.add_argument('--start-from',type=Path,default=None,
       help='fitting/ folder, fit/ folder or parameter table to start from (bounds centred there)')
    p.add_argument('--workers',type=int,default=None,help='systems evaluated in parallel (default: all, up to the CPUs)')
    p.add_argument('--no-export',action='store_true',help='skip the per-system fitting/ export of H(R), S(R)')
    p.add_argument('--outdir',type=Path,default=Path('outputs/family_fit'))
    args=p.parse_args(argv)
    try:
        path,m,train,test=load_manifest(args.manifest)
    except (ValueError,FileNotFoundError) as exc:
        p.error(str(exc))
    pairs=int(m.get('n_pairs',1))
    out=args.outdir.resolve()
    report={'mode':'shared_train_then_frozen_evaluation','n_pairs':pairs,
        'training':[s['name'] for s in train],'test':[s['name'] for s in test],
        'kinds':{s['name']:s['kind'] for s in train+test},
        'training_structure_sha256s':{s['name']:sha(s['structure_file']) for s in train if s.get('structure_file')},
        'test_structure_sha256s':{s['name']:sha(s['structure_file']) for s in test if s.get('structure_file')},
        'test_references_opened_before_freeze':False,
        'notes':('Test reference data are not read or used in fitting. Without test systems every score is a '
                 'training (calibration) error, not transferability.')}
    if not test:report['warning']='no test system: all reported errors are training errors'
    if args.validate_only:
        print(json.dumps(report,indent=2))
        return
    out.mkdir(exist_ok=True,parents=True)
    radius=int(m.get('overlap_radius',DEFAULT_OVERLAP_RADIUS))
    try:
        entries=[preparation(s,out,pairs) for s in train]
    except (ValueError,FileNotFoundError) as exc:
        p.error(str(exc))
    for s,e in zip(train,entries):
        e['overlap_radius']=radius
        if s['kind']=='dft':
            info=s['loaded']['info']
            print('PREPARED',s['name'],'metal' if info['metallic'] else f"gap {info['gap_on_path_eV']:.3f} eV",
                  'semicore dropped',info['semicore_bands_dropped'],'dim',info['periodic_dim'],flush=True)
            for note in info['notes']:print('  NOTE',note,flush=True)
    try:
        training_config=build_training_config(m,entries,args,path.parent,radius)
    except (ValueError,FileNotFoundError) as exc:
        p.error(str(exc))
    generated=out/'frozen_train_only.yaml'
    generated.write_text(yaml.safe_dump(training_config,sort_keys=False))
    report.update(prepared_train_input=str(generated),objective=training_config['objective'],
                  metallic={s['name']:s['loaded']['metallic'] for s in train if s['kind']=='dft'})
    (out/'family_manifest_audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    if args.prepare_only:
        print('TRAIN_PREPARED_NO_TEST_REFERENCE_READ_NO_SOLVER',generated)
        return
    if args.evaluate:
        fitjson=args.evaluate.resolve()
        saved=json.loads(fitjson.read_text());native=fitjson.parent/'param_best_native.dat'
        param_txt=fitjson.parent/'param_best.txt'
        mode_name='evaluation of an earlier fit (no optimization)'
    else:
        from fit_sparse_window import Fitter
        probe=Fitter(training_config,generated.parent,out/'fit_probe')
        n_active=len(probe.names);probe.progress.close();shutil.rmtree(out/'fit_probe',ignore_errors=True)
        minbudget=minimum_staged_budget(n_active)
        if args.optimizer=='nelder-mead' and args.budget<minbudget:
            p.error(f'This family has {n_active} active EHT parameters; --budget must be >= {minbudget} '
                    'for the staged initializer.')
        if args.polish and args.optimizer not in GLOBAL_METHODS:p.error('--polish only follows a global optimizer')
        print('FITTING_ONE_SHARED_PARAMETER_VECTOR_TRAIN_ONLY',n_active,'parameters',len(train),'systems',flush=True)
        run_optimizer(generated,out,args.optimizer,args.budget,args.polish,args.seed,p)
        fitjson=out/'fit/fit.json';saved=json.loads(fitjson.read_text())
        native=out/'fit/param_best_native.dat';param_txt=out/'fit/param_best.txt'
        mode_name='shared family fit'
    if not native.is_file():raise FileNotFoundError(native)
    K=float(saved['best_x']['GLOBAL__K'])
    frozen={'native_parameters':str(native),'K_EHT':K,
        'train_cif_sha256s':list(report['training_structure_sha256s'].values()),
        'train_systems':report['training'],'overlap_radius':radius,'objective':training_config['objective']}
    (out/'shared_frozen_source.json').write_text(json.dumps(frozen,indent=2)+'\n')
    objset=training_config['objective_settings']
    results={};panels=[]
    # Training systems: the backend wrote OUTDIR/fit/<name>_bands.npz with the frozen vector
    for s,e in zip(train,entries):
        npz=out/'fit'/f"{s['name']}_bands.npz"
        sysdir=out/'systems'/s['name']
        bm=(saved.get('metrics') or {}).get(s['name'],{}) if not args.evaluate else {}
        if args.evaluate or not npz.is_file():
            if s['kind']!='dft':raise RuntimeError(f"{s['name']}: no band file from the fit")
            w=sysdir/'eval';w.mkdir(parents=True,exist_ok=True)
            bm=(evaluate_metal(e,native,K,w,s['name'],objset,_window(s)) if s['loaded']['metallic']
                else evaluate_dft(e,native,K,w,s['name'],pairs,_window(s)))
            npz=w/f"{s['name']}_bands.npz"
        met=_system_outputs(s,npz,sysdir,f"{s['name']} [train]",not args.no_export,native,param_txt,K,radius,pairs)
        for k in ('window_rms_eV','rms_within_2eV_of_EF_eV','rms_within_1eV_of_EF_eV','gap_error_eV','total'):
            if bm.get(k) is not None and k not in met:met[k]=bm[k]
        if isinstance(bm.get('window'),dict):
            met['window_rms_eV']=bm['window'].get('window_rms_eV');met['window_total']=bm['window'].get('total')
        results[s['name']]=dict(role='train',kind=s['kind'],**met)
        panels.append((s['name'],npz,'train',_metric_text(met)))
    # Test systems: opened only now, scored with the frozen vector
    trained_elements={e for t in train for e in (t['symbols'] or t['loaded']['ref'].atoms.get_chemical_symbols())}
    for s in test:
        hd=out/'heldout'/s['name'];npz=None
        if s['kind']=='dft':
            L=load_dft(s,hd/'dft_reference',pairs);s['loaded']=L
            unseen=set(L['ref'].atoms.get_chemical_symbols())-trained_elements
            if unseen:raise ValueError(f"{s['name']}: elements absent from training {sorted(unseen)}")
            e=dict(L['entry'],overlap_radius=radius,role='test')
            ev=(evaluate_metal(e,native,K,hd,s['name'],objset,_window(s)) if L['metallic']
                else evaluate_dft(e,native,K,hd,s['name'],pairs,_window(s)))
            npz=hd/f"{s['name']}_bands.npz"
            met=_system_outputs(s,npz,out/'systems'/s['name'],f"{s['name']} [test, frozen]",
                                not args.no_export,native,param_txt,K,radius,pairs)
            for k in ('window_rms_eV','rms_within_1eV_of_EF_eV','rms_within_2eV_of_EF_eV','gap_error_eV'):
                if ev.get(k) is not None and k not in met:met[k]=ev[k]
        else:
            ev=evaluate(s['cif'],s['ini'],s['image'],native,K,hd,pairs,mode=s['image_mode'],bbox=s['bbox'],
                        colors=s['colors'],overlap_radius=radius,source=frozen,structure_format=s['structure_format'])
            met=dict(selected_rmse_eV=ev['selected_rmse_eV'])
        results[s['name']]=dict(role='test',kind=s['kind'],**met)
        if npz is not None and Path(npz).is_file():panels.append((s['name'],npz,'test',_metric_text(met)))
    ov=overview_figure(panels,out/'family_overview.png',f"{mode_name}: {len(train)} train + {len(test)} test, K = {K:.3f}")
    # the shared table, once, for the user
    pdir=out/'parameters';pdir.mkdir(exist_ok=True)
    shutil.copy(native,pdir/'parameters_huckel.dat')
    if param_txt.is_file():shutil.copy(param_txt,pdir/'parameters_huckel.txt')
    else:(pdir/'parameters_huckel.txt').write_text(f'K_EHT {K:.12g}\n')
    report.update({'parameters_sha256':sha(native),'K_EHT':K,'mode':mode_name,
        'shared_source_summary':str(out/'shared_frozen_source.json'),
        'test_references_opened_only_after_fit':True,
        'systems':results,'overview_figure':str(ov) if ov else None,
        'best_training_loss':saved.get('best_training_loss'),
        'objective_optimizer_evaluations':saved.get('objective_evaluations'),
        'optimizer':None if args.evaluate else args.optimizer,'budget':None if args.evaluate else args.budget,
        'objective_settings':objset,'basis_added':saved.get('basis_added'),
        'basis_double_zeta':saved.get('basis_double_zeta'),
        'trainer_model':'physical YAeHMOP multi-system backend (fit_sparse_window.py)'})
    (out/'family_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('FAMILY_DONE',json.dumps({k:{kk:(round(v,4) if isinstance(v,float) else v) for kk,v in r.items()}
                                     for k,r in results.items()}),flush=True)
    print('OVERVIEW',ov);print('PARAMETERS',pdir/'parameters_huckel.dat')

if __name__=='__main__':main()
