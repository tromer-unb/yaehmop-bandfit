#!/usr/bin/env python3
"""One shared EHT vector from several gapped CIF+PNG/CSV sources (2/4/6/8 bands).

Training images only are prepared and opened before parameters are frozen;
withheld reference images are used *only after* fitting, to score genuine
whole-material frozen transfer. Runs actual YAeHMOP only upon --fit.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import yaml
from ase.io import read
from bands_fit import conf,read_nodes,csv_path
from prepare_band_png import prepare,parse_bbox
from frozen_transfer import evaluate,sha
ROOT=Path(__file__).resolve().parents[1]

def load_manifest(path):
    path=Path(path).resolve()
    a=yaml.safe_load(path.read_text())
    if not isinstance(a,dict) or not isinstance(a.get('systems'),list):
        raise ValueError('Manifest requires systems: list')
    if int(a.get('n_pairs',1)) not in (1,2,3,4):raise ValueError('n_pairs must be 1..4')
    names=set()
    train=[];test=[]
    for row in a['systems']:
        if not isinstance(row,dict):raise ValueError('Each system must be a mapping')
        for key in ('name','role','structure','settings'):
            if not row.get(key):raise ValueError('System missing '+key)
        name=str(row['name'])
        if name in names or '/' in name or '..' in name:
            raise ValueError('Repeated or invalid system name '+name)
        names.add(name)
        role=str(row['role']).lower()
        if role not in ('train','test'):raise ValueError('role must be train or test')
        cif=(path.parent/str(row['structure'])).resolve()
        ini=(path.parent/str(row['settings'])).resolve()
        if not cif.is_file() or not ini.is_file():
            raise FileNotFoundError('CIF/INI missing for '+name)
        cfg=conf(ini);read_nodes(cfg)
        img=(path.parent/str(row['reference'])).resolve() if row.get('reference') else csv_path(cfg['reference'].get('image',''),ini.parent)
        if not img.is_file():raise FileNotFoundError('Reference missing for '+name+': '+str(img))
        symbols=read(cif).get_chemical_symbols()
        ref=cfg['reference']
        entry={'name':name,'role':role,'cif':cif,'ini':ini,'cfg':cfg,
               'image':img,'symbols':symbols,'weight':float(row.get('weight',1.0)),
               'image_mode':str(row.get('mode',ref.get('mode','auto'))),
               'bbox':row.get('bbox',ref.get('bbox',fallback=None)),
               'colors':row.get('colors',ref.get('colors',fallback=None))}
        if entry['weight']<=0:raise ValueError('Weight must be positive for '+name)
        (train if role=='train' else test).append(entry)
    if len(train)<2:raise ValueError('At least two whole training systems required')
    if not test:raise ValueError('At least one withheld TEST system is required')
    trained_elements={e for row in train for e in row['symbols']}
    unseen={e for row in test for e in row['symbols']}-trained_elements
    if unseen:raise ValueError('Test contains elements absent from training: '+str(sorted(unseen)))
    thashes={sha(s['cif']) for s in train}
    for s in test:
        if sha(s['cif']) in thashes:
            raise ValueError('Heldout CIF is identical to a training CIF: '+s['name'])
    return path,a,train,test

def preparation(row,out,pairs):
    ref=row['cfg']['reference']
    prepared,qa=prepare(row['image'],out/'prepared'/row['name'],pairs,
        float(ref['emin']),float(ref['emax']),mode=row['image_mode'],
        bbox=parse_bbox(row['bbox']),colors=row['colors'],
        tolerance=ref.getint('color_tolerance',fallback=36),
        nx=ref.getint('trace_nx',fallback=301),
        min_coverage=ref.getfloat('min_coverage',fallback=.75))
    dim,nodes,labels,step=read_nodes(row['cfg'])
    return {'name':row['name'],'role':'train','weight':row['weight'],
       'structure':str(row['cif']),'reference':str(prepared),
       'emin':float(ref['emin']),'emax':float(ref['emax']),
       'point_dim':dim,'symmetry_points':nodes,'labels':labels,
       'n_pairs':pairs,'image_mode':'colored','points_per_line':step,
       'trace_nx':ref.getint('trace_nx',fallback=301)}

def minimum_staged_budget(n_active):
    """The staged native DE initializer needs two populations of 2*d calls,
    plus the stock point, and at least one local call. Check before launching."""
    return max(80,4*int(n_active)+2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true',
       help='schema/CIF/role checks, no PNG extraction or solver')
    mode.add_argument('--prepare-only',action='store_true',
       help='prepare TRAIN PNGs only, inspect QA; NO training/test solver')
    mode.add_argument('--fit',action='store_true',help='actual native family optimization and frozen held-out evaluation')
    p.add_argument('--budget',type=int,default=400)
    p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--outdir',type=Path,default=Path('outputs/family_fit'))
    args=p.parse_args()
    path,m,train,test=load_manifest(args.manifest)
    pairs=int(m.get('n_pairs',1))
    out=args.outdir.resolve()
    report={'mode':'shared_train_then_whole_material_frozen_test',
        'n_pairs':pairs,'n_scored_bands':2*pairs,
        'training':[s['name'] for s in train],
        'test':[s['name'] for s in test],
        'training_cif_sha256s':{s['name']:sha(s['cif']) for s in train},
        'test_cif_sha256s':{s['name']:sha(s['cif']) for s in test},
        'test_references_opened_before_freeze':False,
        'notes':'Test-reference DATA are not digitized or used in fitting. Split selection/preinspection must also be documented; this alone does not guarantee prospective blind validation.'}
    if args.validate_only:
        print(json.dumps(report,indent=2))
        return
    out.mkdir(exist_ok=True,parents=True)
    training_config={'n_pairs':pairs,
        'gap_weight':float(m.get('gap_weight',.35)),
        'reg_weight':float(m.get('reg_weight',.04)),
        'systems':[preparation(s,out,pairs) for s in train]}
    # No withheld system entry or its reference is passed to optimizer.
    generated=out/'frozen_train_only.yaml'
    generated.write_text(yaml.safe_dump(training_config,sort_keys=False))
    report['prepared_train_input']=str(generated)
    (out/'family_manifest_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.prepare_only:
        print('TRAIN_PREPARED_NO_TEST_REFERENCE_READ_NO_SOLVER',generated)
        return
    # Detect the backend's real active dimension before any objective call.
    # This catches the legacy DE initialization budget overrun up front.
    sys.path.insert(0,str(ROOT/'parameter_fitter'))
    from fit_sparse_window import Fitter
    probe=Fitter(training_config,generated.parent,out/'fit')
    minbudget=minimum_staged_budget(len(probe.names))
    if args.budget<minbudget:
        p.error(f'This family has {len(probe.names)} active EHT parameters; '
                f'--budget must be >= {minbudget} for the staged initializer. '
                'A larger budget (e.g. 400) is recommended for actual science.')
    cmd=[sys.executable,str(ROOT/'parameter_fitter/fit_sparse_window.py'),
         str(generated),'--outdir',str(out/'fit'),'--initialization','staged',
         '--budget',str(args.budget),'--seed',str(args.seed)]
    print('FITTING_ONE_SHARED_PARAMETER_VECTOR_TRAIN_ONLY',flush=True)
    subprocess.run(cmd,check=True)
    saved=json.loads((out/'fit/fit.json').read_text())
    native=out/'fit/param_best_native.dat'
    if not native.is_file():raise FileNotFoundError(native)
    K=float(saved['best_x']['GLOBAL__K'])
    frozen={'native_parameters':str(native),'K_EHT':K,
        'train_cif_sha256s':list(report['training_cif_sha256s'].values()),
        'train_systems':report['training']}
    (out/'shared_frozen_source.json').write_text(json.dumps(frozen,indent=2)+'\n')
    results={}
    # Only now open each heldout reference and score the *frozen* vector.
    for s in test:
        results[s['name']]=evaluate(s['cif'],s['ini'],s['image'],
            native,K,out/'heldout'/s['name'],pairs,
            mode=s['image_mode'],bbox=s['bbox'],colors=s['colors'],
            source=frozen)
    report.update({'parameters_sha256':sha(native),'K_EHT':K,
        'shared_source_summary':str(out/'shared_frozen_source.json'),
        'test_references_opened_only_after_fit':True,
        'test_rmse_eV':{k:v['selected_rmse_eV'] for k,v in results.items()},
        'objective_optimizer_evaluations':saved.get('objective_evaluations'),
        'trainer_model':'physical YAeHMOP multi-system sparse-window backend'})
    (out/'family_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FAMILY_SHARED_FIT_DONE; frozen heldouts:',report['test_rmse_eV'])

if __name__=='__main__':main()