#!/usr/bin/env python3
"""Evaluate a frozen, previously fitted YAeHMOP table on another CIF + PNG/CSV.

ZERO optimization, physical YAeHMOP recomputation, 2/4/6/8 matched bands.
This is an *evaluation* of already frozen coefficients, not multi-system
training and not automatically a prospective blind test.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from ase.io import read
from bands_fit import conf,read_nodes,csv_path,recompute,write_bands_plot
from prepare_band_png import prepare,parse_bbox
from generate_mc2d_multiband import read_colored_png
from yaehmop_generic import parse_parameter_table,subset_for_elements,orbital_count,valence_electrons

def sha(p:Path)->str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source_from_summary(path:Path)->tuple[Path,float,dict]:
    """Do not silently borrow the target reference or refit the source."""
    path=Path(path).resolve()
    row=json.loads(path.read_text())
    if not isinstance(row,dict) or 'native_parameters' not in row or 'K_EHT' not in row:
        raise ValueError('Source summary must contain native_parameters and K_EHT')
    native=Path(row['native_parameters'])
    if not native.is_absolute():native=(path.parent/native)
    if not native.is_file():
        native=path.parent/'fit'/'param_best_native.dat'
    if not native.is_file():raise FileNotFoundError('Frozen source parameter table missing: '+str(native))
    return native.resolve(),float(row['K_EHT']),row

def evaluate(cif,ini,image,parameters,K,outdir,pairs,
             *,mode=None,bbox=None,colors=None,allow_ambiguous=False,
             overlap_radius=4,source=None):
    cif=Path(cif).resolve();ini=Path(ini).resolve();outdir=Path(outdir).resolve()
    if pairs not in (1,2,3,4):raise ValueError('pairs must be 1, 2, 3 or 4')
    if not cif.is_file():raise FileNotFoundError(cif)
    if overlap_radius not in range(1,9):raise ValueError('overlap radius must be in 1..8')
    c=conf(ini);read_nodes(c)
    r=c['reference']; f=c['fit']
    img=Path(image).resolve() if image else csv_path(r.get('image',''),ini.parent)
    if not img.is_file():raise FileNotFoundError(img)
    native=Path(parameters).resolve()
    if not native.is_file():raise FileNotFoundError(native)
    outdir.mkdir(exist_ok=True,parents=True)
    emin,emax=float(r['emin']),float(r['emax'])
    actualmode=mode or r.get('mode','auto')
    prepared,qa=prepare(img,outdir/'reference_prepared',pairs,emin,emax,
        mode=actualmode,bbox=parse_bbox(bbox or r.get('bbox',fallback=None)),
        colors=colors or r.get('colors',fallback=None),
        tolerance=r.getint('color_tolerance',fallback=36),
        nx=r.getint('trace_nx',fallback=301),
        min_coverage=r.getfloat('min_coverage',fallback=.75),
        allow_ambiguous=allow_ambiguous)
    # Never evaluate against the source material used to optimize the vector
    # when a training provenance summary is supplied.
    if source is not None and sha(cif) in (
            ([source['cif_sha256']] if source.get('cif_sha256') else [])
             + list(source.get('train_cif_sha256s',[]))):
        raise ValueError('Target CIF equals source training CIF; use direct evaluation, not frozen transfer')
    table=parse_parameter_table(native)
    symbols=read(cif).get_chemical_symbols()
    params=subset_for_elements(table,list(dict.fromkeys(symbols)))
    nelec=valence_electrons(params,symbols)
    norb=orbital_count(params,symbols)
    if nelec%2:raise ValueError('Odd electron count unsupported by spinless benchmark')
    nocc=nelec//2
    if nocc<pairs or norb-nocc<pairs:
        raise ValueError('Requested frontier bands exceed the eligible physical orbital basis')
    # One real periodic calculation with no optimizer. Full orbitals saved.
    k,all_bands=recompute(cif,c,native,float(K),outdir,overlap_radius)
    rx,refs=read_colored_png(prepared,pairs,emin,emax,nx=r.getint('trace_nx',fallback=301))
    target=np.column_stack([np.interp(k,rx,refs[:,j]) for j in range(2*pairs)])
    ref_gap=float(target[:,pairs].min()-target[:,pairs-1].max())
    if ref_gap<=.02:raise ValueError('Reference is not a gapped spinless semiconductor')
    refmid=.5*(float(target[:,pairs-1].max())+float(target[:,pairs].min()))
    srcmid=.5*(float(all_bands[:,nocc-1].max())+float(all_bands[:,nocc].min()))
    selected=all_bands[:,nocc-pairs:nocc+pairs]-srcmid
    reference=target-refmid
    e=np.sqrt(np.mean((selected-reference)**2))
    gap=float(all_bands[:,nocc].min()-all_bands[:,nocc-1].max())
    npz=outdir/'frozen_bands.npz'
    np.savez_compressed(npz,k=k,reference=reference,YAeHMOP=selected,
        all_orbital_eigenvalues_eV=all_bands,nocc=nocc,n_pairs=pairs,
        reference_midgap_eV=refmid,eht_midgap_eV=srcmid)
    reported=write_bands_plot(npz,outdir,2*pairs,
        title=f'Frozen EHT evaluation — {pairs}VB + {pairs}CB (NO RE-FITTING)')
    assert abs(reported-e)<1e-10
    report={
        'mode':'frozen_parameters_no_optimization',
        'warning':'Retrospective frozen evaluation; not necessarily prospective blind transfer.',
        'target_cif':str(cif),'target_cif_sha256':sha(cif),
        'target_reference':str(img),'target_reference_sha256':sha(img),
        'source_summary_supplied':source is not None,
        'source_train_cif_sha256':source.get('cif_sha256') if source else None,
        'parameter_table':str(native),'parameter_table_sha256':sha(native),
        'K_EHT':float(K),'n_pairs':pairs,'n_scored_bands':2*pairs,
        'reference_input_extraction':qa,
        'selected_rmse_eV':float(e),'reference_gap_eV':ref_gap,
        'model_gap_eV':gap,'gap_error_eV':gap-ref_gap,
        'refit_optimizer_calls':0,'native_solver_executions':1,
        'overlap_radius':overlap_radius,
        'full_spectrum':'bands_full.npz',
        'reference_comparison':'bands_comparison.png',
        'notes':'Midgap aligned separately for reference and EHT, as in original gapped sparse-band fitter; full-basis eigenvalues are not extra fitted bands.'}
    (outdir/'transfer_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('FROZEN_EVALUATION_DONE',cif.name,'RMSE_eV',round(float(e),6),
          'optimizer_calls',0)
    print('FULL_BANDS',outdir/'bands_full.npz')
    print('COMPARISON',outdir/'bands_comparison.png')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cif',type=Path)
    p.add_argument('input_file',type=Path)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--source-run',type=Path,
        help='Run summary of a separately fitted SOURCE system; automatically finds table and K')
    choice.add_argument('--parameters',type=Path,
        help='Frozen native parameter table; requires --k')
    p.add_argument('--k',type=float,
        help='Native EHT global K, required if --parameters was chosen')
    p.add_argument('--png','--image',dest='png',type=Path)
    p.add_argument('--pairs',type=int,choices=(1,2,3,4))
    p.add_argument('--mode',choices=('auto','colors','mono','layered','csv'))
    p.add_argument('--bbox',help='L,T,R,B of real plot frame')
    p.add_argument('--colors',help='ordered actual color values for exactly 2*--pairs bands')
    p.add_argument('--allow-ambiguous',action='store_true')
    p.add_argument('--overlap-radius',type=int,default=4)
    p.add_argument('--outdir',type=Path,default=Path('outputs/frozen_transfer'))
    args=p.parse_args()
    if args.source_run is not None:
        if args.k is not None:p.error('--k belongs only with --parameters')
        params,K,source=source_from_summary(args.source_run)
    else:
        if args.k is None:p.error('--k is required when using --parameters')
        params,K,source=args.parameters,args.k,None
    pairs=args.pairs or int(conf(args.input_file)['fit'].get('n_pairs','1'))
    evaluate(args.cif,args.input_file,args.png,params,K,args.outdir,pairs,
        mode=args.mode,bbox=args.bbox,colors=args.colors,
        allow_ambiguous=args.allow_ambiguous,
        overlap_radius=args.overlap_radius,source=source)

if __name__=='__main__':
    main()