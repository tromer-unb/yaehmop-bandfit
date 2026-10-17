#!/usr/bin/env python3
"""Transparent *recommendation* of YAeHMOP workflows; no physics is calculated.

This is a deterministic rules-based guide from user-declared task, not a
validated automatic structure/PNG classifier or optimizer selector.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from ase.io import read

def recommendation(*,goal='direct',electronic='unknown',bands='auto',
                   phase='unknown',elements=(),has_reference=False):
    if goal not in ('direct','transfer'):
        raise ValueError('goal must be direct or transfer')
    if electronic not in ('semiconductor','metal','unknown'):
        raise ValueError('electronic must be semiconductor, metal or unknown')
    if bands not in ('auto','2','4','6','8'):
        raise ValueError('bands must be auto, 2, 4, 6 or 8')
    n=2 if bands=='auto' else int(bands)
    info={
      'kind':'RULE_BASED_WORKFLOW_ADVICE_NOT_AUTOMATIC_MODEL_SELECTION',
      'goal':goal,'electronic_type_user_declared':electronic,
      'phase_user_declared':phase,'chemical_elements':sorted(set(elements)),
      'n_selected_bands':n,'pairs':n//2,'reference_present':has_reference,
      'tested_workflow':False,
      'warnings':['The CIF alone does not determine a band gap, Fermi surface,'
                  ' or electronic phase; verify against reference data.',
                  'An ordinary PNG may hide crossings or use unknown k/energy axes.',
                  'A 1D high-symmetry PNG cannot establish metallic DOS/Fermi topology.']}
    if electronic!='semiconductor':
        info['recommendation']='SPECIALIZED_REFERENCE_REQUIRED'
        info['tested_workflow']=False
        info['why']='The public image fitter validates a positive spinless gap; metal/gapless/spin-orbit states require special objectives and possibly full-BZ data.'
        info['next_step']='Supply electronic classification and numerical BZ bands if metallic. Do not force a gapped 2-band PNG fit.'
    elif goal=='direct':
        info['recommendation']='MATCHED_WINDOW_DIRECT_YAEHMOP'
        info['tested_workflow']=True
        info['why']='Start with the least complex eligible 1V+1C fit; move to 2V+2C etc. only with corresponding reliable reference curves and a validated error criterion.'
        info['entrypoint']='USER_WORKFLOW/bands_fit.py'
        info['commands']=['python USER_WORKFLOW/bands_fit.py STRUCTURE.cif SETTINGS.ini --prepare-only --outdir outputs/material',
           f'python USER_WORKFLOW/bands_fit.py STRUCTURE.cif SETTINGS.ini --pairs {n//2} --budget 400 --outdir outputs/material',
           'python USER_WORKFLOW/bands_fit.py STRUCTURE.cif SETTINGS.ini --from-fit outputs/material/fit/fit.json --outdir outputs/material_full']
    else:
        info['recommendation']='FROZEN_TRANSFER_NO_REFIT'
        info['tested_workflow']=True
        info['why']='Use coefficients from an already fitted source crystal; apply them to a distinct compatible target without changing them; compare the same N bands and report error with zero optimizer calls. This does not train a new shared model.'
        info['entrypoint']='USER_WORKFLOW/frozen_transfer.py'
        info['commands']=[f'python USER_WORKFLOW/frozen_transfer.py TARGET.cif TARGET.ini --source-run outputs/source/run_summary.json --pairs {n//2} --outdir outputs/transfer_target']
        info['shared_fit_note']='Expert parameter_fitter/fit_shared.py exists but uses full-orbital images, not the turnkey 2/4/6/8-band PNG contract. A unified multi-CIF fit is not validated.'
    if not has_reference:
        info['warnings'].append('No reference image/CSV was supplied to the advisor; the recommendation cannot assess physical accuracy.')
    info['recommended_model_status']='Provisional workflow recommendation, NOT a fitted parameter vector'
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cif',type=Path,required=True)
    p.add_argument('--reference',type=Path,help='PNG/CSV path; existence checked only (not parsed)')
    p.add_argument('--goal',choices=('direct','transfer'),default='direct')
    p.add_argument('--electronic',choices=('semiconductor','metal','unknown'),default='unknown')
    p.add_argument('--phase',choices=('unknown','1H','1T','bulk'),default='unknown')
    p.add_argument('--bands',choices=('auto','2','4','6','8'),default='auto')
    p.add_argument('--outdir',type=Path,default=Path('outputs/advice'))
    a=p.parse_args()
    if not a.cif.is_file():p.error('CIF does not exist')
    if a.reference and not a.reference.is_file():p.error('Reference file does not exist')
    atoms=read(a.cif)
    if len(atoms)==0:p.error('CIF contains no atoms')
    info=recommendation(goal=a.goal,electronic=a.electronic,
        phase=a.phase,bands=a.bands,elements=atoms.get_chemical_symbols(),
        has_reference=a.reference is not None)
    info['cif']=str(a.cif.resolve())
    info['reference']=str(a.reference.resolve()) if a.reference else None
    a.outdir.mkdir(parents=True,exist_ok=True)
    path=a.outdir/'recommendation.json'
    path.write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
    print('SUGGESTED:',info['recommendation'])
    print('WHY:',info['why'])
    print('REQUIRES:', 'actual energy bounds, reciprocal-space path and orbital eligibility')
    for cmd in info.get('commands',[]):print('  '+cmd)
    print('ADVICE_JSON:',path)
    print('NOTE: no fitting, no YAeHMOP run, no automatic PNG digitization.')

if __name__=='__main__':main()