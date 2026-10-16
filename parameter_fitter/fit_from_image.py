#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,copy,csv,json,sys,time
import numpy as np
from scipy.optimize import differential_evolution,minimize
from ase.io import read

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from yaehmop_generic import (parse_parameter_table,subset_for_elements,orbital_count,valence_electrons,
    write_parameter_file,write_periodic_input_generic,BandPath,run_bind,parse_band,normalized_path_coordinate,resolve_yaehmop_assets)
from image_target import load_band_image,energies_at_x,image_gap



def parse_points(vals,dim):
    a=np.asarray(vals,float)
    if len(a)%dim: raise ValueError('symmetry-point coordinate count is not divisible by point dimension')
    a=a.reshape(-1,dim); out=np.zeros((len(a),3)); out[:,:dim]=a; return out

def build_specs(params,elements):
    specs=[]; x0=[]; bounds=[]; scales=[]
    for el in elements:
        spec=params[el]
        for orb in ('s','p','d','f'):
            if orb not in spec: continue
            o=spec[orb]
            v=float(o['Hii']); specs.append((el,orb,'Hii')); x0.append(v); bounds.append((v-8.0,v+8.0)); scales.append(4.0)
            v=float(o['zeta1']); specs.append((el,orb,'zeta1')); x0.append(v); bounds.append((max(.55,.45*v),min(9.0,2.0*v))); scales.append(max(.5,.4*v))
            if int(o['nzeta'])==2:
                v=float(o['zeta2']); specs.append((el,orb,'zeta2')); x0.append(v); bounds.append((max(.45,.45*v),min(6.0,2.0*v))); scales.append(max(.4,.4*v))
                ratio=float(o['c2']/o['c1']) if abs(float(o['c1']))>1e-12 else 1.0
                specs.append((el,orb,'ratio')); x0.append(ratio); bounds.append((.05,2.5)); scales.append(.5)
    specs.append(('GLOBAL','','K')); x0.append(1.75); bounds.append((1.0,2.8)); scales.append(.4)
    return specs,np.asarray(x0,float),bounds,np.asarray(scales,float)

def apply_x(defaults,specs,x):
    p=copy.deepcopy(defaults); K=1.75
    for (el,orb,kind),v in zip(specs,x):
        v=float(v)
        if kind=='K': K=v
        elif kind=='ratio': p[el][orb]['c1']=1.0; p[el][orb]['c2']=v
        else: p[el][orb][kind]=v
    return p,K

def physical_ok(p):
    for el,s in p.items():
        for orb in ('s','p','d','f'):
            if orb in s and int(s[orb]['nzeta'])==2:
                if s[orb]['zeta1'] <= s[orb]['zeta2']+0.12: return False
    return True

def image_shape_metrics(target,xcoord,M_aligned):
    mt_num=mt_den=tm_num=tm_den=0.0; ncols=0
    for xx,m in zip(xcoord,M_aligned):
        t=energies_at_x(target,float(xx),half_window=2,cluster_gap_px=3)
        t=t[(t>=target['emin'])&(t<=target['emax'])]
        if not len(t): continue
        m=m[(m>=target['emin']-.5)&(m<=target['emax']+.5)]
        if not len(m): continue
        w_m=1.0+2.0*np.exp(-(m/4.0)**2); d_m=np.min(np.abs(m[:,None]-t[None,:]),axis=1)
        mt_num += float(np.sum(w_m*d_m*d_m)); mt_den += float(np.sum(w_m))
        w_t=1.0+2.0*np.exp(-(t/4.0)**2); d_t=np.min(np.abs(t[:,None]-m[None,:]),axis=1)
        tm_num += float(np.sum(w_t*d_t*d_t)); tm_den += float(np.sum(w_t)); ncols+=1
    if not ncols or mt_den==0 or tm_den==0: return 99.,99.
    return float(np.sqrt(mt_num/mt_den)),float(np.sqrt(tm_num/tm_den))

def main():
    ap=argparse.ArgumentParser(description='Generic YAeHMOP parameter fitting from a band-structure image.')
    ap.add_argument('cif'); ap.add_argument('image')
    ap.add_argument('--emin',type=float,required=True); ap.add_argument('--emax',type=float,required=True); ap.add_argument('--fermi',type=float,default=0.0)
    ap.add_argument('--symmetry-points','--symetry-points',dest='symmetry_points',type=float,nargs='+',required=True); ap.add_argument('--point-dim',type=int,choices=(1,2,3),default=2)
    ap.add_argument('--labels',nargs='+',required=True); ap.add_argument('--points-per-line',type=int,default=30)
    ap.add_argument('--maxiter',type=int,default=6); ap.add_argument('--popsize',type=int,default=5); ap.add_argument('--local-maxiter',type=int,default=160)
    ap.add_argument('--seed',type=int,default=20260927); ap.add_argument('--gap-weight',type=float,default=2.0); ap.add_argument('--reg-weight',type=float,default=.025)
    ap.add_argument('--outdir',default='.'); ap.add_argument('--tag',default='image_fit')
    BIND, PARAM_TABLE = resolve_yaehmop_assets()
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); work=out/'work'; work.mkdir(exist_ok=True)
    cif=Path(args.cif).resolve(); image=Path(args.image).resolve(); atoms=read(cif); symbols=atoms.get_chemical_symbols(); elements=list(dict.fromkeys(s.capitalize() for s in symbols))
    table=parse_parameter_table(PARAM_TABLE); defaults=subset_for_elements(table,elements); norb=orbital_count(defaults,symbols); nelec=valence_electrons(defaults,symbols); nocc=nelec//2
    if norb<=nocc: raise ValueError('Invalid basis/electron count')
    points=parse_points(args.symmetry_points,args.point_dim)
    if len(points)!=len(args.labels): raise ValueError('labels and symmetry points count differ')
    path=BandPath(tuple(args.labels),points,args.points_per_line); target=load_band_image(image,args.emin,args.emax); tv,tc,tgap=image_gap(target,fermi=args.fermi); tmid=.5*(tv+tc)
    specs,x0,bounds,scales=build_specs(defaults,elements); names=[f'{a}_{b}_{c}'.strip('_') for a,b,c in specs]
    inp=work/f'{args.tag}.bind'; parm=work/f'{args.tag}.parms'; runlog=work/f'{args.tag}.run.log'; history=[]; neval=0; best=[1e99,None,None]
    # x coordinate depends only on the requested path/cell.
    pts=[]
    for a,b in zip(points[:-1],points[1:]):
        for j in range(args.points_per_line): pts.append(a+(b-a)*(j/args.points_per_line))
    pts.append(points[-1]); xcoord=normalized_path_coordinate(np.asarray(pts),atoms.cell)
    def objective(x,record=True):
        nonlocal neval,best; neval+=1; p,K=apply_x(defaults,specs,x)
        if not physical_ok(p): return 100.0
        try:
            write_parameter_file(parm,p); write_periodic_input_generic(cif,inp,path,K,True,nelec); run_bind(BIND,inp,parm,runlog); _,M=parse_band(str(inp)+'.band')
            if M.shape!=(len(xcoord),norb) or not np.isfinite(M).all(): return 100.0
            mv=float(M[:,nocc-1].max()); mc=float(M[:,nocc].min()); mgap=mc-mv; mmid=.5*(mv+mc); shift=tmid-mmid; A=M+shift
            r_mt,r_tm=image_shape_metrics(target,xcoord,A); gaperr=mgap-tgap; reg=float(np.sqrt(np.mean(((np.asarray(x)-x0)/scales)**2)))
            loss=.65*r_mt+.35*r_tm+args.gap_weight*abs(gaperr)+args.reg_weight*reg
        except Exception as e:
            return 100.0
        if record:
            row={'eval':neval,'loss':loss,'rmse_model_to_image':r_mt,'rmse_image_to_model':r_tm,'model_gap_eV':mgap,'image_gap_eV':tgap,'gap_error_eV':gaperr,'shift_eV':shift,'regularization':reg,**{n:float(v) for n,v in zip(names,x)}}
            history.append(row)
            if loss<best[0]: best=[loss,np.array(x,float),row.copy()]; print('BEST',json.dumps(row),flush=True)
        return float(loss)
    print(json.dumps({'elements':elements,'basis_orbitals':norb,'valence_electrons':nelec,'occupied_spatial':nocc,'image_bbox':target['bbox'],'image_gap_eV':tgap,'image_VBM_eV':tv,'image_CBM_eV':tc,'variables':names,'n_k_fit':len(xcoord)},indent=2),flush=True)
    t0=time.time(); base=objective(x0); print('BASE_LOSS',base,flush=True)
    de=differential_evolution(objective,bounds,maxiter=args.maxiter,popsize=args.popsize,seed=args.seed,x0=x0,polish=False,workers=1,updating='immediate',tol=1e-5)
    loc=minimize(objective,de.x,method='Nelder-Mead',bounds=bounds,options={'maxiter':args.local_maxiter,'maxfev':max(250,args.local_maxiter*12),'xatol':1e-5,'fatol':1e-6,'adaptive':True})
    xb=loc.x if loc.fun<de.fun else de.x; fin=objective(xb); pbest,Kbest=apply_x(defaults,specs,xb)
    with (out/f'{args.tag}_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    result={'tag':args.tag,'source_image':str(image),'cif':str(cif),'elements':elements,'basis_orbitals':norb,'valence_electrons':nelec,'occupied_spatial_bands':nocc,
      'image':{'emin':args.emin,'emax':args.emax,'fermi':args.fermi,'bbox':target['bbox'],'VBM_eV':tv,'CBM_eV':tc,'gap_eV':tgap},'path':{'labels':args.labels,'points':points.tolist(),'points_per_line':args.points_per_line,'n_k_fit':len(xcoord)},
      'variables':names,'bounds':bounds,'default_x':{n:float(v) for n,v in zip(names,x0)},'best_x':{n:float(v) for n,v in zip(names,xb)},'K':float(Kbest),'base_loss':float(base),'final_loss':float(fin),'best_seen':best[2],
      'optimizer':{'de_fun':float(de.fun),'de_success':bool(de.success),'de_message':str(de.message),'local_fun':float(loc.fun),'local_success':bool(loc.success),'local_message':str(loc.message),'local_nit':int(loc.nit),'local_nfev':int(loc.nfev)},'evaluations':neval,'elapsed_s':time.time()-t0}
    (out/f'{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n'); write_parameter_file(out/f'param_{args.tag}.txt',pbest)
    # prepend global metadata to convenient param file
    pp=out/f'param_{args.tag}.txt'; txt=pp.read_text(); pp.write_text(f'# Generic image-fitted YAeHMOP parameters\nK = {Kbest:.14g}\nWEIGHTED_HIJ = true\n\n'+txt)
    print('FINAL',json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
