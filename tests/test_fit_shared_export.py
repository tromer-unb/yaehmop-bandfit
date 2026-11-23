"""fit_shared.py writes the standard per-system outputs (figure + fitting/ with H(R), S(R)).

Needs the YAeHMOP bind (skipped otherwise); the reference is synthetic."""
from pathlib import Path
import json,sys,tempfile,unittest
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"parameter_fitter"));sys.path.insert(0,str(BASE/"USER_WORKFLOW"))

def bind_ok():
    try:
        from yaehmop_generic import resolve_yaehmop_assets
        resolve_yaehmop_assets();return True
    except Exception:
        return False

@unittest.skipUnless(bind_ok(),"YAeHMOP bind not available")
class FitSharedExport(unittest.TestCase):
    def test_export_systems(self):
        from fit_shared import export_systems,valid_basis
        from structure_io import read_structure
        from yaehmop_generic import (resolve_yaehmop_assets,parse_parameter_table,subset_for_elements,
            write_parameter_file,write_periodic_input_generic,run_bind,parse_band,BandPath,
            overlap_images_for,normalized_path_coordinate)
        cif=BASE/"examples/structures/graphene.cif"
        bind,table=resolve_yaehmop_assets()
        params=valid_basis(subset_for_elements(parse_parameter_table(table),["C"]))
        pts=np.array([[0,0,0],[.5,0,0],[1/3,1/3,0],[0,0,0]],float)
        path=BandPath(("G","M","K","G"),pts,6)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);native=tmp/"p.dat";write_parameter_file(native,params)
            inp=write_periodic_input_generic(cif,tmp/"g.bind",path,1.75,True,8,overlap_images=overlap_images_for(2,3),periodic_dim=2)
            run_bind(bind,inp,native,tmp/"g.log")
            kp,model=parse_band(str(inp)+".band")
            x=normalized_path_coordinate(kp,read_structure(cif).cell)
            s=dict(name="graphene",role="train",cif=cif,structure_format=None,image=tmp/"none.png",path=path,
                   xcoord=x,T=model+0.05,W=np.ones_like(model),mask=np.ones_like(model,bool),nocc=4,
                   target_gap=float(model[:,4].min()-model[:,3].max()),target_mid=0.0,fit_window=[-10,5],
                   periodic_dim=2,overlap_images=overlap_images_for(2,3))
            s["target_mid"]=.5*(model[:,4].min()+model[:,3].max())+0.05
            out=export_systems([s],{"graphene":model},native,1.75,tmp/"out")
            d=Path(out["graphene"])
            self.assertTrue((d/"bands_comparison.png").is_file())
            for f in ("parameters_huckel.dat","huckel-NP.dat","basis_set-NP","fit_summary.json","POSCAR"):
                self.assertTrue((d/"fitting"/f).is_file(),f)
            summ=json.loads((d/"fitting/fit_summary.json").read_text())
            self.assertAlmostEqual(summ["metrics"]["fit_rmse_eV"],0.0,places=4)
            self.assertEqual(summ["tight_binding_files"]["overlap_radius"],3)

if __name__=="__main__":unittest.main()
