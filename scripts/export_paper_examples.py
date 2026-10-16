#!/usr/bin/env python3
"""One-time archival exporter: original P5 Ge/Sn-chalcogenide EHT bands only.

Needs the research project itself; NOT executed in the public GitHub CI.
The public distribution contains only generated, small, precomputed CSVs.
Original third-party PBE reference arrays are deliberately NOT exported.
"""
from pathlib import Path
import csv,json
import numpy as np

def export(source_root,public_root):
    source_root=Path(source_root);public_root=Path(public_root)
    source=source_root/"results/per_material.csv"
    rows=list(csv.DictReader(source.open(newline="")))
    rows=[r for r in rows if r["group"]=="MC2D external"]
    materials=["GeS","GeSe","GeTe","SnS2","SnSe","SnTe"]
    windows=[2,4,6,8]
    data=[]
    edir=public_root/"examples/chalcogenides"
    edir.mkdir(exist_ok=True,parents=True)
    for material in materials:
        for n in windows:
            recs=[r for r in rows if r["material"]==material and int(r["n_bands"])==n]
            fit=next((r for r in recs if r["mode"]=="direct"),None)
            holdout=next((r for r in recs if r["mode"]=="zero-shot shared"),None)
            if fit is not None:
                source_npz=source_root/f"results/direct/{material}/{n}bands/nm_refined/{material}_bands.npz"
                if not source_npz.exists():
                    raise FileNotFoundError(source_npz)
                z=np.load(source_npz)
                k=np.asarray(z["k"],float)
                et=np.asarray(z["reference"],float)
                eh=np.asarray(z["YAeHMOP"],float)
                assert et.shape==eh.shape==(len(k),n)
                measured=float(np.sqrt(np.mean((eh-et)**2)))
                assert np.isclose(measured,float(fit["selected_rmse_eV"]),atol=1e-8)
                assert len(k)>10 and k.max()>k.min()
                x=(k-k.min())/(k.max()-k.min())
                out=edir/material/f"{n}bands_eht.csv"
                out.parent.mkdir(exist_ok=True,parents=True)
                with out.open("w",newline="") as f:
                    writer=csv.writer(f)
                    writer.writerow(["x"]+[f"EHT_band_{j+1}_eV" for j in range(n)])
                    writer.writerows(np.column_stack((x,eh)))
                saved=str(out.relative_to(public_root))
                print("PRECOMPUTED_P5_EHT",material,n,len(k),"k-points",measured)
            else:saved=""
            data.append({
             "material":material,"bands":n,
             "direct_rmse_eV":(f'{float(fit["selected_rmse_eV"]):.12f}' if fit else ""),
             "direct_calls":(fit["calls"] if fit else ""),
             "zero_shot_rmse_eV":(f'{float(holdout["selected_rmse_eV"]):.12f}' if holdout else ""),
             "zero_shot_calls":(holdout["calls"] if holdout else ""),
             "saved_EHT_only":saved,
            })
    out=edir/"direct_and_transfer.csv"
    with out.open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(data[0]))
        wr.writeheader();wr.writerows(data)
    summary={}
    for mode,key in (("direct","direct_rmse_eV"),("zero-shot shared","zero_shot_rmse_eV")):
        summary[mode]={}
        for n in windows:
            vals=[float(r[key]) for r in data if r["bands"]==n and r[key]]
            summary[mode][str(n)]=dict(n_materials=len(vals),mean_rmse_eV=float(np.mean(vals)) if vals else None)
    meta={
      "title":"Matched 2/4/6/8-band calibration and chemical zero-shot transfer, six Ge/Sn chalcogenides",
      "source":"Original P5 Materials Cloud/C2DB PBE direct-fit and leave-one-chemical-system-out arrays; paper Figs 2–5",
      "materials":materials,
      "windows":windows,
      "selected_states":"Exactly N/2 reference valence + N/2 reference conduction bands, with same N EHT eigenvalues in the score",
      "all_csv_spectra":"YAeHMOP EHT predictions only; third-party PBE target arrays are not included",
      "direct_vs_holdout":"Direct fits were trained on each target; zero-shot shared predictions had the material excluded from the fit",
      "ineligible_cases":"Missing entries are missing/unsupported folds, never zeros. GeTe has no eligible 8-band direct fit.",
      "summary":summary,
      "reference_provenance":"Materials Cloud PBE; consult source paper for settings and external attribution; these archives are not independent PBE redistributions."
    }
    (edir/"benchmark_summary.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("P5_EXPORT_OK",len(data),"rows",sum(bool(r["saved_EHT_only"]) for r in data),"individual EHT spectra",summary)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root",type=Path,required=True,
                    help="Research project containing P5 saved fit arrays")
    ap.add_argument("--out-root",type=Path,default=Path(__file__).resolve().parents[1])
    a=ap.parse_args()
    export(a.source_root,a.out_root)
