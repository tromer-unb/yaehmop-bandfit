#!/usr/bin/env python3
"""Opt-in local physical YAeHMOP band calculation for carbon crystal examples.

This script is NEVER used by GitHub Actions. It requires the user's
separately installed YAeHMOP bind binary and authorized atomic table.
It computes a full physical EHT band set on the system's k path, not DFT.
"""
from pathlib import Path
import argparse,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
SYSTEMS={"graphene":"graphene","diamond":"diamond","graphite":"graphite_AB"}
def run(system,parameters,K,overlap_radius,outdir):
    if not os.environ.get("YAEHMOP_BIND") or not os.environ.get("YAEHMOP_PARAM_FILE"):
        raise EnvironmentError(
          "Set YAEHMOP_BIND and YAEHMOP_PARAM_FILE to your separately installed solver and original parameter table")
    if not Path(os.environ["YAEHMOP_BIND"]).is_file():
        raise FileNotFoundError(os.environ["YAEHMOP_BIND"])
    if not Path(os.environ["YAEHMOP_PARAM_FILE"]).is_file():
        raise FileNotFoundError(os.environ["YAEHMOP_PARAM_FILE"])
    if not Path(parameters).is_file():raise FileNotFoundError(parameters)
    name=SYSTEMS[system]
    cmd=[sys.executable,str(ROOT/"USER_WORKFLOW/bands_fit.py"),
      str(ROOT/"examples/structures"/(name+".cif")),
      str(ROOT/"examples/configs"/(name+".ini")),
      "--from-parameters",str(Path(parameters).resolve()),
      "--k",str(float(K)),"--overlap-radius",str(int(overlap_radius)),
      "--outdir",str(outdir)]
    print("Opt-in local YAeHMOP calculation:",name,
          f"radius={overlap_radius}",flush=True)
    subprocess.run(cmd,check=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system",choices=tuple(SYSTEMS),required=True)
    p.add_argument("--parameters",type=Path,required=True,
                   help="Your authorized YAeHMOP-format parameter table; independent of the one used by PBE fits")
    p.add_argument("--k",type=float,default=1.75,
                   help="Extended Hückel Wolfsberg–Helmholtz constant, default 1.75")
    p.add_argument("--overlap-radius",type=int,default=4,
                   help="Images in each periodic lattice direction; default 4")
    p.add_argument("--outdir",type=Path,default=Path("outputs/carbon"))
    a=p.parse_args()
    if not 1<=a.overlap_radius<=8:p.error("--overlap-radius must be 1..8")
    run(a.system,a.parameters,a.k,a.overlap_radius,a.outdir)
