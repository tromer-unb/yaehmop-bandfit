"""Collinear spin-polarized references: reading one channel and joining two channel fits.
No YAeHMOP."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
import numpy as np, scipy.linalg as sl
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"USER_WORKFLOW"),str(BASE/"parameter_fitter")]
from ase.io import read,write
from dft_bands import read_dft_bands
from spin_combine import read_tb

def eigenval_sp(d,k,Eu,Ed,nelect,efermi):
    d.mkdir(parents=True,exist_ok=True)
    write(d/"POSCAR",read(BASE/"examples/structures/diamond.cif"),format="vasp",direct=True)
    L=["    2    2    1    2"," 0.1E+02 0.1E-09 0.1E-09 0.1E-08 0.5E-15"," 1.0E-004","  CAR"," x",f"  {nelect}  {len(k)}  {Eu.shape[1]}"]
    for kk,eu,ed in zip(k,Eu,Ed):
        L+=["",f"  {kk[0]:.7E}  {kk[1]:.7E}  {kk[2]:.7E}  0.0000000E+00"]
        L+=[f"  {j+1}  {a:.6f}  {b:.6f}  1.0  1.0" for j,(a,b) in enumerate(zip(eu,ed))]
    (d/"EIGENVAL").write_text("\n".join(L)+"\n")
    (d/"OUTCAR").write_text(f" ISPIN  =      2\n LSORBIT =      F\n E-fermi :   {efermi:.4f}     XC(G=0)\n")

def tbfile(path,cell,H0,S0,H1,S1):
    n=len(H0);cells=[(np.zeros(3),H0,S0),(cell[0],H1,S1),(-cell[0],H1.T,S1.T)]
    L=["NP","0.000000","0.000000"]+[" ".join(f"{v:.8f}" for v in r) for r in cell]+[str(n),str(len(cells)),"#rcell x   rcell y   rcell z   i   j   ReH   ImH   S"]
    for R,H,S in cells:
        for i in range(n):
            for j in range(n): L.append(f"{R[0]:.8f} {R[1]:.8f} {R[2]:.8f} {i+1} {j+1} {H[i,j]:.10f} 0.0 {S[i,j]:.10f}")
    Path(path).write_text("\n".join(L)+"\n")

class SpinTests(unittest.TestCase):
    def test_read_channels(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/"v";k=np.array([[0,0,0],[.25,0,.25],[.5,0,.5]])
            Eu=np.array([[-5,-1,2,4]]*3,float);Ed=np.array([[-4.5,1.0,3,5]]*3,float)   # up: 2 occupied, down: 1
            eigenval_sp(d,k,Eu,Ed,3,0.0)
            with self.assertRaisesRegex(ValueError,"Spin-polarized"):read_dft_bands(d,"vasp")
            u=read_dft_bands(d,"vasp",spin_channel="up");w=read_dft_bands(d,"vasp",spin_channel="down")
            self.assertEqual((u.nocc,w.nocc),(2,1))
            np.testing.assert_allclose(w.energies,Ed)
            self.assertEqual(u.spin_channel,"up")
    def test_combine(self):
        rng=np.random.default_rng(0);n=3;cell=np.diag([3.,3.,15.])
        def ch(seed):
            r=np.random.default_rng(seed);A=r.normal(size=(n,n));H0=(A+A.T)/2;B=r.normal(size=(n,n))*.05;S0=np.eye(n)+(B+B.T)/2
            return H0,S0,r.normal(size=(n,n))*.2,r.normal(size=(n,n))*.02
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);fits={}
            for tag,seed,mid in (("up",1,-1.0),("down",2,-0.4)):
                d=t/tag;d.mkdir();H0,S0,H1,S1=ch(seed);tbfile(d/"huckel-NP.dat",cell,H0,S0,H1,S1)
                (d/"fit_summary.json").write_text(json.dumps({"K_EHT":1.75,"dft_reference":{"spin_channel":tag,"midgap_eV":mid,"fermi_eV":-0.7}}))
                (d/"basis_set-NP").write_text("bindex aspecie ax ay az l m spin\n"+"".join(f"{i+1} C 0 0 0 1 0 0\n" for i in range(n)))
                (d/"parameters_huckel.dat").write_text("END\n");fits[tag]=(H0,S0,H1,S1,mid+0.7)
            subprocess.run([sys.executable,str(BASE/"USER_WORKFLOW/spin_combine.py"),"--up",str(t/"up"),"--down",str(t/"down"),"--outdir",str(t/"sp")],check=True,capture_output=True)
            L=(t/"sp/huckel-SP.dat").read_text().splitlines()
            self.assertEqual(L[0],"SP");self.assertEqual(L[6],str(2*n))
            B=(t/"sp/basis_set-SP").read_text().splitlines()[1:]
            self.assertEqual([b.split()[-1] for b in B],["1"]*n+["-1"]*n)                  # spin up first
            # at k = 0 the eigenvalues are those of each channel, shifted by its own amount
            d=np.array([[float(x) for x in l.split()] for l in L[9:]]);nc=int(L[7]);m=2*n
            H=d[:,5].reshape(nc,m,m).sum(0);S=d[:,7].reshape(nc,m,m).sum(0)
            self.assertTrue(np.allclose(H[:n,n:],0) and np.allclose(S[n:,:n],0))          # up-down blocks zero
            want=[]
            for tag in ("up","down"):
                H0,S0,H1,S1,sh=fits[tag];Hk=H0+H1+H1.T+sh*(S0+S1+S1.T);Sk=S0+S1+S1.T
                want+=list(sl.eigh(Hk,Sk,eigvals_only=True))
            np.testing.assert_allclose(sl.eigh(H,S,eigvals_only=True),sorted(want),atol=1e-7)

if __name__=="__main__":unittest.main()
