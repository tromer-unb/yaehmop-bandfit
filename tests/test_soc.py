"""On-site spin-orbit term, spinor ordering and the WanTiBEXOS SOC file. No YAeHMOP."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"parameter_fitter")]
from soc import angular_momentum,soc_matrix,soc_shells,spinor,soc_bands,write_wtb_soc,SHELL_ORBITALS

class SOCTests(unittest.TestCase):
    def test_splittings(self):
        for l,sh in ((1,"p"),(2,"d"),(3,"f")):
            labs=[f"X1:{o}" for o in SHELL_ORBITALS[l]]
            e=np.linalg.eigvalsh(soc_matrix(labs,{("X",sh):0.2}))/0.2
            v,c=np.unique(np.round(e,8),return_counts=True)
            self.assertEqual(dict(zip(v,c)),{-(l+1)/2:2*l,l/2:2*l+2})
    def test_angular_momentum(self):
        for l in (1,2,3):
            Lx,Ly,Lz=angular_momentum(l)
            np.testing.assert_allclose(Lx@Ly-Ly@Lx,1j*Lz,atol=1e-12)     # [Lx,Ly] = i Lz
            np.testing.assert_allclose(Lx@Lx+Ly@Ly+Lz@Lz,l*(l+1)*np.eye(2*l+1),atol=1e-12)
        Lx,Ly,Lz=angular_momentum(1)
        v=np.array([1,1j,0])/np.sqrt(2)
        np.testing.assert_allclose(Lz@v,v)                                 # px + i py has m = +1
    def test_spin_up_first(self):
        labs=["A1:s","A1:px","A1:py","A1:pz","B2:s"]
        H=soc_matrix(labs,{("A","p"):0.3})
        n=len(labs)
        np.testing.assert_allclose(H,H.conj().T)
        self.assertTrue(np.allclose(H[0],0) and np.allclose(H[n-1],0))    # s orbitals untouched
        np.testing.assert_allclose(H[1:4,1:4],-H[n+1:n+4,n+1:n+4])        # up-up = -(dn-dn) (Lz)
        self.assertEqual(soc_shells(labs),[("A","p")])
    def test_spinor_and_file(self):
        labs=["A1:px","A1:py","A1:pz"];n=3
        Hk=np.diag([-1.,-1.,-1.])[None];Sk=np.eye(3)[None]
        Hs,Ss=spinor(Hk[0],Sk[0],soc_matrix(labs,{("A","p"):0.4}))
        self.assertEqual(Hs.shape,(6,6));np.testing.assert_allclose(Ss,np.eye(6))
        np.testing.assert_allclose(sorted(soc_bands(Hk,Sk,soc_matrix(labs,{("A","p"):0.4}))[0]),[-1.4,-1.4,-.8,-.8,-.8,-.8])
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);cell=np.eye(3)*3
            write_wtb_soc(d/"h.dat",[(np.zeros(3,int),np.diag([-1.,-1.,-1.]),np.eye(3))],cell,labs,
                          soc_matrix(labs,{("A","p"):0.4}),-1.0,np.zeros((1,3)),["A"],d/"b")
            L=(d/"h.dat").read_text().splitlines()
            self.assertEqual(L[0],"SOC");self.assertEqual(L[6],"6");self.assertEqual(L[7],"1")
            self.assertEqual(len(L),9+36)
            B=(d/"b").read_text().splitlines()
            self.assertEqual([l.split()[-1] for l in B[1:]],["1"]*3+["-1"]*3)  # spin up first

if __name__=="__main__":unittest.main()
