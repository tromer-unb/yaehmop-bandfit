"""DFT members of a family fit: spin averaging, semicore removal, slab paths with
out-of-plane segments and the family manifest with DFT folders. No YAeHMOP solver."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np,yaml
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"USER_WORKFLOW"),str(BASE/"parameter_fitter"),str(BASE/"source"),str(BASE/"tests")]
from ase.io import read,write
from ase.build import mx2
from dft_bands import read_dft_bands,drop_lowest_bands,semicore_band_count,eht_reference
from test_spin import eigenval_sp
import family_fit as fm

class FamilyDFT(unittest.TestCase):
    def test_spin_average(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/"v";k=np.array([[0,0,0],[.25,0,.25],[.5,0,.5]])
            Eu=np.array([[-5,-1,2,4]]*3,float);Ed=np.array([[-4.8,-0.8,2.2,4.4]]*3,float)
            eigenval_sp(d,k,Eu,Ed,4,0.5)
            r=read_dft_bands(d,"vasp",spin_channel="average")
            np.testing.assert_allclose(r.energies,(Eu+Ed)/2)
            self.assertEqual((r.nocc,r.spin_channel,r.nelect),(2,"average",4.0))

    def test_semicore_drop(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/"v";k=np.array([[0,0,0],[.25,0,.25],[.5,0,.5]])
            E=np.array([[-40,-12,-1,2,4]]*3,float)
            eigenval_sp(d,k,E,E,6,0.5)
            r=read_dft_bands(d,"vasp")
            self.assertEqual(semicore_band_count(r,4),1)
            with self.assertRaises(ValueError):semicore_band_count(r,5)
            drop_lowest_bands(r,1)
            self.assertEqual((r.nocc,r.nelect,r.skipped_bands,r.energies.shape[1]),(2,4.0,1,4))
            self.assertAlmostEqual(r.energies.min(),-12)

    def test_slab_path_drops_out_of_plane_segments(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/"v";atoms=mx2("TiS2",kind="1T",a=3.4,vacuum=10)
            nodes=[[0,0,0],[.5,0,0],[1/3,1/3,0],[0,0,0],[0,0,.5],[.5,0,.5]]   # G M K G A L
            nper=5;ks=[]
            for a,b in zip(nodes[:-1],nodes[1:]):
                ks+=[np.array(a)+(np.array(b)-a)*i/(nper-1) for i in range(nper)]
            ks=np.array(ks)
            c=np.cos(2*np.pi*ks[:,:2]).sum(1)
            E=np.column_stack([-3+c,1+.5*c])
            eigenval_sp(d,ks,E,E,2,-1.0)
            write(d/"POSCAR",atoms,format="vasp",direct=True)
            (d/"KPOINTS").write_text("k\n%d\nLine\nReciprocal\n"%nper+"".join(
                f"{a[0]} {a[1]} {a[2]} ! X\n{b[0]} {b[1]} {b[2]} ! Y\n\n" for a,b in zip(nodes[:-1],nodes[1:])))
            r=read_dft_bands(d,"vasp")
            er=eht_reference(r,4,2)
            self.assertEqual(len(er.nodes),5)            # G M K G (=A) L: the G-A line is gone
            self.assertTrue(all(er.real_lines))
            # A-L in 2D equals G-M: same DFT energies
            np.testing.assert_allclose(er.energies[-5:],er.energies[:5],atol=1e-9)

    def test_manifest_with_dft_folders(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);k=np.array([[0,0,0],[.25,0,.25],[.5,0,.5]]);E=np.array([[-5,-1,2,4]]*3,float)
            for n in ("a","b","c"):eigenval_sp(t/n,k,E,E,4,0.5)
            for i,n in enumerate(("b","c")):        # distinct structures (same cell would be a leak)
                a=read(t/n/"POSCAR");a.positions[0]+=0.01*(i+1);write(t/n/"POSCAR",a,format="vasp",direct=True)
            m={"n_pairs":1,"defaults":{"dft_code":"vasp","spin_channel":"average"},
               "systems":[{"name":"a","dft_dir":"a"},{"name":"b","dft_dir":"b","weight":2},
                          {"name":"c","dft_dir":"c","role":"test"}]}
            (t/"f.yaml").write_text(yaml.safe_dump(m))
            _,_,train,test=fm.load_manifest(t/"f.yaml")
            self.assertEqual([s["kind"] for s in train+test],["dft"]*3)
            self.assertEqual(train[1]["weight"],2.0)
            self.assertEqual(train[0]["spin_channel"],"average")
            self.assertEqual(set(train[0]["symbols"]),{"C"})
            m["systems"][2]["dft_dir"]="a"
            (t/"f.yaml").write_text(yaml.safe_dump(m))
            with self.assertRaisesRegex(ValueError,"also a training folder|identical"):fm.load_manifest(t/"f.yaml")

class SlabContiguity(unittest.TestCase):
    def test_wrapped_atom_brought_back(self):
        from yaehmop_generic import contiguous_along_vacuum
        a=mx2("TiS2",kind="1T",a=3.4,vacuum=10)
        f=a.get_scaled_positions();f[:,2]-=f[:,2].min()          # bottom S at z = 0
        a.set_scaled_positions(f)
        w=a.copy();g=w.get_scaled_positions(wrap=False);g[0,2]+=1.0;w.set_scaled_positions(g)   # wrapped
        b=contiguous_along_vacuum(w,2)
        np.testing.assert_allclose(b.get_scaled_positions(wrap=False)[:,2],a.get_scaled_positions(wrap=False)[:,2],atol=1e-9)
        c=contiguous_along_vacuum(a,2)                            # already contiguous: unchanged
        np.testing.assert_allclose(c.positions,a.positions,atol=1e-9)
        d=contiguous_along_vacuum(w,3)                            # 3D: never touched
        np.testing.assert_allclose(d.positions,w.positions,atol=1e-9)

class DoubleZeta(unittest.TestCase):
    def test_make_double_zeta(self):
        from band_objective import make_double_zeta
        from fit_from_image import build_specs,physical_ok
        P={"S":{"atnum":16,"nval":6,"s":dict(nzeta=1,n=3,Hii=-20.,zeta1=2.122,zeta2=0.,c1=1.,c2=0.),
                "p":dict(nzeta=1,n=3,Hii=-11.,zeta1=1.827,zeta2=0.,c1=1.,c2=0.)}}
        Q=make_double_zeta(P,"S:p")
        self.assertEqual(P["S"]["p"]["nzeta"],1)                 # copy, original untouched
        self.assertEqual(Q["S"]["p"]["nzeta"],2)
        self.assertLess(Q["S"]["p"]["zeta2"],Q["S"]["p"]["zeta1"])
        self.assertTrue(physical_ok(Q))
        names=[f"{a}_{b}_{c}" for a,b,c in build_specs(Q,["S"])[0]]
        self.assertIn("S_p_zeta2",names);self.assertIn("S_p_ratio",names)
        with self.assertRaises(ValueError):make_double_zeta(P,"S:d")

    def test_binary_guard(self):
        from yaehmop_generic import check_double_zeta_sp,write_parameter_file
        from band_objective import make_double_zeta
        P={"S":{"atnum":16,"nval":6,"s":dict(nzeta=1,n=3,Hii=-20.,zeta1=2.122,zeta2=0.,c1=1.,c2=0.),
                "p":dict(nzeta=1,n=3,Hii=-11.,zeta1=1.827,zeta2=0.,c1=1.,c2=0.)}}
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);inp=t/"r.bind"
            write_parameter_file(t/"sz.dat",P);write_parameter_file(t/"dz.dat",make_double_zeta(P,"S:p"))
            (t/"r.bind.out").write_text("ATOM: S\n\t3S:   -20.0000     2.1220\n\t3P:   -11.0000     1.8270\n")
            check_double_zeta_sp("bind",inp,t/"sz.dat")              # single zeta: nothing to check
            with self.assertRaisesRegex(RuntimeError,"ignores the second exponent"):
                check_double_zeta_sp("bind",inp,t/"dz.dat")          # original program's listing
            (t/"r.bind.out").write_text("ATOM: S\n\t3S:   -20.0000     2.1220\n\t3P DZ:   -11.0  0.8  1.8  0.3  0.9\n")
            check_double_zeta_sp("bind",inp,t/"dz.dat")              # fork with double-zeta p

if __name__=="__main__":unittest.main()
