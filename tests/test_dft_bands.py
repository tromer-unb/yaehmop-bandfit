"""Numerical DFT band references: readers for VASP, QE, SIESTA and FHI-aims,
path segmentation (with discontinuities) and the YAeHMOP sample built from them.

Synthetic files in each package's format; no DFT code, YAeHMOP or network.
The real examples in validation/ are checked too when present.
"""
from pathlib import Path
import sys,tempfile,unittest,warnings
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"USER_WORKFLOW"),str(BASE/"parameter_fitter"),str(BASE/"source")]
from ase.io import read,write
from dft_bands import (read_dft_bands,eht_reference,auto_segments,save_reference,load_reference,
                       detect_periodic_dim,DFTBands,HARTREE,BOHR)
from fit_sparse_window import parse_sample,write_sample_input,scores

DIAMOND=read(BASE/"examples/structures/diamond.cif")
NODES=[("Γ",[0,0,0]),("X",[.5,0,.5]),("W",[.5,.25,.75]),("L",[.5,.5,.5]),("Γ",[0,0,0])]
NB,NOCC=8,4

def model(k):
    """Smooth gapped 8-band model on fractional k (eV)."""
    c=np.cos(2*np.pi*np.asarray(k)).sum(axis=-1)
    base=np.array([-12,-9,-8,-7,2,3,5,8],float)
    return np.sort(base+np.outer(c,[.5,.4,.3,.2,-.3,-.4,.5,.6]).reshape(-1,NB),axis=1)   # sorted, as the reader returns

def line_kpoints(n):
    """VASP line mode: n points per segment, nodes repeated."""
    ks=[]
    for (_,a),(_,b) in zip(NODES[:-1],NODES[1:]):
        a,b=np.array(a,float),np.array(b,float)
        ks+= [a+(b-a)*i/(n-1) for i in range(n)]
    return np.array(ks)

def write_vasp(d,n=11):
    d.mkdir(parents=True,exist_ok=True)
    write(d/"POSCAR",DIAMOND,format="vasp",direct=True)
    k=line_kpoints(n);E=model(k)
    lines=["    2    2    1    1"," 0.1E+02 0.1E-09 0.1E-09 0.1E-08 0.5E-15"," 1.0E-004","  CAR"," diamond",
           f"     {2*NOCC}  {len(k)}  {NB}"]
    for kk,e in zip(k,E):
        lines+=["",f"  {kk[0]:.7E}  {kk[1]:.7E}  {kk[2]:.7E}  0.0000000E+00"]
        lines+=[f"  {j+1:4d}  {v:14.6f}  {1.0 if j<NOCC else 0.0:9.6f}" for j,v in enumerate(e)]
    (d/"EIGENVAL").write_text("\n".join(lines)+"\n")
    kp=["path",str(n),"Line-mode","rec"]
    for (la,a),(lb,b) in zip(NODES[:-1],NODES[1:]):
        kp+=[f"{a[0]} {a[1]} {a[2]} ! {'G' if la=='Γ' else la}",f"{b[0]} {b[1]} {b[2]} ! {'G' if lb=='Γ' else lb}",""]
    (d/"KPOINTS").write_text("\n".join(kp)+"\n")
    (d/"INCAR").write_text("ISPIN = 1\nLSORBIT = .FALSE.\n")
    return k,E

def write_qe_xml(d,n=10):
    d.mkdir(parents=True,exist_ok=True)
    cell=np.asarray(DIAMOND.cell);alat=np.linalg.norm(cell[0])/BOHR
    k=line_kpoints(n);E=model(k)
    kc=(k@(2*np.pi*np.linalg.inv(cell).T))*alat*BOHR/(2*np.pi)
    ks="".join(f"<ks_energies><k_point weight=\"1\">{a} {b} {c}</k_point><npw>1</npw>"
               f"<eigenvalues size=\"{NB}\">{' '.join(str(v/HARTREE) for v in e)}</eigenvalues>"
               f"<occupations size=\"{NB}\">{' '.join('1' if j<NOCC else '0' for j in range(NB))}</occupations></ks_energies>"
               for (a,b,c),e in zip(kc,E))
    atoms="".join(f"<atom name=\"{s}\" index=\"{i+1}\">{p[0]/BOHR} {p[1]/BOHR} {p[2]/BOHR}</atom>"
                  for i,(s,p) in enumerate(zip(DIAMOND.get_chemical_symbols(),DIAMOND.positions)))
    cellx="".join(f"<a{i+1}>{v[0]/BOHR} {v[1]/BOHR} {v[2]/BOHR}</a{i+1}>" for i,v in enumerate(cell))
    xml=(f"<?xml version=\"1.0\"?><qes:espresso xmlns:qes=\"http://www.quantum-espresso.org/ns/qes/qes-1.0\">"
         f"<input><control_variables><calculation>bands</calculation></control_variables></input><output>"
         f"<atomic_structure nat=\"2\" alat=\"{alat}\"><atomic_positions>{atoms}</atomic_positions><cell>{cellx}</cell></atomic_structure>"
         f"<band_structure><lsda>false</lsda><noncolin>false</noncolin><spinorbit>false</spinorbit><nbnd>{NB}</nbnd>"
         f"<nelec>{2*NOCC}.0</nelec><highestOccupiedLevel>{E[:,NOCC-1].max()/HARTREE}</highestOccupiedLevel>"
         f"<nks>{len(k)}</nks>{ks}</band_structure></output></qes:espresso>")
    (d/"c.save").mkdir(exist_ok=True)
    (d/"c.save"/"data-file-schema.xml").write_text(xml)
    bl="\n".join(f"{p[0]} {p[1]} {p[2]} {n} ! {lab}" for lab,p in NODES)
    (d/"bands.in").write_text(f"&control\n calculation='bands'\n/\nK_POINTS crystal_b\n{len(NODES)}\n{bl}\n")
    return k,E

def write_siesta(d,n=10,jump=False):
    d.mkdir(parents=True,exist_ok=True)
    cell=np.asarray(DIAMOND.cell);frac=DIAMOND.get_scaled_positions()
    rows=[(1,NODES[0])]+[(n,x) for x in NODES[1:]]
    if jump:  # Γ-X | W-L-Γ : the W line starts a new piece (n = 1)
        rows=[(1,NODES[0]),(n,NODES[1]),(1,NODES[2]),(n,NODES[3]),(n,NODES[4])]
    k=[];segs=[]
    for i,(m,(lab,p)) in enumerate(rows):
        p=np.array(p,float)
        if i==0 or m==1:k.append(p);continue
        a=k[-1];k+=[a+(p-a)*j/m for j in range(1,m+1)]
    k=np.array(k);E=model(k)
    rec=2*np.pi*np.linalg.inv(cell).T
    cart=k@rec*BOHR;steps=np.linalg.norm(np.diff(cart,axis=0),axis=1)
    if jump:
        steps[n]=0.0   # SIESTA does not advance the path coordinate over a jump
    x=np.r_[0,np.cumsum(steps)]
    body="".join(f"{xx:12.6f}"+"".join(f"{v:12.4f}" for v in e)+"\n" for xx,e in zip(x,E))
    labs="".join(f"{xx:12.6f}   '{lab}'\n" for xx,(lab,_) in zip([0],[NODES[0]]))
    (d/"c.bands").write_text(f"  {(E[:,NOCC-1].max()+E[:,NOCC].min())/2:.6f}\n  0.0 {x[-1]:.6f}\n  {E.min():.3f} {E.max():.3f}\n"
                             f"  {NB} 1 {len(k)}\n"+body+"  1\n"+labs)
    bl="\n".join(f"{m} {p[0]} {p[1]} {p[2]} {lab}" for m,(lab,p) in rows)
    coords="\n".join(f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f} 1" for f in frac)
    lv="\n".join(" ".join(f"{v:.10f}" for v in r) for r in cell)
    (d/"c.fdf").write_text(f"SystemLabel c\nNumberOfAtoms 2\nLatticeConstant 1.0 Ang\n%block LatticeVectors\n{lv}\n%endblock LatticeVectors\n"
        f"%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\nAtomicCoordinatesFormat Fractional\n"
        f"%block AtomicCoordinatesAndAtomicSpecies\n{coords}\n%endblock AtomicCoordinatesAndAtomicSpecies\n"
        f"BandLinesScale ReciprocalLatticeVectors\n%block BandLines\n{bl}\n%endblock BandLines\n")
    return k,E

def write_aims(d,n=10):
    d.mkdir(parents=True,exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore");write(d/"geometry.in",DIAMOND,format="aims",scaled=True)
    ctrl=["xc pbe","spin none"];allk=[];allE=[]
    for i,((la,a),(lb,b)) in enumerate(zip(NODES[:-1],NODES[1:])):
        a,b=np.array(a,float),np.array(b,float)
        k=np.array([a+(b-a)*j/(n-1) for j in range(n)]);E=model(k)
        ctrl.append(f"output band {a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]} {n} {la} {lb}")
        rows=[f"{j+1} {kk[0]} {kk[1]} {kk[2]} "+" ".join(f"{2.0 if m<NOCC else 0.0} {v}" for m,v in enumerate(e))
              for j,(kk,e) in enumerate(zip(k,E))]
        (d/f"band1{i+1:03d}.out").write_text("\n".join(rows)+"\n")
        allk.append(k);allE.append(E)
    (d/"control.in").write_text("\n".join(ctrl)+"\n")
    (d/"aims.out").write_text("Invoking FHI-aims ...\n  | Chemical potential (Fermi level):    0.5 eV\n")
    return np.vstack(allk),np.vstack(allE)

def write_w90(d,n=10,fermi_in_win=True):
    """Wannier90 seedname.win + seedname_band.kpt/.dat/.labelinfo.dat (bands_plot output)."""
    d.mkdir(parents=True,exist_ok=True)
    cell=np.asarray(DIAMOND.cell);frac=DIAMOND.get_scaled_positions()
    k=[];lab=[]
    for i,((la,a),(lb,b)) in enumerate(zip(NODES[:-1],NODES[1:])):
        a,b=np.array(a,float),np.array(b,float)
        pts=[a+(b-a)*j/n for j in range(n+(1 if i==len(NODES)-2 else 0))]
        lab.append((len(k)+1,la,a));k+=pts
    lab.append((len(k),NODES[-1][0],np.array(NODES[-1][1],float)))
    k=np.array(k);E=model(k)
    uc="\n".join(" ".join(f"{v:.10f}" for v in r) for r in cell)
    at="\n".join(f"C {f[0]:.10f} {f[1]:.10f} {f[2]:.10f}" for f in frac)
    win=f"num_wann = {NB}\nbands_plot = true\n"+(f"fermi_energy = {(E[:,NOCC-1].max()+E[:,NOCC].min())/2:.6f}\n" if fermi_in_win else "")
    (d/"w.win").write_text(win+f"begin unit_cell_cart\nang\n{uc}\nend unit_cell_cart\nbegin atoms_frac\n{at}\nend atoms_frac\n")
    (d/"w_band.kpt").write_text(f"{len(k)}\n"+"".join(f"{a:.8f} {b:.8f} {c:.8f} 1.0\n" for a,b,c in k))
    x=np.r_[0,np.cumsum(np.linalg.norm(np.diff(k@(2*np.pi*np.linalg.inv(cell).T),axis=0),axis=1))]
    (d/"w_band.dat").write_text("".join("".join(f"{xx:.8E} {e:.8E}\n" for xx,e in zip(x,E[:,j]))+"\n" for j in range(NB)))
    (d/"w_band.labelinfo.dat").write_text("".join(f"{'G' if l=='Γ' else l} {i} 0.0 {kk[0]} {kk[1]} {kk[2]}\n" for i,l,kk in lab))
    return k,E

class ReaderTests(unittest.TestCase):
    def check(self,code,writer,**kw):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/code
            k,E=writer(d,**kw)
            r=read_dft_bands(d,code)
            self.assertEqual(r.nocc,NOCC)
            self.assertEqual(r.atoms.get_chemical_symbols(),["C","C"])
            np.testing.assert_allclose(np.asarray(r.atoms.cell),np.asarray(DIAMOND.cell),atol=1e-5)
            np.testing.assert_allclose(r.kfrac,k,atol=2e-6)
            np.testing.assert_allclose(r.energies,E,atol=2e-4)
            er=eht_reference(r,7,3)
            self.assertEqual([l for _,l in er.ticks],[l for l,_ in NODES] if not kw.get("jump") else ["Γ","X|W","L","Γ"])
            self.assertEqual(len(er.mask),7*(len(er.nodes)-1)+1)
            # DFT values interpolated at the YAeHMOP k points equal the model there
            L=len(er.nodes)-1
            pts=[]
            nodes=np.array([n for n in er.nodes])
            for line in range(L):
                for i in range(7+(1 if line==L-1 else 0)):
                    pts.append(nodes[line]+(nodes[line+1]-nodes[line])*i/7)
            pts=np.array(pts)[er.mask]
            # linear interpolation of a smooth model: small but nonzero error
            self.assertLess(np.abs(er.energies-model(pts)).max(),0.35)
            return r,er

    def test_vasp(self):
        r,_=self.check("vasp",write_vasp)
        self.assertEqual(len(r.segments),4)
    def test_qe_xml(self):self.check("qe",write_qe_xml)
    def test_siesta(self):self.check("siesta",write_siesta)
    def test_siesta_jump(self):
        r,er=self.check("siesta",write_siesta,jump=True)
        self.assertEqual(er.real_lines,[True,False,True,True])
        self.assertEqual(int((~er.mask).sum()),6)   # interior of the X->W connecting line
    def test_aims(self):self.check("aims",write_aims)
    def test_wannier90(self):self.check("wannier90",write_w90)
    def test_wannier90_needs_fermi(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/"w";write_w90(d,fermi_in_win=False)
            with self.assertRaisesRegex(ValueError,"Fermi"):read_dft_bands(d,"wannier90")
            (d/"OUTCAR").write_text(f" E-fermi :   {(model(np.zeros((1,3)))[0,NOCC-1]):.4f}     XC(G=0)\n")
            self.assertEqual(read_dft_bands(d,"wannier90").nocc,NOCC)

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/"v";write_vasp(d)
            (d/"INCAR").write_text("LSORBIT = .TRUE.\n")
            with self.assertRaisesRegex(ValueError,"spin-orbit"):read_dft_bands(d,"vasp")
            (d/"INCAR").write_text("ISPIN = 1\n")
            with self.assertRaisesRegex(ValueError,"gap"):read_dft_bands(d,"vasp",valence_bands=3)
            with self.assertRaises(ValueError):read_dft_bands(d,"castep")
            with self.assertRaises(FileNotFoundError):read_dft_bands(Path(tmp)/"none","vasp")
            s=Path(tmp)/"s";write_siesta(s)
            (s/"c.fdf").write_text((s/"c.fdf").read_text()+"SpinOrbit .true.\n")
            with self.assertRaisesRegex(ValueError,"spin-orbit"):read_dft_bands(s,"siesta")

    def test_auto_segments_and_dimension(self):
        k=line_kpoints(6)
        dedup=np.array([k[0]]+[x for a,x in zip(k[:-1],k[1:]) if not np.allclose(a,x)])
        segs=auto_segments(dedup,DIAMOND.cell)
        self.assertEqual([len(s) for s in segs],[6,6,6,6])
        self.assertEqual(detect_periodic_dim(DIAMOND),3)
        g=read(BASE/"examples/structures/graphene.cif")
        self.assertEqual(detect_periodic_dim(g),2)

    def test_sample_for_fitter(self):
        carbon={"C":{"atnum":6,"nval":4,
            "s":{"nzeta":1,"n":2,"Hii":-21.4,"zeta1":1.625,"zeta2":0.,"c1":1.,"c2":0.},
            "p":{"nzeta":1,"n":2,"Hii":-11.4,"zeta1":1.625,"zeta2":0.,"c1":1.,"c2":0.}}}
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);write_siesta(tmp/"s",jump=True)
            r=read_dft_bands(tmp/"s","siesta")
            npz=save_reference(r,tmp/"ref.npz")
            write(tmp/"POSCAR",r.atoms,format="vasp",direct=True)
            back=load_reference(npz,read(tmp/"POSCAR"))
            np.testing.assert_allclose(back.energies,r.energies)
            e={"name":"c","structure":str(tmp/"POSCAR"),"reference":str(npz),"reference_format":"dft",
               "point_dim":3,"n_pairs":2,"points_per_line":5}
            s=parse_sample(e,tmp,carbon,{})
            self.assertEqual(s["target"].shape,(int(s["kmask"].sum()),4))
            self.assertAlmostEqual(float(s["target"][:,1].max()+s["target"][:,2].min()),0.0,places=10)
            inp=write_sample_input(s,tmp/"c.bind",1.75).read_text()
            self.assertIn("Lattice\n3",inp)
            self.assertIn(f"Band\n5\n{len(s['path'].labels)}",inp)
            # scores applies the mask: full-length EHT bands, scored on DFT points only
            fake=np.zeros((len(s["kmask"]),s["norb"]));fake[:,s["nocc"]:]=1.
            m,sel=scores(fake,s)
            self.assertEqual(sel.shape,s["target"].shape)
            with self.assertRaisesRegex(ValueError,"cell"):
                load_reference(npz,read(BASE/"examples/structures/graphene.cif"))

VALID=BASE/"validation"
EXPECT={"vasp/TiBr2_monolayer_pbe":(13,2),"vasp/ZnI2_monolayer_pbe":(13,2),"vasp/In2S2_monolayer_pbe":(27,2),
        "qe/Si_diamond_lda":(4,3),"qe/Si_wurtzite_lda":(8,3),"qe/SiC_zincblende_pbe":(4,3),
        "siesta/Al2O2_bilayer_test":(9,2),"siesta/CNR_nanoribbon_1d_pbe":(112,1),
        "aims/MoS2_monolayer_pbe":(37,2),"aims/MoS2_monolayer_hse06_soc":(37,2)}

@unittest.skipUnless(VALID.is_dir() and any(VALID.glob("*/*")),"validation/ examples not present")
class ValidationFolderTests(unittest.TestCase):
    def test_examples(self):
        for rel,(nocc,dim) in EXPECT.items():
            d=VALID/rel
            if not d.is_dir():continue
            with self.subTest(example=rel):
                r=read_dft_bands(d,rel.split("/")[0])
                self.assertEqual(r.nocc,nocc)
                self.assertEqual(detect_periodic_dim(r.atoms),dim)
                er=eht_reference(r,8,dim)
                self.assertTrue(er.mask.any())
    def test_soc_refused(self):
        d=VALID/"siesta/MoS2_monolayer_pbe_soc"
        if d.is_dir():
            with self.assertRaisesRegex(ValueError,"spin-orbit"):read_dft_bands(d,"siesta")

if __name__=="__main__":unittest.main()


class MatrixDumpTests(unittest.TestCase):
    def test_read_matrix_dump_and_labels(self):
        from yaehmop_generic import read_matrix_dump,orbital_labels
        from bands_fit import path_kpoints
        rng=np.random.default_rng(1);n=4;nk=3
        H=rng.normal(size=(nk,n,n))+1j*rng.normal(size=(nk,n,n));H=H+np.conj(np.transpose(H,(0,2,1)))
        packed=np.zeros((nk,n,n))
        for q in range(nk):
            iu=np.triu_indices(n,1)
            packed[q][np.diag_indices(n)]=H[q].real.diagonal()
            packed[q][iu]=H[q][iu].real
            packed[q][(iu[1],iu[0])]=H[q][iu].imag
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"x.HAM"
            f.write_bytes(np.array([nk,n],np.int32).tobytes()+packed.astype(np.float64).tobytes())
            np.testing.assert_allclose(read_matrix_dump(f),np.conj(H))  # usual e^{+ik.R} convention
        params={"Mo":{"s":{},"p":{},"d":{}},"S":{"s":{},"p":{}}}
        lab=orbital_labels(params,["Mo","S","S"])
        self.assertEqual(len(lab),9+4+4)
        self.assertEqual(lab[4],"Mo1:dx2-y2");self.assertEqual(lab[9],"S2:s")
        k=path_kpoints([[0,0,0],[.5,0,0],[.5,.5,0]],4)
        self.assertEqual(len(k),9);np.testing.assert_allclose(k[4],[.5,0,0])
