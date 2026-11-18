"""Structure files from VASP, Quantum ESPRESSO, SIESTA and FHI-aims are
equivalent to the CIF: same atoms, cell and identical YAeHMOP input.

No YAeHMOP, DFT, fitting or network.
"""
from pathlib import Path
import re,sys,tempfile,unittest,warnings
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"USER_WORKFLOW"),str(BASE/"parameter_fitter"),str(BASE/"source")]
from ase.io import read,write
from structure_io import (read_structure,detect_structure_format,normalize_structure_format,
                          structure_label,STRUCTURE_FORMATS)
from yaehmop_generic import BandPath,write_periodic_input_generic,overlap_images_for
from fit_sparse_window import parse_sample,write_sample_input
from prepare_band_png import prepare

BOHR=0.529177210903
MATERIALS={"diamond":3,"graphene":2,"graphite_AB":3}

def cif_atoms(name):
    return read(BASE/"examples/structures"/f"{name}.cif")

def numbers(text):
    return np.array([float(x) for x in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?",text)])

def species_block(atoms):
    zs=list(dict.fromkeys(atoms.numbers))
    return zs,"\n".join(f"{i} {z} X{z}" for i,z in enumerate(zs,1))

def write_variants(atoms,d):
    """One file per supported writer/variant, all holding the same crystal."""
    d.mkdir(parents=True,exist_ok=True)
    out={}
    write(d/"POSCAR",atoms,format="vasp",direct=True); out["POSCAR"]="vasp"
    write(d/"CONTCAR",atoms,format="vasp",direct=False); out["CONTCAR"]="vasp"
    write(d/"material.vasp",atoms,format="vasp",direct=True); out["material.vasp"]="vasp"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write(d/"geometry.in",atoms,format="aims",scaled=True); out["geometry.in"]="aims"
        write(d/"aims_cart.in",atoms,format="aims",scaled=False); out["aims_cart.in"]="aims"
    pseudo={s:f"{s}.UPF" for s in set(atoms.get_chemical_symbols())}
    write(d/"pw.pwi",atoms,format="espresso-in",pseudopotentials=pseudo); out["pw.pwi"]="qe"
    write(d/"scf.in",atoms,format="espresso-in",pseudopotentials=pseudo,crystal_coordinates=True); out["scf.in"]="qe"
    cell=np.asarray(atoms.cell); frac=atoms.get_scaled_positions(wrap=False); pos=atoms.positions
    zs,spec=species_block(atoms); idx=[zs.index(z)+1 for z in atoms.numbers]
    def rows(m,fmt="{: .12f}"):
        return "\n".join(" ".join(fmt.format(v) for v in r) for r in m)
    coords=lambda m:"\n".join(" ".join(f"{v: .12f}" for v in r)+f" {i}" for r,i in zip(m,idx))
    # 1) fractional, LatticeConstant in Ang, comments, %include of the species
    (d/"species.fdf").write_text(f"%block ChemicalSpeciesLabel\n{spec}\n%endblock ChemicalSpeciesLabel\n")
    (d/"frac.fdf").write_text(
        "# SIESTA test input\nSystemLabel test ! comment\nNumberOfAtoms %d\n"
        "LatticeConstant 1.0 Ang\n%%block LatticeVectors\n%s\n%%endblock LatticeVectors\n"
        "%%include species.fdf\nAtomic-Coordinates-Format  Fractional\n"
        "%%block AtomicCoordinatesAndAtomicSpecies\n%s\n%%endblock AtomicCoordinatesAndAtomicSpecies\n"
        %(len(atoms),rows(cell),coords(frac)))
    out["frac.fdf"]="siesta"
    # 2) cartesian Ang, LatticeConstant in Bohr (vectors in units of it)
    alat=2.0
    (d/"ang.fdf").write_text(
        f"LatticeConstant {alat} Bohr\n%block LatticeVectors\n{rows(cell/(alat*BOHR))}\n%endblock LatticeVectors\n"
        f"%block ChemicalSpeciesLabel\n{spec}\n%endblock ChemicalSpeciesLabel\n"
        f"AtomicCoordinatesFormat NotScaledCartesianAng\n"
        f"%block AtomicCoordinatesAndAtomicSpecies\n{coords(pos)}\n%endblock AtomicCoordinatesAndAtomicSpecies\n")
    out["ang.fdf"]="siesta"
    # 3) ScaledCartesian (units of LatticeConstant), LatticeParameters, default Bohr-free
    a0=float(atoms.cell.lengths()[0]); cp=atoms.cell.cellpar()
    from ase.geometry import cellpar_to_cell
    std=cellpar_to_cell(cp)  # SIESTA/ASE orientation of LatticeParameters
    stdpos=frac@std
    (d/"scaled.fdf").write_text(
        f"LatticeConstant {a0} Ang\n%block LatticeParameters\n"
        f"1.0 {cp[1]/a0:.14f} {cp[2]/a0:.14f} {cp[3]:.12f} {cp[4]:.12f} {cp[5]:.12f}\n%endblock LatticeParameters\n"
        f"%block ChemicalSpeciesLabel\n{spec}\n%endblock ChemicalSpeciesLabel\n"
        f"AtomicCoordinatesFormat ScaledCartesian\n"
        f"%block AtomicCoordinatesAndAtomicSpecies\n{coords(stdpos/a0)}\n%endblock AtomicCoordinatesAndAtomicSpecies\n")
    out["scaled.fdf"]="siesta"
    # 4) default coordinate format (Bohr) and a block read from another file
    (d/"coords.txt").write_text(coords(pos/BOHR)+"\n")
    (d/"bohr.fdf").write_text(
        f"LatticeConstant 1.0 Ang\n%block LatticeVectors\n{rows(cell)}\n%endblock LatticeVectors\n"
        f"%block ChemicalSpeciesLabel\n{spec}\n%endblock ChemicalSpeciesLabel\n"
        f"%block AtomicCoordinatesAndAtomicSpecies < coords.txt\n")
    out["bohr.fdf"]="siesta"
    # SIESTA outputs: .XV (Bohr, cartesian + velocities), .STRUCT_OUT (Ang, fractional)
    (d/"siesta.XV").write_text(
        "\n".join(" ".join(f"{v: .12f}" for v in r)+"  0.0 0.0 0.0" for r in cell/BOHR)+
        f"\n{len(atoms)}\n"+"\n".join(f"{i} {z} "+" ".join(f"{v: .12f}" for v in r)+" 0.0 0.0 0.0"
                                     for i,z,r in zip(idx,atoms.numbers,pos/BOHR))+"\n")
    out["siesta.XV"]="siesta-xv"
    (d/"siesta.STRUCT_OUT").write_text(
        rows(cell)+f"\n{len(atoms)}\n"+"\n".join(f"{i} {z} "+" ".join(f"{v: .12f}" for v in r)
                                              for i,z,r in zip(idx,atoms.numbers,frac))+"\n")
    out["siesta.STRUCT_OUT"]="siesta-struct-out"
    return out

class StructureFormatTests(unittest.TestCase):
    def test_all_formats_match_cif(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for name,dim in MATERIALS.items():
                ref=cif_atoms(name)
                path=BandPath(("G","X"),np.array([[0,0,0],[.5,0,0]],float),3)
                images=overlap_images_for(dim,4)
                cif_inp=write_periodic_input_generic(BASE/"examples/structures"/f"{name}.cif",
                    tmp/f"{name}_cif.bind",path,1.75,True,8,overlap_images=images,periodic_dim=dim)
                want=numbers(cif_inp.read_text())
                for fname,fmt in write_variants(ref,tmp/name).items():
                    with self.subTest(material=name,file=fname):
                        f=tmp/name/fname
                        self.assertEqual(detect_structure_format(f),fmt)
                        at=read_structure(f)
                        self.assertEqual(at.get_chemical_symbols(),ref.get_chemical_symbols())
                        np.testing.assert_allclose(at.cell.cellpar(),ref.cell.cellpar(),atol=1e-6)
                        np.testing.assert_allclose(at.get_scaled_positions(wrap=True),
                                                   ref.get_scaled_positions(wrap=True),atol=1e-6)
                        inp=write_periodic_input_generic(f,tmp/f"{name}_{fname}.bind",path,1.75,True,8,
                                                         overlap_images=images,periodic_dim=dim)
                        got=numbers(inp.read_text())
                        self.assertEqual(got.shape,want.shape)
                        np.testing.assert_allclose(got,want,atol=1e-6)

    def test_content_detection_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); write_variants(cif_atoms("diamond"),tmp)
            renamed={"POSCAR":"vasp","pw.pwi":"qe","geometry.in":"aims","frac.fdf":"siesta"}
            for src,fmt in renamed.items():
                with self.subTest(file=src):
                    dst=tmp/f"renamed_{src.replace('.','_')}.txt"
                    dst.write_text((tmp/src).read_text())
                    self.assertEqual(detect_structure_format(dst),fmt)
            # XV has no recognizable content: name it oddly and force the format
            odd=tmp/"diamond_relaxed.dat"; odd.write_text((tmp/"siesta.XV").read_text())
            with self.assertRaises(ValueError):read_structure(odd)
            self.assertEqual(len(read_structure(odd,"siesta-xv")),2)
            self.assertEqual(len(read_structure(odd,"xv")),2)
            # a forced, wrong format is reported with the path and the format
            with self.assertRaisesRegex(ValueError,"siesta"):read_structure(tmp/"POSCAR","siesta")

    def test_qe_ibrav_input(self):
        """Hand-written pw.x input with ibrav/celldm and crystal coordinates."""
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"diamond.scf.in"
            a=2.52699457*np.sqrt(2)/BOHR
            f.write_text(f"""&control
  calculation='scf', prefix='c'
/
&system
  ibrav=2, celldm(1)={a:.10f}, nat=2, ntyp=1, ecutwfc=60
/
&electrons
/
ATOMIC_SPECIES
 C 12.011 C.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
 C 0.00 0.00 0.00
 C 0.25 0.25 0.25
K_POINTS automatic
 8 8 8 0 0 0
""")
            self.assertEqual(detect_structure_format(f),"qe")
            at=read_structure(f)
            np.testing.assert_allclose(at.cell.cellpar(),cif_atoms("diamond").cell.cellpar(),atol=1e-5)
            self.assertEqual(at.get_chemical_symbols(),["C","C"])
            d=at.get_distance(0,1,mic=True)
            self.assertAlmostEqual(d,1.5474,places=3)

    def test_qe_bravais_volumes(self):
        """Every pw.x ibrav: cell volume and lengths from the analytic formulas."""
        from structure_io import qe_ibrav_cell
        a=5.0;ba=1.3;ca=1.7;b=a*ba*BOHR;c=a*ca*BOHR;A=a*BOHR
        cg,cb=.2,-.3;ct=.4
        tri=np.sqrt(1+2*ct*cb*cg-ct**2-cb**2-cg**2)
        cases={1:(A**3,None),2:(A**3/4,None),3:(A**3/2,None),-3:(A**3/2,None),
               4:(np.sqrt(3)/2*A*A*c,None),6:(A*A*c,None),7:(A*A*c/2,None),
               8:(A*b*c,(A,b,c)),9:(A*b*c/2,None),-9:(A*b*c/2,None),91:(A*b*c/2,None),
               10:(A*b*c/4,None),11:(A*b*c/2,None),
               12:(A*b*c*np.sqrt(1-cg**2),(A,b,c)),13:(A*b*c*np.sqrt(1-cg**2)/2,None),
               -12:(A*b*c*np.sqrt(1-cb**2),(A,b,c)),-13:(A*b*c*np.sqrt(1-cb**2)/2,None),
               14:(A*b*c*tri,(A,b,c))}
        for ibrav,(vol,lengths) in cases.items():
            with self.subTest(ibrav=ibrav):
                cd=[a,ba,ca,cg,cb,0.]
                if ibrav==14:cd=[a,ba,ca,ct,cb,cg]
                cell=qe_ibrav_cell(ibrav,cd)
                self.assertAlmostEqual(abs(np.linalg.det(cell)),vol,places=8)
                if lengths:np.testing.assert_allclose(np.linalg.norm(cell,axis=1),lengths,atol=1e-10)
        for ibrav in (5,-5):
            with self.subTest(ibrav=ibrav):
                cell=qe_ibrav_cell(ibrav,[a,0,0,ct,0,0])
                np.testing.assert_allclose(np.linalg.norm(cell,axis=1),[A]*3,atol=1e-10)
                cos=[cell[i]@cell[j]/A**2 for i,j in ((0,1),(0,2),(1,2))]
                np.testing.assert_allclose(cos,[ct]*3,atol=1e-10)
        # A/B/C/cosAB input equals celldm input (ibrav=12)
        from structure_io import _qe_celldm
        cd=_qe_celldm({"ibrav":12,"a":3.0,"b":4.0,"c":5.0,"cosab":.1})
        np.testing.assert_allclose(cd,[3/BOHR,4/3,5/3,.1,0,0])

    def test_format_names(self):
        self.assertIsNone(normalize_structure_format("auto"))
        self.assertIsNone(normalize_structure_format(None))
        self.assertEqual(normalize_structure_format("POSCAR"),"vasp")
        self.assertEqual(normalize_structure_format("fhi-aims"),"aims")
        self.assertEqual(normalize_structure_format("espresso-in"),"qe")
        self.assertEqual(normalize_structure_format("fdf"),"siesta")
        for fmt in STRUCTURE_FORMATS:self.assertEqual(normalize_structure_format(fmt),fmt)
        with self.assertRaises(ValueError):normalize_structure_format("castep")

    def test_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            mol=tmp/"geometry.in"; mol.write_text("atom 0 0 0 C\natom 1.2 0 0 C\n")
            with self.assertRaisesRegex(ValueError,"periodic cell"):read_structure(mol)
            ghost=tmp/"ghost.fdf"
            ghost.write_text("LatticeConstant 3.0 Ang\n%block ChemicalSpeciesLabel\n1 -6 C_ghost\n%endblock ChemicalSpeciesLabel\n"
                             "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n%endblock AtomicCoordinatesAndAtomicSpecies\n")
            with self.assertRaisesRegex(ValueError,"ghost"):read_structure(ghost)
            noalat=tmp/"noalat.fdf"
            noalat.write_text("%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\n"
                              "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n%endblock AtomicCoordinatesAndAtomicSpecies\n")
            with self.assertRaisesRegex(ValueError,"LatticeConstant"):read_structure(noalat)
            unknown=tmp/"notes.txt"; unknown.write_text("just some text\n")
            with self.assertRaisesRegex(ValueError,"structure-format"):read_structure(unknown)
            with self.assertRaises(FileNotFoundError):read_structure(tmp/"missing.cif")

    def test_labels(self):
        self.assertEqual(structure_label(Path("/x/MoS2/POSCAR")),"MoS2")
        self.assertEqual(structure_label(Path("/x/GeS/geometry.in")),"GeS")
        self.assertEqual(structure_label(Path("/x/runs/structure.cif")),"runs")
        self.assertEqual(structure_label(Path("/x/runs/MoS2.fdf")),"MoS2")

    def test_fitter_sample_from_other_formats(self):
        carbon={"C":{"atnum":6,"nval":4,
            "s":{"nzeta":1,"n":2,"Hii":-21.4,"zeta1":1.625,"zeta2":0.,"c1":1.,"c2":0.},
            "p":{"nzeta":1,"n":2,"Hii":-11.4,"zeta1":1.625,"zeta2":0.,"c1":1.,"c2":0.}}}
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            ref,_=prepare(BASE/"examples/synthetic/toy_2bands.csv",tmp/"ref",1,-3.5,3.5,mode="csv",nx=151)
            write_variants(cif_atoms("diamond"),tmp/"d")
            odd=tmp/"d/relaxed.dat"; odd.write_text((tmp/"d/siesta.XV").read_text())
            texts={}
            cases=[(BASE/"examples/structures/diamond.cif",None),(tmp/"d/POSCAR",None),
                   (tmp/"d/frac.fdf",None),(tmp/"d/scf.in",None),(odd,"siesta-xv")]
            for f,fmt in cases:
                with self.subTest(file=f.name):
                    entry={"name":"diamond","structure":str(f),"reference":str(ref),"emin":-3.5,"emax":3.5,
                           "point_dim":3,"symmetry_points":[[0,0,0],[.5,0,.5],[.5,.5,.5]],
                           "labels":["G","X","L"],"n_pairs":1,"image_mode":"colored",
                           "points_per_line":4,"trace_nx":151}
                    if fmt:entry["structure_format"]=fmt
                    s=parse_sample(entry,tmp,carbon,{})
                    texts[f.name]=numbers(write_sample_input(s,tmp/f"{f.name}.bind",1.75).read_text())
            want=texts.pop("diamond.cif")
            for name,got in texts.items():
                with self.subTest(file=name):np.testing.assert_allclose(got,want,atol=1e-6)

if __name__=="__main__":unittest.main()
