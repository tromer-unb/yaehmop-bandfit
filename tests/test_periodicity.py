"""The fitter must write the same lattice periodicity as the recomputation.

Regression test for fitted 3D crystals being written as a 2D slab
(Lattice 2, images 5 5 0) whatever the [band_path] dimension.
No YAeHMOP, DFT, fitting or network.
"""
from pathlib import Path
import sys,tempfile,unittest
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"USER_WORKFLOW"),str(BASE/"parameter_fitter"),str(BASE/"source")]
from prepare_band_png import prepare
from yaehmop_generic import DEFAULT_OVERLAP_RADIUS,overlap_images_for
from fit_sparse_window import parse_sample,write_sample_input

# Minimal carbon basis; the real atomic table is not bundled with the repo.
CARBON={"C":{"atnum":6,"nval":4,
    "s":{"nzeta":1,"n":2,"Hii":-21.4,"zeta1":1.625,"zeta2":0.,"c1":1.,"c2":0.},
    "p":{"nzeta":1,"n":2,"Hii":-11.4,"zeta1":1.625,"zeta2":0.,"c1":1.,"c2":0.}}}
CASES={"graphene":(2,[[0,0],[.5,0],[1/3,1/3],[0,0]],["G","M","K","G"]),
       "diamond":(3,[[0,0,0],[.5,0,.5],[.5,.5,.5],[0,0,0]],["G","X","L","G"]),
       "graphite_AB":(3,[[0,0,0],[.5,0,0],[0,0,.5]],["G","M","A"])}

def lattice_block(text):
    lines=text.splitlines()
    i=lines.index("Lattice")
    return int(lines[i+1]),tuple(int(v) for v in lines[i+2].split())

class PeriodicityTests(unittest.TestCase):
    def test_overlap_images_helper(self):
        self.assertEqual(overlap_images_for(2),(DEFAULT_OVERLAP_RADIUS,)*2+(0,))
        self.assertEqual(overlap_images_for(3,4),(4,4,4))
        self.assertEqual(overlap_images_for(1,3),(3,0,0))
        for bad in ((0,5),(4,5),(3,0),(3,9)):
            with self.assertRaises(ValueError):overlap_images_for(*bad)

    def test_fitter_input_uses_band_path_dimension(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            ref,_=prepare(BASE/"examples/synthetic/toy_2bands.csv",d/"ref",1,
                          -3.5,3.5,mode="csv",nx=151)
            for name,(dim,nodes,labels) in CASES.items():
                for radius in (None,3):
                    with self.subTest(material=name,radius=radius):
                        entry={"name":name,"structure":str(BASE/"examples/structures"/f"{name}.cif"),
                               "reference":str(ref),"emin":-3.5,"emax":3.5,"point_dim":dim,
                               "symmetry_points":nodes,"labels":labels,"n_pairs":1,
                               "image_mode":"colored","points_per_line":4,"trace_nx":151}
                        shared={} if radius is None else {"overlap_radius":radius}
                        s=parse_sample(entry,d,CARBON,shared)
                        want=overlap_images_for(dim,radius or DEFAULT_OVERLAP_RADIUS)
                        self.assertEqual(s["periodic_dim"],dim)
                        self.assertEqual(s["overlap_images"],want)
                        inp=write_sample_input(s,d/f"{name}.bind",1.75)
                        self.assertEqual(lattice_block(inp.read_text()),(dim,want))

if __name__=="__main__":unittest.main()
