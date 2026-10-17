"""Multi-material selected-window manifest validation (no solver/optimizer)."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from ase.io import read,write
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'USER_WORKFLOW'))
import family_fit as fm
import frozen_transfer as tr

class FamilyInterface(unittest.TestCase):
 def test_staged_budget_guard(self):
  self.assertEqual(fm.minimum_staged_budget(21),86)
  self.assertEqual(fm.minimum_staged_budget(2),80)
  self.assertEqual(fm.minimum_staged_budget(55),222)

 def make_case(self,path):
  atoms=read(ROOT/'examples/structures/graphene.cif')
  for i,name in enumerate(('alpha','beta','gamma')):
   a=atoms.copy();cell=a.cell.array.copy();cell[0]*=1+.01*i
   a.set_cell(cell,scale_atoms=True)
   write(path/f'{name}.cif',a)
  ref=ROOT/'examples/synthetic/toy_2bands.csv'
  ini=ROOT/'examples/configs/synthetic_2bands.ini'
  m={'n_pairs':1,'systems':[
     {'name':name,'role':'train' if name!='gamma' else 'test',
      'structure':str(path/f'{name}.cif'),'settings':str(ini),
      'reference':str(ref),'mode':'csv'} for name in ('alpha','beta','gamma')]}
  yml=path/'family.yaml';yml.write_text(yaml.safe_dump(m))
  return yml
 def test_family_validate_train_test_and_no_heldout_opening(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);yml=self.make_case(path)
   _,a,train,test=fm.load_manifest(yml)
   self.assertEqual(len(train),2)
   self.assertEqual(len(test),1)
   self.assertEqual(a['n_pairs'],1)
   self.assertEqual({s['name'] for s in train},{'alpha','beta'})
 def test_family_duplicate_CIF_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);yml=self.make_case(path)
   m=yaml.safe_load(yml.read_text())
   m['systems'][2]['structure']=m['systems'][0]['structure']
   yml.write_text(yaml.safe_dump(m))
   with self.assertRaisesRegex(ValueError,'identical to a training CIF'):
    fm.load_manifest(yml)
 def test_family_prepare_train_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);yml=self.make_case(path)
   _,m,train,test=fm.load_manifest(yml)
   data=[fm.preparation(x,path/'output',1) for x in train]
   self.assertEqual(len(data),2)
   self.assertTrue(all(s['role']=='train' for s in data))
   self.assertTrue(all(Path(s['reference']).is_file() for s in data))
   self.assertFalse((path/'output/prepared/gamma').exists())
 def test_frozen_multisource_hash_leak_prevented(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);yml=self.make_case(path)
   _,m,train,test=fm.load_manifest(yml)
   tab=path/'table.dat';tab.write_text('not used')
   with patch.object(tr,'recompute',side_effect=AssertionError('solver called')):
    with self.assertRaisesRegex(ValueError,'Target CIF equals source training CIF'):
     tr.evaluate(train[0]['cif'],train[0]['ini'],train[0]['image'],
         tab,1.75,path/'transfer',1,mode='csv',
         source={'train_cif_sha256s':[tr.sha(train[0]['cif'])]})

if __name__=='__main__':unittest.main()