"""CI-safe tests for frozen transfer and conservative task advice.

No YAeHMOP binary, DFT or optimizer is called.
"""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'USER_WORKFLOW'))
import frozen_transfer as tr
import strategy_advisor as advisor

class FrozenPublicTests(unittest.TestCase):
 def test_advisor_direct_semiconductor_2_and_4_bands(self):
  x=advisor.recommendation(goal='direct',electronic='semiconductor',
        bands='auto',phase='1H',elements=('Mo','S'),has_reference=True)
  self.assertEqual(x['pairs'],1)
  self.assertTrue(x['tested_workflow'])
  self.assertIn('--pairs 1',x['commands'][1])
  y=advisor.recommendation(goal='direct',electronic='semiconductor',
        bands='4',elements=('Mo','S'),has_reference=True)
  self.assertEqual(y['pairs'],2)
  self.assertIn('--pairs 2',y['commands'][1])
 def test_advisor_metal_never_claims_png_fermi(self):
  x=advisor.recommendation(goal='direct',electronic='metal',
        bands='2',phase='1T',elements=('W','S'),has_reference=True)
  self.assertFalse(x['tested_workflow'])
  self.assertNotIn('commands',x)
  self.assertIn('SPECIALIZED',x['recommendation'])
 def test_advisor_transfer_is_frozen_not_shared_fit(self):
  x=advisor.recommendation(goal='transfer',electronic='semiconductor',
       bands='4',elements=('Mo','S'),has_reference=True)
  self.assertIn('FROZEN_TRANSFER',x['recommendation'])
  self.assertIn('--source-run',x['commands'][0])
  self.assertIn('does not train',x['why'].lower())
  self.assertIn('not validated',x['shared_fit_note'])
 def test_advisor_invalid_goal_rejected(self):
  with self.assertRaises(ValueError):
   advisor.recommendation(goal='nonsense',electronic='semiconductor')
 def test_source_summary_reads_K_without_target_retraining(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);native=p/'fit'/'param_best_native.dat';native.parent.mkdir()
   native.write_text('dummy source table')
   summary=p/'run_summary.json'
   summary.write_text(json.dumps({'native_parameters':str(native),
     'K_EHT':1.654,'cif_sha256':'source'}))
   out,K,row=tr.source_from_summary(summary)
   self.assertEqual(out,native.resolve())
   self.assertEqual(K,1.654)
   self.assertEqual(row['cif_sha256'],'source')
 def test_missing_k_or_table_fails(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'run_summary.json'
   p.write_text(json.dumps({'native_parameters':'missing.dat'}))
   with self.assertRaises(ValueError):tr.source_from_summary(p)
 def test_same_CIF_source_target_is_blocked_before_solver(self):
  cif=ROOT/'examples/structures/graphene.cif'
  ini=ROOT/'examples/configs/synthetic_2bands.ini'
  csvfile=ROOT/'examples/synthetic/toy_2bands.csv'
  with tempfile.TemporaryDirectory() as tmp:
   tab=Path(tmp)/'table.dat';tab.write_text('not used')
   with patch.object(tr,'recompute',side_effect=AssertionError('solver should never run')):
    with self.assertRaisesRegex(ValueError,'Target CIF equals source training CIF'):
     tr.evaluate(cif,ini,csvfile,tab,1.75,Path(tmp)/'out',1,
        mode='csv',source={'cif_sha256':tr.sha(cif)})
 def test_no_optimizer_ever_in_frozen_module(self):
  s=(ROOT/'USER_WORKFLOW/frozen_transfer.py').read_text()
  self.assertNotIn('scipy.optimize',s)
  self.assertNotIn('subprocess.run',s)
  self.assertIn("'refit_optimizer_calls':0",s)

if __name__=='__main__':unittest.main()