import unittest
import numpy as np
import torch
from mil import *
class TestMIL(unittest.TestCase):
 def test_group_integrity_and_coverage(self):
  main,manual,data=load();folds=grouped_folds(main)
  self.assertEqual(sorted(i for f in folds for i in f),list(range(len(main))))
  groups=[{main[i]['group'] for i in f} for f in folds]
  for a,b in itertools.combinations(groups,2):self.assertFalse(a&b)
  self.assertFalse({r['group'] for r in main}&{r['group'] for r in manual})
 def test_gradient_reaches_both_heads_and_matches_finite_difference(self):
  main,manual,_=load();d=Packed(main[:4]+manual[:2],'prior_residual');m=Model(d.x.shape[1],True)
  torch.manual_seed(3)
  with torch.no_grad():
   for p in m.parameters():p.add_(torch.randn_like(p)*.03)
  cfg={'localL2':.01,'mapL2':.0001,'strongWeight':1.,'effortPower':1.,'confidencePower':1.}
  loss=m.objective(d,cfg);loss.backward()
  for p,ix in [(m.local.weight,(2,6)),(m.local.bias,(3,)),(m.mix.weight,(2,8)),(m.fine_bias,(5,)),(m.fine_slope,(3,))]:
   analytical=float(p.grad[ix]);eps=1e-6
   with torch.no_grad():p[ix]+=eps
   high=float(m.objective(d,cfg).detach())
   with torch.no_grad():p[ix]-=2*eps
   low=float(m.objective(d,cfg).detach())
   with torch.no_grad():p[ix]+=eps
   self.assertAlmostEqual(analytical,(high-low)/(2*eps),places=6)
  p,z,bag=m(d,cfg)
  self.assertTrue(torch.allclose(p.sum(1),torch.ones(len(p))))
  self.assertTrue(torch.allclose(bag.sum(1),torch.ones(len(bag))))
 def test_frozen_preprocessing_and_shared_segment_predictions(self):
  main,manual,_=load();d=Packed(main[:4]+manual[:2],'prior_residual');m=Model(d.x.shape[1],True)
  held=Packed(main[4:6],d.variant,d.mean,d.scale)
  np.testing.assert_array_equal(d.mean,held.mean)
  cfg={'effortPower':0.,'confidencePower':0.}
  p,_,bag=m(held,cfg)
  offset=0
  for i,r in enumerate(held.rows):
   n=len(r['segments']);dur=np.array([max(1,s['end']-s['start']) for s in r['segments']])
   expected=np.average(p.detach().numpy()[offset:offset+n],axis=0,weights=dur)
   np.testing.assert_allclose(expected,bag.detach().numpy()[i],rtol=1e-12);offset+=n
if __name__=='__main__':unittest.main()
