import unittest,copy
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
 def all_break_example(self):
  main,_,_=load();r=copy.deepcopy(main[0]);r['tags']=['chordjack'];r['segments']=r['segments'][:1]
  p=np.full(len(CLASSES),.001/(len(CLASSES)-2));p[CLASSES.index('chordjack')]=.009;p[-1]=.99
  r['segments'][0]['prior']=p.tolist();d=Packed([r],'prior_residual');m=Model(d.x.shape[1],True)
  return r,d,m
 def test_all_break_chart_cannot_receive_a_confident_fine_tag(self):
  r,d,m=self.all_break_example();cfg={'effortPower':0.,'confidencePower':0.}
  prob,local,_=predict(m,d,[r],cfg)
  self.assertTrue(np.all(local[0].argmax(1)==CLASSES.index('break')))
  self.assertEqual(int(predict_tags(prob,.2).sum()),0)
 def test_nonempty_chart_label_supervises_active_evidence_not_only_conditional_composition(self):
  r,d,m=self.all_break_example();cfg={'localL2':.01,'mapL2':.0001,'strongWeight':1.,'effortPower':0.,'confidencePower':0.}
  m.objective(d,cfg).backward()
  self.assertGreater(float(m.local.bias.grad[CLASSES.index('break')]),.1)
 def test_control_local_objective_has_the_derivative_used_by_line_search(self):
  main,manual,_=load();cfg={'localL2':.03,'mapL2':.0003,'strongWeight':.5,'effortPower':1.,'confidencePower':1.}
  for variant in ['strong_only_prior','copied_prior']:
   d=Packed(main[:5]+manual[:2],variant);m=Model(d.x.shape[1],True)
   loss=m.local_control_objective(d,cfg);loss.backward();parameter=m.local.weight;ix=(2,6)
   analytic=float(parameter.grad[ix]);eps=1e-6
   with torch.no_grad():parameter[ix]+=eps
   high=float(m.local_control_objective(d,cfg).detach())
   with torch.no_grad():parameter[ix]-=2*eps
   low=float(m.local_control_objective(d,cfg).detach())
   self.assertAlmostEqual(analytic,(high-low)/(2*eps),places=6)
   for p in m.local.parameters():p.requires_grad_(False)
   m.zero_grad();m.objective(d,cfg).backward();parameter=m.mix.weight;ix=(2,8)
   analytic=float(parameter.grad[ix])
   with torch.no_grad():parameter[ix]+=eps
   high=float(m.objective(d,cfg).detach())
   with torch.no_grad():parameter[ix]-=2*eps
   low=float(m.objective(d,cfg).detach())
   self.assertAlmostEqual(analytic,(high-low)/(2*eps),places=6)
 def test_refinement_stops_on_training_gradient_without_using_validation(self):
  from convergence import polish,POLICY
  main,manual,_=load();d=Packed(main[:3]+manual[:1],'frozen_prior');m=Model(d.x.shape[1],True)
  for p in m.local.parameters():p.requires_grad_(False)
  cfg={'localL2':.03,'mapL2':.0003,'strongWeight':.5,'effortPower':0.,'confidencePower':0.}
  before={k:v.clone()for k,v in m.state_dict().items()}
  result=polish(m,d,cfg,policy={**POLICY,'gradientMax':1e10})
  self.assertEqual(result['status'],'small_gradient');self.assertFalse(result['chunks'])
  for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
if __name__=='__main__':unittest.main()
