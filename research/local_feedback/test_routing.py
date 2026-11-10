import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from routing import Data,Kernel,PLANS,ROUTES


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=Data();s=PLANS[0]['inner'][0];cls.cfg=dict(features='pressure',priorStrength=4.,ridge=.03,teacherWeight=1.,wholeWeight=1.,strongWeight=1.)
        cls.k=Kernel(cls.d,cls.d.main[s['train']],cls.d.manual[s['strongTrain']],cls.cfg.copy())

    def test_individual_objective_pullbacks(self):
        k=self.k;rng=np.random.default_rng(15);t=rng.normal(0,.03,k.dimension);original=k.cfg
        try:
            for active in ['wholeWeight','strongWeight','teacherWeight','ridge']:
                k.cfg={**original,**{c:float(c==active)for c in ['wholeWeight','strongWeight','teacherWeight','ridge']}}
                _,g=k.objective(t)
                for _ in range(3):
                    direction=rng.normal(size=k.dimension);direction/=np.linalg.norm(direction);eps=1e-5
                    fd=(k.objective(t+eps*direction)[0]-k.objective(t-eps*direction)[0])/(2*eps)
                    self.assertLess(abs(fd-g@direction),1e-7,active)
        finally:k.cfg=original

    def test_baseline_primary_and_counterfactual_family_inventory(self):
        d=self.d;k=self.k;p,support,share,family,_=k.forward(np.zeros(k.dimension),all_rows=True)
        self.assertEqual(p.argmax(1).tolist(),d.route.tolist())
        self.assertEqual([r['tag']for r in d.routed_scores(p)],[f['tag']for _,f in d.frame_records])
        np.testing.assert_allclose(p.sum(1),1,atol=1e-14)
        self.assertTrue(np.all((support>=0)&(support<1)))
        self.assertTrue(np.all(family.sum(1)<=1+1e-12))
        self.assertTrue(np.all(d.expert[d.route==0].sum((1,2))>0))

    def test_chart_relative_context_is_duration_centered(self):
        d=self.d
        for r in d.records:
            a,b=r['offset'];np.testing.assert_allclose(np.average(d.context[a:b],axis=0,weights=d.dt[a:b]),0,atol=1e-12)

    def test_exported_coefficients_reproduce_training_frame_logits(self):
        d=self.d;k=self.k;rng=np.random.default_rng(3);theta=rng.normal(0,.1,k.dimension);m=k.export(theta);p,*_=k.forward(theta)
        coef=np.asarray(m['coefficients']);x=k.all_x[k.frame_ids];z=np.c_[np.ones(len(x)),x]@coef.T+k.prior;z-=z.max(1,keepdims=True);expected=np.exp(z);expected/=expected.sum(1,keepdims=True)
        np.testing.assert_allclose(p,expected,atol=1e-12)


if __name__=='__main__':unittest.main()
