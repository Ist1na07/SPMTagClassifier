import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import Data,Kernel,PLANS,allocate,whiten,frame_readout


class CompetitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=Data();s=PLANS[0]['inner'][0]
        cls.cfg=dict(features='hand',ridge=.03,teacherWeight=1.,wholeWeight=1.,strongWeight=1.)
        cls.k=Kernel(cls.d,cls.d.main[s['train']],cls.d.manual[s['strongTrain']],cls.cfg.copy())

    def test_exact_zero_support_and_primary_inventory(self):
        d=self.d;k=self.k;support,scores,_,_=k.forward(np.zeros(k.dimension),all_rows=True)
        np.testing.assert_allclose(support,d.support,atol=1e-14)
        self.assertEqual([f['tag']for f in frame_readout(d,scores)],[f['tag']for _,f in d.frame_records])

    def test_budget_common_shift_and_zero_candidates(self):
        rng=np.random.default_rng(9)
        for g in self.d.groups:
            prior=g['prior'][:100];budget=g['budget'][:100];delta=rng.normal(0,2,prior.shape)
            a=allocate(prior,delta,budget);b=allocate(prior,delta+7.3,budget)
            np.testing.assert_allclose(a.sum(1),budget,atol=2e-13)
            np.testing.assert_allclose(a,b,atol=2e-13)
            self.assertTrue(np.all(a[np.isneginf(prior)]==0))
            self.assertTrue(np.all((a>=0)&(a<1)))

    def test_whitening_removes_flat_feature_direction(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(50,3));x=np.c_[x,x[:,0]+x[:,1],np.ones(50)]
        z,s=whiten(x,np.ones(50));self.assertEqual(s['retainedRank'],3)
        np.testing.assert_allclose(z[:,1:].T@z[:,1:]/50,np.eye(3),atol=1e-12)

    def test_individual_loss_pullbacks(self):
        k=self.k;rng=np.random.default_rng(11);t=rng.normal(0,.06,k.dimension);eps=2e-5
        original=k.cfg.copy()
        try:
            for active in ['wholeWeight','strongWeight','teacherWeight','ridge']:
                k.cfg={**original,**{c:float(c==active)for c in ['wholeWeight','strongWeight','teacherWeight','ridge']}}
                _,g=k.objective(t)
                for _ in range(3):
                    direction=rng.normal(size=k.dimension);direction/=np.linalg.norm(direction)
                    fd=(k.objective(t+eps*direction)[0]-k.objective(t-eps*direction)[0])/(2*eps)
                    self.assertLess(abs(fd-g@direction),1e-7,active)
        finally:k.cfg=original

    def test_train_only_frame_inventory_and_identifiable_contrasts(self):
        d=self.d;k=self.k;allowed={j for i in list(k.train)+list(k.manual)for j in range(*d.records[i]['offset'])}
        for g in k.groups:
            self.assertTrue(set(g['frame_ids']).issubset(allowed))
            np.testing.assert_allclose(g['contrast'].T@g['contrast'],np.eye(len(g['classes'])-1),atol=1e-14)
            np.testing.assert_allclose(g['contrast'].sum(0),0,atol=1e-14)


if __name__=='__main__':unittest.main()
