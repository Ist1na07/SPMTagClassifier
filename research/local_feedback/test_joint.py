import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from joint_activity import *
from competition import allocate as reference
from joint_study import configurations


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=Data();cls.s=PLANS[0]['inner'][0];cls.cfg=dict(widthId=1,context=True,obligations=True,ridge=.3,strongWeight=1.,extraWeight=4.,wholeWeight=1.,trainRoute=True,trainType=True);cls.k=Kernel(cls.d,cls.d.main[cls.s['train']],cls.d.manual[cls.s['strongTrain']],cls.cfg)

    def test_actual_zero_type_embedding_and_budget(self):
        k=self.k;r,u,q,s,z,f=k.forward(np.zeros(k.dimension))
        for v,t in zip(q,k.type):
            np.testing.assert_allclose(v,t['q0'][f],atol=2e-13);np.testing.assert_allclose(v.sum(1),t['budget'][f],atol=2e-13);self.assertTrue(np.all((v>=0)&(v<=1)))

    def test_compiled_saturated_projection_matches_independent_reference(self):
        rng=np.random.default_rng(44)
        for scale in [.001,.1,1,10,100,500]:
            q=rng.uniform(.001,.999,(128,7));p=np.log(q)-np.log1p(-q);delta=rng.normal(0,scale,q.shape);b=q.sum(1)
            np.testing.assert_allclose(allocate(p,delta,b),reference(p,delta,b),atol=1e-11)

    def test_terms_and_exact_local_compaction_in_all_architectures(self):
        rng=np.random.default_rng(45);d=self.d;s=self.s
        for route,typ in [(True,True),(False,True),(True,False)]:
            cfg=dict(self.cfg,trainRoute=route,trainType=typ);k=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],cfg);theta=rng.normal(0,.02,k.dimension)
            for term in ['strongWeight','wholeWeight','ridge']:
                c=dict(cfg,**{key:float(key==term)for key in ['strongWeight','wholeWeight','ridge']});k.set_config(c);loss,g=k.objective(theta);v=rng.normal(size=k.dimension);v/=np.linalg.norm(v);eps=1e-5;fd=(k.objective(theta+eps*v)[0]-k.objective(theta-eps*v)[0])/(2*eps);self.assertLess(abs(fd-g@v),1e-7,(route,typ,term))
                if c['wholeWeight']==0:
                    other,gg=k.objective(theta,compact=False);np.testing.assert_allclose(loss,other,atol=1e-12);np.testing.assert_allclose(g,gg,atol=1e-12)

    def test_reused_extra_weight_and_fresh_kernel_match(self):
        d=self.d;s=self.s;k=self.k;rng=np.random.default_rng(46);theta=rng.normal(0,.02,k.dimension)
        for weight in [0.,1.,4.]:
            c=dict(self.cfg,extraWeight=weight);k.set_config(c);fresh=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],c);a,g=k.objective(theta);b,h=fresh.objective(theta);np.testing.assert_allclose(a,b,atol=1e-12);np.testing.assert_allclose(g,h,atol=1e-12)
        k.set_config(self.cfg)

    def test_real_observations_respect_source_holdout_and_bag_scope(self):
        k=self.k;d=self.d;s=self.s;held={d.records[i]['group']for i in d.main[s['validation']]}
        self.assertTrue(all(e['group']not in held for e in k.base.extra_receipt if e['group']is not None));self.assertTrue(all(abs(e['weight'].sum()-1)<1e-12 for e in k.extra));self.assertTrue(all(len(e['at'])>1 for e in k.extra));self.assertEqual(len(k.frame_ids),len(set(k.frame_ids)))
        self.assertEqual(len(configurations()),102)

    def test_physical_eligibility_even_against_extreme_family_bias(self):
        d=self.d;s=self.s;c=dict(self.cfg,physicalEligibility=True);k=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],c);theta=np.zeros(k.dimension);block=k.blocks[0];coef=theta[:block['size']].reshape(block['shape']);coef[0]=block['contrast'].T@np.array([0.,0.,100.]);r,_,_,support,_,f=k.forward(theta,all_rows=True)
        np.testing.assert_array_equal(r[~k.eligible[:,2],2],0);np.testing.assert_array_equal(r[~k.eligible[:,1],1],0);np.testing.assert_allclose(r.sum(1),1,atol=1e-14)
        ln_columns=[d.fine.index(c)for c in LN if c in d.fine];np.testing.assert_array_equal(support[~k.eligible[:,2]][:,ln_columns],0)
        rng=np.random.default_rng(47);theta=rng.normal(0,.02,k.dimension);v=rng.normal(size=k.dimension);v/=np.linalg.norm(v);eps=1e-5;_,g=k.objective(theta);fd=(k.objective(theta+eps*v)[0]-k.objective(theta-eps*v)[0])/(2*eps);self.assertLess(abs(fd-g@v),1e-7)


if __name__=='__main__':unittest.main()
