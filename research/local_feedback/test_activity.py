import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from activity import *
from activity_evaluate import evaluate
from competition import local_metrics,summary
from reanalyze_allowed import corrected


class ActivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=Data();s=PLANS[0]['inner'][0];cls.s=s;cls.cfg=dict(widthId=1,context=True,obligations=True,ridge=.03,wholeWeight=1.,strongWeight=1.,extraWeight=1.)
        cls.k=Kernel(cls.d,cls.d.main[s['train']],cls.d.manual[s['strongTrain']],cls.cfg)

    def test_baseline_refinement_and_three_exact_consumers(self):
        d=self.d;old=parent.Data();np.testing.assert_allclose(summary(corrected(local_metrics(d,d.manual)))['accuracy'],summary(corrected(local_metrics(old,old.manual)))['accuracy'],atol=1e-14)
        from route_consumers import Consumers
        consumer=Consumers()
        for name in d.consumer_names:np.testing.assert_array_equal(d.fixed[name],consumer.fixed[name])

    def test_no_observed_source_in_validation(self):
        d=self.d;s=self.s;k=self.k;held={d.records[j]['group']for j in d.main[s['validation']]}
        self.assertTrue(all(e['group']not in held for e in k.extra_receipt if e['group']is not None))
        self.assertEqual(len(k.frame_ids),len(set(k.frame_ids)))

    def test_individual_terms_and_exact_execution_optimization(self):
        k=self.k;rng=np.random.default_rng(17);theta=rng.normal(0,.05,k.dimension);old=k.cfg.copy()
        try:
            for active in ['wholeWeight','strongWeight','ridge']:
                k.set_config(dict(old,**{key:float(key==active)for key in ['wholeWeight','strongWeight','ridge']}))
                loss,g=k.objective(theta);ll,gg=parent.Kernel.objective(k,theta)
                np.testing.assert_allclose(loss,ll,atol=1e-12);np.testing.assert_allclose(g,gg,atol=1e-12)
                v=rng.normal(size=k.dimension);v/=np.linalg.norm(v);eps=1e-5
                self.assertLess(abs((k.objective(theta+eps*v)[0]-k.objective(theta-eps*v)[0])/(2*eps)-g@v),1e-7)
        finally:k.set_config(old)

    def test_heldout_only_evaluator_matches_full_actual_path(self):
        d=self.d;k=self.k;rng=np.random.default_rng(9);theta=rng.normal(0,.07,k.dimension);s=self.s;main=d.main[s['validation']];manual=d.manual[s['strongValidation']]
        pred,loc=evaluate(k,theta,main,manual);p,support,*_=k.forward(theta,all_rows=True)
        for name in d.consumer_names:np.testing.assert_array_equal(pred[name],d.consume(support,p,name)[main])
        self.assertEqual(loc,corrected(local(d,manual,p)))

    def test_export_and_obligation_ablation(self):
        k=self.k;rng=np.random.default_rng(4);theta=rng.normal(0,.1,k.dimension);model=k.export(theta);coef=np.array(model['coefficients']);p,*_=k.forward(theta)
        logits=np.c_[np.ones(len(k.frame_ids)),k.all_x[k.frame_ids]]@coef.T+k.prior
        np.testing.assert_allclose(parent.softmax(logits,axis=1),p,atol=1e-12)
        for (w,context,obligation),(names,x)in self.d.inputs.items():
            if not obligation:self.assertFalse(any('Obligation'in n or 'independentTail'in n or 'ReleasePair'in n for n in names))


if __name__=='__main__':unittest.main()
