import unittest,copy
import ln_adapter as a
from family import classify,load_metadata,model_scores


class LnAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main,manual,_=a.load();cls.main=main;cls.manual=manual;known=[r for r in main if set(a.target(r))&a.KNOWN]
        cls.weak=[r for r in known if 'release'in a.target(r)][:2]+[r for r in known if 'release'not in a.target(r)][:2]
        cls.config={'localL2':.1,'strongWeight':1.,'balance':'balanced'}

    def test_exact_weak_impression_and_local_partial_label_objective_gradients(self):
        d=a.Packed(self.weak,self.manual[:1],'geometry3',self.config);a.torch.manual_seed(713)
        t=a.torch.randn(len(d.keys)+3)*.03;t[-2]+=a.SLOPE_INITIAL;t.requires_grad_(True)
        d.objective(t,self.config).backward();g=t.grad.detach().numpy().copy();eps=1e-6
        for j in [0,3,4,5]:
            x=t.detach().clone();x[j]+=eps;y=t.detach().clone();y[j]-=eps
            numerical=(float(d.objective(x,self.config))-float(d.objective(y,self.config)))/(2*eps)
            self.assertAlmostEqual(g[j],numerical,places=6)
        self.assertAlmostEqual(float(d.weak_weight[d.y==1].sum()),.5)
        self.assertAlmostEqual(float(d.weak_weight[d.y==0].sum()),.5)

    def test_zero_update_exactly_preserves_the_actual_archived_conditional_head_and_consumer(self):
        records=self.weak[:1]+self.manual[:1];metadata=load_metadata()
        model={'variant':'geometry3','keys':['a','b','ab'],'mean':[0,0,0],'scale':[1,1,1],'delta':[0,0,0,0],'clip':6}
        predictions=a.predict(model,records)
        for r,p in zip(records,predictions):
            x=a.np.array([s['original']for s in r['segments']]);expected=model_scores(a.ARCHIVE['ln'],x)
            self.assertTrue(a.np.array_equal(p,expected))
            rc=model_scores(a.ARCHIVE['rc'],x)
            for ratio in [.35,.45,.55]:
                for scorer in ['model','blend']:
                    baseline=classify(r,metadata[r['uid']],a.ARCHIVE,ln_ratio=ratio,scorer=scorer)
                    actual=classify(r,metadata[r['uid']],a.ARCHIVE,ln_ratio=ratio,scorer=scorer,precomputed={'rc':rc,'ln':p})
                    self.assertEqual(actual,baseline)

    def test_release_offset_leaves_other_ln_log_odds_unchanged(self):
        d=a.Packed(self.weak,self.manual[:1],'geometry3',self.config);t=a.torch.zeros(len(d.keys)+3);t[0]=.2;t[-2]=a.SLOPE_INITIAL
        logp=a.torch.log_softmax(d.logits+a.torch.nn.functional.one_hot(a.torch.tensor(a.RELEASE),len(a.CLASSES))*(d.x@t[:-2])[:,None],1)
        prior=a.torch.log_softmax(d.logits,1)
        others=[i for i in range(len(a.CLASSES))if i!=a.RELEASE]
        for i in others[1:]:self.assertLess(float((logp[:,i]-logp[:,others[0]]-prior[:,i]+prior[:,others[0]]).abs().max()),1e-10)


if __name__=='__main__':unittest.main()
