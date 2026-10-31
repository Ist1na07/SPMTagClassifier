import unittest
from family_train import *
from family import classify,model_scores

class TestFamily(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main,cls.manual,_=load();cls.metadata=load_metadata()
        cls.cfg={'localL2':.0005,'strongWeight':1.,'balance':'cap6'}

    def test_copy_and_bag_have_identical_inputs_and_strong_supervision(self):
        weak=[r for r in self.main if target(r)==['inverse']][:4];manual=self.manual[:2]
        a=HeadData(weak,manual,self.metadata,'ln','original','copy_partial',self.cfg)
        b=HeadData(weak,manual,self.metadata,'ln','original','bag_partial',self.cfg)
        self.assertTrue(torch.equal(a.x,b.x));self.assertTrue(torch.equal(a.mask,b.mask));self.assertTrue(torch.equal(a.strong_w,b.strong_w))
        torch.manual_seed(31);W=torch.randn(len(a.classes),a.x.shape[1])*.05;bias=torch.randn(len(a.classes))*.1
        # Jensen: a bag label does not impose the same loss as copying it onto
        # all its rows; the new likelihood cannot be greater at the same model.
        self.assertLessEqual(float(b.objective(W,bias,self.cfg)),float(a.objective(W,bias,self.cfg))+1e-12)

    def test_bag_gradient_matches_the_scalar_objective(self):
        weak=[r for r in self.main if target(r)==['inverse']][:4]
        d=HeadData(weak,self.manual[:2],self.metadata,'ln','original','bag_partial',self.cfg)
        W=torch.nn.Parameter(torch.randn(len(d.classes),d.x.shape[1])*.03);b=torch.nn.Parameter(torch.zeros(len(d.classes)))
        d.objective(W,b,self.cfg).backward();ix=(2,5);analytic=float(W.grad[ix]);eps=1e-6
        with torch.no_grad():W[ix]+=eps
        high=float(d.objective(W,b,self.cfg).detach())
        with torch.no_grad():W[ix]-=2*eps
        low=float(d.objective(W,b,self.cfg).detach())
        self.assertAlmostEqual(analytic,(high-low)/(2*eps),places=6)

    def test_frozen_head_precomputation_preserves_the_actual_hierarchy(self):
        models=json.loads(Path('.cache/archived-model-reference.json').read_text(encoding='utf8'));r=self.manual[0]
        x=np.array([s['original']for s in r['segments']]);p={f:model_scores(models[f],x)for f in ['rc','ln']}
        direct=classify(r,self.metadata[r['uid']],models)
        cached=classify(r,self.metadata[r['uid']],models,precomputed=p)
        self.assertEqual(direct,cached)
        rules=classify(r,self.metadata[r['uid']],None,scorer='rules')
        self.assertEqual(len(rules),len(r['segments']));self.assertTrue(all('probability'in s for s in rules))

    def test_joint_seed_candidates_consume_the_actual_single_case_wrapper(self):
        import family_search as search
        search._META=self.metadata
        models=json.loads(Path('.cache/archived-model-reference.json').read_text(encoding='utf8'))
        for name in ['count_primary_original','rules_original']:
            spec=search.VARIANTS[name];configs=search.joint_candidates(spec)
            self.assertEqual(len(configs),8192)
            self.assertTrue(all(c['partition']==c['head']==0 and c['burdenGate']==0 for c in configs))
            values=search.predict_cases(self.manual[:1],models if spec['recipe']!='rules'else None,spec,
                [.35,.45,.55],['rules']if spec['recipe']=='rules'else['model','blend'])
            # Exercise the seed (formerly native partition4/head2) and every
            # distinct actual routing/display path against the real consumer.
            for ratio,scorer,display in itertools.product([.35,.45,.55],['rules']if spec['recipe']=='rules'else['model','blend'],['legacy','coherent']):
                c=next(c for c in configs if c['lnRatio']==ratio and c['scorer']==scorer and c['smoothing']==display)
                data=search.pack(values[(ratio,scorer)],c)
                p=search.decisions(search.summaries(data,c),c)
                self.assertEqual(p.shape,(1,len(MAP_CLASSES)))
                self.assertTrue(np.isfinite(data['p']).all())

if __name__=='__main__':unittest.main()
