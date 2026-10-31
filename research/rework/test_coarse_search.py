import unittest,tempfile
from unittest.mock import patch
import coarse_search as s

class TestCoarseSearch(unittest.TestCase):
    def test_completed_equivalent_feature_spec_is_reused_not_refitted(self):
        with tempfile.TemporaryDirectory(prefix='spm-equivalent-fit-')as tmp:
            root=s.Path(tmp);source=root/'sequence44'/'0.json.gz';target=root/'no_release'/'0.json.gz'
            spec=s.model.SPECS['sequence44'];cfg={'windowMs':2000};fitted={'fixture':True};trace={'status':'small_projected_gradient'}
            s.write(source,{'spec':spec,'config':cfg,'trainUids':[],'model':fitted,'fit':trace})
            with patch.object(s.model,'fit',side_effect=AssertionError('Equivalent expert was fitted again')):
                m,t=s.fit_or_load(target,[],spec,cfg)
            self.assertEqual(m,fitted);self.assertEqual(t,trace);self.assertEqual(s.read(target)['equivalentFitSource'],str(source))

    def test_full_declared_joint_grid_and_source_controls_have_no_duplicate_fits(self):
        for name in s.model.SPECS:
            configs=s.configurations(name)
            expected=18 if name=='weak4_only'else 36 if name in ['weak4_manual7','weak7_manual7']else 108
            self.assertEqual(len(configs),expected)
            self.assertEqual(len({s.json.dumps(c,sort_keys=True)for c in configs}),expected)

    def test_four_key_baseline_selection_does_not_use_seven_key_tie_breaks(self):
        weak=[{'uid':str(K)+tag,'keys':K,'coarseTags':[tag],'recordWeight':1.}for K in[4,7]for tag in s.model.FINE]
        split={'inner':[{'validation':list(range(10)),'strongValidation':[0]}]}
        with tempfile.TemporaryDirectory(prefix='spm-coarse-test-')as tmp:
            root=s.Path(tmp);directory=root/'weak4_only'/'0'/'0';directory.mkdir(parents=True)
            s.write(directory/'complete.json',{'mapUids':[r['uid']for r in weak]})
            # Both heads have the same perfect 4K predictions; only candidate1
            # is good on 7K/manual labels. Pure 4K must retain candidate0.
            p=s.np.array([list(range(5))+[0]*5,list(range(5))*2],dtype=s.np.uint8)
            s.np.savez(directory/'inner-predictions.npz',map=p,segmentAccuracy=s.np.array([[0.],[1.]]))
            with patch.object(s,'configurations',return_value=[{},{}]):
                four,_=s.select_outer(root,'weak4_only',0,weak,[],split,'4k')
                shared,_=s.select_outer(root,'weak4_only',0,weak,[],split,'shared')
            self.assertEqual(four['configId'],0);self.assertEqual(shared['configId'],1)

    def test_empty_evidence_is_an_error_not_an_invented_jack(self):
        rows=[{'keys':4,'coarseTags':[t],'recordWeight':1.}for t in s.model.FINE]
        self.assertEqual(s.source_metric(rows,[5]*5,4)['macroF1'],0.)

    def test_fixed_pool_ablation_optimizes_and_predicts_without_target_labels(self):
        weak,manual,_=s.load_headers()
        headers=[next(r for r in weak if r['keys']==K and r['coarseTags']==[t])for K in[4,7]for t in s.model.FINE]+manual[:1]
        rows=s.model.materialize(headers)
        spec={**s.model.SPECS['full53'],'effort':False,'confidence':False}
        cfg={'windowMs':4000,'burdenGate':0.,'localL2':.003,'strongWeight':.5,'source4Weight':.5}
        fitted,trace=s.model.fit(rows,spec,cfg)
        self.assertEqual(fitted['pool'],{'effortPower':0.,'confidencePower':0.})
        self.assertNotEqual(trace['status'],'budget_exhausted')
        test={**rows[0],'keys':6,'dataset':'unlabelled6'};test.pop('coarseTags',None)
        prediction=s.model.predict(fitted,[test])
        self.assertEqual(prediction['composition'].shape,(1,5));self.assertAlmostEqual(float(prediction['composition'].sum()),1.)
        self.assertTrue(s.np.allclose(prediction['local'][0].sum(1),1.))

if __name__=='__main__':unittest.main()
