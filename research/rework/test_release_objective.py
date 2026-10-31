import unittest,importlib.util,tempfile,json
from pathlib import Path
import numpy as np
from scipy.special import expit

spec=importlib.util.spec_from_file_location('release_study',Path(__file__).with_name('release-study.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class BinaryObjectiveTest(unittest.TestCase):
    def test_gradient_and_hessian_match_the_identical_weighted_logistic_loss(self):
        rng=np.random.default_rng(713);z=rng.normal(size=(9,4));y=np.arange(9)%2;w=rng.random(9)+.1;t=rng.normal(size=5)*.2;C=100000.
        loss,g,h=s.binary_objective(t,z,y,w,C);eps=1e-5
        p=expit(z@t[:-1]+t[-1]);res=w*(p-y)
        expected=np.r_[z.T@res+t[:-1]/C,res.sum()]/w.sum()
        np.testing.assert_allclose(g,expected,atol=1e-14)
        for j in range(len(t)):
            a=t.copy();b=t.copy();a[j]+=eps;b[j]-=eps
            high=s.binary_objective(a,z,y,w,C);low=s.binary_objective(b,z,y,w,C)
            self.assertAlmostEqual(g[j],(high[0]-low[0])/(2*eps),places=8)
            np.testing.assert_allclose(h[:,j],(high[1]-low[1])/(2*eps),atol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(h).min(),0)

    def test_partial_reuse_restores_ancestor_scores_and_deduplicates_fit_records(self):
        cfg={'C':1.,'classWeight':None,'effortPower':0.};cfg2={**cfg,'C':10.}
        protocol={'sourceSnapshot':{'release-geometry.cjs':'same','mechanics-probe.cjs':'same'},'records':[],'plans':[],'featureSets':{},
                  'grid':{'C':[1.,10.],'classWeight':[None],'effortPower':[0.]}}
        with tempfile.TemporaryDirectory()as d:
            parent=Path(d)/'parent';child=Path(d)/'partial';parent.mkdir();child.mkdir()
            score=lambda c:{'variant':'geometry','outer':0,'config':c,'threshold':.5}
            trace={'variant':'geometry','outer':0,'inner':0,'configId':0,'gradientMax':1e-8}
            for root,p,search,fits in[(parent,protocol,[score(cfg),score(cfg2)],[trace]),
                                      (child,{**protocol,'extension':{'source':str(parent)}},[score(cfg)],[trace,trace])]:
                for name,value in [('protocol.json',p),('search.json',search),('fits.json',fits),('selection.json',{'selections':[]}),('outer-predictions.json',[])]:
                    (root/name).write_text(json.dumps(value),encoding='utf8')
            search,choices,predictions,traces,models,lineage=s.reuse_lineage(child,[cfg,cfg2])
            self.assertEqual(len(search),2);self.assertEqual(len(traces),1);self.assertEqual(len(lineage),2)


if __name__=='__main__':unittest.main()
