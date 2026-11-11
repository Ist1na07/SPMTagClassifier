"""Changed scientific contracts; requires the retained local research cache."""
import unittest
from joint_core import *


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.d=Data();cls.w=Whole(cls.d)

    def test_exact_product_baseline(self):
        actual=self.w.evaluate(self.d.baseline);expected=read('research/local_feedback/results/product-readout-report.json')
        self.assertAlmostEqual(self.d.local(self.d.baseline)['accuracy'],expected['local']['accuracy'],places=13)
        for n,m in actual.items():
            for key in ['macroPrecision','macroRecall','macroF1','microF1','exactMatch']:self.assertAlmostEqual(m[key],expected['whole'][n][key],places=13)

    def test_sample_f1_counts_boolean_matches_twice_in_denominator(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
        from model import metrics
        y=np.array([[True,True],[True,False]]);p=np.array([[True,False],[True,False]]);a=metrics(y,p,np.ones(2));b=metrics(y.astype(float),p.astype(float),np.ones(2));self.assertEqual(a,b);self.assertAlmostEqual(a['sampleF1'],(2/3+1)/2);self.assertLessEqual(a['sampleF1'],1.)

    def test_original_coordinate_export_and_class_gauge(self):
        d=self.d;uids=set(d.manual)
        for family in ['rc','ln']:
            k=Kernel(d,family,uids);theta=np.random.default_rng(12).normal(0,.01,k.size);model=k.export(theta);b=d.blocks[family];z=b['z']@np.array(model['W']).reshape(len(model['classes']),-1).T+np.array(model['b']);np.testing.assert_allclose(z,b['logits']+k.changes(theta),atol=1e-10)
            np.testing.assert_allclose(softmax(z,axis=1),softmax(z+10.,axis=1),atol=1e-13)

    def test_local_and_integral_strength_gradients(self):
        d=self.d;w=self.w;plan=read('research/rework/results/mil-stage1-r4/protocol.json')['plans'][0]['inner'][0];train=np.array(w.main)[plan['train']];groups={w.records[i]['group']for i in train};uids={d.manual[i]for i in plan['strongTrain']}|{e['uid']for e in d.extra if d.records[e['uid']]['group']in groups}
        for family in ['rc','ln']:
            for whole in [0.,1.]:
                k=JointKernel(d,w,family,uids,train,whole_weight=whole,learn_gain=True,normalization='smooth32');rng=np.random.default_rng(54);theta=rng.normal(0,.01,k.size);direction=rng.normal(size=k.size);direction/=np.linalg.norm(direction);eps=1e-5;_,g=k.objective(theta);fd=(k.objective(theta+eps*direction)[0]-k.objective(theta-eps*direction)[0])/(2*eps);self.assertLess(abs(fd-g@direction),1e-6)
                used={w.records[w.bag[j]]['group']for j in k.ids};self.assertFalse(used&{w.records[w.main[i]]['group']for i in plan['validation']})

    def test_parameter_changes_reach_actual_gate(self):
        d=self.d;changed=d.frames({},gains={'ln':.7});self.assertGreater(sum(a['tag']!=b['tag']for a,b in zip(d.baseline,changed)),0)
        relaxed=d.frames({},gains={'ln':.7},confidence_floor=.25);self.assertGreater(sum(a['tag']!=b['tag']for a,b in zip(changed,relaxed)),0)

    def test_frozen_parent_split_preserves_allowed_label_coverage(self):
        d=Data('.cache/core-retune-rc-context-r1.jsonl.gz',frozen_routes=True);a=self.d.local(self.d.baseline);b=d.local(d.baseline);self.assertAlmostEqual(a['accuracy'],b['accuracy'],places=13);self.assertEqual(a['techCsConfusionMs'],b['techCsConfusionMs'])


if __name__=='__main__':unittest.main()
