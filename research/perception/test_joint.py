import unittest
import numpy as np
from model import Data
from study import PLANS
from joint import Kernel,infer,smooth_mix_features,local_frames


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=Data();p=PLANS[0]['inner'][0];wid=next(i for i,c in enumerate(cls.d.manifest['weights'])if c==dict(alpha=1.,beta=1.,width=1000.,hold=1.))
        c=dict(weightId=wid,power=1.,components=True,hard=False,conditional=True,ridge=.003,balance=.5,localL2=.3,teacherWeight=1.,strongWeight=1.)
        cls.k=Kernel(cls.d,cls.d.main[p['train']],cls.d.manual[p['strongTrain']],c)
    def test_actual_gradient(self):
        k=self.k;rng=np.random.default_rng(7);t=k.initial.copy();t[:k.sizes[0]]=rng.normal(0,.02,k.sizes[0]);dr=rng.normal(size=k.dimension);dr/=np.linalg.norm(dr);e=1e-5;l,g=k.objective(t)
        fd=(k.objective(t+e*dr)[0]-k.objective(t-e*dr)[0])/(2*e)
        self.assertAlmostEqual(fd,float(g@dr),places=7)
    def test_numpy_torch_and_zero_local_protection(self):
        k=self.k;t=k.initial;model=k.model(t);out=infer(self.d,model);o=k.forward(__import__('torch').as_tensor(t))
        np.testing.assert_allclose(out['support'][k.frame_ids],o['support'].detach().numpy(),atol=1e-12)
        np.testing.assert_allclose(out['share'][k.train],o['share'].detach().numpy()[k.train],atol=1e-12)
        np.testing.assert_allclose(out['slot'][k.train],__import__('torch').softmax(o['slot_logits'],1).detach().numpy()[k.train],atol=1e-11)
        for index in self.d.manual:
            fs=local_frames(self.d,index,out['support'],k.cfg)
            self.assertEqual([f['tag']for f in fs],[f['tag']for f in self.d.records[index]['frames']])
            self.assertEqual([f['secondary']for f in fs],[f['secondary']for f in self.d.records[index]['frames']])
    def test_absent_family_zero_diversity_and_no_teacher_leak(self):
        d=self.d;k=self.k;s=d.representation(k.wid)[2];family=d.representation(k.wid)[3];x,av=smooth_mix_features(d,s,family)
        np.testing.assert_allclose(x[family[:,1]==0,5:],0)
        np.testing.assert_allclose(x[family[:,0]==0,2:5],0)
        self.assertTrue(set(k.frame_ids).issubset({j for i in list(k.train)+list(k.manual)for j in range(*d.records[i]['offset'])}))


if __name__=='__main__':unittest.main()
