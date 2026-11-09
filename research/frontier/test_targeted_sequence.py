import unittest
import numpy as np
from targeted_sequence_model import World,Packed,frames,local_metrics,FEATURES
from pressure_data import read
from rc_action_runtime import sufficient

class TargetedSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds={c:World(c)for c in ['native','local2s']}
        cls.scope=read('research/rework/results/mil-stage1-r4/protocol.json')['plans'][0]['inner'][0]
    def pack(self,context,kind):
        w=self.worlds[context];return Packed(w,[w.main[i]for i in self.scope['train']],[w.manual[i]for i in self.scope['strongTrain']],kind=kind)
    def test_zero_model_and_frozen_frames_exactly_preserve_shipped_parent_scores(self):
        for context,w in self.worlds.items():
            for uid in ['manual/DS',w.main[0]['uid'],'local/F02']:
                p=self.pack(context,'both');m=dict(kind='both',scaler=p.scaler,theta=np.zeros(p.dimension).tolist())
                self.assertEqual(frames(w,uid,m),w.inputs[uid]['parents'])
    def test_actual_gradient_with_pressure_and_error_cost(self):
        rng=np.random.default_rng(20261110)
        for context in self.worlds:
            p=self.pack(context,'both');theta=rng.normal(0,.2,p.dimension);v=rng.normal(size=p.dimension);v/=np.linalg.norm(v)
            for boost in [0,3]:
                cfg=dict(localL2=.003,strongWeight=4.,conceptWeight=0.,effortPower=1.,weakBalance=True,primaryPower=4.,errorBoost=boost)
                loss,g=p.objective(theta,cfg);h=1e-5;fd=(p.objective(theta+h*v,cfg)[0]-p.objective(theta-h*v,cfg)[0])/(2*h)
                self.assertAlmostEqual(fd,g@v,places=7)
    def test_native_and_local_bias_have_identical_training_problem(self):
        cfg=dict(localL2=.003,strongWeight=4.,conceptWeight=0.,effortPower=1.,weakBalance=True,primaryPower=4.,errorBoost=3.)
        a=self.pack('native','bias');b=self.pack('local2s','bias')
        for value in [-1.,0.,.5]:
            aa=a.objective(np.array([value]),cfg);bb=b.objective(np.array([value]),cfg)
            self.assertAlmostEqual(aa[0],bb[0],places=11);np.testing.assert_allclose(aa[1],bb[1],atol=1e-11)
    def test_noneligible_parents_never_change(self):
        for context,w in self.worlds.items():
            p=self.pack(context,'both');m=dict(kind='both',scaler=p.scaler,theta=np.full(p.dimension,2.).tolist())
            uid='local/F02';out=frames(w,uid,m)
            for parent in w.inputs[uid]['parents']:
                if parent['tag']not in ['tech','dense chordstream','fast chordstream']:
                    self.assertIn(parent,out)

if __name__=='__main__':unittest.main()
