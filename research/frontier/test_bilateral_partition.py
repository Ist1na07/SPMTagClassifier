import unittest
import numpy as np
from bilateral_partition_model import World, Packed, lift_bias, read
from bilateral_partition_study import reusable_original_bias, PLANS


class BilateralPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World('physical', 4)
        scope = PLANS[0]['inner'][0]
        cls.train = [cls.world.main[i] for i in scope['train']]
        cls.manual = [cls.world.manual[i] for i in scope['strongTrain']]
        cls.cfg = dict(localL2=.003,strongWeight=1.,conceptWeight=1.,effortPower=1.,weakBalance=True,primaryPower=4.)

    def test_actual_fixed_grips_and_real_cuts_have_different_bilateral_support(self):
        for uid, value in [('concept/H01',0.),('concept/H02',1.),('concept/H03',0.),('concept/H04',1.)]:
            np.testing.assert_array_equal(self.world.features(uid,'bilateral'),[[value,value]])

    def test_original_partition_can_reuse_actual_accepted_baseline_inputs(self):
        world = World('original',4)
        self.assertEqual(str(reusable_original_bias(world,0)), str(__import__('pathlib').Path('research/overlay-quality/cut-consistency-r2')))

    def test_gradient_and_nested_train_objective(self):
        bias = Packed(self.world,self.train,self.manual,kind='bias')
        p = Packed(self.world,self.train,self.manual,kind='bilateral')
        mapped = lift_bias(bias,p,np.array([.3]),self.cfg)
        self.assertAlmostEqual(bias.objective(np.array([.3]),self.cfg)[0],p.objective(mapped,self.cfg)[0],places=12)
        theta = np.linspace(-.2,.3,p.dimension); direction = np.linspace(.8,-.4,p.dimension)
        _, g = p.objective(theta,self.cfg)
        for step in [1e-5,3e-6]:
            numerical = (p.objective(theta+step*direction,self.cfg)[0]-p.objective(theta-step*direction,self.cfg)[0])/(2*step)
            self.assertAlmostEqual(numerical,g@direction,places=6)

    def test_new_main_segment_lengths_and_effort_match_their_actual_partition(self):
        for r in self.world.main+self.world.manual:
            data = self.world.inputs[r['uid']]
            self.assertEqual(len(r['segments']),len(data['frames']))
            self.assertEqual([[s['start'],s['end']]for s in r['segments']],data['intervals'])
            self.assertTrue(np.isfinite(data['bilateral']).all())
            self.assertTrue(np.isfinite(data['effort']).all())


if __name__=='__main__':
    unittest.main()
