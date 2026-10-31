import unittest
from copy import deepcopy
import numpy as np
from policy_capacity import nested_choice,changes
from native_search import BASE


class CapacityTest(unittest.TestCase):
    def test_capacity_counts_each_real_coordinate_not_synthetic_head_index(self):
        c={**BASE,'source':'logical_neutral','smoothing':'coherent','mode':'normalized'}
        sources={'logical_neutral':{'metricStrength':0,'releaseMs':None}}
        self.assertEqual(changes(c,sources),[])
        self.assertEqual(set(changes({**c,'head':5,'partition':8},sources)),{'lnRatio','scorer','minLength','prominence'})

    def test_outer_labels_predictions_and_metrics_do_not_change_nested_choice(self):
        rng=np.random.default_rng(19);p=rng.integers(0,2,(4,8,3),dtype=np.uint8);a=rng.random((4,4))
        y=rng.integers(0,2,(8,3));weights=np.ones(8);cost=np.array([0,1,2,4])
        plan={'train':list(range(6)),'test':[6,7],'strongTrain':[0,1,2],'strongTest':[3],
              'inner':[{'train':[2,3,4,5],'validation':[0,1],'strongTrain':[1,2],'strongValidation':[0]},
                       {'train':[0,1,4,5],'validation':[2,3],'strongTrain':[0,2],'strongValidation':[1]},
                       {'train':[0,1,2,3],'validation':[4,5],'strongTrain':[0,1],'strongValidation':[2]}]}
        first=nested_choice(p,a,y,weights,plan,cost)
        p2=p.copy();p2[:,6:]=1-p2[:,6:];a2=a.copy();a2[:,3]=1-a2[:,3];y2=y.copy();y2[6:]=1-y2[6:]
        self.assertEqual(first,nested_choice(p2,a2,y2,weights,plan,cost))
        self.assertTrue(all(x['candidate']==0 for x in first[0]['innerPolicyChoices']if x['budget']==0))
        self.assertEqual(first[1],0)


if __name__=='__main__':unittest.main()
