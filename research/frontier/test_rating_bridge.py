import unittest
import numpy as np
from rating_bridge import RatingEvidence


class RatingBridgeTests(unittest.TestCase):
    def result(self):
        return dict(model='detail',model_identity='fixture',key_count=7,rate=1.,raw_score=3.,
                    query=[0., 1., 1000.],field=[2., 4., 10.])

    def test_existing_result_needs_no_scorer_and_keeps_sampling_meaning(self):
        e=RatingEvidence(self.result(),expected_keys=7,expected_rate=1.)
        self.assertEqual(e.window(0,1000)['channels']['demand']['rowMean'],3.)
        self.assertIsNone(e.window(2,999)['channels']['demand']['rowMean'])
        self.assertEqual(e.window(2,999)['headRows'],0)

    def test_clock_or_native_key_mismatch_is_rejected(self):
        for k,r in [(6,1.),(7,1.5)]:
            with self.assertRaises(ValueError):RatingEvidence(self.result(),expected_keys=k,expected_rate=r)

    def test_component_grid_must_match_and_inputs_remain_unchanged(self):
        r=self.result();e=RatingEvidence(r,expected_keys=7,expected_rate=1.)
        r['field'][0]=99;self.assertEqual(e.channels['demand'][0],2.)
        with self.assertRaises(ValueError):RatingEvidence(self.result(),expected_keys=7,expected_rate=1.,channels={'x':[1.]})
        with self.assertRaises(ValueError):RatingEvidence(self.result(),expected_keys=7,expected_rate=1.,channels={'demand':[0.,0.,0.]})


if __name__=='__main__':unittest.main()
