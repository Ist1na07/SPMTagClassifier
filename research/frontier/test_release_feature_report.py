import unittest
import numpy as np
from sklearn.metrics import average_precision_score
from release_feature_report import weighted_ap,binary


class ReleaseReportTests(unittest.TestCase):
    def test_weighted_ap_preserves_tied_score_semantics(self):
        y=np.array([1,0,1,0]);p=np.array([.9,.9,.3,.1]);w=np.array([1.,2.,3.,1.])
        self.assertAlmostEqual(float(weighted_ap(y,p,w)),average_precision_score(y,p,sample_weight=w))

    def test_vectorized_bootstrap_uses_the_same_metric(self):
        y=np.array([1,0,1]);p=np.array([1,1,0]);w=np.array([1.,2.,1.])
        a=binary(y,p,w);b=binary(y,p,np.stack([w,w]))
        for k in a:self.assertTrue(np.array_equal(b[k],np.array([a[k],a[k]])))


if __name__=='__main__':unittest.main()
