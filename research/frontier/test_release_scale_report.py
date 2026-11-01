import unittest
import numpy as np
from release_scale_report import threshold_optima


class ThresholdProfileTests(unittest.TestCase):
    def test_vectorized_reselection_matches_explicit_partitions(self):
        y=np.array([1,0,1,0]);p=np.array([.9,.9,.3,.1]);weights=np.array([[1.,2.,3.,1.],[2.,1.,0.,4.]])
        thresholds=np.r_[0.,.2,.6,np.nextafter(1.,2.)]
        t,m,f=threshold_optima(y,p,weights)
        for i,w in enumerate(weights):
            choices=[]
            for threshold in thresholds:
                q=p>=threshold;tp=w@(q*(y==1));fp=w@(q*(y==0));fn=w@((~q)*(y==1));tn=w@((~q)*(y==0))
                positive=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else 0
                negative=2*tn/(2*tn+fp+fn)if 2*tn+fp+fn else 0
                choices.append(((positive+negative)/2,positive,-abs(threshold-.5),-threshold,threshold))
            best=max(choices);self.assertAlmostEqual(t[i],best[-1]);self.assertAlmostEqual(m[i],best[0]);self.assertAlmostEqual(f[i],best[1])


if __name__=='__main__':unittest.main()
