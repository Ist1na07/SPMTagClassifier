import unittest
import numpy as np
from integral import atoms,summarize,shares


def fixture():
    notes=[dict(t=t,col=c,lnEnd=e)for t,c,e in [(0,0,3000),(0,1,None),(100,1,None),(900,1,None),(2000,1,None)]]
    row=dict(uid='test',map=dict(keys=4,firstNote=0,lastNote=3000,notes=notes),frames=[dict(start=0,end=1000),dict(start=1000,end=3000)],geometry=[dict(releaseIndependence=.5,crossCut=.2)]*2)
    curve=dict(query=[0,100,900,2000],demand=[1,9,1,2],Rbar=[1]*4,Pbar=[2]*4,Cbar=[3]*4,Jm=[4]*4)
    return row,curve


class IntegralTests(unittest.TestCase):
    def test_time_not_row_sampling_and_hold_tail(self):
        r,c=fixture();a=atoms(r,c)
        # Exact step-field integral, unequal sample spacing. Not sample mean.
        self.assertAlmostEqual(a['dt']@a['logDemand'],100*np.log(2)+800*np.log(10)+1100*np.log(2)+1000*np.log(3))
        self.assertTrue(np.all(a['action'][a['held']>0]>0))
        self.assertEqual(a['ratingEndpointExtensionMs'],1000)

    def test_partition_invariance_and_overlap(self):
        r,c=fixture();a=atoms(r,c);m,x,_=summarize(r,a,1,1);p=np.array([[.9,.7],[.2,.8]])
        expected=shares(m,p)
        r['frames']=[dict(start=0,end=400),dict(start=400,end=1000),dict(start=1000,end=3000)];r['geometry']*=2;r['geometry']=r['geometry'][:3]
        b=atoms(r,c);n,y,_=summarize(r,b,1,1)
        np.testing.assert_allclose(expected,shares(n,p[[0,0,1]]),atol=1e-14)
        self.assertGreater(shares(n,np.ones((3,2))).sum(),1)

    def test_scaling_common_weight_and_exact_duration(self):
        r,c=fixture();a=atoms(r,c);m,_,_=summarize(r,a,0,0)
        np.testing.assert_allclose(m,[1/3,2/3]);self.assertAlmostEqual(m.sum(),1)
        bad={**c,'query':[0,100,901,2000]}
        with self.assertRaisesRegex(ValueError,'clock'):atoms(r,bad)


if __name__=='__main__':unittest.main()
