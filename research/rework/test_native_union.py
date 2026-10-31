import unittest
from native_union import scope_metrics,best,np
from policy_capacity import candidate_metrics


class SharedUnionTest(unittest.TestCase):
    def test_bitpacked_block_scoring_matches_actual_dense_final_task_scoring(self):
        rng=np.random.default_rng(713);p=rng.integers(0,2,(11,9,17),dtype=np.uint8);a=rng.random((11,6));y=rng.integers(0,2,(9,17));w=rng.random(9)
        ids=[0,2,3,5,6];strong=[0,2,4];packed=np.packbits(p,axis=-1,bitorder='little')
        actual=scope_metrics(packed,a,y,w,ids,strong,block=3);expected=candidate_metrics(p,a,y,w,ids,strong)
        for x,z in zip(actual,expected):np.testing.assert_allclose(x,z,rtol=0,atol=0)
        eligible=np.array([0,2,4,6,8]);chosen=max(eligible,key=lambda i:(expected[2][i],expected[1][i],expected[0][i],-i))
        self.assertEqual(best(actual,eligible),chosen)


if __name__=='__main__':unittest.main()
