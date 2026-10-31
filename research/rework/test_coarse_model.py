import unittest
import coarse_model as c
from mil import np,torch

class TestCoarseModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weak,manual,_=c.load_headers()
        selected=[next(r for r in weak if r['keys']==K and r['coarseTags']==[tag])for K in[4,7]for tag in c.FINE]
        cls.rows=c.materialize(selected+manual[:1]);cls.config={'windowMs':4000,'burdenGate':0.,'localL2':.003,'strongWeight':.5,'source4Weight':.5}

    def test_head_and_both_pooling_parameter_gradients_match_actual_scalar_loss(self):
        d=c.Packed(self.rows,c.SPECS['full53'],self.config);torch.manual_seed(713)
        x=torch.randn(len(c.FINE)*(d.d+1)+2)*.03;x[-2:]=torch.tensor([.4,.6]);x.requires_grad_(True)
        d.objective(x,self.config).backward();gradient=x.grad.detach().numpy().copy();eps=1e-6
        for j in[3,-1,-2]:
            with torch.no_grad():x[j]+=eps
            high=float(d.objective(x,self.config).detach())
            with torch.no_grad():x[j]-=2*eps
            low=float(d.objective(x,self.config).detach())
            with torch.no_grad():x[j]+=eps
            self.assertAlmostEqual(gradient[j],(high-low)/(2*eps),places=6)

    def test_copy_and_bag_use_the_same_inputs_and_weights_but_distinct_weak_losses(self):
        a=c.Packed(self.rows,c.SPECS['full53'],self.config);b=c.Packed(self.rows,c.SPECS['copied'],self.config)
        self.assertTrue(torch.equal(a.x,b.x));self.assertTrue(torch.equal(a.mask,b.mask));self.assertTrue(torch.equal(a.weak_weight,b.weak_weight))
        x=torch.randn(len(c.FINE)*(a.d+1)+2)*.05;x[-2:]=torch.tensor([.5,.5])
        self.assertLessEqual(float(a.objective(x,self.config)),float(b.objective(x,self.config))+1e-12)
        for K in[4,7]:
            indices=[j for j,i in enumerate(a.weak)if a.records[i]['keys']==K]
            self.assertAlmostEqual(float(a.weak_weight[indices].sum()),.5)

    def test_inference_requires_no_labels_and_keeps_headless_holds_eligible(self):
        d=c.Packed(self.rows,c.SPECS['full53'],self.config)
        r={**self.rows[0],'keys':6,'dataset':'unlabelled6'};r.pop('coarseTags',None);r.pop('annotations',None)
        prediction=c.Packed([r],c.SPECS['full53'],self.config,d.mean,d.scale,training=False)
        self.assertEqual(prediction.weak,[])
        zero=[0.]*len(c.SCHEMA['keys'])
        windows=[{'effort':0.,'start':0,'end':2000,'empty':False,'occupancy':.5,'x':zero},
                 {'effort':1.,'start':2000,'end':4000,'empty':False,'occupancy':0.,'x':zero},
                 {'effort':0.,'start':4000,'end':6000,'empty':True,'occupancy':0.,'x':zero}]
        active,_=c.eligibility(windows,.2);self.assertEqual(active.tolist(),[True,True,False])

if __name__=='__main__':unittest.main()
