import unittest,tempfile,copy
from unittest.mock import patch
import convergence as c
from mil import *

class TestContinuation(unittest.TestCase):
    def fixture(self,path):
        main,manual,_=load();rows=main[:4]+manual[:1];variant='original56'
        cfg={'localL2':.003,'mapL2':.0003,'strongWeight':.5,'effortPower':0.,'confidencePower':0.}
        data=Packed(rows,variant);model=Model(data.x.shape[1],False)
        with torch.no_grad():model.local.bias.add_(torch.arange(len(CLASSES))*.01)
        state={'identity':{'variant':variant,'config':cfg,'trainUids':[r['uid']for r in rows]},'policy':copy.deepcopy(c.POLICY),
          'state':model.state_dict(),'mean':data.mean.tolist(),'scale':data.scale.tolist(),'base':{},'complete':True,'restarts':0,
          'refinement':{'status':'budget_exhausted','gradientMax':1.,'loss':2.,'chunks':[]}}
        torch.save(state,path)
        return rows,variant,cfg,state

    def test_extended_cap_must_preserve_every_training_stop_threshold(self):
        new={**c.POLICY,'id':'test-expanded','maxChunks':64}
        self.assertTrue(c.expanded_budget_policy(c.POLICY,new))
        self.assertFalse(c.expanded_budget_policy(c.POLICY,{**new,'gradientMax':1e-3}))
        self.assertFalse(c.expanded_budget_policy(c.POLICY,{**new,'maxChunks':16}))

    def test_exhausted_continuation_loads_weights_instead_of_refitting(self):
        with tempfile.TemporaryDirectory(prefix='spm-continuation-')as tmp:
            path=Path(tmp)/'state.pt';rows,variant,cfg,old=self.fixture(path);new={**c.POLICY,'id':'test-expanded','maxChunks':64}
            result={'policy':new,'status':'train_prediction_stable','chunks':[],'loss':1.,'gradientMax':.01}
            with patch.object(c,'fit',side_effect=AssertionError('must not refit')),patch.object(c,'polish',return_value=result)as polish:
                model,data,trace=c.refined_fit(rows,variant,cfg,path,new,True)
            for k,v in old['state'].items():self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
            self.assertEqual(polish.call_args.kwargs['policy'],new)
            saved=torch.load(path,weights_only=True)
            self.assertTrue(saved['complete']);self.assertEqual(saved['policy'],new)
            self.assertEqual(saved['base']['continuedFrom']['policy'],old['policy'])

    def test_changed_policy_is_rejected_without_explicit_continuation(self):
        with tempfile.TemporaryDirectory(prefix='spm-continuation-')as tmp:
            path=Path(tmp)/'state.pt';rows,variant,cfg,old=self.fixture(path)
            with self.assertRaises(RuntimeError):c.refined_fit(rows,variant,cfg,path,{**c.POLICY,'maxChunks':64})

if __name__=='__main__':unittest.main()
