import unittest
from optimizer_audit import audit_rows


class OptimizerAuditTest(unittest.TestCase):
    def test_unaccepted_local_stage_is_not_hidden_by_accepted_global_head(self):
        result = audit_rows([{'variant':'copy', 'fit': {
            'refinement':{'status':'small_gradient', 'gradientMax':1e-7},
            'localStage':{'refinement':{'status':'budget_exhausted', 'gradientMax':.1}}}}])
        self.assertEqual(result['headStatuses'], {'small_gradient':1})
        self.assertEqual(result['unaccepted'][0]['stage'], 'local')
        self.assertEqual(result['maxReportedGradient'], .1)

    def test_stability_is_accepted_and_missing_evidence_is_explicit(self):
        result = audit_rows([{'fit':{'refinement':{'status':'train_prediction_stable'}}}, {'fit':{}}])
        self.assertEqual(len(result['unaccepted']),1)
        self.assertEqual(result['unaccepted'][0]['status'],'missing_stop_evidence')


if __name__=='__main__':unittest.main()
