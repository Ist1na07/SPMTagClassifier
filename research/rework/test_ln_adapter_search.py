import unittest
import ln_adapter_search as s


class LnPolicyTest(unittest.TestCase):
    def test_common_policy_design_is_unique_and_covers_all_actual_routing_gate_mode_tuples(self):
        policies=s.policies()
        self.assertEqual(len(policies),512)
        self.assertEqual(len({s.json.dumps(c,sort_keys=True)for c in policies}),512)
        actual={(c['lnRatio'],c['scorer'],c['burdenGate'],c['mode'])for c in policies}
        expected=set(s.itertools.product([.35,.45,.55],['model','blend'],[0.,.15,.3],['normalized','soft']))
        self.assertEqual(actual,expected)
        self.assertTrue(all(c['partition']==c['head']==0 and c['smoothing']=='coherent'and c['excludeVibroFromMix']for c in policies))


if __name__=='__main__':unittest.main()
