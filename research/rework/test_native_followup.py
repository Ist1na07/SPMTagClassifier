import unittest
import native_followup as f

class TestFollowup(unittest.TestCase):
    def test_missing_factorial_cell_is_matched_to_the_existing_base_pool(self):
        a=f.candidates('factorial_scale');b=f.native.candidates('logical_neutral')
        self.assertEqual([{**c,'source':'logical_neutral'}for c in a],b)

    def test_all_repaired_controls_are_unique_valid_and_cover_their_local_tuples(self):
        for name,spec in f.GROUPS.items():
            if spec.get('historicalControl'):continue
            configs=f.candidates(name)
            self.assertEqual(len(configs),8192);self.assertEqual(len({f.json.dumps(c,sort_keys=True)for c in configs}),8192)
            self.assertTrue(all(c['smoothing']=='coherent'and c['mode']in ['normalized','soft']and c['excludeVibroFromMix']for c in configs))
            modes=[spec['fixed']['mode']]if'mode'in spec.get('fixed',{})else['normalized','soft']
            expected=set(f.itertools.product(spec['sources'],f.GRID['partition'],f.GRID['head'],f.GRID['burdenGate'],modes))
            actual={tuple(c[k]for k in ['source','partition','head','burdenGate','mode'])for c in configs}
            self.assertEqual(actual,expected)

if __name__=='__main__':unittest.main()
