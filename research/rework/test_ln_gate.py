import unittest
from native_search import CLASSES,BASE,pack,np
from ln_gate import pack_gated


class GateTest(unittest.TestCase):
    def test_ln_protection_is_not_unconditional_ln_or_break_rescue(self):
        def row(i,tag,family,burden):
            p=[float(c==tag)for c in CLASSES]
            return dict(start=i*2000,end=(i+1)*2000,tag=tag,family=family,confidence=1.,burden=burden,probability=p)
        rows=[row(0,'release','ln',1.),row(1,'speed','rc',1.),row(2,'break','break',1.),row(3,'chordjack','rc',100.)]
        rows[-1]['end']=12000
        values=[{'partitions':[{'cases':[{'raw':rows}]}]}];c={**BASE,'partition':0,'head':0,'smoothing':'coherent','burdenGate':.3}
        original=pack(values,c);all_=pack_gated(values,c,'all');protected=pack_gated(values,c);off=pack_gated(values,c,'off')
        self.assertEqual([r['tag']for r in original['chunks'][0]],['break','break','break','chordjack'])
        self.assertEqual([r['tag']for r in protected['chunks'][0]],['release','break','break','chordjack'])
        self.assertEqual([r['tag']for r in off['chunks'][0]],['release','speed','break','chordjack'])
        for k in ['p','duration','effort','tag','family','confidence']:np.testing.assert_array_equal(original[k],all_[k])
        for mode in ['all','ln_protected','off']:
            self.assertEqual(pack_gated(values,{**c,'burdenGate':0.},mode)['chunks'][0],rows)


if __name__=='__main__':unittest.main()
