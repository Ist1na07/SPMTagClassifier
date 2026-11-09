"""INNER mean, active-coordinate and frozen-guard contract for finalization."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import bilateral_partition_final as final
from whole_reference import write
from pressure_data import read


class FinalSelection(unittest.TestCase):
    def test_full_selection_uses_inner_means_and_canonical_frozen_guard(self):
        root=Path('.cache/bilateral-final-selector-fixture-r1')
        if root.exists():self.fail('Preserve existing test fixture; use a new root for a changed test')
        source=root/'inputs';output=root/'outputs'
        post=[dict(dominantShare=.52,mixShare=.07,secondRel=.75),dict(dominantShare=.65,mixShare=.1,secondRel=.9)]
        write(source/'complete.json',{});write(source/'protocol.json',dict(postPolicies=post))
        # Outer decoy deliberately contradicts INNER; it must never be read.
        write(source/'analysis.json',dict(winner='physical/8/both/head1'))
        for variant in ['original','physical']:
            for partition in range(9):
                for kind in final.KINDS:
                    for outer in range(6):
                        f=np.full((3,2),.5);r=np.full((3,2),.6);a=np.full((3,5),.6)
                        f[1]=.99;a[1]=.99  # conceptWeight0: ineligible active head
                        f[2]=.52;a[2]=.7
                        if variant=='physical'and partition==3:f[0]=.51
                        if variant=='physical'and partition==0 and kind=='both':f[2]=.505;a[2]=.999 # fails guard
                        if variant=='original'and partition==4 and kind=='hand':f[2]=[.9 if outer%2==0 else .1,.53]
                        folder=source/variant/str(partition)/kind/str(outer);folder.mkdir(parents=True)
                        np.savez_compressed(folder/'inner-bases.npz',macroF1=f,macroRecall=r,manualAccuracy=a)
        with patch.object(final,'ROOT',source),patch.object(final,'OUTPUT',output),patch.object(final,'CONFIGS',[dict(conceptWeight=0),dict(conceptWeight=1)]):
            selected=final.select()
        self.assertEqual((selected['variant'],selected['partition'],selected['kind'],selected['headId'],selected['policyId']),('original',4,'hand',2,1))
        self.assertAlmostEqual(selected['meanInnerMacroF1'],.53)
        saved=read(output/'selection.json');self.assertFalse(saved['usesOuterTruth'])
        frozen=saved['bestEligibleFrozen']
        self.assertEqual((frozen['variant'],frozen['partition'],frozen['kind'],frozen['headId']),('physical',3,'bias',0))
        self.assertIsNone(frozen['config'])


if __name__=='__main__':unittest.main()
