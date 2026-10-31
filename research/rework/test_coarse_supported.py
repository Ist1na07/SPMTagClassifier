import unittest
import coarse_model as model
import coarse_supported as supported


class SupportTest(unittest.TestCase):
    def test_structural_predicates_use_full_native_facts_not_selected_learning_features(self):
        keys=model.SCHEMA['keys'];base=[0.]*len(keys)
        def window(notes,**values):
            x=base.copy()
            for key,value in values.items():x[keys.index(key)]=value
            return {'start':0,'end':2000,'contextStart':0,'contextEnd':2000,'notes':notes,'x':x}
        cases=[window(16,rowRate=8,overlap=1,rowBpm=120),window(16,rowRate=8,overlap=1,rowBpm=120,chordChange=.5),
               window(0,occupancy=1/6),window(1,rowRate=.5,lnHeadFraction=1,occupancy=1/6)]
        records=[{'windows':{'2000':cases}}];config={'windowMs':2000}
        self.assertEqual(supported.masks(records,config,'both').tolist(),[0,-1,4,4])
        self.assertEqual(supported.masks(records,config,'jack').tolist(),[0,-1,-1,-1])
        self.assertEqual(supported.masks(records,config,'ln').tolist(),[-1,-1,4,4])


if __name__=='__main__':unittest.main()
