import unittest
from coarse_data import *

class TestCoarseData(unittest.TestCase):
    def row(self,uid,K,title='',artist='',tags=None,subset=None):
        return {'uid':uid,'keys':K,'dataset':'test','source':uid,'meta':{'Title':title,'Artist':artist},'tags':tags or [],'subset':subset}

    def test_same_song_across_keys_and_rate_variants_stays_one_group(self):
        a=self.row('a',4,'Song 1.2x','Artist');b=self.row('b',7,'Song','Artist');c=self.row('c',4);d=self.row('d',7)
        group([a,b,c,d]);self.assertEqual(a['group'],b['group']);self.assertNotEqual(c['group'],d['group'])

    def test_coarse_ambiguity_is_not_resolved_by_dropping_unknown_alternatives(self):
        self.assertEqual(whole_target(self.row('a',7,tags=['chordjack','minijack'])),['jack'])
        self.assertEqual(whole_target(self.row('a',7,tags=['ln mix'])),['ln'])
        self.assertEqual(whole_target(self.row('a',7,tags=['hybrid'])),[])
        self.assertEqual(whole_target(self.row('a',7,tags=['speed','tech'])),[])
        a=annotate({'annotations':[{'start':0,'end':100,'tags':['speed','wildcard']}]})
        self.assertEqual(a[0]['tags'],[]);self.assertEqual(a[0]['fineTags'],['speed','wildcard'])

    def test_four_key_source_bucket_is_not_a_general_stamina_attribute(self):
        self.assertEqual(whole_target(self.row('a',4,subset='stamina')),['chordstream'])
        self.assertEqual(whole_target(self.row('a',4,subset='course')),[])

if __name__=='__main__':unittest.main()
