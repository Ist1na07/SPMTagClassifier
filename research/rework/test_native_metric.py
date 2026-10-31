import unittest,copy
import native_metric_search as metric
import native_followup as s

class TestMetricSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved={k:copy.deepcopy(getattr(s,k))for k in ['GROUPS','CATALOGUE','STUDY_ID','SCOPE','SOURCE_FILES','SOURCE_PARAMETERS','EXTRA_CACHE_SPECIFICATIONS']}
        metric.configure()

    @classmethod
    def tearDownClass(cls):
        for k,value in cls.saved.items():setattr(s,k,value)

    def test_source_catalogue_covers_the_full_metric_release_factorial(self):
        actual={tuple(s.SOURCE_PARAMETERS[n][k]for k in ['releaseMs','metricStrength'])for n in s.GROUPS['metric_repaired']['sources']}
        self.assertEqual(actual,set(s.itertools.product([None,20,40,80],[0,.05,.2,1])))

    def test_all_retuned_controls_have_unique_budgets_and_valid_local_mode_coverage(self):
        for name,spec in s.GROUPS.items():
            configs=s.candidates(name,16384)
            self.assertEqual(len(configs),16384);self.assertEqual(len({s.json.dumps(c,sort_keys=True)for c in configs}),16384)
            modes=[spec['fixed']['mode']]if'mode'in spec.get('fixed',{})else['normalized','soft']
            expected=set(s.itertools.product(spec['sources'],s.GRID['partition'],s.GRID['head'],s.GRID['burdenGate'],modes))
            self.assertEqual({tuple(c[k]for k in ['source','partition','head','burdenGate','mode'])for c in configs},expected)
            self.assertTrue(all(c['smoothing']=='coherent'and c['excludeVibroFromMix']for c in configs))

    def test_outer_labels_and_scores_cannot_change_that_outer_selection(self):
        main=[{'tags':['speed'],'recordWeight':1.},{'tags':['tech'],'recordWeight':1.}]
        predictions=s.np.zeros((2,2,len(s.MAP_CLASSES)),dtype=s.np.uint8)
        predictions[0,:,s.MAP_CLASSES.index('speed')]=1;predictions[1,:,s.MAP_CLASSES.index('tech')]=1
        accuracy=s.np.array([[.8,.1],[.7,.9]]);plan=[{'train':[0],'test':[1],'strongTrain':[0],'strongTest':[1]}]
        before,_=s.choose(predictions,accuracy,main,plan,[{'id':0},{'id':1}])
        main[1]['tags']=['release'];predictions[:,1,:]=1;accuracy[:,1]=[1.,0.]
        after,_=s.choose(predictions,accuracy,main,plan,[{'id':0},{'id':1}])
        self.assertEqual(before,after)

    def test_new_metric_cache_consumes_the_actual_single_source_schema(self):
        main,manual,_=s.load();record=manual[0];name='metric_none_5'
        value=s.load_source(name,[record]);config={**s.BASE,'source':name,'smoothing':'coherent','mode':'normalized','excludeVibroFromMix':True}
        data=s.pack(value,config);p=s.decisions(s.summaries(data,config),config)
        self.assertEqual(p.shape,(1,len(s.MAP_CLASSES)));self.assertTrue(s.np.isfinite(data['p']).all())

if __name__=='__main__':unittest.main()
