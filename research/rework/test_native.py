import unittest,subprocess
from native_search import *

class TestNative(unittest.TestCase):
    def test_joint_pool_is_unique_and_covers_every_local_tuple(self):
        for group in ['original','full_soft_no_confidence']:
            configs=candidates(group)
            self.assertEqual(len(configs),8192)
            self.assertEqual(len({json.dumps(c,sort_keys=True)for c in configs}),8192)
            observed={tuple(c[k]for k in ['source','partition','head','burdenGate','smoothing'])for c in configs}
            expected=set(itertools.product(GROUPS[group]['sources'],GRID['partition'],GRID['head'],GRID['burdenGate'],GRID['smoothing']))
            self.assertEqual(observed,expected)
            if group.endswith('no_confidence'):
                self.assertTrue(all(c['mode']=='soft'and c['confidencePower']==0 for c in configs))
        with self.assertRaises(ValueError):candidates('original',1)

    def test_js_python_complete_policy_and_aggregation_parity(self):
        files=list((CACHE/'original').glob('*.json.gz'))[:6]
        self.assertTrue(files,'run the label-free native extractor first')
        values=[]
        for file in files:
            with gzip.open(file,'rt',encoding='utf8')as f:values.append(json.load(f))
        rng=np.random.default_rng(31);configs=candidates('original');payload=[];expected=[]
        for ci in [4*3*2*2,1400,3200,6300]+rng.choice(len(configs),20,replace=False).tolist():
            config=configs[ci];data=pack(values,config);summary=summaries(data,config);prediction=decisions(summary,config)
            for i,value in enumerate(values):
                payload.append({'value':value,'config':config})
                expected.append({'tags':sorted(c for c,b in zip(MAP_CLASSES,prediction[i])if b),
                  'shares':dict(zip(FINE,summary['shares'][i])),'diversity':summary['diversity'][i],
                  'segments':[(s['start'],s['end'],s['tag'])for s in select_segments(value,config)]})
        program="const fs=require('node:fs'),{evaluateCachedNative,selectNativeSegments}=require('./research/rework/native-policy.cjs');const input=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(input.map(x=>{const r=evaluateCachedNative(x.value,x.config);return {tags:r.overall.tags.sort(),shares:r.overall.shares,diversity:r.overall.diversity,segments:selectNativeSegments(x.value,x.config).map(s=>[s.start,s.end,s.tag])}})));"
        output=subprocess.run(['node','-e',program],input=json.dumps(payload),text=True,encoding='utf8',capture_output=True,check=True)
        actual=json.loads(output.stdout)
        for i,(a,e)in enumerate(zip(actual,expected)):
            self.assertEqual(a['tags'],e['tags'],str(i)+' tags')
            self.assertEqual(a['segments'],[list(s)for s in e['segments']],str(i)+' native gate')
            self.assertAlmostEqual(a['diversity'],e['diversity'],places=10)
            for c,v in e['shares'].items():self.assertAlmostEqual(a['shares'].get(c,0),v,places=10,msg=str(i)+' '+c)

    def test_all_break_never_receives_a_fabricated_argmax_label(self):
        probability=[0.]*len(CLASSES);probability[BI]=.6;probability[CLASSES.index('chordjack')]=.4
        value={'partitions':[{'cases':[{'raw':[{'start':0,'end':1000,'tag':'break','family':'break','confidence':1.,'burden':0.,'probability':probability}]}]}]}
        c={**BASE,'partition':0,'head':0,'smoothing':'coherent','mode':'soft'}
        summary=summaries(pack([value],c),c)
        self.assertEqual(int(decisions(summary,c).sum()),0)
        self.assertEqual(float(summary['total'][0]),0.)
    def test_vibro_is_not_a_third_ordinary_mix_type(self):
        rows=[]
        for start,end,tag in [(0,4000,'chordjack'),(4000,7000,'speed'),(7000,10000,'vibro')]:
            p=[float(c==tag)for c in CLASSES]
            rows.append({'start':start,'end':end,'tag':tag,'family':'rc','confidence':1.,'burden':1.,'probability':p})
        value={'partitions':[{'cases':[{'raw':rows,'legacy':rows}]}]}
        c={**BASE,'partition':0,'head':0,'smoothing':'coherent','mode':'soft'}
        data=summaries(pack([value],c),c);pred=decisions(data,c)
        self.assertEqual(int(pred[0,MAP_CLASSES.index('rc mix')]),0)
        pred=decisions(data,{**c,'excludeVibroFromMix':False})
        self.assertEqual(int(pred[0,MAP_CLASSES.index('rc mix')]),1)

if __name__=='__main__':unittest.main()
