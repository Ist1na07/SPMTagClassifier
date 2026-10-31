"""Actual consumer parity against accepted training code, without refitting.

Selected pure-4K outer models are used as numeric fixtures, not adopted as the
shared final model. Source 4K/7K fixtures are each model's held-out records;
native-6K fixtures have no pattern truth. Tests arithmetic/interval semantics.
"""
from coarse_model import predict, materialize, np, json, Path, SCHEMA
from coarse_data import load_headers, plans, CLASSES
import subprocess
from copy import deepcopy
import argparse
import coarse_supported


def run(mode='none',output='research/rework/results/coarse-runtime-parity-r2'):
    weak, manual, _ = load_headers()
    six_root = Path('.cache/rework-six-physical-r2')
    inventory = json.loads((six_root/'inventory.json').read_text(encoding='utf8'))['records']
    # Label-independent selection covers sparse/long actual note sequences.
    selected = sorted(inventory, key=lambda r: (r['notes'], r['uid']))
    six = [json.loads((six_root/(r['uid'].replace('/','__')+'.json')).read_text(encoding='utf8'))
           for r in [selected[0], selected[len(selected)//2], selected[-1]]]
    requests, references, maximum = [], [], {}
    for o, plan in enumerate(plans(weak, manual)):
        model_path=Path(f'research/rework/results/coarse-4k-baseline-r1/weak4_only/{o}/outer-model.json')
        model=json.loads(model_path.read_text(encoding='utf8'))['model']
        chosen=[next(weak[i] for i in plan['test'] if weak[i]['keys']==K) for K in [4,7]]+[manual[o]]
        records=materialize(chosen)+[six[o%3]]
        models=[model]
        if o==0:
            # Unselected 4s configuration checks the supported middle context;
            # these are arithmetic tests, not alternative performance scores.
            middle=deepcopy(model);middle['config'].update(windowMs=4000,burdenGate=.2)
            models.append(middle)
        for m in models:
            pred=coarse_supported.predict(m,records,mode)
            for j,r in enumerate(records):
                requests.append({'uid':r['uid'],'source':r['source'],'model':m,'options':{'physicalSupport':mode}})
                references.append((r,m,pred,j))
    scratch=Path('.cache/rework-coarse-runtime-parity');scratch.mkdir(exist_ok=True)
    (scratch/'requests.json').write_text(json.dumps(requests),encoding='utf8')
    subprocess.run(['node','research/rework/coarse-parity-driver.cjs',str(scratch/'requests.json'),str(scratch/'actual.json')],check=True)
    actual=json.loads((scratch/'actual.json').read_text(encoding='utf8'))
    counts={str(K):0 for K in [4,6,7]};windows=0
    def near(name,a,b,tol=1e-9):
        error=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
        maximum[name]=max(maximum.get(name,0),error)
        if error>tol:raise AssertionError(f'{name}: {error} > {tol}')
    for a,(r,m,pred,j) in zip(actual,references,strict=True):
        assert a['uid']==r['uid'] and a['keys']==r['keys']
        ws=r['windows'][str(m['config']['windowMs'])];windows+=len(ws);counts[str(r['keys'])]+=1
        assert len(a['windows'])==len(ws)
        near('composition',a['composition'],pred['composition'][j])
        near('mergedComposition',a['mergedComposition'],a['composition'])
        near('mergedDiversity',a['mergedDiversity'],a['diversity'])
        near('mass',a['mass'],pred['mass'][j],1e-7)
        near('local', [w['probability'] for w in a['windows']],pred['local'][j])
        near('weights',[w['weight'] for w in a['windows']],pred['weights'][j],1e-8)
        source=SCHEMA['baseKeys']if m['spec']['representation']=='base'else SCHEMA['keys']
        index=[source.index(k)for k in m['keys']];array='base'if m['spec']['representation']=='base'else'x'
        near('features',[w['features']for w in a['windows']],[[w[array][k]for k in index]for w in ws])
        expected=[]
        for w,local in zip(ws,pred['local'][j],strict=True):
            tag=CLASSES[int(np.argmax(local))]
            if expected and expected[-1]['tag']==tag and expected[-1]['end']==w['start']:expected[-1]['end']=w['end']
            else:expected.append({'start':w['start'],'end':w['end'],'tag':tag})
        assert a['intervals']==expected
        assert a['dominant']==(CLASSES[int(np.argmax(pred['composition'][j]))]if pred['mass'][j]>0 else None)
        for got,w in zip(a['windows'],ws,strict=True):
            for k in ['start','end','contextStart','contextEnd']:assert got[k]==w[k]
            assert got['empty']==w['empty']
        if mode!='none':
            assert [w['structuralSupport']for w in a['windows']]==[None if k<0 else CLASSES[int(k)]for k in pred['rules'][j]]
    result={'scope':'real JavaScript full parsing/native features/head/composition vs Python training prediction; baseline fixture models only, not shared final validation',
      'cases':len(actual),'byKeys':counts,'windows':windows,'contextMs':sorted({r['model']['config']['windowMs']for r in requests}),
      'gate':sorted({r['model']['config']['burdenGate']for r in requests}),'maximumAbsoluteErrors':maximum,
      'intervalsAndPrimaryLabels':'exact','sixPatternAccuracy':'not measured; no human pattern labels',
      'fixtures':[{'uid':r['uid'],'windowMs':r['model']['config']['windowMs'],'gate':r['model']['config']['burdenGate']}for r in requests],
      'physicalSupport':mode,'runtimeSource':Path('research/rework/coarse-runtime.cjs').read_text(encoding='utf8'),
      'physicalSupportSource':Path('research/rework/physical-support.cjs').read_text(encoding='utf8'),
      'trainingPredictionSource':Path('research/rework/coarse_supported.py').read_text(encoding='utf8')}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items()if k not in ['runtimeSource','physicalSupportSource','trainingPredictionSource','fixtures']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--physical-support',choices=['none','jack','ln','both'],default='none');p.add_argument('--output',default='research/rework/results/coarse-runtime-parity-r2');a=p.parse_args();run(a.physical_support,a.output)
