"""Compare existing R1 and shipped R3; no search, refit or new scoring."""
from pathlib import Path
import json
import numpy as np
from model import Data,read,metrics
from anchored import base,predictions
from extended import ExtendedData,WEIGHTS
ROOT=Path('research/perception/results/selection-review-r4')

def run():
    d=Data();e=ExtendedData();b=base(d)
    original={r['uid']:r['predicted']for r in read('.cache/perception-original-v110.json')}
    if set(original)!=set(d.manifest['mainUids']):raise ValueError('Original baseline inventory changed')
    legacy=np.array([[c in original[d.records[i]['uid']]for c in d.classes]for i in d.main])
    baseline=metrics(d.y[d.main],legacy,d.weight[d.main]);out={}
    r1=read('research/perception/results/anchored-r1/final-inner-selection.json')['choices']['combined/protected']
    r3=read('research/perception/results/linear-support-r3/detail/selection.json')['common']['combined']
    for name,data,c in [('R1',d,r1),('R3',e,r3)]:
        r=c['representation'];_,_,share,family=data.representation(r['weightId'],r['power'],r['hard']);q=predictions(data,share,family,b,[c['policy']],[c['family']])[0,0]
        if name=='R1':
            lookup={r['uid']:r for o in range(6)for r in read(f'research/perception/results/anchored-r1/{o}/outer-predictions.json')['combined/protected']}
            oof=np.array([[c in lookup[d.records[i]['uid']]['predicted']for c in d.classes]for i in d.main])
        else:
            lookup={r['uid']:r for r in read('research/perception/results/linear-support-r3/detail/oof.json')['predictions']['combined']}
            oof=np.array([[c in lookup[d.records[i]['uid']]['predicted']for c in d.classes]for i in d.main])
        out[name]=dict(profile=c,recipe=data.manifest['weights'][r['weightId']],oof=metrics(d.y[d.main],oof,d.weight[d.main]),
            commonReplay=metrics(d.y[d.main],q[d.main],d.weight[d.main]),
            noTagAfterOldTag=[row['uid']for i,row in enumerate(d.records)if b[i].any()and not q[i].any()],
            query=[dict(uid=row['uid'],predicted=[c for c,y in zip(d.classes,q[i])if y])for i,row in enumerate(d.records)if row['dataset']in ['query','derived']],
            outerPredictions=[dict(uid=d.records[i]['uid'],predicted=[c for c,y in zip(d.classes,oof[k])if y])for k,i in enumerate(d.main)],
            commonPredictions=[dict(uid=d.records[i]['uid'],predicted=[c for c,y in zip(d.classes,q[i])if y])for i in d.main])
    ROOT.mkdir(exist_ok=True)
    folds={}
    for name,data in [('R1',d),('R3',e)]:
        rows=[]
        for o in range(6):
            c=read(f'research/perception/results/anchored-r1/{o}/selection.json')['choices']['combined/protected']if name=='R1'else read('research/perception/results/linear-support-r3/detail/selection.json')['folds'][o]['combined']
            rows.append(dict(recipe=data.manifest['weights'][c['representation']['weightId']],supportPower=c['representation']['power'],policy=c['policy'],family=c['family']))
        folds[name]=rows
    (ROOT/'comparison.json').write_text(json.dumps(dict(researchBaseline='v1.1.0',frozenIntermediatePrior='v1.2.2',classOrder=d.classes,scope='Original349, identical class order/weights/source groups; all values retrospective; no new fits or scoring',originalBaseline=baseline,originalPredictions=[dict(uid=d.records[i]['uid'],predicted=original[d.records[i]['uid']])for i in d.main],intermediateControl=metrics(d.y[d.main],b[d.main],d.weight[d.main]),foldSelections=folds,candidates=out),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (ROOT/'selected-detail.json').write_text(json.dumps(dict(researchBaseline='v1.1.0',study='anchored-r1',profile=r1,recipe=d.manifest['weights'][r1['representation']['weightId']],decision='Restore R1 after retrospective review; original parameters are pooled-INNER selection, not a new OUTER refit.'),indent=2)+'\n',encoding='utf8')
    keys=['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch']
    print('v1.1.0',{k:round(100*baseline[k],2)for k in keys})
    for name,r in out.items():
        for mode in ['oof','commonReplay']:print(name,mode,{k:round(100*r[mode][k],2)for k in keys})
        print(name,'no-tag',r['noTagAfterOldTag'],'queries',r['query'])
if __name__=='__main__':run()
