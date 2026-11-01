"""Inner-only boundary extension; reuse all valid original fits exactly.

Compare240/480/960ms and the analytically rescaled infinity shape against
15/30/60/120ms. Refit the complete effective C/balance/effort grid at each
new shape. This diagnoses identifiability; no new outer performance claim.
"""
import argparse
import concurrent.futures
import itertools
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
import release_feature_study as base

SHAPES=[240,480,960,'infinity']
VARIANTS=['retained3','rating_plus_retained3']
_DATA=None


def build():
    d=base.build_data();source=Path('research/frontier/results/release-feature-r1')
    protocol=base.read(source/'protocol.json')
    if protocol['sourceSnapshots']['release_feature_study.py']!=Path(base.__file__).read_text(encoding='utf8'):
        raise ValueError('Original producer changed; review reuse rather than silently mixing experiments')
    weak,_,_=base.mil.load();records={r['uid']:r for r in weak}
    geometry={r['uid']:r['features']for r in map(json.loads,Path('.cache/rework-release-geometry-r2.jsonl').read_text().splitlines())}
    actions={r['uid']:r for r in map(json.loads,Path('.cache/frontier-release-actions-r2.jsonl').read_text().splitlines())}
    keys=base.read('.cache/rework-release-geometry-r2-schema.json')['features']
    d['matrices']={v:{}for v in VARIANTS}
    for item in d['records']:
        r=records[item['uid']];a=actions[r['uid']];g=np.asarray(geometry[r['uid']]);isolation=g[:,keys.index('headIsolation')];tailrate=g[:,keys.index('tailRatePerKey')]
        rating=base.read(Path('.cache/frontier-rating-r1/detail')/(r['uid'].replace('/','__')+'.json'))
        intervals=[[s['start'],s['end']]for s in r['segments']]
        if a['intervals']!=intervals:raise ValueError('Extension interval mismatch')
        channels=sorted(rating['wholeCurve']['channels'])
        rx=np.array([[w['channels'][c]['rowMean']if w['channels'][c]['rowMean']is not None else np.nan for c in channels]for w in rating['windows']])
        duration=np.array([s['end']-s['start']for s in r['segments']],float);effort=np.maximum(1e-6,[s['effort']for s in r['segments']])
        for shape in SHAPES:
            rate=np.array([w['retainedPairsPerKeySecond']for w in a['largeScaleLimit']])if shape=='infinity'else np.array([w[SHAPES.index(shape)]['retainedPairsPerKeySecond']for w in a['values']])
            after=np.divide(rate,tailrate*6,out=np.zeros_like(rate),where=tailrate>0)
            if np.any(after<0)or (shape!='infinity'and np.any(after>1+1e-12)):raise ValueError('Extension shape normalization mismatch')
            physical=np.c_[isolation,after,isolation*after]
            for variant in VARIANTS:
                x=physical if variant=='retained3'else np.c_[rx,physical]
                for power in [0.,1.]:d['matrices'][variant].setdefault((shape,power),[]).append(base.pool_missing(x,duration*effort**power))
    d['matrices']={v:{k:np.array(x)for k,x in m.items()}for v,m in d['matrices'].items()}
    C=list(dict.fromkeys(c['C']for c in d['configs']['retained3']))
    d['extensionConfigs']=[{'C':c,'classWeight':b,'effortPower':p,'releaseScaleMs':t}for c,b,p,t in itertools.product(C,[None,'balanced'],[0.,1.],SHAPES)]
    return d


def init(data):
    global _DATA
    _DATA=data


def job(args):
    variant,outer,output=args;root=Path(output)/variant/str(outer)
    if(root/'complete.json').exists():return variant,outer,'resumed'
    d=_DATA;y,w=d['y'],d['w'];plan=d['plans'][outer];traces=[];search=[];predictions=[]
    source=Path('research/frontier/results/release-feature-r1')/variant/str(outer)
    existing=base.read(source/'search.json');offset=len(existing)
    with np.load(source/'inner-predictions.npz')as saved:original=saved['probability'];expected_ids=saved['validationIds']
    for j,cfg in enumerate(d['extensionConfigs']):
        x=d['matrices'][variant][(cfg['releaseScaleMs'],cfg['effortPower'])];p=[];ids=[];cid=offset+j
        for inner,val in enumerate(plan['innerValidation']):
            train=[i for i in plan['train']if i not in set(val)];scaler,model,trace=base.old.fit_model(x[train],y[train],w[train],cfg)
            p.extend(model.predict_proba(scaler.transform(x[val]))[:,1]);ids.extend(val);traces.append({'configId':cid,'inner':inner,**trace})
        if ids!=expected_ids.tolist():raise ValueError('Cannot reuse unmatched validation predictions')
        p=np.array(p);thresholds,macro,positive=base.old.threshold_scores(y[ids],p,w[ids])
        at=max(range(len(thresholds)),key=lambda j:(macro[j],positive[j],-abs(thresholds[j]-.5),-thresholds[j]))
        search.append({'configId':cid,'config':cfg,'threshold':float(thresholds[at]),'macroF1':float(macro[at]),
            'releaseF1':float(positive[at]),'AP':float(average_precision_score(y[ids],p,sample_weight=w[ids])),'thresholdPartitions':len(thresholds)})
        predictions.append(p)
    combined=existing+search
    best=max(combined,key=lambda r:(r['macroF1'],r['releaseF1'],r['AP'],-r['config']['C'],-r['configId']))
    base.write(root/'selection.json',{'usesOuterLabels':False,**best});base.write(root/'search.json',combined);base.write(root/'new-fits.json',traces)
    np.savez_compressed(root/'inner-predictions.npz',probability=np.concatenate([original,np.array(predictions)]),validationIds=expected_ids)
    base.write(root/'complete.json',{'newInnerFits':len(traces),'reusedInnerFits':3*len(existing),'outerFits':0})
    return variant,outer,len(traces)


def run(args):
    d=build();root=Path(args.output)
    protocol={'scope':'inner-only boundary/shape diagnosis; no new outer predictions or generalization claims',
        'sourceStudy':'research/frontier/results/release-feature-r1','records':d['records'],'plans':d['plans'],
        'variants':VARIANTS,'addedShapes':SHAPES,'extensionConfigs':d['extensionConfigs'],
        'limitMeaning':'tau^2/1e6 rescaled exponential-product limit: held-age seconds times remaining-hold seconds; same train-only standardization',
        'sourceSnapshots':{p.name:p.read_text(encoding='utf8')for p in [Path(__file__),Path(base.__file__),Path(__file__).with_name('release-actions.cjs')]},
        'inputSchema':base.read('.cache/frontier-release-actions-r2.jsonl.schema.json')}
    if(root/'protocol.json').exists()and base.read(root/'protocol.json')!=protocol:raise ValueError('Extension protocol changed')
    base.write(root/'protocol.json',protocol);jobs=[(v,o,str(root))for v in VARIANTS for o in range(6)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(d,))as pool:
        for result in pool.map(job,jobs):print('scale boundary',result,flush=True)
    counts=[base.read(root/v/str(o)/'complete.json')for v,o,_ in jobs]
    base.write(root/'complete.json',{'jobs':len(jobs),'newInnerFits':sum(c['newInnerFits']for c in counts),
        'reusedInnerFits':sum(c['reusedInnerFits']for c in counts),'outerFits':0,'scope':protocol['scope']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='research/frontier/results/scale-boundary-r2');p.add_argument('--workers',type=int,default=4);run(p.parse_args())
