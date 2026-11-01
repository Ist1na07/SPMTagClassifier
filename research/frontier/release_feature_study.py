"""Jointly retuned mechanism study: whole-chart Release impression, not tags.

Same97 known-LN records and6x3 song splits as the historical diagnostic.
Every effective C/balance/pooling/release-scale configuration gets its own
inner fits AND exhaustive threshold partitions. Synthetic annotations define
the hypotheses; they are not scored or fitted in this benchmark.
"""
import argparse
import concurrent.futures
import importlib.util
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rework'))
import mil
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score
from rating_bridge import RATING_COMMIT

spec=importlib.util.spec_from_file_location('prior_release_study',Path(__file__).resolve().parents[1]/'rework/release-study.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
VARIANTS=['old3','retained3','retained_no_isolation','retained_no_interaction',
          'rating_D','rating_components','rating_plus_old3','rating_plus_retained3']
SCALES=[15,30,60,120]
TAU_VARIANTS={'retained3','retained_no_isolation','retained_no_interaction','rating_plus_retained3'}
_DATA=None


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)


def pool_missing(x,weights):
    values=[];deviations=[];tails=[]
    for column in x.T:
        take=np.isfinite(column)
        if not take.any():raise ValueError('A whole chart has no observed values for a rating channel')
        w=weights[take];w=w/w.sum();v=column[take];mean=float(w@v)
        order=np.argsort(v,kind='stable');index=min(int(np.searchsorted(np.cumsum(w[order]),.9)),len(v)-1)
        values.append(mean);deviations.append(float(np.sqrt(w@((v-mean)**2))));tails.append(float(v[order[index]]))
    return np.r_[values,deviations,tails]


def build_data():
    parent=read('research/rework/results/release-impression-r5/protocol.json')
    weak,_,_=mil.load();byid={r['uid']:r for r in weak};records=[byid[r['uid']]for r in parent['records']]
    if any(int('release'in mil.target(r))!=p['truth']or r['recordWeight']!=p['weight']or r['group']!=p['group']for r,p in zip(records,parent['records'])):
        raise ValueError('Historical diagnostic record/label/group contract differs')
    geometry={r['uid']:r['features']for r in map(json.loads,Path('.cache/rework-release-geometry-r2.jsonl').read_text().splitlines())}
    actions={r['uid']:r for r in map(json.loads,Path('.cache/frontier-release-actions-r1.jsonl').read_text().splitlines())}
    gkeys=read('.cache/rework-release-geometry-r2-schema.json')['features']
    rating_root=Path('.cache/frontier-rating-r1');complete=read(rating_root/'complete.json')
    if complete['records']!=399 or complete['ratingCommit']!=RATING_COMMIT:raise ValueError('Rating extraction is incomplete or a different version')
    channels=None;matrices={v:{}for v in VARIANTS};feature_names={}
    for r in records:
        intervals=[[s['start'],s['end']]for s in r['segments']];a=actions[r['uid']]
        rating=read(rating_root/'detail'/(r['uid'].replace('/','__')+'.json'))
        if intervals!=a['intervals']or intervals!=[[s['start'],s['end']]for s in rating['windows']]:raise ValueError('Feature intervals differ')
        names=sorted(rating['wholeCurve']['channels'])
        if channels is None:channels=names
        if names!=channels:raise ValueError('Rating component inventory differs')
        gx=np.asarray(geometry[r['uid']]);isolation=gx[:,gkeys.index('headIsolation')];before=gx[:,gkeys.index('heldOtherAtTail')]
        tailrate=gx[:,gkeys.index('tailRatePerKey')]
        rx=np.array([[w['channels'][c]['rowMean']if w['channels'][c]['rowMean']is not None else np.nan for c in channels]for w in rating['windows']])
        duration=np.array([s['end']-s['start']for s in r['segments']],float)
        effort=np.maximum(1e-6,np.array([s['effort']for s in r['segments']],float))
        for variant in VARIANTS:
            for tau in (SCALES if variant in TAU_VARIANTS else [None]):
                after=None
                if tau is not None:
                    rate=np.array([w[SCALES.index(tau)]['retainedPairsPerKeySecond']for w in a['values']])
                    after=np.divide(rate,tailrate*6,out=np.zeros_like(rate),where=tailrate>0)
                    if np.any(after < -1e-12)or np.any(after > 1+1e-12):raise ValueError('Retained-finger normalization differs')
                x=np.c_[isolation,before,isolation*before];keys=['headIsolation','heldBefore','isolation*heldBefore']
                if variant in TAU_VARIANTS:
                    x=np.c_[isolation,after,isolation*after];keys=['headIsolation','heldAfter','isolation*heldAfter']
                if variant=='retained_no_isolation':x=after[:,None];keys=['heldAfter']
                if variant=='retained_no_interaction':x=np.c_[isolation,after];keys=['headIsolation','heldAfter']
                if variant=='rating_D':x=rx[:,[channels.index('demand')]];keys=['rating:demand']
                if variant=='rating_components':x=rx;keys=['rating:'+c for c in channels]
                if variant.startswith('rating_plus_'):x=np.c_[rx,x];keys=['rating:'+c for c in channels]+keys
                feature_names[variant]=keys
                for power in [0.,1.]:
                    identity=(tau,power);matrices[variant].setdefault(identity,[]).append(pool_missing(x,duration*effort**power))
    for variant in VARIANTS:matrices[variant]={k:np.array(v)for k,v in matrices[variant].items()}
    configs={v:[{'C':C,'classWeight':balance,'effortPower':power,'releaseScaleMs':tau}
        for C,balance,power,tau in itertools.product(parent['grid']['C'],[None,'balanced'],[0.,1.],SCALES if v in TAU_VARIANTS else[None])]for v in VARIANTS}
    return dict(records=[{'uid':r['uid'],'group':r['group'],'truth':int('release'in mil.target(r)),'weight':r['recordWeight']}for r in records],
        plans=parent['plans'],features=feature_names,configs=configs,matrices=matrices,
        y=np.array(['release'in mil.target(r)for r in records],int),w=np.array([r['recordWeight']for r in records]))


def init(data):
    global _DATA;_DATA=data


def job(arguments):
    variant,outer,output=arguments;root=Path(output)/variant/str(outer)
    if(root/'complete.json').exists():return variant,outer,'resumed'
    d=_DATA;y,w=d['y'],d['w'];plan=d['plans'][outer];search=[];traces=[];predictions=[]
    for cid,cfg in enumerate(d['configs'][variant]):
        x=d['matrices'][variant][(cfg['releaseScaleMs'],cfg['effortPower'])];p=[];ids=[]
        for inner,val in enumerate(plan['innerValidation']):
            train=[i for i in plan['train']if i not in set(val)];scaler,model,trace=old.fit_model(x[train],y[train],w[train],cfg)
            p.extend(model.predict_proba(scaler.transform(x[val]))[:,1]);ids.extend(val)
            traces.append({'configId':cid,'inner':inner,**trace})
        p=np.array(p);thresholds,macro,positive=old.threshold_scores(y[ids],p,w[ids])
        at=max(range(len(thresholds)),key=lambda j:(macro[j],positive[j],-abs(thresholds[j]-.5),-thresholds[j]))
        ap=float(average_precision_score(y[ids],p,sample_weight=w[ids]))
        search.append({'configId':cid,'config':cfg,'threshold':float(thresholds[at]),'macroF1':float(macro[at]),
            'releaseF1':float(positive[at]),'AP':ap,'thresholdPartitions':len(thresholds)})
        predictions.append(p)
    best=max(search,key=lambda r:(r['macroF1'],r['releaseF1'],r['AP'],-r['config']['C'],-r['configId']))
    write(root/'selection.json',{'usesOuterLabels':False,**best});write(root/'search.json',search);write(root/'fits.json',traces)
    np.savez_compressed(root/'inner-predictions.npz',probability=np.array(predictions),validationIds=np.array(ids))
    cfg=best['config'];x=d['matrices'][variant][(cfg['releaseScaleMs'],cfg['effortPower'])];train,test=plan['train'],plan['test']
    scaler,model,trace=old.fit_model(x[train],y[train],w[train],cfg);p=model.predict_proba(scaler.transform(x[test]))[:,1]
    write(root/'outer-model.json',{'features':d['features'][variant],'config':cfg,'threshold':best['threshold'],
        'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'W':model.coef_[0].tolist(),'b':float(model.intercept_[0]),'fit':trace,
        'trainUids':[d['records'][i]['uid']for i in train]})
    write(root/'outer-predictions.json',[{**d['records'][i],'outer':outer,'variant':variant,'probability':float(v),'predicted':int(v>=best['threshold'])}for i,v in zip(test,p)])
    write(root/'complete.json',{'innerFits':len(traces),'jointConfigurations':len(search),'outerFits':1})
    return variant,outer,len(traces)


def run(args):
    d=build_data();root=Path(args.output)
    protocol={k:v for k,v in d.items()if k not in ['matrices','y','w']}
    protocol.update(id='retained-release-feature-study-r1',primaryComparisons=[['retained3','old3'],['rating_plus_retained3','rating_plus_old3']],
        scope='whole-chart Release impression among97explicitLN records; not local classification or complete tag performance',
        ratingPrior='frozen SPMRating v1.2.0 detail, auxiliary training exposure not fold-clean',ratingCommit=RATING_COMMIT,
        pooling='same duration*modernEffort^a for all representations; per-coordinate observation masks exclude missing head rows, never impute zero',
        selection='complete effective joint grid and all distinct threshold partitions; inner binary macroF1, ReleaseF1, AP, smallerC, deterministic index',
        syntheticCases='mechanism hypotheses only, not fitted or used for validation',
        sourceSnapshots={p.name:p.read_text(encoding='utf8')for p in [Path(__file__),Path(__file__).with_name('release-actions.cjs'),Path(__file__).parents[1]/'rework/release-study.py']})
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=protocol:raise ValueError('Feature study protocol changed')
    write(root/'protocol.json',protocol)
    jobs=[(v,o,str(root))for v in VARIANTS for o in range(6)]
    pending=[j for j in jobs if not(root/j[0]/str(j[1])/'complete.json').exists()]
    if args.max_jobs:pending=pending[:args.max_jobs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(d,))as pool:
        for result in pool.map(job,pending):print('feature study',result,flush=True)
    if not all((root/v/str(o)/'complete.json').exists()for v,o,_ in jobs):
        print('Partial resource check; no aggregate performance or complete.json.',flush=True);return
    rows=[r for v,o,_ in jobs for r in read(root/v/str(o)/'outer-predictions.json')];write(root/'outer-predictions.json',rows)
    metrics={}
    for v in VARIANTS:
        r=[s for s in rows if s['variant']==v];metrics[v]=old.metrics(np.array([s['truth']for s in r]),[s['probability']for s in r],[s['predicted']for s in r],np.array([s['weight']for s in r]))
        metrics[v]['meanFoldAP']=float(np.mean([old.metrics(np.array([s['truth']for s in r if s['outer']==o]),[s['probability']for s in r if s['outer']==o],[s['predicted']for s in r if s['outer']==o],np.array([s['weight']for s in r if s['outer']==o]))['averagePrecision']for o in range(6)]))
    write(root/'metrics.json',metrics);write(root/'complete.json',{'variants':VARIANTS,'outerSelections':len(jobs),
        'innerFits':sum(read(root/v/str(o)/'complete.json')['innerFits']for v,o,_ in jobs),'scope':protocol['scope']})
    print(metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='research/frontier/results/release-feature-r1')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-jobs',type=int);run(p.parse_args())
