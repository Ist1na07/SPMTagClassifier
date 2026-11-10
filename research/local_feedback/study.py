"""Nested, checkpointed matched local-only / whole-only / joint experiment."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,itertools,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import Data,Kernel,fit,metrics,local_metrics,summary,read,write,PLANS,GEOMETRY,HAND

ROOT=Path('research/local_feedback/results/competition-r2')
CACHE=Path('.cache/local-feedback-competition-r2')
D=None


def configurations():
    # Descending ridge provides an exact-scope continuation start, in addition
    # to the zero start. Each ablation reselects every active parameter.
    return [dict(features=f,strongWeight=s,wholeWeight=w,teacherWeight=t,ridge=r)
        for f,s,w,t,r in itertools.product(['geometry','hand'],[0.,1.],[0.,.3,1.,3.],[0.,1.],[3.,.3,.03,.003])if s or w]


def initialize():
    global D
    D=Data()


def eligible(c,arm):
    if arm=='all':return True
    if arm=='local_only':return c['strongWeight']>0 and c['wholeWeight']==0
    if arm=='whole_only':return c['strongWeight']==0 and c['wholeWeight']>0
    if arm=='joint':return c['strongWeight']>0 and c['wholeWeight']>0
    if arm=='joint_no_hand':return eligible(c,'joint')and c['features']=='geometry'
    if arm=='joint_hand':return eligible(c,'joint')and c['features']=='hand'
    if arm=='joint_no_teacher':return eligible(c,'joint')and c['teacherWeight']==0
    if arm=='joint_teacher':return eligible(c,'joint')and c['teacherWeight']>0
    raise ValueError(arm)


ARMS=['all','local_only','whole_only','joint','joint_no_hand','joint_hand','joint_no_teacher','joint_teacher']


def score(r):
    m=r['whole'];return .5*r['local']['accuracy']+.25*m['macroF1']+.125*m['microF1']+.125*m['sampleF1']


def select(rows,baseline,arm,protected):
    values=[r for r in rows if eligible(r['config'],arm)]
    if protected:
        # Product priority is fewer visible mistakes, with bounded whole-tag
        # cost. All figures are on this scope's INNER validation, not OUTER.
        values=[r for r in values if r['local']['accuracy']>=baseline['local']['accuracy']-1e-12
            and r['local']['techCsConfusionMs']<=baseline['local']['techCsConfusionMs']+1e-8
            and all(r['whole'][k]>=baseline['whole'][k]-.01 for k in ['macroPrecision','macroRecall','macroF1','microF1'])]
        values.append(baseline)
    return max(values,key=lambda r:(score(r),-r['local']['techCsConfusionMs'],r['whole']['macroRecall'],-r['configId']))


def compact_metrics(d,ids,pred,locals):
    return dict(whole=metrics(d.y[ids],pred,d.weight[ids]),local=summary(locals))


def job(outer):
    d=D;folder=ROOT/str(outer);cache=CACHE/str(outer);cache.mkdir(parents=True,exist_ok=True)
    if(folder/'complete.json').exists():return outer,'resumed'
    plan=PLANS[outer];begin=time.monotonic();configs=configurations();va=np.concatenate([d.main[s['validation']]for s in plan['inner']]);baseline_pred,_=d.chart_predictions(d.support)
    baseline_loc=[v for s in plan['inner']for v in local_metrics(d,d.manual[s['strongValidation']])]
    baseline=dict(configId=-1,config=dict(features='frozen',strongWeight=0.,wholeWeight=0.,teacherWeight=0.,ridge=0.),
        **compact_metrics(d,va,baseline_pred[va],baseline_loc))
    rows=[];predictions=[];warm={};kernels={}
    for ci,c in enumerate(configs):
        path=cache/f'config-{ci}.json'
        if path.exists():record=read(path)
        else:
            inner=[];ps=[];ls=[]
            for j,s in enumerate(plan['inner']):
                key=(j,c['features']);continuation=(key,c['strongWeight'],c['wholeWeight'],c['teacherWeight'])
                if key not in kernels:kernels[key]=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],c)
                k=kernels[key];k.cfg=c
                theta,receipt=fit(k,starts=2,warm=warm.get(continuation));warm[continuation]=theta
                support,scores,share,_=k.forward(theta,all_rows=True);pred,_=d.chart_predictions(support);ps.append(pred[d.main[s['validation']]])
                ls.extend(local_metrics(d,d.manual[s['strongValidation']],scores))
                inner.append(dict(theta=theta.tolist(),receipt=receipt))
            q=np.concatenate(ps);row=dict(configId=ci,config=c,**compact_metrics(d,va,q,ls))
            record=dict(row=row,inner=inner,predictions=q.tolist(),localDetails=ls);write(path,record)
        for j,r in enumerate(record['inner']):warm[((j,c['features']),c['strongWeight'],c['wholeWeight'],c['teacherWeight'])]=np.array(r['theta'])
        rows.append(record['row']);predictions.append(record['predictions'])
        if ci%8==0:print('competition',outer,ci,'/',len(configs),round(time.monotonic()-begin,1),'seconds',flush=True)
    choices={arm+'/'+guard:select(rows,baseline,arm,guard=='protected')for arm in ARMS for guard in ['balanced','protected']}
    write(folder/'inner-grid.json',rows);write(folder/'selection.json',dict(usesOuterTruth=False,baseline=baseline,choices=choices))
    np.savez_compressed(folder/'inner-vectors.npz',ids=va,predictions=np.array(predictions,bool))
    # All decisions above are persisted before reading any OUTER outcomes.
    models={};outputs={};local={};queries={}
    test=d.main[plan['test']];strong_test=d.manual[plan['strongTest']]
    for name,row in choices.items():
        ci=row['configId']
        if ci not in models:
            if ci<0:support=d.support;scores=None;models[ci]=dict(model=None,receipt=None)
            else:
                k=Kernel(d,d.main[plan['train']],d.manual[plan['strongTrain']],row['config']);theta,receipt=fit(k);support,scores,_,_=k.forward(theta,all_rows=True);models[ci]=dict(model=k.export(theta),receipt=receipt)
            pred,share=d.chart_predictions(support);models[ci]['predictions']=pred;models[ci]['support']=support;models[ci]['scores']=scores;models[ci]['share']=share
        state=models[ci];pred=state['predictions'];share=state['share'];scores=state['scores']
        pack=lambda i:dict(uid=d.records[i]['uid'],actual=d.records[i]['tags'],predicted=[c for c,v in zip(d.classes,pred[i])if v],percentages=dict(zip(d.fine,(100*share[i]).tolist())))
        outputs[name]=[pack(i)for i in test];local[name]=local_metrics(d,strong_test,scores)
        # E/F/G answers are NOT fitted or selected in this study.
        queries[name]=dict(charts=[pack(i)for i,r in enumerate(d.records)if r['dataset']in ['query','derived']],
            localScores={d.records[i]['uid']:scores[slice(*d.records[i]['offset'])].tolist()for i in test}if scores is not None else {})
    write(folder/'outer-models.json',{str(ci):dict(model=s['model'],receipt=s['receipt'])for ci,s in models.items()})
    write(folder/'outer-predictions.json',outputs);write(folder/'outer-local.json',local);write(folder/'queries.json',queries)
    write(folder/'complete.json',dict(configs=len(configs),innerFits=len(configs)*3,outerFits=sum(ci>=0 for ci in models),seconds=time.monotonic()-begin))
    return outer,read(folder/'complete.json')


def declare():
    d=Data();p=ROOT/'protocol.json'
    protocol=dict(id='local-feedback-competition-r2',researchBaseline='v1.1.0',fixedProductControl='v1.2.3',configs=configurations(),plans=PLANS,
        readout=d.profile,scorerCalls=0,
        local='Smooth fixed-family-budget Bernoulli-entropy projection; Helmert class contrasts; TRAIN-only chart-duration weighted feature whitening. Intercept identifiable because whole readout is fixed. Attributes, Break, family routing and zero candidates inherited, not silently synthesized.',
        weak='Original349 whole labels only. Fixed R1 integral/anchored readout, differentiable log-odds distance from its applicable keep/add/override threshold; equal costs within each observable fine class. Mix primitives unknown. Fixed-family Hybrid overrides have no local fine derivative and remain metric errors.',
        strong='Original6 partial allowed-tag interval annotations, actual natural-frame overlap, chart-equal duration weighting. Break/family/zero-candidate infeasibility audited separately. No active E/F/G/I or constructed-label training.',
        selection='INNER pooled source-group validation. Score=.5localMeanAccuracy+.25wholeMacroF1+.125microF1+.125sampleF1. Protected: no local mean/Tech-CS confusion regression; whole macroP/R/F1/micro each within1pp of the fixed product control. Fallback is explicit frozen, not an adopted unchanged local model. Each ablation reselects all its live coordinates.',
        optimizer='Two starts: zero + deterministic random/within-scope larger-ridge continuation. Whitened positive-curvature parameter penalty; no sorting/capping in the allocation map. Every accepted max gradient<=1e-5. Multistart spread and rank/eigenvalues retained, no global-optimum claim.',
        retrospective='Same historically reused349+6 and fixed v1.2.3 prior/readout, not a new blind full-pipeline estimate. Newly trained local parameters and their hyperparameters use only their original grouped TRAIN/INNER scopes.',
        resource=read('research/local_feedback/results/resource-r2.json'),
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['competition.py','study.py','extract.cjs']})
    if p.exists()and read(p)!=protocol:
        old=read(p);repair=read(ROOT/'numerical-repair.json')
        # Reuse only with an explicit accepted-state equation/gradient audit.
        # The original declared protocol/source remains untouched on disk.
        allowed=dict(old);allowed['sources']=dict(old['sources']);allowed['sources']['competition.py']=repair['sourceAfter'];allowed['sources']['study.py']=protocol['sources']['study.py']
        if allowed!=protocol or old['sources']['competition.py']!=repair['sourceBefore']or repair['maxGradientDifference']>1e-8:
            raise ValueError('Frozen protocol changed beyond audited numeric root solver')
    if not p.exists():write(p,protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--declare-only',action='store_true');args=parser.parse_args();declare()
    if not args.declare_only:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize)as pool:
            futures=[pool.submit(job,o)for o in range(6)]
            for f in as_completed(futures):print('complete',f.result(),flush=True)
        cs=[read(ROOT/str(o)/'complete.json')for o in range(6)];write(ROOT/'complete.json',dict(scopes=6,innerFits=sum(r['innerFits']for r in cs),outerFits=sum(r['outerFits']for r in cs)))
