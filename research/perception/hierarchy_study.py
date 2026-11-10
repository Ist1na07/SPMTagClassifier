"""Matched conditional supervision and categorical slot ablations."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time,itertools,argparse,sys
import numpy as np
from model import Data,read,metrics
from study import PLANS,NAMES,configurations,objective,key
from hierarchy import fit,probabilities,decode
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write

ROOT=Path('research/perception/results/hierarchy-r2')
_D=None


def init():
    global _D
    _D=Data()


def configs(d):return [dict(c,conditional=flag)for flag in [False,True]for c in configurations(d)]


def job(outer):
    d=_D;folder=ROOT/str(outer)
    if(folder/'complete.json').exists():return outer,'resumed'
    plan=PLANS[outer];ids=[i for s in plan['inner']for i in s['validation']];va=d.main[ids];rows=[];receipts=[];vectors=[];start=time.monotonic()
    for ci,c in enumerate(configs(d)):
        raw,av,share,family=d.representation(c['weightId'],c['power'],c['hard'],c['components']);fines=[];slots=[]
        for inner,s in enumerate(plan['inner']):
            m,t=fit(d,raw,av,share,family,d.main[s['train']],c);fine,slot=probabilities(d,raw,av,share,family,m);fines.append(fine[d.main[s['validation']]]);slots.append(slot[d.main[s['validation']]])
            receipts.append(dict(configId=ci,inner=inner,slot=t['slot'],maximumFineProjectedGradient=max(r.get('projectedGradient',0.)for r in t['fine']['classes'])))
        fine=np.concatenate(fines);slot=np.concatenate(slots)
        for fallback in [False,True]:
            q=decode(d,fine,slot,fallback);m=metrics(d.y[va],q,d.weight[va]);rows.append(dict(configId=len(rows),fitId=ci,config=dict(c,fallback=fallback),metrics=m));vectors.append(q)
        if ci%400==0:print('hierarchy',outer,ci,'/',len(configs(d)),round(time.monotonic()-start,1),'s',flush=True)
    choices={}
    arms=NAMES+['all','no_rating','nonconditional','conditional','no_fallback','with_fallback']
    for arm in arms:
        if arm=='all':rs=rows
        elif arm=='no_rating':rs=[r for r in rows if r['config']['arm']in ['hard_time','branch_time','actions']]
        elif arm in ['conditional','nonconditional']:rs=[r for r in rows if r['config']['conditional']==(arm=='conditional')]
        elif arm in ['no_fallback','with_fallback']:rs=[r for r in rows if r['config']['fallback']==(arm=='with_fallback')]
        else:rs=[r for r in rows if r['config']['arm']==arm]
        for criterion in ['balanced','macro']:choices[arm+'/'+criterion]=max(rs,key=lambda r:key(r,criterion))
    write(folder/'inner-grid.json',rows);write(folder/'selection.json',dict(usesOuterTruth=False,choices=choices));write(folder/'fit-receipts.json',receipts)
    np.savez_compressed(folder/'inner-vectors.npz',validationIds=va,predictions=np.array(vectors))
    models={};out={};queries={}
    for name,candidate in choices.items():
        ci=candidate['fitId'];c=candidate['config'];raw,av,share,family=d.representation(c['weightId'],c['power'],c['hard'],c['components'])
        if ci not in models:m,t=fit(d,raw,av,share,family,d.main[plan['train']],c);models[ci]=dict(model=m,receipt=t)
        fine,slot=probabilities(d,raw,av,share,family,models[ci]['model']);q=decode(d,fine,slot,c['fallback'])
        pack=lambda i:dict(uid=d.records[i]['uid'],predicted=[c for c,v in zip(d.classes,q[i])if v],slotProbability=slot[i].tolist(),fineProbability=fine[i].tolist(),percentages=dict(zip(d.fine,(share[i]*100).tolist())))
        out[name]=[pack(i)for i in d.main[plan['test']]];groups={d.records[j]['group']for j in d.main[plan['train']]}
        queries[name]=[dict(pack(i),heldOutSourceGroup=r['group']not in groups)for i,r in enumerate(d.records)if r['dataset']in ['query','derived']]
    write(folder/'outer-models.json',models);write(folder/'outer-predictions.json',out);write(folder/'queries.json',queries)
    write(folder/'complete.json',dict(fits=len(receipts),predictedConfigurations=len(rows),seconds=time.monotonic()-start,
        maximumFineProjectedGradient=max(t['maximumFineProjectedGradient']for t in receipts),maximumSlotGradient=max(t['slot']['gradientMax']for t in receipts)))
    return outer,read(folder/'complete.json')


def declare(d):
    p=ROOT/'protocol.json';v=dict(id='perception-hierarchy-r2',configs=configs(d),fallback=[False,True],plans=PLANS,
        reason='A whole Mix slot does not deny its primitive actions. Compare conditional fine supervision vs complete endpoint BCE with the same categorical ordinary/RC-mix/LN-mix/Hybrid gate. Fallback only when ordinary chosen and some positive fine support; empty physical evidence stays empty.',
        selection='Same balanced and macro criteria as integral-r1, every arm retuned including supervision and fallback. Gate decision.5 fixed; slot uses categorical argmax with ordinary reference logit0. No OUTER selection.',
        numerical='Convex positive-ridge fine and slot objectives; same projected-gradient guards. Mixture gauge removed by ordinary logit0; no duplicate RC/LN fraction columns.',
        correction='r1 incomplete: one outer failed stationarity because a physically unsupported target entered a -1e6 masked-logit likelihood, creating a spurious normalizer gradient and large constant. r2 treats unsupported targets as constant clipped-NLL with zero derivative, reports their count/mass, retains complete metric errors. Up to3 numerical continuation passes; never accepts bad gradient. All6 scopes recomputed; no r1 output reused as this candidate.',
        support='Exact same frozen branch supports/integrals as integral-r1; no local classifier adaptation in this phase.',
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['model.py','hierarchy.py','hierarchy_study.py']})
    if p.exists()and read(p)!=v:raise ValueError('Frozen hierarchy protocol changed')
    if not p.exists():write(p,v)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);a=p.parse_args();d=Data();declare(d)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init)as pool:
        for r in pool.map(job,range(6)):print('complete hierarchy',r,flush=True)
    cs=[read(ROOT/str(o)/'complete.json')for o in range(6)];write(ROOT/'complete.json',dict(scopes=6,fits=sum(c['fits']for c in cs),maxFineGradient=max(c['maximumFineProjectedGradient']for c in cs),maxSlotGradient=max(c['maximumSlotGradient']for c in cs)))
