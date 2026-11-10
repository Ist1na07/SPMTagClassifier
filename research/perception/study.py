"""Paired nested selection of the integral representation and chart gates."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,itertools,argparse,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from model import Data,read,mil,fit_gate,probability,predict,metrics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write
from rc_action_runtime import sufficient,policy_predictions,POST

ROOT=Path('research/perception/results/integral-r1')
PLANS=read('research/rework/results/mil-stage1-r4/protocol.json')['plans']
REG=[.0003,.003,.03,.3]
BALANCE=[0.,.5,1.]
NAMES=['hard_time','branch_time','difficulty','actions','combined','components']
_D=None


def configurations(d):
    out=[]
    for arm in NAMES:
        for wid,w in enumerate(d.manifest['weights']):
            a,b=w['alpha'],w['beta']
            valid={'hard_time':a==b==0,'branch_time':a==b==0,'difficulty':a>0 and b==0,
                'actions':a==0 and b>0,'combined':a>0 and b>0,'components':True}[arm]
            if not valid:continue
            for power,ridge,balance in itertools.product([1.]if arm=='hard_time'else[1.,2.],REG,BALANCE):
                out.append(dict(arm=arm,weightId=wid,power=power,hard=arm=='hard_time',components=arm=='components',ridge=ridge,balance=balance))
    return out


def objective(m):return .5*m['macroF1']+.25*m['microF1']+.25*m['sampleF1']


def key(row,criterion='balanced'):
    m=row['metrics']
    if criterion=='macro':return m['macroF1'],m['macroRecall'],-row['configId']
    return objective(m),m['macroRecall'],m['macroF1'],-row['configId']


def init():
    global _D
    _D=Data()


def job(outer):
    d=_D;folder=ROOT/str(outer)
    if(folder/'complete.json').exists():return outer,'resumed'
    plan=PLANS[outer];ids=[i for s in plan['inner']for i in s['validation']];vaids=d.main[ids]
    if set(ids)!=set(plan['train'])or len(ids)!=len(set(ids)):raise ValueError('Inner coverage mismatch')
    for s in plan['inner']:
        if {d.records[i]['group']for i in d.main[s['train']]}&{d.records[i]['group']for i in d.main[s['validation']]}:raise ValueError('Song group leakage')
    configs=configurations(d);rows=[];trace=[];predictions=[];scores=[];start=time.monotonic()
    for ci,c in enumerate(configs):
        raw,av,share,family=d.representation(c['weightId'],c['power'],c['hard'],c['components']);ps=[]
        for inner,s in enumerate(plan['inner']):
            # Exact original main inventory, no newly cropped labels fitted.
            model,t=fit_gate(d,raw,av,d.main[s['train']],c);ps.append(probability(raw[d.main[s['validation']]],av[d.main[s['validation']]],model))
            trace.append(dict(configId=ci,inner=inner,seconds=t['seconds'],maximumProjectedGradient=max(r.get('projectedGradient',0.)for r in t['classes']),
                loss=[r.get('loss')for r in t['classes']]))
        p=np.concatenate(ps);q=predict(p,d.classes);m=metrics(d.y[vaids],q,d.weight[vaids]);rows.append(dict(configId=ci,config=c,metrics=m));predictions.append(q);scores.append(p)
        if ci%200==0:print('integral',outer,ci,'/',len(configs),round(time.monotonic()-start,1),'s',flush=True)
    baseline_q=np.array([[c in d.records[i]['legacyOverall']['tags']for c in d.classes]for i in vaids]);base=metrics(d.y[vaids],baseline_q,d.weight[vaids])
    choices={}
    for arm in NAMES+['all','no_rating']:
        available=rows if arm=='all'else[r for r in rows if r['config']['arm']==arm]if arm in NAMES else[r for r in rows if r['config']['arm']in ['hard_time','branch_time','actions']]
        for criterion in ['balanced','macro']:choices[arm+'/'+criterion]=max(available,key=lambda r:key(r,criterion))
    # Independent retuning of the inherited consumer; no expert refitting.
    stats=[sufficient(d.records[i]['frames'])for i in vaids];old=policy_predictions(stats);oldrows=[dict(configId=pi,config=POST[pi],metrics=metrics(d.y[vaids],q,d.weight[vaids]))for pi,q in enumerate(old)]
    oldchoice=max(oldrows,key=key)
    write(folder/'inner-grid.json',rows);write(folder/'fit-receipts.json',trace);write(folder/'selection.json',dict(usesOuterTruth=False,choices=choices,baseline=base,legacyRetuned=oldchoice))
    np.savez_compressed(folder/'inner-vectors.npz',predictions=np.array(predictions),probability=np.array(scores),validationIds=vaids)
    selected={};outer_preds={};queries={}
    for name,choice in choices.items():
        ci=choice['configId'];c=choice['config']
        if ci not in selected:
            raw,av,share,family=d.representation(c['weightId'],c['power'],c['hard'],c['components']);m,t=fit_gate(d,raw,av,d.main[plan['train']],c);selected[ci]=dict(model=m,receipt=t)
        model=selected[ci]['model'];raw,av,share,family=d.representation(c['weightId'],c['power'],c['hard'],c['components']);prob=probability(raw,av,model);q=predict(prob,d.classes)
        outer_preds[name]=[dict(uid=d.records[i]['uid'],predicted=[c for c,v in zip(d.classes,q[i])if v],probability=prob[i].tolist(),percentages=dict(zip(d.fine,(100*share[i]).tolist())))for i in d.main[plan['test']]]
        queries[name]=[dict(uid=r['uid'],predicted=[c for c,v in zip(d.classes,q[i])if v],probability=prob[i].tolist(),percentages=dict(zip(d.fine,(100*share[i]).tolist())),
            heldOutSourceGroup=r['group']not in{d.records[j]['group']for j in d.main[plan['train']]})for i,r in enumerate(d.records)if r['dataset']in ['query','derived']]
    oldouter=policy_predictions([sufficient(d.records[i]['frames'])for i in d.main[plan['test']]],[oldchoice['config']])[0]
    outer_preds['legacy_retuned']=[dict(uid=d.records[i]['uid'],predicted=[c for c,v in zip(d.classes,q)if v])for i,q in zip(d.main[plan['test']],oldouter)]
    write(folder/'outer-models.json',selected);write(folder/'outer-predictions.json',outer_preds);write(folder/'queries.json',queries)
    write(folder/'complete.json',dict(configs=len(configs),innerGateFits=len(trace),outerGateFits=len(selected),maximumProjectedGradient=max(t['maximumProjectedGradient']for t in trace),seconds=time.monotonic()-start))
    return outer,read(folder/'complete.json')


def declare(d):
    protocol=dict(id='perception-integral-r1',baselineVersion='v1.2.2',configurations=configurations(d),plans=PLANS,
        records=[dict(uid=d.records[i]['uid'],group=d.records[i]['group'],tags=d.records[i]['tags'],weight=float(d.weight[i]))for i in d.main],
        selection='INNER pooled OOF: balanced=.5macroF1+.25microF1+.25sampleF1, then macroRecall,macroF1,deterministic ID. Macro-first secondary. All arms independently retuned; no fixed default P/R guards. Final common model uses mean INNER criteria across6 plans, never OUTER selection.',
        fitting='Convex per-tag logistic gates, first feature coefficient nonnegative; strictly positive ridge to physical own-share prior. Train-only scaling, constants removed; balance exponent0/.5/1. No intercept/threshold gauge: prediction threshold fixed.5. Mixture must outrank fine gates to replace them.',
        geometry='Exact continuous-time integrals of branch support*log1p(demand)^alpha*log1p(action rate)^beta. Actions=head+union-tail box rates +held union credit/s/key; 1/2s widths, held0/1. Difficulty prior frozen SPMRatingv1.2.0, historical exposure unknown.',
        local='Frozen v1.2.2 natural raw frames, support=max(0,score)/max(1,maxScore), cap.999 for subsequent log-odds continuity; power1/2. Hard-time control uses literal one-hot. Break zero support but denominator retains physical mass. Percentages overlap and need not sum100.',
        components='Gate conditional means: logDemand plus release/coordination/jack log1p contrasts against Pbar; no raw Rbar-only rule. Other arms do not read component features. No-rating arm uses only hard/branch/actions; no demand dependence.',
        labels='Exact original349 main only. Original6 manual local unchanged in this frozen phase. Cropped I01/I02 and constructed contexts are diagnostics, never fitted or selected. All follow-up joint supervision must remain source-TRAIN only.',
        retrospective='Repeatedly studied original songs/annotations; no fresh blind test claim. Separate observed metric effects, mechanistic evidence and deployment decision.',
        numerical='Every accepted class fit projected gradient<=1e-5. Positive ridge gives convex conditional gate objective; representation grid still discrete, not a global model optimum claim.',
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['integral.py','model.py','study.py']})
    p=ROOT/'protocol.json'
    if p.exists()and read(p)!=protocol:raise ValueError('Frozen protocol changed; use a new study revision')
    if not p.exists():write(p,protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--declare-only',action='store_true');a=p.parse_args();d=Data();declare(d)
    if not a.declare_only:
        with ProcessPoolExecutor(max_workers=a.workers,initializer=init)as pool:
            for r in pool.map(job,range(6)):print('complete outer',r,flush=True)
        counts=[read(ROOT/str(o)/'complete.json')for o in range(6)];write(ROOT/'complete.json',dict(scopes=6,innerGateFits=sum(r['innerGateFits']for r in counts),maxProjectedGradient=max(r['maximumProjectedGradient']for r in counts)))
