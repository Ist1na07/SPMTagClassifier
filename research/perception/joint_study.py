"""Bounded full joint continuation, checkpointed after every config/fold."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools,time,argparse,sys
import numpy as np
from model import Data,read,metrics
from study import PLANS,objective,key
from joint import Kernel,fit,infer,local_frames,smooth_mix_features
from hierarchy import decode,slot_probability
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write
from rc_action_runtime import smooth

ROOT=Path('research/perception/results/joint-r1');CACHE=Path('.cache/perception-joint-r1')
_D=None


def configurations(d):
    weights=[i for i,w in enumerate(d.manifest['weights'])if w['width']==1000. and w['hold']==1. and w['beta']in [0.,1.]]
    return [dict(weightId=w,power=1.,hard=False,components=True,conditional=True,ridge=r,balance=b,teacherWeight=t,strongWeight=s,localL2=l)
        for w,r,b,t,s,l in itertools.product(weights,[.003,.03],[0.,.5,1.],[0.,1.],[0.,1.],[3.,.3,.03])]


def init():
    global _D
    _D=Data()


def local_metrics(d,ids,support=None,config=None):
    out=[]
    for i in ids:
        row=d.records[i];raw=local_frames(d,i,support,config)if support is not None else row['frames'];fs=smooth(raw);correct=total=techdense=confusion=0.
        for a in row['annotations']:
            if not a['tags']:continue
            total+=a['end']-a['start']
            for f in fs:
                dt=max(0.,min(a['end'],f['end'])-max(a['start'],f['start']));correct+=dt*(f['tag']in a['tags'])
                if 'tech'in a['tags']and f['tag']=='dense chordstream':techdense+=dt
                if (f['tag']=='tech'and any(c in a['tags']for c in ['dense chordstream','fast chordstream']))or(f['tag']in ['dense chordstream','fast chordstream']and'tech'in a['tags']):confusion+=dt
        out.append(dict(uid=row['uid'],accuracy=correct/max(total,1),techToDenseMs=techdense,techCsConfusionMs=confusion))
    return out


def choice(rows,criterion='balanced',baseline=None):
    if criterion=='protected':
        rows=[r for r in rows if r['manualAccuracy']>=baseline['manualAccuracy']-1e-12 and r['techCsConfusionMs']<=baseline['techCsConfusionMs']+1e-9]
    return max(rows,key=lambda r:key(r,'macro'if criterion=='macro'else'balanced'))


def job(outer):
    d=_D;folder=ROOT/str(outer);cache=CACHE/str(outer);cache.mkdir(parents=True,exist_ok=True)
    if(folder/'complete.json').exists():return outer,'resumed'
    plan=PLANS[outer];ids=[i for s in plan['inner']for i in s['validation']];va=d.main[ids];rows=[];vectors=[];parents={};continuation={};start=time.monotonic();configs=configurations(d)
    for ci,c in enumerate(configs):
        p=cache/('config-'+str(ci)+'.json')
        if p.exists():record=read(p)
        else:
            inner=[];joint_p=[];frozen_p=[];joint_l=[];frozen_l=[]
            for j,s in enumerate(plan['inner']):
                parentkey=(j,c['weightId'],c['ridge'],c['balance']);warmkey=(parentkey,c['teacherWeight'],c['strongWeight'])
                k=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],c,parent=parents.get(parentkey));parents[parentkey]=k.parent
                # Include the actual frozen parent with the SAME smooth readout.
                zero=k.model(k.initial);frozen=infer(d,zero);frozen_p.append(decode(d,frozen['fine'][d.main[s['validation']]],frozen['slot'][d.main[s['validation']]],True));frozen_l.extend(local_metrics(d,d.manual[s['strongValidation']]))
                warm=continuation.get(warmkey);theta,t=fit(k,starts=2,warm=warm);continuation[warmkey]=theta
                m=k.model(theta);out=infer(d,m);q=decode(d,out['fine'][d.main[s['validation']]],out['slot'][d.main[s['validation']]],True);joint_p.append(q)
                joint_l.extend(local_metrics(d,d.manual[s['strongValidation']],out['support'],c));inner.append(dict(model=m,receipt=t,trainFrameIds=k.frame_ids.tolist()))
            record=dict(config=c,inner=inner,jointPredictions=np.concatenate(joint_p).tolist(),frozenPredictions=np.concatenate(frozen_p).tolist(),jointLocal=joint_l,frozenLocal=frozen_l)
            write(p,record)
        # Resume continuation from the accepted exact same objective family.
        for j,r in enumerate(record['inner']):
            parentkey=(j,c['weightId'],c['ridge'],c['balance']);warmkey=(parentkey,c['teacherWeight'],c['strongWeight']);best=min(r['receipt']['starts'],key=lambda r:r['loss']);continuation[warmkey]=best['theta']
        for variant in ['frozen','joint']:
            q=np.asarray(record[variant+'Predictions'],bool);ls=record[variant+'Local'];m=metrics(d.y[va],q,d.weight[va]);rows.append(dict(configId=len(rows),fitId=ci,variant=variant,config=c,metrics=m,
                manualAccuracy=float(np.mean([r['accuracy']for r in ls])),techCsConfusionMs=float(sum(r['techCsConfusionMs']for r in ls)),techToDenseMs=float(sum(r['techToDenseMs']for r in ls))))
            vectors.append(q)
        if ci%12==0:print('joint integral',outer,ci,'/',len(configs),round(time.monotonic()-start,1),'s',flush=True)
    baseline=next(r for r in rows if r['variant']=='frozen');choices={}
    for arm in ['all','joint','frozen','no_teacher','teacher','no_strong','strong','rated_integral']:
        eligible=rows if arm=='all'else[r for r in rows if r['variant']==arm]if arm in ['joint','frozen']else[r for r in rows if (r['config']['teacherWeight']==(1. if arm=='teacher'else 0.))]if arm in ['teacher','no_teacher']else[r for r in rows if r['config']['strongWeight']==(1. if arm=='strong'else 0.)]if arm in ['strong','no_strong']else[r for r in rows if d.manifest['weights'][r['config']['weightId']]['alpha']>0 and d.manifest['weights'][r['config']['weightId']]['beta']>0]
        for criterion in ['balanced','macro','protected']:choices[arm+'/'+criterion]=choice(eligible,criterion,baseline)
    write(folder/'inner-grid.json',rows);write(folder/'selection.json',dict(usesOuterTruth=False,choices=choices));np.savez_compressed(folder/'inner-vectors.npz',validationIds=va,predictions=np.array(vectors))
    selected={};outer_preds={};outer_local={};queries={};outgroups={d.records[i]['group']for i in d.main[plan['train']]}
    for name,r in choices.items():
        c=r['config'];ident=r['fitId'],r['variant']
        if ident not in selected:
            k=Kernel(d,d.main[plan['train']],d.manual[plan['strongTrain']],c)
            if r['variant']=='frozen':theta=k.initial;t=None
            else:theta,t=fit(k,starts=2)
            m=k.model(theta);selected[ident]=dict(model=m,receipt=t)
        m=selected[ident]['model'];out=infer(d,m);q=decode(d,out['fine'],out['slot'],True)
        pack=lambda i:dict(uid=d.records[i]['uid'],predicted=[c for c,v in zip(d.classes,q[i])if v],percentages=dict(zip(d.fine,(out['share'][i]*100).tolist())))
        outer_preds[name]=[pack(i)for i in d.main[plan['test']]];outer_local[name]=local_metrics(d,d.manual[plan['strongTest']],out['support'],c)
        queries[name]=[dict(pack(i),heldOutSourceGroup=r['group']not in outgroups)for i,r in enumerate(d.records)if r['dataset']in ['query','derived']]
    write(folder/'outer-models.json',{str(k):v for k,v in selected.items()});write(folder/'outer-predictions.json',outer_preds);write(folder/'outer-local.json',outer_local);write(folder/'queries.json',queries)
    write(folder/'complete.json',dict(configs=len(configs),innerFits=len(configs)*3,seconds=time.monotonic()-start))
    return outer,read(folder/'complete.json')


def declare(d):
    p=ROOT/'protocol.json';v=dict(id='perception-joint-r1',configs=configurations(d),plans=PLANS,
        hypothesis='Whole-tag supervision through the physical integral may correct local prototype affinities. Conditional fine supervision excludes Mix-as-false-primitive-negatives; categorical ordinary/Mix output is separate.',
        scope='Exact same349 original whole labels and original6 manual, original6x3 song/physical groups. Manual loss and unlabeled teacher protection TRAIN-only; no active-query annotation fitted. Unsupported fixed-routing targets remain metric errors and have constant clipped-NLL/zero learning derivative.',
        representation='Natural raw parents, independent prototype supports. Right-continuous SPM demand and head+tail+hold time integral. Joint domain alpha0/1/2 beta0/1,1s action kernel,hold1; raw support power1 is definition, no additional concentration exponent. Full frozen-r1/r2 controls independently tune1/2s,hold0/1,power1/2.',
        gates='Own-share monotone fine logits; train-fitted conditional component moments always accompany Rbar with demand and other roles. Smooth categorical mix: family fraction,active mass,entropy,two-/three-draw diversity (not sorted top-k) to make gradients continuous.',
        local='No intercept; per-prototype TRAIN centering/scaling. Existing branch support0 and Break/family routing stay0. Existing expert odds plus local feature residual: relative demand/actions, R/P,C/P,J/P contrasts,total demand,release independence,crossCut. Teacher Bernoulli KL0/1,local L2.03/.3/3,strong partial local loss0/1 independently joint-tuned.',
        optimizer='Two starts: frozen parent plus random for strongest local L2, frozen parent plus same-scope stronger-L2 continuation thereafter. Up to3 L-BFGS continuation passes. Every accepted projected gradient<=1e-5, objective no worse than both actual starting candidates. Record multistart spread, not a global optimum claim.',
        selection='Same balanced and macro objectives as frozen study. Protected secondary requires INNER manual mean accuracy>=frozen and Tech/CS confusion<=frozen. Rated-integral selection requires both alpha>0,beta>0. Whole-tag OUTER truth never selects; all-grid INNER means select common profile before reading OUTER.',
        resource='Resource full209-dim real-fold directional FD error1.5e-11. Sorted-top mixture caused a nondifferentiable convergence failure; replaced by smooth collision/triple diversity before formal fits. Two-start compact resource~9.4s during other work. Exact frame compaction excludes all non-TRAIN rows from fitting.',
        runtime='Checkpoints per config+scope under.cache/perception-joint-r1, immutable protocol source snapshots. Finished scope receipt required before pooled report. Research6h+integration2h cap from03:50 Asia/Taipei, aim earlier.',
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['integral.py','model.py','hierarchy.py','joint.py','joint_study.py']})
    if p.exists()and read(p)!=v:raise ValueError('Frozen joint protocol changed; do not mix sources')
    if not p.exists():write(p,v)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--declare-only',action='store_true');a=p.parse_args();d=Data();declare(d)
    if not a.declare_only:
        with ProcessPoolExecutor(max_workers=a.workers,initializer=init)as pool:
            for r in pool.map(job,range(6)):print('complete joint',r,flush=True)
        cs=[read(ROOT/str(o)/'complete.json')for o in range(6)];write(ROOT/'complete.json',dict(scopes=6,innerFits=sum(c['innerFits']for c in cs)))
