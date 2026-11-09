"""Short product experiment: exact joint grid, same source groups, no new labels."""
from pathlib import Path
import argparse,itertools,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from targeted_sequence_model import World,Packed,FEATURES,frames,local_metrics,basic_fit,KEYS
from pressure_data import mil,read
from rc_action_runtime import sufficient,policy_predictions,macro_scores,POST,aggregate
from whole_reference import write

ROOT=Path('research/overlay-quality/targeted-sequence-r1')
PLANS=read('research/rework/results/mil-stage1-r4/protocol.json')['plans']
CONFIGS=[dict(localL2=l,strongWeight=s,errorBoost=b,effortPower=e,primaryPower=p,weakBalance=True,conceptWeight=0.)
    for l,s,b,e,p in itertools.product([.0003,.003,.03],[0.,1.,4.],[0.,3.],[0.,1.],[1.,4.])if s or not b]
FIXED=POST.index(dict(dominantShare=.52,mixShare=.07,secondRel=.75))
PARENTS={'bias':[],'sequence':['bias'],'time':['bias'],'global':['bias'],'both':['sequence','time']}
_WORLDS={};_FITS={}


def world_for(context):
    if context not in _WORLDS:_WORLDS[context]=World(context)
    return _WORLDS[context]


def fit(p,cfg):
    key=p.world.context,p.kind,tuple(r['uid']for r in p.main_train),tuple(r['uid']for r in p.manual_train),tuple(cfg.items())
    if key in _FITS:
        m,t=_FITS[key];return m,{**t,'reusedWithinWorker':True}
    seeds=[];proofs=[]
    names=lambda pack:['intercept']+[k for k,keep in zip(pack.scaler['expandedKeys'],pack.scaler['keep'])if keep]
    for kind in PARENTS[p.kind]:
        parent=Packed(p.world,p.main_train,p.manual_train,kind=kind);m,t=fit(parent,cfg);theta=np.zeros(p.dimension)
        for name,value in zip(names(parent),m['theta']):theta[names(p).index(name)]=value
        np.testing.assert_allclose(parent.x@np.asarray(m['theta']),p.x@theta,atol=1e-11)
        loss=p.objective(theta,cfg)[0];old=parent.objective(np.asarray(m['theta']),cfg)[0]
        if abs(loss-old)>1e-10:raise ValueError('Nested TRAIN objective changed')
        seeds.append((loss,theta));proofs.append(dict(parent=kind,embeddedLoss=loss))
    model,trace=basic_fit(p,cfg,min(seeds,key=lambda r:r[0])[1]if seeds else None)
    selected=min(r['loss']for r in trace['starts'])
    if seeds and selected>min(r[0]for r in seeds)+1e-8:raise ValueError('Expanded TRAIN optimum worse than embedded parent')
    trace['nestedTrainProofs']=proofs;_FITS[key]=model,trace;return model,trace


def choose(rows,name,ref):
    if name=='frozen':rows=[r for r in rows if r['headId']==0]
    elif name.startswith('experience'):
        rows=[r for r in rows if r['macroF1']>=ref['macroF1']-1e-12 and r['macroRecall']>=ref['macroRecall']-1e-12]
        if name=='experience_fixed':rows=[r for r in rows if r['policyId']==FIXED]
        elif name=='experience_no_error':rows=[r for r in rows if not r['headId']or r['config']['errorBoost']==0]
        return max(rows,key=lambda r:(-r['pressureConfusion'],r['manualAccuracy'],r['macroF1'],r['macroRecall'],r['headId']==0,r['policyId']==FIXED,-r['headId'],-r['policyId']))
    return max(rows,key=lambda r:(r['macroF1'],r['macroRecall'],r['manualAccuracy'],r['headId']==0,r['policyId']==FIXED,-r['headId'],-r['policyId']))


def job(task):
    context,kind,outer=task;root=ROOT/context/kind/str(outer)
    if(root/'complete.json').exists():return [context,kind,outer,'resumed']
    w=world_for(context);plan=PLANS[outer];ids=[i for s in plan['inner']for i in s['validation']]
    if len(ids)!=len(set(ids))or set(ids)!=set(plan['train']):raise ValueError('INNER inventory changed')
    packs=[]
    for s in plan['inner']:
        if {w.main[i]['group']for i in s['train']}&{w.main[i]['group']for i in s['validation']}:raise ValueError('Group leakage')
        packs.append(Packed(w,[w.main[i]for i in s['train']],[w.manual[i]for i in s['strongTrain']],kind=kind))
    y=mil.encode([w.main[i]for i in ids]);weight=np.array([w.main[i]['recordWeight']for i in ids]);rows=[];models=[];traces=[];scores=[]
    for hid,cfg in enumerate([None]+CONFIGS):
        stats=[];manual=[]
        for inner,s in enumerate(plan['inner']):
            if cfg is None:m=None
            else:
                m,t=fit(packs[inner],cfg);traces.append(dict(headId=hid,inner=inner,**t));models.append(dict(headId=hid,inner=inner,model=m,trainUids=[w.main[i]['uid']for i in s['train']]))
            stats.extend(sufficient(frames(w,w.main[i]['uid'],m))for i in s['validation'])
            manual.extend(local_metrics(w,[w.manual[i]for i in s['strongValidation']],m))
        f,r=macro_scores(y,policy_predictions(stats),weight);acc=float(np.mean([x['accuracy']for x in manual]));cost=float(np.mean([x['pressureConfusion']for x in manual]))
        rows.extend(dict(headId=hid,config=cfg,policyId=j,policy=POST[j],macroF1=float(f[j]),macroRecall=float(r[j]),manualAccuracy=acc,pressureConfusion=cost)for j in range(len(POST)))
        scores.append(dict(headId=hid,manual=manual));
        if hid%20==0:print('targeted',context,kind,outer,hid,'/',len(CONFIGS),flush=True)
    ref=next(r for r in rows if r['headId']==0 and r['policyId']==FIXED)
    choices={n:choose(rows,n,ref)for n in ['frozen','macro','experience','experience_fixed','experience_no_error']}
    choices['shipped_fixed']=ref;write(root/'inner-search.json',rows);write(root/'inner-manual.json',scores)
    write(root/'inner-models.json',models);write(root/'fits.json',traces);write(root/'selection.json',dict(choices=choices,usesOuterTruth=False))
    outer_models={};whole=[];local=[];regressions=[]
    for name,c in choices.items():
        h=c['headId']
        if h not in outer_models:
            if h:m,t=fit(Packed(w,[w.main[i]for i in plan['train']],[w.manual[i]for i in plan['strongTrain']],kind=kind),c['config'])
            else:m=t=None
            outer_models[h]=dict(model=m,trace=t,config=c['config'])
        m=outer_models[h]['model']
        for i in plan['test']:
            r=aggregate(sufficient(frames(w,w.main[i]['uid'],m)),c['policy']);whole.append(dict(name=name,uid=w.main[i]['uid'],predicted=r['tags'],sort=r['sort']))
        local.extend(dict(name=name,**r)for r in local_metrics(w,[w.manual[i]for i in plan['strongTest']],m))
        for uid in ['local/F01','local/F02']:
            fs=frames(w,uid,m);a,b=9000,16500;coverage=sum(max(0,min(b,s['end'])-max(a,s['start']))for s in fs if s['tag']=='speed')/(b-a)
            regressions.append(dict(name=name,uid=uid,speedPrimaryCoverage=coverage,scope='All-fold diagnostic only; no query labels in fit or selection'))
    write(root/'outer-models.json',outer_models);write(root/'outer-whole.json',whole);write(root/'outer-local.json',local);write(root/'regressions.json',regressions)
    write(root/'complete.json',dict(targetInnerFits=len(traces),maximumProjectedGradient=max(r['maximumProjectedGradient']for r in traces),nestedComparisons=sum(len(r['nestedTrainProofs'])for r in traces)))
    return [context,kind,outer,{n:(c['headId'],c['policyId'],c['pressureConfusion'])for n,c in choices.items()}]


def declare():
    protocol=dict(startedLocal='2026-11-10 01:56 Asia/Taipei',hardDeadlineLocal='2026-11-10 04:56 Asia/Taipei',
        purpose='At most2-3hours targeted overlay improvement; no broad model/version iteration. Negative result means retain v1.2.2.',
        hypothesis='Ordered consecutive-chord evidence and per-hand timing distinguish mixed actions lost by global averages. Local2s context tests whether coarse native aggregation hides them.',
        contexts=['native','local2s'],representations=list(FEATURES),configs=CONFIGS,post=POST,plans=PLANS,
        targetInnerFits=2*len(FEATURES)*len(CONFIGS)*18,changedParameters='Only declared grid; weak class balance fixed, H conceptual loss disabled. errorBoost0 when strongWeight0 canonicalised.',
        training='Original singleton Tech/CS whole bags + original partial manual intervals; no new player queries fitted. Boost0/3 scales TRAIN teacher-error interval loss by relative physical effort, symmetrically Tech<->CS. No annotation boundaries in extraction.',
        selection='Joint all head+245post. Experience: INNER macroF1 AND recall >= shipped fixedv122, then minimize displayed pressure-weighted Tech/CS confusion per annotated duration, maximize displayed manual accuracy,F1,R,prefer frozen and original policy. Fixed-post and no-error-boost controls independent full retuning; macro-first and retuned frozen retained.',
        localContract='Native parent boundaries retained; local2s subdivides only native Tech/CS parents into equal >=500ms cells with clipped2s context. Other parents exact. Equal adjacent displayed labels within a parent use duration-weighted affinities; original smoothing+aggregation applied. Zero head exact original parent sequence.',
        parentChecks='Within same context: bias nested in each family, sequence/time nested in both; exact TRAIN prediction+objective inclusion and accepted embedded-parent multistart. Contexts are different representations, not claimed nested.',
        identifiability='Drop jointly missing duplicate indicators by physical feature-pair contract; constants removed from actual TRAIN only. No global optimum claim.',
        retrospective='Same repeatedly studied349whole/6manual/source plans; historical teacher exposure unknown. F01/F02 only diagnostics, not random population evidence.',
        externalSources=read('.cache/overlay-targeted-external-r1/manifest.json'),inputSchema=read('.cache/overlay-targeted-sequence-r1.jsonl.schema.json'),
        sourceSnapshots={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['targeted_sequence_model.py','targeted_sequence_study.py','type_bag_objective.py','rc_type_model.py','cut_consistency_model.py','rc_action_runtime.py']})
    f=ROOT/'protocol.json'
    if f.exists()and read(f)!=protocol:raise ValueError('Frozen targeted protocol changed')
    if not f.exists():write(f,protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--declare-only',action='store_true');p.add_argument('--workers',type=int,default=6);a=p.parse_args();declare()
    if not a.declare_only:
        read(ROOT/'resource-check.json');start=time.monotonic()
        with ProcessPoolExecutor(max_workers=a.workers)as pool:
            for r in pool.map(job,itertools.product(['native','local2s'],FEATURES,range(6))):print(r,flush=True)
        counts=[read(ROOT/c/k/str(o)/'complete.json')for c,k,o in itertools.product(['native','local2s'],FEATURES,range(6))]
        write(ROOT/'complete.json',dict(scopes=len(counts),targetInnerFits=sum(r['targetInnerFits']for r in counts),maximumProjectedGradient=max(r['maximumProjectedGradient']for r in counts),nestedComparisons=sum(r['nestedComparisons']for r in counts),seconds=time.monotonic()-start))
