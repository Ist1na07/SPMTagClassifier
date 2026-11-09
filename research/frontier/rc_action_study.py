"""Full grouped validation of a restricted motor-action residual and its post.

Local coefficients are trained by bags/actual intervals; complete Cartesian
hyperparameters and245matched legacy post policies are selected together.
Frozen head gets exactly the same post search. No new LN or Break routing.
"""
import argparse,concurrent.futures,itertools,time
from pathlib import Path
import numpy as np
from rc_action_model import World,Packed,fit,read,mil,CONCEPTS
from rc_action_runtime import frames,sufficient,aggregate,policy_predictions,macro_scores,segment_accuracy,POST
from whole_reference import write

CONFIGS=[{'localL2':l,'strongWeight':s,'conceptWeight':c,'effortPower':a,'weakBalance':b}
    for l,s,c,a,b in itertools.product([.0001,.001,.01,.1,1.,10.,100.],[0.,.25,1.,4.],[0.,.25,1.,4.],[0.,1.,2.],[False,True])]
KINDS=['frequency','hand'];_WORLD=None


def init():
    global _WORLD;_WORLD=World()


def pack_stats(stats):
    order=np.full((len(stats),len(mil.MAP_CLASSES)),-1,np.int16);weights=np.zeros_like(order,float);index={c:i for i,c in enumerate(mil.MAP_CLASSES)}
    for j,s in enumerate(stats):
        for k,c in enumerate(s['ranked']):order[j,k]=index[c];weights[j,index[c]]=s['weights'][c]
    return {'order':order,'weights':weights,'total':np.array([s['total']for s in stats]),'familyCount':np.array([s['familyCount']for s in stats],np.int16),'sort':np.array([['rc','ln','hb'].index(s['sort'])for s in stats],np.int8)}


def select(heads,condition):
    eligible=[h for h in heads if h['headId']==0 or condition(h['config'])]
    return max(eligible,key=lambda h:(h['macroF1'],h['macroRecall'],h['segmentAccuracy'],h['headId']==0,-h['headId'],-h['policyId']))


def load_parent_bases(path,ids):
    # NpzFile is lazy: saved[key][i] inside a per-head loop decompresses the
    # ENTIRE matrix each time, and its row view keeps that matrix alive. Load
    # each member once; all per-head views then share one bounded allocation.
    with np.load(path)as saved:z={k:saved[k]for k in saved.files}
    if not np.array_equal(z['validationIds'],ids):raise ValueError('RC reusable validation scope changed')
    stats=[{k:z[k][i]for k in ['order','weights','total','familyCount','sort']}for i in range(len(z['macroF1']))]
    return list(z['macroF1']),list(z['macroRecall']),list(z['manualAccuracy']),stats,sum(a.nbytes for a in z.values())


def evaluate(job,config_grid=None,packed_type=Packed,parent=None,*,fit_function=fit,frame_function=frames,accuracy_function=segment_accuracy,extra_conditions=None,concept_uids=None,checkpoint_dir=None,extra_selections=None):
    kind,o,output=job;world=_WORLD;root=Path(output)/kind/str(o)
    if(root/'complete.json').exists():return kind,o,'resumed'
    grid=CONFIGS if config_grid is None else config_grid
    plan=read('research/rework/results/mil-stage1-r4/protocol.json')['plans'][o];main=world.main;ids=[i for s in plan['inner']for i in s['validation']];truth=mil.encode([main[i]for i in ids]);weight=np.array([main[i]['recordWeight']for i in ids]);fitted=[]
    packs=[packed_type(world,[main[i]for i in s['train']],[world.manual[i]for i in s['strongTrain']],kind=kind)for s in plan['inner']]
    searches=[];score_f1=[];score_recall=[];allstats=[];manual_scores=[];models=[]
    previous=Path(parent)/kind/str(o)if parent else None;previous_outer={}
    if previous is not None:
        if not(previous/'complete.json').exists():raise ValueError('Parent RC scope is incomplete')
        old=read(previous/'search.json')
        for row in old:
            hid=row['headId'];cfg=None if hid==0 else grid[hid-1]
            if hid:
                parent_config=row['config'];expected={**parent_config,**({'primaryPower':1.}if'primaryPower'not in parent_config else{})}
                if cfg!=expected:raise ValueError('RC reusable head configuration changed')
            searches.append({**row,'config':cfg})
        score_f1,score_recall,manual_scores,allstats,_=load_parent_bases(previous/'inner-bases.npz',ids)
        convention='same declared primary-slot objective'if'primaryPower'in old[1]['config']else'original mean-share objective; primaryPower1 differs only by a parameter-constant normalization'
        fitted=[{**r,'reusedFrom':str(previous),'lossConvention':convention}for r in read(previous/'fits.json')]
        models=[{**r,'reusedFrom':str(previous)}for r in read(previous/'inner-models.json')];previous_outer=read(previous/'outer-models.json')
    reused=len(fitted)
    for hid,cfg in enumerate([None]+grid):
        if hid<len(searches):continue
        checkpoint=Path(checkpoint_dir)/kind/str(o)/str(hid)if checkpoint_dir else None
        if checkpoint is not None and checkpoint.with_suffix('.json').exists():
            saved=read(checkpoint.with_suffix('.json'))
            if saved['search']['config']!=cfg or saved['search']['headId']!=hid:raise ValueError('Checkpoint head/config mismatch')
            with np.load(checkpoint.with_suffix('.npz'))as z:
                if not np.array_equal(z['validationIds'],ids):raise ValueError('Checkpoint validation IDs differ')
                searches.append(saved['search']);score_f1.append(z['macroF1'].copy());score_recall.append(z['macroRecall'].copy());manual_scores.append(z['manualAccuracy'].copy());allstats.append({k:z[k].copy()for k in ['order','weights','total','familyCount','sort']})
            fitted.extend(saved['fits']);models.extend(saved['models']);reused+=len(saved['fits']);continue
        first_fit=len(fitted);first_model=len(models)
        stats=[];manual=[];inner_models=[]
        for inner,s in enumerate(plan['inner']):
            if cfg is None:model=None
            else:
                model,trace=fit_function(packs[inner],cfg);fitted.append({'headId':hid,'inner':inner,**trace});models.append({'headId':hid,'inner':inner,'trainUids':[main[i]['uid']for i in s['train']],'model':model})
            inner_models.append(model)
            stats.extend(sufficient(frame_function(world,main[i]['uid'],model))for i in s['validation'])
            manual.extend(accuracy_function(world,[world.manual[i]for i in s['strongValidation']],model))
        probability=policy_predictions(stats);f,r=macro_scores(truth,probability,weight);acc=float(np.mean(manual));best=max(range(len(POST)),key=lambda j:(f[j],r[j],-j))
        searches.append({'headId':hid,'config':cfg,'policyId':best,'policy':POST[best],'macroF1':float(f[best]),'macroRecall':float(r[best]),'segmentAccuracy':acc})
        score_f1.append(f);score_recall.append(r);allstats.append(pack_stats(stats));manual_scores.append(manual)
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            # The small JSON commit marker is written LAST. An interrupted
            # NPZ cannot be treated as a valid head on recovery.
            np.savez_compressed(checkpoint.with_suffix('.npz'),validationIds=np.array(ids),macroF1=f,macroRecall=r,manualAccuracy=np.array(manual),**allstats[-1])
            write(checkpoint.with_suffix('.json'),{'search':searches[-1],'fits':fitted[first_fit:],'models':models[first_model:]})
        if hid and hid%100==0:print('RC heads',kind,o,hid,'/',len(grid),flush=True)
    choices={'full':select(searches,lambda c:c['conceptWeight']>0),'no_concept':select(searches,lambda c:c['conceptWeight']==0.),
        'no_strong':select(searches,lambda c:c['conceptWeight']>0 and c['strongWeight']==0.),'adaptive':select(searches,lambda c:True),'frozen':searches[0]}
    if 'primaryPower'in grid[0]:
        choices['share_control']=select(searches,lambda c:c['conceptWeight']>0 and c['primaryPower']==1.)
        choices['only_sharpened']=select(searches,lambda c:c['conceptWeight']>0 and c['primaryPower']>1.)
    for name,condition in (extra_conditions or{}).items():choices[name]=select(searches,condition)
    for name,chooser in (extra_selections or{}).items():choices[name]=chooser(searches)
    write(root/'search.json',searches);write(root/'selection.json',{'usesOuterLabels':False,'selections':choices});write(root/'fits.json',fitted);write(root/'inner-models.json',models)
    np.savez_compressed(root/'inner-bases.npz',validationIds=np.array(ids),macroF1=np.array(score_f1),macroRecall=np.array(score_recall),manualAccuracy=np.array(manual_scores),**{k:np.array([s[k]for s in allstats])for k in allstats[0]})
    outputs=[];outer_models={};local=[];concept=[]
    for treatment,choice in choices.items():
        hid=choice['headId']
        if hid not in outer_models:
            if hid==0:model=None;trace=None
            elif str(hid)in previous_outer:
                saved=previous_outer[str(hid)];model=saved['model'];trace={**saved['fit'],'reusedFrom':str(previous)}
            else:model,trace=fit_function(packed_type(world,[main[i]for i in plan['train']],[world.manual[i]for i in plan['strongTrain']],kind=kind),choice['config'])
            outer_models[hid]={'config':choice['config'],'model':model,'fit':trace,'trainUids':[main[i]['uid']for i in plan['train']],'strongTrainUids':[world.manual[i]['uid']for i in plan['strongTrain']]}
        model=outer_models[hid]['model']
        for i in plan['test']:
            result=aggregate(sufficient(frame_function(world,main[i]['uid'],model)),choice['policy']);outputs.append({'uid':main[i]['uid'],'outer':o,'kind':kind,'variant':treatment,'predicted':result['tags'],'sort':result['sort']})
        for i,accuracy in zip(plan['strongTest'],accuracy_function(world,[world.manual[i]for i in plan['strongTest']],model)):
            local.append({'uid':world.manual[i]['uid'],'outer':o,'kind':kind,'variant':treatment,'rawPrimaryIntervalAccuracy':accuracy,'frames':frame_function(world,world.manual[i]['uid'],model)})
        for uid in (CONCEPTS if concept_uids is None else concept_uids):
            f=frame_function(world,uid,model);concept.append({'uid':uid,'outer':o,'kind':kind,'variant':treatment,'frames':f,'overall':aggregate(sufficient(f),choice['policy'])})
    write(root/'outer-models.json',outer_models);write(root/'outer-predictions.json',outputs);write(root/'outer-local.json',local);write(root/'concept-predictions.json',concept)
    write(root/'complete.json',{'innerFits':len(fitted),'innerTrainingStarts':sum(len(r['starts'])for r in fitted),'reusedInnerFits':reused,'headChoices':len(searches),'postPoliciesPerHead':len(POST),'jointHeadPostChoices':len(searches)*len(POST),'selectedOuterHeadFits':sum(k!=0 for k in outer_models),
        'reusedOuterHeadFits':sum(v['fit']is not None and'reusedFrom'in v['fit']for v in outer_models.values()),'treatments':list(choices)})
    return kind,o,{k:{'head':v['headId'],'policy':v['policyId'],'innerMacroF1':v['macroF1']}for k,v in choices.items()}


def run(args):
    root=Path(args.output);world=World();protocol={'scope':'retrospective restricted roll-motor residual, full349whole17and6manualgrouped evaluation; LN/Break routes and non-triad local mass frozen',
        'configs':CONFIGS,'kinds':KINDS,'postPolicies':POST,'plans':read('research/rework/results/mil-stage1-r4/protocol.json')['plans'],
        'records':[{'uid':r['uid'],'group':r['group'],'tags':mil.target(r),'weight':r['recordWeight']}for r in world.main],
        'localModel':'roll*[1,train-standardized action] times theta times zero-sum orthonormal contrasts; redistribute existing Speed/Tech/Minijack mass only, exact unchanged at zero roll',
        'weakTask':'single-triad whole-slot labels supervise duration*headEffort^a mixture, never copied to windows; all349charts enter full inner/outer selection',
        'strongTask':'actual manual overlap allowed sets; constructed C16/24/32/48 allowed sets are shared mechanism supervision, not a random accuracy sample; C48Anchor side not learned',
        'featureScaling':'original weak training charts only, per-chart equal RC roll-positive mass; no constructed feature statistics even when their labels are used',
        'selection':'complete local hypergrid crossed with245Cartesian dominantShare/mixShare/secondRel policies; full17macroF1,macroRecall,manualrawprimaryaccuracy,preferfrozen,tiedeterministic',
        'postScope':'other original policy constants/family routing frozen for local attribution; not a claim to re-tune every legacy coordinate; frozen head receives exactly the same245post search',
        'treatments':'full=conceptWeight>0; no_concept=0; no_strong=strongWeight0withconcept>0; adaptive=all; frozen=head0,postretuned. All allow frozen head on equal-or-better inner metrics.',
        'sourceSnapshots':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['rc_action_model.py','rc_action_runtime.py','rc_action_study.py']}}
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=protocol:raise ValueError('RC action protocol changed')
    write(root/'protocol.json',protocol)
    if args.resource_check:
        p=protocol['plans'][0];start=time.perf_counter();s=p['inner'][0];cfg=CONFIGS[0];checks=[]
        for kind in KINDS:
            packed=Packed(world,[world.main[i]for i in s['train']],[world.manual[i]for i in s['strongTrain']],kind=kind);model,trace=fit(packed,cfg);stats=[sufficient(frames(world,world.main[i]['uid'],model))for i in s['validation']];q=policy_predictions(stats);f,r=macro_scores(mil.encode([world.main[i]for i in s['validation']]),q,np.array([world.main[i]['recordWeight']for i in s['validation']]))
            checks.append({'kind':kind,'fit':trace,'weakTrainingBags':len(packed.weak),'postChoices':len(f),'bestInnerMacroF1':float(f.max())})
        write(root/'resource-check.json',{'scope':'one predeclared inner scope/first grid config; timing/convergence only, no outer scoring','config':cfg,'checks':checks,'seconds':time.perf_counter()-start});print('resource',checks,'seconds',time.perf_counter()-start,flush=True);return
    jobs=[(kind,o,str(root))for kind,o in itertools.product(KINDS,range(6))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init)as pool:
        for r in pool.map(evaluate,jobs):print('RC action',r,flush=True)
    counts=[read(root/kind/str(o)/'complete.json')for kind,o,_ in jobs]
    for filename in ['outer-predictions','outer-local','concept-predictions']:write(root/(filename+'.json'),[r for kind,o,_ in jobs for r in read(root/kind/str(o)/(filename+'.json'))])
    write(root/'complete.json',{'kinds':KINDS,'innerFits':sum(r['innerFits']for r in counts),'innerTrainingStarts':sum(r['innerTrainingStarts']for r in counts),
        'jointHeadPostChoices':sum(r['jointHeadPostChoices']for r in counts),'outerDecisions':len(jobs)*5,'selectedOuterHeadFits':sum(r['selectedOuterHeadFits']for r in counts)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='research/frontier/results/rc-action-r1');p.add_argument('--resource-check',action='store_true');p.add_argument('--workers',type=int,default=4);run(p.parse_args())
