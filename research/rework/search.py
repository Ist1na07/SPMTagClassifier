"""Nested grouped, FULL JOINT grid for the shared local/aggregation model.

Stage 1 intentionally fixes the original segmentation. Segmenter coupling is
studied separately; neither this script nor its report may call it full-pipeline
validation. No local proxy preselects a candidate.
"""
import argparse,concurrent.futures,subprocess,traceback,shutil
from mil import *
OUT=Path('research/rework/results/mil-stage1-r2')
GRID={'localL2':[.003,.03],'mapL2':[.00003,.0003],'strongWeight':[.5,4.],
      'effortPower':[0.,1.],'confidencePower':[0.,1.]}
THRESHOLDS=[{'fine':f,'mix':m} for f,m in itertools.product([.2,.35,.5],[.2,.35,.5])]
def configs(variant):
    grid=GRID.copy()
    if variant=='frozen_prior':grid={**grid,'localL2':[.03],'strongWeight':[.5]}
    return [dict(zip(grid,values)) for values in itertools.product(*grid.values())]
_MAIN=_MANUAL=_PLAN=None
_REFINE=False
_STATES=None
_POLICY=None
_CONTINUE_EXHAUSTED=False
_REUSE_STATES=None
def init_worker(plan,refine=False,states=None,policy=None,continue_exhausted=False,reuse_states=None):
    global _MAIN,_MANUAL,_PLAN,_REFINE,_STATES,_POLICY,_CONTINUE_EXHAUSTED,_REUSE_STATES
    _MAIN,_MANUAL,_=load();_PLAN=plan;_REFINE=refine;_STATES=states
    _POLICY=policy;_CONTINUE_EXHAUSTED=continue_exhausted;_REUSE_STATES=reuse_states

def fit_candidate(rows,variant,cfg,key,refine=False,states=None,policy=None,continue_exhausted=False):
    if not refine:return fit(rows,variant,cfg)
    from convergence import refined_fit,POLICY
    return refined_fit(rows,variant,cfg,Path(states)/(key.replace('/','-')+'.pt')if states else None,
                       policy=policy or POLICY,continue_exhausted=continue_exhausted)
def run(job):
    variant,outer,inner,cid=job;split=_PLAN[outer]['inner'][inner]
    train=[_MAIN[i] for i in split['train']]+[_MANUAL[i] for i in split['strongTrain']]
    val=[_MAIN[i] for i in split['validation']];strong=[_MANUAL[i] for i in split['strongValidation']]
    cfg=configs(variant)[cid];start=time.perf_counter()
    key='/'.join(map(str,job))
    if _CONTINUE_EXHAUSTED:
        destination=Path(_STATES)/(key.replace('/','-')+'.pt');source=Path(_REUSE_STATES)/destination.name
        if not destination.exists():
            if not source.exists():raise RuntimeError('Missing train checkpoint for exhausted continuation: '+key)
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix('.copy.tmp');shutil.copyfile(source,temporary);temporary.replace(destination)
    model,packed,fittrace=fit_candidate(train,variant,cfg,key,_REFINE,_STATES,_POLICY,_CONTINUE_EXHAUSTED)
    prob,_,_=predict(model,packed,val,cfg);_,local,_=predict(model,packed,strong,cfg)
    return {'key':'/'.join(map(str,job)),'variant':variant,'outer':outer,'inner':inner,'configId':cid,
      'config':cfg,'fit':{**fittrace,'seconds':time.perf_counter()-start},
      'mapIndices':split['validation'],'mapProbability':prob.tolist(),'segmentMetrics':segment_metrics(strong,local)}
def plan(main,manual):
    folds=grouped_folds(main,6);plans=[]
    for outer,test in enumerate(folds):
        train=[i for i in range(len(main)) if i not in set(test)]
        inner=grouped_folds([main[i] for i in train],3)
        sm=[i for i in range(len(manual)) if i!=outer]
        inner_splits=[]
        for j,validation in enumerate(inner):
            va=[train[i] for i in validation];tr=[i for i in train if i not in set(va)]
            sv=[i for k,i in enumerate(sm) if k%3==j];st=[i for i in sm if i not in sv]
            inner_splits.append({'train':tr,'validation':va,'strongTrain':st,'strongValidation':sv})
        plans.append({'train':train,'test':test,'strongTrain':sm,'strongTest':[outer],'inner':inner_splits})
    return plans
def record_support(main,ids):
    return {'records':len(ids),'groups':len({main[i]['group'] for i in ids}),
      'labelRecords':dict(Counter(t for i in ids for t in target(main[i]))),
      'labelGroups':{c:len({main[i]['group'] for i in ids if c in target(main[i])}) for c in MAP_CLASSES}}
def save_model(path,model,packed,cfg,threshold,scope):
    dump(path,{'variant':packed.variant,'config':cfg,'threshold':threshold,'scope':scope,
      'featureKeys':[k+':'+f for k in VARIANTS[packed.variant][0] for f in SCHEMA[k]],
      'classes':CLASSES,'mapClasses':MAP_CLASSES,'mean':packed.mean.tolist(),'scale':packed.scale.tolist(),
      'state':{k:v.detach().numpy().tolist() for k,v in model.state_dict().items()},
      'prior':model.prior,'clip':6,'composition':'duration * effort^a * sum(p^2)^b; normalize; conditional non-Break composition',
      'fineChartHead':'positive slope times logit(own composition) plus bias; no cross-class relabeling'})
def main_run(args):
    global OUT
    OUT=Path(args.output)
    main,manual,data=load();plans=plan(main,manual);variants=args.variants.split(',') if args.variants else list(VARIANTS)
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'id':'7k-mil-fixed-original-segmentation-r2','sourceCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'variants':variants,'grid':GRID,'thresholds':THRESHOLDS,'selection':'.5 * pooled inner map macroF1 + .5 * equal-chart inner segment acceptable-label accuracy',
      'tieBreak':'segment accuracy, map macroF1, then deterministic first grid order',
      'outer':6,'inner':3,'partition':'fixed original automatic segment boundaries for ALL feature representations',
      'manualWeights':'each annotated chart has total strong weight 1; duration-normalized acceptable-label NLL',
      'mapWeights':'each distinct physical source file total weight 1; conflicting duplicate-file annotations remain separate, divided weight',
      'confidence':'sum of squared segment probabilities: concentration, NOT empirically calibrated correctness',
      'weakExistence':'nonempty chart label supplies a max-MIL active-evidence likelihood in addition to final chart-tag likelihood; detached in strong-only/copy-local controls',
      'consistency':'if EVERY displayed local primary is Break, overall prediction is empty; zero-score vector remains empty, never an invented argmax class',
      'exposure':'all local data have been seen in prior development; retrospective nested grouped CV; historical original-model exposure unknown',
      'optimizer':'full batch Adam max300 then LBFGS max100; train-only convergence; fit loss and residual gradient recorded',
      'classSupport':record_support(main,list(range(len(main)))),'plans':plans,
      'records':[{'index':i,'uid':r['uid'],'group':r['group'],'tags':target(r),'recordWeight':r['recordWeight']} for i,r in enumerate(main)],
      'manual':[{'index':i,'uid':r['uid'],'group':r['group']}for i,r in enumerate(manual)]}
    if args.refine:
        from convergence import POLICY
        fit_policy=POLICY
        protocol.update(id='7k-mil-fixed-original-segmentation-r3',optimizer=POLICY,
          refinementScope='same splits/grid/final inner objective; train-only optimizer refinement; detached controls fit their local objective before fitting the frozen-local map head',
          reusedFits='r2 fits already satisfying the new gradient stopping condition are retained unchanged')
    else:fit_policy=None
    if args.continue_exhausted:
        if not(args.refine and args.reuse and args.reuse_states):raise RuntimeError('Exhausted continuation requires refined parent results and their train checkpoints')
        fit_policy={**POLICY,'id':'train-stability-r4','maxChunks':args.max_chunks}
        protocol.update(id='7k-mil-fixed-original-segmentation-r4',optimizer=fit_policy,
          reusedFits='all r3 fits already satisfying the IDENTICAL gradient or two-check training-stability criterion, including valid local stages; continue only exhausted fits from their saved weights',
          sourceBaseCommit=protocol.pop('sourceCommit'),
          sourceSnapshot={n:(Path(__file__).parent/n).read_text(encoding='utf8')for n in ['search.py','convergence.py','mil.py']},
          outerReuse='after inner-only reselection, retain an existing accepted outer model/output only when config and threshold are identical; fit changed choices only')
    old=OUT/'protocol.json'
    if old.exists() and json.loads(old.read_text(encoding='utf8'))!=protocol:
        raise RuntimeError('Protocol differs from existing run; use a new result directory/revision, not a silent mixed resume')
    dump(old,protocol)
    dump(OUT/'supports.json',[{'outer':i,'train':record_support(main,p['train']),'test':record_support(main,p['test']),
      'inner':[{'train':record_support(main,s['train']),'validation':record_support(main,s['validation'])}for s in p['inner']]}for i,p in enumerate(plans)])
    journal=Path(args.journal);done={}
    if journal.exists():
        with journal.open(encoding='utf8') as f:
            for line in f:r=json.loads(line);done[r['key']]=r
    if args.reuse:
        if not args.refine:raise RuntimeError('Reuse requires explicit train-only refinement')
        source=Path(args.reuse);previous=json.loads((source/'protocol.json').read_text(encoding='utf8'))
        for key in ['variants','grid','thresholds','plans','records','manual']:
            if protocol[key]!=previous[key]:raise RuntimeError('Cannot reuse different '+key)
        previous_fits=json.loads((source/'fits.json').read_text(encoding='utf8'))
        if args.continue_exhausted:
            from convergence import expanded_budget_policy
            if previous['id']!='7k-mil-fixed-original-segmentation-r3'or not expanded_budget_policy(previous['optimizer'],fit_policy):raise RuntimeError('Only an expanded cap with identical r3 stopping criteria can reuse stable fits')
            valid={'small_gradient','small_gradient_reused','train_prediction_stable'}
            eligible={r['key']for r in previous_fits if r['fit']['refinement']['status']in valid and
                      ('localStage'not in r['fit']or r['fit']['localStage']['refinement']['status']in valid)}
        else:eligible={r['key']for r in previous_fits if r['fit']['gradientMax']<=POLICY['gradientMax']
                       and r['variant']not in ['strong_only_prior','copied_prior']}
        with Path(args.reuse_journal).open(encoding='utf8')as f,journal.open('a',encoding='utf8')as destination:
            for line in f:
                r=json.loads(line)
                if r['key']not in eligible or r['key']in done:continue
                r['reusedFrom']={'directory':str(source),'sourceCommit':previous['sourceCommit'],
                  'reason':'already satisfies identical training stop criteria'}
                if not args.continue_exhausted:
                    r['fit']['refinement']={'policy':POLICY,'status':'small_gradient_reused','chunks':[],
                      'loss':r['fit']['loss'],'gradientMax':r['fit']['gradientMax']}
                done[r['key']]=r;destination.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    jobs=[(v,o,i,c) for v in variants for o in range(6) for i in range(3) for c in range(len(configs(v))) if '/'.join(map(str,(v,o,i,c))) not in done]
    print('pending fits',len(jobs),'workers',args.workers,flush=True)
    with journal.open('a',encoding='utf8') as f,concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(plans,args.refine,args.states,fit_policy,args.continue_exhausted,args.reuse_states)) as pool:
        futures={pool.submit(run,j):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            r=future.result();done[r['key']]=r;f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            if len(done)%40==0:print('completed',len(done),r['variant'],'elapsed fit',round(r['fit']['seconds'],2),flush=True)
    # Materialize complete per-fit summaries and pooled per-configuration scores.
    fit_summaries=[];search=[];selections=[]
    for r in done.values():
        if r['variant'] not in variants:continue
        val=[main[i]for i in r['mapIndices']]
        fit_summaries.append({k:v for k,v in r.items() if k not in ['mapIndices','mapProbability']}|
          {'mapDiagnosticAt035':map_metrics(val,np.array(r['mapProbability']),{'fine':.35,'mix':.35})})
    dump(OUT/'fits.json',fit_summaries)
    if args.continue_exhausted:
        exhausted=[r['key']for r in done.values()if r['fit']['refinement']['status']=='budget_exhausted']
        if exhausted:
            dump(OUT/'STOPPED.json',{'reason':'training safety cap still exhausted; no final selection or outer opening','keys':exhausted})
            raise RuntimeError('Unresolved train optimization: '+str(len(exhausted))+' fits; see STOPPED.json')
    for variant in variants:
      for outer in range(6):
        best=None
        for cid,cfg in enumerate(configs(variant)):
            rs=[done[f'{variant}/{outer}/{inner}/{cid}'] for inner in range(3)]
            indices=[i for r in rs for i in r['mapIndices']];val=[main[i]for i in indices]
            p=np.array([v for r in rs for v in r['mapProbability']])
            strong=[s for r in rs for s in r['segmentMetrics']['charts']]
            acc=float(np.mean([s['accuracy']for s in strong]))
            for threshold in THRESHOLDS:
                metrics=map_metrics(val,p,threshold);score=.5*(metrics['macroF1']+acc)
                row={'variant':variant,'outer':outer,'configId':cid,'config':cfg,'threshold':threshold,'score':score,
                  'map':metrics,'segmentChartAccuracy':acc,'segmentCharts':strong}
                search.append(row);rank=(score,acc,metrics['macroF1'])
                if best is None or rank>best[0]:best=(rank,row)
        selections.append(best[1])
    dump(OUT/'search.json',search);dump(OUT/'selection.json',{'outerOpened':False,'selections':selections})
    outerpred=[];outersegment=[];outerfits=[];reused_outer=0
    if args.continue_exhausted:
        old_selections={(s['variant'],s['outer']):s for s in json.loads((source/'selection.json').read_text(encoding='utf8'))['selections']}
        old_outer=json.loads((source/'outer-predictions.json').read_text(encoding='utf8'));old_segments=json.loads((source/'outer-segments.json').read_text(encoding='utf8'))
        old_fits={(r['variant'],r['outer']):r for r in json.loads((source/'outer-fits.json').read_text(encoding='utf8'))}
    for selection in selections:
        v,o,cfg,t=selection['variant'],selection['outer'],selection['config'],selection['threshold'];split=plans[o]
        train=[main[i]for i in split['train']]+[manual[i]for i in split['strongTrain']]
        test=[main[i]for i in split['test']];strong=[manual[i]for i in split['strongTest']]
        if args.continue_exhausted:
            old=old_selections[(v,o)];prior=old_fits[(v,o)]
            if old['config']==cfg and old['threshold']==t and prior['fit']['refinement']['status']in valid:
                reuse={'directory':str(source),'reason':'same outer train/test scope, identical inner-selected config and threshold; previous optimizer already accepted'}
                saved=json.loads((source/'models'/f'{v}-{o}.json').read_text(encoding='utf8'));dump(OUT/'models'/f'{v}-{o}.json',{**saved,'reusedFrom':reuse})
                outerpred.extend(r for r in old_outer if r['variant']==v and r['outer']==o)
                outersegment.extend(r for r in old_segments if r['variant']==v and r['outer']==o);outerfits.append({**prior,'reusedFrom':reuse});reused_outer+=1
                dump(OUT/'outer-predictions.json',outerpred);dump(OUT/'outer-segments.json',outersegment);dump(OUT/'outer-fits.json',outerfits)
                print('outer reused',v,o,flush=True);continue
        model,packed,trace=fit_candidate(train,v,cfg,'outer/'+v+'/'+str(o),args.refine,args.states,fit_policy)
        if args.continue_exhausted and trace['refinement']['status']=='budget_exhausted':raise RuntimeError('New outer fit exhausted its training cap; do not accept '+v+'/'+str(o))
        prob,_,bag=predict(model,packed,test,cfg);_,local,_=predict(model,packed,strong,cfg)
        pred=predict_tags(prob,t);seg=segment_metrics(strong,local)
        save_model(OUT/'models'/f'{v}-{o}.json',model,packed,cfg,t,{'outer':o,'trainUids':[r['uid']for r in train],
          'testUids':[r['uid']for r in test+strong],'selection':'inner CV only'})
        for j,r in enumerate(test):outerpred.append({'variant':v,'outer':o,'uid':r['uid'],'group':r['group'],'truth':target(r),
          'weight':r['recordWeight'],'predicted':[c for c,b in zip(MAP_CLASSES,pred[j])if b],
          'probability':prob[j].tolist(),'composition':bag[j].tolist()})
        for r,p in zip(strong,local):
            for s,probability in zip(r['segments'],p):outersegment.append({'variant':v,'outer':o,'uid':r['uid'],
              'start':s['start'],'end':s['end'],'tag':CLASSES[int(probability.argmax())],'probability':probability.tolist()})
        outerfits.append({'variant':v,'outer':o,'fit':trace,'map':map_metrics(test,prob,t),'segment':seg})
        print('outer scored',v,o,'map',round(outerfits[-1]['map']['macroF1'],4),'segment',round(seg['chartMeanAccuracy'],4),flush=True)
        dump(OUT/'outer-predictions.json',outerpred);dump(OUT/'outer-segments.json',outersegment);dump(OUT/'outer-fits.json',outerfits)
    dump(OUT/'complete.json',{'fits':len(fit_summaries),'outerFits':len(outerfits),'newOuterFits':len(outerfits)-reused_outer,'reusedOuterFits':reused_outer,'variants':variants,
      'scope':'fixed original segmentation only; no release or product replacement is implied'})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--variants')
    parser.add_argument('--output',default=str(OUT));parser.add_argument('--journal',default='.cache/rework-mil-stage1-r2-fits.jsonl')
    parser.add_argument('--refine',action='store_true');parser.add_argument('--states',default='.cache/rework-r3-states')
    parser.add_argument('--reuse');parser.add_argument('--reuse-journal',default='.cache/rework-mil-stage1-r2-fits.jsonl')
    parser.add_argument('--continue-exhausted',action='store_true');parser.add_argument('--max-chunks',type=int,default=64);parser.add_argument('--reuse-states')
    args=parser.parse_args()
    main_run(args)
