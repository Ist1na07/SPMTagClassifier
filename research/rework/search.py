"""Nested grouped, FULL JOINT grid for the shared local/aggregation model.

Stage 1 intentionally fixes the original segmentation. Segmenter coupling is
studied separately; neither this script nor its report may call it full-pipeline
validation. No local proxy preselects a candidate.
"""
import argparse,concurrent.futures,subprocess,traceback
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
def init_worker(plan):
    global _MAIN,_MANUAL,_PLAN
    _MAIN,_MANUAL,_=load();_PLAN=plan
def run(job):
    variant,outer,inner,cid=job;split=_PLAN[outer]['inner'][inner]
    train=[_MAIN[i] for i in split['train']]+[_MANUAL[i] for i in split['strongTrain']]
    val=[_MAIN[i] for i in split['validation']];strong=[_MANUAL[i] for i in split['strongValidation']]
    cfg=configs(variant)[cid];start=time.perf_counter()
    model,packed,fittrace=fit(train,variant,cfg)
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
    jobs=[(v,o,i,c) for v in variants for o in range(6) for i in range(3) for c in range(len(configs(v))) if '/'.join(map(str,(v,o,i,c))) not in done]
    print('pending fits',len(jobs),'workers',args.workers,flush=True)
    with journal.open('a',encoding='utf8') as f,concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(plans,)) as pool:
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
    outerpred=[];outersegment=[];outerfits=[]
    for selection in selections:
        v,o,cfg,t=selection['variant'],selection['outer'],selection['config'],selection['threshold'];split=plans[o]
        train=[main[i]for i in split['train']]+[manual[i]for i in split['strongTrain']]
        test=[main[i]for i in split['test']];strong=[manual[i]for i in split['strongTest']]
        model,packed,trace=fit(train,v,cfg)
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
    dump(OUT/'complete.json',{'fits':len(fit_summaries),'outerFits':len(outerfits),'variants':variants,
      'scope':'fixed original segmentation only; no release or product replacement is implied'})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--variants')
    parser.add_argument('--output',default=str(OUT));parser.add_argument('--journal',default='.cache/rework-mil-stage1-r2-fits.jsonl');args=parser.parse_args()
    main_run(args)
