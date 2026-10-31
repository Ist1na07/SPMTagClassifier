"""Train-only original-family refits and a matched weak-supervision ablation.

All head configurations cross the final native gating/aggregation pool. No local
proxy selects a head. Raw boundaries are fixed here; the distinct native study
handles boundary coupling with frozen historical experts, not cold refits.
"""
from family_train import *
from family import classify,smooth,model_scores
from native_search import pack,summaries,decisions,f1_counts,BASE,GRID as NATIVE_GRID
from scipy.stats import qmc
import argparse,gzip,subprocess,concurrent.futures

OUT=Path('research/rework/results/family-joint-r2')
VARIANTS={**SPECS,**{f'rules_{r}':{'recipe':'rules','representation':r}for r in ['original','corrected']}}
_MAIN=_MANUAL=_META=_PLAN=None

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.gz':
        with gzip.open(path,'wt',encoding='utf8')as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    else:path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))

def joint_candidates(spec,budget=8192):
    grid={k:v for k,v in NATIVE_GRID.items()if k not in ['partition','head','burdenGate','smoothing']}
    grid={'modelId':[0]if spec['recipe']=='rules'else list(range(len(CONFIGS))),
          'lnRatio':[.35,.45,.55],'scorer':['rules']if spec['recipe']=='rules'else['model','blend'],
          'smoothing':['legacy','coherent'],**{k:v for k,v in grid.items()if k!='mode'},'mode':['legacy','normalized','soft']}
    found={}
    def add(c):
        # These are synthetic wrappers around one fixed raw partition and the
        # replaced expert pair, NOT native cache partition/head indices.
        c={**c,'partition':0,'head':0,'burdenGate':0.}
        c['excludeVibroFromMix']=c['mode']!='legacy'
        if c['mode']!='soft':c['minEvidence']=.25;c['minimumDiversity']=0.
        key=json.dumps(c,sort_keys=True)
        if key not in found:found[key]=c
    for model_id,ratio,scorer,display,mode in itertools.product(grid['modelId'],grid['lnRatio'],grid['scorer'],grid['smoothing'],grid['mode']):
        add({**BASE,'modelId':model_id,'lnRatio':ratio,'scorer':scorer,'smoothing':display,'mode':mode})
    if budget<len(found):raise ValueError('Budget omits required head/gating tuples')
    for point in qmc.Sobol(len(grid),scramble=True,seed=20261031).random_base2(16):
        if len(found)>=budget:break
        add({k:v[min(len(v)-1,int(point[j]*len(v)))]for j,(k,v)in enumerate(grid.items())})
    if len(found)<budget:raise ValueError('Insufficient distinct joint configurations')
    return list(found.values())[:budget]

def init(plans):
    global _MAIN,_MANUAL,_META,_PLAN
    _MAIN,_MANUAL,_=load();_META=load_metadata();_PLAN=plans

def predict_cases(records,models,spec,ratios,scorers):
    if spec['recipe']=='rules':precomputed=None
    else:
        x=np.array([s[spec['representation']]for r in records for s in r['segments']]);lengths=np.cumsum([len(r['segments'])for r in records])[:-1]
        probabilities={f:np.split(model_scores(models[f],x),lengths)for f in ['rc','ln']}
        precomputed=[{f:probabilities[f][i]for f in ['rc','ln']}for i in range(len(records))]
    result={}
    for ratio,scorer in itertools.product(ratios,scorers):
        values=[]
        for i,r in enumerate(records):
            raw=classify(r,_META[r['uid']],models,spec['representation'],ratio,scorer,precomputed[i]if precomputed else None)
            legacy=smooth(raw)
            # Burden is a new evidence field. Pool it over the actual legacy
            # display interval, rather than inheriting the first raw burden.
            for s in legacy:
                mass=total=0.
                for a in raw:
                    d=max(0,min(a['end'],s['end'])-max(a['start'],s['start']));mass+=d*a['burden'];total+=d
                s['burden']=mass/total if total else 0.
            values.append({'partitions':[{'cases':[{'raw':raw,'legacy':legacy}]}]})
        result[(ratio,scorer)]=values
    return result

def actual_segment_accuracy(record,rows):
    total=correct=0.
    for a in record['annotations']:
        if not a['tags']:continue
        total+=a['end']-a['start']
        correct+=sum(max(0,min(a['end'],s['end'])-max(a['start'],s['start']))for s in rows if s['tag']in a['tags'])
    return correct/max(1,total)

def run_job(job):
    variant,outer,budget,output=job;root=Path(output)/variant/str(outer);spec=VARIANTS[variant];plan=_PLAN[outer]
    if(root/'complete.json').exists():return {'variant':variant,'outer':outer,'resumed':True}
    configs=joint_candidates(spec,budget);write(root/'candidates.json.gz',configs)
    val_records=[_MAIN[i]for i in plan['train']]+[_MANUAL[i]for i in plan['strongTrain']]
    position={r['uid']:i for i,r in enumerate(val_records)};nmap=len(plan['train']);nstrong=len(plan['strongTrain'])
    predicted=np.zeros((len(configs),nmap,len(MAP_CLASSES)),dtype=np.uint8);accuracy=np.zeros((len(configs),nstrong))
    checkpoint=root/'inner-models.json.gz'
    if checkpoint.exists():
        with gzip.open(checkpoint,'rt',encoding='utf8')as f:models_saved=json.load(f)
    else:models_saved=[]
    fitted={(r['modelId'],r['inner']):r for r in models_saved}
    if len(fitted)!=len(models_saved):raise RuntimeError('Duplicate fitted head checkpoint')
    by_model=defaultdict(list)
    for i,c in enumerate(configs):by_model[c['modelId']].append(i)
    for model_id,indices in by_model.items():
        ratios=sorted({configs[i]['lnRatio']for i in indices});scorers=sorted({configs[i]['scorer']for i in indices})
        combined={(r,s):[None]*len(val_records)for r,s in itertools.product(ratios,scorers)}
        for inner,split in enumerate(plan['inner']):
            weak=[_MAIN[i]for i in split['train']];strong=[_MANUAL[i]for i in split['strongTrain']]
            validation=[_MAIN[i]for i in split['validation']]+[_MANUAL[i]for i in split['strongValidation']]
            saved=fitted.get((model_id,inner))
            expected=None if spec['recipe']=='rules'else CONFIGS[model_id]
            if saved is not None:
                if saved['config']!=expected:raise RuntimeError('Fitted head checkpoint configuration differs')
                models,trace=saved['models'],saved['fit']
            else:
                if spec['recipe']=='rules':models=None;trace={}
                else:models,trace=fit_pair(weak,strong,_META,spec,CONFIGS[model_id])
                saved={'inner':inner,'modelId':model_id,'config':expected,
                  'models':models,'fit':trace,'scope':'matching protocol outer/inner train and strongTrain only'}
                models_saved.append(saved);fitted[(model_id,inner)]=saved
                # Persist each completed fit BEFORE prediction/selection. A
                # downstream consumer error must not discard valid training.
                write(checkpoint,models_saved)
            values=predict_cases(validation,models,spec,ratios,scorers)
            for key,rows in values.items():
                for record,value in zip(validation,rows):combined[key][position[record['uid']]]=value
        for ratio,scorer,display in itertools.product(ratios,scorers,['legacy','coherent']):
            ids=[i for i in indices if configs[i]['lnRatio']==ratio and configs[i]['scorer']==scorer and configs[i]['smoothing']==display]
            if not ids:continue
            c=configs[ids[0]];data=pack(combined[(ratio,scorer)],c);cache={}
            strong_accuracy=np.array([actual_segment_accuracy(r,rows)for r,rows in zip(val_records[nmap:],data['chunks'][nmap:])])
            for i in ids:
                c=configs[i];key=tuple(c[k]for k in ['mode','effortPower','confidencePower'])
                if key not in cache:cache[key]=summaries(data,c)
                predicted[i]=decisions(cache[key],c)[:nmap];accuracy[i]=strong_accuracy
    y=encode(val_records[:nmap]);weight=np.array([r['recordWeight']for r in val_records[:nmap]])
    tp=(predicted*y[None,:,:]*weight[None,:,None]).sum(1);fp=(predicted*(1-y[None,:,:])*weight[None,:,None]).sum(1)
    fn=((1-predicted)*y[None,:,:]*weight[None,:,None]).sum(1);macro=f1_counts(tp,fp,fn);segment=accuracy.mean(1);joint=.5*(macro+segment)
    best=max(range(len(configs)),key=lambda i:(joint[i],segment[i],macro[i],-i));choice=configs[best]
    write(root/'selection.json',{'outerOpened':False,'candidate':best,'config':choice,
      'inner':{'joint':float(joint[best]),'mapMacroF1':float(macro[best]),'segmentChartAccuracy':float(segment[best])}})
    write(root/'search.json.gz',[{'candidate':i,'joint':float(joint[i]),'mapMacroF1':float(macro[i]),'segmentChartAccuracy':float(segment[i])}for i in range(len(configs))])
    np.savez_compressed(root/'inner-predictions.npz',map=predicted,segmentAccuracy=accuracy)
    weak=[_MAIN[i]for i in plan['train']];strong=[_MANUAL[i]for i in plan['strongTrain']]
    if spec['recipe']=='rules':models=None;trace={}
    else:models,trace=fit_pair(weak,strong,_META,spec,CONFIGS[choice['modelId']])
    write(root/'outer-model.json',{'models':models,'fit':trace,'config':choice,'headConfig':None if models is None else CONFIGS[choice['modelId']],
      'trainUids':[r['uid']for r in weak+strong],'scope':'cold refit on outer training fold after inner-only selection'})
    test=[_MAIN[i]for i in plan['test']]+[_MANUAL[i]for i in plan['strongTest']]
    values=predict_cases(test,models,spec,[choice['lnRatio']],[choice['scorer']])[(choice['lnRatio'],choice['scorer'])]
    data=pack(values,choice);summary=summaries(data,choice);p=decisions(summary,choice);rows=[];segments=[]
    for i,r in enumerate(test):
        if r['dataset']=='manual':segments.extend({'variant':variant,'outer':outer,'uid':r['uid'],**s}for s in data['chunks'][i])
        else:rows.append({'variant':variant,'outer':outer,'uid':r['uid'],'group':r['group'],'truth':target(r),'weight':r['recordWeight'],
               'predicted':[c for c,b in zip(MAP_CLASSES,p[i])if b],'composition':summary['distribution'][i].tolist()})
    write(root/'outer-predictions.json',rows);write(root/'outer-segments.json',segments)
    write(root/'complete.json',{'variant':variant,'outer':outer,'innerHeadPairs':len(models_saved)if spec['recipe']!='rules'else 0,
      'candidateBudget':len(configs),'coldStart':spec['recipe']!='rules'})
    return {'variant':variant,'outer':outer,'innerSelectedJoint':float(joint[best])}

def main_run(args):
    from search import plan
    main,manual,_=load();plans=plan(main,manual);root=Path(args.output);variants=args.variants.split(',')if args.variants else list(VARIANTS)
    protocol={'id':'family-cold-joint-r2','sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'sourceSnapshot':{n:(Path(__file__).parent/n).read_text(encoding='utf8')for n in ['family_search.py','family_train.py','family.py','family-meta-extract.cjs','native_search.py','mil.py']},
      'scope':'fixed original boundaries; cold two-family experts or no learned expert; retrospective grouped validation, not a new blind dataset',
      'variants':{v:VARIANTS[v]for v in variants},'fitGrid':FIT_GRID,'candidatesPerOuter':args.candidates,
      'selection':'.5 pooled inner map17 macroF1 + .5 equal-chart inner acceptable-label accuracy; all fitted head configs crossed with final pipeline choices, no local proxy filter',
      'weakLoss':'count_primary source-style count/primary recipe; copy_partial and bag_partial share data/scaler/strong loss and differ only E[-log p] versus -log E[p] for a weak chart label',
      'optimizer':'cold zero initialization; full-batch Adam <=250 then LBFGS up to4x1000, training-only gradient/prediction stopping; fit traces saved',
      'fixedPrior':'rule formulas, Break structure, and feature design arose in past research on these data; no historical trained expert weights/scalers initialize cold refits',
      'zeroRulePolicy':'an all-zero rule vector abstains as the routed family wildcard, not an arbitrary first dictionary type; actual errors remain counted',
      'strongCoverage':'definite-family acceptable sets use duration-overlap partial NLL; cross-family alternatives excluded ONLY from head training, not validation; source-style recipe uses primary >=60% overlap',
      'plans':plans,'records':[{'uid':r['uid'],'group':r['group'],'tags':target(r),'weight':r.get('recordWeight',1)}for r in main+manual]}
    path=root/'protocol.json'
    if path.exists()and json.loads(path.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Family study protocol changed; use a new revision')
    write(path,protocol);jobs=[(v,o,args.candidates,str(root))for v in variants for o in range(6)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(plans,))as pool:
        futures=[pool.submit(run_job,j)for j in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):print('cold family',future.result(),flush=True)
        except BaseException:
            for future in futures:future.cancel()
            raise
    for filename in ['outer-predictions.json','outer-segments.json']:
        allrows=[]
        for v in variants:
            for o in range(6):allrows.extend(json.loads((root/v/str(o)/filename).read_text(encoding='utf8')))
        write(root/filename,allrows)
    write(root/'complete.json',{'variants':variants,'outerSelections':len(jobs),'scope':protocol['scope']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--variants');p.add_argument('--candidates',type=int,default=8192)
    p.add_argument('--output',default=str(OUT));main_run(p.parse_args())
