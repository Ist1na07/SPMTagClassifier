"""Exhaustive declared discrete grid; head and pool jointly trained per fit.

One owner writes the journal and final selections. Inner checkpoints preserve
every fit; the public evidence keeps all choices, hard validation predictions
and traces, with full outer models/probabilities. Raw notes stay local.
"""
import argparse,gzip,itertools,concurrent.futures,subprocess,shutil
import coarse_model as model
from coarse_data import load_headers,plans,CLASSES,FINE_TO_COARSE
from mil import np,json,Path,Counter,torch
from sklearn.metrics import f1_score,accuracy_score

GRID={'windowMs':[2000,4000,8000],'burdenGate':[0.,.2],'localL2':[.0003,.003,.03],
      'strongWeight':[.5,4.],'source4Weight':[.25,.5,.75]}
_WEAK=_MANUAL=_PLANS=_ROOT=_CHECK=None

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    if path.suffix=='.gz':
        with gzip.open(temporary,'wt',encoding='utf8')as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    else:temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)

def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf8')as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf8'))

def configurations(variant):
    grid={**GRID};spec=model.SPECS[variant]
    if spec.get('weak4Only'):grid.update(strongWeight=[0.],source4Weight=[1.])
    if spec.get('weakDomain'):grid['source4Weight']=[float(spec['weakDomain']==4)]
    return [dict(zip(grid,v))for v in itertools.product(*grid.values())]

def source_metric(rows,labels,K):
    ids=[i for i,r in enumerate(rows)if r['keys']==K]
    y=[model.FINE.index(rows[i]['coarseTags'][0])for i in ids];p=[int(labels[i])for i in ids]
    w=[rows[i]['recordWeight']for i in ids]
    return {'macroF1':float(f1_score(y,p,labels=list(range(5)),average='macro',sample_weight=w,zero_division=0)),
      'accuracy':float(accuracy_score(y,p,sample_weight=w)),'records':len(ids)}

def segment_metric(record,probabilities,config):
    ws=record['windows'][str(config['windowMs'])];labels=probabilities.argmax(1);correct=total=0.
    for a in record['annotations']:
        if not a['tags']:continue
        total+=a['end']-a['start']
        correct+=sum(max(0,min(w['end'],a['end'])-max(w['start'],a['start']))for j,w in enumerate(ws)if CLASSES[int(labels[j])]in a['tags'])
    return {'uid':record['uid'],'accuracy':correct/max(1,total),'correctMs':correct,'labelledMs':total}

def init(root,check):
    global _WEAK,_MANUAL,_PLANS,_ROOT,_CHECK
    _WEAK,_MANUAL,_=load_headers();_PLANS=plans(_WEAK,_MANUAL);_ROOT=Path(root);_CHECK=Path(check)

def fit_or_load(path,records,spec,cfg):
    scope=[r['uid']for r in model.training_rows(records,spec)]
    # These two declared ablation names have the same effective objective.
    # Reuse an already completed canonical fit, including outer refits.
    if not path.exists()and 'no_release'in path.parts and spec==model.SPECS['sequence44']:
        source=Path(*['sequence44'if p=='no_release'else p for p in path.parts])
        if source.exists():
            saved=read(source)
            if saved['config']!=cfg or saved['spec']!=spec or saved['trainUids']!=scope:raise RuntimeError('Equivalent fit scope differs')
            write(path,{**saved,'equivalentFitSource':str(source)})
    if path.exists():
        saved=read(path)
        if saved['config']!=cfg or saved['spec']!=spec or saved['trainUids']!=scope:raise RuntimeError('Coarse fit checkpoint scope differs')
        return saved['model'],saved['fit']
    fitted,trace=model.fit(records,spec,cfg)
    write(path,{'model':fitted,'fit':trace,'config':cfg,'spec':spec,'trainUids':scope})
    return fitted,trace

def run_inner(job):
    variant,outer,inner=job;directory=_ROOT/variant/str(outer)/str(inner);spec=model.SPECS[variant]
    if(directory/'complete.json').exists():return {'variant':variant,'outer':outer,'inner':inner,'resumed':True}
    if variant=='no_release'and spec==model.SPECS['sequence44']and configurations(variant)==configurations('sequence44'):
        source=_ROOT/'sequence44'/str(outer)/str(inner)
        if(source/'complete.json').exists():
            directory.mkdir(parents=True,exist_ok=True)
            for name in ['fits.json.gz','inner-predictions.npz']:shutil.copyfile(source/name,directory/name)
            for cid in range(len(configurations(variant))):
                saved=read(_CHECK/'sequence44'/str(outer)/str(inner)/(str(cid)+'.json.gz'))
                write(_CHECK/variant/str(outer)/str(inner)/(str(cid)+'.json.gz'),{**saved,'equivalentFitSource':str(_CHECK/'sequence44'/str(outer)/str(inner)/(str(cid)+'.json.gz'))})
            write(directory/'complete.json',{**read(source/'complete.json'),'equivalentFitReuse':len(configurations(variant)),'equivalentVariant':'sequence44'})
            return {'variant':variant,'outer':outer,'inner':inner,'equivalentFitsReused':len(configurations(variant))}
    split=_PLANS[outer]['inner'][inner];cfgs=configurations(variant)
    train=model.materialize([_WEAK[i]for i in split['train']]+[_MANUAL[i]for i in split['strongTrain']])
    validation=model.materialize([_WEAK[i]for i in split['validation']]);strong=model.materialize([_MANUAL[i]for i in split['strongValidation']])
    predicted=np.zeros((len(cfgs),len(validation)),dtype=np.uint8);accuracy=np.zeros((len(cfgs),len(strong)));traces=[]
    for cid,cfg in enumerate(cfgs):
        path=_CHECK/variant/str(outer)/str(inner)/(str(cid)+'.json.gz')
        fitted,trace=fit_or_load(path,train,spec,cfg)
        prediction=model.predict(fitted,validation)
        predicted[cid]=np.where(prediction['mass']>0,prediction['composition'].argmax(1),len(model.FINE))
        sp=model.predict(fitted,strong)
        metrics=[segment_metric(r,p,cfg)for r,p in zip(strong,sp['local'])];accuracy[cid]=[s['accuracy']for s in metrics]
        traces.append({'configId':cid,'config':cfg,'fit':trace,'pool':fitted['pool'],'segment':metrics})
    write(directory/'fits.json.gz',traces)
    np.savez_compressed(directory/'inner-predictions.npz',map=predicted,segmentAccuracy=accuracy)
    write(directory/'complete.json',{'fits':len(cfgs),'mapUids':[r['uid']for r in validation],'strongUids':[r['uid']for r in strong]})
    return {'variant':variant,'outer':outer,'inner':inner,'fits':len(cfgs)}

def select_outer(root,variant,outer,weak,manual,plan,criterion='shared'):
    configs=configurations(variant);p=[];acc=[];validation=[]
    for inner,s in enumerate(plan['inner']):
        directory=root/variant/str(outer)/str(inner)
        complete=read(directory/'complete.json')
        if complete['mapUids']!=[weak[i]['uid']for i in s['validation']]:raise RuntimeError('Inner validation ordering differs')
        with np.load(directory/'inner-predictions.npz')as saved:p.append(saved['map']);acc.append(saved['segmentAccuracy'])
        validation.extend(weak[i]for i in s['validation'])
    p=np.concatenate(p,axis=1);acc=np.concatenate(acc,axis=1);search=[]
    for cid,cfg in enumerate(configs):
        four=source_metric(validation,p[cid],4);seven=source_metric(validation,p[cid],7);segment=float(acc[cid].mean())
        score=four['macroF1']if criterion=='4k'else .25*four['macroF1']+.25*seven['macroF1']+.5*segment
        search.append({'configId':cid,'config':cfg,'score':score,'map4':four,'map7':seven,'segmentChartAccuracy':segment})
    if criterion=='4k':best=max(range(len(configs)),key=lambda i:(search[i]['score'],search[i]['map4']['accuracy'],-i))
    else:best=max(range(len(configs)),key=lambda i:(search[i]['score'],search[i]['segmentChartAccuracy'],search[i]['map4']['macroF1'],search[i]['map7']['macroF1'],-i))
    return search[best],search

def finish(root,check,variants,weak,manual,ps):
    expanded=[(v,v,'4k'if v=='weak4_only'else'shared')for v in variants]
    if'weak4_only'in variants:expanded.append(('weak4_only_shared_selection','weak4_only','shared'))
    outer_rows=[];outer_segments=[]
    for name,variant,criterion in expanded:
        for outer,split in enumerate(ps):
            directory=root/name/str(outer);selection_path=directory/'selection.json'
            choice,search=select_outer(root,variant,outer,weak,manual,split,criterion)
            write(directory/'search.json.gz',search);write(selection_path,{'outerOpened':False,'criterion':criterion,**choice})
            train=model.materialize([weak[i]for i in split['train']]+[manual[i]for i in split['strongTrain']])
            cfg=choice['config'];spec=model.SPECS[variant]
            fitted,trace=fit_or_load(check/'outer'/variant/str(outer)/(str(choice['configId'])+'.json.gz'),train,spec,cfg)
            write(directory/'outer-model.json',{'model':fitted,'fit':trace,'trainUids':[r['uid']for r in model.training_rows(train,spec)],'selection':'matching inner folds only'})
            if trace['status'] not in ['small_projected_gradient','train_prediction_stable']:
                write(root/'STOPPED.json',{'reason':'selected outer fit has not met the declared training stop condition','variant':name,'outer':outer,'fit':trace})
                raise RuntimeError('Unaccepted selected outer optimizer fit; predictions not opened')
            test=model.materialize([weak[i]for i in split['test']]);strong=model.materialize([manual[i]for i in split['strongTest']])
            prediction=model.predict(fitted,test);sp=model.predict(fitted,strong)
            rows=[{'variant':name,'outer':outer,'uid':r['uid'],'group':r['group'],'keys':r['keys'],'truth':r['coarseTags'][0],
              'predicted':model.FINE[int(p.argmax())]if p.sum()>0 else None,'composition':p.tolist(),'weight':r['recordWeight']}for r,p in zip(test,prediction['composition'])]
            segments=[]
            for r,p,weights in zip(strong,sp['local'],sp['weights']):
                for w,probability,weight in zip(r['windows'][str(cfg['windowMs'])],p,weights):
                    segments.append({'variant':name,'outer':outer,'uid':r['uid'],'start':w['start'],'end':w['end'],
                      'tag':CLASSES[int(probability.argmax())],'probability':probability.tolist(),'aggregationWeight':float(weight)})
            write(directory/'outer-predictions.json',rows);write(directory/'outer-segments.json',segments)
            outer_rows.extend(rows);outer_segments.extend(segments)
            print('coarse outer',name,outer,flush=True)
    write(root/'outer-predictions.json',outer_rows);write(root/'outer-segments.json',outer_segments)
    equivalent_reuse=sum(read(root/v/str(o)/str(i)/'complete.json').get('equivalentFitReuse',0)for v in variants for o in range(6)for i in range(3))
    write(root/'complete.json',{'variants':[n for n,v,c in expanded],'innerFits':sum(len(configurations(v))*18 for v in variants),'equivalentInnerFitsReused':equivalent_reuse,
      'outerSelections':len(expanded)*6,'scope':'complete declared discrete grid; jointly trained local head/pool; retrospective grouped cross-key validation'})

def run(args):
    weak,manual,allrows=load_headers();ps=plans(weak,manual);root=Path(args.output);check=Path(args.checkpoints)
    variants=args.variants.split(',')if args.variants else list(model.SPECS)
    protocol={'id':'coarse-shared-joint-r3','sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'sourceSnapshot':{n:(Path(__file__).parent/n).read_text(encoding='utf8')for n in ['coarse_search.py','coarse_model.py','coarse_data.py','physical.cjs','coarse-extract.cjs']},
      'featureSpecification':model.SCHEMA,'scope':'4K/7K shared five-class coarse morphology; retrospective grouped CV, no human 6K pattern accuracy',
      'grid':GRID,'configs':{v:configurations(v)for v in variants},'variants':{v:model.SPECS[v]for v in variants},
      'selection':'.25 inner 4K five-class macroF1 + .25 inner 7K five-class macroF1 + .5 equal-chart 7K coarse segment accuracy',
      'weak4Control':'same weak4_only fits: 4K-only selection baseline and shared-objective transfer selection separately; no repeated training for identical outer choice',
      'continuousFit':'five-way zero-initialized softmax W/b and bounded effort/concentration exponents [0,1] fitted on same training-only bag/partial-label objective',
      'equivalentFitReuse':'sequence44 and no_release have identical effective specs and grids; completed canonical inner predictions/fits and matching outer fits are copied with explicit lineage, not trained twice',
      'optimizer':'full-batch L-BFGS-B up to3x1000; projected gradient<=1e-5 or two stable training-loss/prediction checks; refuse selection until every inner fit is accepted; selected outer fits must also be accepted',
      'runtimeRepair':'one active-count validation pass replaces repeated whole-array conversion; unchanged packed values/objective/gradient. Existing valid fits retained; packing-equivalence.json records the actual whole-fold comparison. Atomic checkpoint publication added.',
      'releaseOrderRepair':'unique timestamp groups, not sorted chord endpoints, define sameHandOrder. Updated coarse-r2 cache records all changed windows and bitwise invariance of every other coordinate. Prior fits reused only where training inputs are unchanged; validation predictions regenerated on the repaired representation.',
      'plans':ps,'records':[{'uid':r['uid'],'group':r['group'],'keys':r['keys'],'tags':r['coarseTags'],'weight':r['recordWeight']}for r in weak],
      'manual':[{'uid':r['uid'],'group':r['group'],'annotations':r['annotations']}for r in manual]}
    path=root/'protocol.json'
    if path.exists()and read(path)!=protocol:raise RuntimeError('Coarse study protocol changed; use a new revision')
    write(path,protocol)
    jobs=[(v,o,i)for v in variants for o in range(6)for i in range(3)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(str(root),str(check)))as pool:
        futures=[pool.submit(run_inner,j)for j in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):print('coarse inner',future.result(),flush=True)
        except BaseException:
            for future in futures:future.cancel()
            raise
    unaccepted=[]
    for variant,outer,inner in jobs:
        for row in read(root/variant/str(outer)/str(inner)/'fits.json.gz'):
            if row['fit']['status'] not in ['small_projected_gradient','train_prediction_stable']:
                unaccepted.append({'variant':variant,'outer':outer,'inner':inner,'configId':row['configId'],'fit':row['fit']})
    if unaccepted:
        write(root/'STOPPED.json',{'reason':'inner fits have not met the declared training stop condition; no outer selection opened','fits':unaccepted})
        raise RuntimeError(f'{len(unaccepted)} inner optimizer fits unaccepted; final selection refused')
    finish(root,check,variants,weak,manual,ps)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--variants')
    p.add_argument('--output',default='research/rework/results/coarse-shared-r3');p.add_argument('--checkpoints',default='.cache/rework-coarse-shared-r3-fits')
    run(p.parse_args())
