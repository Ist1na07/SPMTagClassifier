"""Common complete head x post-policy design for a small anchored LN update.

Every fitted head crosses the SAME 512-policy set. No proxy filters heads and
every updating representation also has the exact zero-update head available.
"""
from mil import np,torch,json,Path,load,target,encode,MAP_CLASSES,Counter,defaultdict,itertools
import ln_adapter as adapter
from family import classify,load_metadata,model_scores
from native_search import BASE,GRID,pack,summaries,decisions,f1_counts
from scipy.stats import qmc
import argparse,gzip,concurrent.futures,subprocess

FIT_GRID={'localL2':[.01,.1,1.,10.,100.],'strongWeight':[.25,1.,4.],'balance':['none','balanced']}
CONFIGS=[dict(zip(FIT_GRID,v))for v in itertools.product(*FIT_GRID.values())]
VARIANTS=['geometry3','no_isolation','no_hold','no_interaction','bias','old_summary','geometry9','no_update']
_MAIN=_MANUAL=_META=_PLANS=_ROOT=_CHECK=_RC=_LN=None


def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf8')as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    if path.suffix=='.gz':
        with gzip.open(tmp,'wt',encoding='utf8')as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    else:tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(path)


def policies(n=512):
    grid={'lnRatio':[.35,.45,.55],'scorer':['model','blend'],'burdenGate':GRID['burdenGate'],
          **{k:v for k,v in GRID.items()if k not in ['partition','head','smoothing','mode','burdenGate']},'mode':['normalized','soft']}
    found={}
    def add(c):
        c={**BASE,**c,'partition':0,'head':0,'smoothing':'coherent','excludeVibroFromMix':True}
        if c['mode']!='soft':c['minEvidence']=.25;c['minimumDiversity']=0.
        found.setdefault(json.dumps(c,sort_keys=True),c)
    for ratio,scorer,gate,mode in itertools.product(grid['lnRatio'],grid['scorer'],grid['burdenGate'],grid['mode']):
        add({'lnRatio':ratio,'scorer':scorer,'burdenGate':gate,'mode':mode})
    if n<len(found):raise ValueError('Policy budget omits mandatory routing tuples')
    for point in qmc.Sobol(len(grid),scramble=True,seed=20261031).random_base2(14):
        if len(found)>=n:break
        add({k:v[min(len(v)-1,int(point[j]*len(v)))]for j,(k,v)in enumerate(grid.items())})
    if len(found)!=n:raise RuntimeError('Unique common policy count differs')
    return list(found.values())


def init(root,check,plans):
    global _MAIN,_MANUAL,_META,_PLANS,_ROOT,_CHECK,_RC,_LN
    _MAIN,_MANUAL,_=load();_META=load_metadata();_PLANS=plans;_ROOT=Path(root);_CHECK=Path(check);_RC={};_LN={}
    for r in _MAIN+_MANUAL:
        x=np.array([s['original']for s in r['segments']]);_RC[r['uid']]=model_scores(adapter.ARCHIVE['rc'],x);_LN[r['uid']]=model_scores(adapter.ARCHIVE['ln'],x)


def zero_model(variant):
    v=variant if variant!='no_update'else'bias';keys=adapter.SPECS[v]
    return {'variant':v,'keys':keys,'mean':[0.]*len(keys),'scale':[1.]*len(keys),'delta':[0.]*(len(keys)+1),'clip':6.,'frozenNoUpdate':True}


def fit_or_load(path,weak,manual,variant,config):
    scope=[r['uid']for r in weak+manual]
    if path.exists():
        value=read(path)
        if value['scope']!=scope or value['variant']!=variant or value['config']!=config:raise RuntimeError('LN fit checkpoint scope differs')
        if value['fit']['status']!='small_gradient':raise RuntimeError('Stored LN adapter has not met the gradient guard')
        return value['model'],value['fit']
    model,trace=adapter.fit(weak,manual,variant,config);write(path,{'variant':variant,'config':config,'scope':scope,'model':model,'fit':trace})
    if trace['status']!='small_gradient':raise RuntimeError('LN adapter training gradient guard not met')
    return model,trace


def cases(records,ln_probabilities,ratio,scorer):
    result=[]
    for r,ln in zip(records,ln_probabilities):
        raw=classify(r,_META[r['uid']],adapter.ARCHIVE,ln_ratio=ratio,scorer=scorer,precomputed={'rc':_RC[r['uid']],'ln':ln})
        result.append({'partitions':[{'cases':[{'raw':raw}]}]})
    return result


def segment_accuracy(record,rows):
    total=sum(a['end']-a['start']for a in record['annotations']if a['tags'])
    correct=sum(max(0,min(a['end'],s['end'])-max(a['start'],s['start']))for a in record['annotations']if a['tags']for s in rows if s['tag']in a['tags'])
    return correct/max(1,total)


def run_job(job):
    variant,o,budget=job;root=_ROOT/variant/str(o);plan=_PLANS[o]
    if(root/'complete.json').exists():return {'variant':variant,'outer':o,'resumed':True}
    post=policies(budget);head_count=1 if variant=='no_update'else len(CONFIGS)+1
    validation=[_MAIN[i]for i in plan['train']]+[_MANUAL[i]for i in plan['strongTrain']];position={r['uid']:i for i,r in enumerate(validation)}
    nmap=len(plan['train']);nstrong=len(plan['strongTrain']);pred=np.zeros((head_count*len(post),nmap,len(MAP_CLASSES)),dtype=np.uint8);acc=np.zeros((len(pred),nstrong));traces=[]
    local_jobs=defaultdict(list)
    for i,c in enumerate(post):local_jobs[(c['lnRatio'],c['scorer'],c['burdenGate'])].append(i)
    for head in range(head_count):
        probabilities=[None]*len(validation)
        for inner,split in enumerate(plan['inner']):
            weak=[_MAIN[i]for i in split['train']];manual=[_MANUAL[i]for i in split['strongTrain']]
            records=[_MAIN[i]for i in split['validation']]+[_MANUAL[i]for i in split['strongValidation']]
            if head==0:model=zero_model(variant);trace={'status':'frozen_archived_prior'};local=[_LN[r['uid']]for r in records]
            else:
                model,trace=fit_or_load(_CHECK/variant/str(o)/str(inner)/(str(head-1)+'.json.gz'),weak,manual,variant,CONFIGS[head-1]);local=adapter.predict(model,records)
                traces.append({'head':head,'inner':inner,'configId':head-1,'fit':trace})
            for r,p in zip(records,local):probabilities[position[r['uid']]]=p
        for ratio,scorer in itertools.product([.35,.45,.55],['model','blend']):
            values=cases(validation,probabilities,ratio,scorer)
            for gate in GRID['burdenGate']:
                indices=local_jobs[(ratio,scorer,gate)];c=post[indices[0]];data=pack(values,c);cache={}
                local_accuracy=np.array([segment_accuracy(r,s)for r,s in zip(validation[nmap:],data['chunks'][nmap:])])
                for j in indices:
                    c=post[j];key=tuple(c[k]for k in ['mode','effortPower','confidencePower'])
                    if key not in cache:cache[key]=summaries(data,c)
                    at=head*len(post)+j;pred[at]=decisions(cache[key],c)[:nmap];acc[at]=local_accuracy
        print('ln head',variant,o,head,'/',head_count-1,flush=True)
    y=encode(validation[:nmap]);w=np.array([r['recordWeight']for r in validation[:nmap]])
    macro=f1_counts((pred*y[None,:,:]*w[None,:,None]).sum(1),(pred*(1-y[None,:,:])*w[None,:,None]).sum(1),((1-pred)*y[None,:,:]*w[None,:,None]).sum(1))
    segment=acc.mean(1);joint=.5*(macro+segment)
    at=max(range(len(pred)),key=lambda i:(joint[i],segment[i],macro[i],-i));head,policy=divmod(at,len(post));choice=post[policy]
    write(root/'selection.json',{'selectionUsesOuterLabels':False,'head':head,'fitConfig':CONFIGS[head-1]if head else None,'policyId':policy,'policy':choice,
      'inner':{'joint':float(joint[at]),'mapMacroF1':float(macro[at]),'segmentAccuracy':float(segment[at])}})
    write(root/'search.json.gz',[{'head':i//len(post),'policy':i%len(post),'joint':float(joint[i]),'mapMacroF1':float(macro[i]),'segmentAccuracy':float(segment[i])}for i in range(len(pred))])
    np.savez_compressed(root/'inner-predictions.npz',map=pred,segmentAccuracy=acc);write(root/'inner-fits.json.gz',traces)
    weak=[_MAIN[i]for i in plan['train']];manual=[_MANUAL[i]for i in plan['strongTrain']]
    if head:model,trace=fit_or_load(_CHECK/'outer'/variant/str(o)/(str(head-1)+'.json.gz'),weak,manual,variant,CONFIGS[head-1])
    else:model=zero_model(variant);trace={'status':'frozen_archived_prior'}
    write(root/'outer-model.json',{'model':model,'fit':trace,'policy':choice,'trainUids':[r['uid']for r in weak+manual]})
    test=[_MAIN[i]for i in plan['test']]+[_MANUAL[i]for i in plan['strongTest']];local=adapter.predict(model,test)
    values=cases(test,local,choice['lnRatio'],choice['scorer']);data=pack(values,choice);p=decisions(summaries(data,choice),choice);rows=[];segments=[]
    for i,r in enumerate(test):
        if r['dataset']=='manual':segments.extend({'variant':variant,'outer':o,'uid':r['uid'],**s}for s in data['chunks'][i])
        else:rows.append({'variant':variant,'outer':o,'uid':r['uid'],'group':r['group'],'truth':target(r),'weight':r['recordWeight'],
                         'predicted':[c for c,b in zip(MAP_CLASSES,p[i])if b]})
    write(root/'outer-predictions.json',rows);write(root/'outer-segments.json',segments)
    write(root/'complete.json',{'learnedInnerFits':len(traces),'jointCandidates':len(pred),'zeroHeadAvailable':True})
    return {'variant':variant,'outer':o,'selectedHead':head,'inner':float(joint[at])}


def run(args):
    from search import plan
    main,manual,_=load();plans=plan(main,manual);root=Path(args.output);check=Path(args.checkpoints)
    variants=args.variants.split(',')if args.variants else VARIANTS
    protocol={'id':'ln-anchored-release-r1','scope':'fixed original raw automatic boundaries; small conditional Release logit update, archived RC/LN prior exposure unknown; retrospective grouped CV',
      'sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sourceSnapshot':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['ln_adapter.py','ln_adapter_search.py','family.py','native_search.py','mil.py']},
      'archivedExperts':adapter.ARCHIVE,'variants':{v:adapter.SPECS.get(v,[])for v in variants},'fitGrid':FIT_GRID,'postPolicies':policies(args.policies),
      'selection':f'ALL30 fitted update configs plus exact zero head cross ALL common{args.policies} post policies, final .5 map17macroF1 + .5 equal-chart segmentaccuracy; no proxy head selection',
      'plans':plans,'records':[{'uid':r['uid'],'group':r['group'],'tags':target(r),'weight':r.get('recordWeight',1)}for r in main+manual],
      'geometrySpecification':read('.cache/rework-release-geometry-r2-schema.json'),'primary':'geometry3','zeroBaseline':'exact archived head, available within every updating candidate pool'}
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=protocol:raise RuntimeError('LN study protocol changed')
    write(root/'protocol.json',protocol)
    jobs=[(v,o,args.policies)for v in variants for o in range(6)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(str(root),str(check),plans))as pool:
        try:
            for result in pool.map(run_job,jobs):print('ln outer',result,flush=True)
        except BaseException as e:
            write(root/'STOPPED.json',{'reason':str(e),'validCheckpointsRetained':str(check)});raise
    for filename in ['outer-predictions.json','outer-segments.json']:
        rows=[]
        for variant in variants:
            for o in range(6):rows.extend(read(root/variant/str(o)/filename))
        write(root/filename,rows)
    write(root/'complete.json',{'variants':variants,'outerSelections':len(jobs),'learnedInnerFits':sum(0 if v=='no_update'else 30*18 for v in variants)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--policies',type=int,default=512);p.add_argument('--variants')
    p.add_argument('--output',default='research/rework/results/ln-adapter-r1');p.add_argument('--checkpoints',default='.cache/rework-ln-adapter-r1-fits');run(p.parse_args())
