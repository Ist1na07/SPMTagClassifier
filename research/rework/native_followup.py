"""Complete the missing factorial cell and evaluate a semantically valid pool.

Uses the native producer/consumer unchanged. Processes one source at a time so
the expanded joint pool does not require eight native corpora resident in RAM.
"""
import argparse,gzip,subprocess
import native_search as native
from native_search import pack,summaries,decisions,f1_counts,BASE,GRID
from mil import *
from scipy.stats import qmc

PARENT=Path('research/rework/results/native-joint-r1')
EXTRA=Path('.cache/rework-native-factorial-r1')
STUDY_ID='native-factorial-repaired-r1'
SCOPE='retrospective followup prompted by observed native full-combination regression; frozen historical expert exposure unknown'
SOURCE_FILES=['native_followup.py','native_search.py','native-factorial-extract.cjs','family.py']
CATALOGUE={'logical_neutral_scale':{'root':str(EXTRA),'id':'native-factorial-r1'}}
EXTRA_CACHE_SPECIFICATIONS={}
SOURCE_PARAMETERS={}
SOURCES=['logical_neutral','logical_neutral_scale']+[f'{p}_{s}'for p in ['logical_release','full']for s in [20,40,80]]
GROUPS={
 'factorial_scale':{'sources':['logical_neutral_scale'],'historicalControl':True},
 'repaired':{'sources':SOURCES},
 'repaired_no_release':{'sources':['logical_neutral','logical_neutral_scale']},
 'repaired_no_scale':{'sources':['logical_neutral']+[f'logical_release_{s}'for s in [20,40,80]]},
 'repaired_normalized':{'sources':SOURCES,'fixed':{'mode':'normalized'}},
 'repaired_soft':{'sources':SOURCES,'fixed':{'mode':'soft'}},
 'repaired_no_effort':{'sources':SOURCES,'fixed':{'effortPower':0.}},
 'repaired_no_confidence':{'sources':SOURCES,'fixed':{'confidencePower':0.}},
 'repaired_no_diversity':{'sources':SOURCES,'fixed':{'minimumDiversity':0.}},
}

def candidates(name,budget=8192):
    spec=GROUPS[name]
    if spec.get('historicalControl'):
        return [{**c,'source':'logical_neutral_scale'}for c in native.candidates('logical_neutral',budget)]
    fixed={'smoothing':'coherent','excludeVibroFromMix':True,**spec.get('fixed',{})}
    grid={'source':spec['sources'],**GRID,'smoothing':['coherent'],'mode':['normalized','soft']}
    found={}
    def add(c):
        c={**c,**fixed}
        if c['mode']!='soft':c['minEvidence']=.25;c['minimumDiversity']=0.
        key=json.dumps(c,sort_keys=True)
        if key not in found:found[key]=c
    for source,partition,head,gate,mode in itertools.product(spec['sources'],GRID['partition'],GRID['head'],GRID['burdenGate'],grid['mode']):
        add({**BASE,'source':source,'partition':partition,'head':head,'burdenGate':gate,'mode':mode})
    if budget<len(found):raise ValueError('Budget cannot cover mandatory native/gating/mode tuples')
    for point in qmc.Sobol(len(grid),scramble=True,seed=20261031).random_base2(16):
        if len(found)>=budget:break
        add({k:v[min(len(v)-1,int(point[j]*len(v)))]for j,(k,v)in enumerate(grid.items())})
    if len(found)<budget:raise RuntimeError('Not enough unique joint candidates')
    return list(found.values())[:budget]

def load_source(name,records):
    if name not in CATALOGUE:return native.load_source(name,records)
    spec=CATALOGUE[name]
    out=[]
    for r in records:
        with gzip.open(Path(spec['root'])/name/(r['uid'].replace('/','__')+'.json.gz'),'rt',encoding='utf8')as f:value=json.load(f)
        if value['id']!=spec['id']or value['uid']!=r['uid']or value['treatment']!=name:raise RuntimeError('Followup source identity differs')
        out.append(value)
    return out

def choose(predictions,accuracy,main,plans,configs):
    y=encode(main);weight=np.array([r['recordWeight']for r in main]);search=[];selected=[]
    for outer,split in enumerate(plans):
        train=split['train'];p=predictions[:,train,:]
        tp=(p*y[train][None,:,:]*weight[train][None,:,None]).sum(1)
        fp=(p*(1-y[train][None,:,:])*weight[train][None,:,None]).sum(1)
        fn=((1-p)*y[train][None,:,:]*weight[train][None,:,None]).sum(1)
        macro=f1_counts(tp,fp,fn);segment=accuracy[:,split['strongTrain']].mean(1);joint=.5*(macro+segment)
        best=max(range(len(configs)),key=lambda i:(joint[i],segment[i],macro[i],-i))
        selected.append({'outer':outer,'candidate':best,'config':configs[best],
          'inner':{'joint':float(joint[best]),'mapMacroF1':float(macro[best]),'segmentChartAccuracy':float(segment[best])}})
        search.extend({'outer':outer,'candidate':i,'joint':float(joint[i]),'mapMacroF1':float(macro[i]),'segmentChartAccuracy':float(segment[i])}for i in range(len(configs)))
    return selected,search

def run_group(name,root,main,manual,plans,budget):
    directory=root/name
    if(directory/'complete.json').exists():return
    configs=candidates(name,budget);dump(directory/'candidates.json',configs)
    predictions=np.zeros((len(configs),len(main),len(MAP_CLASSES)),dtype=np.uint8);accuracy=np.zeros((len(configs),len(manual)))
    for source in GROUPS[name]['sources']:
        ids=[i for i,c in enumerate(configs)if c['source']==source];checkpoint=directory/('source-'+source+'.npz')
        if checkpoint.exists():
            with np.load(checkpoint)as saved:
                if not np.array_equal(saved['ids'],ids):raise RuntimeError('Source candidate checkpoint differs')
                predictions[ids]=saved['map'];accuracy[ids]=saved['accuracy']
            print(name,'reused',source,flush=True);continue
        values=load_source(source,main+manual);jobs=defaultdict(list)
        for i in ids:jobs[tuple(configs[i][k]for k in ['partition','head','smoothing','burdenGate'])].append(i)
        for j,indices in enumerate(jobs.values()):
            c=configs[indices[0]];data=pack(values,c);cache={}
            # Score the actual primary label, not an implementation-specific
            # probability tie-break. Uncovered annotated time is counted wrong.
            acc=[]
            for r,rows in zip(manual,data['chunks'][len(main):]):
                total=sum(a['end']-a['start']for a in r['annotations']if a['tags'])
                correct=sum(max(0,min(a['end'],s['end'])-max(a['start'],s['start']))for a in r['annotations']if a['tags']for s in rows if s['tag']in a['tags'])
                acc.append(correct/max(1,total))
            for i in indices:
                c=configs[i];key=tuple(c[k]for k in ['mode','effortPower','confidencePower'])
                if key not in cache:cache[key]=summaries(data,c)
                predictions[i]=decisions(cache[key],c)[:len(main)];accuracy[i]=acc
            if(j+1)%80==0:print(name,source,'native tuples',j+1,'/',len(jobs),flush=True)
        np.savez_compressed(checkpoint,ids=ids,map=predictions[ids],accuracy=accuracy[ids]);del values
    np.savez_compressed(directory/'candidate-predictions.npz',map=predictions,segmentAccuracy=accuracy)
    selected,search=choose(predictions,accuracy,main,plans,configs)
    native.write_gzip(directory/'search.json.gz',search);dump(directory/'selection.json',{'selectionUsesOuterLabels':False,'outerReportsWrittenAfterSelections':True,'selections':selected})
    rows=[];segments=[]
    for choice in selected:
        o,i,c=choice['outer'],choice['candidate'],choice['config'];split=plans[o]
        for k in split['test']:
            r=main[k];rows.append({'variant':name,'outer':o,'uid':r['uid'],'group':r['group'],'truth':target(r),'weight':r['recordWeight'],
              'predicted':[label for label,b in zip(MAP_CLASSES,predictions[i,k])if b]})
        for k in split['strongTest']:
            r=manual[k];value=load_source(c['source'],[r])[0]
            segments.extend({'variant':name,'outer':o,'uid':r['uid'],**s}for s in native.select_segments(value,c))
    dump(directory/'outer-predictions.json',rows);dump(directory/'outer-segments.json',segments)
    dump(directory/'complete.json',{'candidates':len(configs),'outerFits':0,'outerSelections':6})
    print('complete native followup',name,flush=True)

def run(args):
    main,manual,_=load();parent=json.loads((PARENT/'protocol.json').read_text(encoding='utf8'));plans=parent['plans']
    if not(PARENT/'complete.json').exists():raise RuntimeError('Parent study is incomplete')
    for spec in [json.loads((EXTRA/'spec.json').read_text(encoding='utf8')),*EXTRA_CACHE_SPECIFICATIONS.values()]:
        if spec['baseSpecification']['classes']!=CLASSES:raise RuntimeError('Additional native class order differs')
    identity=[{'uid':r['uid'],'group':r['group'],'tags':target(r),'weight':r.get('recordWeight',1)}for r in main+manual]
    if identity!=parent['records']:raise RuntimeError('Parent and followup populations differ')
    names=args.groups.split(',')if args.groups else list(GROUPS);root=Path(args.output)
    protocol={'id':STUDY_ID,'sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'sourceSnapshot':{n:(Path(__file__).parent/n).read_text(encoding='utf8')for n in SOURCE_FILES},
      'parentProtocol':parent,'factorialCacheSpecification':json.loads((EXTRA/'spec.json').read_text(encoding='utf8')),
      'additionalCacheSpecifications':EXTRA_CACHE_SPECIFICATIONS,'sourceCatalogue':CATALOGUE,'sourceParameters':SOURCE_PARAMETERS,'scope':SCOPE,
      'candidateBudgetPerAblation':args.candidates,'groups':{n:GROUPS[n]for n in names},'grid':GRID,
      'candidateMethod':'factorial_scale when present exactly matches logical_neutral configs except source; repaired pools: all source/partition/head/gate/valid-mode tuple seeds then unique scrambled joint Sobol, seed20261031',
      'selection':parent['selection'],'plans':plans,'records':identity,
      'requiredSemantics':'repaired requires logical5+neutral+coverage, coherent atomic evidence, normalized or soft aggregation, Vibro excluded as ordinary third Mix component',
      'labelAccess':'fixed label-independent candidate predictions and per-chart metric arrays can be cached for every record; each outer choice indexes ONLY its training groups/charts. Changing held-out labels/metrics leaves that outer choice invariant. Retrospective grouped selection; development data have been seen.',
      'factorialReuse':{'base':'native-joint-r1/logical_neutral','release':'native-joint-r1/logical_release','release_and_scale':'native-joint-r1/full'},
      'coefficientTraining':'none; no cold-refit claim'}
    path=root/'protocol.json'
    if path.exists()and json.loads(path.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Followup protocol changed; use a new revision')
    dump(path,protocol)
    for name in names:run_group(name,root,main,manual,plans,args.candidates)
    for filename in ['outer-predictions.json','outer-segments.json']:
        rows=[]
        for name in names:rows.extend(json.loads((root/name/filename).read_text(encoding='utf8')))
        dump(root/filename,rows)
    dump(root/'complete.json',{'variants':names,'candidatesPerAblation':args.candidates,'outerFits':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups');p.add_argument('--candidates',type=int,default=8192)
    p.add_argument('--output',default='research/rework/results/native-followup-r1');run(p.parse_args())
