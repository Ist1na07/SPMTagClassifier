"""Joint native-pipeline calibration, retaining the complete historical experts.

Each ablation gets the same deterministic, label-independent joint candidate
budget. This is a Sobol study over a declared finite space, NOT an exhaustive
Cartesian grid or a retraining of the historical expert's unknown source data.
No local proxy discards configurations before their final-task evaluation.
"""
import argparse,gzip,subprocess
from mil import *
from scipy.stats import qmc

CACHE=Path('.cache/rework-native-r1')
GRID={'partition':list(range(9)),'head':list(range(6)),'burdenGate':[0.,.15,.3],
      'smoothing':['legacy','coherent'],'mode':['legacy','normalized','soft'],
      'effortPower':[0.,1.],'confidencePower':[0.,1.],
      'dominantShare':[.4,.52,.65],'mixShare':[.04,.07,.12],'secondRel':[.6,.75,.9],
      'twoTagMaxShare':[.45,.55,.65],'familyThresholds':[[.1,.6],[.18,.68],[.3,.8]],
      'minEvidence':[.1,.25,.4],'minimumDiversity':[0.,.1,.2]}
BASE={'partition':4,'head':2,'burdenGate':0.,'smoothing':'legacy','mode':'legacy',
      'effortPower':0.,'confidencePower':1.,'dominantShare':.52,'mixShare':.07,'secondRel':.75,
      'twoTagMaxShare':.55,'familyThresholds':[.18,.68],'minEvidence':.25,'minimumDiversity':0.}
LN={'coordination','density','inverse','release','technical','ln wildcard'}
FINE=[c for c in CLASSES if c not in ['break','wildcard','ln wildcard']]
FI=[CLASSES.index(c)for c in FINE];BI=CLASSES.index('break')
GROUPS={k:{'sources':[k]}for k in ['original','area','adjacency','heldOrder','phase','sustaining',
          'neutralHands','coverage','transformedScale','logical','logical_neutral']}
for prefix in ['release','logical_release','full']:
    GROUPS[prefix]={'sources':[f'{prefix}_{ms}'for ms in [20,40,80]]}
for mode in ['legacy','normalized','soft']:
    GROUPS['full_'+mode]={'sources':GROUPS['full']['sources'],'fixed':{'mode':mode}}
for name,parameter in [('confidence','confidencePower'),('effort','effortPower'),('diversity','minimumDiversity')]:
    GROUPS['full_soft_no_'+name]={'sources':GROUPS['full']['sources'],'fixed':{'mode':'soft',parameter:0.}}
for name,source in [('original_soft_coherent','original'),('neutral_soft_coherent','neutralHands'),('mirror_ensemble','mirror_ensemble')]:
    GROUPS[name]={'sources':[source],'fixed':{'mode':'soft','smoothing':'coherent'}}
GROUPS['full_share_only']={'sources':GROUPS['full']['sources'],'fixed':{'mode':'normalized','excludeVibroFromMix':False}}
GROUPS['full_soft_include_vibro']={'sources':GROUPS['full']['sources'],'fixed':{'mode':'soft','excludeVibroFromMix':False}}

def write_gzip(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf8')as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def candidates(group,n=8192):
    spec=GROUPS[group];grid={'source':spec['sources'],**GRID};fixed=spec.get('fixed',{})
    found={}
    def add(config):
        c={**config,**fixed}
        c.setdefault('excludeVibroFromMix',c['mode']!='legacy')
        if c['mode']!='soft':c['minEvidence']=.25;c['minimumDiversity']=0.
        key=json.dumps(c,sort_keys=True)
        if key not in found:found[key]=c
    # All local partition/head/gate/smoothing combinations occur at least once.
    # These are seeds in the SAME final-task pool, not local preselection.
    for source,partition,head,gate,smoothing in itertools.product(spec['sources'],GRID['partition'],GRID['head'],GRID['burdenGate'],GRID['smoothing']):
        add({**BASE,'source':source,'partition':partition,'head':head,'burdenGate':gate,'smoothing':smoothing})
    if n<len(found):raise ValueError(f'Candidate budget {n} cannot cover all {len(found)} mandatory local tuples')
    points=qmc.Sobol(d=len(grid),scramble=True,seed=20261031).random_base2(16)
    for point in points:
        if len(found)>=n:break
        add({k:v[min(len(v)-1,int(point[j]*len(v)))]for j,(k,v)in enumerate(grid.items())})
    if len(found)<n:raise RuntimeError('Not enough distinct joint candidates; do not inflate with duplicate evaluations')
    return list(found.values())[:n]

def load_source(name,records):
    result=[]
    for r in records:
        base=Path('.cache/rework-native-ensemble-r1')if name=='mirror_ensemble'else CACHE
        path=base/name/(r['uid'].replace('/','__')+'.json.gz')
        with gzip.open(path,'rt',encoding='utf8')as f:value=json.load(f)
        expected='native-ensemble-r1'if name=='mirror_ensemble'else'native-label-free-r1'
        if value['id']!=expected or value['uid']!=r['uid']or value['treatment']!=name:raise RuntimeError('Native source identity differs')
        result.append(value)
    return result

def select_segments(value,config):
    part=value['partitions'][config['partition']];case=part['cases'][config['head']]
    rows=case['legacy'if config['smoothing']=='legacy'else'raw']
    if not config['burdenGate']:return rows
    active=sorted((s['burden'],s['end']-s['start'])for s in rows if s['burden']>0)
    mass=sum(w for v,w in active);acc=0.;reference=1.
    for v,w in active:
        acc+=w;reference=v
        if acc>=.75*mass:break
    result=[]
    for s in rows:
        if s['family']!='break'and s['burden']/reference<config['burdenGate']:
            probability=[0.]*len(CLASSES);probability[BI]=1.
            result.append({**s,'family':'break','tag':'break','confidence':1.,'probability':probability})
        else:result.append(s)
    return result

def pack(values,config):
    chunks=[select_segments(v,config)for v in values];n=len(chunks)
    lengths=[len(s)for s in chunks];rows=[s for chunk in chunks for s in chunk]
    if any(n==0 for n in lengths):raise RuntimeError('Native source has an empty partition; resolve its interval contract')
    bag=np.repeat(np.arange(n),lengths);duration=np.array([s['end']-s['start']for s in rows])/1000
    burden=np.array([s['burden']for s in rows]);reference=np.ones(n)
    offset=0;first=np.full((n,len(FINE)),np.inf)
    for i,chunk in enumerate(chunks):
        active=sorted((s['burden'],s['end']-s['start'])for s in chunk if s['burden']>0)
        mass=sum(w for v,w in active);acc=0.
        for v,w in active:
            acc+=w;reference[i]=v
            if acc>=.75*mass:break
        for j,s in enumerate(chunk):
            if s['family']!='break'and s['tag']in FINE:
                k=FINE.index(s['tag']);first[i,k]=min(first[i,k],offset+j)
        offset+=len(chunk)
    return {'chunks':chunks,'bag':bag,'duration':duration,'effort':np.maximum(1e-4,burden/reference[bag]),
            'p':np.array([s['probability']for s in rows]),'confidence':np.array([s['confidence']for s in rows]),
            'tag':np.array([CLASSES.index(s['tag'])for s in rows]),
            'family':np.array([s['family']for s in rows]),'n':n,'first':first}

def summaries(data,config):
    bag,n,p=data['bag'],data['n'],data['p'];soft=config['mode']=='soft'
    base=(data['duration']if soft else np.maximum(.5,data['duration']))*data['effort']**config['effortPower']
    confidence=(p*p).sum(1)if soft else np.maximum(.4,data['confidence'])
    weight=base*confidence**config['confidencePower'];weights=np.zeros((n,len(CLASSES)))
    if soft:
        weight=weight*(data['tag']!=BI)
        for k in range(len(CLASSES)):weights[:,k]=np.bincount(bag,weights=weight*p[:,k],minlength=n)
        total=weights[:,:BI].sum(1);q=weights/np.maximum(total[:,None],1e-300);q[:,BI]=0.
        entropy=-(q*np.log(np.maximum(q,1e-300))).sum(1)/math.log(len(CLASSES)-1)
        active=1-p[:,BI];local=p[:,:BI]/np.maximum(active[:,None],1e-300)
        h=-(local*np.log(np.maximum(local,1e-300))).sum(1)/math.log(len(CLASSES)-1)
        mean=np.bincount(bag,weights=weight*active*h,minlength=n)/np.maximum(total,1e-300)
        diversity=np.maximum(0.,entropy-mean)
        ln=weights[:,[CLASSES.index(c)for c in LN]].sum(1);rc=total-ln
        first=np.broadcast_to(np.arange(len(FINE)),(n,len(FINE)))
    else:
        eligible=(data['family']!='break')&np.isin(data['tag'],FI)
        np.add.at(weights,(bag[eligible],data['tag'][eligible]),weight[eligible])
        total=np.bincount(bag[eligible],weights=(base if config['mode']=='legacy'else weight)[eligible],minlength=n)
        q=weights/np.maximum(total[:,None],1e-300)
        ln=np.bincount(bag,weights=base*(data['family']=='ln'),minlength=n)
        rc=np.bincount(bag,weights=base*(data['family']=='rc'),minlength=n)
        diversity=np.zeros(n);first=data['first']
    fine=q[:,FI];order=np.lexsort((first,-fine),axis=1)
    ranked=np.take_along_axis(fine,order,axis=1)
    return {'shares':fine,'ranked':ranked,'order':order,'total':total,'distribution':q,
            'familyRatio':ln/(np.maximum(ln+rc,1e-300)if soft else np.maximum(1.,ln+rc)),
            'fineCount':(weights[:,FI]>0).sum(1),'diversity':diversity}

def decisions(summary,config):
    shares,ranked,order=summary['shares'],summary['ranked'],summary['order'];n=len(shares)
    count_mask=np.array([not(config.get('excludeVibroFromMix',config['mode']!='legacy')and c=='vibro')for c in FINE])
    top=ranked[:,0];significant=((shares>=config['mixShare'])&count_mask[None,:]).sum(1)
    family=np.where(summary['familyRatio']<config['familyThresholds'][0],0,np.where(summary['familyRatio']>config['familyThresholds'][1],1,2))
    soft=config['mode']=='soft';eligible=summary['total']>0
    if soft:eligible&=top>=config['minEvidence']
    # A Vibro component does not supply the third ordinary type required by
    # the stated Mix/Hybrid ontology. The historical behavior is a control.
    mix=eligible&(significant>=3)&(top<config['dominantShare'])
    accepted_mix=mix&((summary['diversity']>=config['minimumDiversity'])if soft else True)
    two=eligible&~mix&(ranked[:,1]>=config['secondRel']*top)&(top<config['twoTagMaxShare'])
    one=eligible&~mix&~two
    output=np.zeros((n,len(MAP_CLASSES)),dtype=np.uint8);ids=np.arange(n)
    fine_map=np.array([MAP_CLASSES.index(c)for c in FINE]);mix_map=np.array([MAP_CLASSES.index(c)for c in ['rc mix','ln mix','hybrid']])
    output[ids[accepted_mix],mix_map[family[accepted_mix]]]=1
    output[ids[one|two],fine_map[order[one|two,0]]]=1
    output[ids[two],fine_map[order[two,1]]]=1
    return output

def scores(y,p,w):
    tp=(y*p*w[:,None]).sum(0);fp=((1-y)*p*w[:,None]).sum(0);fn=(y*(1-p)*w[:,None]).sum(0)
    return float(f1_counts(tp,fp,fn))

def f1_counts(tp,fp,fn):
    return np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0).mean(-1)

def run(args):
    from search import plan,record_support
    main,manual,_=load();records=main+manual;plans=plan(main,manual);root=Path(args.output)
    cache_spec=json.loads((CACHE/'spec.json').read_text(encoding='utf8'))
    if cache_spec['classes']!=CLASSES:raise RuntimeError('Native and training class order differs')
    groups=args.groups.split(',')if args.groups else list(GROUPS)
    pool={g:candidates(g,args.candidates)for g in groups}
    source_files=['native_search.py','native.cjs','native-extract.cjs','native-policy.cjs','evidence.cjs','composition.cjs',
                  'legacy-lab.cjs','patches.cjs','mil.py','search.py','data.cjs',
                  'native-ensemble.cjs','native-mirror-extract.cjs','native-ensemble-build.cjs','mirror.cjs']
    protocol={'id':'native-joint-sobol-r1','sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'sourceSnapshot':{name:(Path(__file__).parent/name).read_text(encoding='utf8')for name in source_files},
      'sourceIdentity':'exact producing research modules embedded before scoring; archived engine and fixed model reside at sourceBaseCommit; no hash substitutes for source',
      'cacheSpecification':cache_spec,
      'ensembleSpecification':json.loads(Path('.cache/rework-native-ensemble-r1/spec.json').read_text(encoding='utf8'))if'mirror_ensemble'in groups else None,
      'scope':'retrospective grouped outer6; fixed historical experts; per-chart unsupervised normalization; complete native segmentation/gating/smoothing/aggregation path calibrated jointly',
      'historicalExposure':'unknown for pretrained expert coefficients; not a matched-from-scratch data comparison',
      'candidateBudgetPerAblation':args.candidates,'candidateMethod':'all local tuple seeds then deduplicated joint scrambled Sobol; same seed and budget for each ablation, not exhaustive Cartesian',
      'grid':GRID,'groups':{g:GROUPS[g]for g in groups},'selection':'.5 pooled inner map17 macroF1 + .5 equal-chart inner segment acceptable-label accuracy',
      'innerMeaning':'fixed expert pipeline has no supervised fit inside a candidate; pooled inner validation predictions equal predictions on the outer training groups; all thresholds selected without outer labels',
      'burdenGate':'after chosen native display policy, compare segment burden with duration-weighted positive-burden chart q75; threshold0 disables the extra gate',
      'confidence':'conditional native decision support, not calibrated correctness; soft weights use concentration and Mix may require between-segment diversity',
      'plans':plans,'records':[{'uid':r['uid'],'group':r['group'],'tags':target(r),'weight':r.get('recordWeight',1)}for r in records]}
    path=root/'protocol.json'
    if path.exists()and json.loads(path.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Protocol changed; do not silently mix runs')
    dump(path,protocol)
    dump(root/'supports.json',[{'outer':i,'train':record_support(main,s['train']),'test':record_support(main,s['test'])}for i,s in enumerate(plans)])
    for group in groups:
        directory=root/group
        if(directory/'complete.json').exists():continue
        configs=pool[group];dump(directory/'candidates.json',configs)
        sources={s:load_source(s,records)for s in GROUPS[group]['sources']}
        predictions=np.zeros((len(configs),len(main),len(MAP_CLASSES)),dtype=np.uint8)
        accuracies=np.zeros((len(configs),len(manual)));jobs=defaultdict(list)
        for i,c in enumerate(configs):jobs[tuple(c[k]for k in ['source','partition','head','smoothing','burdenGate'])].append(i)
        for j,(key,indices)in enumerate(jobs.items()):
            c=configs[indices[0]];data=pack(sources[c['source']],c);cache={}
            strong=[{**r,'segments':data['chunks'][len(main)+k]}for k,r in enumerate(manual)]
            metric=segment_metrics(strong,[np.array([s['probability']for s in r['segments']])for r in strong])
            accuracy=np.array([s['accuracy']for s in metric['charts']])
            for i in indices:
                c=configs[i];weight_key=tuple(c[k]for k in ['mode','effortPower','confidencePower'])
                if weight_key not in cache:cache[weight_key]=summaries(data,c)
                predictions[i]=decisions(cache[weight_key],c)[:len(main)];accuracies[i]=accuracy
            if(j+1)%40==0:print(group,'native tuples',j+1,'/',len(jobs),flush=True)
        np.savez_compressed(directory/'candidate-predictions.npz',map=predictions,segmentAccuracy=accuracies)
        y=encode(main);weight=np.array([r['recordWeight']for r in main]);search=[];selections=[]
        for outer,split in enumerate(plans):
            train=split['train'];strong=split['strongTrain'];p=predictions[:,train,:]
            tp=(p*y[train][None,:,:]*weight[train][None,:,None]).sum(1)
            fp=(p*(1-y[train][None,:,:])*weight[train][None,:,None]).sum(1)
            fn=((1-p)*y[train][None,:,:]*weight[train][None,:,None]).sum(1)
            f1=f1_counts(tp,fp,fn);acc=accuracies[:,strong].mean(1);joint=.5*(f1+acc)
            best=max(range(len(configs)),key=lambda i:(joint[i],acc[i],f1[i],-i))
            selections.append({'outer':outer,'candidate':best,'config':configs[best],
              'inner':{'joint':float(joint[best]),'mapMacroF1':float(f1[best]),'segmentChartAccuracy':float(acc[best])}})
            search.extend({'outer':outer,'candidate':i,'joint':float(joint[i]),'mapMacroF1':float(f1[i]),'segmentChartAccuracy':float(acc[i])}for i in range(len(configs)))
        write_gzip(directory/'search.json.gz',search);dump(directory/'selection.json',{'outerOpened':False,'selections':selections})
        outer_predictions=[];outer_segments=[]
        for choice in selections:
            o,i,c=choice['outer'],choice['candidate'],choice['config'];split=plans[o]
            for k in split['test']:
                r=main[k];outer_predictions.append({'variant':group,'outer':o,'uid':r['uid'],'group':r['group'],'truth':target(r),'weight':r['recordWeight'],
                  'predicted':[label for label,b in zip(MAP_CLASSES,predictions[i,k])if b]})
            for k in split['strongTest']:
                r=manual[k];rows=select_segments(sources[c['source']][len(main)+k],c)
                outer_segments.extend({'variant':group,'outer':o,'uid':r['uid'],**s}for s in rows)
        dump(directory/'outer-predictions.json',outer_predictions);dump(directory/'outer-segments.json',outer_segments)
        dump(directory/'complete.json',{'candidates':len(configs),'outerFits':0,'outerSelections':6,
          'scope':'native frozen-expert joint calibration; no historical coefficient retraining; each candidate evaluated on complete final path'})
        print('complete native ablation',group,flush=True)
    for filename in ['outer-predictions.json','outer-segments.json']:
        combined=[]
        for group in groups:combined.extend(json.loads((root/group/filename).read_text(encoding='utf8')))
        dump(root/filename,combined)
    dump(root/'complete.json',{'variants':groups,'outerFits':0,'candidatesPerAblation':args.candidates,
      'scope':'complete native path, fixed historical experts, jointly calibrated bounded candidate space; not coefficient retraining'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='research/rework/results/native-joint-r1')
    parser.add_argument('--groups');parser.add_argument('--candidates',type=int,default=8192)
    run(parser.parse_args())
