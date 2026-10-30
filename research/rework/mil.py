"""One differentiable segment -> weighted composition -> chart-tag pipeline.

The chart head sees ONLY the composition and its summaries, never a separate
pooled raw-feature representation. Chart labels supervise bags, not every
segment. Strong annotations supervise overlapping predicted intervals with a
partial-label likelihood. All preprocessing is fitted on training charts.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import json,re,math,itertools,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
SCHEMA=json.loads(Path('.cache/rework-schema.json').read_text(encoding='utf8'))
SCHEMA['constant']=['zero']
CLASSES=SCHEMA['classes']
MAP_CLASSES=sorted(set(CLASSES)-{'break','wildcard'}|{'rc mix','ln mix','hybrid'})
VARIANTS={
 'original56':(['original'],False),
 'original56_context':(['original','context'],False),
 'modern40_context':(['modern','context'],False),
 'corrected56_context':(['corrected','context'],False),
 'union_context':(['corrected','modern','context'],False),
 'prior_residual':(['corrected','modern','context'],True),
 'frozen_prior':(['corrected','modern','context'],True),
 'strong_only_prior':(['corrected','modern','context'],True),
 'copied_prior':(['corrected','modern','context'],True),
 'context_residual':(['context'],True),
 'bias_only_prior':(['constant'],True),
}
def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def target(r):
    return sorted(set('dense chordstream' if t=='chordstream' else t.lower() for t in r['tags']))
def canonical(s):
    s=re.sub(r'\b(?:x\s*)?\d+(?:\.\d+)?\s*x\b|\bx\s*\d+(?:\.\d+)?\b','',s.lower())
    return re.sub(r'[^\w]+','',s)
def load():
    with Path('.cache/rework7.jsonl').open(encoding='utf8') as f:
        data=[json.loads(line) for line in f]
    parent={r['uid']:r['uid'] for r in data}
    def find(x):
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def join(a,b):parent[find(b)]=find(a)
    identities={}
    for r in data:
        if r.get('duplicateOf'):join(r['uid'],r['duplicateOf'])
        m=r['meta'];ids=[('song',canonical(m.get('Artist','')),canonical(m.get('Title','')))]
        for k in ['BeatmapID','BeatmapSetID']:
            v=m.get(k,'')
            if v.isdigit() and int(v)>0:ids.append((k,v))
        if r.get('extra',{}).get('base_title'):ids.append(('base',canonical(r['extra']['base_title'])))
        ids.append(('file',r['source']))
        for identity in ids:
            if identity in identities:join(r['uid'],identities[identity])
            else:identities[identity]=r['uid']
    for r in data:r['group']=find(r['uid'])
    manual=[r for r in data if r['dataset']=='manual'];mg={r['group'] for r in manual}
    main=[r for r in data if r['dataset']!='manual' and target(r) and r['group'] not in mg]
    # Duplicate physical files remain separate annotation records, each with 1/n
    # weight. Their conflicting tags are not silently unioned into a new truth.
    counts=Counter(r['source'] for r in main)
    for r in main:r['recordWeight']=1/counts[r['source']]
    return main,manual,data
def grouped_folds(records,n=6):
    """Deterministic iterative multilabel group allocation (no model scores).

    Allocate a group containing the scarcest remaining label to a fold needing
    that label most; use relative label/record balance to break ties. Single-
    group classes necessarily remain unidentifiable in their held-out fold.
    """
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r['group']].append(i)
    keys=sorted(groups);labels=sorted({t for r in records for t in target(r)})
    vectors=np.array([[sum(t in target(records[i]) for i in groups[g]) for t in labels] for g in keys],float)
    sizes=np.array([len(groups[g]) for g in keys]);totals=vectors.sum(0)
    counts=np.zeros((n,len(labels)));used=np.zeros(n);remaining=set(range(len(keys)));assignment={}
    rng=np.random.default_rng(20261030);tie=rng.permutation(n).tolist()
    while remaining:
        present=(vectors[list(remaining)]>0).sum(0)
        nonzero=np.flatnonzero(present)
        label=min(nonzero,key=lambda j:(present[j],totals[j],labels[j]))
        options=[i for i in remaining if vectors[i,label]>0]
        g=max(options,key=lambda i:(np.sum((vectors[i]>0)/np.maximum(totals,1)),sizes[i],keys[i]))
        def cost(f):
            after=counts[f]+vectors[g]
            return (counts[f,label],np.sum((after/(totals/n+1))**2)+((used[f]+sizes[g])/(sizes.sum()/n))**2,tie.index(f))
        fold=min(range(n),key=cost);assignment[keys[g]]=fold
        counts[fold]+=vectors[g];used[fold]+=sizes[g];remaining.remove(g)
    return [[i for i,r in enumerate(records) if assignment[r['group']]==f] for f in range(n)]
def encode(rows):return np.array([[float(c in target(r)) for c in MAP_CLASSES] for r in rows])
def predict_tags(prob,threshold):
    fine=threshold['fine'] if isinstance(threshold,dict) else threshold
    mix=threshold['mix'] if isinstance(threshold,dict) else threshold
    mi=[MAP_CLASSES.index(c) for c in ['rc mix','ln mix','hybrid']]
    fi=[i for i in range(len(MAP_CLASSES)) if i not in mi]
    p=np.zeros_like(prob,dtype=bool)
    for i,row in enumerate(prob):
        if row.max()<=0:continue  # actual empty evidence remains empty
        best=mi[int(np.argmax(row[mi]))]
        if row[best]>=mix and row[best]>=max(row[fi]):p[i,best]=True
        else:p[i,fi]=row[fi]>=fine
        if not p[i].any():p[i,np.argmax(row)]=True
    return p.astype(int)
def map_metrics(rows,prob,threshold):
    y=encode(rows);p=predict_tags(prob,threshold);w=np.array([r.get('recordWeight',1) for r in rows])
    return {'macroF1':float(f1_score(y,p,average='macro',sample_weight=w,zero_division=0)),
      'microF1':float(f1_score(y,p,average='micro',sample_weight=w,zero_division=0)),
      'sampleF1':float(f1_score(y,p,average='samples',sample_weight=w,zero_division=0)),
      'exactMatch':float(accuracy_score(y,p,sample_weight=w)),
      'support':dict(zip(MAP_CLASSES,(y*w[:,None]).sum(0).tolist()))}
def segment_metrics(rows,probs):
    metrics=[]
    for r,p in zip(rows,probs):
        pred=p.argmax(1);correct=total=nll=0
        for a in r['annotations']:
            allowed=[CLASSES.index(t) for t in a['tags'] if t in CLASSES]
            if not allowed:continue
            total+=a['end']-a['start'];covered=0
            for j,s in enumerate(r['segments']):
                d=max(0,min(a['end'],s['end'])-max(a['start'],s['start']))
                covered+=d
                correct+=d*(pred[j] in allowed)
                nll-=d*np.log(max(p[j,allowed].sum(),1e-15))
            nll-=(a['end']-a['start']-covered)*np.log(1e-15)
        metrics.append({'chart':r['id'],'accuracy':correct/max(total,1),'correctMs':correct,'labelledMs':total,'nll':nll/max(total,1)})
    return {'chartMeanAccuracy':float(np.mean([m['accuracy'] for m in metrics])) if metrics else 0.,
      'durationAccuracy':sum(m['correctMs'] for m in metrics)/max(1,sum(m['labelledMs'] for m in metrics)),'charts':metrics}
def matrix(rows,variant):
    names,_=VARIANTS[variant]
    return np.array([sum(([0.] if k=='constant' else s[k] for k in names),[]) for r in rows for s in r['segments']],float)
class Packed:
    def __init__(self,rows,variant,mean=None,scale=None):
        self.rows=rows;self.lengths=[len(r['segments']) for r in rows]
        raw=matrix(rows,variant);self.length=len(raw)
        # Per-chart equal total mass when estimating scaling; no held-out chart.
        weights=np.concatenate([np.full(n,1/n) for n in self.lengths])
        self.mean=np.average(raw,axis=0,weights=weights) if mean is None else mean
        self.scale=np.sqrt(np.average((raw-self.mean)**2,axis=0,weights=weights)) if scale is None else scale
        self.scale=np.maximum(self.scale,1e-6)
        self.x=torch.tensor(np.clip((raw-self.mean)/self.scale,-6,6))
        self.bag=torch.tensor(np.repeat(np.arange(len(rows)),self.lengths),dtype=torch.long)
        self.durations=torch.tensor([max(1,s['end']-s['start']) for r in rows for s in r['segments']])
        self.effort=torch.tensor([max(1e-4,s['effort']) for r in rows for s in r['segments']])
        self.prior=torch.tensor(np.log([[max(1e-8,v) for v in s['prior']] for r in rows for s in r['segments']]))
        self.weak=[i for i,r in enumerate(rows) if r['dataset']!='manual']
        self.y=torch.tensor(encode([rows[i] for i in self.weak]))
        self.weak_weight=torch.tensor([rows[i].get('recordWeight',1) for i in self.weak])
        self.edges=[];masks=[];strongw=[];offset=0
        nstrong=sum(r['dataset']=='manual' for r in rows)
        for r in rows:
            if r['dataset']=='manual':
                labelled=sum(a['end']-a['start'] for a in r['annotations'] if a['tags'])
                for a in r['annotations']:
                    allowed=[CLASSES.index(t) for t in a['tags'] if t in CLASSES]
                    if not allowed:continue
                    for j,s in enumerate(r['segments']):
                        d=max(0,min(a['end'],s['end'])-max(a['start'],s['start']))
                        if d:
                            self.edges.append(offset+j);mask=np.zeros(len(CLASSES));mask[allowed]=1;masks.append(mask)
                            strongw.append(d/labelled/max(1,nstrong))
            offset+=len(r['segments'])
        self.mask=torch.tensor(np.array(masks))
        self.strongw=torch.tensor(strongw)
        self.copy_edges=[];copy_y=[];copy_w=[];offset=0
        for r,n in zip(rows,self.lengths):
            labels=target(r)
            if r['dataset']!='manual' and len(labels)==1 and labels[0] in CLASSES:
                dur=np.array([max(1,s['end']-s['start']) for s in r['segments']],float)
                self.copy_edges.extend(range(offset,offset+n));copy_y.extend([CLASSES.index(labels[0])]*n)
                copy_w.extend((dur/dur.sum()*r.get('recordWeight',1)).tolist())
            offset+=n
        self.copy_y=torch.tensor(copy_y,dtype=torch.long)
        self.copy_w=torch.tensor(copy_w)/max(1,sum(copy_w))
        self.variant=variant
class Model(torch.nn.Module):
    def __init__(self,d,prior):
        super().__init__();self.prior=prior
        self.local=torch.nn.Linear(d,len(CLASSES));torch.nn.init.zeros_(self.local.weight);torch.nn.init.zeros_(self.local.bias)
        # A fine chart tag is monotone in ITS OWN segment composition. A free
        # cross-class chart matrix could relabel latent local classes and make
        # apparently named segment probabilities scientifically meaningless.
        self.fine_classes=[c for c in MAP_CLASSES if c not in ['rc mix','ln mix','hybrid']]
        self.fine_slope=torch.nn.Parameter(torch.full((len(self.fine_classes),),math.log(math.expm1(.75))))
        self.fine_bias=torch.nn.Parameter(torch.zeros(len(self.fine_classes)))
        self.mix=torch.nn.Linear(11,3);torch.nn.init.zeros_(self.mix.weight);torch.nn.init.constant_(self.mix.bias,-2.)
    def forward(self,data,config):
        z=self.local(data.x)+(data.prior if self.prior else 0)
        p=torch.softmax(z,dim=1)
        # Smooth concentration (agreement of two draws), not an empirical
        # correctness probability. Whether it helps is an explicit ablation.
        concentration=p.square().sum(1)
        weight=data.durations*(data.effort**config['effortPower'])*(concentration**config['confidencePower'])
        total=torch.zeros(len(data.rows)).index_add(0,data.bag,weight)
        bag=torch.zeros((len(data.rows),len(CLASSES))).index_add(0,data.bag,p*weight[:,None])/total[:,None]
        q=bag[:,:-1]/torch.clamp(1-bag[:,-1:],min=1e-8)
        entropy=-(q*torch.log(torch.clamp(q,min=1e-12))).sum(1,keepdim=True)/math.log(q.shape[1])
        top=torch.topk(q,3,dim=1).values
        ln=q[:,[CLASSES.index(c) for c in ['coordination','density','inverse','release','technical','ln wildcard']]].sum(1,keepdim=True)
        # Conditional family entropy distinguishes LN mix from RC mix; balance
        # provides the non-monotonic family evidence required for Hybrid.
        ent=[]
        for indices in [[0,1,2,3,4,5,7],[8,9,10,11,12]]:
            v=q[:,indices];v=v/torch.clamp(v.sum(1,keepdim=True),min=1e-8)
            ent.append(-(v*torch.log(torch.clamp(v,min=1e-12))).sum(1,keepdim=True)/math.log(len(indices)))
        features=torch.cat([entropy,*ent,ln,4*ln*(1-ln),top,bag[:,-1:],q[:,6:7],q[:,13:14]],1)
        if data.variant in ['strong_only_prior','copied_prior']:features=features.detach();q=q.detach()
        slopes=.25+torch.nn.functional.softplus(self.fine_slope)
        own=torch.clamp(q[:,[CLASSES.index(c) for c in self.fine_classes]],min=1e-8,max=1-1e-8)
        fine=torch.log(own/(1-own))*slopes+self.fine_bias
        mix=self.mix(features)
        columns={c:fine[:,i] for i,c in enumerate(self.fine_classes)}
        columns.update({c:mix[:,i] for i,c in enumerate(['rc mix','ln mix','hybrid'])})
        return p,torch.stack([columns[c] for c in MAP_CLASSES],1),bag
    def local_control_objective(self,data,config):
        """The actual local training target for the no-weak-feedback controls.

        A detached weak chart loss is NOT a constant when line search moves
        local parameters. Optimize this explicit local target first, freeze the
        local model, then fit the chart head. Hyperparameters still get selected
        together by the final inner task, not by a local proxy.
        """
        p=torch.softmax(self.local(data.x)+(data.prior if self.prior else 0),dim=1)
        strong=-(torch.log(torch.clamp((p[data.edges]*data.mask).sum(1),min=1e-15))*data.strongw).sum() if data.edges else p.sum()*0
        loss=config['strongWeight']*strong
        if data.variant=='copied_prior' and data.copy_edges:
            loss=loss-(torch.log(torch.clamp(p[data.copy_edges,data.copy_y],min=1e-15))*data.copy_w).sum()
        return loss+config['localL2']*(self.local.weight.square().sum()+self.local.bias.square().sum())/2
    def objective(self,data,config):
        p,z,_=self(data,config)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(z[data.weak],data.y,reduction='none').mean(1)
        # A nonempty chart annotation also asserts that some segment is active.
        # Conditional non-Break composition ALONE has zero gradient to the
        # Break gate. Max-MIL supplies this missing observation without copying
        # the chart's type onto every segment or assuming an active duration.
        active=torch.zeros(len(data.rows)).scatter_reduce(0,data.bag,1-p[:,-1],reduce='amax',include_self=True)
        existence=-torch.log(torch.clamp(active[data.weak],min=1e-15))
        if data.variant in ['strong_only_prior','copied_prior']:existence=existence.detach()
        bce=bce+existence
        weak=(bce*data.weak_weight).sum()/data.weak_weight.sum()
        strong=-(torch.log(torch.clamp((p[data.edges]*data.mask).sum(1),min=1e-15))*data.strongw).sum() if data.edges else p.sum()*0
        if data.variant=='copied_prior' and data.copy_edges:
            weak=weak-(torch.log(torch.clamp(p[data.copy_edges,data.copy_y],min=1e-15))*data.copy_w).sum()
        reg=config['localL2']*(self.local.weight.square().sum()+self.local.bias.square().sum())/2
        reg+=config['mapL2']*(self.mix.weight.square().sum()+self.mix.bias.square().sum()+
          self.fine_bias.square().sum()+(.25+torch.nn.functional.softplus(self.fine_slope)-1).square().sum())/2
        return weak+config['strongWeight']*strong+reg
def fit(rows,variant,config,steps=300):
    torch.manual_seed(20261030);data=Packed(rows,variant)
    model=Model(data.x.shape[1],VARIANTS[variant][1]);optim=torch.optim.Adam(model.parameters(),lr=.04)
    if variant=='frozen_prior':
        for p in model.local.parameters():p.requires_grad_(False)
    previous=None;stable=0
    for step in range(steps):
        optim.zero_grad();loss=model.objective(data,config);loss.backward();optim.step()
        v=float(loss.detach())
        if previous is not None and abs(v-previous)<1e-7:stable+=1
        else:stable=0
        previous=v
        if stable>=15:break
    # Finish against a convergence criterion, not a favorable validation epoch.
    lb=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=100,tolerance_grad=1e-6,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        calls+=1;lb.zero_grad();loss=model.objective(data,config);loss.backward();return loss
    lb.step(closure);final=closure()
    grad=max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
    return model,data,{'adamSteps':step+1,'lbfgsEvaluations':calls,'loss':float(final.detach()),'gradientMax':grad}
def predict(model,train,rows,config):
    packed=Packed(rows,train.variant,train.mean,train.scale)
    with torch.no_grad():p,z,bag=model(packed,config)
    parts=np.split(p.numpy(),np.cumsum(packed.lengths)[:-1])
    probability=torch.sigmoid(z).numpy()
    for i,local in enumerate(parts):
        if np.all(local.argmax(1)==CLASSES.index('break')):probability[i]=0
    return probability,parts,bag.numpy()
if __name__=='__main__':
    main,manual,data=load();cfg={'localL2':.03,'mapL2':.03,'strongWeight':1.,'effortPower':1.,'confidencePower':0.}
    start=time.perf_counter();model,train,trace=fit(main[:250]+manual[:4],'prior_residual',cfg)
    p,s,b=predict(model,train,main[250:],cfg)
    print(len(main),len(manual),'features',train.x.shape,'seconds',time.perf_counter()-start,trace,flush=True)
    print('development-only fit, not a selected result',map_metrics(main[250:],p,.4)['macroF1'])
