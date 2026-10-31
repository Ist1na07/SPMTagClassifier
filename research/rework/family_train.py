"""Cold linear conditional experts; source-style, copied and bag supervision.

The two partial-label recipes have identical data, scaling and strong loss.
Their sole difference is E[-log p] versus -log E[p] for a weak chart label.
"""
from mil import *
from family import load_metadata
HEAD_CLASSES={'rc':['chordjack','dense chordstream','fast chordstream','minijack','speed','tech','wildcard'],
              'ln':['coordination','density','inverse','ln wildcard','release','technical']}
FIT_GRID={'localL2':[.0001,.0005,.0025,.0125,.0625],'strongWeight':[.25,1.,4.],'balance':['none','cap6']}
CONFIGS=[dict(zip(FIT_GRID,v))for v in itertools.product(*FIT_GRID.values())]
SPECS={recipe+'_'+representation:{'recipe':recipe,'representation':representation}
       for recipe in ['count_primary','copy_partial','bag_partial']for representation in ['original','corrected']}

class HeadData:
    def __init__(self,weak,manual,metadata,family,representation,recipe,config):
        self.classes=HEAD_CLASSES[family];K=len(self.classes);self.recipe=recipe;self.family=family
        xs=[];weak_ids=[];row_duration=[];bag_y=[];bag_weight=[];source_w=[];source_y=[]
        for r in weak:
            labels=target(r)
            if len(labels)!=1 or labels[0]not in self.classes:continue
            selected=[i for i,s in enumerate(r['segments'])if metadata[r['uid']][representation][i]['family0']!='break'and metadata[r['uid']][representation][i]['gates']['rowCount']]
            if not selected:continue
            total=sum(r['segments'][i]['end']-r['segments'][i]['start']for i in selected);label=self.classes.index(labels[0])
            for i in selected:
                s=r['segments'][i];xs.append(s[representation]);weak_ids.append(len(bag_y));row_duration.append((s['end']-s['start'])/total)
                source_w.append(r.get('recordWeight',1));source_y.append(label)
            bag_y.append(label);bag_weight.append(r.get('recordWeight',1))
        self.nweak=len(xs);self.nbags=len(bag_y)
        if not self.nweak:raise ValueError('No supported weak chart in family '+family)
        masks=[];strong_w=[];strong_primary_w=[];strong_y=[];strong_chart_count=0
        if recipe=='count_primary':
            for r in manual:
                for i,s in enumerate(r['segments']):
                    if not metadata[r['uid']][representation][i]['gates']['rowCount']:continue
                    overlaps=[max(0,min(s['end'],a['end'])-max(s['start'],a['start']))for a in r['annotations']]
                    if not overlaps:continue
                    j=int(np.argmax(overlaps));a=r['annotations'][j]
                    if not a['tags']or a['tags'][0]not in self.classes or overlaps[j]<.6*(s['end']-s['start']):continue
                    tag=a['tags'][0];label=self.classes.index(tag);xs.append(s[representation]);strong_y.append(label)
                    strong_primary_w.append(5. if tag in ['wildcard','ln wildcard','tech']else 3.)
            base=np.r_[source_w,np.array(strong_primary_w)*config['strongWeight']]
            labels=np.r_[source_y,strong_y].astype(int);freq=np.bincount(labels,weights=base,minlength=K)
            cw=np.minimum(6.,base.sum()/np.maximum(1e-300,K*freq))if config['balance']=='cap6'else np.ones(K)
            self.primary_y=torch.tensor(labels,dtype=torch.long);self.primary_weight=torch.tensor(base*cw[labels]);self.primary_normalizer=float(base.sum())
            scale_w=base
        else:
            for r in manual:
                annotations=[a for a in r['annotations']if a['tags']and set(a['tags'])<=set(self.classes)]
                labelled=sum(a['end']-a['start']for a in annotations)
                if not labelled:continue
                strong_chart_count+=1
                for a in annotations:
                    mask=np.array([float(c in a['tags'])for c in self.classes])
                    for i,s in enumerate(r['segments']):
                        if not metadata[r['uid']][representation][i]['gates']['rowCount']:continue
                        d=max(0,min(a['end'],s['end'])-max(a['start'],s['start']))
                        if d:
                            xs.append(s[representation]);masks.append(mask);strong_w.append(d/labelled)
            strong_w=np.array(strong_w)/max(1,strong_chart_count)
            bw=np.array(bag_weight);by=np.array(bag_y);freq=np.bincount(by,weights=bw,minlength=K)
            cw=np.minimum(6.,bw.sum()/np.maximum(1e-300,K*freq))if config['balance']=='cap6'else np.ones(K)
            self.bag_weight=torch.tensor(bw*cw[by]);self.weak_normalizer=float(bw.sum());self.bag_y=torch.tensor(by,dtype=torch.long)
            self.bag=torch.tensor(weak_ids,dtype=torch.long);self.duration=torch.tensor(row_duration)
            self.mask=torch.tensor(np.array(masks).reshape(-1,K));self.strong_w=torch.tensor(strong_w)
            # Each weak/annotated chart has equal input-moment mass; delta only
            # changes supervised strength, not the common copy-vs-bag scaler.
            scale_w=np.r_[np.array(row_duration)*bw[np.array(weak_ids)],strong_w*max(1,strong_chart_count)]
        x=np.array(xs);self.mean=np.average(x,axis=0,weights=scale_w)
        self.std=np.maximum(1e-6,np.sqrt(np.average((x-self.mean)**2,axis=0,weights=scale_w)))
        self.x=torch.tensor((x-self.mean)/self.std);self.labels=Counter(source_y)
        self.scope={'weakCharts':len(bag_y),'weakSegments':self.nweak,'strongRows':len(xs)-self.nweak,
                    'definiteFamilyStrongCharts':strong_chart_count if recipe!='count_primary'else None}

    def objective(self,W,b,config):
        p=torch.softmax(self.x@W.T+b,1)
        if self.recipe=='count_primary':
            loss=-(torch.log(torch.clamp(p[torch.arange(len(p)),self.primary_y],min=1e-15))*self.primary_weight).sum()/self.primary_normalizer
        else:
            if self.recipe=='copy_partial':
                per_row=-torch.log(torch.clamp(p[:self.nweak][torch.arange(self.nweak),self.bag_y[self.bag]],min=1e-15))*self.duration
                per_bag=torch.zeros(self.nbags).index_add(0,self.bag,per_row)
            else:
                pooled=torch.zeros((self.nbags,len(self.classes))).index_add(0,self.bag,p[:self.nweak]*self.duration[:,None])
                per_bag=-torch.log(torch.clamp(pooled[torch.arange(self.nbags),self.bag_y],min=1e-15))
            weak=(per_bag*self.bag_weight).sum()/self.weak_normalizer
            strong=-(torch.log(torch.clamp((p[self.nweak:]*self.mask).sum(1),min=1e-15))*self.strong_w).sum()if len(self.mask)else p.sum()*0
            loss=(weak+config['strongWeight']*strong)/(1+config['strongWeight']if len(self.mask)else 1)
        return loss+config['localL2']*W.square().sum()/2

def fit_head(weak,manual,metadata,family,representation,recipe,config):
    data=HeadData(weak,manual,metadata,family,representation,recipe,config)
    W=torch.nn.Parameter(torch.zeros(len(data.classes),data.x.shape[1]));b=torch.nn.Parameter(torch.zeros(len(data.classes)))
    optimizer=torch.optim.Adam([W,b],lr=.04);previous=None;stable=0
    for step in range(250):
        optimizer.zero_grad();loss=data.objective(W,b,config);loss.backward();optimizer.step();value=float(loss.detach())
        stable=stable+1 if previous is not None and abs(value-previous)<1e-8 else 0;previous=value
        if stable>=15:break
    optimizer=torch.optim.LBFGS([W,b],lr=1,max_iter=1000,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    calls=0;checks=[];status='budget_exhausted';stable=0;old_p=None;old_loss=None
    def closure():
        nonlocal calls
        calls+=1;optimizer.zero_grad();loss=data.objective(W,b,config);loss.backward();return loss
    for chunk in range(4):
        optimizer.step(closure);loss=closure();gradient=max(float(W.grad.abs().max()),float(b.grad.abs().max()))
        with torch.no_grad():p=torch.softmax(data.x@W.T+b,1)
        row={'chunk':chunk+1,'loss':float(loss.detach()),'gradientMax':gradient,'evaluations':calls}
        if old_p is not None:
            dp=(p-old_p).abs();row.update(meanProbabilityChange=float(dp.mean()),maxProbabilityChange=float(dp.max()),lossChange=abs(row['loss']-old_loss))
            stable=stable+1 if row['meanProbabilityChange']<=1e-5 and row['maxProbabilityChange']<=1e-3 and row['lossChange']<=1e-8 else 0
        checks.append(row);old_p=p.clone();old_loss=row['loss']
        if gradient<=1e-5:status='small_gradient';break
        if stable>=2:status='train_prediction_stable';break
    trace={'adamSteps':step+1,'status':status,'checks':checks,**data.scope}
    model={'classes':data.classes,'keys':SCHEMA[representation],'mean':data.mean.tolist(),'std':data.std.tolist(),
           'W':W.detach().numpy().ravel().tolist(),'b':b.detach().numpy().tolist()}
    return model,trace

def fit_pair(weak,manual,metadata,spec,config):
    models={};traces={}
    for family in ['rc','ln']:
        models[family],traces[family]=fit_head(weak,manual,metadata,family,spec['representation'],spec['recipe'],config)
    return models,traces

if __name__=='__main__':
    import time
    weak,manual,_=load();metadata=load_metadata();cfg={'localL2':.0005,'strongWeight':1.,'balance':'cap6'}
    for recipe in ['count_primary','copy_partial','bag_partial']:
        start=time.perf_counter();models,traces=fit_pair(weak[:250],manual[:5],metadata,{'recipe':recipe,'representation':'original'},cfg)
        print(recipe,round(time.perf_counter()-start,2),{k:(v['status'],v['checks'][-1]['gradientMax'],v['checks'][-1]['evaluations'])for k,v in traces.items()},flush=True)
