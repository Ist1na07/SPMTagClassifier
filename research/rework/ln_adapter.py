"""Low-capacity Release-logit adaptation of the archived conditional LN head.

No RC replacement, no copied whole-map segment truth. Weak impressions train a
monotone map-level calibration of the SAME local conditional Release support.
"""
from mil import np,torch,json,Path,SCHEMA,target,load,math,time
from family import model_scores
from scipy.optimize import minimize
import torch.nn.functional as F

ARCHIVE=json.loads(Path('.cache/archived-model-reference.json').read_text(encoding='utf8'))
CLASSES=ARCHIVE['ln']['classes'];RELEASE=CLASSES.index('release')
KNOWN={'coordination','density','inverse','release','technical'}
GEOMETRY_SCHEMA=json.loads(Path('.cache/rework-release-geometry-r2-schema.json').read_text(encoding='utf8'))['features']
with Path('.cache/rework-release-geometry-r2.jsonl').open(encoding='utf8')as _f:
    GEOMETRY={r['uid']:r['features']for r in map(json.loads,_f)}
SPECS={'geometry3':['a','b','ab'],'no_isolation':['b'],'no_hold':['a'],'no_interaction':['a','b'],
       'bias':[],'old_summary':['old:tailIndependence','old:lnAreaRatio','old:lnWeightRatio'],
       'geometry9':['geometry:'+k for k in GEOMETRY_SCHEMA]}
SLOPE_INITIAL=math.log(math.expm1(1.))


def feature_rows(record,variant):
    keys=SPECS[variant];rows=[]
    if len(GEOMETRY[record['uid']])!=len(record['segments']):raise RuntimeError('Release geometry partition mismatch')
    for s,g in zip(record['segments'],GEOMETRY[record['uid']]):
        a=g[GEOMETRY_SCHEMA.index('headIsolation')];b=g[GEOMETRY_SCHEMA.index('heldOtherAtTail')]
        fields={'a':a,'b':b,'ab':a*b,**{'geometry:'+k:g[j]for j,k in enumerate(GEOMETRY_SCHEMA)}}
        fields.update({'old:'+k:s['original'][SCHEMA['original'].index(k)]for k in ['tailIndependence','lnAreaRatio','lnWeightRatio']})
        rows.append([fields[k]for k in keys])
    return np.array(rows).reshape(-1,len(keys))if keys else np.zeros((len(record['segments']),0))


def prior_logits(record):
    head=ARCHIVE['ln'];x=np.array([s['original']for s in record['segments']])
    return (x-np.array(head['mean']))/np.array(head['std'])@np.array(head['W']).reshape(len(CLASSES),-1).T+np.array(head['b'])


class Packed:
    def __init__(self,weak,manual,variant,config):
        weak=[r for r in weak if set(target(r))&KNOWN];self.variant=variant;self.keys=SPECS[variant]
        if not weak:raise RuntimeError('No explicit LN impression records in training scope')
        records=weak+manual;xs=[];logits=[];bag=[];physical=[];scaling=[];lengths=[];weak_y=[];weak_weight=[]
        strong_index=[];masks=[];strong_weight=[];offset=0
        actual_manual=[r for r in manual if any(a['tags']and set(a['tags'])<=set(CLASSES)for a in r['annotations'])]
        for i,r in enumerate(records):
            ws=r['segments'];x=feature_rows(r,variant);lengths.append(len(ws));xs.extend(x);logits.extend(prior_logits(r));bag.extend([i]*len(ws))
            duration=np.array([s['end']-s['start']for s in ws],float)
            ln=np.array([max(s['modern'][SCHEMA['modern'].index('occupancy')],s['modern'][SCHEMA['modern'].index('lnHeadFraction')])for s in ws])
            mass=duration*np.clip(ln,0,1);physical.extend(mass)
            if i<len(weak):
                if mass.sum()<=0:raise RuntimeError('Explicit LN chart has no physical LN evidence: '+r['uid'])
                weak_y.append(float('release'in target(r)));weak_weight.append(r.get('recordWeight',1))
                scale_weight=mass/mass.sum()*r.get('recordWeight',1)/sum(t.get('recordWeight',1)for t in weak)
            else:
                annotations=[a for a in r['annotations']if a['tags']and set(a['tags'])<=set(CLASSES)]
                total=sum(a['end']-a['start']for a in annotations);scale_weight=np.zeros(len(ws))
                for a in annotations:
                    for j,s in enumerate(ws):
                        overlap=max(0,min(a['end'],s['end'])-max(a['start'],s['start']))
                        if overlap:
                            w=overlap/max(1,total)/max(1,len(actual_manual));strong_index.append(offset+j)
                            masks.append([float(c in a['tags'])for c in CLASSES]);strong_weight.append(w);scale_weight[j]+=w
            scaling.extend(scale_weight);offset+=len(ws)
        raw=np.array(xs).reshape(offset,len(self.keys))if self.keys else np.zeros((offset,0));sw=np.array(scaling)
        # Equal total weak/manual domain mass when both have usable LN evidence.
        weak_rows=sum(lengths[:len(weak)]);sw[:weak_rows]/=max(1e-300,sw[:weak_rows].sum())
        if sw[weak_rows:].sum()>0:sw[weak_rows:]/=sw[weak_rows:].sum()
        self.mean=np.average(raw,axis=0,weights=sw);self.scale=np.maximum(1e-6,np.sqrt(np.average((raw-self.mean)**2,axis=0,weights=sw)))
        self.x=torch.tensor(np.c_[np.clip((raw-self.mean)/self.scale,-6,6),np.ones(offset)])
        self.logits=torch.tensor(np.array(logits));self.bag=torch.tensor(bag,dtype=torch.long);self.physical=torch.tensor(physical)
        self.n=len(records);self.nweak=len(weak);self.y=torch.tensor(weak_y);w=np.array(weak_weight);y=np.array(weak_y)
        if config['balance']=='balanced':
            if len(set(y))!=2:raise RuntimeError('Both weak impression labels required in this training scope')
            w=np.array([v/w[y==label].sum()/2 for v,label in zip(w,y)])
        else:w=w/w.sum()
        self.weak_weight=torch.tensor(w);self.strong_index=torch.tensor(strong_index,dtype=torch.long)
        self.mask=torch.tensor(np.array(masks).reshape(-1,len(CLASSES)));self.strong_weight=torch.tensor(strong_weight)
        self.scope={'weakCharts':len(weak),'manualChartsWithLn':len(actual_manual),'windows':offset,'strongOverlapRows':len(strong_index),
          'weakPositiveCharts':int(y.sum()),'trainUids':[r['uid']for r in records],'featureKeys':self.keys,
          'scaling':'training only; equal usable weak/manual domain mass, physical LN/duration and physical-file weighting'}

    def probabilities(self,theta):
        delta=self.x@theta[:-2];shift=torch.zeros_like(self.logits);shift[:,RELEASE]=delta
        return torch.softmax(self.logits+shift,1)

    def objective(self,theta,config):
        p=self.probabilities(theta);mass=torch.zeros(self.n).index_add(0,self.bag,self.physical)
        q=torch.zeros(self.n).index_add(0,self.bag,p[:,RELEASE]*self.physical)/mass.clamp(min=1e-300)
        logit=torch.logit(q[:self.nweak].clamp(1e-12,1-1e-12));score=F.softplus(theta[-2])*logit+theta[-1]
        weak=(F.binary_cross_entropy_with_logits(score,self.y,reduction='none')*self.weak_weight).sum()
        strong=-(torch.log((p[self.strong_index]*self.mask).sum(1).clamp(min=1e-15))*self.strong_weight).sum()
        ratio=config['strongWeight']if len(self.strong_index)else 0.
        prior=theta.clone();prior[-2]-=SLOPE_INITIAL
        return (weak+ratio*strong)/(1+ratio)+config['localL2']*prior.square().sum()/2


def fit(weak,manual,variant,config):
    data=Packed(weak,manual,variant,config);initial=np.zeros(len(data.keys)+3);initial[-2]=SLOPE_INITIAL;start=time.perf_counter();checks=[]
    def objective(x):
        t=torch.tensor(x,requires_grad=True);loss=data.objective(t,config);loss.backward();return float(loss.detach()),t.grad.numpy()
    status='budget_exhausted'
    for chunk in range(3):
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-8,'ftol':1e-14,'maxls':50})
        g=float(np.abs(result.jac).max());checks.append({'chunk':chunk+1,'loss':float(result.fun),'gradientMax':g,'iterations':int(result.nit),'message':str(result.message)})
        initial=result.x
        if g<=1e-5:status='small_gradient';break
    model={'variant':variant,'keys':data.keys,'mean':data.mean.tolist(),'scale':data.scale.tolist(),'delta':result.x[:-2].tolist(),
      'weakCalibrator':{'slope':float(F.softplus(torch.tensor(result.x[-2]))),'bias':float(result.x[-1])},'config':config,'clip':6}
    return model,{'status':status,'checks':checks,'seconds':time.perf_counter()-start,**data.scope}


def predict(model,records):
    result=[]
    for r in records:
        if not any(model['delta']):result.append(model_scores(ARCHIVE['ln'],np.array([s['original']for s in r['segments']])));continue
        x=feature_rows(r,model['variant']);x=np.clip((x-np.array(model['mean']))/np.array(model['scale']),-model['clip'],model['clip'])
        delta=np.c_[x,np.ones(len(x))]@np.array(model['delta']);z=prior_logits(r);z[:,RELEASE]+=delta;z-=z.max(1,keepdims=True)
        p=np.exp(z);result.append(p/p.sum(1,keepdims=True))
    return result


if __name__=='__main__':
    from search import plan
    main,manual,_=load();split=plan(main,manual)[0]['inner'][0]
    cfg={'localL2':1.,'strongWeight':1.,'balance':'balanced'}
    model,trace=fit([main[i]for i in split['train']],[manual[i]for i in split['strongTrain']],'geometry3',cfg)
    print(json.dumps({'model':model,'fit':trace},ensure_ascii=False,indent=2),flush=True)
    Path('.cache/ln-adapter-pilot.json').write_text(json.dumps({'model':model,'fit':trace},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
