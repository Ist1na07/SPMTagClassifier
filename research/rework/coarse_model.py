"""One native-key local head and its SAME weighted coarse composition.

Five supported morphology classes; structural/relative-load Break gating is
label-independent. Window/gate/head regularization/strong and domain weights
are for inner selection. Pool exponents are fitted with the head, not selected
by a local proxy. No independent raw-feature whole-chart classifier exists.
"""
from mil import np,torch,json,Path,math
from coarse_data import CACHE,CLASSES,load_headers,plans
from scipy.optimize import minimize
import time

FINE=CLASSES[:-1]
SCHEMA=json.loads((CACHE/'spec.json').read_text(encoding='utf8'))
SPECS={
 'base40':{'representation':'base'},
 'sequence44':{'representation':'physical','drop':['release']},
 'full53':{'representation':'physical'},
 **{'no_'+g:{'representation':'physical','drop':[g]}for g in ['geometry','rhythm','holds','sequence','release','burden']},
 'copied':{'representation':'physical','recipe':'copy'},
 'no_class_balance':{'representation':'physical','balance':False},
 'no_effort':{'representation':'physical','effort':False},
 'no_confidence':{'representation':'physical','confidence':False},
 'weak4_only':{'representation':'physical','weak4Only':True,'noStrong':True},
 'weak4_manual7':{'representation':'physical','weakDomain':4},
 'weak7_manual7':{'representation':'physical','weakDomain':7},
}

def materialize(headers):
    out=[]
    for r in headers:
        value=json.loads((CACHE/(r['uid'].replace('/','__')+'.json')).read_text(encoding='utf8'))
        if value['source']!=r['source']:raise RuntimeError('Coarse source differs')
        out.append({**r,'windows':value['windows']})
    return out

def feature_keys(spec):
    keys=SCHEMA['baseKeys']if spec['representation']=='base'else SCHEMA['keys']
    removed=set(k for g in spec.get('drop',[])for k in SCHEMA['groups'][g])
    # Composite effort would reintroduce removed hold/geometry evidence.
    if {'holds','geometry'}&set(spec.get('drop',[])):removed.add('effort')
    return [k for k in keys if k not in removed]

def eligibility(windows,gate):
    positive=sorted((w['effort'],w['end']-w['start'])for w in windows if w['effort']>0)
    total=sum(d for v,d in positive);acc=0.;reference=1.
    for v,d in positive:
        acc+=d;reference=v
        if acc>=.75*total:break
    tail_index=SCHEMA['keys'].index('release:tailRatePerKey')
    active=np.array([not w['empty']and(not gate or w['effort']/reference>=gate or w['occupancy']>0 or w['x'][tail_index]>0)for w in windows])
    effort=np.array([max(1e-4,w['effort']/reference)for w in windows])
    return active,effort

class Packed:
    def __init__(self,records,spec,config,mean=None,scale=None,training=True):
        self.records=records;self.spec=spec;self.keys=feature_keys(spec);self.lengths=[];xs=[];eligible=[];efforts=[];durations=[];bag=[];scaling=[]
        source=SCHEMA['baseKeys']if spec['representation']=='base'else SCHEMA['keys'];index=[source.index(k)for k in self.keys]
        domains=[('manual'if r['dataset']=='manual'else str(r['keys']))for r in records]
        domain_mass={d:sum(r.get('recordWeight',1)for r,s in zip(records,domains)if s==d)for d in set(domains)}
        for i,(r,domain)in enumerate(zip(records,domains)):
            ws=r['windows'][str(config['windowMs'])];active,effort=eligibility(ws,config['burdenGate'])
            self.lengths.append(len(ws));array='base'if spec['representation']=='base'else'x'
            xs.extend([[w[array][j]for j in index]for w in ws]);eligible.extend(active.tolist());efforts.extend(effort)
            duration=np.array([w['end']-w['start']for w in ws],float);durations.extend(duration/1000);bag.extend([i]*len(ws))
            sw=duration*active;sw=sw/max(1.,sw.sum())*r.get('recordWeight',1)/domain_mass[domain]
            scaling.extend(sw)
        raw=np.array(xs);sw=np.array(scaling)
        self.mean=np.average(raw,axis=0,weights=sw)if mean is None else np.array(mean)
        self.scale=np.maximum(1e-6,np.sqrt(np.average((raw-self.mean)**2,axis=0,weights=sw)))if scale is None else np.array(scale)
        self.x=torch.tensor(np.clip((raw-self.mean)/self.scale,-6,6));self.eligible=torch.tensor(eligible,dtype=torch.bool)
        self.effort=torch.tensor(efforts);self.duration=torch.tensor(durations);self.bag=torch.tensor(bag,dtype=torch.long)
        self.n=len(records);self.d=len(self.keys);self.weak=[i for i,r in enumerate(records)if r['dataset']!='manual']if training else[]
        # Validate once, rather than rebuilding both whole-corpus arrays for
        # every chart. This guard does not change any training input or loss.
        active_counts=np.bincount(np.asarray(bag,dtype=np.int64),weights=np.asarray(eligible,dtype=np.int64),minlength=self.n)
        if any(active_counts[i]==0 for i in self.weak):raise RuntimeError('A labelled weak chart has no physical active window')
        self.labels=torch.tensor([FINE.index(records[i]['coarseTags'][0])for i in self.weak],dtype=torch.long)
        # Each source has its own class-balanced likelihood and total mass.
        weights=np.zeros(len(self.weak))
        available={records[i]['keys']for i in self.weak}
        for K in available:
            ids=[j for j,i in enumerate(self.weak)if records[i]['keys']==K]
            base=np.array([records[self.weak[j]].get('recordWeight',1)for j in ids]);y=self.labels[ids].numpy()
            if spec.get('balance',True):
                freq=np.bincount(y,weights=base,minlength=len(FINE));base=base/np.maximum(freq[y],1e-300)
            mass=(config['source4Weight']if K==4 else 1-config['source4Weight'])if len(available)>1 else 1.
            weights[ids]=mass*base/base.sum()
        self.weak_weight=torch.tensor(weights)
        strong_index=[];masks=[];strong_weight=[];missing=0.;offset=0;nmanual=sum(r['dataset']=='manual'for r in records)
        for r,length in zip(records,self.lengths):
            if training and r['dataset']=='manual':
                annotations=[a for a in r['annotations']if a['tags']and set(a['tags'])<=set(FINE)]
                total=sum(a['end']-a['start']for a in annotations)
                for a in annotations:
                    covered=0.
                    for j,w in enumerate(r['windows'][str(config['windowMs'])]):
                        overlap=max(0,min(w['end'],a['end'])-max(w['start'],a['start']))
                        if not overlap:continue
                        weight=overlap/max(1,total)/max(1,nmanual)
                        if eligible[offset+j]:
                            strong_index.append(offset+j);masks.append([float(c in a['tags'])for c in FINE]);strong_weight.append(weight)
                        else:missing+=weight
                        covered+=overlap
                    missing+=(a['end']-a['start']-covered)/max(1,total)/max(1,nmanual)
            offset+=length
        self.strong_index=torch.tensor(strong_index,dtype=torch.long);self.mask=torch.tensor(np.array(masks).reshape(-1,len(FINE)))
        self.strong_weight=torch.tensor(strong_weight);self.missing=missing
        self.scope={'records':len(records),'weakCharts':len(self.weak),'manualCharts':nmanual,'windows':len(xs),'strongOverlapRows':len(strong_index),
          'strongFineCoverageMissingMass':missing,'scaling':'training-only duration/physical-file weighted, each present source domain equal total mass; same scaler for copy and bag'}

    def forward(self,theta):
        D=self.d;K=len(FINE);W=theta[:K*D].reshape(K,D);bias=theta[K*D:K*D+K];a,b=theta[-2:]
        p=torch.softmax(self.x@W.T+bias,1)
        weight=self.duration*self.eligible*self.effort.pow(a)*(p.square().sum(1)).pow(b)
        mass=torch.zeros(self.n).index_add(0,self.bag,weight)
        composition=torch.zeros((self.n,K)).index_add(0,self.bag,p*weight[:,None])/mass.clamp(min=1e-300)[:,None]
        return p,composition,weight,mass

    def objective(self,theta,config):
        p,composition,weight,mass=self.forward(theta)
        if self.spec.get('recipe')=='copy':
            labels=torch.zeros(self.n,dtype=torch.long);labels[self.weak]=self.labels
            loss=-torch.log(p[torch.arange(len(p)),labels[self.bag]].clamp(min=1e-15))*weight
            per_bag=torch.zeros(self.n).index_add(0,self.bag,loss)/mass.clamp(min=1e-300)
            weak=(per_bag[self.weak]*self.weak_weight).sum()
        else:weak=-(torch.log(composition[self.weak,self.labels].clamp(min=1e-15))*self.weak_weight).sum()
        strong=-(torch.log((p[self.strong_index]*self.mask).sum(1).clamp(min=1e-15))*self.strong_weight).sum()+self.missing*(-math.log(1e-15))
        delta=config['strongWeight']if len(self.strong_index)or self.missing else 0.
        W=theta[:len(FINE)*self.d]
        return (weak+delta*strong)/(1+delta)+config['localL2']*W.square().sum()/2

def training_rows(records,spec):
    if spec.get('weak4Only'):records=[r for r in records if r['keys']==4 and r['dataset']!='manual']
    if spec.get('weakDomain'):records=[r for r in records if r['dataset']=='manual'or r['keys']==spec['weakDomain']]
    if spec.get('noStrong'):records=[r for r in records if r['dataset']!='manual']
    return records

def fit(records,spec,config):
    records=training_rows(records,spec)
    data=Packed(records,spec,config);size=len(FINE)*(data.d+1)+2;initial=np.zeros(size);initial[-2:]=.5
    allow_effort=spec.get('effort',True)and not({'holds','geometry'}&set(spec.get('drop',[])))
    bounds=[(None,None)]*(size-2)+[(0.,1.)if allow_effort else(0.,0.),(0.,1.)if spec.get('confidence',True)else(0.,0.)]
    initial[-2:]=[.5 if allow_effort else 0.,.5 if spec.get('confidence',True)else 0.]
    evaluations=0;checks=[];previous=None;stable=0;status='budget_exhausted';start=time.perf_counter()
    def objective(x):
        nonlocal evaluations
        t=torch.tensor(x,requires_grad=True);loss=data.objective(t,config);loss.backward();evaluations+=1
        return float(loss.detach()),t.grad.numpy()
    for chunk in range(3):
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':1000,'gtol':1e-7,'ftol':1e-12,'maxls':40})
        gradient=result.jac.copy()
        for j,(lo,hi)in enumerate(bounds):
            if lo==hi and lo is not None:gradient[j]=0.
            elif lo is not None and result.x[j]<=lo+1e-12 and gradient[j]>0 or hi is not None and result.x[j]>=hi-1e-12 and gradient[j]<0:gradient[j]=0.
        with torch.no_grad():p=data.forward(torch.tensor(result.x))[0].numpy()
        check={'chunk':chunk+1,'loss':float(result.fun),'projectedGradientMax':float(np.abs(gradient).max()),'iterations':int(result.nit),'message':str(result.message),'evaluations':evaluations}
        if previous is not None:
            dp=np.abs(p-previous[0]);check.update(meanProbabilityChange=float(dp.mean()),maxProbabilityChange=float(dp.max()),lossChange=abs(float(result.fun)-previous[1]))
            stable=stable+1 if dp.mean()<=1e-5 and dp.max()<=1e-3 and check['lossChange']<=1e-8 else 0
        checks.append(check);previous=(p,float(result.fun));initial=result.x
        if check['projectedGradientMax']<=1e-5:status='small_projected_gradient';break
        if stable>=2:status='train_prediction_stable';break
    model={'classes':FINE,'keys':data.keys,'mean':data.mean.tolist(),'scale':data.scale.tolist(),'theta':result.x.tolist(),
      'pool':{'effortPower':float(result.x[-2]),'confidencePower':float(result.x[-1])},'spec':spec,'config':config,'clip':6.}
    trace={'status':status,'seconds':time.perf_counter()-start,'checks':checks,**data.scope}
    return model,trace

def predict(model,records):
    data=Packed(records,model['spec'],model['config'],model['mean'],model['scale'],training=False)
    with torch.no_grad():p,composition,weight,mass=data.forward(torch.tensor(model['theta']))
    local=np.zeros((len(p),len(CLASSES)));active=data.eligible.numpy();local[active,:-1]=p.numpy()[active];local[~active,-1]=1.
    return {'composition':composition.numpy(),'local':np.split(local,np.cumsum(data.lengths)[:-1]),'weights':np.split(weight.numpy(),np.cumsum(data.lengths)[:-1]),'mass':mass.numpy()}

if __name__=='__main__':
    weak,manual,_=load_headers();split=plans(weak,manual)[0]['inner'][0]
    records=materialize([weak[i]for i in split['train']]+[manual[i]for i in split['strongTrain']])
    config={'windowMs':4000,'burdenGate':0.,'localL2':.003,'strongWeight':.5,'source4Weight':.5}
    model,trace=fit(records,SPECS['full53'],config)
    print(json.dumps({'pool':model['pool'],'trace':trace},indent=2),flush=True)
    Path('.cache/coarse-pilot-model.json').write_text(json.dumps(model),encoding='utf8')
