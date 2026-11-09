"""Roll-gated redistribution within Speed/Tech/Minijack, eight coefficients.

The triad's original probability mass is fixed, all other classes are fixed,
and LN/Break routing is fixed. Weak single-slot chart labels supervise bags,
not copied segment targets. Human combinations are allowed sets, not invented
equal probabilities. This restricted test does not learn Anchor or Vibro.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from pressure_data import mil,read

TRIAD=['speed','tech','minijack']
INDICES=[mil.CLASSES.index(c)for c in TRIAD]
CONTRAST=np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0.,-2/np.sqrt(6)]])
CONCEPTS={'concept/C16':['speed'],'concept/C24':['tech'],'concept/C32':['tech','minijack'],'concept/C48':['minijack','anchor']}


class World:
    def __init__(self):
        self.main,self.manual,records=mil.load();self.records={r['uid']:r for r in records};self.frames={r['uid']:r for r in map(json.loads,Path('.cache/frontier-rc-frames-r1.jsonl').read_text().splitlines())};self.inputs={};self._probability={}
        for uid,row in self.frames.items():
            self.inputs[uid]=read(Path('.cache/frontier-rc-action-r1')/(uid.replace('/','__')+'.json'))
            if self.inputs[uid]['intervals']!=row['intervals']:raise ValueError('RC motor/frame intervals differ')
            if uid.startswith('concept/'):
                keys=read('.cache/frontier-rc-frames-r1.jsonl.schema.json')['eventKeys'];effort=[w['values'][keys.index('effort')]for w in row['eventWindows']]
                self.records[uid]={'uid':uid,'dataset':'concept','source':row['source'],'segments':[{'start':a,'end':b,'effort':e}for (a,b),e in zip(row['intervals'],effort)]}
    def probability(self,uid):
        if uid in self._probability:return self._probability[uid]
        out=[]
        for f in self.frames[uid]['frames']:
            values=np.array([f['scores'].get(c,0.)for c in mil.CLASSES]);total=values.sum();out.append(values/total if total>0 else values)
        result=np.array(out);result.flags.writeable=False;self._probability[uid]=result;return result
    def scaler(self,main_train):
        x=[];weight=[]
        for r in main_train:
            row=self.inputs[r['uid']];a=np.array(row['action']);take=np.array([f['family']=='rc'for f in self.frames[r['uid']]['frames']])&(np.array(row['roll'])>0)
            if take.any():x.extend(a[take]);weight.extend(np.full(take.sum(),r['recordWeight']/take.sum()))
        x=np.array(x);w=np.array(weight);mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));return mean,np.maximum(scale,1e-6)


class Packed:
    def __init__(self,world,main_train,manual_train,*,kind='hand',reference=None):
        self.world=world;self.kind=kind;self.mean,self.scale=reference if reference is not None else world.scaler(main_train)
        weak=[r for r in main_train if len(mil.target(r))==1 and mil.target(r)[0]in TRIAD]
        self.rows=weak+manual_train+[world.records[uid]for uid in CONCEPTS];self.offsets={};features=[];prior=[];mass=[];teacher=[];self.weak=[];self.strong=[];self.concepts=[];offset=0
        weak_mass=sum(r['recordWeight']for r in weak)
        for row in self.rows:
            uid=row['uid'];inputs=world.inputs[uid];p=world.probability(uid);n=len(p);self.offsets[uid]=(offset,offset+n);base=p[:,INDICES];m=base.sum(1);q=np.divide(base,m[:,None],out=np.full_like(base,1/3),where=m[:,None]>0)
            # Numerical positive floor only inside the learnable triad. The
            # actual runtime bypasses an exactly-zero residual to preserve it.
            q=np.maximum(q,1e-12);q/=q.sum(1,keepdims=True);raw=(np.array(inputs['action'])-self.mean)/self.scale
            if kind=='frequency':raw=raw[:,:1]
            roll=np.array(inputs['roll'])*np.array([f['family']=='rc'for f in world.frames[uid]['frames']]);features.extend(np.c_[roll,roll[:,None]*raw]);prior.extend(np.log(q));mass.extend(m);teacher.extend(p)
            duration=np.array([s['end']-s['start']for s in row['segments']],float);effort=np.maximum(1e-6,[s['effort']for s in row['segments']])
            if row in weak:self.weak.append({'range':(offset,offset+n),'target':TRIAD.index(mil.target(row)[0]),'duration':duration,'effort':effort,'weight':row['recordWeight']/weak_mass})
            if row['dataset']=='manual':
                total=sum(a['end']-a['start']for a in row['annotations']if a['tags'])
                for annotation in row['annotations']:
                    mask=np.array([c in annotation['tags']for c in mil.CLASSES],float)
                    if not mask[INDICES].any():continue  # omitted terms are parameter-constant
                    for j,s in enumerate(row['segments']):
                        d=max(0,min(s['end'],annotation['end'])-max(s['start'],annotation['start']))
                        if d:self.strong.append((offset+j,mask,d/total/max(1,len(manual_train))))
            if uid in CONCEPTS:
                mask=np.array([c in CONCEPTS[uid]for c in mil.CLASSES],float)
                self.concepts.extend((offset+j,mask,d/duration.sum()/len(CONCEPTS))for j,d in enumerate(duration))
            offset+=n
        self.x=np.array(features);self.log_prior=np.array(prior);self.mass=np.array(mass);self.teacher=np.array(teacher)
        self.dimension=self.x.shape[1]*2
        self.class_mass=np.array([sum(r['weight']for r in self.weak if r['target']==c)for c in range(3)])

    def distribution(self,theta):return softmax(self.log_prior+self.x@theta.reshape(self.x.shape[1],2)@CONTRAST.T,axis=1)

    def objective(self,theta,cfg):
        q=self.distribution(theta);gradient=np.zeros_like(q);loss=0.
        for row in self.weak:
            a,b=row['range'];c=row['target'];v=row['duration']*row['effort']**cfg['effortPower'];v/=v.sum();p=v@(self.mass[a:b]*q[a:b,c]);safe=max(float(p),1e-15);weight=row['weight']
            if cfg.get('weakBalance',False):weight/=np.count_nonzero(self.class_mass)*self.class_mass[c]
            loss-=weight*np.log(safe)
            if p>1e-15:
                residual=q[a:b].copy();residual[:,c]-=1;gradient[a:b]+=(weight*v*self.mass[a:b]*q[a:b,c]/safe)[:,None]*residual
        for edges,coefficient in [(self.strong,cfg['strongWeight']),(self.concepts,cfg['conceptWeight'])]:
            if not coefficient:continue
            for j,mask,w in edges:
                allowed=mask[INDICES];selected=float(q[j]@allowed);fixed=float(self.teacher[j]@mask-self.teacher[j,INDICES]@allowed);p=fixed+self.mass[j]*selected;safe=max(p,1e-15);loss-=coefficient*w*np.log(safe)
                if p>1e-15:gradient[j]+=coefficient*w*self.mass[j]/safe*q[j]*(selected-allowed)
        loss+=cfg['localL2']*(theta@theta)/2;g=(self.x.T@gradient@CONTRAST).ravel()+cfg['localL2']*theta
        return float(loss),g


def fit(packed,cfg):
    start=time.perf_counter();rng=np.random.default_rng(20261102);starts=[np.zeros(packed.dimension),rng.normal(0,.05,packed.dimension),rng.normal(0,.05,packed.dimension)];results=[]
    for index,theta in enumerate(starts):
        history=[]
        for chunk in range(3):
            result=minimize(packed.objective,theta,args=(cfg,),jac=True,method='L-BFGS-B',options={'maxiter':1500,'gtol':1e-8,'ftol':1e-14,'maxls':50});theta=result.x;value,gradient=packed.objective(theta,cfg);g=float(np.abs(gradient).max());history.append({'chunk':chunk+1,'loss':value,'gradientMax':g,'iterations':int(result.nit),'message':str(result.message)})
            if g<=1e-5:break
        if g>1e-5:raise RuntimeError('RC action fit failed its gradient condition')
        results.append({'start':index,'theta':theta.tolist(),'loss':value,'gradientMax':g,'checks':history})
    best=min(results,key=lambda r:(r['loss'],r['start']));return {'kind':packed.kind,'theta':best['theta'],'mean':packed.mean.tolist(),'scale':packed.scale.tolist(),'triad':TRIAD}, {
        'seconds':time.perf_counter()-start,'selectedStart':best['start'],'starts':results,'maximumGradient':max(r['gradientMax']for r in results),
        'interpretation':'three training-only starts of the same small nonconvex bag objective; no global-optimality claim'}


def redistribute(world,uid,model):
    row=world.inputs[uid];frames=world.frames[uid]['frames'];p=world.probability(uid);base=p[:,INDICES];mass=base.sum(1);q=np.divide(base,mass[:,None],out=np.full_like(base,1/3),where=mass[:,None]>0);q=np.maximum(q,1e-12);q/=q.sum(1,keepdims=True)
    raw=(np.array(row['action'])-np.array(model['mean']))/np.array(model['scale'])
    if model['kind']=='frequency':raw=raw[:,:1]
    roll=np.array(row['roll'])*np.array([f['family']=='rc'for f in frames]);x=np.c_[roll,roll[:,None]*raw];delta=x@np.array(model['theta']).reshape(x.shape[1],2)@CONTRAST.T;new=softmax(np.log(q)+delta,axis=1)
    out=p.copy();out[:,INDICES]=new*mass[:,None]
    unchanged=(np.ptp(delta,axis=1)<1e-15)|(roll==0)|(mass==0);out[unchanged]=p[unchanged]
    return out,unchanged
