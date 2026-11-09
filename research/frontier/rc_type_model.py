"""Full RC/Break head: compare existing global cuts and actual hand history.

No roll-only gate or three-class redistribution. LN affinities stay fixed.
This is a bounded mechanism experiment, not a complete new classifier.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.linalg import helmert
from scipy.special import softmax
from scipy.optimize import minimize
from pressure_data import mil,read
from local_target_manifest import eligible_targets
from type_bag_objective import TypeBagObjective

SCHEMA=read('.cache/frontier-rc-type-inputs-r1.jsonl.schema.json')
TYPES=SCHEMA['rcClasses']+['break'];INDEX=[mil.CLASSES.index(c)for c in TYPES];OTHER=[i for i in range(len(mil.CLASSES))if i not in INDEX];CONTRAST=helmert(len(TYPES),full=False).T
CONCEPTS={'concept/C16':['speed'],'concept/C24':['tech'],'concept/C32':['tech','minijack'],'concept/C48':['minijack','anchor'],'concept/A02':['minijack'],'concept/CTX-H':['dense chordstream'],'concept/CTX-H-target':['break']}
COMMON=['log1pNpsPerKey','log1pRowRate','chordFraction','multiFraction','repeatFraction','rollFraction','log1pRepeatHzP90','oldRcMass','relativeNps']
GLOBAL=['crossCut','cutReturn','sameHand'];HAND=['handRepeatOverlap','handReturn','log1pHandFixedGripRatePerKey','handFirstOrderUncertainty','handSecondOrderUncertainty']


class World:
    def __init__(self):
        self.main,self.manual,records=mil.load();self.records={r['uid']:r for r in records};self.inputs={r['uid']:r for r in map(json.loads,Path('.cache/frontier-rc-type-inputs-r1.jsonl').read_text().splitlines())};self.locals=read('.cache/frontier-local-targets-r1.json')['records'];self.cache={};self.references={}
        for uid,r in self.inputs.items():
            if uid not in self.records:self.records[uid]={'uid':uid,'source':r['source'],'dataset':'local'if uid.startswith('local/')else'concept','segments':[{'start':a,'end':b,'effort':v[SCHEMA['eventKeys'].index('effort')]}for (a,b),v in zip(r['intervals'],r['event'])]}
    def teacher(self,uid):
        if uid not in self.cache:
            out=np.array([[f.get('scores',{}).get(c,0.)for c in mil.CLASSES]for f in self.inputs[uid]['frames']])
            for j,f in enumerate(self.inputs[uid]['frames']):
                if not out[j].sum():out[j,mil.CLASSES.index(f['tag'])]=1.
            out/=out.sum(1,keepdims=True);out.flags.writeable=False;self.cache[uid]=out
        return self.cache[uid]
    def reference(self,uid):
        source=self.inputs[uid].get('sourceUid')or('concept/CTX-H'if uid=='concept/CTX-H-target'else uid)
        if source not in self.references:
            row=self.inputs[source];values=np.array(row['event'])[:,SCHEMA['eventKeys'].index('npsPerKey')];duration=np.diff(np.array(row['intervals']),axis=1)[:,0];take=values>0
            if take.any():
                order=np.argsort(values[take],kind='stable');v=values[take][order];w=duration[take][order];ref=v[np.searchsorted(np.cumsum(w),.75*w.sum(),side='left')]
            else:ref=1.
            self.references[source]=max(1e-12,float(ref))
        return self.references[source]
    def features(self,uid,kind):
        row=self.inputs[uid];e=np.array(row['event']);get=lambda k:e[:,SCHEMA['eventKeys'].index(k)];nps=get('npsPerKey');common=np.c_[np.log1p(nps),np.log1p(get('rowRate')),get('chordFraction'),get('multiFraction'),get('repeatFraction'),get('rollFraction'),np.log1p(get('repeatHzP90')),self.teacher(uid)[:,INDEX[:-1]].sum(1),nps/self.reference(uid)]
        if kind in ['global','combined']:common=np.c_[common,*[get(k)for k in GLOBAL]]
        if kind in ['hands','combined']:
            h=np.array(row['hand'],float);common=np.c_[common,h[:,0],h[:,2],np.log1p(h[:,4]),h[:,6],h[:,7]]
        return common
    def scaler(self,records,kind):
        arrays=[];weights=[]
        for r in records:
            uid=r['uid'];take=self.teacher(uid)[:,INDEX].sum(1)>0
            if take.any():arrays.append(self.features(uid,kind)[take]);weights.append(np.full(take.sum(),r['recordWeight']/take.sum()))
        x=np.concatenate(arrays);w=np.concatenate(weights);mu=[]
        for j in range(x.shape[1]):
            take=np.isfinite(x[:,j]);v=x[take,j];mu.append(float(v[0])if len(v)and np.ptp(v)==0 else float(np.average(v,weights=w[take]))if len(v)else 0.)
        filled=np.where(np.isfinite(x),x,mu);expanded=np.c_[filled,~np.isfinite(x)];mean=np.average(expanded,axis=0,weights=w);sd=np.sqrt(np.average((expanded-mean)**2,axis=0,weights=w));keep=np.ptp(expanded,axis=0)>0
        return {'rawImputation':mu,'mean':mean.tolist(),'scale':np.where(keep,sd,1.).tolist(),'keep':keep.tolist(),'removedConstantColumns':np.flatnonzero(~keep).tolist()}
    def design(self,uid,kind,scaler):
        raw=self.features(uid,kind);filled=np.where(np.isfinite(raw),raw,scaler['rawImputation']);expanded=np.c_[filled,~np.isfinite(raw)];x=((expanded-scaler['mean'])/scaler['scale'])[:,scaler['keep']];return np.c_[np.ones(len(x)),x]
    def prior(self,uid):return np.c_[self.inputs[uid]['rcPriorLogProbabilities'],np.zeros(len(self.inputs[uid]['intervals']))]


class Packed(TypeBagObjective):
    def __init__(self,world,main_train,manual_train,*,kind='combined',real_enabled=True):
        self.world=world;self.kind=kind;self.types=TYPES;self.index=INDEX;self.other=OTHER;self.contrast=CONTRAST;self.scaler=world.scaler(main_train,kind);groups={r['group']for r in main_train};real=[r for r in eligible_targets(world.locals,groups)if r['primary']in TYPES]if real_enabled else[];self.localTrainUids=[r['uid']for r in real]
        weak=[r for r in main_train if len(mil.target(r))==1 and mil.target(r)[0]in TYPES[:-1]];self.rows=weak+manual_train+[world.records[r['uid']]for r in real]+[world.records[u]for u in CONCEPTS];self.weak=[];self.strong=[];self.ranks=[];self.concepts=[];self.offsets={};xs=[];priors=[];teachers=[];offset=0;weak_weight=sum(r['recordWeight']for r in weak);real_lookup={r['uid']:r for r in real}
        for row in self.rows:
            uid=row['uid'];teacher=world.teacher(uid);n=len(teacher);self.offsets[uid]=(offset,offset+n);xs.extend(world.design(uid,kind,self.scaler));priors.extend(world.prior(uid));teachers.extend(teacher);interval=np.array(world.inputs[uid]['intervals']);duration=interval[:,1]-interval[:,0];effort=np.maximum(1e-6,[s['effort']for s in row['segments']])
            if row in weak:self.weak.append({'range':(offset,offset+n),'target':TYPES.index(mil.target(row)[0]),'duration':duration,'effort':effort,'weight':row['recordWeight']/weak_weight})
            if row['dataset']=='manual':
                total=sum(a['end']-a['start']for a in row['annotations']if a['tags'])
                for a in row['annotations']:
                    mask=np.array([c in a['tags']for c in mil.CLASSES],float)
                    if not mask[INDEX].any():continue
                    for j,(start,end)in enumerate(interval):
                        d=max(0,min(end,a['end'])-max(start,a['start']))
                        if d:self.strong.append((offset+j,mask,d/total/max(1,len(manual_train))))
            if uid in real_lookup:
                r=real_lookup[uid];rank=[TYPES.index(r['primary'])]+([TYPES.index(r['component'])]if r.get('component')in TYPES else[]);self.ranks.extend((offset+j,rank,float(d/duration.sum()/max(1,len(real))))for j,d in enumerate(duration))
            if uid in CONCEPTS:self.concepts.append({'range':(offset,offset+n),'allowed':[TYPES.index(c)for c in CONCEPTS[uid]],'duration':duration,'effort':effort,'weight':1/len(CONCEPTS)})
            offset+=n
        self.x=np.array(xs);self.log_prior=np.array(priors);self.teacher=np.array(teachers);self.mass=self.teacher[:,INDEX].sum(1);self.dimension=self.x.shape[1]*(len(TYPES)-1);self.class_mass=np.array([sum(r['weight']for r in self.weak if r['target']==c)for c in range(len(TYPES))])


def projected_gradient(theta,gradient,bounds):
    out=np.array(gradient,copy=True)
    if bounds is not None:
        for j,(low,high)in enumerate(bounds):
            if low is not None and theta[j]<=low+1e-10 and out[j]>0:out[j]=0.
            if high is not None and theta[j]>=high-1e-10 and out[j]<0:out[j]=0.
    return out


def fit(packed,cfg,seed_theta=None):
    start=time.monotonic();rng=np.random.default_rng(20261102);traces=[]
    initial=[np.zeros(packed.dimension),rng.normal(0,.05,packed.dimension),rng.normal(0,.05,packed.dimension)]
    if seed_theta is not None:
        if np.shape(seed_theta)!=(packed.dimension,):raise ValueError('Nested TRAIN seed has wrong dimension')
        initial.append(np.array(seed_theta,copy=True))
    for i,theta in enumerate(initial):
        checks=[]
        for _ in range(3):
            bounds=getattr(packed,'bounds',None);a=minimize(packed.objective,theta,args=(cfg,),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':1500,'gtol':1e-8,'ftol':1e-14,'maxls':50});theta=a.x;loss,g=packed.objective(theta,cfg);maximum=float(np.abs(g).max());projected=float(np.abs(projected_gradient(theta,g,bounds)).max());checks.append({'loss':loss,'gradientMax':maximum,'projectedGradientMax':projected,'iterations':int(a.nit),'message':str(a.message)})
            if projected<=1e-5:break
        if projected>1e-5:raise RuntimeError('RC full type fit failed projected-gradient guard')
        traces.append({'start':i,'theta':theta.tolist(),'loss':loss,'gradientMax':maximum,'projectedGradientMax':projected,'checks':checks})
    best=min(traces,key=lambda r:(r['loss'],r['start']));return {'kind':packed.kind,'theta':best['theta'],'scaler':packed.scaler,'types':TYPES},{'seconds':time.monotonic()-start,'starts':traces,'selectedStart':best['start'],'maximumGradient':max(r['gradientMax']for r in traces),'maximumProjectedGradient':max(r['projectedGradientMax']for r in traces),'localTrainUids':packed.localTrainUids}


def probability(world,uid,model):
    teacher=world.teacher(uid);x=world.design(uid,model['kind'],model['scaler']);q=softmax(world.prior(uid)+x@np.array(model['theta']).reshape(x.shape[1],len(TYPES)-1)@CONTRAST.T,axis=1);out=teacher.copy();out[:,INDEX]=q*teacher[:,INDEX].sum(1,keepdims=True);return out
