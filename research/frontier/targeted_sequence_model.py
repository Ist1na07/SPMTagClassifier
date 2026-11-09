"""Bounded ordered-row and hand-clock test, against the shipped v1.2.2 gate.

Native vs local2s contexts are separate treatments, not nested feature claims.
Only parents already labelled Tech/CS may change; all others stay exact.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp
from pressure_data import mil,read
from cut_consistency_model import Packed as BasePacked,fit as basic_fit,INDEX,TYPES
from rc_type_model import World as GenericWorld
from rc_action_runtime import smooth

INPUT=Path('.cache/overlay-targeted-sequence-r1.jsonl')
FEATURES={'bias':[],'sequence':[0,1],'time':[2,3],'global':[4,5],'both':[0,1,2,3]}
KEYS=read(str(INPUT)+'.schema.json')['keys']


class World:
    teacher=GenericWorld.teacher
    def __init__(self,context):
        self.context=context;self.main,self.manual,records=mil.load()
        self.records={r['uid']:r for r in records};self.inputs={};self.cache={};self.scalers={};self.designs={}
        read(str(INPUT)+'.complete.json')
        for row in map(json.loads,INPUT.read_text(encoding='utf8').splitlines()):
            uid=row['uid'];local=context=='local2s';f=row['native']['frames']if local else row['parents']
            effort=row['native']['effort']if local else row['originalEffort']
            self.inputs[uid]=dict(frames=f,intervals=[[s['start'],s['end']]for s in f],
                values=row['native']['values']if local else row['originalValues'],parents=row['parents'],
                parentIds=row['native']['parentIds']if local else list(range(len(f))))
            if uid not in self.records:self.records[uid]=dict(uid=uid,dataset='concept',source=row['source'])
            self.records[uid]['segments']=[dict(start=s['start'],end=s['end'],effort=e)for s,e in zip(f,effort)]
    def features(self,uid,kind):
        x=np.asarray(self.inputs[uid]['values'],float);return x[:,FEATURES[kind]]
    def feature_keys(self,kind):return [KEYS[i]for i in FEATURES[kind]]
    def design(self,uid,kind,scaler):
        key=uid,kind,id(scaler)
        if key not in self.designs:self.designs[key]=GenericWorld.design(self,uid,kind,scaler)
        return self.designs[key]
    def scaler(self,records,kind):
        key=tuple(r['uid']for r in records),kind
        if key in self.scalers:return self.scalers[key]
        arrays=[];weights=[]
        for r in records:
            row=self.inputs[r['uid']];take=np.array([s['family']=='rc'and s['tag']in TYPES for s in row['frames']])
            if take.any():arrays.append(self.features(r['uid'],kind)[take]);weights.append(np.full(take.sum(),r['recordWeight']/take.sum()))
        x=np.concatenate(arrays);w=np.concatenate(weights)
        mu=np.array([np.average(x[np.isfinite(x[:,j]),j],weights=w[np.isfinite(x[:,j])])if np.isfinite(x[:,j]).any()else 0. for j in range(x.shape[1])])
        expanded=np.c_[np.where(np.isfinite(x),x,mu),~np.isfinite(x)];mean=np.average(expanded,axis=0,weights=w)
        sd=np.sqrt(np.average((expanded-mean)**2,axis=0,weights=w));keep=np.ptp(expanded,axis=0)>0
        keys=self.feature_keys(kind)
        # Each pair is jointly missing by the extraction contract. Keep one
        # indicator per pair, not two redundant directions on a flat plateau.
        for j,index in enumerate(FEATURES[kind]):
            if index%2:
                if not np.array_equal(expanded[:,len(keys)+j-1],expanded[:,len(keys)+j]):raise ValueError('Joint observation contract changed')
                keep[len(keys)+j]=False
        out=dict(rawImputation=mu.tolist(),mean=mean.tolist(),scale=np.where(keep,sd,1.).tolist(),keep=keep.tolist(),expandedKeys=keys+[k+':missing'for k in keys])
        self.scalers[key]=out;return out
    def prior(self,uid):
        value=self.teacher(uid)[:,INDEX];mass=value.sum(1)
        q=np.divide(value,mass[:,None],out=np.full_like(value,1/3),where=mass[:,None]>0)
        with np.errstate(divide='ignore'):return np.log(q)


class Packed(BasePacked):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.active=np.zeros(len(self.x));self.pressure=np.ones(len(self.x));tags=[]
        for row in self.rows:
            a,b=self.offsets[row['uid']];f=self.world.inputs[row['uid']]['frames']
            self.active[a:b]=[s['family']=='rc'and s['tag']in TYPES for s in f]
            effort=np.maximum(1e-6,[s['effort']for s in row['segments']]);dt=np.array([s['end']-s['start']for s in f])
            self.pressure[a:b]=effort/np.average(effort,weights=dt);tags.extend(s['tag']for s in f)
        self.hard=np.array([float((tags[i]in TYPES)and not mask[mil.CLASSES.index(tags[i])]and mask[INDEX].any())*self.pressure[i]for i,mask,_ in self.strong])
    def log_distribution(self,theta):
        z=self.log_prior+((self.x@theta)*self.active)[:,None]*self.contrast.T
        return z-logsumexp(z,axis=1,keepdims=True)
    def pullback(self,theta,gradient):return (self.x.T@(gradient*self.active[:,None])@self.contrast).ravel()
    def objective(self,theta,cfg):
        loss,g=super().objective(theta,cfg)
        if cfg['strongWeight']and cfg['errorBoost']and self.strong:
            ix,subset,fixed,weight=self._strong_cache;q=self.distribution(theta);selected=np.sum(q[ix]*subset,axis=1)
            prob=fixed+self.mass[ix]*selected;extra=cfg['strongWeight']*cfg['errorBoost']*weight*self.hard
            loss-=float(extra@np.log(np.maximum(prob,1e-15)))
            factor=np.divide(extra*self.mass[ix],prob,out=np.zeros_like(prob),where=prob>1e-15);grad=np.zeros_like(q)
            np.add.at(grad,ix,factor[:,None]*q[ix]*(selected[:,None]-subset));g+=self.pullback(theta,grad)
        return float(loss),g


def frames(world,uid,model=None):
    row=world.inputs[uid]
    if model is None or not np.any(model['theta']):return [dict(f)for f in row['parents']]
    if not any(s['family']=='rc'and s['tag']in TYPES for s in row['parents']):return [dict(f)for f in row['parents']]
    x=world.design(uid,model['kind'],model['scaler']);source=row['frames'];teacher=world.teacher(uid)
    active=np.array([s['family']=='rc'and s['tag']in TYPES for s in source])
    z=world.prior(uid)+((x@np.asarray(model['theta']))*active)[:,None]*np.array([[0.,0.,1.]])
    q=np.exp(z-logsumexp(z,axis=1,keepdims=True));out=[]
    for j,f in enumerate(source):
        if not active[j]:out.append(dict(f));continue
        scores=dict(f['scores']);total=sum(scores.values())
        for c,i,v in zip(TYPES,INDEX,q[j]):scores[c]=float(v*teacher[j,INDEX].sum()*total)
        ranked=sorted(scores.items(),key=lambda r:-r[1]);tag,confidence=ranked[0]
        if tag in ['wildcard','ln wildcard']:
            for c,v in ranked[1:]:
                if c not in ['wildcard','ln wildcard']and v>.3:tag,confidence=c,v;break
        secondary=ranked[1][0]if len(ranked)>1 and ranked[1][1]>=ranked[0][1]*.75 and ranked[1][1]>.3 else None
        out.append({**f,'scores':scores,'tag':tag,'confidence':min(1.,confidence),'secondary':secondary})
    # No display flicker from identical adjacent labels. Coalesce only inside
    # one original parent; duration-weighted affinities are explicit new readout.
    merged=[];parents=[]
    for f,p in zip(out,row['parentIds']):
        if merged and parents[-1]==p and all(merged[-1].get(k)==f.get(k)for k in ['tag','secondary','family']):
            old=merged[-1];a=old['end']-old['start'];b=f['end']-f['start'];old['end']=f['end']
            old['scores']={k:(v*a+f['scores'][k]*b)/(a+b)for k,v in old['scores'].items()}
            old['confidence']=min(1.,old['scores'].get(old['tag'],1.))
        else:merged.append({**f,'scores':dict(f['scores'])});parents.append(p)
    return merged


def local_metrics(world,records,model=None):
    out=[]
    for row in records:
        raw=frames(world,row['uid'],model);fs=smooth(raw);eff=np.maximum(1e-6,[s['effort']for s in row['segments']])
        src=world.inputs[row['uid']]['frames'];starts=np.array([s['start']for s in src]);ends=np.array([s['end']for s in src]);dt=ends-starts;mean=np.average(eff,weights=dt)
        rates=eff/mean;prefix=np.r_[0.,np.cumsum(dt*rates)]
        def integral(t):
            i=int(np.searchsorted(ends,t,side='right'))
            return float(prefix[i]+(max(0,t-starts[i])*rates[i]if i<len(src)else 0.))
        correct=labelled=confusion=weighted=0.;details=[]
        for a in row['annotations']:
            if not a['tags']:continue
            labelled+=a['end']-a['start']
            for f in fs:
                d=max(0,min(a['end'],f['end'])-max(a['start'],f['start']))
                if not d:continue
                ok=f['tag']in a['tags'];correct+=d*ok
                bad=not ok and((f['tag']=='tech'and any(c in a['tags']for c in TYPES[:2]))or(f['tag']in TYPES[:2]and'tech'in a['tags']))
                if bad:
                    confusion+=d;pressure=integral(min(a['end'],f['end']))-integral(max(a['start'],f['start']))
                    weighted+=pressure;details.append(dict(start=max(a['start'],f['start']),end=min(a['end'],f['end']),truth=a['tags'],predicted=f['tag'],weightedMs=pressure))
        raw_correct=sum(max(0,min(a['end'],f['end'])-max(a['start'],f['start']))*(f['tag']in a['tags'])for a in row['annotations']if a['tags']for f in raw)
        out.append(dict(uid=row['uid'],accuracy=correct/labelled,rawAccuracy=raw_correct/labelled,confusionMs=confusion,pressureConfusion=weighted/labelled,details=details))
    return out
