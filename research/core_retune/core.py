"""Direct original softmax parameter fitting. Research only; no runtime edits.

Keep native parents, original vector/standardization, routing, rule attributes,
wildcard replacement, demotion and smoothing. Fit actual expert b/W, not an
external residual feature model. Local real excerpts are bags, not per-row truth.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,gzip,sys,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.linalg import helmert
from scipy.optimize import minimize


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def smooth(frames,minimum=1800):
    out=[]
    for f in frames:
        s=dict(f)
        if out and all(out[-1].get(k)==s.get(k)for k in ['tag','secondary','family'])and s['end']-s['start']<minimum:out[-1]['end']=s['end']
        else:out.append(s)
    result=[]
    for j,s in enumerate(out):
        previous=result[-1]if result else None;following=out[j+1]if j+1<len(out)else None
        if s['family']=='break'and s['end']-s['start']<minimum and previous is not None and following is not None and following['tag']==previous['tag']:previous['end']=s['end']
        else:result.append(s)
    return result


class Data:
    def __init__(self,input_path='.cache/core-retune-parents-r2.jsonl.gz',frozen_routes=False):
        self.frozen_routes=frozen_routes
        self.schema=read('.cache/core-retune-parents-r1.jsonl.gz.schema.json')
        with gzip.open(input_path,'rt',encoding='utf8')as f:self.rows=[json.loads(s)for s in f]
        manifest=read('.cache/local-feedback-activity-r6b/manifest.json');self.records={r['uid']:r for r in manifest['records']};self.manual=manifest['manualUids'];self.lookup={r['uid']:r for r in self.rows}
        self.offset={};parents=[]
        for r in self.rows:self.offset[r['uid']]=(len(parents),len(parents)+len(r['parents']));parents.extend(r['parents'])
        self.parents=parents;self.x=np.array([p['vector']for p in parents]);self.models=self.schema['model'];self.blocks={};self.classes=[]
        for family in ['rc','ln']:
            m=self.models[family];classes=m['classes'];wild='wildcard'if family=='rc'else'ln wildcard'
            names=classes+(['vibro']if family=='rc'else[]);mask=np.array([c!=wild for c in classes],float)
            z=(self.x-np.array(m['mean']))/np.array(m['std']);logits=z@np.array(m['W']).reshape(len(classes),-1).T+np.array(m['b'])
            fixed=np.array([[p['ruleRc']['wildcard']*.6, p['ruleRc']['vibro']*2]if family=='rc'else[p['ruleLn']['ln wildcard']*.6]for p in parents]);fixed=np.minimum(fixed,1.)
            self.blocks[family]=dict(classes=classes,names=names,wild=classes.index(wild),z=z,logits=logits,mask=mask,fixed=fixed)
            self.classes.extend(c for c in names if c not in self.classes)
        self.extra=[]
        for rnd in ['r4','r5','r6']:
            for name,a in read('research/frontier/annotations-'+rnd+'.json')['cases'].items():self.extra.append(dict(id=name,uid=a['sourceUid'],interval=a['sourceTargetMs'],tags=[a['localCategory']]))
        for name,a in read('research/frontier/annotations-r10.json')['originalLocalRevision']['cases'].items():self.extra.append(dict(id=name,uid=a['sourceUid'],interval=a['sourceTargetMs'],tags=[a['primaryCategory']]))
        self.observations=[]
        for uid in self.manual:
            record=self.records[uid];duration=sum(a['end']-a['start']for a in record['annotations']if a['tags'])
            for a in record['annotations']:
                if not a['tags']:continue
                for j,w in self.overlap(uid,a['start'],a['end']):self.observations.append(dict(id=uid,uid=uid,at=[j],weight=[1.],cost=w/duration/len(self.manual),tags=a['tags'],kind='strong'))
        for e in self.extra:
            overlap=self.overlap(e['uid'],*e['interval']);ids=[j for j,w in overlap];weights=np.array([w for j,w in overlap]);weights=weights/weights.sum()
            # Native-duration bag. Difficulty weighting is a separate comparison,
            # not silently substituted into the existing local annotation.
            self.observations.append(dict(id=e['id'],uid=e['uid'],at=ids,weight=weights.tolist(),cost=1/len(self.extra),tags=e['tags'],kind='bag'))
        self.baseline=self.frames({})
        for j,(p,f)in enumerate(zip(self.parents,self.baseline)):
            b=p['baseline']
            if b['tag']!=f['tag']or b.get('secondary')!=f.get('secondary'):raise ValueError('Native decision parity '+str(j))
            for c,v in b.get('scores',{}).items():
                if abs(v-f['scores'][c])>1e-12:raise ValueError('Native score parity '+str(j))

    def overlap(self,uid,start,end):
        a,b=self.offset[uid];return[(j,w)for j in range(a,b)if(w:=max(0,min(end,self.parents[j]['end'])-max(start,self.parents[j]['start'])))>0]

    def scores(self,family,delta=None,gain=1.4,wild_scale=1.):
        b=self.blocks[family];p=softmax(b['logits']+(0 if delta is None else delta),axis=1);s=gain*p*b['mask'];s[:,b['wild']]=wild_scale*b['fixed'][:,0]
        if family=='rc':s=np.c_[s,b['fixed'][:,1]]
        return s,p

    def frames(self,changes,gains=None,wild_scales=None,confidence_floor=.5,wildcard_rest_exemption=False):
        scores={f:self.scores(f,changes.get(f),(gains or{}).get(f,1.4),(wild_scales or{}).get(f,1.))[0]for f in self.blocks};out=[]
        # Native post-competition easyAmbiguous check must be recomputed below.
        for j,p in enumerate(self.parents):
            old=dict(p['preLate']);family=old['family']
            if family=='break':out.append(old);continue
            b=self.blocks[family];values=dict(zip(b['names'],map(float,scores[family][j])));ranked=sorted(values.items(),key=lambda v:-v[1]);tag,value=ranked[0]
            if tag in ['wildcard','ln wildcard']:
                for c,v in ranked[1:]:
                    if c not in ['wildcard','ln wildcard']and v>.3:tag,value=c,v;break
            second=ranked[1][0]if ranked[1][1]>=ranked[0][1]*.75 and ranked[1][1]>.3 else None
            old.update(tag=tag,secondary=second,confidence=min(1.,value),scores=values);out.append(old)
        # Initial family routing and early Break stay frozen. Re-evaluate the
        # confidence-dependent late gate for initially non-Break parents.
        for row in []if self.frozen_routes else self.rows:
            a,b=self.offset[row['uid']];ps=self.parents[a:b];fs=out[a:b];active=[j for j,p in enumerate(ps)if p['preLate']['family']!='break']
            diffs=[p['f']['nps']*(.5+p['f']['chordMean']/3)*(.6+p['f']['jackSpeedBpm']/150)for p in ps]
            ds=sorted(diffs[j]for j in active);p15=ds[int(len(ds)*.15)]if ds else float('inf')
            ns=sorted(p['f']['nps']for p in ps if p['extra']['rowCount']);mapn=ns[len(ns)//2]if ns else 0
            alln=sorted(p['f']['nps']for p in ps);medn=alln[len(alln)//2]if alln else 0
            for j in active:
                n=ps[j]['f']['nps'];context=ps[j].get('context',[ps[j]['start'],ps[j]['end']]);dur=context[1]-context[0];easy=fs[j]['confidence']<confidence_floor and diffs[j]<=max(p15,1)and n<mapn and not(wildcard_rest_exemption and fs[j]['tag']=='ln wildcard');sparse=dur<20000 and 0<n<medn*.4 and n<4
                if easy or sparse:fs[j].update(tag='break',family='break',secondary=None)
        return out

    def local(self,frames,uids=None):
        details=[]
        for uid in self.manual if uids is None else uids:
            a,b=self.offset[uid];fs=smooth(frames[a:b]);total=correct=td=conf=0.;counts={}
            for ann in self.records[uid]['annotations']:
                tags=ann['tags']
                if not tags:continue
                total+=ann['end']-ann['start']
                for f in fs:
                    dt=max(0.,min(ann['end'],f['end'])-max(ann['start'],f['start']));tag=f['tag'];ok=tag in tags;correct+=dt*ok
                    td+=dt*(not ok and 'tech'in tags and tag=='dense chordstream');conf+=dt*(not ok and(('tech'in tags and tag in ['dense chordstream','fast chordstream'])or(tag=='tech'and any(c in tags for c in ['dense chordstream','fast chordstream']))))
                    for c in set(tags)|{tag}:
                        v=counts.setdefault(c,dict(tp=0.,fp=0.,fn=0.));v['tp']+=dt*(tag==c and ok);v['fp']+=dt*(tag==c and not ok);v['fn']+=dt*(c in tags and not ok)
            details.append(dict(uid=uid,accuracy=correct/total,techToDenseMs=td,techCsConfusionMs=conf,perClass=counts))
        return dict(accuracy=float(np.mean([r['accuracy']for r in details])),techToDenseMs=sum(r['techToDenseMs']for r in details),techCsConfusionMs=sum(r['techCsConfusionMs']for r in details),details=details)

    def real(self,frames):
        result=[]
        for e in self.extra:
            a,b=self.offset[e['uid']];fs=smooth(frames[a:b]);s,t=e['interval'];duration=t-s;correct=sum(max(0,min(t,f['end'])-max(s,f['start']))*(f['tag']in e['tags'])for f in fs)
            result.append(dict(id=e['id'],uid=e['uid'],accuracy=correct/duration))
        return dict(accuracy=float(np.mean([r['accuracy']for r in result])),details=result)


class Kernel:
    def __init__(self,d,family,uids,mode='weights',ridge=.01,extra_weight=1.):
        self.d=d;self.family=family;self.ridge=ridge;b=d.blocks[family];self.block=b;self.contrast=helmert(len(b['classes']),full=False).T;self.obs=[]
        for o in d.observations:
            if o['uid']not in uids:continue
            at=np.array(o['at']);weight=np.array(o['weight']);valid=np.array([d.parents[j]['preLate']['family']==family for j in at]);at=at[valid];weight=weight[valid]
            allowed=np.array([c in o['tags']for c in b['names']],float)
            if not len(at)or not allowed.any():continue
            weight=weight/weight.sum();self.obs.append(dict(at=at,w=weight,cost=o['cost']*(extra_weight if o['kind']=='bag'else 1),allowed=allowed))
        at=np.unique(np.concatenate([o['at']for o in self.obs]))if self.obs else np.array([],int)
        self.ids=at;look={int(j):i for i,j in enumerate(at)}
        for o in self.obs:o['at']=np.array([look[int(j)]for j in o['at']])
        # Scale correction directions using TRAIN only. Original b/W and their
        # standardization remain the exact zero-correction starting point.
        x=b['z'];mean=x[at].mean(0);sd=x[at].std(0);keep=np.flatnonzero(sd>1e-6)if mode=='weights'else np.array([],int)
        self.keep=keep;self.mean=mean;self.sd=sd;self.x=np.c_[np.ones(len(x)),(x[:,keep]-mean[keep])/sd[keep]]
        self.shape=(self.x.shape[1],len(b['classes'])-1);self.size=int(np.prod(self.shape));self.training_x=self.x[at]

    def objective(self,theta):
        b=self.block;z=b['logits'][self.ids]+self.training_x@theta.reshape(self.shape)@self.contrast.T;p=softmax(z,axis=1);s=1.4*p*b['mask'];s[:,b['wild']]=b['fixed'][self.ids,0]
        if self.family=='rc':s=np.c_[s,b['fixed'][self.ids,1]]
        ds=np.zeros_like(s);loss=0.
        for o in self.obs:
            at=o['at'];w=o['w'];allowed=o['allowed'];v=w@s[at];num=max(1e-300,float(v@allowed));den=max(1e-300,float(v.sum()));cost=o['cost'];loss+=cost*(np.log(den)-np.log(num));np.add.at(ds,at,cost*w[:,None]*(1/den-allowed/num))
        dp=1.4*ds[:,:len(b['classes'])]*b['mask'];dz=p*(dp-(dp*p).sum(1,keepdims=True));coef=theta.reshape(self.shape)
        loss+=.5*self.ridge*np.sum(coef*coef);g=self.training_x.T@dz@self.contrast+self.ridge*coef
        return float(loss),g.ravel()

    def changes(self,theta):return self.x@theta.reshape(self.shape)@self.contrast.T

    def export(self,theta):
        # Fold correction back into actual archived W/b coordinates.
        dc=theta.reshape(self.shape)@self.contrast.T;dw=np.zeros((len(self.block['classes']),self.block['z'].shape[1]));dw[:,self.keep]=dc[1:].T/self.sd[self.keep];db=dc[0]-dw@self.mean
        original=self.d.models[self.family];return dict(classes=self.block['classes'],keys=original['keys'],mean=original['mean'],std=original['std'],W=(np.array(original['W']).reshape(dw.shape)+dw).ravel().tolist(),b=(np.array(original['b'])+db).tolist())


def fit(k):
    receipts=[];best=None
    for seed in [0,1]:
        start=time.monotonic();x=np.zeros(k.size)if seed==0 else np.random.default_rng(8021).normal(0,.02,k.size)
        r=minimize(k.objective,x,jac=True,method='L-BFGS-B',options=dict(maxiter=2000,gtol=1e-7,ftol=1e-13,maxls=40));loss,g=k.objective(r.x);receipts.append(dict(start=seed,loss=loss,maximumGradient=float(abs(g).max()),iterations=r.nit,success=bool(r.success),message=str(r.message),seconds=time.monotonic()-start))
        if best is None or loss<best.fun:best=r
    return best.x,receipts
