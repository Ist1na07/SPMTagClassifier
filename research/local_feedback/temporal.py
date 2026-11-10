"""A one-direction temporal refinement, with a genuinely identifiable scalar.

Only Dense/Fast parents may expose a persistent local Tech subsegment. The
proposal is a newly computed short-time expert, NOT the parent score copied
to smaller intervals. All other primary classes and family routing stay put.
"""
from pathlib import Path
import sys,time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import read,write,logit,expit,metrics,PLANS
from anchored import predictions,base

ROOT=Path('.cache/local-feedback-temporal-r4b');TYPES=['dense chordstream','fast chordstream','tech']


class Data:
    def __init__(self):
        read(ROOT/'complete.json');m=read(ROOT/'manifest.json');self.records=m['records'];self.fine=m['fineClasses'];self.classes=m['mapClasses'];self.local_classes=self.fine+['wildcard'];self.lookup={r['uid']:i for i,r in enumerate(self.records)};self.main=np.array([self.lookup[u]for u in m['mainUids']]);self.manual=np.array([self.lookup[u]for u in m['manualUids']]);self.profile=m['profile'];self.variants=m['variants']
        with np.load(ROOT/'arrays.npz')as z:self.a={k:z[k].copy()for k in z.files}
        self.q0=self.a['base'];self.bag=self.a['bag'];self.mass=self.a['mass'];self.ti=np.array([self.fine.index(c)for c in TYPES]);self.triad=self.q0[:,self.ti];self.budget=self.triad.sum(1);self.cap=np.minimum(.999,self.budget);self.baseT=self.triad[:,2];self.ratio=np.divide(self.triad[:,:2],self.triad[:,:2].sum(1,keepdims=True),out=np.zeros_like(self.triad[:,:2]),where=self.triad[:,:2].sum(1,keepdims=True)>0)
        self.y=np.array([[c in r['tags']for c in self.classes]for r in self.records],float);self.weight=np.array([r['weight']for r in self.records]);self.baseline=base(self)
        self.frames=[f for r in self.records for f in r['frames']];self.starts=np.array([f['start']for f in self.frames]);self.ends=np.array([f['end']for f in self.frames]);self.parent=self.a['parent'];self.refinable=np.array([f['family']=='rc'and f['tag']in TYPES[:2]for f in self.frames])
        oldOdds=np.log(np.maximum(self.triad[:,2],1e-300))-np.log(np.maximum(self.triad[:,:2].sum(1),1e-300));self.boost=np.maximum(0.,self.a['local']-oldOdds);self.boost[:,~self.refinable]=0
        fraction=np.divide(self.baseT,self.cap,out=np.zeros_like(self.baseT),where=self.cap>0);self.prior=logit(fraction)
        self.base_share=np.zeros((len(self.records),len(self.fine)));np.add.at(self.base_share,self.bag,self.mass[:,None]*self.q0[:,:len(self.fine)])
        self.family=np.zeros((len(self.records),2));np.add.at(self.family,self.bag,self.mass[:,None]*np.array([[f['family']=='rc',f['family']=='ln']for f in self.frames]))
        self.frame_weight=np.zeros(len(self.frames));self.truthTech=np.zeros(len(self.frames));self.truthCS=np.zeros(len(self.frames));self.truthDense=np.zeros(len(self.frames));self.truthParent=np.zeros(len(self.frames));self.confusion_change=np.zeros(len(self.frames));self.tech_dense_error=np.zeros(len(self.frames));self.annotation_mass={};self.baseline_local={}
        for i,r in enumerate(self.records):
            a,b=r['offset'];self.frame_weight[a:b]=self.a['dt'][a:b]/self.a['dt'][a:b].sum()*self.weight[i]
            if i not in self.manual:continue
            total=correct=techdense=confusion=0.
            for ann in r['annotations']:
                if not ann['tags']:continue
                overlap=np.maximum(0,np.minimum(self.ends[a:b],ann['end'])-np.maximum(self.starts[a:b],ann['start']));total+=ann['end']-ann['start'];tags=ann['tags'];isT='tech'in tags;isCS=any(c in TYPES[:2]for c in tags)
                parent_allowed=np.array([f['tag']in tags for f in self.frames[a:b]]);self.truthParent[a:b]+=overlap*parent_allowed;self.truthTech[a:b]+=overlap*isT;self.truthCS[a:b]+=overlap*isCS;self.truthDense[a:b]+=overlap*('dense chordstream'in tags)
                old_td=np.array([isT and f['tag']=='dense chordstream'and not parent_allowed[j]for j,f in enumerate(self.frames[a:b])]);old_conf=np.array([not parent_allowed[j]and((isT and f['tag']in TYPES[:2])or(isCS and f['tag']=='tech'))for j,f in enumerate(self.frames[a:b])]);new_conf=isCS and not isT
                self.tech_dense_error[a:b]+=overlap*old_td;self.confusion_change[a:b]+=overlap*(new_conf-old_conf.astype(float))
                correct+=float(overlap@parent_allowed);techdense+=float(overlap@old_td);confusion+=float(overlap@old_conf)
            self.annotation_mass[i]=total;self.baseline_local[i]=dict(uid=r['uid'],accuracy=correct/total,correctMs=correct,totalMs=total,techToDenseMs=techdense,techCsConfusionMs=confusion)
        # A new Tech must actually beat all existing candidates, not merely the
        # inherited CS. Dense/Fast remain in their original conditional ratio.
        other=self.a['raw']/self.a['div'][:,None];other[:,self.ti]=0
        self.primary_threshold=np.maximum(self.budget*self.ratio.max(1)/(1+self.ratio.max(1)),other.max(1))

    def evidence(self,cfg):
        return self.boost[cfg['variant']]*((self.a['local'][cfg['variant']]>=cfg['minimumLocalLogOdds'])&(self.a['margin'][cfg['variant']]>=0)&self.refinable)

    def forward(self,kappa,cfg):
        feature=self.evidence(cfg);fraction=expit(self.prior+kappa*feature);q=self.cap*fraction;dq=self.cap*fraction*(1-fraction)*feature
        # Exact identity for unaffected/zero-evidence rows, including breaks.
        q=np.where(feature>0,q,self.baseT);dq=np.where(feature>0,dq,0.)
        delta=q-self.baseT;share=self.base_share.copy();share[:,self.ti[2]]+=np.bincount(self.bag,weights=self.mass*delta,minlength=len(self.records))
        for j in range(2):share[:,self.ti[j]]-=np.bincount(self.bag,weights=self.mass*self.ratio[:,j]*delta,minlength=len(self.records))
        return q,dq,share

    def predict(self,share):
        return predictions(self,share,self.family,self.baseline,[self.profile['policy']],[self.profile['family']])[0,0]

    def local(self,ids,q,minimum_run):
        changed=q>self.primary_threshold;changed&=self.refinable;out=[]
        for i in ids:
            a,b=self.records[i]['offset'];on=changed[a:b].copy()
            if minimum_run:
                boundary=np.r_[True,(on[1:]!=on[:-1])|(self.parent[a+1:b]!=self.parent[a:b-1])];starts=np.flatnonzero(boundary);ends=np.r_[starts[1:],b-a]
                for lo,hi in zip(starts,ends):
                    if on[lo]and self.ends[a+hi-1]-self.starts[a+lo]<minimum_run:on[lo:hi]=False
            base=self.baseline_local[int(i)];gain=float(on@(self.truthTech[a:b]-self.truthParent[a:b]));techdense=float(on@self.tech_dense_error[a:b]);confusion=float(on@self.confusion_change[a:b])
            out.append(dict(base,accuracy=(base['correctMs']+gain)/base['totalMs'],correctMs=base['correctMs']+gain,techToDenseMs=base['techToDenseMs']-techdense,techCsConfusionMs=base['techCsConfusionMs']+confusion))
        return out


class Kernel:
    def __init__(self,d,train,manual,cfg):
        self.d=d;self.train=np.array(train);self.manual=np.array(manual);self.cfg=cfg;self.dimension=1
        self.observed=np.zeros(len(d.frames),bool)
        for i in list(train)+list(manual):self.observed[slice(*d.records[i]['offset'])]=True
        self.tw=d.frame_weight*self.observed;self.tw/=self.tw.sum()
        fi=[d.classes.index(c)for c in TYPES];self.y=d.y[train][:,fi];w=d.weight[train];w/=w.sum();ordinary=np.array([not any(c in ['hybrid','rc mix','ln mix']for c in d.records[i]['tags'])for i in train]);target_family=np.any(self.y>0,axis=1)
        fixed_hybrid=(d.family[train,0]>=d.profile['family']['minRc'])&(d.family[train,1]>=d.profile['family']['minLn']);observable=(d.base_share[train][:,d.ti]>0)&(ordinary&target_family&~fixed_hybrid)[:,None]
        cw=w[:,None]*observable;self.cw=np.divide(cw,cw.sum(0,keepdims=True),out=np.zeros_like(cw),where=cw.sum(0,keepdims=True)>0)/3
        policy=d.profile['policy'];prior_mixed=d.baseline[train][:,[d.classes.index(c)for c in ['hybrid','rc mix','ln mix']]].any(1);self.threshold=np.where(d.baseline[train][:,fi],policy['keepFloor'],policy['addFloor']);self.threshold[prior_mixed]=policy['mixOverride']
        self.strong=[];self.sw=[];self.num0=[];self.slope=[];self.den=[]
        for i in manual:
            r=d.records[i];a,b=r['offset'];duration=d.annotation_mass[int(i)]
            for ann in r['annotations']:
                if not ann['tags']:continue
                mask=np.array([c in ann['tags']for c in d.local_classes],float)
                slope=mask[d.ti[2]]-d.ratio[a:b]@mask[d.ti[:2]];num=d.q0[a:b]@mask;den=d.q0[a:b].sum(1);dt=np.maximum(0,np.minimum(d.ends[a:b],ann['end'])-np.maximum(d.starts[a:b],ann['start']));valid=(dt>0)&(num>0)
                at=np.flatnonzero(valid);self.strong.extend((a+at).tolist());self.sw.extend((dt[at]/max(duration,1)/max(len(manual),1)).tolist());self.num0.extend(num[at].tolist());self.slope.extend(slope[at].tolist());self.den.extend(den[at].tolist())
        for k in ['strong','sw','num0','slope','den']:setattr(self,k,np.array(getattr(self,k),int if k=='strong'else float))

    def objective(self,theta):
        kappa=float(theta[0]);d=self.d;c=self.cfg;q,dq,share=d.forward(kappa,c);s=np.clip(share[self.train][:,d.ti],1e-12,1-1e-12);z=logit(s)-logit(self.threshold);weak=c['wholeWeight']*np.sum(self.cw*(np.logaddexp(0,z)-self.y*z));gs=np.zeros((len(d.records),3));gs[self.train]=c['wholeWeight']*self.cw*(expit(z)-self.y)/(s*(1-s))
        pull=self.d.mass*(gs[d.bag,2]-np.sum(gs[d.bag,:2]*d.ratio,axis=1));gradient=float(pull@dq);loss=weak
        p0=np.divide(d.baseT,d.budget,out=np.zeros_like(q),where=d.budget>0);p=np.divide(q,d.budget,out=np.zeros_like(q),where=d.budget>0);p=np.clip(p,1e-15,1-1e-15)
        kl=p0*np.log(np.maximum(p0,1e-15)/p)+(1-p0)*np.log(np.maximum(1-p0,1e-15)/(1-p));loss+=c['teacherWeight']*float(self.tw@kl)
        klpull=np.divide((p-p0),p*(1-p)*d.budget,out=np.zeros_like(q),where=d.budget>0);gradient+=c['teacherWeight']*float((self.tw*klpull)@dq)
        if c['strongWeight']and len(self.strong):
            at=self.strong;num=np.maximum(self.num0+self.slope*(q[at]-d.baseT[at]),1e-300);loss-=c['strongWeight']*float(self.sw@np.log(num/self.den));gradient-=c['strongWeight']*float((self.sw*self.slope/num)@dq[at])
        loss+=.5*c['ridge']*kappa*kappa;gradient+=c['ridge']*kappa
        return float(loss),np.array([gradient])


def fit(k):
    begin=time.monotonic();states=[]
    for start in [0.,1.,4.]:
        r=minimize(k.objective,np.array([start]),jac=True,method='L-BFGS-B',bounds=[(0.,None)],options=dict(gtol=1e-9,ftol=1e-14,maxiter=300,maxls=40));loss,g=k.objective(r.x);pg=0. if r.x[0]<=1e-9 and g[0]>0 else abs(float(g[0]))
        if pg>1e-5:raise ValueError('Temporal scalar not converged '+str(pg))
        states.append(dict(kappa=float(r.x[0]),loss=loss,projectedGradient=pg,iterations=int(r.nit),start=start))
    best=min(states,key=lambda s:s['loss']);return best['kappa'],dict(seconds=time.monotonic()-begin,starts=states,startLossSpread=max(s['loss']for s in states)-best['loss'])


if __name__=='__main__':
    d=Data();s=PLANS[0]['inner'][0];cfg=dict(variant=2,minimumLocalLogOdds=0.,ridge=.003,teacherWeight=.1,strongWeight=1.,wholeWeight=1.)
    k=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],cfg);t=np.array([.7]);eps=1e-5;l,g=k.objective(t);fd=(k.objective(t+eps)[0]-k.objective(t-eps)[0])/(2*eps)
    if abs(fd-g[0])>1e-7:raise ValueError('Temporal gradient mismatch')
    x,receipt=fit(k);q,_,share=d.forward(x,cfg);pred=d.predict(share);write('research/local_feedback/results/temporal-resource-r4b.json',dict(config=cfg,kappa=x,receipt=receipt,gradientError=float(abs(fd-g[0])),
        local=d.local(d.manual[s['strongValidation']],q,500),whole=metrics(d.y[d.main[s['validation']]],pred[d.main[s['validation']]],d.weight[d.main[s['validation']]])))
    print('temporal scalar',x,receipt,'local',d.local(d.manual[s['strongValidation']],q,500),flush=True)
