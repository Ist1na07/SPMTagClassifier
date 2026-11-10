"""Whole labels backpropagate through their actual time-integral readout.

No free pooled chart head or copied per-window truth. Local adapters have no
intercept, preserve unsupported branches/routing, and stay near the teacher.
"""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from model import Data,logit,design
from hierarchy import fit as fit_parent,mix_features,slot_targets,fit_slots

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
T=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float64)


def smooth_mix_features(d,share,family):
    values=[np.divide(family[:,1],family.sum(1),out=np.zeros(len(family)),where=family.sum(1)>0),family.sum(1)]
    for mask in [d.rc,d.ln]:
        p=share[:,mask];q=np.divide(p,p.sum(1,keepdims=True),out=np.zeros_like(p),where=p.sum(1,keepdims=True)>0)
        second=(q*q).sum(1);third=(q*q*q).sum(1)
        present=p.sum(1)>0
        values.extend([-np.sum(q*np.log(np.maximum(q,1e-15)),axis=1)/np.log(mask.sum()),present*(1-second),present*(1-3*second+2*third)])
    _,available=mix_features(d,share,family)
    return np.array(values).T,available


class Kernel:
    def __init__(self,d,train,manual,cfg,parent=None):
        self.d=d;self.cfg=cfg;self.train=np.array(train);self.manual=np.array(manual);wid=cfg['weightId'];self.wid=wid
        self.raw,self.av,self.share,self.family=d.representation(wid,cfg['power'],False,True)
        if parent is None:
            parent,_=fit_parent(d,self.raw,self.av,self.share,self.family,train,cfg)
            mr,ma=smooth_mix_features(d,self.share,self.family);parent['slot'],_=fit_slots(d,mr,ma,train,cfg)
        self.parent=parent;self.fi=[d.classes.index(c)for c in d.fine];nf=len(self.fi);nb=len(d.records)
        all_train=list(train)+list(manual)
        self.frame_ids=np.concatenate([np.arange(*d.records[i]['offset'])for i in all_train]);self.frame_mapper={int(j):i for i,j in enumerate(self.frame_ids)}
        self.base=np.maximum(0,d.soft[self.frame_ids]**cfg['power']);self.prior_logit=T(logit(self.base));self.supported=T(self.base>0)
        self.mass=T(d.arrays['mass'][wid,self.frame_ids]);self.bag=torch.tensor(d.arrays['bag'][self.frame_ids],dtype=torch.int64);self.xraw=T(d.arrays['x'][wid,self.frame_ids]);self.fam=T(self.family)
        weight=d.weight[train];weight=weight/weight.sum();self.weight=weight;self.y=d.y[train][:,self.fi]
        ordinary=slot_targets(d,train)==0;self.fine_available=self.av[train][:,self.fi]&ordinary[:,None]
        prev=weight@self.y;balance=cfg['balance'];cw=weight[:,None]*self.fine_available*np.where(self.y>0,np.maximum(prev,.01)**(-balance),np.maximum(1-prev,.01)**(-balance))
        cw=np.divide(cw,cw.sum(0,keepdims=True),out=np.zeros_like(cw),where=cw.sum(0,keepdims=True)>0);self.cw=T(cw);self.yt=T(self.y)
        s=parent['fine']['scaler'];self.fm=T(np.asarray(s['mean'])[self.fi]);self.fs=T(np.asarray(s['scale'])[self.fi]);self.fk=T(np.asarray(s['keep'])[self.fi])
        prior=np.zeros((nf,9));prior[:,0]=np.asarray(s['mean'])[self.fi,0];prior[:,1]=np.where(np.asarray(s['keep'])[self.fi,0],np.asarray(s['scale'])[self.fi,0],0.)
        self.fine_prior=T(prior)
        self.slot_scaler=parent['slot']['scaler'];self.sm=T(self.slot_scaler['mean'][0]);self.ss=T(self.slot_scaler['scale'][0]);self.sk=T(self.slot_scaler['keep'][0]);self.slot_prior=T(parent['slot']['prior'])
        self.slot_y=torch.tensor(slot_targets(d,train),dtype=torch.int64);freq=np.bincount(self.slot_y.numpy(),weights=weight,minlength=4);sw=weight*np.maximum(freq[self.slot_y.numpy()],.01)**(-balance);self.sw=T(sw/sw.sum())
        _,slotav=mix_features(d,self.share,self.family);self.slot_available=torch.tensor(slotav,dtype=torch.bool)
        self.sw*=T(slotav[train,self.slot_y.numpy()])
        # Fit only real observed feature directions. Per-class centering removes
        # an otherwise nearly confounded local intercept/whole threshold shift.
        fw=np.zeros(len(d.soft))
        for i in all_train:
            a,b=d.records[i]['offset'];dt=np.array([f['end']-f['start']for f in d.records[i]['frames']]);fw[a:b]=dt/dt.sum()*d.weight[i]/sum(d.weight[j]for j in all_train)
        observed=(self.base>0);lw=fw[self.frame_ids,None]*observed;den=lw.sum(0)
        raw=d.arrays['x'][wid,self.frame_ids];mean=np.divide(lw.T@raw,den[:,None],out=np.zeros((nf,raw.shape[1])),where=den[:,None]>0)
        variance=np.divide(np.einsum('nc,ncf->cf',lw,(raw[:,None,:]-mean)**2),den[:,None],out=np.zeros_like(mean),where=den[:,None]>0)
        keep=variance>1e-12;scale=np.where(keep,np.sqrt(variance),1.)
        self.local_scaler=dict(mean=mean.tolist(),scale=scale.tolist(),keep=keep.tolist())
        self.local_x=T(np.where(keep,(raw[:,None,:]-mean)/scale,0.));self.teacher_weight=T(lw/max(1,nf));self.teacher=T(self.base)
        fine_keep=np.c_[self.fine_available.any(0),np.asarray(s['keep'])[self.fi]];slot_keep=np.tile(np.r_[True,np.asarray(self.slot_scaler['keep'])[0]],(3,1))
        self.positions=[np.flatnonzero(keep.ravel()),np.flatnonzero(fine_keep.ravel()),np.flatnonzero(slot_keep.ravel())]
        self.shapes=[keep.shape,(nf,9),(3,9)];self.sizes=[len(p)for p in self.positions];self.dimension=sum(self.sizes)
        fine0=np.asarray(parent['fine']['coef'])[self.fi];slot0=np.asarray(parent['slot']['coef']);self.initial=np.concatenate([np.zeros(self.sizes[0]),fine0.ravel()[self.positions[1]],slot0.ravel()[self.positions[2]]])
        self.bounds=[(None,None)]*self.dimension;at=self.sizes[0]
        for j,p in enumerate(self.positions[1]):
            if p%9==1:self.bounds[at+j]=(0.,None)
        self.strong=[];self.strong_weight=[];self.strong_allowed=[];base_index=[j for j,c in enumerate(d.fine)if c not in ['anchor','vibro']]
        self.base_index=base_index
        for i in manual:
            row=d.records[i];labelled=sum(a['end']-a['start']for a in row['annotations']if a['tags']);offset=row['offset'][0]
            for a in row['annotations']:
                allowed=[j for j,c in enumerate(d.fine)if c in a['tags']and j in base_index]
                if not allowed:continue
                mask=np.zeros(nf);mask[allowed]=1
                for j,f in enumerate(row['frames']):
                    dt=max(0.,min(a['end'],f['end'])-max(a['start'],f['start']))
                    if dt:
                        self.strong.append(self.frame_mapper[offset+j]);self.strong_weight.append(dt/max(labelled,1)/max(1,len(manual)));self.strong_allowed.append(mask)
        self.strong_weight=T(self.strong_weight);self.strong_allowed=T(np.asarray(self.strong_allowed).reshape(-1,nf))

    def unpack(self,t):
        out=[];offset=0
        for shape,p,n in zip(self.shapes,self.positions,self.sizes):
            a=torch.zeros(int(np.prod(shape)),dtype=t.dtype).index_copy(0,torch.tensor(p,dtype=torch.int64),t[offset:offset+n]);out.append(a.reshape(shape));offset+=n
        return out

    def forward(self,t):
        local,coef,slotcoef=self.unpack(t)
        support=torch.sigmoid(self.prior_logit+(self.local_x*local[None,:,:]).sum(2))*self.supported
        weighted=self.mass[:,None]*support;nb=len(self.d.records);nf=len(self.d.fine)
        share=torch.zeros((nb,nf)).index_add(0,self.bag,weighted)
        moments=torch.zeros((nb,nf,8)).index_add(0,self.bag,weighted[:,:,None]*self.xraw[:,None,:])
        moments=moments/torch.clamp(share[:,:,None],min=1e-15)
        own=torch.log(torch.clamp(share,min=1e-12))-torch.log(torch.clamp(1-share,min=1e-12))
        raw=torch.cat([own[:,:,None],moments[:,:,[5,2,3,4]],torch.zeros((nb,nf,3))],dim=2)
        x=torch.cat([torch.ones((nb,nf,1)),((raw-self.fm)/self.fs)*self.fk],dim=2);fine_logits=(x*coef).sum(2)
        mix=[torch.where(self.fam.sum(1)>0,self.fam[:,1]/torch.clamp(self.fam.sum(1),min=1e-15),0),self.fam.sum(1)]
        for mask in [self.d.rc,self.d.ln]:
            p=share[:,mask];q=p/torch.clamp(p.sum(1,keepdim=True),min=1e-15);second=(q*q).sum(1);third=(q*q*q).sum(1)
            present=(p.sum(1)>0).to(q.dtype)
            mix.extend([-(q*torch.log(torch.clamp(q,min=1e-15))).sum(1)/np.log(mask.sum()),present*(1-second),present*(1-3*second+2*third)])
        mr=torch.stack(mix,dim=1);mx=torch.cat([torch.ones((nb,1)),((mr-self.sm)/self.ss)*self.sk],dim=1)
        slot_logits=torch.cat([torch.zeros((nb,1)),mx@slotcoef.T],dim=1);slot_logits=torch.where(self.slot_available,slot_logits,-1e6)
        return dict(support=support,share=share,fine_logits=fine_logits,slot_logits=slot_logits,local=local,coef=coef,slotcoef=slotcoef)

    def objective(self,theta):
        t=T(theta).requires_grad_(True);o=self.forward(t);c=self.cfg
        z=o['fine_logits'][self.train];fine=(self.cw*(torch.nn.functional.softplus(z)-self.yt*z)).sum()/len(self.d.fine)
        fine+=.5*c['ridge']*torch.sum((o['coef']-self.fine_prior)**2)/len(self.d.fine)
        logs=torch.log_softmax(o['slot_logits'][self.train],dim=1);slot=-(self.sw*logs[torch.arange(len(self.train)),self.slot_y]).sum()+.5*c['ridge']*torch.sum((o['slotcoef']-self.slot_prior)**2)
        q=torch.clamp(o['support'],min=1e-15,max=1-1e-15);p=self.teacher
        kl=(p*torch.log(torch.clamp(p,min=1e-15)/q)+(1-p)*torch.log(torch.clamp(1-p,min=1e-15)/(1-q)))
        anchor=c['teacherWeight']*(self.teacher_weight*kl).sum();regularization=.5*c['localL2']*torch.sum(o['local']**2)/len(self.d.fine)
        strong=torch.zeros(())
        if c['strongWeight']and self.strong:
            observed=o['support'][self.strong];den=observed[:,self.base_index].sum(1);num=(observed*self.strong_allowed).sum(1)
            strong=-c['strongWeight']*(self.strong_weight*torch.log(torch.clamp(num/torch.clamp(den,min=1e-15),min=1e-15))).sum()
        loss=fine+slot+anchor+regularization+strong;g,=torch.autograd.grad(loss,t)
        return float(loss.detach()),g.detach().numpy()

    def model(self,theta):
        with torch.no_grad():local,coef,slots=self.unpack(T(theta))
        return dict(config=self.cfg,localCoef=local.numpy().tolist(),localScaler=self.local_scaler,
            fineCoef=coef.numpy().tolist(),fineScaler=self.parent['fine']['scaler'],slotCoef=slots.numpy().tolist(),slotScaler=self.slot_scaler,
            trainUids=[self.d.records[i]['uid']for i in self.train],manualTrainUids=[self.d.records[i]['uid']for i in self.manual])


def fit(kernel,*,starts=2,warm=None):
    begin=time.monotonic();rng=np.random.default_rng(20261110);receipts=[];initial_loss=kernel.objective(kernel.initial)[0]
    seeds=[kernel.initial.copy()]
    for j in range(starts-1):
        t=kernel.initial.copy();t[:kernel.sizes[0]]=rng.normal(0,.03,kernel.sizes[0]);seeds.append(np.asarray(warm).copy()if warm is not None and j==0 else t)
    for seed in seeds:
        initial=kernel.objective(seed)[0];theta=seed.copy();passes=[]
        for _ in range(3):
            result=minimize(kernel.objective,theta,jac=True,method='L-BFGS-B',bounds=kernel.bounds,options=dict(gtol=1e-8,ftol=1e-15,maxiter=1600,maxls=50));theta=result.x
            loss,g=kernel.objective(theta);pg=g.copy()
            for j,(lo,hi)in enumerate(kernel.bounds):
                if lo is not None and theta[j]<=lo+1e-9 and pg[j]>0:pg[j]=0
                if hi is not None and theta[j]>=hi-1e-9 and pg[j]<0:pg[j]=0
            maximum=float(np.abs(pg).max());passes.append(dict(loss=loss,maximumProjectedGradient=maximum,iterations=int(result.nit),message=str(result.message)))
            if maximum<=1e-5:break
        if maximum>1e-5:raise ValueError('Unconverged joint model '+str(maximum))
        receipts.append(dict(theta=theta.tolist(),initialLoss=initial,loss=loss,maximumProjectedGradient=maximum,passes=passes))
    best=min(receipts,key=lambda r:r['loss'])
    if best['loss']>initial_loss+1e-8:raise ValueError('Joint fit worse than accepted frozen parent objective')
    if best['loss']>min(r['initialLoss']for r in receipts)+1e-8:raise ValueError('Joint fit worse than warm continuation seed')
    return best['theta'],dict(initialParentLoss=initial_loss,starts=receipts,seconds=time.monotonic()-begin,dimension=kernel.dimension,
        maximumProjectedGradient=max(r['maximumProjectedGradient']for r in receipts),startLossSpread=max(r['loss']for r in receipts)-best['loss'])


def infer(d,model):
    c=model['config'];wid=c['weightId'];base=d.soft**c['power'];s=model['localScaler'];x=d.arrays['x'][wid]
    local=np.where(s['keep'],(x[:,None,:]-np.array(s['mean']))/s['scale'],0.)
    q=1/(1+np.exp(-np.clip(logit(base)+np.sum(local*np.array(model['localCoef'])[None,:,:],axis=2),-700,700)));q*=base>0
    from model import map_features
    raw,av,share,family=map_features(d,d.arrays['mass'][wid],q,x,True);fi=[d.classes.index(c)for c in d.fine]
    full=design(raw,model['fineScaler']);fine=np.zeros((len(d.records),len(d.classes)));fine[:,fi]=1/(1+np.exp(-np.clip(np.sum(full[:,fi]*np.array(model['fineCoef'])[None,:,:],axis=2),-700,700)))*av[:,fi]
    mr,ma=smooth_mix_features(d,share,family);mx=design(mr[:,None,:],model['slotScaler'])[:,0,:];z=np.c_[np.zeros(len(mx)),mx@np.array(model['slotCoef']).T];z=np.where(ma,z,-1e6);z-=z.max(1,keepdims=True);slot=np.exp(z);slot/=slot.sum(1,keepdims=True)
    return dict(fine=fine,slot=slot,support=q,share=share,family=family)


def local_frames(d,index,support,config):
    record=d.records[index];a,b=record['offset'];base=d.soft[a:b]**config['power'];out=[]
    for j,f in enumerate(record['frames']):
        scores=dict(f['scores'])
        for k,c in enumerate(d.fine):
            if base[j,k]>0:scores[c]*=float(support[a+j,k]/base[j,k])
        ranked=sorted(scores.items(),key=lambda r:-r[1]);tag,value=ranked[0]
        if f['family']=='break':out.append(dict(f));continue
        if tag in ['wildcard','ln wildcard']:
            for c,v in ranked[1:]:
                if c not in ['wildcard','ln wildcard']and v>.3:tag,value=c,v;break
        secondary=ranked[1][0]if len(ranked)>1 and ranked[1][1]>=ranked[0][1]*.75 and ranked[1][1]>.3 else None
        out.append(dict(f,scores=scores,tag=tag,confidence=min(1.,value),secondary=secondary))
    return out
