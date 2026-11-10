"""A mixture slot is not absence of its primitive actions.

Factorized chart outcome: categorical ordinary/RC-mix/LN-mix/Hybrid gate,
then own-share fine gates conditional on an ordinary output. This prevents
whole Mix labels from becoming false primitive negatives in local feedback.
"""
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
from model import scaler,design,expit,logit,fit_gate,probability,MIX

SLOTS=['ordinary','rc mix','ln mix','hybrid']
MIX_KEYS=['lnFraction','activeMass','rcEntropy','rcSecond','rcThird','lnEntropy','lnSecond','lnThird']


def mix_features(d,share,family):
    values=[np.divide(family[:,1],family.sum(1),out=np.zeros(len(family)),where=family.sum(1)>0),family.sum(1)]
    possible=[np.ones(len(family),bool)]
    for mask,fi in [(d.rc,0),(d.ln,1)]:
        p=share[:,mask];q=np.divide(p,p.sum(1,keepdims=True),out=np.zeros_like(p),where=p.sum(1,keepdims=True)>0);top=np.sort(q,axis=1)[:,::-1]
        values.extend([-np.sum(q*np.log(np.maximum(q,1e-15)),axis=1)/np.log(mask.sum()),top[:,1],top[:,2]])
        possible.append((family[:,fi]>0)&(np.count_nonzero(p>0,axis=1)>=2))
    possible.append((family>0).all(1))
    return np.array(values).T,np.array(possible).T


def slot_targets(d,ids):
    return np.array([next((SLOTS.index(c)for c in d.records[i]['tags']if c in SLOTS),0)for i in ids])


def fit_slots(d,raw,available,train,cfg):
    w=d.weight[train];w=w/w.sum();v=slot_targets(d,train);s=scaler(raw[train,None,:],np.ones((len(train),1),bool),w);x=design(raw[:,None,:],s)[:,0,:]
    freq=np.bincount(v,weights=w,minlength=4);adjusted=w*np.maximum(freq[v],.01)**(-cfg['balance']);adjusted/=adjusted.sum()
    prior=np.zeros((3,x.shape[1]));prior[:,0]=(1-cfg['balance'])*np.log(np.maximum(freq[1:],.01)/max(freq[0],.01))
    target=np.eye(4)[v];a=x[train];mask=available[train];ridge=cfg['ridge']
    infeasible=~mask[np.arange(len(v)),v];infeasible_mass=float(adjusted[infeasible].sum());adjusted=adjusted*(~infeasible)
    def objective(theta):
        coef=theta.reshape(prior.shape);z=np.c_[np.zeros(len(a)),a@coef.T];z=np.where(mask,z,-1e6)
        logp=z-logsumexp(z,axis=1,keepdims=True);p=np.exp(logp)
        loss=-np.sum(adjusted[:,None]*target*logp)+.5*ridge*np.sum((coef-prior)**2)
        g=((adjusted[:,None]*(p-target)*mask)[:,1:].T@a)+ridge*(coef-prior)
        return float(loss),g.ravel()
    theta=prior.ravel();passes=[]
    for _ in range(3):
        result=minimize(objective,theta,jac=True,method='L-BFGS-B',options=dict(gtol=1e-9,ftol=1e-15,maxiter=1500,maxls=40));theta=result.x;loss,g=objective(theta);passes.append(dict(loss=loss,gradientMax=float(np.abs(g).max()),iterations=int(result.nit)))
        if np.max(np.abs(g))<=1e-5:break
    if np.max(np.abs(g))>1e-5:raise ValueError('Slot gate not converged')
    model=dict(coef=result.x.reshape(prior.shape).tolist(),scaler=s,prior=prior.tolist(),config=cfg)
    return model,dict(loss=loss,gradientMax=float(np.max(np.abs(g))),iterations=int(result.nit),frequency=freq.tolist(),passes=passes,
        infeasibleTargetCount=int(infeasible.sum()),infeasibleWeightedMass=infeasible_mass,constantClippedNll=-np.log(1e-15)*infeasible_mass)


def slot_probability(raw,available,model):
    x=design(raw[:,None,:],model['scaler'])[:,0,:];z=np.c_[np.zeros(len(x)),x@np.asarray(model['coef']).T]
    return softmax(np.where(available,z,-1e6),axis=1)


def fit(d,raw,av,share,family,train,cfg):
    fit_av=av.copy()
    # Output slot supervision; primitives inside Mix remain unobserved rather
    # than false negative. The other treatment retains complete endpoint BCE.
    if cfg['conditional']:
        mixed=slot_targets(d,np.arange(len(d.records)))>0
        for c in d.fine:fit_av[mixed,d.classes.index(c)]=False
    fine,receipt=fit_gate(d,raw,fit_av,train,cfg)
    mixraw,mixav=mix_features(d,share,family);slot,trace=fit_slots(d,mixraw,mixav,train,cfg)
    return dict(fine=fine,slot=slot,config=cfg),dict(fine=receipt,slot=trace)


def probabilities(d,raw,av,share,family,model):
    fine=probability(raw,av,model['fine']);mr,ma=mix_features(d,share,family);slot=slot_probability(mr,ma,model['slot'])
    return fine,slot


def decode(d,fine,slot,fallback=True):
    out=np.zeros_like(fine,dtype=bool);fi=[d.classes.index(c)for c in d.fine];which=np.argmax(slot,axis=1);ordinary=which==0
    out[:,fi]=(fine[:,fi]>=.5)&ordinary[:,None]
    if fallback:
        empty=ordinary&(~out.any(1))&(fine[:,fi].max(1)>0);best=np.asarray(fi)[np.argmax(fine[:,fi],axis=1)];out[np.flatnonzero(empty),best[empty]]=True
    for j,name in enumerate(SLOTS[1:],1):out[which==j,d.classes.index(name)]=True
    return out
