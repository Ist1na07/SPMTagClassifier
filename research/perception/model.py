"""Small identifiable, monotone own-share chart gates; optional local residual.

Fine gates never inspect other class names or chart identities. Mixture gates
inspect only the same integral composition and physical family coverage.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from pressure_data import read,mil

ROOT=Path('.cache/perception-r1')
MIX=['hybrid','rc mix','ln mix']


def logit(x):return np.log(np.maximum(x,1e-12))-np.log(np.maximum(1-x,1e-12))


class Data:
    def __init__(self):
        read(ROOT/'complete.json');self.manifest=read(ROOT/'manifest.json');self.records=self.manifest['records'];self.lookup={r['uid']:i for i,r in enumerate(self.records)}
        self.main=np.array([self.lookup[u]for u in self.manifest['mainUids']]);self.manual=np.array([self.lookup[u]for u in self.manifest['manualUids']]);self.fine=self.manifest['fineClasses'];self.classes=self.manifest['mapClasses']
        with np.load(ROOT/'arrays.npz')as z:self.arrays={k:z[k].copy()for k in z.files}
        self.y=np.array([[float(c in r['tags'])for c in self.classes]for r in self.records]);self.weight=np.array([r['weight']for r in self.records]);self.cache={}
        # Independent degree of prototype support; overlap allowed. Numerical
        # endpoint regularization leaves room for a fitted log-odds residual.
        self.soft=np.minimum(self.arrays['support'],.999);self.hard=self.arrays['hard']
        self.rc=np.array([c in ['chordjack','minijack','dense chordstream','fast chordstream','speed','tech','vibro','anchor']for c in self.fine]);self.ln=~self.rc
    def complete_train(self,ids):
        ids=list(ids);groups={self.records[i]['group']for i in ids}
        return np.array(ids+[i for i,r in enumerate(self.records)if r['dataset']=='derived'and r['group']in groups],int)
    def representation(self,wid,power=1,hard=False,components=False):
        key=wid,power,hard,components
        if key not in self.cache:
            m=self.arrays['mass'][wid];s=self.hard if hard else self.soft**power;x=self.arrays['x'][wid]
            f,available,shares,family=map_features(self,m,s,x,components)
            self.cache[key]=(f,available,shares,family)
        return self.cache[key]


def map_features(d,mass,support,x,components=False):
    n=len(d.records);c=len(d.classes);features=np.zeros((n,c,8));available=np.zeros((n,c),bool);share=np.zeros((n,len(d.fine)));family=np.zeros((n,2));moments=np.zeros((n,len(d.fine),x.shape[1]))
    np.add.at(share,d.arrays['bag'],mass[:,None]*support)
    np.add.at(family,d.arrays['bag'],mass[:,None]*d.arrays['family'])
    if components:
        np.add.at(moments,d.arrays['bag'],mass[:,None,None]*support[:,:,None]*x[:,None,:])
        moments=np.divide(moments,share[:,:,None],out=np.zeros_like(moments),where=share[:,:,None]>0)
    for j,name in enumerate(d.fine):
        k=d.classes.index(name);features[:,k,0]=logit(share[:,j]);available[:,k]=share[:,j]>0
        if components:
            # Rbar always accompanied by total demand and other pressure roles.
            features[:,k,1:5]=moments[:,j,[5,2,3,4]]
    for name,mask,fi in [('rc mix',d.rc,0),('ln mix',d.ln,1)]:
        k=d.classes.index(name);p=share[:,mask];total=p.sum(1);q=np.divide(p,total[:,None],out=np.zeros_like(p),where=total[:,None]>0)
        top=np.sort(q,axis=1)[:,::-1];entropy=-np.sum(q*np.log(np.maximum(q,1e-15)),axis=1)/np.log(mask.sum())
        features[:,k,:6]=np.c_[logit(family[:,fi]),entropy,1-top[:,0],top[:,1],top[:,2],family[:,1-fi]]
        available[:,k]=(family[:,fi]>0)&(np.count_nonzero(p>0,axis=1)>=2)
    k=d.classes.index('hybrid');balance=2*np.minimum(family[:,0],family[:,1]);active=family.sum(1)
    features[:,k,:4]=np.c_[logit(np.clip(balance,0,1)),active,family[:,0],family[:,1]]
    available[:,k]=(family>0).all(1)
    return features,available,share,family


def scaler(raw,available,weight):
    w=weight[:,None]*available;den=w.sum(0);mean=np.divide(np.sum(raw*w[:,:,None],axis=0),den[:,None],out=np.zeros(raw.shape[1:]),where=den[:,None]>0)
    var=np.divide(np.sum((raw-mean)**2*w[:,:,None],axis=0),den[:,None],out=np.zeros_like(mean),where=den[:,None]>0)
    scale=np.maximum(np.sqrt(var),1e-6);keep=var>1e-12
    return dict(mean=mean.tolist(),scale=scale.tolist(),keep=keep.tolist())


def design(raw,s):
    z=np.where(s['keep'],(raw-np.array(s['mean']))/s['scale'],0.)
    # Intercept and actual TRAIN-standardized features; no validation clipping.
    return np.concatenate([np.ones((*raw.shape[:2],1)),z],axis=2)


def fit_gate(d,raw,available,train,cfg):
    start=time.monotonic();w=d.weight[train];w=w/w.sum();y=d.y[train];av=available[train];s=scaler(raw[train],av,w);x=design(raw[train],s);dimension=x.shape[-1];coef=np.zeros((len(d.classes),dimension));traces=[]
    for j,name in enumerate(d.classes):
        valid=av[:,j];a=x[valid,j];v=y[valid,j];ww=w[valid]
        if not valid.any():traces.append(dict(className=name,unobserved=True));continue
        prevalence=float(w@y[:,j]);power=cfg['balance'];pos=max(prevalence,.01)**(-power);neg=max(1-prevalence,.01)**(-power)
        weights=ww*np.where(v>0,pos,neg);weights/=max(weights.sum(),1e-15)
        prior=np.zeros(dimension);prior[0]=float(np.array(s['mean'])[j,0]);prior[1]=float(np.array(s['scale'])[j,0])if np.array(s['keep'])[j,0]else 0.
        # Prior is exactly z=logit(share); ridge makes parameter wells explicit.
        ridge=cfg['ridge'];keep=np.r_[True,np.array(s['keep'])[j]];bounds=[(None,None)]+[(0.,None)if i==0 else(None,None)for i in range(dimension-1)]
        for k in np.flatnonzero(~keep):bounds[k]=(0.,0.)
        def objective(theta):
            z=a@theta;loss=np.sum(weights*(np.logaddexp(0,z)-v*z))+.5*ridge*np.sum((theta-prior)**2)
            g=a.T@(weights*(expit(z)-v))+ridge*(theta-prior)
            return float(loss),g
        result=minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(gtol=1e-9,ftol=1e-14,maxiter=1000,maxls=40))
        loss,g=objective(result.x);pg=g.copy()
        for k,(lo,hi)in enumerate(bounds):
            if lo is not None and result.x[k]<=lo+1e-9 and pg[k]>0:pg[k]=0
            if hi is not None and result.x[k]>=hi-1e-9 and pg[k]<0:pg[k]=0
        maximum=float(np.max(np.abs(pg)))
        if maximum>1e-5:raise ValueError('Gate did not converge: '+name+' '+str(maximum))
        coef[j]=result.x;traces.append(dict(className=name,loss=loss,projectedGradient=maximum,iterations=int(result.nit),positiveMass=float(w@y[:,j])))
    return dict(coef=coef.tolist(),scaler=s,config=cfg),dict(seconds=time.monotonic()-start,classes=traces)


def probability(raw,available,model):return expit(np.sum(design(raw,model['scaler'])*np.asarray(model['coef'])[None,:,:],axis=2))*available


def predict(p,classes):
    out=p>=.5;mi=[classes.index(c)for c in MIX];fi=[j for j,c in enumerate(classes)if c not in MIX]
    best=np.argmax(p[:,mi],axis=1);score=np.max(p[:,mi],axis=1);use=(score>=.5)&(score>=np.max(p[:,fi],axis=1))
    out[:,mi]=False;out[use]=False;out[np.flatnonzero(use),np.asarray(mi)[best[use]]]=True
    return out


def metrics(y,p,w):
    tp=np.sum(w[:,None]*y*p,axis=0);fp=np.sum(w[:,None]*(1-y)*p,axis=0);fn=np.sum(w[:,None]*y*(1-p),axis=0)
    div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    precision=div(tp,tp+fp);recall=div(tp,tp+fn);f1=div(2*tp,2*tp+fp+fn)
    return dict(macroPrecision=float(precision.mean()),macroRecall=float(recall.mean()),macroF1=float(f1.mean()),
        microF1=float(2*tp.sum()/max(1e-15,2*tp.sum()+fp.sum()+fn.sum())),exactMatch=float(np.average(np.all(y==p,axis=1),weights=w)),
        sampleF1=float(np.average(div(2*np.sum(y*p,axis=1),np.sum(y+p,axis=1)),weights=w)),
        perClass=[dict(precision=float(a),recall=float(b),f1=float(c),tp=float(t),fp=float(f),fn=float(n))for a,b,c,t,f,n in zip(precision,recall,f1,tp,fp,fn)])
