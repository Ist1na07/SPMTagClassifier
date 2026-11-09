"""Shared bag/interval/rank math with explicit type indices and allowed sets.

No global taxonomy mutation. Active historical LN producer remains unchanged;
its objective is an independent parity control for singleton bag targets.
"""
import numpy as np
from scipy.special import softmax,logsumexp


class TypeBagObjective:
    def distribution(self,theta):
        return np.exp(self.log_distribution(theta))

    def log_distribution(self,theta):
        z=self.log_prior+self.x@theta.reshape(self.x.shape[1],len(self.types)-1)@self.contrast.T
        return z-logsumexp(z,axis=1,keepdims=True)

    def pullback(self,theta,gradient):
        return (self.x.T@gradient@self.contrast).ravel()

    def bag_arrays(self,cfg):
        key=(cfg['effortPower'],cfg.get('weakBalance',False))
        if not hasattr(self,'_bag_cache'):self._bag_cache={}
        if key in self._bag_cache:return self._bag_cache[key]
        bags=self.weak+self.concepts;n=len(bags);length=max(r['range'][1]-r['range'][0]for r in bags);index=np.zeros((n,length),int);valid=np.zeros((n,length),bool);weight=np.zeros((n,length));factor=np.array([r['weight']for r in bags]);allowed=np.zeros((n,len(self.types)),bool)
        for j,row in enumerate(bags):
            a,b=row['range'];count=b-a;index[j,:count]=np.arange(a,b);valid[j,:count]=True;v=row['duration']*row['effort']**cfg['effortPower'];weight[j,:count]=v/v.sum();allowed[j,row.get('allowed',[row.get('target')])]=True
        if key[1]:
            target=np.array([r['target']for r in self.weak]);factor[:len(self.weak)]/=np.count_nonzero(self.class_mass)*self.class_mass[target]
        wm=weight*self.mass[index];logwm=np.full(wm.shape,-np.inf);logwm[wm>0]=np.log(wm[wm>0]);fixed=np.sum(weight[:,:,None]*self.teacher[index][:,:,self.other],axis=1);logfixed=np.full(fixed.shape,-np.inf);logfixed[fixed>0]=np.log(fixed[fixed>0]);eligible=np.any(wm>0,axis=1)
        result=(index,valid,logwm,logfixed,eligible,factor,allowed);self._bag_cache[key]=result;return result

    def objective(self,theta,cfg):
        power=cfg['primaryPower'];logq=self.log_distribution(theta);q=np.exp(logq);gradient=np.zeros_like(q)
        index,valid,logwm,logfixed,eligible,factor,allowed=self.bag_arrays(cfg);factor=factor.copy();factor[len(self.weak):]*=cfg['conceptWeight'];local_log=logq[index];logQ=logsumexp(logwm[:,:,None]+local_log,axis=1);logQ[~eligible]=0.;logits=power*np.c_[logQ,logfixed];primary=softmax(logits,axis=1);restricted=np.where(allowed,logits[:,:len(self.types)],-np.inf);nll=logsumexp(logits,axis=1)-logsumexp(restricted,axis=1);nll=np.where(eligible,nll,-np.log(1e-15));loss=float(factor@nll);gQ=primary[:,:len(self.types)]-softmax(restricted,axis=1);gQ*=(power*factor*eligible)[:,None];safe_logQ=np.where(np.isfinite(logQ),logQ,0.);responsibility=np.exp(logwm[:,:,None]+local_log-safe_logQ[:,None,:])*gQ[:,None,:];local_gradient=responsibility-q[index]*responsibility.sum(2,keepdims=True);np.add.at(gradient,index[valid],local_gradient[valid])
        if self.strong and cfg['strongWeight']:
            if not hasattr(self,'_strong_cache'):
                ix=np.array([r[0]for r in self.strong]);mask=np.array([r[1]for r in self.strong]);weight=np.array([r[2]for r in self.strong]);subset=mask[:,self.index];fixed=np.sum(self.teacher[ix]*mask,axis=1)-np.sum(self.teacher[ix][:,self.index]*subset,axis=1);self._strong_cache=(ix,subset,fixed,weight)
            ix,subset,fixed,weight=self._strong_cache;selected=np.sum(q[ix]*subset,axis=1);prob=fixed+self.mass[ix]*selected;loss-=cfg['strongWeight']*float(weight@np.log(np.maximum(prob,1e-15)));factor=np.divide(cfg['strongWeight']*weight*self.mass[ix],prob,out=np.zeros_like(prob),where=prob>1e-15);np.add.at(gradient,ix,factor[:,None]*q[ix]*(selected[:,None]-subset))
        for j,rank,w in self.ranks:
            if not cfg['strongWeight']or not self.mass[j]:continue
            available=np.ones(len(self.types),bool)
            for target in rank:
                logits=np.where(available,power*logq[j],-np.inf);p=softmax(logits);loss+=cfg['strongWeight']*w*(logsumexp(logits)-logits[target]);g=p;g[target]-=1;gradient[j]+=cfg['strongWeight']*w*power*g;available[target]=False
        loss+=cfg['localL2']*(theta@theta)/2;g=self.pullback(theta,gradient)+cfg['localL2']*theta;return float(loss),g
