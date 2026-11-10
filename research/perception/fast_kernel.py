"""Algebraically identical NumPy pullback for the frozen joint objective.

The torch Kernel remains the independent specification and gradient oracle.
This changes evaluation cost, never folds, targets, priors, or selection.
"""
import numpy as np
from scipy.special import expit,logsumexp
from joint import Kernel as TorchKernel


class FastKernel(TorchKernel):
    def objective(self,theta):
        if not hasattr(self,'_fast'):
            self._fast={k:getattr(self,k).detach().numpy()for k in ['prior_logit','supported','mass','bag','xraw','fam','cw','yt','fm','fs','fk','fine_prior','sm','ss','sk','slot_prior','slot_y','sw','slot_available','local_x','teacher_weight','teacher','strong_weight','strong_allowed']}
            bag=self._fast['bag'];starts=np.r_[0,np.flatnonzero(np.diff(bag)!=0)+1];groups=bag[starts]
            if len(np.unique(groups))!=len(groups):raise ValueError('Compact TRAIN groups are not contiguous')
            self._fast['starts']=starts;self._fast['groups']=groups
        a=self._fast;blocks=[];offset=0
        for shape,pos,n in zip(self.shapes,self.positions,self.sizes):
            value=np.zeros(int(np.prod(shape)));value[pos]=theta[offset:offset+n];blocks.append(value.reshape(shape));offset+=n
        local,coef,slotcoef=blocks;nf=len(self.d.fine);nb=len(self.d.records);c=self.cfg
        q=expit(a['prior_logit']+np.einsum('ncf,cf->nc',a['local_x'],local))*a['supported'];weighted=a['mass'][:,None]*q
        share=np.zeros((nb,nf));share[a['groups']]=np.add.reduceat(weighted,a['starts'],axis=0)
        moments=np.zeros((nb,nf,8));moments[a['groups']]=np.add.reduceat(weighted[:,:,None]*a['xraw'][:,None,:],a['starts'],axis=0);safe=np.maximum(share,1e-15);v=moments/safe[:,:,None]
        own=np.log(np.maximum(share,1e-12))-np.log(np.maximum(1-share,1e-12));raw=np.concatenate([own[:,:,None],v[:,:,[5,2,3,4]],np.zeros((nb,nf,3))],axis=2)
        x=np.concatenate([np.ones((nb,nf,1)),(raw-a['fm'])/a['fs']*a['fk']],axis=2);z=np.sum(x*coef[None,:,:],axis=2)
        train=self.train;zt=z[train];fine=np.sum(a['cw']*(np.logaddexp(0,zt)-a['yt']*zt))/nf+.5*c['ridge']*np.sum((coef-a['fine_prior'])**2)/nf
        gz=np.zeros_like(z);gz[train]=a['cw']*(expit(zt)-a['yt'])/nf
        gf=np.einsum('bc,bcf->cf',gz,x)+c['ridge']*(coef-a['fine_prior'])/nf
        gr=gz[:,:,None]*coef[None,:,1:]/a['fs']*a['fk'];gs=gr[:,:,0]*((share>=1e-12)/np.maximum(share,1e-12)+((1-share)>=1e-12)/np.maximum(1-share,1e-12))
        gv=np.zeros_like(moments);gv[:,:,[5,2,3,4]]=gr[:,:,1:5]
        # Smooth composition moments, including the exact masked-zero case.
        family=a['fam'];active=family.sum(1);mix=[np.divide(family[:,1],active,out=np.zeros(nb),where=active>0),active];groups=[]
        for mask in [self.d.rc,self.d.ln]:
            u=share[:,mask];total=u.sum(1,keepdims=True);p=u/np.maximum(total,1e-15);second=(p*p).sum(1);third=(p*p*p).sum(1);present=total[:,0]>0
            mix.extend([-np.sum(p*np.log(np.maximum(p,1e-15)),axis=1)/np.log(mask.sum()),present*(1-second),present*(1-3*second+2*third)]);groups.append((mask,p,total,present))
        mr=np.array(mix).T;mx=np.c_[np.ones(nb),(mr-a['sm'])/a['ss']*a['sk']];sz=np.c_[np.zeros(nb),mx@slotcoef.T];sz=np.where(a['slot_available'],sz,-1e6);logs=sz-logsumexp(sz,axis=1,keepdims=True)
        slot=-np.sum(a['sw']*logs[train,a['slot_y']])+.5*c['ridge']*np.sum((slotcoef-a['slot_prior'])**2)
        gsz=np.zeros_like(sz);gt=np.exp(logs[train]);gt[np.arange(len(train)),a['slot_y']]-=1;gsz[train]=a['sw'][:,None]*gt*a['slot_available'][train]
        gm=gsz[:,1:].T@mx+c['ridge']*(slotcoef-a['slot_prior']);gmr=(gsz[:,1:]@slotcoef[:,1:])/a['ss']*a['sk']
        for i,(mask,p,total,present)in enumerate(groups):
            g=gmr[:,2+3*i:5+3*i];gp=g[:,0,None]*(-(np.log(np.maximum(p,1e-15))+(p>=1e-15))/np.log(mask.sum()))
            gp+=g[:,1,None]*(-2*p)*present[:,None]+g[:,2,None]*(-6*p+6*p*p)*present[:,None]
            gu=gp/np.maximum(total,1e-15)-((total>=1e-15)*np.sum(gp*p,axis=1,keepdims=True)/np.maximum(total,1e-15))
            gs[:,mask]+=gu
        # Conditional moment quotient; its denominator clamp is part of spec.
        gshare=gs-np.sum(gv*v,axis=2)/safe*(share>=1e-15);gq=a['mass'][:,None]*(gshare[a['bag']]+np.einsum('ncf,nf->nc',gv[a['bag']],a['xraw'])/safe[a['bag']])
        qc=np.clip(q,1e-15,1-1e-15);p=a['teacher'];kl=p*np.log(np.maximum(p,1e-15)/qc)+(1-p)*np.log(np.maximum(1-p,1e-15)/(1-qc));anchor=c['teacherWeight']*np.sum(a['teacher_weight']*kl)
        gq+=c['teacherWeight']*a['teacher_weight']*(qc-p)/(qc*(1-qc))*((q>=1e-15)&(q<=1-1e-15))
        regularization=.5*c['localL2']*np.sum(local*local)/nf;strong=0.
        if c['strongWeight']and self.strong:
            obs=q[self.strong];den=obs[:,self.base_index].sum(1);num=np.sum(obs*a['strong_allowed'],axis=1);ratio=num/np.maximum(den,1e-15);weight=c['strongWeight']*a['strong_weight'];strong=-np.sum(weight*np.log(np.maximum(ratio,1e-15)))
            valid=ratio>=1e-15;gn=np.divide(-weight*valid,num,out=np.zeros_like(num),where=num>0);gd=np.divide(weight*valid*(den>=1e-15),den,out=np.zeros_like(den),where=den>0)
            primary=np.zeros(nf);primary[self.base_index]=1;np.add.at(gq,self.strong,gn[:,None]*a['strong_allowed']+gd[:,None]*primary)
        gl=np.einsum('nc,ncf->cf',gq*q*(1-q),a['local_x'])+c['localL2']*local/nf
        gradient=np.concatenate([g.ravel()[pos]for g,pos in zip([gl,gf,gm],self.positions)])
        return float(fine+slot+anchor+regularization+strong),gradient
