"""Exact metric factorization for the same anchored Boolean tag function."""
import numpy as np


def metrics_under_hybrid(y,pre,family_trigger,w,hybrid_index):
    # pre: fine-policy x chart x class; trigger: family-policy x chart.
    p,n,c=pre.shape;ordinary=(~family_trigger).astype(float);active=family_trigger.astype(float)
    values=pre.transpose(1,0,2).reshape(n,p*c)
    tp=(ordinary@((w[:,None]*y)[:,None,:]*pre.transpose(1,0,2)).reshape(n,p*c)).reshape(len(ordinary),p,c)
    pm=(ordinary@(w[:,None]*values)).reshape(len(ordinary),p,c)
    tp[:,:,hybrid_index]+=(active@(w*y[:,hybrid_index]))[:,None];pm[:,:,hybrid_index]+=(active@w)[:,None]
    support=w@y;fp=pm-tp
    div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    precision=div(tp,pm).mean(-1);recall=div(tp,support).mean(-1);f1=div(2*tp,support+pm).mean(-1);micro=div(2*tp.sum(-1),support.sum()+pm.sum(-1))
    count=y.sum(1);sample=div(2*np.sum(pre&y.astype(bool),axis=2),pre.sum(2)+count);exact=np.all(pre==y,axis=2)
    hs=2*y[:,hybrid_index]/(1+count);he=(y[:,hybrid_index]>0)&(count==1)
    sample=(ordinary@(w[:,None]*sample.T)+(active@(w*hs))[:,None])/w.sum();exact=(ordinary@(w[:,None]*exact.T)+(active@(w*he))[:,None])/w.sum()
    return np.stack([precision,recall,f1,micro,sample,exact],axis=-1)
