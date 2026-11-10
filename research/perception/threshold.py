"""Direct percentage gates: finite exact TRAIN threshold partitions.

This deliberately avoids a second inferred correctness-probability scale.
Hybrid is a two-FAMILY question, not a requirement for three fine types.
"""
import itertools
import numpy as np

POLICIES=[dict(hybrid=False,minRc=0.,minLn=0.,mixFloor=f,dominant=d)for f,d in itertools.product([.05,.1,.15,.2,.25],[.35,.5,.65,.8])]+[
    dict(hybrid=True,minRc=r,minLn=l,mixFloor=f,dominant=d)for r,l,f,d in itertools.product([.1,.18,.2,.3,.32,.4],[.1,.18,.2,.3,.32,.4],[.05,.1,.15,.2,.25],[.35,.5,.65,.8])]
SHRINK=[dict(reference=t,strength=l)for t,l in itertools.product([.1,.25,.4],[0.,.001,.01,.1])]


def coarse(d,share,family,policies=POLICIES):
    n=len(share);out=np.zeros((len(policies),n,len(d.classes)),bool);use=np.zeros((len(policies),n),bool)
    rc=family[:,0]>=family[:,1]
    for j,p in enumerate(policies):
        hybrid=(family[:,0]>=p['minRc'])&(family[:,1]>=p['minLn'])if p['hybrid']else np.zeros(n,bool)
        out[j,hybrid,d.classes.index('hybrid')]=True;use[j]|=hybrid
        for main,mask,name in [(rc,d.rc,'rc mix'),(~rc,d.ln,'ln mix')]:
            a=share[:,mask];mix=main&(~hybrid)&(np.sum(a>=p['mixFloor'],axis=1)>=3)&(a.max(1)<p['dominant'])
            out[j,mix,d.classes.index(name)]=True;use[j]|=mix
    return out,use


def train_thresholds(d,share,family,train,policies=POLICIES,shrink=SHRINK):
    _,use=coarse(d,share[train],family[train],policies);ordinary=~use;y=d.y[train];w=d.weight[train];fine=[d.classes.index(c)for c in d.fine]
    threshold=np.ones((len(policies),len(shrink),len(fine)));interval=np.zeros((*threshold.shape,2));objectives=np.zeros_like(threshold)
    for j,c in enumerate(fine):
        s=share[train,j];positive=np.unique(s[s>0]);order=np.argsort(-s,kind='stable');v=s[order]
        if not len(positive):continue
        count=np.searchsorted(-v,-positive,side='right');count=np.r_[count,0]
        lower=np.r_[0.,positive[:-1],positive[-1]];upper=np.r_[positive,np.nextafter(max(1.,positive[-1]),np.inf)]
        lower=np.nextafter(lower,upper)
        tp=np.c_[np.zeros(len(policies)),np.cumsum(ordinary[:,order]*(w*y[:,c])[order],axis=1)][:,count]
        fp=np.c_[np.zeros(len(policies)),np.cumsum(ordinary[:,order]*(w*(1-y[:,c]))[order],axis=1)][:,count]
        support=float(w@y[:,c]);den=support+tp+fp;f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)
        for k,h in enumerate(shrink):
            tau=np.maximum(lower,np.minimum(upper,h['reference']));penalty=h['strength']/(1+support)*(np.log(tau/h['reference'])**2)
            utility=f1-penalty[None,:]
            # Select a TRAIN-equivalent partition by utility, recall and lower
            # false positives; never use an outer chart to choose a flat point.
            selected=utility>=utility.max(1,keepdims=True)-1e-14
            max_tp=np.max(np.where(selected,tp,-np.inf),axis=1,keepdims=True);selected&=tp>=max_tp-1e-14
            min_fp=np.min(np.where(selected,fp,np.inf),axis=1,keepdims=True);selected&=fp<=min_fp+1e-14
            distance=np.abs(np.log(tau/h['reference']));best=np.argmin(np.where(selected,distance[None,:],np.inf),axis=1)
            threshold[:,k,j]=tau[best];interval[:,k,j,0]=lower[best];interval[:,k,j,1]=upper[best];objectives[:,k,j]=utility[np.arange(len(policies)),best]
    return threshold,interval,objectives


def predict(d,share,family,threshold,policies=POLICIES):
    out,use=coarse(d,share,family,policies);out=np.repeat(out[:,None,:,:],threshold.shape[1],axis=1)
    ordinary=(~use)[:,None,:,None];q=(share[None,None,:,:]>=threshold[:,:,None,:])&(share[None,None,:,:]>0)&ordinary
    fine=[d.classes.index(c)for c in d.fine]
    out[:,:,:,fine]=q
    return out


def batch_metrics(y,p,w):
    # p leading axes are representation/policy/shrink; reduce only chart axis.
    tp=np.einsum('...nc,nc->...c',p,y*w[:,None]);pm=np.einsum('...nc,n->...c',p,w);fp=pm-tp;support=w@y
    div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    precision=div(tp,tp+fp).mean(-1);recall=div(tp,support).mean(-1);f1=div(2*tp,support+tp+fp).mean(-1)
    micro=div(2*tp.sum(-1),support.sum()+tp.sum(-1)+fp.sum(-1))
    sample=np.sum(div(2*np.sum(p&y.astype(bool),axis=-1),np.sum(p,axis=-1)+np.sum(y,axis=-1))*w,axis=-1)/w.sum();exact=np.sum(np.all(p==y,axis=-1)*w,axis=-1)/w.sum()
    return np.stack([precision,recall,f1,micro,sample,exact],axis=-1)
