"""Exact joint fine/mix operating-point search, with no forced argmax tag."""
import numpy as np
from pressure_data import mil

MIX=[mil.MAP_CLASSES.index(c)for c in ['rc mix','ln mix','hybrid']]
FINE=[i for i in range(len(mil.MAP_CLASSES))if i not in MIX]


def thresholds(p):
    u=np.unique(p[p>0])
    if not len(u):return np.array([np.nextafter(1.,2.)])
    middle=(u[:-1]+u[1:])/2;middle=np.where(middle<=u[:-1],u[1:],middle)
    return np.r_[0.,middle,np.nextafter(1.,2.)]


def predict(p,fine,mix,*,gate=None):
    q=np.zeros(p.shape,int);mi=np.argmax(p[:,MIX],axis=1);mp=p[np.arange(len(p)),np.array(MIX)[mi]]
    use=(mp>=mix)&(mp>0)&(mp>=p[:,FINE].max(1))if gate is None else(gate>=mix)&(gate>0)
    q[np.flatnonzero(use),np.array(MIX)[mi[use]]]=1
    q[:,FINE]=((p[:,FINE]>=fine)&(p[:,FINE]>0)&(~use[:,None])).astype(int)
    return q


def optimize(y,p,w,*,gate=None):
    tf=thresholds(p[:,FINE]);mi=np.argmax(p[:,MIX],axis=1);mp=p[np.arange(len(p)),np.array(MIX)[mi]]
    if gate is None:valid=(mp>0)&(mp>=p[:,FINE].max(1))
    else:mp=gate;valid=mp>0
    tm=thresholds(mp[valid]);use=valid[None,:]&(mp[None,:]>=tm[:,None]);eligible=~use
    support=w@y;score=np.zeros((len(tm),len(tf)));recall=np.zeros_like(score)
    for c in FINE:
        order=np.argsort(-p[:,c],kind='stable');v=p[order,c]
        counts=np.searchsorted(-v,-tf,side='right');counts=np.minimum(counts,int((v>0).sum()))
        tp=np.c_[np.zeros(len(tm)),np.cumsum(eligible[:,order]*(w*y[:,c])[order],axis=1)][:,counts]
        fp=np.c_[np.zeros(len(tm)),np.cumsum(eligible[:,order]*(w*(1-y[:,c]))[order],axis=1)][:,counts]
        denominator=support[c]+tp+fp
        score+=np.divide(2*tp,denominator,out=np.zeros_like(tp),where=denominator>0)
        if support[c]>0:recall+=tp/support[c]
    for j,c in enumerate(MIX):
        q=use*(mi[None,:]==j);tp=q@(w*y[:,c]);fp=q@(w*(1-y[:,c]));denominator=support[c]+tp+fp
        score+=np.divide(2*tp,denominator,out=np.zeros_like(tp),where=denominator>0)[:,None]
        if support[c]>0:recall+=(tp/support[c])[:,None]
    score/=len(mil.MAP_CLASSES);recall/=len(mil.MAP_CLASSES)
    # Equal macro-F1 prefers recall, then the lower threshold representatives.
    candidates=np.flatnonzero(score.ravel()==score.max())
    best_recall=recall.ravel()[candidates].max()
    # Ascending row-major tm/tf already implements the remaining lower-
    # threshold tie break. Avoid sorting every inferior joint partition.
    at=candidates[np.flatnonzero(recall.ravel()[candidates]==best_recall)[0]]
    best=np.unravel_index(at,score.shape)
    return {'fine':float(tf[best[1]]),'mix':float(tm[best[0]]),'macroF1':float(score[best]),'macroRecall':float(recall[best]),
        'distinctFinePartitions':len(tf),'distinctMixPartitions':len(tm),'jointPartitions':int(score.size)}
