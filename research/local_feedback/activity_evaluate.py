"""Evaluate only the held-out rows, with the identical consumer/visible metric."""
import copy
import numpy as np
from activity import parent,ROUTES
from anchored import predictions
from competition import local_metrics
from reanalyze_allowed import corrected


def evaluate(k,theta,main,manual):
    d=k.d;ids=list(main)+list(manual);f=np.concatenate([np.arange(*d.records[i]['offset'])for i in ids]);x=k.all_x[f]
    z=np.c_[np.ones(len(f)),(x-k.scaler['mean'])@k.scaler['matrix']];p=parent.softmax(np.eye(3)[d.route[f]]*2+z@theta.reshape(k.shape)@k.contrast.T,axis=1)
    support=np.sum(p[:,1:,None]*d.expert[f],axis=1);starts=np.r_[0,np.flatnonzero(np.diff(d.arrays['bag'][f]))+1];out={}
    for j,name in enumerate(d.consumer_names):
        share=np.zeros((len(d.records),len(d.fine)));family=np.zeros((len(d.records),2));mass=d.a['mass'][j,f,None]
        share[ids]=np.add.reduceat(mass*support,starts,axis=0);family[ids]=np.add.reduceat(mass*np.eye(3)[p.argmax(1),1:],starts,axis=0);profile=d.profiles[name]
        out[name]=predictions(d,share,family,d.baseline,[profile['policy']],[profile['family']])[0,0][main]
    # Use the original actual score readout and smoothing without constructing
    # every unrelated chart's 500-ms frames at each validation call.
    proxy=copy.copy(d);proxy.records=[];proxy.frame_records=[];proxy.expert_raw=[];offset=0;pp=[];lookup={int(j):i for i,j in enumerate(f)}
    for i in manual:
        a,b=d.records[i]['offset'];proxy.records.append(dict(d.records[i],offset=(offset,offset+b-a)))
        proxy.frame_records.extend(d.frame_records[a:b]);proxy.expert_raw.extend(d.expert_raw[a:b]);pp.extend(p[[lookup[j]for j in range(a,b)]]);offset+=b-a
    if len(manual):
        proxy.frame_records=[(i,v)for (i,_),v in zip(proxy.frame_records,proxy.routed_scores(np.array(pp)))];local=corrected(local_metrics(proxy,np.arange(len(manual))))
    else:local=[]
    return out,local
