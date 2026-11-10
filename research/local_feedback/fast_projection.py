"""Compiled execution of the same safeguarded scalar root, one CPU per worker."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def allocate(prior,delta,budget):
    n,k=prior.shape;out=np.empty((n,k));trial=np.empty(k)
    for i in range(n):
        lo=delta[i,0];hi=lo
        for j in range(1,k):lo=min(lo,delta[i,j]);hi=max(hi,delta[i,j])
        lam=(lo+hi)/2;total=0.
        for j in range(k):
            x=prior[i,j]+delta[i,j]-lam
            if x>=0:v=1/(1+math.exp(-x))
            else:e=math.exp(x);v=e/(1+e)
            out[i,j]=v;total+=v
        res=total-budget[i];done=False
        for _ in range(100):
            if abs(res)<=2e-13:done=True;break
            if res>0:lo=lam
            else:hi=lam
            mid=(lo+hi)/2;slope=0.
            for j in range(k):slope+=out[i,j]*(1-out[i,j])
            step=lam+res/slope if slope>0 else mid
            if not(lo<step<hi)or not math.isfinite(step):step=mid
            total=0.
            for j in range(k):
                x=prior[i,j]+delta[i,j]-step
                if x>=0:v=1/(1+math.exp(-x))
                else:e=math.exp(x);v=e/(1+e)
                trial[j]=v;total+=v
            new_res=total-budget[i]
            if abs(new_res)>.5*abs(res)and step!=mid:
                step=mid;total=0.
                for j in range(k):
                    x=prior[i,j]+delta[i,j]-step
                    if x>=0:v=1/(1+math.exp(-x))
                    else:e=math.exp(x);v=e/(1+e)
                    trial[j]=v;total+=v
                new_res=total-budget[i]
            lam=step;res=new_res
            for j in range(k):out[i,j]=trial[j]
        if not done and abs(res)>2e-13:raise ValueError('Compiled safeguarded projection did not converge')
    return out
