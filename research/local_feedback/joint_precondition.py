"""Exact change of coordinates, preserving the original loss and ridge.

Global unlabeled-chart whitening is not the curvature of an objective with
burden-weighted interval bags. Use TRAIN exposure covariance to precondition
the optimizer, not to silently redefine the statistical regularizer.
"""
import time
import numpy as np
from scipy.optimize import minimize


def covariance(k):
    z=k.z;strong=np.zeros((z.shape[1],z.shape[1]));extra=strong.copy();weak=strong.copy()
    if len(k.strong_at):
        x=z[k.strong_at];strong=x.T@(k.strong_cost[:,None]*x)
    for e in k.extra:
        x=z[e['at']];extra+=x.T@(e['weight'][:,None]*x)/max(len(k.extra),1)
    weights=k.d.weight[k.train];weights=weights/weights.sum();lookup={int(i):v for i,v in zip(k.train,weights)};w=np.array([lookup.get(int(b),0.)for b in k.bag])*k.mass
    weak=z.T@(w[:,None]*z)
    return strong,extra,weak


class Coordinates:
    def __init__(self,k,components=None):
        self.k=k;self.components=components or covariance(k);a,b,c=self.components;cfg=k.cfg;matrix=cfg['strongWeight']*(a+cfg['extraWeight']*b)+cfg['wholeWeight']*c;e,v=np.linalg.eigh(matrix);e=np.maximum(e,0);self.transforms=[]
        for block in k.blocks:
            if not block['size']:self.transforms.append(None);continue
            ridge=cfg['ridge']/block['shape'][1];values=e+ridge;M=(v/np.sqrt(values))@v.T;inverse=(v*np.sqrt(values))@v.T;self.transforms.append((M,inverse))
        positive=e[e>1e-12];self.receipt=dict(covarianceMinimum=float(e.min()),covarianceMaximum=float(e.max()),nonzeroCondition=float(positive.max()/positive.min())if len(positive)else None,rows=len(k.frame_ids),observedRows=len(k.observed),scope='TRAIN exposure covariance; objective, original coefficients, ridge and starts unchanged')

    def to_original(self,eta):
        out=[]
        for block,m in zip(self.k.blocks,self.transforms):
            if m is not None:out.append((m[0]@eta[block['offset']:block['offset']+block['size']].reshape(block['shape'])).ravel())
        return np.concatenate(out)

    def from_original(self,theta):
        out=[]
        for block,m in zip(self.k.blocks,self.transforms):
            if m is not None:out.append((m[1]@theta[block['offset']:block['offset']+block['size']].reshape(block['shape'])).ravel())
        return np.concatenate(out)

    def objective(self,eta):
        loss,g=self.k.objective(self.to_original(eta));out=[]
        for block,m in zip(self.k.blocks,self.transforms):
            if m is not None:out.append((m[0].T@g[block['offset']:block['offset']+block['size']].reshape(block['shape'])).ravel())
        return loss,np.concatenate(out)


def fit(k,*,warm=None,components=None):
    start=time.monotonic();coords=Coordinates(k,components);rng=np.random.default_rng(20261111);seeds=[np.zeros(k.dimension),np.asarray(warm).copy()if warm is not None else rng.normal(0,.02,k.dimension)];receipts=[]
    for seed in seeds:
        eta=coords.from_original(seed);passes=[]
        for _ in range(3):
            result=minimize(coords.objective,eta,jac=True,method='BFGS',options=dict(gtol=1e-8,maxiter=1000,xrtol=1e-12));eta=result.x;theta=coords.to_original(eta);loss,g=k.objective(theta);gm=float(np.max(np.abs(g)));passes.append(dict(loss=loss,gradientMax=gm,etaGradientMax=float(np.max(np.abs(result.jac))),iterations=int(result.nit),message=str(result.message)))
            if gm<=1e-5:break
        if gm>1e-5:raise ValueError('Original-coordinate joint gradient did not converge '+str(gm))
        receipts.append(dict(theta=theta.tolist(),loss=loss,gradientMax=gm,passes=passes))
    best=min(receipts,key=lambda r:r['loss']);return np.array(best['theta']),dict(seconds=time.monotonic()-start,dimension=k.dimension,starts=receipts,startLossSpread=max(r['loss']for r in receipts)-best['loss'],coordinates=coords.receipt,optimizer='dense BFGS on an exact invertible TRAIN-only preconditioner')
