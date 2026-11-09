"""Task-specific whole-tag reference: test information before scaling up MIL.

This is NOT a strict performance upper bound and produces no local classes.
Same349 annotation records and6x3 historical song-group folds;17labels retained.
All kernels, ridge strengths, target scales, pooling powers and threshold
partitions are jointly selected inside training folds. No copied local truth.
"""
import argparse,concurrent.futures,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from pressure_data import attach,mil,scale_matrix,read
from pressure_thresholds import optimize,predict

VARIANTS=['original','events','events_rating']
RIDGE=[.0001,.001,.01,.1,1.,10.,100.]
KERNELS=[('linear',None)]+[('rbf',g)for g in [1/16,1/4,1.,4.,16.]]
CONFIGS=[{'kernel':k,'gamma':g,'ridge':ridge,'target':target,'effortPower':power}
    for (k,g),ridge,target,power in itertools.product(KERNELS,RIDGE,['raw','standardized'],[0.,1.,2.])]
_DATA=None


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)


def pooled(x,w):
    out=[]
    for c in x.T:
        take=np.isfinite(c)
        if not take.any():out.extend([np.nan]*3);continue
        v=c[take];a=w[take];a=a/a.sum();mean=a@v;order=np.argsort(v,kind='stable');j=min(np.searchsorted(np.cumsum(a[order]),.9),len(v)-1)
        out.extend([float(mean),float(np.sqrt(a@((v-mean)**2))),float(v[order[j]])])
    return out


def load():
    main,manual,allrows=mil.load();schema=attach(allrows)
    protocol=read('research/rework/results/mil-stage1-r4/protocol.json')
    if [r['uid']for r in main]!=[r['uid']for r in protocol['records']]:raise ValueError('Historical whole inventory changed')
    matrices={(v,p):[]for v in VARIANTS for p in [0.,1.,2.]}
    for r in main:
        f=r['_frontier'];original=np.c_[[s['original']for s in r['segments']],f['typePrior']]
        # Every representation shares the same frozen expert-prior input.
        pressure=f['pressure'];no_rating=np.delete(pressure,[6,7,14,15,16],axis=1)
        events=np.c_[f['core'],f['motion'],f['hand'],no_rating,f['typePrior']]
        rating=np.c_[f['core'],f['motion'],f['hand'],pressure,f['typePrior']]
        duration=np.array([s['end']-s['start']for s in r['segments']],float)
        # Common head+tail count scale; no fixed class gate. Both terms have
        # actions/s/key units; this is a pooling descriptor, not calibrated SR.
        effort=np.sqrt(np.array([s['effort']for s in r['segments']])**2+f['tailRate']**2+1e-12)
        for v,x in [('original',original),('events',events),('events_rating',rating)]:
            for power in [0.,1.,2.]:matrices[v,power].append(pooled(x,duration*effort**power))
    return{'main':main,'plans':protocol['plans'],'schema':schema,'matrices':{k:np.array(v)for k,v in matrices.items()},
        'y':mil.encode(main),'w':np.array([r['recordWeight']for r in main])}


class KernelFit:
    def __init__(self,raw,y,w,kernel,gamma):
        self.w=w/w.sum();x,self.mean,self.scale=scale_matrix(raw,self.w);self.x=x;self.prior=self.w@y
        self.constant=(self.prior==0)|(self.prior==1);self.kernel=kernel;self.gamma=gamma
        distance=np.maximum(0,(x*x).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*x@x.T)
        positive=distance[distance>1e-12];self.median=float(np.median(positive))if len(positive)else 1.
        k=x@x.T/x.shape[1]if kernel=='linear'else np.exp(-gamma*distance/self.median)
        self.row=k@self.w;self.grand=float(self.w@self.row);centered=k-self.row[:,None]-self.row[None,:]+self.grand
        self.sqrt=np.sqrt(self.w);self.B=self.sqrt[:,None]*centered*self.sqrt[None,:]
        self.eigen,self.vectors=np.linalg.eigh(self.B)
        if self.eigen.min() < -1e-9:raise ValueError('Kernel lost positive-semidefinite structure')
        self.eigen=np.maximum(self.eigen,0);self.y=y;self.solves=0

    def solve(self,ridge,target):
        scale=np.sqrt(self.prior*(1-self.prior))if target=='standardized'else np.ones_like(self.prior)
        scale=np.where(self.constant,1,scale);rhs=self.sqrt[:,None]*(self.y-self.prior)/scale
        solution=self.vectors@((self.vectors.T@rhs)/(self.eigen[:,None]+ridge))
        residual=float(np.abs(self.B@solution+ridge*solution-rhs).max());self.solves+=1
        if residual>1e-9:raise ValueError('Unaccepted kernel normal-equation residual')
        return self.sqrt[:,None]*solution,residual

    def predict(self,raw,coefficient,target):
        x,_,_=scale_matrix(raw,np.ones(len(raw)),self.mean,self.scale)
        if self.kernel=='linear':k=x@self.x.T/x.shape[1]
        else:
            distance=np.maximum(0,(x*x).sum(1)[:,None]+(self.x*self.x).sum(1)[None,:]-2*x@self.x.T)
            k=np.exp(-self.gamma*distance/self.median)
        centered=k-(k@self.w)[:,None]-self.row[None,:]+self.grand
        score=centered@coefficient
        p=expit(score)if target=='standardized'else np.clip(score+self.prior,0,1)
        p[:,self.constant]=self.prior[self.constant]
        return p


def init(d):
    global _DATA
    _DATA=d


def evaluate(args):
    variant,outer,output=args;root=Path(output)/variant/str(outer)
    if(root/'complete.json').exists():return variant,outer,'resumed'
    d=_DATA;plan=d['plans'][outer];inner=[None]*len(CONFIGS);traces=[];factorizations=0
    # Factorization reuse is exact linear algebra, not skipped configurations.
    for power,(kernel,gamma)in itertools.product([0.,1.,2.],KERNELS):
        x=d['matrices'][variant,power];configs=[(i,c)for i,c in enumerate(CONFIGS)if c['effortPower']==power and c['kernel']==kernel and c['gamma']==gamma]
        for fold,split in enumerate(plan['inner']):
            tr,va=split['train'],split['validation'];model=KernelFit(x[tr],d['y'][tr],d['w'][tr],kernel,gamma);factorizations+=1
            for i,cfg in configs:
                coefficient,residual=model.solve(cfg['ridge'],cfg['target']);p=model.predict(x[va],coefficient,cfg['target'])
                if inner[i]is None:inner[i]=[]
                inner[i].append(p);traces.append({'configId':i,'inner':fold,'normalEquationResidual':residual})
    ids=[i for split in plan['inner']for i in split['validation']];probability=np.array([np.concatenate(p)for p in inner]);search=[]
    if set(ids)!=set(plan['train'])or len(ids)!=len(set(ids)):raise ValueError('Inner coverage mismatch')
    for i,cfg in enumerate(CONFIGS):
        best=optimize(d['y'][ids],probability[i],d['w'][ids]);search.append({'configId':i,'config':cfg,**best})
    choice=max(search,key=lambda r:(r['macroF1'],r['macroRecall'],r['config']['ridge'],-r['configId']))
    write(root/'search.json',search);write(root/'inner-fits.json',traces);write(root/'selection.json',{'usesOuterLabels':False,**choice})
    np.savez_compressed(root/'inner-predictions.npz',probability=probability,validationIds=np.array(ids))
    c=choice['config'];x=d['matrices'][variant,c['effortPower']];tr,te=plan['train'],plan['test'];model=KernelFit(x[tr],d['y'][tr],d['w'][tr],c['kernel'],c['gamma'])
    coefficient,residual=model.solve(c['ridge'],c['target']);p=model.predict(x[te],coefficient,c['target']);q=predict(p,choice['fine'],choice['mix'])
    np.savez_compressed(root/'outer-model.npz',trainX=model.x,mean=model.mean,scale=model.scale,prior=model.prior,constant=model.constant,
        weight=model.w,row=model.row,grand=np.array(model.grand),median=np.array(model.median),coefficient=coefficient)
    write(root/'outer-predictions.json',[{'uid':d['main'][i]['uid'],'outer':outer,'variant':variant,'probability':a.tolist(),
        'predicted':[c for c,t in zip(mil.MAP_CLASSES,b)if t]}for i,a,b in zip(te,p,q)])
    write(root/'complete.json',{'innerCoefficientSolves':len(traces),'innerKernelFactorizations':factorizations,'outerCoefficientSolves':1,'outerResidual':residual})
    return variant,outer,len(traces)


def run(args):
    d=load();root=Path(args.output)
    protocol={'id':'whole-information-reference-r1','scope':'task-specific whole-tag reference, not a strict ceiling or a local classifier; retrospective grouped CV, frozen expert/rating auxiliary exposure unknown',
        'variants':VARIANTS,'configs':CONFIGS,'classes':mil.MAP_CLASSES,'plans':d['plans'],
        'records':[{'uid':r['uid'],'group':r['group'],'tags':mil.target(r),'weight':r['recordWeight']}for r in d['main']],
        'selection':'exact joint fine/mix threshold partitions; inner full17macroF1, macroRecall, larger ridge, deterministic config order',
        'features':d['schema'],'pooling':'per-coordinate masked duration*sqrt(headEffort^2+tailRatePerKey^2)^a mean/std/q90; all variants same pooling and frozen expert logits',
        'standardization':'training-only physical-file-weighted moments; missing observations centered with explicit masks; train-only median squared distance for RBF',
        'scoreMeaning':'raw-target ridge clipped to[0,1], or sigmoid standardized-target margin; continuous tag scores, not empirically calibrated confidence',
        'sourceSnapshots':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['whole_reference.py','pressure_data.py','pressure_thresholds.py']}}
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=protocol:raise ValueError('Whole reference protocol changed')
    write(root/'protocol.json',protocol);jobs=[(v,o,str(root))for v in VARIANTS for o in range(6)]
    if args.max_jobs:jobs=jobs[:args.max_jobs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(d,))as pool:
        for result in pool.map(evaluate,jobs):print('whole reference',result,flush=True)
    alljobs=[(v,o)for v in VARIANTS for o in range(6)]
    if not all((root/v/str(o)/'complete.json').exists()for v,o in alljobs):print('Partial resource run; no aggregate results.',flush=True);return
    rows=[r for v,o in alljobs for r in read(root/v/str(o)/'outer-predictions.json')];write(root/'outer-predictions.json',rows)
    counts=[read(root/v/str(o)/'complete.json')for v,o in alljobs]
    write(root/'complete.json',{'variants':VARIANTS,'recordsPerVariant':len(d['main']),
        'innerCoefficientSolves':sum(r['innerCoefficientSolves']for r in counts),'innerKernelFactorizations':sum(r['innerKernelFactorizations']for r in counts),'outerFits':len(counts)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='research/frontier/results/whole-reference-r1');p.add_argument('--workers',type=int,default=4);p.add_argument('--max-jobs',type=int);run(p.parse_args())
