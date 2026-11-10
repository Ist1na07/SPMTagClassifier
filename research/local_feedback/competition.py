"""A local score-competition adapter with a frozen perceptual readout.

No chart classifier is fitted. Whole-label derivatives pass through the actual
duration/burden integral into local class contrasts. Family support budgets are
fixed, so increasing all local scores cannot improve the whole objective.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.special import softmax,logsumexp,expit
from scipy.linalg import helmert
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from model import Data as PerceptionData,logit,metrics
from anchored import predictions,base
from study import PLANS
from rc_action_runtime import smooth

RC=['chordjack','dense chordstream','fast chordstream','minijack','speed','tech','wildcard']
LN=['coordination','density','inverse','ln wildcard','release','technical']
RATE_KEYS={'npsPerKey','rowRate','rowBpm','tailRate','effort','handTransitionRatePerKey','handSubstitutionRatePerKey','handFixedGripRatePerKey'}
GEOMETRY=['npsPerKey','rowRate','chordFraction','multiFraction','lnHeadFraction','overlap','repeatFraction','cutReturn','chordChange','columnEntropy','handImbalance','sameHand','crossCut','logIoiSd','ioiChange','graceFraction','gridResidual','ioiCv','occupancy','minOccupancy','occupancySd','heldAtHead','lockFraction','releaseIndependence','releaseScatter','tailRate','shortHoldFraction','inverseGap','lnHeadOverlap','anchorShare','repeatRun','maskEntropy','trillFraction','rollFraction','burstiness','activeFraction']
HAND=['handMaskEntropy','handFirstOrderUncertainty','handSecondOrderUncertainty','handTransitionRatePerKey','handRepeatOverlap','handChangeJaccard','handReturn','handSubstitutionRatePerKey','handFixedGripRatePerKey']


def read(p):
    with Path(p).open(encoding='utf8')as f:return json.load(f)


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


class Data(PerceptionData):
    def __init__(self):
        super().__init__()
        self.profile=read('src/models/perception7k.json')['profiles']['detail']
        self.wid=self.manifest['weights'].index(self.profile['recipe'])
        self.mass=self.arrays['mass'][self.wid]
        self.baseline=base(self)
        self.family=self.representation(self.wid)[3]
        self.local_classes=list(self.fine)+['wildcard'];self.lidx={c:i for i,c in enumerate(self.local_classes)}
        self.frame_records=[(i,f)for i,r in enumerate(self.records)for f in r['frames']]
        self.raw=np.array([[max(0.,f['scores'].get(c,0.))for c in self.local_classes]for _,f in self.frame_records])
        self.div=np.array([max(1.,max(f['scores'].values(),default=0.))for _,f in self.frame_records])
        self.support=np.minimum(.999,self.raw/self.div[:,None])
        self.active=np.array([f['family']!='break'for _,f in self.frame_records]);self.support[~self.active]=0
        self.dt=np.array([f['end']-f['start']for _,f in self.frame_records])
        self.frame_weight=np.zeros(len(self.raw))
        for i,r in enumerate(self.records):
            a,b=r['offset'];self.frame_weight[a:b]=self.dt[a:b]/self.dt[a:b].sum()*self.weight[i]
        inputs=[json.loads(line)for line in Path('.cache/local-feedback-r2-inputs.jsonl').read_text(encoding='utf8').splitlines()]
        if [r['uid']for r in inputs]!=[r['uid']for r in self.records]:raise ValueError('Feature inventory/order mismatch')
        self.feature_names=GEOMETRY+HAND
        self.x=np.concatenate([np.array([[g[k]for k in GEOMETRY]+h for g,h in zip(r['geometry'],r['hand'])])for r in inputs])
        for j,k in enumerate(self.feature_names):
            if k in RATE_KEYS:self.x[:,j]=np.log1p(self.x[:,j])
        if not np.isfinite(self.x).all():raise ValueError('Nonfinite physical features')
        self.groups=[]
        for family,classes in [('rc',RC),('ln',LN)]:
            columns=np.array([self.lidx[c]for c in classes]);frames=np.array([j for j,(_,f)in enumerate(self.frame_records)if f['family']==family])
            scores=self.support[frames][:,columns];total=scores.sum(1);p=np.divide(scores,total[:,None],out=np.zeros_like(scores),where=total[:,None]>0)
            if np.any(total<=0):raise ValueError('Empty actual candidate family')
            prior=np.full_like(scores,-np.inf);np.log(scores,out=prior,where=scores>0);prior-=np.log1p(-scores)
            self.groups.append(dict(family=family,classes=classes,columns=columns,frames=frames,prior=prior,p=p,q0=scores,
                budget=total,contrast=helmert(len(classes),full=False).T))

    def chart_predictions(self,support):
        share=np.zeros((len(self.records),len(self.fine)))
        np.add.at(share,self.arrays['bag'],self.mass[:,None]*support[:,:len(self.fine)])
        q=predictions(self,share,self.family,self.baseline,[self.profile['policy']],[self.profile['family']])[0,0]
        return q,share


def frame_readout(d,scores):
    """Exact inherited decision semantics, including wildcard demotion."""
    out=[]
    for j,(_,f)in enumerate(d.frame_records):
        if f['family']=='break':out.append(dict(f));continue
        values={c:float(scores[j,d.lidx[c]])for c in f['scores']}
        ranked=sorted(values.items(),key=lambda v:-v[1]);tag,value=ranked[0]
        if tag in ['wildcard','ln wildcard']:
            for c,v in ranked[1:]:
                if c not in ['wildcard','ln wildcard']and v>.3:tag,value=c,v;break
        secondary=ranked[1][0]if len(ranked)>1 and ranked[1][1]>=ranked[0][1]*.75 and ranked[1][1]>.3 else None
        out.append(dict(f,scores=values,tag=tag,confidence=min(1.,value),secondary=secondary))
    return out


def local_metrics(d,ids,scores=None):
    all_frames=frame_readout(d,scores)if scores is not None else [f for _,f in d.frame_records]
    out=[]
    for i in ids:
        r=d.records[i];a,b=r['offset'];frames=smooth(all_frames[a:b]);correct=total=confusion=techdense=0.;counts={};transitions={}
        for truth in r['annotations']:
            tags=truth['tags']
            if not tags:continue
            total+=truth['end']-truth['start']
            for f in frames:
                dt=max(0.,min(truth['end'],f['end'])-max(truth['start'],f['start']))
                if not dt:continue
                tag=f['tag'];correct+=dt*(tag in tags)
                techdense+=dt*('tech'in tags and tag=='dense chordstream')
                confusion+=dt*((tag=='tech'and any(c in tags for c in ['dense chordstream','fast chordstream']))or(tag in ['dense chordstream','fast chordstream']and'tech'in tags))
                key='|'.join(tags)+' -> '+tag;transitions[key]=transitions.get(key,0.)+dt
                for c in set(tags)|{tag}:
                    v=counts.setdefault(c,dict(tp=0.,fp=0.,fn=0.));v['tp']+=dt*(c==tag and c in tags);v['fp']+=dt*(c==tag and c not in tags);v['fn']+=dt*(c!=tag and c in tags)
        out.append(dict(uid=r['uid'],accuracy=correct/max(total,1),correctMs=correct,totalMs=total,techToDenseMs=techdense,techCsConfusionMs=confusion,perClass=counts,transitions=transitions))
    return out


def summary(rows):
    return dict(accuracy=float(np.mean([r['accuracy']for r in rows])),techToDenseMs=sum(r['techToDenseMs']for r in rows),techCsConfusionMs=sum(r['techCsConfusionMs']for r in rows))


def whiten(x,w):
    w=w/w.sum();mean=w@x;xc=x-mean;cov=xc.T@(w[:,None]*xc);e,v=np.linalg.eigh(cov);keep=e>max(1e-10,e.max()*1e-8)
    matrix=v[:,keep]/np.sqrt(e[keep]);z=np.c_[np.ones(len(x)),xc@matrix]
    return z,dict(mean=mean.tolist(),matrix=matrix.tolist(),eigenvalues=e.tolist(),retainedRank=int(keep.sum()),removedDirections=int((~keep).sum()))


def allocate(prior,delta,budget):
    """Smooth Bernoulli-entropy projection onto the fixed-budget open cube.

    Solve sum(sigmoid(logit(q0)+delta-lambda))=sum(q0). Lambda is
    bracketed by the smallest/largest contrast; no cap/max winner kink.
    """
    lo=delta.min(1);hi=delta.max(1);lam=(lo+hi)/2
    for _ in range(70):
        q=expit(prior+delta-lam[:,None]);res=q.sum(1)-budget
        if np.max(np.abs(res))<2e-13:return q
        lo=np.where(res>0,lam,lo);hi=np.where(res<0,lam,hi)
        # A merely bracketed Newton proposal can bounce between almost-flat
        # sigmoid plateaus without shrinking the bracket. Bisection gives a
        # deterministic error bound even on extreme line-search candidates.
        lam=(lo+hi)/2
    raise ValueError('Support budget projection did not converge')


class Kernel:
    def __init__(self,d,train,manual,cfg):
        self.d=d;self.train=np.array(train);self.manual=np.array(manual);self.cfg=cfg
        self.features=np.arange(len(GEOMETRY)+(len(HAND)if cfg['features']=='hand'else 0));self.groups=[];self.dimension=0
        train_frames={j for i in list(train)+list(manual)for j in range(*d.records[i]['offset'])}
        manual_weight={int(i):1/max(1,len(manual))for i in manual}
        for g in d.groups:
            frame_ids=np.array([j for j in g['frames']if j in train_frames]);look={int(j):i for i,j in enumerate(g['frames'])};ix=np.array([look[int(j)]for j in frame_ids]);bag=d.arrays['bag'][frame_ids]
            z,scaler=whiten(d.x[frame_ids][:,self.features],d.frame_weight[frame_ids]);nc=len(g['classes']);shape=(z.shape[1],nc-1);size=int(np.prod(shape))
            strong=[];strong_weights=[];allowed=[];unreachable=0.
            fmap={int(j):i for i,j in enumerate(frame_ids)}
            for i in manual:
                r=d.records[i];a,b=r['offset'];duration=sum(v['end']-v['start']for v in r['annotations']if v['tags'])
                for ann in r['annotations']:
                    if not ann['tags']:continue
                    mask=np.array([c in ann['tags']for c in g['classes']])
                    for j in range(a,b):
                        if j not in fmap:continue
                        f=d.frame_records[j][1];dt=max(0.,min(ann['end'],f['end'])-max(ann['start'],f['start']))
                        if not dt:continue
                        w=dt/max(duration,1)*manual_weight[int(i)]
                        if not np.any(mask&(g['p'][look[j]]>0)):unreachable+=w;continue
                        strong.append(fmap[j]);strong_weights.append(w);allowed.append(mask)
            weights=d.frame_weight[frame_ids].copy();weights/=weights.sum()
            self.groups.append(dict(**g,frame_ids=frame_ids,ix=ix,bag=bag,z=z,scaler=scaler,shape=shape,size=size,offset=self.dimension,
                mass=d.mass[frame_ids],weights=weights,prior0=g['prior'][ix],p0=g['p'][ix],budget0=g['budget'][ix],
                strong=np.array(strong,int),strong_weights=np.array(strong_weights),allowed=np.array(allowed,bool).reshape(-1,nc),unreachableStrongMass=unreachable))
            self.dimension+=size
        fine_ids=[d.classes.index(c)for c in d.fine];self.y=d.y[train][:,fine_ids];weights=d.weight[train];weights/=weights.sum()
        mixed=np.array([any(c in ['hybrid','rc mix','ln mix']for c in d.records[i]['tags'])for i in train])
        family=d.profile['family'];fixed_hybrid=(d.family[train,0]>=family['minRc'])&(d.family[train,1]>=family['minLn'])if family['enabled']else np.zeros(len(train),bool)
        available=np.zeros((len(d.records),len(d.fine)),bool)
        for g in self.groups:
            for c in g['classes']:
                if c in d.fine:available[:,d.fine.index(c)]=np.bincount(g['bag'],minlength=len(d.records))>0
        mask=available[train]&(~mixed[:,None])&(~fixed_hybrid[:,None]);cw=weights[:,None]*mask
        # Equal chart costs within each observable class; no estimated class
        # weight or new fine gate can compensate for the local head.
        cw=np.divide(cw,cw.sum(0,keepdims=True),out=np.zeros_like(cw),where=cw.sum(0,keepdims=True)>0);self.cw=cw/len(d.fine)
        prior_mixed=d.baseline[train][:,[d.classes.index(c)for c in ['hybrid','rc mix','ln mix']]].any(1)
        gate=d.profile['policy'];threshold=np.where(d.baseline[train][:,fine_ids],gate['keepFloor'],gate['addFloor']);threshold[prior_mixed]=gate['mixOverride'];self.threshold=np.clip(threshold,1e-6,1-1e-6)

    def forward(self,theta,*,all_rows=False):
        d=self.d;support=d.support.copy();scores=d.raw.copy();states=[]
        for g in self.groups:
            coef=theta[g['offset']:g['offset']+g['size']].reshape(g['shape'])
            if all_rows:
                f=g['frames'];s=g['scaler'];z=np.c_[np.ones(len(f)),(d.x[f][:,self.features]-s['mean'])@s['matrix']];prior=g['prior'];budget=g['budget']
            else:f=g['frame_ids'];z=g['z'];prior=g['prior0'];budget=g['budget0']
            delta=z@coef@g['contrast'].T;allocation=allocate(prior,delta,budget);p=allocation/budget[:,None]
            support[np.ix_(f,g['columns'])]=allocation
            scores[np.ix_(f,g['columns'])]=allocation*d.div[f,None]
            states.append(dict(p=p,allocation=allocation,z=z,coef=coef))
        share=np.zeros((len(d.records),len(d.fine)))
        np.add.at(share,d.arrays['bag'],d.mass[:,None]*support[:,:len(d.fine)])
        return support,scores,share,states

    def objective(self,theta):
        d=self.d;c=self.cfg;support,scores,share,states=self.forward(theta);s=share[self.train];safe=np.clip(s,1e-12,1-1e-12)
        z=logit(safe)-logit(self.threshold);weak=c['wholeWeight']*np.sum(self.cw*(np.logaddexp(0,z)-self.y*z));gs=np.zeros_like(share)
        gs[self.train]=c['wholeWeight']*self.cw*(expit(z)-self.y)/(safe*(1-safe))*((s>=1e-12)&(s<=1-1e-12))
        loss=weak;gradient=[]
        for g,state in zip(self.groups,states):
            p=state['p'];allocation=state['allocation'];gq=np.zeros_like(p)
            for j,cls in enumerate(g['classes']):
                if cls in d.fine:gq[:,j]=g['mass']*gs[g['bag'],d.fine.index(cls)]
            # Categorical teacher acts on the same competition as the local
            # label; no 1e-9 sigmoid derivative is needed to correct a mistake.
            kl=np.sum(g['p0']*(np.log(np.maximum(g['p0'],1e-300))-np.log(np.maximum(p,1e-300))),axis=1)
            loss+=c['teacherWeight']*np.sum(g['weights']*kl)/2
            gq-=c['teacherWeight']*g['weights'][:,None]*g['p0']/np.maximum(allocation,1e-300)/2
            if c['strongWeight']and len(g['strong']):
                at=g['strong'];obs=p[at];num=np.sum(obs*g['allowed'],axis=1);ww=c['strongWeight']*g['strong_weights']
                loss-=np.sum(ww*np.log(np.maximum(num,1e-300)))
                pull=-g['allowed'].astype(float)/np.maximum((num*g['budget0'][at])[:,None],1e-300);np.add.at(gq,at,ww[:,None]*pull)
            sensitivity=allocation*(1-allocation);den=np.sum(sensitivity,axis=1,keepdims=True)
            gz=sensitivity*(gq-np.sum(sensitivity*gq,axis=1,keepdims=True)/np.maximum(den,1e-300))
            coef=state['coef'];loss+=.5*c['ridge']*np.sum(coef*coef)/sum(len(a['classes'])-1 for a in self.groups)
            gc=state['z'].T@gz@g['contrast']+c['ridge']*coef/sum(len(a['classes'])-1 for a in self.groups);gradient.append(gc.ravel())
        return float(loss),np.concatenate(gradient)

    def export(self,theta):
        blocks=[]
        for g in self.groups:
            b=theta[g['offset']:g['offset']+g['size']].reshape(g['shape'])@g['contrast'].T;s=g['scaler'];w=np.array(s['matrix'])@b[1:];intercept=b[0]-np.array(s['mean'])@w
            blocks.append(dict(family=g['family'],classes=g['classes'],coefficients=np.vstack([intercept,w]).T.tolist(),scaler=s))
        return dict(schemaVersion=1,kind='family-budget-competition',features=[self.d.feature_names[j]for j in self.features],blocks=blocks,config=self.cfg,
            trainUids=[self.d.records[i]['uid']for i in self.train],manualTrainUids=[self.d.records[i]['uid']for i in self.manual],
            routing='Inherited family and Break routing unchanged; zero/counterfactual branches not synthesized')


def fit(k,*,starts=2,warm=None):
    begin=time.monotonic();rng=np.random.default_rng(20261110);seeds=[np.zeros(k.dimension)]
    if starts>1:seeds.append(np.asarray(warm).copy()if warm is not None else rng.normal(0,.02,k.dimension))
    receipts=[]
    for seed in seeds:
        theta=seed.copy();passes=[]
        for _ in range(4):
            r=minimize(k.objective,theta,jac=True,method='L-BFGS-B',options=dict(gtol=2e-7,ftol=1e-14,maxiter=1000,maxls=50,maxcor=30));theta=r.x;loss,g=k.objective(theta);gm=float(np.max(np.abs(g)));passes.append(dict(loss=loss,gradientMax=gm,iterations=int(r.nit),message=str(r.message)))
            if gm<=1e-5:break
        if gm>1e-5:raise ValueError('Local competition fit not converged '+str(gm))
        receipts.append(dict(theta=theta.tolist(),loss=loss,gradientMax=gm,passes=passes))
    best=min(receipts,key=lambda r:r['loss'])
    if best['loss']>min(k.objective(seed)[0]for seed in seeds)+1e-8:raise ValueError('Fit worse than an actual start')
    return np.array(best['theta']),dict(seconds=time.monotonic()-begin,dimension=k.dimension,starts=receipts,startLossSpread=max(r['loss']for r in receipts)-best['loss'])


if __name__=='__main__':
    d=Data();p=PLANS[0]['inner'][0];cfg=dict(features='hand',ridge=.03,teacherWeight=1.,wholeWeight=1.,strongWeight=1.)
    k=Kernel(d,d.main[p['train']],d.manual[p['strongTrain']],cfg);rng=np.random.default_rng(7);t=rng.normal(0,.02,k.dimension);direction=rng.normal(size=k.dimension);direction/=np.linalg.norm(direction);eps=1e-5
    l,g=k.objective(t);fd=(k.objective(t+eps*direction)[0]-k.objective(t-eps*direction)[0])/(2*eps)
    print('resource dimension',k.dimension,'directional gradient error',abs(fd-g@direction),flush=True)
    if abs(fd-g@direction)>1e-7:raise ValueError('Analytic pullback mismatch')
    q,s,share,_=k.forward(np.zeros(k.dimension),all_rows=True);np.testing.assert_allclose(q,d.support,atol=1e-14)
    np.testing.assert_array_equal([r['tag']for r in frame_readout(d,s)],[f['tag']for _,f in d.frame_records])
    theta,receipt=fit(k);support,scores,_,_=k.forward(theta,all_rows=True);pred,_=d.chart_predictions(support)
    result=dict(receipt=receipt,gradientError=float(abs(fd-g@direction)),config=cfg,
        baselineLocal=local_metrics(d,d.manual[p['strongValidation']]),candidateLocal=local_metrics(d,d.manual[p['strongValidation']],scores),
        baselineWhole=metrics(d.y[d.main[p['validation']]],d.chart_predictions(d.support)[0][d.main[p['validation']]],d.weight[d.main[p['validation']]]),
        candidateWhole=metrics(d.y[d.main[p['validation']]],pred[d.main[p['validation']]],d.weight[d.main[p['validation']]]))
    write('research/local_feedback/results/resource-r2.json',result);print('seconds',receipt['seconds'],'local',summary(result['baselineLocal']),summary(result['candidateLocal']),flush=True)
