"""Joint route and the established fixed-affinity-budget type competition.

q=sigmoid(logit(q0)+delta-lambda), sum(q)=sum(q0), conditional type mass=q/B.
This reuses competition-r2's exact model and implicit Jacobian. Its original
bisection solver is the independent oracle for a safeguarded execution path.
"""
from pathlib import Path
import os,sys,time
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from scipy.linalg import helmert
from scipy.special import softmax,expit
sys.path.insert(0,str(Path(__file__).resolve().parent))
from activity import Data,Kernel as RouteKernel,PLANS,read,write,parent
from competition import RC,LN,fit,logit,whiten,local_metrics,summary
from reanalyze_allowed import corrected


def allocate_numpy(prior,delta,budget):
    lo=delta.min(1);hi=delta.max(1);lam=(lo+hi)/2;q=expit(prior+delta-lam[:,None]);res=q.sum(1)-budget;active=np.flatnonzero(np.abs(res)>2e-13)
    for _ in range(100):
        if not len(active):return q
        a=active;r=res[a];slope=np.sum(q[a]*(1-q[a]),axis=1);lo[a]=np.where(r>0,lam[a],lo[a]);hi[a]=np.where(r<0,lam[a],hi[a]);mid=(lo[a]+hi[a])/2
        step=lam[a]+np.divide(r,slope,out=np.zeros_like(r),where=slope>0);step=np.where((step>lo[a])&(step<hi[a])&np.isfinite(step),step,mid)
        qq=expit(prior[a]+delta[a]-step[:,None]);rr=qq.sum(1)-budget[a]
        # Reject Newton oscillations on saturated plateaus by measuring the
        # actual residual contraction. A rejected step halves its bracket.
        bad=(np.abs(rr)>.5*np.abs(r))&(step!=mid)
        if bad.any():
            step[bad]=mid[bad];qq[bad]=expit(prior[a[bad]]+delta[a[bad]]-step[bad,None]);rr[bad]=qq[bad].sum(1)-budget[a[bad]]
        lam[a]=step;q[a]=qq;res[a]=rr;active=a[np.abs(rr)>2e-13]
    raise ValueError('Safeguarded fixed-budget solve did not converge')


from fast_projection import allocate


class Kernel:
    def __init__(self,d,train,manual,cfg):
        self.d=d;self.cfg=cfg;self.train=np.array(train);self.manual=np.array(manual)
        # Reuse the established TRAIN inventory, per-class whole masks and
        # anchored thresholds, not its coarse-only local objective.
        rcfg=dict(widthId=cfg['widthId'],context=cfg['context'],obligations=cfg['obligations'],ridge=cfg['ridge'],wholeWeight=cfg['wholeWeight'],strongWeight=1.,extraWeight=cfg['extraWeight'])
        self.base=RouteKernel(d,train,manual,rcfg);b=self.base;self.frame_ids=b.frame_ids;self.all_x=b.all_x;self.features=b.features;self.z=b.z;self.scaler=b.scaler;self.bag=b.bag;self.mass=b.mass
        self.blocks=[];self.dimension=0
        for bi,classes in enumerate([parent.ROUTES,RC,LN]):
            contrast=helmert(len(classes),full=False).T;shape=(self.z.shape[1],len(classes)-1);active=cfg.get('trainRoute',True)if bi==0 else cfg.get('trainType',True);size=int(np.prod(shape))if active else 0;self.blocks.append(dict(classes=classes,contrast=contrast,shape=shape,offset=self.dimension,size=size));self.dimension+=size
        self.eligible=np.ones((len(d.route),3),bool)
        if cfg.get('physicalEligibility',False):
            names={name:i for i,name in enumerate(self.features)};self.eligible[:,1]=self.all_x[:,names['npsPerKey']]>0
            self.eligible[:,2]=np.any(self.all_x[:,[names[key]for key in ['lnHeadFraction','occupancy','tailRate']]]>0,axis=1)
        self.prior=np.where(self.eligible,np.eye(3)[d.route]*6.,-np.inf)
        if not hasattr(d,'joint_type_experts'):
            types=[]
            for family,classes in enumerate([RC,LN]):
                q0=[]
                for scores in d.expert_raw:
                    ss=scores[family];den=max(1.,max(ss.values(),default=0.));q0.append([min(.999,max(0.,ss.get(c,0.))/den)for c in classes])
                q0=np.array(q0);budget=q0.sum(1);prior=np.full_like(q0,-np.inf);np.log(q0,out=prior,where=q0>0);prior-=np.log1p(-q0)
                if np.any(budget<=0):raise ValueError('Empty real type expert')
                types.append(dict(prior=prior,budget=budget,q0=q0,columns=[d.fine.index(c)if c in d.fine else-1 for c in classes]))
            d.joint_type_experts=types
        self.type=d.joint_type_experts
        if cfg.get('physicalEligibility',False):
            available=np.zeros((len(d.records),len(d.fine)),bool)
            for family,typ in enumerate(self.type):
                for j,col in enumerate(typ['columns']):
                    if col>=0:np.maximum.at(available[:,col],d.arrays['bag'],self.eligible[:,family+1]&(typ['q0'][:,j]>0))
            weights=d.weight[train];weights=weights/weights.sum();mixed=np.array([any(c in ['hybrid','rc mix','ln mix']for c in d.records[i]['tags'])for i in train]);cw=weights[:,None]*available[train]*(~mixed[:,None]);b.cw=np.divide(cw,cw.sum(0,keepdims=True),out=np.zeros_like(cw),where=cw.sum(0,keepdims=True)>0)/len(d.fine)
        self.mapper={int(j):i for i,j in enumerate(self.frame_ids)};self.strong=[];self.extra=[]
        for i in manual:
            row=d.records[i];duration=sum(a['end']-a['start']for a in row['annotations']if a['tags'])
            for ann in row['annotations']:
                if not ann['tags']:continue
                for j in range(*row['offset']):
                    f=d.frame_records[j][1];dt=max(0.,min(ann['end'],f['end'])-max(ann['start'],f['start']))
                    if dt:self.strong.append(dict(at=np.array([self.mapper[j]]),weight=np.array([1.]),cost=dt/max(duration,1)/max(len(manual),1),allowed=self.allowed(ann['tags'])))
        categories={'B01':['break'],'B02':['release'],'C16':['speed'],'C24':['tech'],'CTX-L':['dense chordstream','fast chordstream'],'CTX-H':['break'],'LN-L':['release'],'LN-H':['release']}
        for rnd in ['r4','r5','r6']:
            for name,a in read('research/frontier/annotations-'+rnd+'.json')['cases'].items():categories[name]=[a['localCategory']]
        for name in ['I01','I02']:categories[name]=['ln wildcard']
        self.categories=categories
        for e in b.extra_receipt:
            at=[];weight=[];i=e['index'];row=d.records[i]
            for j in range(*row['offset']):
                f=d.frame_records[j][1];dt=max(0.,min(e['end'],f['end'])-max(e['start'],f['start']))
                if dt:at.append(self.mapper[j]);weight.append(d.mass[j]*dt/(f['end']-f['start']))
            weight=np.array(weight);weight=weight/max(weight.sum(),1e-30)
            self.extra.append(dict(at=np.array(at),weight=weight,unitCost=1/max(len(b.extra_receipt),1),allowed=self.allowed(categories[e['id']]),id=e['id']))
        # Combine duplicate one-row observations into vectorized derivatives.
        self.strong_at=np.array([v['at'][0]for v in self.strong],int);self.strong_cost=np.array([v['cost']for v in self.strong]);self.strong_allowed=[np.array([v['allowed'][family]for v in self.strong])for family in range(3)]
        self.observed=np.unique(np.r_[self.strong_at,*[e['at']for e in self.extra]]).astype(int);lookup=np.full(len(self.frame_ids),-1,int);lookup[self.observed]=np.arange(len(self.observed));self.compact_strong=lookup[self.strong_at];self.compact_extra=[lookup[e['at']]for e in self.extra]

    @staticmethod
    def coefficient(theta,b):
        return theta[b['offset']:b['offset']+b['size']].reshape(b['shape'])if b['size']else np.zeros(b['shape'])

    def set_config(self,cfg):
        if any(cfg.get(key,True)!=self.cfg.get(key,True)for key in ['trainRoute','trainType','context','obligations','physicalEligibility']):raise ValueError('Cannot reuse a different joint architecture')
        self.cfg=cfg

    def allowed(self,tags):
        return [np.array([float('break'in tags)]),np.array([float(c in tags)for c in RC]),np.array([float(c in tags)for c in LN])]

    def forward(self,theta,all_rows=False,compact=False):
        f=np.arange(len(self.d.route))if all_rows else self.frame_ids[self.observed]if compact else self.frame_ids;z=np.c_[np.ones(len(f)),(self.all_x[f]-self.scaler['mean'])@self.scaler['matrix']]if all_rows else self.z[self.observed]if compact else self.z
        residual=[]
        for b in self.blocks:residual.append(z@self.coefficient(theta,b)@b['contrast'].T if b['size']else np.zeros((len(f),len(b['classes']))))
        route=softmax(self.prior[f]+residual[0],axis=1);u=[];q=[];support=np.zeros((len(f),len(self.d.fine)))
        for family,t in enumerate(self.type):
            qq=t['q0'][f].copy()
            if self.blocks[family+1]['size']:
                valid=self.eligible[f,family+1];ff=f[valid];qq[valid]=allocate(t['prior'][ff],residual[family+1][valid],t['budget'][ff])
            v=qq/t['budget'][f,None];u.append(v);q.append(qq)
            for j,c in enumerate(t['columns']):
                if c>=0:support[:,c]=route[:,family+1]*qq[:,j]
        return route,u,q,support,z,f

    def objective(self,theta,*,compact=None):
        d=self.d;b=self.base;c=self.cfg;compact=c['wholeWeight']==0 if compact is None else compact
        if compact and c['wholeWeight']:raise ValueError('Nonzero whole loss cannot omit whole rows')
        r,u,q,support,z,f=self.forward(theta,compact=compact);loss=0.;gr=np.zeros_like(r);gu=[np.zeros_like(v)for v in u]
        if not compact:
            bag=d.arrays['bag'][f];starts=np.r_[0,np.flatnonzero(np.diff(bag))+1];groups=bag[starts];share=np.zeros((len(d.records),len(d.fine)));family=np.zeros((len(d.records),2));share[groups]=np.add.reduceat(d.mass[f,None]*support,starts,axis=0);family[groups]=np.add.reduceat(d.mass[f,None]*r[:,1:],starts,axis=0);t=self.train
            s=np.clip(share[t],1e-12,1-1e-12);logits=logit(s)-logit(b.threshold);weak=np.sum(b.cw*(np.logaddexp(0,logits)-b.y*logits));gs=np.zeros_like(share);gs[t]=b.cw*(expit(logits)-b.y)/(s*(1-s))
            ff=np.clip(family[t],1e-12,1-1e-12);ll=logit(ff)-logit(b.family_threshold);floss=np.sum(b.family_weight*(np.logaddexp(0,ll)-b.family_y*ll));gf=np.zeros_like(family);gf[t]=b.family_weight*(expit(ll)-b.family_y)/(ff*(1-ff))
            loss=c['wholeWeight']*(weak+floss)/2;factor=c['wholeWeight']/2*d.mass[f]
            for family,typ in enumerate(self.type):
                gq=np.zeros_like(q[family]);gr[:,family+1]+=factor*gf[bag,family]
                for j,col in enumerate(typ['columns']):
                    if col>=0:
                        g=factor*gs[bag,col];gr[:,family+1]+=g*q[family][:,j];gq[:,j]+=g*r[:,family+1]
                gu[family]+=gq*typ['budget'][f,None]
        # Proper joint route/type categorical mass; Break is one category.
        if c['strongWeight']and len(self.strong_at):
            at=self.compact_strong if compact else self.strong_at;allowed=self.strong_allowed;a0=allowed[0][:,0];a1=np.sum(u[0][at]*allowed[1],axis=1);a2=np.sum(u[1][at]*allowed[2],axis=1);num=np.maximum(r[at,0]*a0+r[at,1]*a1+r[at,2]*a2,1e-300);cost=c['strongWeight']*self.strong_cost;loss-=np.sum(cost*np.log(num));g=-cost/num
            for family,part in enumerate([a0,a1,a2]):np.add.at(gr[:,family],at,g*part)
            for family in range(2):np.add.at(gu[family],at,g[:,None]*r[at,family+1,None]*allowed[family+1])
        # Active answers describe target passages, not a truth copied to each
        # 500-ms child. Burden-normalized interval bags let mixed passages stay
        # mixed. Qualitative grades are retained, not invented numeric labels.
        if c['strongWeight']:
            for ei,e in enumerate(self.extra):
                at=self.compact_extra[ei]if compact else e['at'];w=e['weight'];a=e['allowed'];parts=[np.full(len(at),a[0][0]),u[0][at]@a[1],u[1][at]@a[2]];num=max(float(np.sum(w*np.sum(r[at]*np.stack(parts,axis=1),axis=1))),1e-300);cost=c['strongWeight']*c['extraWeight']*e['unitCost'];loss-=cost*np.log(num);g=-cost/num*w
                for family,part in enumerate(parts):gr[at,family]+=g*part
                for family in range(2):gu[family][at]+=g[:,None]*r[at,family+1,None]*a[family+1]
        gradients=[r*(gr-np.sum(r*gr,axis=1,keepdims=True))]
        for j,(qq,g)in enumerate(zip(q,gu)):
            h=qq*(1-qq);gg=g/self.type[j]['budget'][f,None];den=np.maximum(h.sum(1),1e-300);gradients.append(h*(gg-np.sum(h*gg,axis=1,keepdims=True)/den[:,None]))
        out=[]
        for block,g in zip(self.blocks,gradients):
            if not block['size']:continue
            coef=self.coefficient(theta,block);ridge=c['ridge']/block['shape'][1];loss+=.5*ridge*np.sum(coef*coef);out.append((z.T@g@block['contrast']+ridge*coef).ravel())
        return float(loss),np.concatenate(out)

    def frames(self,theta):
        r,u,q,support,_,_=self.forward(theta,all_rows=True);out=[]
        for j,(_,old)in enumerate(self.d.frame_records):
            route=int(r[j].argmax())
            if not route:out.append(dict(old,family='break',tag='break',secondary=None,confidence=float(r[j,0]),scores={'break':1.}));continue
            typ=self.blocks[route];values=q[route-1][j];scores=dict(zip(typ['classes'],values.tolist()));ranked=sorted(scores,key=lambda c:-scores[c]);tag=ranked[0];secondary=ranked[1]if scores[ranked[1]]>=.75*scores[tag]and scores[ranked[1]]>.3 else None
            out.append(dict(old,family=parent.ROUTES[route],tag=tag,secondary=secondary,scores=scores,confidence=float(scores[tag]*r[j,route])))
        return out,r,support

    def export(self,theta):
        blocks=[]
        for b in self.blocks:
            coef=self.coefficient(theta,b)@b['contrast'].T;s=self.scaler;w=np.array(s['matrix'])@coef[1:];bias=coef[0]-np.array(s['mean'])@w;blocks.append(dict(classes=b['classes'],coefficients=np.vstack([bias,w]).T.tolist()))
        return dict(kind='joint-activity-budget',schemaVersion=1,features=self.features,blocks=blocks,config=self.cfg,windowMs=self.d.widths[self.cfg['widthId']],stepMs=500,priorStrength=6,allocation='fixed-affinity Bernoulli entropy projection',trainUids=[self.d.records[i]['uid']for i in self.train],manualTrainUids=[self.d.records[i]['uid']for i in self.manual],extraObservations=self.base.extra_receipt)


def local(d,ids,frames):
    old=d.frame_records
    try:
        d.frame_records=[(i,f)for(i,_),f in zip(old,frames)];return corrected(local_metrics(d,ids))
    finally:d.frame_records=old


if __name__=='__main__':
    from human_regression import coverage
    d=Data();cfg=dict(widthId=1,context=True,obligations=True,ridge=.03,strongWeight=1.,extraWeight=4.,wholeWeight=1.);k=Kernel(d,d.main,d.manual,cfg);rng=np.random.default_rng(31);t=rng.normal(0,.01,k.dimension);v=rng.normal(size=k.dimension);v/=np.linalg.norm(v);eps=1e-5;l,g=k.objective(t);err=abs((k.objective(t+eps*v)[0]-k.objective(t-eps*v)[0])/(2*eps)-g@v)
    if err>1e-7:raise ValueError('Joint activity gradient mismatch '+str(err))
    _,_,q0,_,_,f=k.forward(np.zeros(k.dimension));baseline_error=max(float(np.max(np.abs(q-t['q0'][f])))for q,t in zip(q0,k.type))
    if baseline_error>1e-12:raise ValueError('Initial bounded expert embedding mismatch')
    print('joint resource',k.dimension,'gradient',err,'zero expert',baseline_error,flush=True);theta,receipt=fit(k);frames,p,s=k.frames(theta);cases=[]
    for e in k.base.extra_receipt:
        a,b=d.records[e['index']]['offset'];case=dict(id=e['id'],coverage=coverage(frames[a:b],e['start'],e['end']));cases.append(case);print(case,flush=True)
    write('research/local_feedback/results/joint-entropy-resource-r7.json',dict(config=cfg,model=k.export(theta),receipt=receipt,gradientError=err,zeroExpertError=baseline_error,local=summary(local(d,d.manual,frames)),cases=cases,consumers={n:parent.metrics(d.y[d.main],d.consume(s,p,n)[d.main],d.weight[d.main])for n in d.consumer_names},scope='All TRAIN feasibility only; no validation claim'))
    print('joint resource seconds',receipt['seconds'],summary(local(d,d.manual,frames)),flush=True)

