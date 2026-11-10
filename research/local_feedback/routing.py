"""Learn the activity/family routing that fixed-family adapters cannot repair."""
from pathlib import Path
import sys,time
import numpy as np
from scipy.special import softmax
from scipy.linalg import helmert
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import Data as LocalData,GEOMETRY,HAND,RC,LN,whiten,logit,expit,fit,read,write,local_metrics,summary,metrics,PLANS
from anchored import predictions

ROUTES=['break','rc','ln']


class Data(LocalData):
    def __init__(self):
        super().__init__();rows=[__import__('json').loads(line)for line in Path('.cache/local-feedback-routing-r3.jsonl').read_text(encoding='utf8').splitlines()]
        if [r['uid']for r in rows]!=[r['uid']for r in self.records]:raise ValueError('Counterfactual expert inventory mismatch')
        self.candidates=[c for r in rows for c in r['candidates']];self.expert=np.zeros((len(self.raw),2,len(self.fine)));self.expert_raw=[]
        for j,c in enumerate(self.candidates):
            self.expert_raw.append([c['rc'],c['ln']])
            for family in range(2):
                scores=c[ROUTES[family+1]];den=max(1.,max(scores.values(),default=0.))
                self.expert[j,family]=[min(.999,max(0.,scores.get(cls,0.))/den)for cls in self.fine]
        self.route=np.array([ROUTES.index(f['family'])for _,f in self.frame_records]);self.context=np.zeros((len(self.raw),4))
        ix=[self.feature_names.index(k)for k in ['npsPerKey','rowRate','tailRate','occupancy']]
        for r in self.records:
            a,b=r['offset'];x=self.x[a:b][:,ix];self.context[a:b]=x-np.average(x,axis=0,weights=self.dt[a:b])
        self.routing_names=self.feature_names+['relativeLogNps','relativeLogRowRate','relativeLogTailRate','relativeOccupancy']
        self.routing_x=np.c_[self.x,self.context]
        self.pressure_names=list(self.manifest['localKeys'][:6]);self.pressure_x=self.arrays['x'][self.wid,:,:6]
        self.family_y=np.array([[any(c in RC+['anchor','vibro','rc mix','hybrid']for c in r['tags']),any(c in LN+['ln mix','hybrid']for c in r['tags'])]for r in self.records],float)

    def chart_predictions_routed(self,support,family_probability):
        share=np.zeros((len(self.records),len(self.fine)));family=np.zeros((len(self.records),2))
        np.add.at(share,self.arrays['bag'],self.mass[:,None]*support)
        np.add.at(family,self.arrays['bag'],self.mass[:,None]*family_probability[:,1:])
        q=predictions(self,share,family,self.baseline,[self.profile['policy']],[self.profile['family']])[0,0]
        return q,share,family

    def routed_scores(self,probability):
        """Primary family is explicit; counterfactual class experts are real."""
        chosen=probability.argmax(1);frames=[]
        for j,(_,old)in enumerate(self.frame_records):
            route=chosen[j]
            if route==0:frames.append(dict(old,tag='break',family='break',secondary=None,confidence=float(probability[j,0]),scores={'break':1.}));continue
            scores=self.expert_raw[j][route-1];ranked=sorted(scores.items(),key=lambda r:-r[1]);tag,value=ranked[0]
            if tag in ['wildcard','ln wildcard']:
                for c,v in ranked[1:]:
                    if c not in ['wildcard','ln wildcard']and v>.3:tag,value=c,v;break
            secondary=ranked[1][0]if len(ranked)>1 and ranked[1][1]>=ranked[0][1]*.75 and ranked[1][1]>.3 else None
            frames.append(dict(old,family=ROUTES[route],tag=tag,scores=scores,secondary=secondary,confidence=min(1.,value)*float(probability[j,route])))
        return frames


def routed_local(d,ids,probability):
    # Reuse the metric definition, not a second local accuracy convention.
    old=d.frame_records
    try:
        fs=d.routed_scores(probability);d.frame_records=[(i,f)for(i,_),f in zip(old,fs)]
        return local_metrics(d,ids)
    finally:
        d.frame_records=old


class Kernel:
    def __init__(self,d,train,manual,cfg):
        self.d=d;self.train=np.array(train);self.manual=np.array(manual);self.cfg=cfg
        self.features=GEOMETRY if cfg['features']=='absolute'else d.routing_names if cfg['features']=='context'else d.routing_names+d.pressure_names
        x=d.x[:,:len(GEOMETRY)]if cfg['features']=='absolute'else d.routing_x if cfg['features']=='context'else np.c_[d.routing_x,d.pressure_x]
        self.all_x=x;self.frame_ids=np.concatenate([np.arange(*d.records[i]['offset'])for i in list(train)+list(manual)]);f=self.frame_ids
        self.z,self.scaler=whiten(x[f],d.frame_weight[f]);self.contrast=helmert(3,full=False).T;self.shape=(self.z.shape[1],2);self.dimension=int(np.prod(self.shape));self.bag=d.arrays['bag'][f];self.mass=d.mass[f]
        self.prior=np.eye(3)[d.route[f]]*cfg['priorStrength'];self.p0=softmax(self.prior,axis=1);self.teacher_weights=d.frame_weight[f]/d.frame_weight[f].sum()
        self.expert=d.expert[f];self.strong=[];self.strong_weights=[];self.allowed=[];mapper={int(j):k for k,j in enumerate(f)}
        for i in manual:
            r=d.records[i];duration=sum(a['end']-a['start']for a in r['annotations']if a['tags'])
            for a in r['annotations']:
                if not a['tags']:continue
                mask=np.array([any(c=='break'for c in a['tags']),any(c in RC+['anchor','vibro']for c in a['tags']),any(c in LN for c in a['tags'])])
                if not mask.any():raise ValueError('Unmapped original local truth')
                for j in range(*r['offset']):
                    row=d.frame_records[j][1];dt=max(0.,min(a['end'],row['end'])-max(a['start'],row['start']))
                    if dt:self.strong.append(mapper[j]);self.strong_weights.append(dt/max(duration,1)/max(1,len(manual)));self.allowed.append(mask)
        self.strong=np.array(self.strong,int);self.strong_weights=np.array(self.strong_weights);self.allowed=np.array(self.allowed,bool).reshape(-1,3)
        fi=[d.classes.index(c)for c in d.fine];self.y=d.y[train][:,fi];w=d.weight[train];w/=w.sum();mixed=np.array([any(c in ['hybrid','rc mix','ln mix']for c in d.records[i]['tags'])for i in train])
        available=np.zeros((len(d.records),len(d.fine)),bool)
        for j in range(len(d.fine)):np.maximum.at(available[:,j],d.arrays['bag'],d.expert[:,:,j].max(1)>0)
        cw=w[:,None]*available[train]*(~mixed[:,None]);self.cw=np.divide(cw,cw.sum(0,keepdims=True),out=np.zeros_like(cw),where=cw.sum(0,keepdims=True)>0)/len(d.fine)
        prior_mixed=d.baseline[train][:,[d.classes.index(c)for c in ['hybrid','rc mix','ln mix']]].any(1);gate=d.profile['policy'];threshold=np.where(d.baseline[train][:,fi],gate['keepFloor'],gate['addFloor']);threshold[prior_mixed]=gate['mixOverride'];self.threshold=np.clip(threshold,1e-6,1-1e-6)
        self.family_y=d.family_y[train];self.family_weight=w[:,None]/2;self.family_threshold=np.array([d.profile['family']['minRc'],d.profile['family']['minLn']])

    def forward(self,theta,*,all_rows=False):
        d=self.d;coef=theta.reshape(self.shape)
        if all_rows:
            f=np.arange(len(d.raw));z=np.c_[np.ones(len(f)),(self.all_x-self.scaler['mean'])@self.scaler['matrix']];prior=np.eye(3)[d.route]*self.cfg['priorStrength'];expert=d.expert
        else:f=self.frame_ids;z=self.z;prior=self.prior;expert=self.expert
        p=softmax(prior+z@coef@self.contrast.T,axis=1);support=np.sum(p[:,1:,None]*expert,axis=1)
        share=np.zeros((len(d.records),len(d.fine)));family=np.zeros((len(d.records),2));np.add.at(share,d.arrays['bag'][f],d.mass[f,None]*support);np.add.at(family,d.arrays['bag'][f],d.mass[f,None]*p[:,1:])
        return p,support,share,family,z

    def objective(self,theta):
        d=self.d;c=self.cfg;p,support,share,family,z=self.forward(theta);t=self.train
        s=np.clip(share[t],1e-12,1-1e-12);logits=logit(s)-logit(self.threshold);weak=np.sum(self.cw*(np.logaddexp(0,logits)-self.y*logits))
        gs=np.zeros_like(share);gs[t]=self.cw*(expit(logits)-self.y)/(s*(1-s))
        f=np.clip(family[t],1e-12,1-1e-12);flogits=logit(f)-logit(self.family_threshold);family_loss=np.sum(self.family_weight*(np.logaddexp(0,flogits)-self.family_y*flogits))
        gf=np.zeros_like(family);gf[t]=self.family_weight*(expit(flogits)-self.family_y)/(f*(1-f))
        # Equal fine/family task means; the family task prevents primitive
        # Mix-as-negative supervision and gives Hybrid an actual route gradient.
        loss=c['wholeWeight']*(weak+family_loss)/2;gp=np.zeros_like(p)
        gp[:,1:]=c['wholeWeight']/2*self.mass[:,None]*(np.einsum('nc,nfc->nf',gs[self.bag],self.expert)+gf[self.bag])
        gz=p*(gp-np.sum(p*gp,axis=1,keepdims=True))
        teacher=c['teacherWeight']*np.sum(self.teacher_weights[:,None]*self.p0*(np.log(self.p0)-np.log(np.maximum(p,1e-300))))
        loss+=teacher;gz+=c['teacherWeight']*self.teacher_weights[:,None]*(p-self.p0)
        if c['strongWeight']and len(self.strong):
            obs=p[self.strong];num=np.sum(obs*self.allowed,axis=1);ww=c['strongWeight']*self.strong_weights;loss-=np.sum(ww*np.log(np.maximum(num,1e-300)))
            np.add.at(gz,self.strong,ww[:,None]*(obs-obs*self.allowed/np.maximum(num[:,None],1e-300)))
        coef=theta.reshape(self.shape);loss+=.5*c['ridge']*np.sum(coef*coef)/2;gradient=z.T@gz@self.contrast+c['ridge']*coef/2
        return float(loss),gradient.ravel()

    def export(self,theta):
        b=theta.reshape(self.shape)@self.contrast.T;s=self.scaler;w=np.array(s['matrix'])@b[1:];bias=b[0]-np.array(s['mean'])@w
        return dict(schemaVersion=1,kind='context-pressure-routing',classes=ROUTES,features=self.features,coefficients=np.vstack([bias,w]).T.tolist(),config=self.cfg,
            scaler=s,trainUids=[self.d.records[i]['uid']for i in self.train],manualTrainUids=[self.d.records[i]['uid']for i in self.manual])


if __name__=='__main__':
    d=Data();s=PLANS[0]['inner'][0];cfg=dict(features='pressure',priorStrength=4.,ridge=.03,teacherWeight=1.,wholeWeight=1.,strongWeight=1.)
    k=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],cfg);rng=np.random.default_rng(12);theta=rng.normal(0,.02,k.dimension);direction=rng.normal(size=k.dimension);direction/=np.linalg.norm(direction);eps=1e-5;l,g=k.objective(theta);fd=(k.objective(theta+eps*direction)[0]-k.objective(theta-eps*direction)[0])/(2*eps)
    if abs(fd-g@direction)>1e-7:raise ValueError('Routing gradient mismatch')
    print('routing resource',k.dimension,'gradient error',abs(fd-g@direction),flush=True);theta,receipt=fit(k);p,support,_,_,_=k.forward(theta,all_rows=True);pred,_,_=d.chart_predictions_routed(support,p)
    baseline=local_metrics(d,d.manual[s['strongValidation']]);candidate=routed_local(d,d.manual[s['strongValidation']],p)
    write('research/local_feedback/results/routing-resource-r3.json',dict(config=cfg,receipt=receipt,gradientError=float(abs(fd-g@direction)),baselineLocal=baseline,candidateLocal=candidate,
        candidateWhole=metrics(d.y[d.main[s['validation']]],pred[d.main[s['validation']]],d.weight[d.main[s['validation']]])))
    print('routing seconds',receipt['seconds'],'local',summary(baseline),summary(candidate),flush=True)
