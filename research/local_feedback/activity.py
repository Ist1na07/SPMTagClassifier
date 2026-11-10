"""Local context router, learning the already confirmed player observations.

The old family is an observed input with learnable coefficients, not a fixed
zero branch. No pseudo-label teacher duplicates known mistakes. Prototype
experts remain real counterfactual candidates, not invented target copies.
"""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import routing as parent
from competition import RATE_KEYS,whiten
from anchored import predictions
read=parent.read;write=parent.write;fit=parent.fit;metrics=parent.metrics;PLANS=parent.PLANS
ROOT=Path('.cache/local-feedback-activity-r6b');ROUTES=parent.ROUTES


class Data:
    def __init__(self):
        read(ROOT/'complete.json');m=read(ROOT/'manifest.json');self.records=m['records'];self.classes=m['classes'];self.fine=m['fine'];self.lookup={r['uid']:i for i,r in enumerate(self.records)};self.main=np.array([self.lookup[u]for u in m['mainUids']]);self.manual=np.array([self.lookup[u]for u in m['manualUids']]);self.profiles=m['profiles'];self.profile=self.profiles['detail'];self.widths=m['widths'];self.keys=m['featureKeys'];self.consumer_names=m['consumers']
        with np.load(ROOT/'arrays.npz')as z:self.a={k:z[k].copy()for k in z.files}
        self.arrays={'bag':self.a['bag']};self.mass=self.a['mass'].mean(0);self.expert=self.a['expert'];self.route=self.a['route'];self.raw=np.empty((len(self.route),0));self.weight=np.array([r['weight']for r in self.records]);self.y=np.array([[c in r['tags']for c in self.classes]for r in self.records],float);self.baseline=np.array([[c in r['legacyOverall']['tags']for c in self.classes]for r in self.records])
        self.frame_records=[(i,f)for i,r in enumerate(self.records)for f in r['frames']];self.frame_weight=np.zeros(len(self.route));self.inputs={};self.extra=[];self.features_by={}
        for i,r in enumerate(self.records):
            a,b=r['offset'];self.frame_weight[a:b]=self.a['dt'][a:b]/self.a['dt'][a:b].sum()*self.weight[i]
        old_family=np.eye(3)[self.route];guide_names=['continuousReleasePairRate','independentReleasePairRate','independentTailRate','heldReleaseObligation','independentHeldObligation'];rate=RATE_KEYS|set(guide_names)
        for wid in range(len(self.widths)):
            x=self.a['x'][wid].copy()
            for j,k in enumerate(self.keys):
                if k in rate:x[:,j]=np.log1p(x[:,j])
            relative=np.zeros((len(x),4));columns=[self.keys.index(k)for k in ['npsPerKey','rowRate','tailRate','occupancy']]
            for r in self.records:
                a,b=r['offset'];v=x[a:b][:,columns];relative[a:b]=v-np.average(v,axis=0,weights=self.a['dt'][a:b])
            for context in [False,True]:
                for obligations in [False,True]:
                    keys=self.keys if obligations else self.keys[:36];xx=x if obligations else x[:,:36]
                    names=keys+(['relativeLogNps','relativeLogRowRate','relativeLogTailRate','relativeOccupancy']if context else[])+['wasBreak','wasRC','wasLN'];values=np.c_[xx,relative,old_family]if context else np.c_[xx,old_family];self.inputs[wid,context,obligations]=(names,values)
        self.family_y=np.array([[any(c in parent.RC+['anchor','vibro','rc mix','hybrid']for c in r['tags']),any(c in parent.LN+['ln mix','hybrid']for c in r['tags'])]for r in self.records],float)
        # Synthetic observations are mechanism constraints, not random tests.
        for name,family,target in [('B01','break',None),('B02','ln',None),('C16','rc',None),('C24','rc',None),('CTX-L','rc',[9000,17000]),('CTX-H','break',[9000,17000]),('LN-L','ln',[9000,17000]),('LN-H','ln',[9000,17000])]:
            i=self.lookup['query/'+name];r=self.records[i];start,end=target or[r['frames'][0]['start'],r['frames'][-1]['end']];self.extra.append(dict(id=name,index=i,start=start,end=end,allowed=[family],group=None,kind='mechanism'))
        for rnd in ['r4','r5','r6']:
            for name,a in read('research/frontier/annotations-'+rnd+'.json')['cases'].items():
                i=self.lookup[a['sourceUid']];tag=a['localCategory'];family='break'if tag=='break'else'rc'if tag in parent.RC else'ln';start,end=a['sourceTargetMs'];self.extra.append(dict(id=name,index=i,start=start,end=end,allowed=[family],group=self.records[i]['group'],kind='real-source'))
        for name,a in read('research/frontier/annotations-r10.json')['originalLocalRevision']['cases'].items():
            i=self.lookup[a['sourceUid']];start,end=a['sourceTargetMs'];self.extra.append(dict(id=name,index=i,start=start,end=end,allowed=['ln'],group=self.records[i]['group'],kind='real-source'))
        old=parent.Data();self.expert_raw=[old.expert_raw[j]for j in self.a['globalParent']];self.fixed={}
        for name,mass in zip(self.consumer_names,self.a['mass']):
            support=np.sum(np.eye(3)[self.route,1:,None]*self.expert,axis=1);share=np.zeros((len(self.records),len(self.fine)));family=np.zeros((len(self.records),2));np.add.at(share,self.arrays['bag'],mass[:,None]*support);np.add.at(family,self.arrays['bag'],mass[:,None]*np.eye(3)[self.route,1:]);p=self.profiles[name];self.fixed[name]=predictions(self,share,family,self.baseline,[p['policy']],[p['family']])[0,0]
        self.routing_names=[];self.routing_x=np.zeros((len(self.route),0));self.pressure_names=[];self.pressure_x=np.zeros((len(self.route),0));self.x=np.empty((len(self.route),0));del self.a['x']

    routed_scores=parent.Data.routed_scores

    def consume(self,support,probabilities,name):
        mass=self.a['mass'][self.consumer_names.index(name)];share=np.zeros((len(self.records),len(self.fine)));family=np.zeros((len(self.records),2));np.add.at(share,self.arrays['bag'],mass[:,None]*support);np.add.at(family,self.arrays['bag'],mass[:,None]*np.eye(3)[probabilities.argmax(1),1:]);p=self.profiles[name]
        return predictions(self,share,family,self.baseline,[p['policy']],[p['family']])[0,0]


class Kernel(parent.Kernel):
    def __init__(self,d,train,manual,cfg):
        train_groups={d.records[i]['group']for i in train};extra=[e for e in d.extra if e['group']is None or e['group']in train_groups];indices=sorted({e['index']for e in extra}-set(train)-set(manual));extended=np.r_[manual,indices].astype(int)
        d.routing_names,d.routing_x=d.inputs[cfg['widthId'],cfg['context'],cfg.get('obligations',True)]
        # The original constructor handles arbitrary context columns. Keep a
        # separate external config: window/context are not fake pressure flags.
        internal=dict(cfg,features='context',priorStrength=2.,teacherWeight=0.);super().__init__(d,train,extended,internal);self.cfg=dict(cfg,priorStrength=2.,teacherWeight=0.);self.original_manual=np.array(manual);self.original_strong_count=len(self.strong)
        self.original_weights=self.strong_weights*len(extended)/max(1,len(manual));mapper={int(j):k for k,j in enumerate(self.frame_ids)};at=[];weight=[];allowed=[]
        for e in extra:
            row=d.records[e['index']];duration=e['end']-e['start'];mask=np.array([c in e['allowed']for c in ROUTES])
            for j in range(*row['offset']):
                f=d.frame_records[j][1];dt=max(0.,min(e['end'],f['end'])-max(e['start'],f['start']))
                if dt:at.append(mapper[j]);weight.append(dt/max(duration,1)/max(1,len(extra)));allowed.append(mask)
        self.strong=np.r_[self.strong,np.array(at,int)];self.allowed=np.r_[self.allowed,np.array(allowed,bool).reshape(-1,3)];self.extra_weights=np.array(weight);self.extra_receipt=extra;self.set_config(cfg)

    def set_config(self,cfg):
        self.cfg=dict(cfg,priorStrength=2.,teacherWeight=0.);self.strong_weights=np.r_[self.original_weights,cfg['extraWeight']*self.extra_weights]

    def forward(self,theta,*,all_rows=False):
        d=self.d;coef=theta.reshape(self.shape)
        if all_rows:
            f=np.arange(len(d.route));z=np.c_[np.ones(len(f)),(self.all_x-self.scaler['mean'])@self.scaler['matrix']];prior=np.eye(3)[d.route]*2.;expert=d.expert
        else:f=self.frame_ids;z=self.z;prior=self.prior;expert=self.expert
        p=parent.softmax(prior+z@coef@self.contrast.T,axis=1)
        family_index=np.array([1 if c in parent.RC+['anchor','vibro']else 2 for c in d.fine]);support=p[:,family_index]*expert.sum(1)
        bag=d.arrays['bag'][f];starts=np.r_[0,np.flatnonzero(np.diff(bag)!=0)+1];groups=bag[starts]
        if len(np.unique(groups))!=len(groups):raise ValueError('Training bags are not contiguous independent records')
        share=np.zeros((len(d.records),len(d.fine)));family=np.zeros((len(d.records),2));share[groups]=np.add.reduceat(d.mass[f,None]*support,starts,axis=0);family[groups]=np.add.reduceat(d.mass[f,None]*p[:,1:],starts,axis=0)
        return p,support,share,family,z

    def objective(self,theta):
        if self.cfg['wholeWeight']:
            return super().objective(theta)
        # Exact compact local-only objective: unlabeled main rows have zero
        # loss derivative, and no pseudo-label teacher is enabled. Keeping
        # them in every objective evaluation would only waste computation.
        mask=self.strong_weights>0;at=self.strong[mask];weight=self.cfg['strongWeight']*self.strong_weights[mask];allowed=self.allowed[mask];z=self.z[at];coef=theta.reshape(self.shape)
        p=parent.softmax(self.prior[at]+z@coef@self.contrast.T,axis=1);num=np.maximum(np.sum(p*allowed,axis=1),1e-300)
        loss=-np.sum(weight*np.log(num))+.5*self.cfg['ridge']*np.sum(coef*coef)/2
        gz=weight[:,None]*(p-p*allowed/num[:,None]);gradient=z.T@gz@self.contrast+self.cfg['ridge']*coef/2
        return float(loss),gradient.ravel()

    def export(self,theta):
        model=super().export(theta);model.update(kind='local-activity-routing',windowMs=self.d.widths[self.cfg['widthId']],stepMs=500,
            originalManualTrainUids=[self.d.records[i]['uid']for i in self.original_manual],extraObservations=self.extra_receipt,
            weakBurden='Mean of separately normalized Detailed/actions/Compact native burden fields; no re-scoring. Fine integral training is differentiable; family bag loss is an auxiliary surrogate, MAP family coverage used for the actual physical Hybrid gate.')
        return model


def local(d,ids,p):
    # Same actual inherited display smoothing and partial allowed-tag metric.
    return parent.routed_local(d,ids,p)


if __name__=='__main__':
    d=Data();s=PLANS[0]['inner'][0];cfg=dict(widthId=1,context=True,ridge=.03,wholeWeight=1.,strongWeight=1.,extraWeight=1.)
    k=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],cfg);rng=np.random.default_rng(21);t=rng.normal(0,.02,k.dimension);v=rng.normal(size=k.dimension);v/=np.linalg.norm(v);eps=1e-5;l,g=k.objective(t);fd=(k.objective(t+eps*v)[0]-k.objective(t-eps*v)[0])/(2*eps)
    if abs(fd-g@v)>1e-7:raise ValueError('Active local router gradient mismatch')
    theta,receipt=fit(k);p,support,_,_,_=k.forward(theta,all_rows=True);write('research/local_feedback/results/activity-resource-r6.json',dict(config=cfg,receipt=receipt,gradientError=float(abs(fd-g@v)),local=local(d,d.manual[s['strongValidation']],p),extraTrain=k.extra_receipt,
        consumers={n:metrics(d.y[d.main[s['validation']]],d.consume(support,p,n)[d.main[s['validation']]],d.weight[d.main[s['validation']]])for n in d.consumer_names}))
    print('active local resource',k.dimension,receipt['seconds'],'gradient',abs(fd-g@v),'local',local(d,d.manual[s['strongValidation']],p),flush=True)
